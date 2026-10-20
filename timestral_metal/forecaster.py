"""Probabilistic forecasting with Timestral, including the clampsym test-time augmentation.

A port of `timestral.forecaster`. `TimestralForecaster.forecast` takes a context `(B, V, T)`
(series, variates, time; NaN marks a missing value) and returns quantile forecasts
`(B, V, H, Q)`. Horizons longer than one decode block are forecast block by block, feeding each
block's median back as context.

Clampsym (both options on by default):

- Symmetric averaging forecasts the context and its negation, reflects the second
  forecast back (`-Q(1 - p)`), and averages the two. Every covariate row is negated
  with the context: numeric covariates, 0/1 indicators, target encodings and raw codes.
  Both forecasts run as one batch.
- Non-negative clipping clamps the forecast at 0 for every series whose observed
  context is entirely non-negative.

Categorical covariates are target-encoded against the series' forward-filled context
(see :mod:`timestral_metal.covariate_encoding`) and then handled like numeric covariates.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from timestral_metal.covariate_encoding import target_encode
from timestral_metal.loading import load_model
from timestral_metal.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET, TimestralModel
from timestral_metal.scaler import advancing_block_frames, causal_patch_loc_scale, clamp_nonfinite

ArrayLike = mx.array | np.ndarray


@dataclass(frozen=True)
class Forecast:
    """Quantile forecast.

    Attributes:
        quantiles: `(B, V, H, Q)` forecast values, sorted along the quantile axis.
        quantile_levels: The `Q` probability levels.
    """

    quantiles: np.ndarray
    quantile_levels: tuple[float, ...]

    @property
    def median(self) -> np.ndarray:
        """`(B, V, H)` median forecast."""
        return self.quantiles[..., self.quantile_levels.index(0.5)]

    def quantile(self, level: float) -> np.ndarray:
        """`(B, V, H)` forecast at one of the model's quantile levels."""
        if level not in self.quantile_levels:
            raise ValueError(f"level {level} is not one of {self.quantile_levels}")
        return self.quantiles[..., self.quantile_levels.index(level)]


def ffill_with_mask(inputs: mx.array, mask: mx.array) -> mx.array:
    """Forward-fill unobserved positions; leading unobserved positions become 0."""
    idx = mx.cummax(mx.where(mask, mx.arange(inputs.shape[-1]), 0), axis=-1)
    return mx.nan_to_num(mx.take_along_axis(inputs, idx, axis=-1), nan=0.0)


def pad_to_patch(inputs: mx.array, mask: mx.array, patch_size: int) -> tuple[mx.array, mx.array]:
    """Left-pad to a multiple of `patch_size`; padding is 0 and unobserved."""
    ctx_len = inputs.shape[-1]
    if ctx_len % patch_size != 0:
        pad = [(0, 0)] * (inputs.ndim - 1) + [(patch_size - ctx_len % patch_size, 0)]
        inputs = mx.pad(inputs, pad)
        mask = mx.pad(mask, pad, constant_values=False)
    return inputs, mask


def replace_extreme_values(x: mx.array, threshold: float, replacement: float = 0.0) -> mx.array:
    return mx.where(mx.isfinite(x) & (mx.abs(x) < threshold), x, replacement)


def nonnegative_context(inputs: mx.array, observation_mask: mx.array) -> mx.array:
    """`(B, V)`: whether every observed context value is non-negative (and one exists)."""
    observed = mx.isfinite(inputs) & observation_mask
    return observed.any(axis=-1) & ((inputs >= 0) | ~observed).all(axis=-1)


class TimestralForecaster:
    """Block-autoregressive quantile forecaster around a :class:`TimestralModel`.

    Args:
        model: The network.
        max_context: Context cap in time steps (most recent kept); defaults to the
            trained context length.
        per_shot_horizon_patches: Patches decoded per block, capped at the trained
            block size.
        quantile_real_cap_k: Clamp forecasts to `[min - k * s, max + k * s]` of the
            observed context, `s` being the scale of the last context patch. 0 disables.
        extreme_value_threshold: Fed-back medians at or above this magnitude become 0.
    """

    def __init__(
        self,
        model: TimestralModel,
        max_context: int | None = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 1e4,
        extreme_value_threshold: float = 1e30,
    ) -> None:
        assert per_shot_horizon_patches >= 1, f"per_shot_horizon_patches must be >= 1, got {per_shot_horizon_patches}"
        self.model = model
        self.config = model.config
        self.max_context = max_context if max_context is not None else model.config.context_length
        self.per_shot_horizon_patches = min(per_shot_horizon_patches, model.config.block_size)
        self.quantile_real_cap_k = quantile_real_cap_k
        self.extreme_value_threshold = extreme_value_threshold
        self.quantile_levels = tuple(model.config.quantile_knots)
        self.median_index = model.config.median_index
        # The refine step is many small operations, and runs once per decoded block.
        self._compiled_refine_block = mx.compile(self._refine_block)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        max_context: int | None = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 1e4,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ) -> "TimestralForecaster":
        """Load an exported checkpoint (`config.json` + `model.safetensors`).

        `path_or_repo_id` is a local directory or a Hugging Face Hub repo id. `revision`,
        `cache_dir`, `token` and `local_files_only` only apply to Hub repos.
        """
        model = load_model(path_or_repo_id, revision=revision, cache_dir=cache_dir, token=token,
                           local_files_only=local_files_only)
        return cls(model, max_context, per_shot_horizon_patches, quantile_real_cap_k)

    @property
    def patch_size(self) -> int:
        return self.config.patch_size

    @property
    def supports_covariates(self) -> bool:
        return self.config.covariate_enabled

    @property
    def supports_known_covariates(self) -> bool:
        return self.config.supports_known_covariates

    def forecast(
        self,
        context: ArrayLike,
        prediction_length: int,
        observation_mask: ArrayLike | None = None,
        past_covariates: ArrayLike | None = None,
        known_covariates: ArrayLike | None = None,
        categorical_past_covariates: ArrayLike | None = None,
        categorical_known_covariates: ArrayLike | None = None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
    ) -> Forecast:
        """Forecast `prediction_length` steps after `context`.

        Args:
            context: `(B, V, T)`, `(B, T)` (one variate) or `(T,)` (one series). NaN marks
                a missing value. The variates of a series are forecast jointly.
            prediction_length: Horizon in time steps.
            observation_mask: Optional boolean mask shaped like `context`; False marks
                additional missing values.
            past_covariates: `(B, C_past, T)` covariates known over the context only.
            known_covariates: `(B, C_known, T + prediction_length)` covariates known over
                the context and the horizon.
            categorical_past_covariates: `(B, C, T)` category codes (any numbers; NaN marks
                a missing value) known over the context only.
            categorical_known_covariates: `(B, C, T + prediction_length)` category codes
                known over the context and the horizon.
                Categorical covariates are target-encoded per series against the
                forward-filled target context. A series with several target variates
                has no single target to encode against, so its codes are used as is.
            symmetric_averaging: Average with the reflected forecast of `-context`, with every
                covariate row negated too.
            clip_nonnegative: Clamp at 0 the series whose observed context is non-negative.

        Returns:
            A :class:`Forecast` with `quantiles` `(B, V, prediction_length, Q)`.
        """
        assert prediction_length >= 1, f"prediction_length must be >= 1, got {prediction_length}"
        inputs, mask = self._as_context(context, observation_mask)
        b = inputs.shape[0]
        t = inputs.shape[-1]
        past = self._as_covariates(past_covariates, b, t, "past_covariates")
        known = self._as_covariates(known_covariates, b, t + prediction_length, "known_covariates")
        cat_past = self._as_categorical(categorical_past_covariates, b, t, "categorical_past_covariates")
        cat_known = self._as_categorical(categorical_known_covariates, b, t + prediction_length, "categorical_known_covariates")
        covariates = {"past_covariates": past, "known_covariates": known}
        if cat_past is not None or cat_known is not None:
            target = self._encoding_target(inputs, mask)
            enc_past = self._encode_categorical(cat_past, target)
            enc_known = self._encode_categorical(cat_known, target)
            covariates = {"past_covariates": _cat_rows(past, enc_past), "known_covariates": _cat_rows(known, enc_known)}
        if covariates["known_covariates"] is not None and not self.supports_known_covariates:
            raise ValueError("this checkpoint cannot condition on known-future covariates")

        if symmetric_averaging:
            levels = self.quantile_levels
            assert all(abs(lo + hi - 1.0) < 1e-9 for lo, hi in zip(levels, reversed(levels))), (
                f"symmetric averaging needs quantile levels symmetric about 0.5, got {levels}"
            )
            # The mirrored pass forecasts -context with every covariate row negated, as B more series of
            # the same batch. Target encodings are linear in the target, so negating them gives the
            # encodings against -context.
            both = {name: None if rows is None else mx.concatenate([rows, -rows]) for name, rows in covariates.items()}
            quantiles = self._forecast_raw(mx.concatenate([inputs, -inputs]), mx.concatenate([mask, mask]),
                                           prediction_length, **both)
            quantiles = 0.5 * (quantiles[:b] - quantiles[b:, ..., ::-1])
        else:
            quantiles = self._forecast_raw(inputs, mask, prediction_length, **covariates)
        if clip_nonnegative:
            eligible = nonnegative_context(inputs, mask)
            quantiles = mx.where(eligible[..., None, None], mx.maximum(quantiles, 0), quantiles)
        quantiles = np.array(quantiles)
        n_nan = int(np.isnan(quantiles).sum())
        if n_nan:
            raise ValueError(f"forecast produced {n_nan} NaN values; the input is degenerate")
        return Forecast(quantiles=quantiles, quantile_levels=self.quantile_levels)

    def _as_context(self, context: ArrayLike, observation_mask: ArrayLike | None) -> tuple[mx.array, mx.array]:
        x = mx.array(np.asarray(context, dtype=np.float32))
        if x.ndim == 1:
            x = x[None, None]
        elif x.ndim == 2:
            x = x[:, None]
        assert x.ndim == 3, f"context must be (T,), (B, T) or (B, V, T), got {tuple(x.shape)}"
        mask = mx.isfinite(x)
        if observation_mask is not None:
            mask = mask & mx.array(np.asarray(observation_mask, dtype=bool)).reshape(x.shape)
        return mx.where(mask, x, 0.0), mask

    def _encoding_target(self, inputs: mx.array, mask: mx.array) -> mx.array | None:
        """`(B, T)` target to encode against, or None when a series has several target variates.

        The forward-filled target from its first observation on (so left padding or a late
        start does not enter the category means), over the last `max_context` steps, the
        ones the model reads; NaN elsewhere.
        """
        if inputs.shape[1] != 1:
            return None
        t = inputs.shape[-1]
        started = mx.cummax(mask[:, 0].astype(mx.int32), axis=-1).astype(mx.bool_)
        in_window = mx.arange(t) >= t - self.max_context
        return mx.where(started & in_window, ffill_with_mask(inputs, mask)[:, 0], mx.nan)

    def _encode_categorical(self, codes: np.ndarray | None, target_context: mx.array | None) -> mx.array | None:
        """Target-encode `(B, C, L)` codes against `(B, T)` targets (NaN outside the encoding window); raw codes when None."""
        if codes is None or target_context is None:
            return mx.array(codes.astype(np.float32)) if codes is not None else None
        ctx = np.array(target_context).astype(np.float64)
        t = ctx.shape[-1]
        out = np.empty(codes.shape, dtype=np.float32)
        for i in range(codes.shape[0]):
            for j in range(codes.shape[1]):
                future = codes[i, j, t:] if codes.shape[-1] > t else None
                past_enc, future_enc = target_encode(ctx[i], codes[i, j, :t], future)
                # A missing code before the target starts is padding, not the missing category.
                out[i, j, :t] = np.where(np.isnan(ctx[i]) & np.isnan(codes[i, j, :t]), np.nan, past_enc)
                if future_enc is not None:
                    out[i, j, t:] = future_enc
        return mx.array(out)

    def _as_categorical(self, values: ArrayLike | None, batch: int, length: int, name: str) -> np.ndarray | None:
        """Category codes as numpy float64, so that distinct codes past 2**24 stay distinct."""
        v = self._as_covariates(values, batch, length, name, dtype=np.float64)
        if v is not None and np.isinf(v).any():
            raise ValueError(f"{name} must hold finite codes or NaN; got an infinite value")
        return v

    def _as_covariates(
        self, values: ArrayLike | None, batch: int, length: int, name: str, dtype: type = np.float32
    ) -> mx.array | np.ndarray | None:
        """`(B, C, length)` covariates: an MLX array for float32, numpy for float64; None when absent or empty."""
        if values is None:
            return None
        if not self.supports_covariates:
            raise ValueError(f"this checkpoint was not trained with covariates; got {name}")
        v = np.asarray(values, dtype=dtype)
        if v.ndim == 2:
            v = v[:, None]
        assert v.ndim == 3 and v.shape[0] == batch and v.shape[-1] == length, (
            f"{name} must be (B={batch}, C, {length}), got {tuple(v.shape)}"
        )
        if v.shape[1] == 0:
            return None
        return mx.array(v) if dtype == np.float32 else v

    def _forecast_raw(
        self,
        inputs: mx.array,
        observation_mask: mx.array,
        prediction_length: int,
        past_covariates: mx.array | None = None,
        known_covariates: mx.array | None = None,
    ) -> mx.array:
        """One forecast pass: `(B, V, T)` inputs to `(B, V, H, Q)` raw-space quantiles."""
        rows, mask, role_ids, known_future, known_future_mask, n_tgt = self._assemble_rows(
            inputs, observation_mask, prediction_length, past_covariates, known_covariates
        )
        quantiles = self._block_ar_decode(rows, mask, prediction_length, role_ids, known_future, known_future_mask)
        quantiles = quantiles[:, :n_tgt]
        if self.quantile_real_cap_k > 0:
            quantiles = self._cap_quantiles(quantiles, rows[:, :n_tgt], mask[:, :n_tgt])
        return quantiles

    def _cap_quantiles(self, quantiles: mx.array, inputs: mx.array, observation_mask: mx.array) -> mx.array:
        not_obs = ~observation_mask
        ctx_max = mx.nan_to_num(mx.where(not_obs, -mx.inf, inputs).max(axis=-1, keepdims=True), neginf=0.0)
        ctx_min = mx.nan_to_num(mx.where(not_obs, mx.inf, inputs).min(axis=-1, keepdims=True), posinf=0.0)
        c = self.config
        _, scale = causal_patch_loc_scale(inputs, observation_mask, c.patch_size, c.minimum_scale)
        anchor_scale = scale[..., -1:] * self.quantile_real_cap_k
        return mx.clip(quantiles, (ctx_min - anchor_scale)[..., None], (ctx_max + anchor_scale)[..., None])

    def _assemble_rows(
        self,
        inputs: mx.array,
        observation_mask: mx.array,
        prediction_length: int,
        past_covariates: mx.array | None,
        known_covariates: mx.array | None,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None, mx.array | None, int]:
        """Stack target and covariate rows; returns `(rows, mask, role_ids, known_future, known_future_mask, n_tgt)`."""
        ps = self.patch_size
        b, n_tgt, ctx0 = inputs.shape
        horizon = math.ceil(prediction_length / ps) * ps

        ctx_rows = [ffill_with_mask(inputs, observation_mask)]
        ctx_masks = [observation_mask]
        roles = [ROLE_TARGET] * n_tgt

        n_past = 0
        if past_covariates is not None:
            assert past_covariates.shape[-1] == ctx0, "past covariates must cover exactly the context"
            n_past = past_covariates.shape[1]
            ctx_masks.append(~mx.isnan(past_covariates))
            ctx_rows.append(mx.nan_to_num(past_covariates, nan=0.0))
            roles += [ROLE_PAST_COV] * n_past

        n_known = 0
        known_future_vals: mx.array | None = None
        known_future_obs: mx.array | None = None
        if known_covariates is not None:
            n_known = known_covariates.shape[1]
            kc_ctx, kc_fut = known_covariates[..., :-prediction_length], known_covariates[..., -prediction_length:]
            assert kc_ctx.shape[-1] == ctx0, "known covariates must cover the context plus the horizon"
            ctx_masks.append(~mx.isnan(kc_ctx))
            ctx_rows.append(mx.nan_to_num(kc_ctx, nan=0.0))
            roles += [ROLE_KNOWN_COV] * n_known
            known_future_obs = ~mx.isnan(kc_fut)
            known_future_vals = mx.nan_to_num(kc_fut, nan=0.0)

        rows = mx.concatenate(ctx_rows, axis=1)
        mask = mx.concatenate(ctx_masks, axis=1)
        if rows.shape[-1] > self.max_context:
            rows = rows[:, :, -self.max_context :]
            mask = mask[:, :, -self.max_context :]
        rows, mask = pad_to_patch(rows, mask, ps)
        v_all = rows.shape[1]
        if n_past == 0 and n_known == 0:
            return rows, mask, None, None, None, n_tgt

        role_ids = mx.array(roles * b)
        known_future: mx.array | None = None
        known_future_mask: mx.array | None = None
        if n_known > 0:
            assert known_future_vals is not None and known_future_obs is not None
            k0 = n_tgt + n_past
            plen = known_future_vals.shape[-1]
            kf = mx.zeros((b, v_all, horizon), dtype=rows.dtype)
            kfm = mx.zeros((b, v_all, horizon), dtype=mx.bool_)
            kf[:, k0 : k0 + n_known, :plen] = known_future_vals
            kfm[:, k0 : k0 + n_known, :plen] = known_future_obs
            known_future = kf.reshape(b * v_all, horizon)
            known_future_mask = kfm.reshape(b * v_all, horizon)
        return rows, mask, role_ids, known_future, known_future_mask, n_tgt

    def _block_ar_decode(
        self,
        inputs: mx.array,
        observation_mask: mx.array,
        prediction_length: int,
        role_ids: mx.array | None,
        known_future: mx.array | None,
        known_future_mask: mx.array | None,
    ) -> mx.array:
        """Decode block by block in the packed `(R, T)` layout; returns `(B, V, H, Q)`.

        After each block the context grows by the block's feedback: targets carry their
        median, known-future covariates their real future, past-only covariates a blank.
        Past `max_context` the oldest steps are dropped.
        """
        ps = self.patch_size
        b, v, ctx_len = inputs.shape
        context = inputs.reshape(b * v, ctx_len)
        mask = observation_mask.reshape(b * v, ctx_len)
        group_ids = mx.array(np.repeat(np.arange(b), v))  # from the host, so the model can read it back for free

        known_rows = past_rows = target_rows = None
        if role_ids is not None:
            known_rows = role_ids == ROLE_KNOWN_COV
            past_rows = (role_ids != ROLE_TARGET) & ~known_rows
            target_rows = role_ids == ROLE_TARGET

        blocks: list[mx.array] = []
        remaining = math.ceil(prediction_length / ps)
        offset = 0
        while remaining > 0:
            block = min(self.per_shot_horizon_patches, remaining)
            blen = block * ps
            kf_block = known_future[:, offset : offset + blen] if known_future is not None else None
            kfm_block = known_future_mask[:, offset : offset + blen] if known_future_mask is not None else None
            quantiles = self.model.predict_block(context, mask, group_ids, block, role_ids, kf_block, kfm_block)
            quantiles = self._compiled_refine_block(quantiles, context, mask, target_rows)
            blocks.append(quantiles)
            remaining -= block
            offset += blen
            if remaining <= 0:
                break
            feedback = replace_extreme_values(quantiles[..., self.median_index], self.extreme_value_threshold)
            feedback_mask = mx.ones(feedback.shape, dtype=mx.bool_)
            if role_ids is not None:
                assert known_rows is not None and past_rows is not None
                if kf_block is not None:
                    assert kfm_block is not None
                    feedback = mx.where(known_rows[:, None], kf_block, feedback)
                    feedback_mask = mx.where(known_rows[:, None], kfm_block, feedback_mask)
                feedback = mx.where(past_rows[:, None], 0.0, feedback)
                feedback_mask = feedback_mask & ~past_rows[:, None]
            context = mx.concatenate([context, feedback], axis=-1)
            mask = mx.concatenate([mask, feedback_mask], axis=-1)
            if context.shape[-1] > self.max_context:
                keep = max(ps, (self.max_context // ps) * ps)
                context = context[..., -keep:]
                mask = mask[..., -keep:]

        packed = mx.concatenate(blocks, axis=-2)[..., :prediction_length, :]
        return packed.reshape(b, v, prediction_length, -1)

    def _refine_block(self, quantiles: mx.array, context: mx.array, mask: mx.array, target_rows: mx.array | None) -> mx.array:
        """Move one decoded block from the anchor frame to the advancing refine frame it was trained in.

        The model denormalizes the whole block `(R, H * patch_size, Q)` with the frame of the
        last context patch. The first horizon patch keeps that frame; each later one is
        re-denormalized with :func:`timestral_metal.scaler.advancing_block_frames`. Covariate rows
        keep their quantiles.
        """
        ps = self.patch_size
        n_pred = quantiles.shape[1] // ps
        if n_pred <= 1:
            return quantiles
        _, loc, scale = self.model.scale(context, mask)
        loc0, scale0 = loc[:, -1, None, None], scale[:, -1, None, None]
        median = quantiles[..., self.median_index].reshape(-1, n_pred, ps)
        loc_j, scale_j = advancing_block_frames(context, mask, (median - loc0) / scale0, self.config.minimum_scale)
        loc_j = mx.repeat(mx.concatenate([loc0[:, 0], loc_j[:, 1:]], axis=1), ps, axis=-1)
        scale_j = mx.repeat(mx.concatenate([scale0[:, 0], scale_j[:, 1:]], axis=1), ps, axis=-1)
        refined = clamp_nonfinite((quantiles - loc0) / scale0 * scale_j[..., None] + loc_j[..., None])
        if target_rows is not None:
            refined = mx.where(target_rows[:, None, None], refined, quantiles)
        return refined


def _cat_rows(a: mx.array | None, b: mx.array | None) -> mx.array | None:
    """Concatenate covariate rows along the channel axis, either side possibly None."""
    if a is None or b is None:
        return a if b is None else b
    return mx.concatenate([a, b], axis=1)

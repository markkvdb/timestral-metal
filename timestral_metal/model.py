"""The Timestral network and its single-block forecast primitive.

A port of `timestral.model`. The model normalizes each row causally per patch, embeds patches
from `[values, missing]` channels, runs a time/variate transformer over `[Enc ; Dec]` tokens
(context patches followed by learned forecast-query tokens), and reads quantiles for every
horizon time step from the decoder tokens.

Upstream runs context and horizon queries as two attention passes when
`split_context_horizon_attn` is set. That split skips work but gives the same result as one
pass with the full mask, which is what this port runs.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from timestral_metal.config import TimestralConfig
from timestral_metal.layers import ResidualBlock, Transformer
from timestral_metal.masks import (
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_positions,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
    group_first_observed_patch,
)
from timestral_metal.scaler import (
    apply_affine_asinh_masked,
    causal_patch_loc_scale,
    denormalize_quantiles,
    instance_norm_stats,
)

ROLE_TARGET = 0
ROLE_PAST_COV = 1
ROLE_KNOWN_COV = 2


class PatchRepresentation(nn.Module):
    """Patch embedding over per-step `[values, missing]` channels, plus the forecast-query token."""

    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = ResidualBlock(2 * patch_size, 4 * d_model, d_model, use_bias=True)
        self.forecast_query = mx.zeros((d_model,))

    def __call__(self, scaled: mx.array, observed: mx.array) -> mx.array:
        """`(R, T)` normalized values and observation mask to `(R, T / patch_size, d_model)`."""
        r, t = scaled.shape
        n = t // self.patch_size
        values = scaled.reshape(r, n, self.patch_size)
        missing = (~observed).reshape(r, n, self.patch_size).astype(scaled.dtype)
        return self.patch_proj(mx.concatenate([values, missing], axis=-1))

    def forecast_tokens(self, rows: int, n_patches: int) -> mx.array:
        return mx.broadcast_to(self.forecast_query, (rows, n_patches, self.forecast_query.shape[0]))


def softplus(x: mx.array) -> mx.array:
    """`torch.nn.functional.softplus` with its defaults, `beta=1` and `threshold=20`."""
    return mx.where(x > 20, x, mx.log1p(mx.exp(x)))


def monotone_quantiles(raw: mx.array, levels: tuple[float, ...], width_factor: float) -> mx.array:
    """Non-crossing quantiles from raw head outputs along the last axis (`monotone_cumsum`).

    The median channel is free. The others fan out from it as
    `median -/+ cumsum(width_factor * dp * softplus(raw))`, `dp` being the probability gap
    to the next level inward, so unequal knot spacing is respected.
    """
    m = levels.index(0.5)
    widths = [0.0] * len(levels)
    for i in range(m + 1, len(levels)):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(m):
        widths[i] = levels[i + 1] - levels[i]
    inc = width_factor * mx.array(widths, dtype=mx.float32) * softplus(raw)
    median = raw[..., m : m + 1]
    up = mx.cumsum(inc[..., m + 1 :], axis=-1)
    down = mx.cumsum(inc[..., :m][..., ::-1], axis=-1)[..., ::-1]
    big = mx.finfo(mx.float32).max
    return mx.clip(mx.concatenate([median - down, median, median + up], axis=-1), -big, big)


class QuantileHead(nn.Module):
    """Maps each decoder token to `patch_size` monotone quantile vectors in normalized space."""

    def __init__(self, config: TimestralConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = ResidualBlock(
            config.d_model, 4 * config.d_model, config.patch_size * config.num_quantiles, use_bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        """`(R, N, D)` to normalized quantiles `(R, N * patch_size, Q)`."""
        c = self.config
        raw = self.projection(x)
        *leading, n_patches, _ = raw.shape
        preds = raw.reshape(*leading, n_patches, c.patch_size, c.num_quantiles)
        preds = monotone_quantiles(preds, c.quantile_knots, c.quantile_width_factor)
        return preds.reshape(*leading, n_patches * c.patch_size, c.num_quantiles)


def _series_index(group_ids: mx.array) -> tuple[mx.array, int]:
    """For each row, the index of its series among the distinct `group_ids`, and the number of series."""
    unique_ids, inverse = np.unique(np.array(group_ids), return_inverse=True)
    return mx.array(inverse), len(unique_ids)


class TimestralModel(nn.Module):
    def __init__(self, config: TimestralConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.representation = PatchRepresentation(c.patch_size, c.d_model)
        self.transformer = Transformer(
            block_schedule=c.block_schedule,
            embed_dim=c.d_model,
            num_heads=c.num_heads,
            mlp_hidden_dim=c.mlp_hidden_dim,
            rope_theta=c.rope_theta,
            rope_dim=c.rope_dim,
            norm_eps=c.norm_eps,
            attn_bias=c.attn_bias,
            mlp_bias=c.mlp_bias,
            time_attn_gate=c.time_attn_gate,
            variate_attn_gate=c.variate_attn_gate,
        )
        self.head = QuantileHead(c)
        # The parameters are implicit inputs, so loading new weights does not leave a stale graph.
        self._compiled_transformer = mx.compile(self.transformer, inputs=self.transformer.state)

    def scale(self, inputs: mx.array, observed: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Causal per-patch normalization of packed `(R, T)` rows.

        Returns `(scaled, loc, scale)`, each `(R, T)`.
        """
        c = self.config
        loc, scale = causal_patch_loc_scale(inputs, observed, c.patch_size, c.minimum_scale)
        return apply_affine_asinh_masked(inputs, loc, scale, observed, use_asinh=c.use_asinh), loc, scale

    def _known_cov_scaling(
        self, inputs: mx.array, observed: mx.array, scaled: mx.array, loc: mx.array, scale: mx.array, known_rows: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Normalize known-future covariate rows with non-causal whole-row statistics.

        Indicator rows (values in {0, 1}) skip the asinh squash.
        """
        if not known_rows.any().item():
            return scaled, loc, scale
        loc_c, scale_c, is_binary = instance_norm_stats(inputs, observed, minimum_scale=self.config.minimum_scale)
        known_scaled = apply_affine_asinh_masked(inputs, loc_c, scale_c, observed, use_asinh=False)
        if self.config.use_asinh:
            known_scaled = mx.where(observed & ~is_binary, mx.arcsinh(known_scaled), known_scaled)
        sel = known_rows[:, None]
        return mx.where(sel, known_scaled, scaled), mx.where(sel, loc_c, loc), mx.where(sel, scale_c, scale)

    def _run_two_stream(
        self, x_enc: mx.array, x_dec: mx.array, time_mask: mx.array, variate_mask: mx.array | None, positions: mx.array
    ) -> mx.array:
        """Transformer over `[Enc ; Dec]`, then the head on the decoder tokens (normalized space)."""
        if variate_mask is None:  # a single row; the compiled stack takes arrays only
            variate_mask = mx.ones((1, 1, 1, 1), dtype=mx.bool_)
        h = self._compiled_transformer(mx.concatenate([x_enc, x_dec], axis=1), time_mask, variate_mask, positions)
        return self.head(h[:, x_enc.shape[1] :, :])

    def predict_block(
        self,
        inputs: mx.array,
        observed: mx.array,
        group_ids: mx.array,
        horizon_patches: int,
        role_ids: mx.array | None = None,
        known_future: mx.array | None = None,
        known_future_mask: mx.array | None = None,
    ) -> mx.array:
        """Forecast one block of `horizon_patches` patches after a packed context.

        Args:
            inputs: `(R, T)` context values, `T` a multiple of the patch size.
            observed: `(R, T)` observation mask.
            group_ids: `(R,)` series id per row; rows of one series attend each other.
            horizon_patches: Number of patches to forecast.
            role_ids: `(R,)` row roles (`ROLE_TARGET`, `ROLE_PAST_COV`, `ROLE_KNOWN_COV`),
                or None when every row is a target.
            known_future: `(R, horizon_patches * patch_size)` known-future covariate
                values over this block; zeros on other rows.
            known_future_mask: Observation mask for `known_future`.

        Returns:
            Raw-space quantiles `(R, horizon_patches * patch_size, Q)`, non-decreasing along Q.
        """
        assert inputs.ndim == 2, f"expected packed (R, T) inputs, got {tuple(inputs.shape)}"
        if role_ids is None or not self.config.covariate_enabled or (role_ids == ROLE_TARGET).all().item():
            return self._predict_context_only(inputs, observed, group_ids, horizon_patches)
        r = inputs.shape[0]
        horizon_len = horizon_patches * self.config.patch_size
        if known_future is None:
            known_future = mx.zeros((r, horizon_len), dtype=inputs.dtype)
            known_future_mask = mx.zeros((r, horizon_len), dtype=mx.bool_)
        assert known_future_mask is not None, "known_future_mask is required alongside known_future"
        assert known_future.shape == (r, horizon_len), f"known_future must be ({r}, {horizon_len}), got {tuple(known_future.shape)}"
        full_inputs = mx.concatenate([inputs, known_future], axis=-1)
        full_observed = mx.concatenate([observed, known_future_mask], axis=-1)
        return self._predict_window(full_inputs, full_observed, group_ids, horizon_patches, role_ids)

    def _predict_context_only(
        self, inputs: mx.array, observed: mx.array, group_ids: mx.array, horizon_patches: int
    ) -> mx.array:
        """No covariates: encoder over the context, `h` forecast-query tokens after it."""
        c = self.config
        ps = c.patch_size
        rows, t = inputs.shape
        assert t % ps == 0, f"context length {t} must be a multiple of patch_size {ps}"
        assert group_ids.shape == (rows,), f"group_ids must be ({rows},), got {tuple(group_ids.shape)}"
        ll, h = t // ps, horizon_patches
        assert h >= 1, f"horizon_patches must be >= 1, got {h}"

        scaled, loc, scale = self.scale(inputs, observed)
        x_enc = self.representation(scaled, observed)
        x_dec = self.representation.forecast_tokens(rows, h)

        obs_per_patch = observed.reshape(rows, ll, ps).any(axis=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        time_mask = build_forecast_time_mask(ll, h, key_valid)
        variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)
        inverse, num_series = _series_index(group_ids)
        first_obs = group_first_observed_patch(obs_per_patch, inverse, num_series)[inverse]
        positions = build_forecast_positions(ll, h, first_obs)

        preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
        anchor = (ll - 1) * ps
        return denormalize_quantiles(preds, loc[:, anchor, None, None], scale[:, anchor, None, None], c.use_asinh)

    def _predict_window(
        self, full_inputs: mx.array, full_observed: mx.array, group_ids: mx.array, horizon_patches: int, role_ids: mx.array
    ) -> mx.array:
        """With covariates: encoder over context plus horizon, so known-future values are keys.

        Target and past-only rows keep a blank horizon. The time and variate masks keep
        target information out of covariate rows and keep any row's future out of the
        targets.
        """
        c = self.config
        ps = c.patch_size
        rows, w = full_inputs.shape
        assert w % ps == 0, f"window length {w} must be a multiple of patch_size {ps}"
        assert group_ids.shape == (rows,) and role_ids.shape == (rows,), "group_ids and role_ids must be (R,)"
        h = horizon_patches
        assert 1 <= h <= c.block_size, f"decode block {h} must be in [1, {c.block_size}] with covariates"
        n = w // ps
        ll = n - h
        assert ll >= 1, f"context must have at least one patch (window {n}, horizon {h})"
        target_rows = role_ids == ROLE_TARGET
        known_rows = role_ids == ROLE_KNOWN_COV

        scaled, loc, scale = self.scale(full_inputs, full_observed)
        scaled, loc, scale = self._known_cov_scaling(full_inputs, full_observed, scaled, loc, scale, known_rows)
        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, h)

        obs_per_patch = full_observed.reshape(rows, n, ps).any(axis=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, c.attend_missing_patches)
        s_rows = mx.full((rows,), ll)
        time_mask = build_two_stream_time_mask(s_rows, n, c.block_size, key_valid, bidirectional_rows=known_rows)
        variate_mask = build_two_stream_variate_mask(
            group_ids, key_valid, n, target_rows=target_rows, bidirectional_rows=known_rows
        )
        inverse, num_series = _series_index(group_ids)
        first_obs = group_first_observed_patch(obs_per_patch, inverse, num_series)[inverse]
        positions = build_two_stream_positions(n, first_obs)

        # Upstream runs a decoder token for every window patch and keeps the last `h`. The others are
        # inactive: they read only themselves, and no kept token reads them. Leaving them out gives the
        # same forecast from N + h tokens instead of 2N.
        keep = mx.concatenate([mx.arange(n), mx.arange(n + ll, 2 * n)])
        time_mask = mx.take(mx.take(time_mask, keep, axis=2), keep, axis=3)
        variate_mask = None if variate_mask is None else mx.take(variate_mask, keep, axis=0)
        horizon_preds = self._run_two_stream(x_enc, x_dec, time_mask, variate_mask, mx.take(positions, keep, axis=1))
        anchor = (ll - 1) * ps
        return denormalize_quantiles(horizon_preds, loc[:, anchor, None, None], scale[:, anchor, None, None], c.use_asinh)

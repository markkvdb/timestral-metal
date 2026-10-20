"""`timestral_metal.scaler` (float32) against `timestral.scaler` (float64 accumulation, float32 results)."""

import mlx.core as mx
import numpy as np
import torch
from timestral import scaler as ref

from timestral_metal import scaler

PS, MIN_SCALE, T = 32, 1e-6, 32768


def hard_rows() -> tuple[np.ndarray, np.ndarray]:
    """Rows that stress float32 accumulation, with a late start, a gap and random gaps."""
    rng = np.random.default_rng(0)
    steps = np.arange(T)
    rows = np.stack([
        50 + rng.standard_normal(T).cumsum(),  # random walk
        1e6 + rng.standard_normal(T),  # large level
        1e6 + 0.01 * rng.standard_normal(T),  # large level, noise below float32 resolution
        1e8 + 10 * rng.standard_normal(T),
        steps * (1e5 / T) + rng.standard_normal(T),  # trend
        1e-6 * rng.standard_normal(T),  # tiny values
        rng.standard_t(1, T),  # heavy-tailed spikes
        300 + 50 * np.sin(2 * np.pi * steps / 96) + 5 * rng.standard_normal(T),  # daily cycle
    ]).astype(np.float32)
    obs = np.ones(rows.shape, dtype=bool)
    obs[1, :5000] = False
    obs[2, 10000:12000] = False
    obs[6, rng.random(T) < 0.3] = False
    return np.where(obs, rows, 0.0).astype(np.float32), obs


def test_causal_patch_loc_scale() -> None:
    x, obs = hard_rows()
    want_loc, want_scale = (a.numpy() for a in ref.causal_patch_loc_scale(torch.from_numpy(x), torch.from_numpy(obs), PS, MIN_SCALE, False))
    loc, scale = (np.array(a) for a in scaler.causal_patch_loc_scale(mx.array(x), mx.array(obs), PS, MIN_SCALE))
    assert loc.shape == want_loc.shape and loc.dtype == np.float32
    np.testing.assert_array_less(np.abs(loc - want_loc) / want_scale, 1e-6)
    np.testing.assert_array_less(np.abs(scale / want_scale - 1), 1e-6)


def test_advancing_block_frames() -> None:
    x, obs = hard_rows()
    x, obs = x[:, -4096:], obs[:, -4096:]
    sinh_median = np.random.default_rng(1).standard_normal((x.shape[0], 8, PS)).cumsum(-1) / 4
    # Upstream returns float64; its caller, `_refine_block`, casts the frames to float32 at once.
    want_loc, want_scale = (a.float().numpy() for a in ref.advancing_block_frames(
        torch.from_numpy(x), torch.from_numpy(obs), torch.from_numpy(sinh_median), MIN_SCALE
    ))
    loc, scale = (np.array(a) for a in scaler.advancing_block_frames(
        mx.array(x), mx.array(obs), mx.array(sinh_median.astype(np.float32)), MIN_SCALE
    ))
    np.testing.assert_array_less(np.abs(loc - want_loc) / want_scale, 1e-6)
    np.testing.assert_array_less(np.abs(scale / want_scale - 1), 1e-6)


def test_instance_norm_and_transforms() -> None:
    rng = np.random.default_rng(2)
    values = np.stack([rng.standard_normal(500) * 3 + 7, (rng.random(500) < 0.4).astype(float), rng.standard_normal(500)]).astype(np.float32)
    obs = rng.random(values.shape) < 0.9
    obs[2] = False
    for got, want in zip(
        scaler.instance_norm_stats(mx.array(values), mx.array(obs), MIN_SCALE),
        ref.instance_norm_stats(torch.from_numpy(values), torch.from_numpy(obs), MIN_SCALE),
    ):
        np.testing.assert_allclose(np.array(got), want.numpy(), rtol=1e-6)
    loc, scale = values.mean(-1, keepdims=True), values.std(-1, keepdims=True) + 0.5
    scaled = scaler.apply_affine_asinh_masked(mx.array(values), mx.array(loc), mx.array(scale), mx.array(obs), True)
    want = ref.apply_affine_asinh_masked(*(torch.from_numpy(a) for a in (values, loc, scale, obs)), use_asinh=True)
    np.testing.assert_allclose(np.array(scaled), want.numpy(), rtol=1e-6, atol=1e-7)
    back = np.array(scaler.invert_affine_asinh(scaled, mx.array(loc), mx.array(scale), True))
    want_back = ref.invert_affine_asinh(want, torch.from_numpy(loc), torch.from_numpy(scale), True).numpy()
    # The result is a difference of two terms, so float32 rounding scales with the terms, not the result.
    terms = np.abs(loc) + np.abs(np.sinh(want.numpy()) * scale)
    np.testing.assert_array_less(np.abs(back - want_back), 4 * np.finfo(np.float32).eps * terms)


def test_clamp_nonfinite() -> None:
    vals = np.array([[1.0, np.inf, 3.0], [-np.inf, 2.0, np.inf], [np.inf, np.inf, -np.inf], [np.nan, 1.0, np.inf]], dtype=np.float32)
    np.testing.assert_array_equal(np.array(scaler.clamp_nonfinite(mx.array(vals))), ref.clamp_nonfinite(torch.from_numpy(vals)).numpy())

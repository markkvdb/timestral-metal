"""Normalization for Timestral: causal per-patch mean/std scaling with an asinh squash.

A port of `timestral.scaler`. Each patch is normalized with the mean and std of every observed
value up to the end of that patch, so the context never sees future statistics. Forecasts
come back in the normalized (asinh) space of the last context patch and are inverted with `sinh`.

Upstream accumulates these statistics in float64. Here they are float32, which the GPU
supports: each row is shifted by its first observed value before accumulating, and the
per-patch states are combined with a log-depth scan. The results match upstream's float32
values to within about 1e-6 of the scale.
"""

import mlx.core as mx

F32_EPS = mx.finfo(mx.float32).eps
F32_TINY = 2.0**-126


def welford_combine(
    w_a: mx.array, mean_a: mx.array, m2_a: mx.array, w_b: mx.array, mean_b: mx.array, m2_b: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    """Merge two Welford states `(weight, mean, m2)` element-wise; `(0, 0, 0)` is the identity."""
    w = w_a + w_b
    w_safe = mx.maximum(w, F32_TINY)
    delta = mean_b - mean_a
    mean = mean_a + delta * w_b / w_safe
    m2 = m2_a + m2_b + delta * delta * w_a * w_b / w_safe
    is_zero = w == 0
    return w, mx.where(is_zero, 0.0, mean), mx.where(is_zero, 0.0, m2)


def cumulative_welford(w: mx.array, mean: mx.array, m2: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Inclusive prefix scan of Welford states along the last axis (Hillis-Steele, `log2(N)` merges)."""
    n = w.shape[-1]
    d = 1
    while d < n:
        pad = [(0, 0)] * (w.ndim - 1) + [(d, 0)]
        shifted = (mx.pad(s[..., : n - d], pad) for s in (w, mean, m2))
        w, mean, m2 = welford_combine(*shifted, w, mean, m2)
        d *= 2
    return w, mean, m2


def _causal_patch_states(
    data: mx.array, observation_mask: mx.array, patch_size: int
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Welford states `(count, mean, m2)` of each `(R, T)` row up to the end of each patch, plus the row shift.

    The means are relative to the shift, the row's first observed value (0 if none), shaped `(R, 1)`.
    """
    r, t = data.shape
    w = observation_mask.astype(mx.float32)
    first = mx.argmax(w, axis=-1, keepdims=True)
    shift = mx.take_along_axis(data, first, axis=-1) * (w.sum(-1, keepdims=True) > 0)
    x = ((data - shift) * w).reshape(r, t // patch_size, patch_size)
    wp = w.reshape(r, t // patch_size, patch_size)
    cnt = wp.sum(-1)
    mean = x.sum(-1) / mx.maximum(cnt, 1.0)
    m2 = (wp * (x - mean[..., None]) ** 2).sum(-1)
    return (*cumulative_welford(cnt, mean, m2), shift)


def _scale(cnt: mx.array, m2: mx.array, minimum_scale: float) -> mx.array:
    """Bessel-corrected std, clamped to `minimum_scale` after the sqrt."""
    variance = m2 / mx.maximum(mx.maximum(cnt, 1.0) - 1.0, 1.0)
    return mx.maximum(mx.sqrt(mx.maximum(variance, 0.0)), minimum_scale)


def causal_patch_loc_scale(
    data: mx.array, observation_mask: mx.array, patch_size: int, minimum_scale: float
) -> tuple[mx.array, mx.array]:
    """Per-patch `(loc, scale)` broadcast back to every time step of the patch, each shaped like `data`.

    Each patch uses the statistics of everything observed up to the end of that patch.
    """
    shape = data.shape
    t = shape[-1]
    assert t % patch_size == 0, f"length {t} is not a multiple of patch_size {patch_size}"
    cnt, mean, m2, shift = _causal_patch_states(data.reshape(-1, t), observation_mask.reshape(-1, t), patch_size)
    loc = mx.where(cnt > 0, mean + shift, 0.0)
    scale = _scale(cnt, m2, minimum_scale)
    return mx.repeat(loc, patch_size, axis=-1).reshape(shape), mx.repeat(scale, patch_size, axis=-1).reshape(shape)


def apply_affine_asinh_masked(
    data: mx.array, loc: mx.array, scale: mx.array, observation_mask: mx.array, use_asinh: bool
) -> mx.array:
    """`asinh((data - loc) / scale)` on observed positions; unobserved positions become 0."""
    scaled = (data - loc) / mx.maximum(scale, F32_EPS)
    if use_asinh:
        scaled = mx.arcsinh(scaled)
    return mx.where(observation_mask, scaled, 0.0)


def invert_affine_asinh(x: mx.array, loc: mx.array, scale: mx.array, use_asinh: bool) -> mx.array:
    """Inverse of the forward transform: `sinh(x) * scale + loc`."""
    return (mx.sinh(x) if use_asinh else x) * scale + loc


def clamp_nonfinite(vals: mx.array) -> mx.array:
    """Map +/-inf (a sinh overflow) to the nearest finite value along the last axis.

    A position whose whole last axis is non-finite becomes 0. NaN is left untouched so
    that real instability still surfaces downstream.
    """
    finite = mx.isfinite(vals)
    hi = mx.where(finite, vals, -mx.inf).max(axis=-1, keepdims=True)
    lo = mx.where(finite, vals, mx.inf).min(axis=-1, keepdims=True)
    clamped = mx.where(vals == mx.inf, hi, mx.where(vals == -mx.inf, lo, vals))
    return mx.where(mx.isinf(clamped), 0.0, clamped)


def denormalize_quantiles(preds: mx.array, loc: mx.array, scale: mx.array, use_asinh: bool) -> mx.array:
    """Invert the normalization and clamp infinities; monotone quantiles stay in order."""
    return clamp_nonfinite(invert_affine_asinh(preds, loc, scale, use_asinh))


def advancing_block_frames(
    context: mx.array, observation_mask: mx.array, sinh_median: mx.array, minimum_scale: float
) -> tuple[mx.array, mx.array]:
    """Per-patch `(loc, scale)` of a decoded block in the advancing refine frame, each `(R, H)`.

    Horizon patch `j` is normalized with the statistics of the observed context plus the
    block's medians of patches `0..j-1`, each median denormalized in its own frame before
    it is folded in. With perfect medians this is causal per-patch normalization of the true
    series.

    Args:
        context: `(R, T)` block context, `T` a multiple of the patch size.
        observation_mask: `(R, T)`.
        sinh_median: `(R, H, patch_size)` the block's median in the last context patch's
            frame with the asinh undone, `(median - loc) / scale`.
        minimum_scale: Scale floor.
    """
    r, n_pred, ps = sinh_median.shape
    cnt, mean, m2, shift = _causal_patch_states(context, observation_mask, ps)
    cnt, mean, m2 = cnt[:, -1], mean[:, -1], m2[:, -1]
    offset = mean  # medians are clamped relative to the context mean, so a large level survives
    clip = mx.finfo(mx.float32).max ** 0.5 / 1e3
    shift = shift[:, 0]
    locs, scales = [], []
    for h in range(n_pred):
        scale = _scale(cnt, m2, minimum_scale)
        locs.append(mean + shift)
        scales.append(scale)
        folded = mx.clip(sinh_median[:, h] * scale[:, None] + (mean - offset)[:, None], -clip, clip) + offset[:, None]
        b_mean = folded.mean(-1)
        b_m2 = ((folded - b_mean[:, None]) ** 2).sum(-1)
        cnt, mean, m2 = welford_combine(cnt, mean, m2, mx.full(cnt.shape, ps, dtype=mx.float32), b_mean, b_m2)
    return mx.stack(locs, axis=1), mx.stack(scales, axis=1)


def instance_norm_stats(
    values: mx.array, observation_mask: mx.array, minimum_scale: float
) -> tuple[mx.array, mx.array, mx.array]:
    """Non-causal per-row mean/std over observed positions, for known-future covariates.

    Rows whose observed values are all 0 or 1 (indicator covariates) keep `loc=0`,
    `scale=1`. Returns `(loc, scale, is_binary)`, each `(R, 1)`.
    """
    mask_f = observation_mask.astype(mx.float32)
    n = mx.maximum(mask_f.sum(axis=-1, keepdims=True), 1.0)
    loc = (values * mask_f).sum(axis=-1, keepdims=True) / n
    var = (((values - loc) * mask_f) ** 2).sum(axis=-1, keepdims=True) / n
    scale = mx.maximum(mx.sqrt(var), minimum_scale)
    is_zero_one = ((values == 0.0) | (values == 1.0) | ~observation_mask).all(axis=-1, keepdims=True)
    is_binary = is_zero_one & observation_mask.any(axis=-1, keepdims=True)
    return mx.where(is_binary, 0.0, loc), mx.where(is_binary, 1.0, scale), is_binary

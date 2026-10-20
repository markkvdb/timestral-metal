"""Attention masks and time positions for Decoupled Horizon Forecasting (DHF) decoding.

A port of `timestral.masks`. The sequence is `[Enc ; Dec]`: encoder tokens embed the observed
context patches; decoder tokens are forecast queries for the horizon patches. All masks are
boolean, True meaning "may attend".
"""

import mlx.core as mx


def _tril(n: int) -> mx.array:
    return mx.tril(mx.ones((n, n), dtype=mx.bool_))


def _eye(n: int) -> mx.array:
    return mx.eye(n, dtype=mx.bool_)


def compute_enc_key_validity(obs_per_patch: mx.array, attend_missing: bool) -> mx.array:
    """`(R, N)` encoder key validity.

    A patch is a valid key if it is observed, or if `attend_missing` is set and the
    row's series has started at or before it. Leading all-unobserved patches are
    left-padding and are never valid keys.
    """
    if not attend_missing:
        return obs_per_patch
    started = mx.cummax(obs_per_patch.astype(mx.int32), axis=-1).astype(mx.bool_)
    return obs_per_patch | started


def enc_causal_mask(key_valid: mx.array) -> mx.array:
    """`(R, L, L)` causal encoder mask over valid keys, with the diagonal always open."""
    ll = key_valid.shape[1]
    return (_tril(ll)[None] & key_valid[:, None, :]) | _eye(ll)[None]


def build_forecast_time_mask(context_len: int, horizon: int, key_valid: mx.array) -> mx.array:
    """`(R, 1, L+h, L+h)` time mask for one block without covariates.

    The encoder is causal; every decoder token reads the whole valid context and its
    own causal decoder chain; the context never reads the horizon.
    """
    rows = key_valid.shape[0]
    enc = enc_causal_mask(key_valid)
    enc_dec = mx.zeros((rows, context_len, horizon), dtype=mx.bool_)
    dec_enc = mx.broadcast_to(key_valid[:, None, :], (rows, horizon, context_len))
    dec_dec = mx.broadcast_to(_tril(horizon)[None], (rows, horizon, horizon))
    top = mx.concatenate([enc, enc_dec], axis=2)
    bottom = mx.concatenate([dec_enc, dec_dec], axis=2)
    return mx.concatenate([top, bottom], axis=1)[:, None]


def build_forecast_positions(context_len: int, horizon: int, first_obs: mx.array) -> mx.array:
    """`(R, L+h)` time positions `0..L+h-1`, anchored at each row's first observed patch."""
    return mx.arange(context_len + horizon)[None, :] - first_obs[:, None]


def compute_block_layout(min_context: mx.array, n_patches: int, block_size: int) -> tuple[mx.array, mx.array]:
    """Per decoder position: the anchor of its block and whether it is active (`p >= s`)."""
    p = mx.arange(n_patches)[None, :]
    s = min_context[:, None]
    active = p >= s
    offset = mx.maximum(p - s, 0)
    anchor = s + block_size * (offset // block_size)
    return mx.where(active, anchor, 0), active


def build_two_stream_time_mask(
    min_context: mx.array,
    n_patches: int,
    block_size: int,
    key_valid: mx.array,
    bidirectional_rows: mx.array | None,
) -> mx.array:
    """`(R, 1, 2N, 2N)` time mask over a full `[Enc ; Dec]` window (the covariate path).

    - Enc to Enc: causal over valid keys; full (non-causal) for `bidirectional_rows`,
      the known-future covariates.
    - Enc to Dec: never.
    - Dec to Enc: a decoder token in the block anchored at `b` reads valid `Enc[j]`, `j < b`.
    - Dec to Dec: causal within the same block; every decoder token reads itself.
    """
    anchor, active = compute_block_layout(min_context, n_patches, block_size)
    rows, n = active.shape
    key_col = key_valid[:, None, :]
    j = mx.arange(n).reshape(1, 1, n)
    eye = _eye(n)[None]
    enc_enc = enc_causal_mask(key_valid)
    if bidirectional_rows is not None:
        enc_bi = mx.broadcast_to(key_col, (rows, n, n)) | eye
        enc_enc = mx.where(bidirectional_rows[:, None, None], enc_bi, enc_enc)
    enc_dec = mx.zeros((rows, n, n), dtype=mx.bool_)
    dec_enc = active[:, :, None] & (j < anchor[:, :, None]) & key_col
    same_block = anchor[:, :, None] == anchor[:, None, :]
    dec_dec = (same_block & _tril(n)[None] & active[:, :, None] & active[:, None, :]) | eye
    top = mx.concatenate([enc_enc, enc_dec], axis=2)
    bottom = mx.concatenate([dec_enc, dec_dec], axis=2)
    return mx.concatenate([top, bottom], axis=1)[:, None]


def variate_firewall(r: int, target_rows: mx.array | None, bidirectional_rows: mx.array | None) -> mx.array | None:
    """`(R, R)` forbidden query-to-key row pairs, or None.

    - A covariate row may not read a target row (covariates are exogenous).
    - A known-future (bidirectional) row may not read a causal row. Without this, a
      covariate's view of the future could relay a target's own future back into it.
    """
    if target_rows is None and bidirectional_rows is None:
        return None
    forbid = mx.zeros((r, r), dtype=mx.bool_)
    if target_rows is not None:
        forbid = forbid | ((~target_rows[:, None]) & target_rows[None, :])
    if bidirectional_rows is not None:
        forbid = forbid | (bidirectional_rows[:, None] & (~bidirectional_rows)[None, :])
    return forbid


def build_two_stream_variate_mask(
    group_ids: mx.array,
    enc_key_valid: mx.array,
    n_dec: int,
    target_rows: mx.array | None = None,
    bidirectional_rows: mx.array | None = None,
) -> mx.array | None:
    """`(N_enc + N_dec, 1, R, R)` variate mask, or None for a single row.

    A row reads another row of the same series; on encoder slots the key row must also
    be a valid key. The firewalls apply on every slot; the diagonal is always open.
    """
    r = enc_key_valid.shape[0]
    if r == 1:
        return None
    row_eq = group_ids[:, None] == group_ids[None, :]
    slot_valid = mx.concatenate([enc_key_valid, mx.ones((r, n_dec), dtype=mx.bool_)], axis=1)
    mask = row_eq[None, None] & slot_valid.T[:, None, None, :]
    forbid = variate_firewall(r, target_rows, bidirectional_rows)
    if forbid is not None:
        mask = mask & ~forbid[None, None]
    return mask | _eye(r)[None, None]


def build_two_stream_positions(n_patches: int, first_obs: mx.array) -> mx.array:
    """`(R, 2N)` positions: encoder and decoder token of the same patch share a time."""
    row = mx.arange(n_patches)[None, :] - first_obs[:, None]
    return mx.concatenate([row, row], axis=-1)


def group_first_observed_patch(obs_per_patch: mx.array, inverse: mx.array, num_groups: int) -> mx.array:
    """`(num_groups,)` index of the first patch any row of the group observes (0 if none)."""
    group_obs = mx.zeros((num_groups, obs_per_patch.shape[-1])).at[inverse].maximum(obs_per_patch.astype(mx.float32))
    return mx.argmax(group_obs, axis=-1)

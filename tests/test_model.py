"""`timestral_metal` models against `timestral` models: one `predict_block` call per path, same weights."""

import mlx.core as mx
import numpy as np
import timestral.loading
import torch

from timestral_metal.loading import load_model
from timestral_metal.model import ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET


def rows(n_rows: int, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return (10 * np.sin(2 * np.pi * t / 24) + rng.standard_normal((n_rows, length)).cumsum(-1)).astype(np.float32)


def assert_close(got: mx.array, want: torch.Tensor, rtol: float = 1e-5) -> None:
    want = want.numpy()
    assert got.shape == want.shape
    assert np.abs(np.array(got) - want).max() <= rtol * np.abs(want).max()


def observation_mask(n_rows: int, length: int) -> np.ndarray:
    obs = np.ones((n_rows, length), dtype=bool)
    obs[1, : length // 3] = False  # late start
    obs[2, length // 2 : length // 2 + 20] = False  # gap
    return obs


def test_predict_block_without_covariates(checkpoint: str) -> None:
    ref, ours = timestral.loading.load_model(checkpoint), load_model(checkpoint)
    ps = ref.config.patch_size
    x, obs = rows(6, 20 * ps, 0), observation_mask(6, 20 * ps)
    x = np.where(obs, x, 0.0).astype(np.float32)
    groups = np.repeat(np.arange(2), 3)
    with torch.inference_mode():
        want = ref.predict_block(torch.from_numpy(x), torch.from_numpy(obs), torch.from_numpy(groups), 3)
    assert_close(ours.predict_block(mx.array(x), mx.array(obs), mx.array(groups), 3), want)


def test_predict_block_with_covariates(checkpoint: str) -> None:
    ref, ours = timestral.loading.load_model(checkpoint), load_model(checkpoint)
    ps, h = ref.config.patch_size, min(3, ref.config.block_size)
    x, obs = rows(6, 20 * ps, 1), observation_mask(6, 20 * ps)
    x = np.where(obs, x, 0.0).astype(np.float32)
    groups = np.repeat(np.arange(2), 3)
    roles = np.array([ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV] * 2)
    known = roles == ROLE_KNOWN_COV
    future = np.where(known[:, None], rows(6, h * ps, 2), 0.0).astype(np.float32)
    future_mask = np.broadcast_to(known[:, None], future.shape).copy()
    with torch.inference_mode():
        want = ref.predict_block(*(torch.from_numpy(a) for a in (x, obs, groups)), h,
                                 *(torch.from_numpy(a) for a in (roles, future, future_mask)))
    got = ours.predict_block(mx.array(x), mx.array(obs), mx.array(groups), h, mx.array(roles), mx.array(future), mx.array(future_mask))
    assert_close(got, want)

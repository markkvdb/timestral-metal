"""`timestral_metal.masks` against `timestral.masks`: every mask and position must match exactly."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from timestral import masks as ref

from timestral_metal import masks

ROWS, N, H, BLOCK = 7, 20, 5, 4
rng = np.random.default_rng(0)
OBS = rng.random((ROWS, N)) < 0.8
OBS[1, :6] = False  # late start
OBS[2, 9:13] = False  # gap
OBS[4] = False  # never observed
GROUPS = np.array([0, 0, 0, 1, 1, 2, 2])
TARGETS = np.array([True, False, False, True, False, True, True])
KNOWN = np.array([False, False, True, False, True, False, False])


def check(got: mx.array | None, want: torch.Tensor | None) -> None:
    assert (got is None) == (want is None)
    if want is not None:
        np.testing.assert_array_equal(np.array(got), want.numpy())


def pair(a: np.ndarray) -> tuple[mx.array, torch.Tensor]:
    return mx.array(a), torch.from_numpy(a)


@pytest.mark.parametrize("attend_missing", [True, False])
def test_context_only_masks(attend_missing: bool) -> None:
    obs_m, obs_t = pair(OBS)
    groups_m, groups_t = pair(GROUPS)
    kv_t = ref.compute_enc_key_validity(obs_t, attend_missing)
    kv_m = masks.compute_enc_key_validity(obs_m, attend_missing)
    check(kv_m, kv_t)
    check(masks.build_forecast_time_mask(N, H, kv_m), ref.build_forecast_time_mask(N, H, kv_t, kv_t.device))
    check(masks.build_two_stream_variate_mask(groups_m, kv_m, H), ref.build_two_stream_variate_mask(groups_t, kv_t, H))
    check(masks.build_two_stream_variate_mask(groups_m[:1], kv_m[:1], H), None)


def test_covariate_window_masks() -> None:
    obs_m, obs_t = pair(OBS)
    kv_m, kv_t = masks.compute_enc_key_validity(obs_m, True), ref.compute_enc_key_validity(obs_t, True)
    min_m, min_t = pair(np.full(ROWS, N - H))
    known_m, known_t = pair(KNOWN)
    targets_m, targets_t = pair(TARGETS)
    for bidirectional_m, bidirectional_t in ((known_m, known_t), (None, None)):
        check(
            masks.build_two_stream_time_mask(min_m, N, BLOCK, kv_m, bidirectional_m),
            ref.build_two_stream_time_mask(min_t, N, BLOCK, kv_t, bidirectional_t),
        )
    check(
        masks.build_two_stream_variate_mask(mx.array(GROUPS), kv_m, N, targets_m, known_m),
        ref.build_two_stream_variate_mask(torch.from_numpy(GROUPS), kv_t, N, targets_t, known_t),
    )


def test_positions() -> None:
    obs_m, obs_t = pair(OBS)
    groups_m, groups_t = pair(GROUPS)
    first_t = ref.group_first_observed_patch(obs_t, groups_t, 3)[groups_t]
    first_m = masks.group_first_observed_patch(obs_m, groups_m, 3)[groups_m]
    check(first_m, first_t)
    check(masks.build_forecast_positions(N, H, first_m), ref.build_forecast_positions(N, H, first_t))
    check(masks.build_two_stream_positions(N, first_m), ref.build_two_stream_positions(N, first_t))

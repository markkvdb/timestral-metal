"""End-to-end forecasts from `timestral_metal` against `timestral`, on the same checkpoints and inputs."""

import functools
import math

import numpy as np
import pytest
import timestral
import timestral.loading

import timestral_metal.loading
from timestral_metal import Forecast, TimestralForecaster


@functools.cache
def forecasters(checkpoint: str) -> tuple[timestral.TimestralForecaster, TimestralForecaster]:
    return timestral.TimestralForecaster.from_pretrained(checkpoint), TimestralForecaster.from_pretrained(checkpoint)


def series(batch: int, variates: int, length: int, seed: int = 0, positive: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    x = 10 * np.sin(2 * math.pi * t / 24) + rng.standard_normal((batch, variates, length)).cumsum(-1)
    return (np.abs(x) + 1 if positive else x).astype(np.float32)


def codes(batch: int, length: int, seed: int) -> np.ndarray:
    c = np.random.default_rng(seed).integers(0, 4, (batch, 1, length)).astype(np.float64)
    c[0, 0, 10:20] = np.nan
    return c


def with_gaps() -> np.ndarray:
    x = series(2, 2, 200, seed=4)
    x[0, 0, 5:9] = np.nan
    x[1, 1, :60] = np.nan  # late start
    return x


CASES = {
    "one block": dict(context=series(2, 1, 100), prediction_length=12),
    "three variates, two blocks on the tiny model": dict(context=series(2, 3, 300, seed=1), prediction_length=50),
    "short context, many blocks": dict(context=series(2, 2, 37, seed=2), prediction_length=70),
    "long horizon": dict(context=series(1, 1, 1500, seed=3)[0, 0], prediction_length=1100),
    "(B, T) input with missing values": dict(context=with_gaps()[:, 0], prediction_length=24),
    "observation mask": dict(context=with_gaps(), prediction_length=24,
                             observation_mask=np.random.default_rng(5).random((2, 2, 200)) > 0.1),
    "past and known covariates": dict(context=series(2, 1, 120, seed=6), prediction_length=24,
                                      past_covariates=series(2, 2, 120, seed=7),
                                      known_covariates=series(2, 1, 144, seed=8)),
    "covariates over several blocks": dict(context=series(1, 1, 300, seed=17), prediction_length=1100,
                                           past_covariates=series(1, 1, 300, seed=18),
                                           known_covariates=series(1, 2, 1400, seed=19)),
    "categorical covariates": dict(context=series(2, 1, 120, seed=9, positive=True), prediction_length=24,
                                   categorical_past_covariates=codes(2, 120, 10),
                                   categorical_known_covariates=codes(2, 144, 11)),
    "categorical codes on two targets": dict(context=series(2, 2, 120, seed=12), prediction_length=24,
                                             categorical_known_covariates=codes(2, 144, 13)),
}


def assert_matches(checkpoint: str, ours: TimestralForecaster | None = None, **kwargs) -> None:
    ref, default = forecasters(checkpoint)
    ours = ours or default
    want = ref.forecast(**kwargs).quantiles.numpy()
    got = ours.forecast(**kwargs)
    assert isinstance(got, Forecast) and isinstance(got.quantiles, np.ndarray)
    assert got.quantiles.shape == want.shape
    assert np.abs(got.quantiles - want).max() <= 1e-4 * np.abs(want).max()


@pytest.mark.parametrize("case", CASES)
def test_forecast_matches_timestral(checkpoint: str, case: str) -> None:
    assert_matches(checkpoint, **CASES[case])


@pytest.mark.parametrize("max_batch_rows", [1, 7, 64])
def test_batch_chunking(checkpoint: str, max_batch_rows: int) -> None:
    """12 series of 3 targets and 1 past covariate: 96 packed rows with symmetric averaging.

    With 64 rows, each call decodes 16 series over the fused variate attention. With 7 or 1 rows, each call
    decodes one series of 4 rows over the plain matmul and softmax.
    """
    assert_matches(checkpoint, TimestralForecaster.from_pretrained(checkpoint, max_batch_rows=max_batch_rows),
                   context=series(12, 3, 200, seed=20), prediction_length=40, past_covariates=series(12, 1, 200, seed=21))


@pytest.mark.parametrize("symmetric_averaging", [True, False])
@pytest.mark.parametrize("clip_nonnegative", [True, False])
def test_clampsym_options(checkpoint: str, symmetric_averaging: bool, clip_nonnegative: bool) -> None:
    assert_matches(checkpoint, context=series(2, 1, 120, seed=14, positive=True), prediction_length=24,
                   past_covariates=series(2, 1, 120, seed=15), known_covariates=series(2, 1, 144, seed=16),
                   symmetric_averaging=symmetric_averaging, clip_nonnegative=clip_nonnegative)


def test_from_pretrained_local_directory_and_hub_repo(tiny_checkpoint: str, monkeypatch) -> None:
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return tiny_checkpoint

    monkeypatch.setattr(timestral_metal.loading, "snapshot_download", fake_snapshot_download)
    x = series(2, 1, 100)
    local = TimestralForecaster.from_pretrained(tiny_checkpoint).forecast(x, prediction_length=12).quantiles
    assert not calls  # an existing directory never touches the Hub
    hub = TimestralForecaster.from_pretrained("org/timestral-tiny", revision="v1", token=False, local_files_only=True)
    assert calls == [("org/timestral-tiny", {"revision": "v1", "cache_dir": None, "token": False,
                                             "local_files_only": True,
                                             "allow_patterns": ["config.json", "model.safetensors"]})]
    np.testing.assert_array_equal(hub.forecast(x, prediction_length=12).quantiles, local)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["three variates, two blocks on the tiny model", "long horizon", "past and known covariates",
                                  "categorical covariates"])
def test_forecast_matches_timestral_123m(case: str) -> None:
    assert_matches("mistralai/Timestral-123M", **CASES[case])

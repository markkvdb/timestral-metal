"""Speed and peak memory of timestral_metal against timestral (PyTorch on the CPU).

Times are the fastest of five runs after a warm-up. Peak memory is MLX's, over those runs.
The year-long covariate cases run on timestral_metal only: timestral's dense attention on them
needs tens of gigabytes. MLX reads `MLX_ENABLE_TF32` once, so run the script once per setting:

    uv run python bench.py
    MLX_ENABLE_TF32=0 uv run python bench.py
"""

import os
import time

import mlx.core as mx
import numpy as np
import timestral

import timestral_metal

rng = np.random.default_rng(0)
panel = rng.standard_normal((2, 3, 2000)).cumsum(-1)
panel[0, 1, :300] = np.nan
single = 50 + rng.standard_normal(3000).cumsum()


def year(steps: int) -> dict:
    t = np.arange(steps + 48)
    daily = np.sin(2 * np.pi * t / 24)
    return dict(
        context=(100 + 30 * daily[:steps] + rng.standard_normal(steps).cumsum())[None],
        prediction_length=48,
        past_covariates=rng.standard_normal((1, 2, steps)),
        known_covariates=np.stack([daily, np.cos(2 * np.pi * t / 168), rng.standard_normal(steps + 48)])[None],
    )


CASES = {
    "2 series x 3 variates, 96 steps": (dict(context=panel, prediction_length=96), True),
    "1 series, 1100 steps": (dict(context=single, prediction_length=1100), True),
    "covariates, 8760-step context": (year(8760), False),
    "covariates, 32768-step context": (year(32768), False),
}


def best_time(fc, kwargs: dict, runs: int = 5) -> float:
    fc.forecast(**kwargs)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fc.forecast(**kwargs)
        times.append(time.perf_counter() - start)
    return min(times)


if __name__ == "__main__":
    print(f"MLX_ENABLE_TF32={os.environ.get('MLX_ENABLE_TF32', 'unset')}, device {mx.default_device()}")
    for repo in ("mistralai/Timestral-5M", "mistralai/Timestral-123M"):
        ours = timestral_metal.TimestralForecaster.from_pretrained(repo)
        ref = timestral.TimestralForecaster.from_pretrained(repo)
        for case, (kwargs, with_ref) in CASES.items():
            mx.reset_peak_memory()
            secs = best_time(ours, kwargs)
            peak = mx.get_peak_memory() / 2**20
            ref_text = f"timestral {best_time(ref, kwargs) * 1e3:8.1f} ms" if with_ref else "timestral   not run"
            print(f"{repo.split('/')[1]:15} {case:32} timestral_metal {secs * 1e3:8.1f} ms, peak {peak:7.0f} MiB | {ref_text}")

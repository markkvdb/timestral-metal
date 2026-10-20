# timestral-metal

`timestral-metal` runs Mistral's [Timestral](https://github.com/mistralai/timestral) forecasting models on Apple silicon with [MLX](https://github.com/ml-explore/mlx). It is a port of `timestral` 1.0.0 that loads the released checkpoints unchanged, supports the same features (including past-only, known-future and categorical covariates), and gives the same forecasts within float32 rounding. It does not need PyTorch.

## Install

```bash
pip install timestral-metal
```

You need a Mac with Apple silicon and Python 3.11 or later.

## Example

```python
import numpy as np
from timestral_metal import TimestralForecaster

forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-123M")
context = np.random.randn(2, 1, 2000).cumsum(-1)  # (series, variates, time); NaN marks a missing value
forecast = forecaster.forecast(context, prediction_length=96)
forecast.median  # numpy array, (2, 1, 96)
```

The API is `timestral`'s, documented in its [README](https://github.com/mistralai/timestral), with two differences: forecasts are numpy arrays, and there is no `device` argument. `TimestralForecaster` and `from_pretrained` also take `max_batch_rows` (see below).

## Performance

Time per forecast on an Apple M5 Pro, from `bench.py`, against `timestral` on the CPU:

| Timestral-123M | timestral | timestral-metal | timestral-metal, full float32 |
|---|---|---|---|
| 2 series x 3 variates, 2,000-step context, 96 steps | 179 ms | 22 ms | 43 ms |
| 1 series, 3,000-step context, 1,100 steps | 173 ms | 20 ms | 34 ms |
| 1 series with 5 covariates, 32,768-step context, 48 steps | not measured | 392 ms | 789 ms |

The covariate case peaks at 2.1 GB of memory. A batch of 128 series of 3 variates, with full float32, takes 2.5 s (20 ms per series) and peaks at 2.3 GB. `max_batch_rows` (default 64) sets how many packed rows of series, variates and covariates go into one model call. Larger batches are decoded in chunks of whole series, which keeps memory bounded.

## Changes from timestral

Besides running on the GPU, `timestral-metal` changes these parts of `timestral`'s computation. The forecasts stay the same within float32 rounding.

- With covariates, the model creates decoder tokens for the horizon patches only. `timestral` creates one for every patch of the window and keeps the predictions of the horizon patches. The other decoder tokens do not affect the forecast. Leaving them out halves the time of covariate forecasts, from 804 ms to 392 ms in the case above.
- The forecast of the context and the forecast of its negation, which symmetric averaging combines, run as one batch instead of two calls. This saves 4 to 5 ms per forecast on Timestral-123M.
- Each attention computes Q, K, V and its output gate with one matrix product, saving up to 0.5 ms per model call on Timestral-123M. `load_model` reorders the checkpoint's QKV weights from (qkv, head_dim, heads) to (qkv, heads, head_dim) and appends the gate weights.
- Variate attention runs over the rows of one model call. For 8 rows or fewer, it uses a plain matrix product and softmax, which is about 3 times faster than MLX's fused attention kernel at that size. For more rows, it uses the fused kernel.
- Apple GPUs have no float64. To keep the normalisation statistics on the GPU, they are computed in float32: each row is shifted by its first observed value before the sums, and per-patch statistics are combined with a log-depth scan. `timestral` computes them in float64.
- `mx.compile` compiles the transformer layers, saving 5 to 10% of their time, and the refine step that runs after each decoded block, saving about 10% on forecasts of more than one block.

The first two changes also apply to `timestral`'s PyTorch code.

## Precision

On M5-class GPUs, MLX computes float32 matrix products at reduced (TF32-class) precision by default. It is faster than full float32 and moves forecasts by up to about 1e-2 of their largest value. For full float32, set `MLX_ENABLE_TF32=0` before your program's first MLX computation:

```bash
MLX_ENABLE_TF32=0 python my_forecast.py
```

Forecasts then match `timestral`'s within 1e-4 of their largest value.

## Development

```bash
uv run pytest            # compare every module with timestral, on a tiny model and Timestral-5M
uv run pytest -m slow    # compare forecasts on Timestral-123M
uv run python bench.py   # time and memory against timestral
```

## Licence

Apache License 2.0. The code is a port of `timestral` (Copyright Mistral AI); see `NOTICE`. The model weights are distributed by Mistral AI under their own licence on the [Hugging Face Hub](https://huggingface.co/mistralai/Timestral-123M).

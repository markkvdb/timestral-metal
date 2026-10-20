import os

# The comparisons with timestral need full float32 matmuls. MLX reads this once, at its first computation.
os.environ["MLX_ENABLE_TF32"] = "0"

import pytest
import torch
from safetensors.torch import save_file
from timestral import TimestralConfig, TimestralModel

KNOTS = (0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> str:
    """Directory of a small random Timestral checkpoint, built as upstream's tests build theirs."""
    config = TimestralConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=("time", "time_variate", "time", "time_variate"),
        quantile_knots=KNOTS,
        block_size=4,
        context_length=256,
    )
    torch.manual_seed(0)
    model = TimestralModel(config)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    path = tmp_path_factory.mktemp("tiny")
    config.save(path / "config.json")
    save_file(model.state_dict(), path / "model.safetensors")
    return str(path)


@pytest.fixture(params=["tiny", "mistralai/Timestral-5M"], scope="session")
def checkpoint(request, tiny_checkpoint) -> str:
    """The tiny random checkpoint, then the released Timestral-5M."""
    return tiny_checkpoint if request.param == "tiny" else request.param

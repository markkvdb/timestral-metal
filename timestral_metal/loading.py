"""Load exported checkpoints: a directory with `config.json` and `model.safetensors`, either on
disk or in a Hugging Face Hub repo."""

from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HFValidationError, RepositoryNotFoundError

from timestral_metal.config import TimestralConfig
from timestral_metal.model import TimestralModel

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def resolve_checkpoint(
    path_or_repo_id: str | Path,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return a local directory holding `config.json` and `model.safetensors`.

    An existing directory is used as is. Anything else is a Hugging Face Hub repo id
    (e.g. `org/timestral-108m`): its two files are downloaded to the Hub cache, or read from it
    when `local_files_only` (or `HF_HUB_OFFLINE=1`) is set.
    """
    path = Path(path_or_repo_id).expanduser()
    if path.is_dir():
        return path
    if path.exists():
        raise NotADirectoryError(
            f"{str(path)!r} is a file; pass the export directory holding {CONFIG_FILE} and {WEIGHTS_FILE}"
        )
    try:
        return Path(
            snapshot_download(
                str(path_or_repo_id),
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                allow_patterns=[CONFIG_FILE, WEIGHTS_FILE],
            )
        )
    except GatedRepoError:
        raise  # the Hub's message says how to request access
    except (RepositoryNotFoundError, HFValidationError) as e:
        raise FileNotFoundError(
            f"{str(path_or_repo_id)!r} is neither a local directory nor a Hugging Face Hub repo you can access"
        ) from e


def merge_attention_weights(weights: dict[str, mx.array], config: TimestralConfig) -> dict[str, mx.array]:
    """Merge each attention's `wQKV` and `_gate` weights into the `proj` of `timestral_metal.layers.Attention`.

    The released `wQKV` lays out its outputs as (qkv, head_dim, heads); `proj` has them as
    (qkv, heads, head_dim), followed by the gate.
    """
    d, heads = config.d_model, config.num_heads
    rows = mx.arange(3 * d).reshape(3, d // heads, heads).transpose(0, 2, 1).reshape(-1)
    for name in [k for k in weights if k.endswith(".wQKV.weight")]:
        prefix = name.removesuffix(".wQKV.weight")
        for kind in ("weight", "bias"):
            qkv = weights.pop(f"{prefix}.wQKV.{kind}", None)
            if qkv is None:
                continue
            gate = weights.pop(f"{prefix}._gate.{kind}", None)
            weights[f"{prefix}.proj.{kind}"] = qkv[rows] if gate is None else mx.concatenate([qkv[rows], gate])
    return weights


def load_model(path_or_repo_id: str | Path, **hub_kwargs) -> TimestralModel:
    """Build the model from `config.json` and load `model.safetensors`.

    `path_or_repo_id` is a local directory or a Hugging Face Hub repo id; `hub_kwargs` go to
    `resolve_checkpoint`. Weights load in float32 with `strict=True`.
    """
    path = resolve_checkpoint(path_or_repo_id, **hub_kwargs)
    config = TimestralConfig.load(path / CONFIG_FILE)
    model = TimestralModel(config)
    weights = {k: v.astype(mx.float32) for k, v in mx.load(str(path / WEIGHTS_FILE)).items()}
    model.load_weights(list(merge_attention_weights(weights, config).items()), strict=True)
    mx.eval(model.parameters())
    return model.eval()

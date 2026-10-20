"""Transformer layers for Timestral inference.

A port of `timestral.layers` to `mlx.nn`. All arrays use the packed layout: one row per variate of
every series in the batch, `(R, N, D)` with `N` patches. Time attention runs per row over patches;
variate attention runs per patch over the rows of the same series.

Parameter names match the released weights except in `Attention`, where
`timestral_metal.loading.load_model` merges `wQKV` and `_gate` into `proj`.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from timestral_metal.config import LAYER_KIND_AXES


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.scale = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.scale, self.eps)


class ResidualBlock(nn.Module):
    """`output(silu(hidden(x))) + residual(x)`, used by the patch embedding and the head."""

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, use_bias: bool) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(input_dims, hidden_dims, bias=use_bias)
        self.output_layer = nn.Linear(hidden_dims, output_dims, bias=use_bias)
        self.residual_layer = nn.Linear(input_dims, output_dims, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output_layer(nn.silu(self.hidden_layer(x))) + self.residual_layer(x)


class FusedSwiGLU(nn.Module):
    """SwiGLU MLP with one fused input projection; computes `gate * silu(up)`."""

    def __init__(self, embed_dim: int, hidden_dim: int, use_bias: bool) -> None:
        super().__init__()
        self.w12 = nn.Linear(embed_dim, 2 * hidden_dim, bias=use_bias)
        self.w3 = nn.Linear(hidden_dim, embed_dim, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.w12(x), 2, axis=-1)
        return self.w3(gate * nn.silu(up))


class SinCosRoPE(nn.Module):
    """Rotary position embedding (sin/cos concatenation), with partial rotation.

    Only the first `rotary_dim` dimensions of each head rotate. The inverse-frequency
    table is zero-padded past `rotary_dim / 2`, so the remaining dimensions get `cos=1`,
    `sin=0`, which is the identity rotation.
    """

    def __init__(self, dim: int, base: float, rotary_dim: int) -> None:
        super().__init__()
        assert 0 < rotary_dim <= dim and rotary_dim % 2 == 0, f"invalid rotary_dim={rotary_dim} for dim={dim}"
        inv_freq = np.zeros(dim // 2, dtype=np.float32)
        idx = np.arange(0, rotary_dim, 2, dtype=np.float32)
        inv_freq[: rotary_dim // 2] = 1.0 / (np.float32(base) ** (idx / np.float32(rotary_dim)))
        self.inv_freq = inv_freq  # numpy, so it is not a model parameter; the checkpoint does not store it

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        """Return `(cos, sin)` tables, each `(R, N, head_dim)`, for integer `position_ids` `(R, N)`."""
        freqs = position_ids.astype(mx.float32)[..., None] * mx.array(self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate `x` `(R, heads, N, head_dim)` with tables `(R, 1, N, head_dim)`."""
    half = x.shape[-1] // 2
    return x * cos + mx.concatenate([-x[..., half:], x[..., :half]], axis=-1) * sin


class Attention(nn.Module):
    """Multi-head attention over the time axis (`is_time=True`) or the variate axis.

    Time attention applies RoPE; variate attention has no positional encoding. With the
    `sdpa_sigmoid` gate, the attention output is multiplied by `sigmoid(x @ W_g)` before
    the output projection.

    `proj` computes Q, K, V and the gate in one matmul. Its output features are Q, K and V,
    each laid out (heads, head_dim), then the gate.
    """

    def __init__(self, embed_dim: int, num_heads: int, is_time: bool, use_bias: bool, gate: str) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.is_time = is_time
        self.gated = gate == "sdpa_sigmoid"
        self.proj = nn.Linear(embed_dim, (4 if self.gated else 3) * embed_dim, bias=use_bias)
        self.wO = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def __call__(self, x: mx.array, mask: mx.array | None, cos: mx.array, sin: mx.array) -> mx.array:
        rows, n, d = x.shape
        hd = d // self.num_heads
        proj = self.proj(x)
        qkv = proj[..., : 3 * d].reshape(rows, n, 3, self.num_heads, hd)
        if self.is_time:
            qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, R, heads, N, head_dim)
            q, k = _apply_rotary(qkv[0], cos, sin), _apply_rotary(qkv[1], cos, sin)
            out = mx.fast.scaled_dot_product_attention(q, k, qkv[2], scale=hd**-0.5, mask="causal" if mask is None else mask)
            out = out.transpose(0, 2, 1, 3)
        else:
            # Sequences of R rows are too short for the fused attention kernel to pay off.
            qkv = qkv.transpose(2, 1, 3, 0, 4)  # (3, N, heads, R, head_dim)
            scores = (qkv[0] @ qkv[1].swapaxes(-1, -2)) * hd**-0.5
            if mask is not None:
                scores = mx.where(mask, scores, -mx.inf)
            out = (mx.softmax(scores, axis=-1) @ qkv[2]).transpose(2, 0, 1, 3)
        out = out.reshape(rows, n, d)
        if self.gated:
            out = out * mx.sigmoid(proj[..., 3 * d :])
        return self.wO(out)


class TransformerBlock(nn.Module):
    """The attention(s) of one layer kind, in order, each a pre-norm residual, then one MLP."""

    def __init__(
        self,
        kind: str,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
    ) -> None:
        super().__init__()
        axes = LAYER_KIND_AXES[kind]
        self.attn_norms = [RMSNorm(embed_dim, norm_eps) for _ in axes]
        self.attns = [
            Attention(embed_dim, num_heads, is_time, attn_bias, time_attn_gate if is_time else variate_attn_gate)
            for is_time in axes
        ]
        self.mlp_norm = RMSNorm(embed_dim, norm_eps)
        self.mlp = FusedSwiGLU(embed_dim, mlp_hidden_dim, mlp_bias)

    def __call__(
        self, x: mx.array, time_mask: mx.array | None, variate_mask: mx.array | None, cos: mx.array, sin: mx.array
    ) -> mx.array:
        for norm, attn in zip(self.attn_norms, self.attns):
            x = x + attn(norm(x), time_mask if attn.is_time else variate_mask, cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(
        self,
        block_schedule: tuple[str, ...],
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        rope_theta: float,
        rope_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
    ) -> None:
        super().__init__()
        self.layers = [
            TransformerBlock(
                kind=kind,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                norm_eps=norm_eps,
                attn_bias=attn_bias,
                mlp_bias=mlp_bias,
                time_attn_gate=time_attn_gate,
                variate_attn_gate=variate_attn_gate,
            )
            for kind in block_schedule
        ]
        self.rope = SinCosRoPE(dim=embed_dim // num_heads, base=rope_theta, rotary_dim=rope_dim)
        self.out_norm = RMSNorm(embed_dim, norm_eps)

    def __call__(
        self, x: mx.array, time_mask: mx.array | None, variate_mask: mx.array | None, positions: mx.array
    ) -> mx.array:
        """Run the stack on `x` `(R, N, D)` with integer time `positions` `(R, N)`.

        `time_mask` is `(R, 1, N, N)` (None for causal) and `variate_mask` `(N, 1, R, R)` (None for full).
        """
        assert positions.shape == x.shape[:2], f"positions {tuple(positions.shape)} do not match x {tuple(x.shape)}"
        cos, sin = self.rope(positions)
        cos, sin = cos[:, None], sin[:, None]  # broadcast over heads
        for layer in self.layers:
            x = layer(x, time_mask, variate_mask, cos, sin)
        return self.out_norm(x)

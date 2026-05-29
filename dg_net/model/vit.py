"""MAE-style Vision Transformer blocks for reproducible SSL experiments.

The implementation follows the block structure used by MAE-style ViTs:
pre-norm attention followed by a residual MLP branch, with no pretrained
weights or external model dependency.

Run ``python vit.py`` to write ``output/vit.log``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Transformer feed-forward network used by ViT/MAE blocks."""

    def __init__(self, in_features: int, hidden_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        hidden_features = hidden_features or 4 * in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Attention(nn.Module):
    """Multi-head self-attention with the qkv projection used in MAE ViTs."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_drop(self.proj(x))


class ViTBlock(nn.Module):
    """MAE-compatible pre-normalized Vision Transformer building block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def build_vit_block(**kwargs: object) -> ViTBlock:
    """Construct a randomly initialized ViT block for SSL training."""

    return ViTBlock(**kwargs)


def parameter_summary(module: nn.Module) -> tuple[int, float]:
    parameters = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    bytes_used = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return parameters, bytes_used / (1024**2)


def write_demo_log(output_path: str | Path | None = None) -> Path:
    """Validate block construction/forward and write the required log report."""

    path = Path(output_path) if output_path else Path(__file__).resolve().parent / "output" / "vit.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    block = ViTBlock(dim=192, num_heads=3, mlp_ratio=4.0)
    output = block(torch.randn(2, 196, 192))
    parameters, size_mib = parameter_summary(block)
    lines = [
        "ViT block construction successful (MAE-style, random initialization; pretrained=False).",
        f"parameters: {parameters:,}",
        f"parameter_size_mib: {size_mib:.4f}",
        f"output_shape: {tuple(output.shape)}",
        str(block),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"Wrote ViT verification log: {write_demo_log()}")

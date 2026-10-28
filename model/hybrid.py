"""Independent hybrid transformer block for DGNet's degradation generator.

Architecture:
``LayerNorm -> MHSA -> residual -> LayerNorm -> CNN -> residual ->
LayerNorm -> FFN -> residual``.

Run ``python hybrid.py`` to write ``output/hybrid.log``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenCNNBranch(nn.Module):
    """Local depthwise/pointwise branch without a residual of its own."""

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to retain the patch grid.")
        self.dim = dim
        self.depthwise = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        batch, tokens, dim = x.shape
        height, width = grid_size
        if dim != self.dim or tokens != height * width:
            raise ValueError(
                f"Expected tokens [B, {height * width}, {self.dim}], got {tuple(x.shape)}."
            )
        x = x.transpose(1, 2).reshape(batch, dim, height, width)
        x = self.dropout(self.act(self.pointwise(self.depthwise(x))))
        return x.flatten(2).transpose(1, 2)


class HybridBlock(nn.Module):
    """Global self-attention plus local convolutional degradation block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.attn_drop = nn.Dropout(proj_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cnn = TokenCNNBranch(dim, kernel_size=cnn_kernel_size, dropout=cnn_dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=ffn_dropout)

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        normalized = self.norm1(x)
        attention, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + self.attn_drop(attention)
        x = x + self.cnn(self.norm2(x), grid_size)
        return x + self.ffn(self.norm3(x))


# The legacy prototype used this public name; retain it for easy migration.
HybridTransformerLayer = HybridBlock


def build_hybrid_block(**kwargs: object) -> HybridBlock:
    return HybridBlock(**kwargs)


def parameter_summary(module: nn.Module) -> tuple[int, float]:
    parameters = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    bytes_used = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return parameters, bytes_used / (1024**2)


def write_demo_log(output_path: str | Path | None = None) -> Path:
    """Execute a dummy block forward and persist construction evidence."""

    path = (
        Path(output_path)
        if output_path
        else Path(__file__).resolve().parent / "output" / "hybrid.log"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    block = HybridBlock(dim=192, num_heads=3, mlp_ratio=4.0)
    output = block(torch.randn(2, 196, 192), (14, 14))
    parameters, size_mib = parameter_summary(block)
    lines = [
        "Hybrid block construction successful (random initialization; pretrained=False).",
        "flow: LayerNorm -> MHSA -> Residual -> LayerNorm -> CNN -> Residual -> LayerNorm -> FFN -> Residual",
        f"parameters: {parameters:,}",
        f"parameter_size_mib: {size_mib:.4f}",
        f"output_shape: {tuple(output.shape)}",
        str(block),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"Wrote hybrid verification log: {write_demo_log()}")

"""Independent CNN building blocks for degradation-network experiments.

The module intentionally builds untrained blocks rather than downloading
pretrained networks.  ``DGNet`` consumes :class:`TokenCNNBlock`, while the
feature-map blocks are exposed for direct ResNet/ConvNeXt experiments.

Run ``python cnn.py`` to write a construction/shape report to
``output/cnn.log``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn


class DropPath(nn.Module):
    """Per-sample stochastic depth used by modern residual blocks."""

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


class ResNetBlock(nn.Module):
    """Untrained ResNet basic block operating on ``[B, C, H, W]`` maps."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ConvNeXtBlock(nn.Module):
    """Untrained ConvNeXt block operating on ``[B, C, H, W]`` maps."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ConvNeXt kernel_size must be odd.")
        self.depthwise = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return identity + self.drop_path(x)


def build_cnn_block(architecture: str, **kwargs: object) -> nn.Module:
    """Construct an untrained feature-map CNN block by architecture name."""

    key = architecture.strip().upper().replace("-", "")
    if key in {"RESNET", "RESNETBLOCK"}:
        return ResNetBlock(**kwargs)
    if key in {"CONVNEXT", "CONVNEXTBLOCK", "CONXNET"}:
        return ConvNeXtBlock(**kwargs)
    raise ValueError(f"Unsupported CNN architecture: {architecture!r}. Use RESNET or CONVNEXT.")


class TokenCNNBlock(nn.Module):
    """Apply a selectable CNN block to patch tokens ``[B, N, D]``."""

    def __init__(
        self,
        dim: int,
        architecture: str = "CONVNEXT",
        kernel_size: int = 7,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        key = architecture.strip().upper().replace("-", "")
        if key == "RESNET":
            self.block = ResNetBlock(dim, dim)
        elif key in {"CONVNEXT", "CONXNET"}:
            self.block = ConvNeXtBlock(dim, kernel_size=kernel_size, drop_path=drop_path)
        else:
            raise ValueError("TokenCNNBlock architecture must be RESNET or CONVNEXT.")

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected tokens [B, N, D], got shape {tuple(x.shape)}.")
        batch, tokens, dim = x.shape
        height, width = grid_size
        if dim != self.dim or tokens != height * width:
            raise ValueError(
                f"Expected dim={self.dim} and tokens={height * width}, got {dim} and {tokens}."
            )
        features = x.transpose(1, 2).reshape(batch, dim, height, width)
        features = self.block(features)
        return features.flatten(2).transpose(1, 2)


def parameter_summary(module: nn.Module) -> tuple[int, float]:
    """Return trainable parameter count and parameter memory in MiB."""

    parameters = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    bytes_used = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return parameters, bytes_used / (1024**2)


def write_demo_log(output_path: str | Path | None = None) -> Path:
    """Create blocks, execute dummy forwards, and persist a readable report."""

    path = Path(output_path) if output_path else Path(__file__).resolve().parent / "output" / "cnn.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    feature_input = torch.randn(2, 32, 8, 8)
    token_input = torch.randn(2, 64, 32)
    blocks = {
        "ResNetBlock": ResNetBlock(32, 32),
        "ConvNeXtBlock": ConvNeXtBlock(32),
        "TokenCNNBlock(CONVNEXT)": TokenCNNBlock(32, architecture="CONVNEXT"),
    }
    lines = ["CNN block construction successful (random initialization; pretrained=False)."]
    for name, block in blocks.items():
        result = (
            block(token_input, (8, 8))
            if isinstance(block, TokenCNNBlock)
            else block(feature_input)
        )
        parameters, size_mib = parameter_summary(block)
        lines.extend(
            [
                f"\n{name}",
                f"parameters: {parameters:,}",
                f"parameter_size_mib: {size_mib:.4f}",
                f"output_shape: {tuple(result.shape)}",
                str(block),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"Wrote CNN verification log: {write_demo_log()}")

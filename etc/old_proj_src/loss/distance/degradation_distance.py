from __future__ import annotations

import torch
import torch.nn as nn

# from src.loss.loss_config import DistanceConfig
from src.config.DG_config import DistanceConfig

class DegradationDistance(nn.Module):
    """
    D_mask = ((x - t) - m)^2

    x : [B, C, H, W]
    m : [B, C, H, W]
    t : scalar threshold
    """

    def __init__(self, cfg: DistanceConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        threshold: float | None = None,
    ) -> torch.Tensor:
        if x.shape != m.shape:
            raise ValueError(f"Shape mismatch: x={tuple(x.shape)} vs m={tuple(m.shape)}")

        t = self.cfg.mask_threshold if threshold is None else threshold
        target = x - t
        loss = (target - m) ** 2
        loss = loss.mean(dim=(1, 2, 3))   # [B]
        return loss
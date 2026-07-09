from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.loss.loss_config import DistanceConfig
from src.config.DG_config import DistanceConfig


class RepresentationDistance(nn.Module):
    """
    D_representation = 1 - cosine_similarity(z, z_dg)

    Input:
        z     : [B, D]
        z_dg  : [B, D]

    Output:
        [B]
    """

    def __init__(self, cfg: DistanceConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, z: torch.Tensor, z_dg: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z_dg.ndim != 2:
            raise ValueError(
                f"RepresentationDistance expects 2D tensors [B, D], "
                f"but got z={tuple(z.shape)}, z_dg={tuple(z_dg.shape)}"
            )
        if z.shape != z_dg.shape:
            raise ValueError(
                f"Shape mismatch: z={tuple(z.shape)} vs z_dg={tuple(z_dg.shape)}"
            )

        z = F.normalize(z, dim=-1, eps=self.cfg.eps)
        z_dg = F.normalize(z_dg, dim=-1, eps=self.cfg.eps)

        cos = torch.sum(z * z_dg, dim=-1)   # [B]
        d_rep = 1.0 - cos                   # [B]
        return d_rep
from __future__ import annotations

import torch
import torch.nn as nn

# from src.loss.loss_config import RegluationConfig
from src.config.DG_config import RegluationConfig

class DegradationRegluation(nn.Module):
    """
    R_deg = lambda_tv * TV(m) + lambda_sparsity * sparsity(x, m)

    Returned value is per-batch tensor [B].
    Outer lambda_deg is applied in the final total loss,
    not inside this module.
    """

    def __init__(self, cfg: RegluationConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def total_variation(m: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(m[:, :, 1:, :] - m[:, :, :-1, :]).mean(dim=(1, 2, 3))
        dw = torch.abs(m[:, :, :, 1:] - m[:, :, :, :-1]).mean(dim=(1, 2, 3))
        return dh + dw  # [B]

    @staticmethod
    def sparsity(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return torch.abs(x - m).mean(dim=(1, 2, 3))  # [B]

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        if x.shape != m.shape:
            raise ValueError(f"Shape mismatch: x={tuple(x.shape)} vs m={tuple(m.shape)}")

        device = x.device
        B = x.shape[0]
        total = torch.zeros(B, device=device, dtype=x.dtype)

        if self.cfg.lambda_tv != 0.0:
            total = total + self.cfg.lambda_tv * self.total_variation(m)

        if self.cfg.lambda_sparsity != 0.0:
            total = total + self.cfg.lambda_sparsity * self.sparsity(x, m)

        return total
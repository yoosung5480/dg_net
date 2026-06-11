from __future__ import annotations

import torch
import torch.nn as nn

# from src.loss.loss_config import RegluationConfig
from src.config.DG_config import RegluationConfig


class RepresentationRegluation(nn.Module):
    """
    R_rep = ReLU(D_r - t) * exp(k * (D_r - t))

    D_r must be batch-wise tensor [B].
    """

    def __init__(self, cfg: RegluationConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, d_rep: torch.Tensor) -> torch.Tensor:
        if d_rep.ndim != 1:
            raise ValueError(f"Expected d_rep [B], got {tuple(d_rep.shape)}")

        diff = d_rep - self.cfg.rep_threshold
        relu = torch.relu(diff)
        penalty = relu * torch.exp(self.cfg.rep_exp_k * relu)
        return penalty
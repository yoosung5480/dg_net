# src/loss/regluation/adios_regluation.py

from __future__ import annotations

import math
import torch
import torch.nn as nn


class AdiosRegulation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def adios_sparsity_penalty(self, mask, eps=1e-6):
        """
        mask: [B,1,H,W] or [B,C,H,W]
        """
        ratio = mask.mean(dim=(1,2,3))   # [B]
        penalty = 1.0 / (torch.sin(math.pi * ratio).clamp_min(eps))
        return penalty.mean()
# loss.ADIOS_loss.py
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

# from src.loss.loss_config import LossConfig
from src.config.DG_config import LossConfig

from src.loss.distance.degradation_distance import DegradationDistance
from src.loss.regluation.degradation_regluation import DegradationRegluation
from src.loss.regluation.representation_regluation import RepresentationRegluation
from src.loss.regluation.adios_regluation import AdiosRegulation

from src.loss.distance.representation_distance import RepresentationDistance
from src.loss.distance.reconstruction_distance import ReconstructionDistance

def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def _zero_like_batch(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

class AELoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.rep_distance = RepresentationDistance(cfg.distance)

    def forward(self, *, z_x, z_x_dg, x):
        details = {}
        total = _zero_like_batch(x)

        # -----------------------------
        # representation
        # -----------------------------
        d_rep = self.rep_distance(z_x, z_x_dg)
        total = total + d_rep
        details["d_rep"] = d_rep.detach()
       
        # -----------------------------
        # final reduction
        # -----------------------------
        loss = _reduce_loss(total, self.cfg.reduction)
        details["loss"] = loss.detach()
        return loss, details

class DegradationLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.rep_distance = RepresentationDistance(cfg.distance)
        self.panelty = AdiosRegulation()                             #  추가

    def forward(self, *,  z_x, z_x_dg, x_m, x):
        ''' 

        '''

        details = {}
        total = _zero_like_batch(x)
        # -----------------------------
        # 1) representation adversarial
        # -----------------------------
        d_rep = self.rep_distance(z_x, z_x_dg)
        total = total + d_rep       

        p = self.panelty(x_m)
        total = total + p

        # -----------------------------
        # final reduction
        # -----------------------------
        loss = _reduce_loss(total, self.cfg.reduction)

        details["loss"] = loss.detach()

        return loss, details


def build_dg_learn_loss(cfg: LossConfig) -> tuple[AELoss, DegradationLoss]:
    ae_loss = AELoss(cfg)
    degradation_loss = DegradationLoss(cfg)
    return ae_loss, degradation_loss
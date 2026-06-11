from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

# from src.loss.loss_config import LossConfig
from src.config.DG_config import LossConfig

from src.loss.distance.degradation_distance import DegradationDistance
from src.loss.regluation.degradation_regluation import DegradationRegluation
from src.loss.regluation.representation_regluation import RepresentationRegluation

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
        self.rec_distance = ReconstructionDistance(cfg.distance)

    def forward(self, *, pred, z_x, z_x_dg, x):
        details = {}
        total = _zero_like_batch(x)
        active_terms = []

        # -----------------------------
        # representation
        # -----------------------------
        if self.cfg.distance.alpha_r != 0.0:
            d_rep = self.rep_distance(z_x, z_x_dg)
            total = total + self.cfg.distance.alpha_r * d_rep

            details["d_rep"] = d_rep.detach()
            active_terms.append("alpha_r * D_r")

        # -----------------------------
        # reconstruction
        # -----------------------------
        if self.cfg.distance.alpha_p != 0.0:
            rec_out = self.rec_distance(pred, x)

            rec_mode = self.cfg.distance.ae_recon_mode
            if rec_mode not in ("D_p", "D_p_hat"):
                raise ValueError(f"Invalid ae_recon_mode: {rec_mode}")

            d_rec = rec_out[rec_mode]
            total = total + self.cfg.distance.alpha_p * d_rec

            details["ae_recon_mode"] = rec_mode
            details["d_rec"] = rec_out["D_p"].detach()
            if "D_p_hat" in rec_out:
                details["d_rec_hat"] = rec_out["D_p_hat"].detach()

            active_terms.append(f"alpha_p * {rec_mode}")

        # -----------------------------
        # all terms off -> backward-safe zero
        # -----------------------------
        if len(active_terms) == 0:
            zero = pred.sum() * 0.0
            if self.cfg.reduction == "none":
                loss = torch.zeros_like(total)
            else:
                loss = zero

            details["loss"] = loss.detach()
            details["active_terms"] = []
            return loss, details

        # -----------------------------
        # final reduction
        # -----------------------------
        loss = _reduce_loss(total, self.cfg.reduction)
        details["loss"] = loss.detach()
        details["active_terms"] = active_terms

        return loss, details

class DegradationLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

        self.rep_distance = RepresentationDistance(cfg.distance)
        self.rec_distance = ReconstructionDistance(cfg.distance)
        self.mask_distance = DegradationDistance(cfg.distance)

        self.deg_reg = DegradationRegluation(cfg.regluation)
        self.rep_reg = RepresentationRegluation(cfg.regluation)

    def forward(self, *, pred, z_x, z_x_dg, x, x_dg, z_mask):
        details = {}
        total = _zero_like_batch(x)
        active_terms = []

        d_rep = None

        # -----------------------------
        # 1) representation adversarial
        # -----------------------------
        if self.cfg.distance.beta_r != 0.0:
            d_rep = self.rep_distance(z_x, z_mask)
            total = total + self.cfg.distance.beta_r * d_rep        # '-' => '+', 26.04.09, z_x_dg -> mask, 26.04.10

            details["d_rep"] = d_rep.detach()
            active_terms.append("beta_r * D_r")

        # -----------------------------
        # 2) reconstruction
        # -----------------------------
        if self.cfg.distance.beta_p != 0.0:
            rec_out = self.rec_distance(pred, x)

            mode = self.cfg.distance.dg_recon_mode
            if mode not in ("D_p", "D_p_hat"):
                raise ValueError(f"Invalid dg_recon_mode: {mode}")

            d_rec = rec_out[mode]
            total = total + self.cfg.distance.beta_p * d_rec

            details["dg_recon_mode"] = mode
            details["d_rec"] = rec_out["D_p"].detach()
            if "D_p_hat" in rec_out:
                details["d_rec_hat"] = rec_out["D_p_hat"].detach()

            active_terms.append(f"beta_p * {mode}")

        # -----------------------------
        # 3) mask distance
        # -----------------------------
        if self.cfg.distance.beta_m != 0.0:
            d_mask = self.mask_distance(x, x_dg)
            total = total + self.cfg.distance.beta_m * d_mask

            details["d_mask"] = d_mask.detach()
            active_terms.append("beta_m * D_m")

        # -----------------------------
        # 4) degradation regulation
        # -----------------------------
        if self.cfg.regluation.lambda_deg != 0.0:
            r_deg = self.deg_reg(x, x_dg)
            total = total + self.cfg.regluation.lambda_deg * r_deg

            details["r_deg"] = r_deg.detach()
            active_terms.append("lambda_deg * R_deg")

        # -----------------------------
        # 5) representation regulation
        # -----------------------------
        if self.cfg.regluation.lambda_rep != 0.0:
            if d_rep is None:
                d_rep = self.rep_distance(z_x, z_x_dg)

            r_rep = self.rep_reg(d_rep)
            total = total + self.cfg.regluation.lambda_rep * r_rep

            details["r_rep"] = r_rep.detach()
            active_terms.append("lambda_rep * R_rep")

        # -----------------------------
        # all off → backward-safe zero
        # -----------------------------
        if len(active_terms) == 0:
            zero = x_dg.sum() * 0.0
            if self.cfg.reduction == "none":
                loss = torch.zeros_like(total)
            else:
                loss = zero

            details["loss"] = loss.detach()
            details["active_terms"] = []
            return loss, details

        # -----------------------------
        # final reduction
        # -----------------------------
        loss = _reduce_loss(total, self.cfg.reduction)

        details["loss"] = loss.detach()
        details["active_terms"] = active_terms

        return loss, details


def build_dg_learn_loss(cfg: LossConfig) -> tuple[AELoss, DegradationLoss]:
    ae_loss = AELoss(cfg)
    degradation_loss = DegradationLoss(cfg)
    return ae_loss, degradation_loss
from __future__ import annotations

import torch
import torch.nn as nn

# from src.loss.loss_config import DistanceConfig
from src.config.DG_config import DistanceConfig


class ReconstructionDistance(nn.Module):
    """
    D_pixel = (x_recon - x)^2

    Here x_recon is MAE-style patch prediction:
        pred: [B, N, patch_dim]
    and x is image:
        x: [B, C, H, W]

    We patchify x to compare in the same space.
    """

    def __init__(self, cfg: DistanceConfig):
        super().__init__()
        self.cfg = cfg

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: [B, C, H, W]
        x:    [B, N, patch_size**2 * C]
        """
        p = self.cfg.patch_size
        c = self.cfg.in_chans

        if imgs.ndim != 4:
            raise ValueError(f"Expected imgs [B, C, H, W], got {tuple(imgs.shape)}")

        B, C, H, W = imgs.shape
        if C != c:
            raise ValueError(f"Expected in_chans={c}, but got C={C}")
        if H != W:
            raise ValueError(f"Only square images supported, got H={H}, W={W}")
        if H % p != 0:
            raise ValueError(f"Image size {H} must be divisible by patch size {p}")

        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def normalize_patch_targets(self, target: torch.Tensor) -> torch.Tensor:
        """
        target: [B, N, patch_dim]
        """
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True, unbiased=False)
        target = (target - mean) / torch.sqrt(var + 1.0e-6)
        return target

    def forward(self, pred: torch.Tensor, x: torch.Tensor):
        if pred.ndim != 3:
            raise ValueError(f"Expected pred [B, N, patch_dim], got {tuple(pred.shape)}")

        target = self.patchify(x)

        if self.cfg.use_norm_pix_loss:
            target = self.normalize_patch_targets(target)

        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction and target shape mismatch: "
                f"pred={tuple(pred.shape)} vs target={tuple(target.shape)}"
            )

        # -----------------------------
        # D_p (original)
        # -----------------------------
        loss = (pred - target) ** 2
        D_p = loss.mean(dim=(1, 2))   # [B]

        # -----------------------------
        # D^_p (target-based)
        # -----------------------------
        target_p = self.cfg.recon_target
        D_p_hat = torch.abs(D_p - target_p)             # 큰~~변화!

        return {
            "D_p": D_p,
            "D_p_hat": D_p_hat
        }
# from __future__ import annotations

# from dataclasses import dataclass, field


# from dataclasses import dataclass, field


# @dataclass
# class DistanceConfig:
#     # AE loss weights
#     alpha_r: float = 1.0
#     alpha_p: float = 1.0

#     # Degradation loss weights
#     beta_r: float = 1.0
#     beta_p: float = 1.0
#     beta_m: float = 0.0

#     # 🔥 추가: recon mode 선택
#     ae_recon_mode: str = "D_p"        # "D_p" | "D_p_hat"
#     dg_recon_mode: str = "D_p_hat"    # "D_p" | "D_p_hat"

#     # Representation distance
#     eps: float = 1e-8

#     # Reconstruction distance
#     patch_size: int = 16
#     in_chans: int = 3
#     use_norm_pix_loss: bool = False

#     # Mask / degradation distance
#     mask_threshold: float = 0.25
#     recon_target: float = 0.1   # D^_p target


# @dataclass
# class RegluationConfig:
#     # outer weights for degradation / representation regulation
#     lambda_deg: float = 0.0
#     lambda_rep: float = 0.0

#     # inner weights for degradation regulation
#     lambda_tv: float = 0.0
#     lambda_sparsity: float = 0.0

#     # representation regulation parameters
#     rep_threshold: float = 0.3
#     rep_exp_k: float = 10.0

#     # numerical stability
#     eps: float = 1e-8


# @dataclass
# class LossConfig:
#     distance: DistanceConfig = field(default_factory=DistanceConfig)
#     regluation: RegluationConfig = field(default_factory=RegluationConfig)

#     reduction: str = "mean"   # "mean" | "sum" | "none"
#     return_details: bool = True
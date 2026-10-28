"""Configuration surface for DGNet's adversarial reconstruction losses.

This module deliberately contains no model or dataset imports.  The training
engine can build one :class:`LossConfig` and pass it to
``loss.prepare_dgloss`` without coupling the loss package to DGNet internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi
from typing import Literal


ScheduleKind = Literal["linear", "cosine"]


@dataclass(frozen=True)
class ScheduleConfig:
    """Optional scalar schedule applied to exactly one loss coefficient.

    A schedule multiplies its term's configured base coefficient.  Thus a
    coefficient set to ``0.0`` remains exactly zero even when scheduling is
    enabled, which is important when ablation experiments turn terms off.
    """

    # ``False`` leaves the coefficient unchanged.  Switching a schedule on
    # therefore needs no change to loss code, only a config update.
    enabled: bool = False

    # Interpolation curve for the multiplicative factor.  ``linear`` changes
    # at a constant rate; ``cosine`` starts and finishes smoothly.
    kind: ScheduleKind = "linear"

    # First global optimization step affected by interpolation.  Before this
    # step the scheduler emits ``start_factor``.
    start_step: int = 0

    # Step at which interpolation is complete.  At and after this step the
    # scheduler emits ``end_factor``.
    end_step: int = 100

    # Multiplier at ``start_step``.  Use ``0.0`` for a term warm-up, or
    # ``1.0`` when annealing a term down from its configured base coefficient.
    start_factor: float = 0.0

    # Multiplier at ``end_step``.  Use ``1.0`` for warm-up and ``0.0`` for
    # scheduled removal of a term.
    end_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ("linear", "cosine"):
            raise ValueError("ScheduleConfig.kind must be 'linear' or 'cosine'.")
        if self.start_step < 0:
            raise ValueError("ScheduleConfig.start_step must be non-negative.")
        if self.end_step <= self.start_step:
            raise ValueError("ScheduleConfig.end_step must be greater than start_step.")
        if self.start_factor < 0.0 or self.end_factor < 0.0:
            raise ValueError("Schedule factors must be non-negative.")

    def factor_at(self, step: int) -> float:
        """Return this schedule's coefficient multiplier at ``step``."""

        if step < 0:
            raise ValueError("Loss step must be non-negative.")
        if not self.enabled:
            return 1.0
        progress = min(max((step - self.start_step) / (self.end_step - self.start_step), 0.0), 1.0)
        if self.kind == "cosine":
            progress = 0.5 * (1.0 - cos(pi * progress))
        return self.start_factor + (self.end_factor - self.start_factor) * progress


@dataclass(frozen=True)
class LossConfig:
    """All hyperparameters controlling the DGNet loss construction.

    Four separately switchable terms are used:

    ``alpha_inference * D_rec``
        Reconstruction objective optimized for the inference encoder/decoder.
    ``alpha_target * (D_rec - tau_deg)^2``
        Difficulty-matching objective optimized for the degradation network.
    ``lambda_budget * |mean(|M(x)|) - beta_mask|``
        Residual magnitude budget objective for the degradation network.
    ``-lambda_reg * KL(P_damage || U)``
        Anti-collapse objective encouraging localized rather than uniform
        brightness damage.

    Set any term coefficient to exactly ``0.0`` to disable that term.  The
    loss implementation then does not include that term in the autograd
    objective, so that disabled term contributes no gradient.
    """

    # Coefficient on inference reconstruction loss D_rec.  It belongs to the
    # inference/decoder optimizer objective, not the degradation objective.
    alpha_inference: float = 1.0

    # Coefficient on degradation target loss (D_rec - tau_deg)^2.  Set to
    # zero when testing the budget or regularization terms in isolation.
    alpha_target: float = 1.0

    # Coefficient on the residual magnitude budget penalty.  Larger values
    # make M(x)'s mean absolute residual adhere more tightly to beta_mask.
    lambda_budget: float = 1.0

    # Coefficient on KL anti-collapse regularization.  The implementation
    # inserts this as ``-lambda_reg * KL`` because the degradation optimizer
    # minimizes its loss while it should maximize non-uniformity.
    lambda_reg: float = 0.1

    # Desired reconstruction difficulty for the degradation network.  Since
    # the default reconstruction reduction is mean squared error, this is a
    # target mean per-pixel reconstruction error by default.
    tau_deg: float = 0.25

    # Desired average absolute magnitude of the image-shaped degradation
    # residual M(x).  It bounds how much damage the degradation model uses.
    beta_mask: float = 0.15

    # Numerical stabilizer in P_damage = A / (sum(A) + eps) and its log.
    # It keeps zero-initialized residuals finite during the first forward.
    eps: float = 1.0e-8

    # Scalar reduction for D_rec and per-image KL: ``"mean"`` is stable when
    # batch/image size changes; ``"sum"`` reproduces an unnormalized squared
    # L2 sum and requires correspondingly scaled tau_deg.
    reduction: Literal["mean", "sum"] = "mean"

    # Optional schedule for alpha_inference.  By default the reconstruction
    # learner begins at full strength without scheduling.
    alpha_inference_schedule: ScheduleConfig = ScheduleConfig()

    # Optional schedule for alpha_target.  For adversarial warm-up, set
    # enabled=True with factors 0 -> 1 over the desired number of steps.
    alpha_target_schedule: ScheduleConfig = ScheduleConfig()

    # Optional schedule for lambda_budget, useful when first allowing M(x) to
    # explore then gradually enforcing its magnitude budget.
    lambda_budget_schedule: ScheduleConfig = ScheduleConfig()

    # Optional schedule for lambda_reg, useful for introducing anti-collapse
    # pressure after a target-difficulty warm-up period.
    lambda_reg_schedule: ScheduleConfig = ScheduleConfig()

    def __post_init__(self) -> None:
        coefficient_names = ("alpha_inference", "alpha_target", "lambda_budget", "lambda_reg")
        for name in coefficient_names:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.tau_deg < 0.0 or self.beta_mask < 0.0:
            raise ValueError("tau_deg and beta_mask must be non-negative.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")

    def weights_at(self, step: int = 0) -> dict[str, float]:
        """Return effective term weights after applying all selected schedules."""

        return {
            "alpha_inference": self.alpha_inference * self.alpha_inference_schedule.factor_at(step),
            "alpha_target": self.alpha_target * self.alpha_target_schedule.factor_at(step),
            "lambda_budget": self.lambda_budget * self.lambda_budget_schedule.factor_at(step),
            "lambda_reg": self.lambda_reg * self.lambda_reg_schedule.factor_at(step),
        }


__all__ = ["LossConfig", "ScheduleConfig", "ScheduleKind"]

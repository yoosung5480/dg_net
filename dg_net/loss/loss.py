"""Config-driven loss functions for DGNet adversarial reconstruction.

Only tensors are consumed by this module.  In particular it does not import
``DGNet`` or a dataset, keeping the loss package usable from any training
engine that supplies an original image, its reconstruction, and a degradation
residual.

Typical use with the project's DGNet output object::

    loss_fn = prepare_dgloss(LossConfig())
    values = loss_fn(model(images), step=global_step)
    values.inference_loss.backward()    # update encoder/decoder while M is frozen
    values.degradation_loss.backward()  # update M while inference model is frozen
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from loss_config import LossConfig, ScheduleConfig
else:
    try:  # package import
        from .loss_config import LossConfig, ScheduleConfig
    except ImportError:  # direct ``python loss.py`` / flat-module import
        from loss_config import LossConfig, ScheduleConfig


def _reduce(value: Tensor, reduction: str) -> Tensor:
    return value.mean() if reduction == "mean" else value.sum()


class ReconstructionDistance(nn.Module):
    """Compute ``D_rec = ||reconstruction - original||_2^2``."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, reconstruction: Tensor, original: Tensor) -> Tensor:
        if reconstruction.shape != original.shape:
            raise ValueError(
                "reconstruction and original must have identical shapes, "
                f"got {tuple(reconstruction.shape)} and {tuple(original.shape)}."
            )
        return _reduce((reconstruction - original).square(), self.reduction)


class DegradationTargetDistance(nn.Module):
    """Compute ``D_target = (D_rec - tau_deg)^2``."""

    def __init__(self, tau_deg: float) -> None:
        super().__init__()
        self.tau_deg = tau_deg

    def forward(self, reconstruction_distance: Tensor) -> Tensor:
        return (reconstruction_distance - self.tau_deg).square()


class MaskBudgetDistance(nn.Module):
    """Compute ``D_budget = |mean(|M(x)|) - beta_mask|``."""

    def __init__(self, beta_mask: float) -> None:
        super().__init__()
        self.beta_mask = beta_mask

    def forward(self, residual: Tensor) -> Tensor:
        return (residual.abs().mean() - self.beta_mask).abs()


class DamageKLDivergence(nn.Module):
    """Compute ``KL(P_damage || Uniform)`` over image spatial positions.

    For each image, channel-wise absolute residuals produce its spatial
    damage map.  A zero residual produces a finite zero KL value, matching the
    zero-initialized degradation head used by DGNet.
    """

    def __init__(self, eps: float = 1.0e-8, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, residual: Tensor) -> Tensor:
        if residual.ndim != 4:
            raise ValueError(f"residual must be BCHW image data, got shape {tuple(residual.shape)}.")
        damage = residual.abs().mean(dim=1).flatten(start_dim=1)
        probabilities = damage / (damage.sum(dim=1, keepdim=True) + self.eps)
        uniform_probability = 1.0 / probabilities.shape[1]
        contribution = torch.where(
            probabilities > 0.0,
            probabilities * (probabilities.clamp_min(self.eps).log() - torch.log(
                probabilities.new_tensor(uniform_probability)
            )),
            torch.zeros_like(probabilities),
        )
        return _reduce(contribution.sum(dim=1), self.reduction)


@dataclass
class DGLossResult:
    """Separate optimization objectives and detached diagnostic terms.

    ``inference_loss`` and ``degradation_loss`` must be optimized in separate
    DGNet phases.  ``combined_loss`` is supplied only for reporting/smoke
    tests; using it for one optimizer removes the intended adversarial split.
    """

    inference_loss: Tensor
    degradation_loss: Tensor
    combined_loss: Tensor
    reconstruction_distance: Tensor
    target_distance: Tensor
    budget_distance: Tensor
    damage_kl: Tensor
    weighted_inference: Tensor
    weighted_target: Tensor
    weighted_budget: Tensor
    weighted_regularization: Tensor
    effective_weights: dict[str, float]

    def metrics(self) -> dict[str, float]:
        """Convert detached scalar diagnostics to plain numbers for logs."""

        values = {
            "inference_loss": self.inference_loss,
            "degradation_loss": self.degradation_loss,
            "combined_loss": self.combined_loss,
            "reconstruction_distance": self.reconstruction_distance,
            "target_distance": self.target_distance,
            "budget_distance": self.budget_distance,
            "damage_kl": self.damage_kl,
            "weighted_inference": self.weighted_inference,
            "weighted_target": self.weighted_target,
            "weighted_budget": self.weighted_budget,
            "weighted_regularization": self.weighted_regularization,
        }
        result = {name: value.detach().item() for name, value in values.items()}
        result.update(self.effective_weights)
        return result


class DGLoss(nn.Module):
    """Four independently controlled DGNet objective terms."""

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.reconstruction = ReconstructionDistance(cfg.reduction)
        self.target = DegradationTargetDistance(cfg.tau_deg)
        self.budget = MaskBudgetDistance(cfg.beta_mask)
        self.damage_kl = DamageKLDivergence(cfg.eps, cfg.reduction)

    @staticmethod
    def _tensor_from_output(output: object | Mapping[str, Tensor], name: str) -> Tensor:
        tensor = output[name] if isinstance(output, Mapping) else getattr(output, name, None)
        if not isinstance(tensor, Tensor):
            raise TypeError(f"loss input must provide a Tensor named {name!r}.")
        return tensor

    @staticmethod
    def _disabled(reference: Tensor) -> Tensor:
        """Create a detached scalar for an inactive term (no term autograd path)."""

        return reference.detach().new_zeros(())

    def from_tensors(
        self,
        original: Tensor,
        reconstruction: Tensor,
        residual: Tensor,
        step: int = 0,
    ) -> DGLossResult:
        """Compute objectives from explicit tensors instead of a model output object."""

        if residual.shape != original.shape:
            raise ValueError(
                "residual and original must have identical shapes, "
                f"got {tuple(residual.shape)} and {tuple(original.shape)}."
            )
        weights = self.cfg.weights_at(step)
        inference_on = weights["alpha_inference"] != 0.0
        target_on = weights["alpha_target"] != 0.0
        budget_on = weights["lambda_budget"] != 0.0
        regularization_on = weights["lambda_reg"] != 0.0

        if inference_on or target_on:
            reconstruction_distance = self.reconstruction(reconstruction, original)
        else:
            with torch.no_grad():
                reconstruction_distance = self.reconstruction(reconstruction, original)
        target_distance = (
            self.target(reconstruction_distance)
            if target_on
            else self.target(reconstruction_distance.detach())
        )
        budget_distance = (
            self.budget(residual) if budget_on else self.budget(residual.detach())
        )
        damage_kl = (
            self.damage_kl(residual)
            if regularization_on
            else self.damage_kl(residual.detach())
        )

        weighted_inference = (
            reconstruction_distance * weights["alpha_inference"]
            if inference_on
            else self._disabled(reconstruction)
        )
        weighted_target = (
            target_distance * weights["alpha_target"]
            if target_on
            else self._disabled(reconstruction)
        )
        weighted_budget = (
            budget_distance * weights["lambda_budget"]
            if budget_on
            else self._disabled(residual)
        )
        weighted_regularization = (
            -damage_kl * weights["lambda_reg"]
            if regularization_on
            else self._disabled(residual)
        )

        # Keep ``backward`` safe for a fully disabled objective while emitting
        # exactly zero gradients.  Active terms do not use this anchor.
        inference_loss = (
            weighted_inference if inference_on else reconstruction.sum() * 0.0
        )
        if target_on or budget_on or regularization_on:
            degradation_loss = weighted_target + weighted_budget + weighted_regularization
        else:
            degradation_loss = reconstruction.sum() * 0.0 + residual.sum() * 0.0
        combined_loss = inference_loss + degradation_loss

        return DGLossResult(
            inference_loss=inference_loss,
            degradation_loss=degradation_loss,
            combined_loss=combined_loss,
            reconstruction_distance=reconstruction_distance.detach(),
            target_distance=target_distance.detach(),
            budget_distance=budget_distance.detach(),
            damage_kl=damage_kl.detach(),
            weighted_inference=weighted_inference.detach(),
            weighted_target=weighted_target.detach(),
            weighted_budget=weighted_budget.detach(),
            weighted_regularization=weighted_regularization.detach(),
            effective_weights=weights,
        )

    def forward(
        self,
        output: object | Mapping[str, Tensor],
        step: int = 0,
    ) -> DGLossResult:
        """Consume ``DGNetOutput`` or a mapping with its three tensor fields."""

        return self.from_tensors(
            original=self._tensor_from_output(output, "original"),
            reconstruction=self._tensor_from_output(output, "reconstruction"),
            residual=self._tensor_from_output(output, "residual"),
            step=step,
        )


def prepare_dgloss(cfg: LossConfig | None = None) -> DGLoss:
    """Build the DGNet loss module solely from ``LossConfig``."""

    return DGLoss(cfg or LossConfig())


def write_loss_report(
    values: DGLossResult,
    output_path: str | Path,
    *,
    heading: str = "DGNet loss verification",
) -> Path:
    """Persist one scalar loss observation without depending on a model API."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = values.metrics()
    lines = [heading] + [f"{name}: {value:.10f}" for name, value in metrics.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _self_test() -> Path:
    """Run independent tensor-level checks and write an executable smoke log."""

    torch.manual_seed(0)
    original = torch.rand(4, 3, 32, 32)
    reconstruction = (original + 0.05 * torch.randn_like(original)).requires_grad_()
    residual = (0.1 * torch.rand_like(original)).requires_grad_()
    scheduled = ScheduleConfig(enabled=True, kind="cosine", end_step=100)
    cfg = LossConfig(alpha_target_schedule=scheduled, lambda_reg_schedule=scheduled)
    loss_fn = prepare_dgloss(cfg)
    start = loss_fn.from_tensors(original, reconstruction, residual, step=0)
    end = loss_fn.from_tensors(original, reconstruction, residual, step=100)
    end.degradation_loss.backward()
    assert torch.isfinite(end.combined_loss)
    assert start.effective_weights["alpha_target"] == 0.0
    assert end.effective_weights["alpha_target"] == cfg.alpha_target

    off_reconstruction = reconstruction.detach().clone().requires_grad_()
    off_residual = residual.detach().clone().requires_grad_()
    off = prepare_dgloss(
        LossConfig(alpha_inference=0.0, alpha_target=0.0, lambda_budget=0.0, lambda_reg=0.0)
    ).from_tensors(original, off_reconstruction, off_residual)
    off.combined_loss.backward()
    assert off_reconstruction.grad is not None
    assert off_residual.grad is not None
    assert torch.count_nonzero(off_reconstruction.grad) == 0
    assert torch.count_nonzero(off_residual.grad) == 0

    report = Path(__file__).resolve().parent / "output" / "loss_tensor_smoke.log"
    write_loss_report(end, report, heading="DGNet independent loss smoke verification")
    with report.open("a", encoding="utf-8") as stream:
        stream.write("schedule_step_0_alpha_target: 0.0000000000\n")
        stream.write(f"schedule_step_100_alpha_target: {cfg.alpha_target:.10f}\n")
        stream.write("all_terms_disabled_zero_gradient: True\n")
    return report


__all__ = [
    "DGLoss",
    "DGLossResult",
    "DamageKLDivergence",
    "DegradationTargetDistance",
    "LossConfig",
    "MaskBudgetDistance",
    "ReconstructionDistance",
    "ScheduleConfig",
    "prepare_dgloss",
    "write_loss_report",
]


if __name__ == "__main__":
    print(f"Wrote loss verification artifact: {_self_test()}")

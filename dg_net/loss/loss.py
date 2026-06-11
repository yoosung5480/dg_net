"""Tensor-only, term-wise scheduled DGNet adversarial losses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from .loss_config import LossConfig, TermWeightConfig
except ImportError:
    from loss_config import LossConfig, TermWeightConfig


def _reduce(values: Tensor, reduction: str) -> Tensor:
    return values.mean() if reduction == "mean" else values.sum()


class InferenceReconstructionLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, reconstruction: Tensor, original: Tensor) -> Tensor:
        if reconstruction.shape != original.shape:
            raise ValueError("reconstruction and original shapes must match.")
        return _reduce((reconstruction - original).square(), self.reduction)


class RepresentationDistance(nn.Module):
    def __init__(self, eps: float = 1.0e-8, reduction: str = "mean") -> None:
        super().__init__()
        self.eps, self.reduction = eps, reduction

    def forward(self, original: Tensor, degraded: Tensor) -> Tensor:
        if original.shape != degraded.shape or original.ndim != 2:
            raise ValueError("representations must be matching [B, D] tensors.")
        cosine = F.cosine_similarity(original, degraded, dim=1, eps=self.eps)
        return _reduce(1.0 - cosine, self.reduction)


class DGTargetRepresentationDistance(nn.Module):
    """Keep each degraded representation at a target cosine distance."""

    def __init__(self, tau_rep: float, eps: float = 1.0e-8, reduction: str = "mean") -> None:
        super().__init__()
        self.tau_rep, self.eps, self.reduction = tau_rep, eps, reduction

    def forward(self, original: Tensor, degraded: Tensor) -> Tensor:
        if original.shape != degraded.shape or original.ndim != 2:
            raise ValueError("representations must be matching [B, D] tensors.")
        cosine = F.cosine_similarity(original, degraded, dim=1, eps=self.eps)
        per_image_distance = 1.0 - cosine
        return _reduce((per_image_distance - self.tau_rep).abs(), self.reduction)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, original: Tensor, degraded: Tensor) -> Tensor:
        if original.shape != degraded.shape or original.ndim != 2:
            raise ValueError("representations must be matching [B, D] tensors.")
        if original.shape[0] < 2:
            raise ValueError("InfoNCE requires local batch size of at least 2.")
        normalized_original = F.normalize(original, dim=1)
        normalized_degraded = F.normalize(degraded, dim=1)
        logits = normalized_original @ normalized_degraded.T / self.temperature
        labels = torch.arange(original.shape[0], device=original.device)
        loss_forward = F.cross_entropy(logits, labels)
        loss_reverse = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_forward + loss_reverse)


class DGTargetDistance(nn.Module):
    def __init__(self, tau_deg: float, reduction: str = "mean") -> None:
        super().__init__()
        self.tau_deg, self.reduction = tau_deg, reduction

    def forward(self, reconstruction: Tensor, original: Tensor) -> Tensor:
        per_image = (reconstruction - original).square().flatten(1).mean(1)
        return _reduce((per_image - self.tau_deg).abs(), self.reduction)


class DGBudgetDistance(nn.Module):
    def __init__(self, tau_budget: float, reduction: str = "mean") -> None:
        super().__init__()
        self.tau_budget, self.reduction = tau_budget, reduction

    def forward(self, residual: Tensor) -> Tensor:
        per_image = residual.square().flatten(1).mean(1)
        return _reduce((per_image - self.tau_budget).abs(), self.reduction)


class ChannelWiseSpatialKL(nn.Module):
    def __init__(self, eps: float = 1.0e-8, reduction: str = "mean") -> None:
        super().__init__()
        self.eps, self.reduction = eps, reduction

    def forward(self, residual: Tensor) -> Tensor:
        if residual.ndim != 4:
            raise ValueError("residual must be BCHW.")
        damage = residual.abs().flatten(2)
        probability = damage / (damage.sum(2, keepdim=True) + self.eps)
        uniform = 1.0 / damage.shape[2]
        terms = torch.where(
            probability > 0,
            probability * (probability.clamp_min(self.eps).log() - probability.new_tensor(uniform).log()),
            torch.zeros_like(probability),
        )
        return _reduce(terms.sum(2), self.reduction)


class SparsePenalty(nn.Module):
    def __init__(self, eps: float = 1.0e-8, reduction: str = "mean") -> None:
        super().__init__()
        self.eps, self.reduction = eps, reduction

    def forward(self, mask: Tensor) -> Tensor:
        if mask.ndim == 4:
            mask = mask.mean(1)
        if mask.ndim != 3:
            raise ValueError("mask must be [B,H,W] or [B,C,H,W].")
        fraction = mask.flatten(1).sum(1) / (mask.shape[-2] * mask.shape[-1])
        return _reduce(torch.sin(torch.pi * fraction).clamp_min(self.eps).reciprocal(), self.reduction)


@dataclass
class DGLossOutput:
    inference_loss: Tensor
    degradation_loss: Tensor
    combined_loss: Tensor
    inference_rec: Tensor
    inference_rep: Tensor
    dg_target: Tensor
    dg_budget: Tensor
    dg_kl: Tensor
    dg_rep: Tensor
    dg_sparse: Tensor
    weighted_inference_rec: Tensor
    weighted_inference_rep: Tensor
    weighted_dg_target: Tensor
    weighted_dg_budget: Tensor
    weighted_dg_kl: Tensor
    weighted_dg_rep: Tensor
    weighted_dg_sparse: Tensor
    effective_weights: dict[str, float]

    def metrics(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                result[name] = float(value.detach().item())
        result.update({f"weight_{k}": v for k, v in self.effective_weights.items()})
        return result


class DGLoss(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.rec = InferenceReconstructionLoss(config.inference_reduction)
        self.rep = self._build_representation_objective(config)
        self.dg_rep_target = DGTargetRepresentationDistance(config.tau_rep, config.eps, config.dg_reduction)
        self.target = DGTargetDistance(config.tau_deg, config.dg_reduction)
        self.budget = DGBudgetDistance(config.tau_budget, config.dg_reduction)
        self.kl = ChannelWiseSpatialKL(config.eps, config.dg_reduction)
        self.sparse = SparsePenalty(config.eps, config.dg_reduction)

    @staticmethod
    def _build_representation_objective(config: LossConfig) -> nn.Module:
        if config.representation_distance == "cosine":
            return RepresentationDistance(config.eps, config.inference_reduction)
        if config.representation_distance == "infonce":
            return InfoNCELoss(config.infonce_temperature)
        raise ValueError(f"Unsupported representation distance: {config.representation_distance!r}.")

    @staticmethod
    def _zero(reference: Tensor) -> Tensor:
        return reference.detach().new_zeros(())

    @staticmethod
    def _tensor(output: object | Mapping[str, Tensor], name: str) -> Tensor:
        value = output[name] if isinstance(output, Mapping) else getattr(output, name, None)
        if not isinstance(value, Tensor):
            raise TypeError(f"loss input must provide Tensor {name!r}.")
        return value

    def forward(
        self,
        output: object | Mapping[str, Tensor] | None = None,
        *,
        original: Tensor | None = None,
        degraded: Tensor | None = None,
        residual: Tensor | None = None,
        reconstruction: Tensor | None = None,
        z_original: Tensor | None = None,
        z_degraded: Tensor | None = None,
        mask: Tensor | None = None,
        step: int = 0,
    ) -> DGLossOutput:
        if output is not None:
            original = self._tensor(output, "original")
            degraded = self._tensor(output, "degraded")
            residual = self._tensor(output, "residual")
            reconstruction = self._tensor(output, "reconstruction")
            z_original = self._tensor(output, "representation")
            z_degraded = self._tensor(output, "degraded_representation")
        required = (original, degraded, residual, reconstruction, z_original, z_degraded)
        if any(value is None for value in required):
            raise TypeError("original, degraded, residual, reconstruction, z_original and z_degraded are required.")
        assert original is not None and residual is not None and reconstruction is not None
        assert z_original is not None and z_degraded is not None
        weights = self.config.weights_at(step)

        inference_rec = self.rec(reconstruction, original) if weights["inference_rec"] else self._zero(original)
        inference_rep = self.rep(z_original, z_degraded) if weights["inference_rep"] else self._zero(original)
        dg_target = self.target(reconstruction, original) if weights["dg_target"] else self._zero(original)
        dg_budget = self.budget(residual) if weights["dg_budget"] else self._zero(original)
        dg_kl = self.kl(residual) if weights["dg_kl"] else self._zero(original)
        dg_rep = self.dg_rep_target(z_original.detach(), z_degraded) if weights["dg_rep"] else self._zero(original)
        if weights["dg_sparse"] and mask is None:
            raise ValueError("dg_sparse is enabled but no explicit mask was provided.")
        dg_sparse = self.sparse(mask) if weights["dg_sparse"] else self._zero(original)

        weighted_inference_rec = weights["inference_rec"] * inference_rec
        weighted_inference_rep = weights["inference_rep"] * inference_rep
        weighted_dg_target = weights["dg_target"] * dg_target
        weighted_dg_budget = weights["dg_budget"] * dg_budget
        weighted_dg_kl = -weights["dg_kl"] * dg_kl
        weighted_dg_rep = weights["dg_rep"] * dg_rep
        weighted_dg_sparse = weights["dg_sparse"] * dg_sparse
        inference_loss = weighted_inference_rec + weighted_inference_rep
        degradation_loss = weighted_dg_target + weighted_dg_budget + weighted_dg_kl + weighted_dg_rep + weighted_dg_sparse
        # Safe backward anchors for fully disabled ablations.
        if not inference_loss.requires_grad:
            inference_loss = inference_loss + reconstruction.sum() * 0.0 + z_degraded.sum() * 0.0
        if not degradation_loss.requires_grad:
            degradation_loss = degradation_loss + residual.sum() * 0.0
        return DGLossOutput(
            inference_loss, degradation_loss, inference_loss + degradation_loss,
            inference_rec, inference_rep, dg_target, dg_budget, dg_kl, dg_rep, dg_sparse,
            weighted_inference_rec, weighted_inference_rep, weighted_dg_target, weighted_dg_budget,
            weighted_dg_kl, weighted_dg_rep, weighted_dg_sparse, weights,
        )

    def from_output(self, output: object | Mapping[str, Tensor], step: int = 0) -> DGLossOutput:
        return self(output, step=step)

    def from_tensors(self, *, step: int = 0, **tensors: Tensor | None) -> DGLossOutput:
        return self(None, step=step, **tensors)


def prepare_dgloss(config: LossConfig | None = None) -> DGLoss:
    return DGLoss(config or LossConfig())



def write_loss_report(values: DGLossOutput, output_path: str | Path, *, heading: str = "DGNet loss verification") -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(heading + "\n" + "\n".join(f"{k}: {v:.10f}" for k, v in values.metrics().items()) + "\n", encoding="utf-8")
    return path



def _fake_inputs() -> dict[str, Tensor]:
    original = torch.rand(4, 3, 16, 16)
    residual = (torch.rand_like(original) * 0.1).requires_grad_()
    return {
        "original": original, "degraded": original - residual, "residual": residual,
        "reconstruction": (original + torch.randn_like(original) * 0.1).requires_grad_(),
        "z_original": torch.randn(4, 8), "z_degraded": torch.randn(4, 8, requires_grad=True),
    }



def _self_test() -> Path:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs: dict[str, str] = {}
    scheduled = TermWeightConfig(weight=2.0, schedule="cosine", start=0.0, end=1.0, total_steps=100)
    assert scheduled.effective_weight(0) == 0.0 and scheduled.effective_weight(100) == 2.0
    logs["loss_scheduler_test.log"] = "linear/cosine clamp and zero-base checks passed\n"
    assert TermWeightConfig(weight=0.0, schedule="linear", start=1.0, end=9.0).effective_weight(10) == 0.0
    for name, cfg in (
        ("representation_only", LossConfig.representation_only()),
        ("reconstruction_only", LossConfig.reconstruction_only()),
        ("hybrid", LossConfig.hybrid()),
        ("representation_only_infonce", LossConfig.representation_only(representation_distance="infonce")),
    ):
        values = prepare_dgloss(cfg)(**_fake_inputs(), step=50)
        assert values.combined_loss.ndim == 0 and torch.isfinite(values.combined_loss)
        values.combined_loss.backward()
        logs.setdefault("loss_ablation_mode_test.log", "")
        logs["loss_ablation_mode_test.log"] += f"{name}: passed\n"
    off = LossConfig(
        inference_rec=TermWeightConfig(weight=0), inference_rep=TermWeightConfig(weight=0),
        dg_target=TermWeightConfig(weight=0), dg_budget=TermWeightConfig(weight=0),
        dg_kl=TermWeightConfig(weight=0), dg_rep=TermWeightConfig(weight=0),
        dg_sparse=TermWeightConfig(weight=0),
    )
    values = prepare_dgloss(off)(**_fake_inputs())
    assert all(v == 0.0 for v in values.effective_weights.values())
    logs["loss_term_onoff_test.log"] = "all disabled terms skipped and zero weighted\n"
    logs["loss_forward_shape_test.log"] = "all modes produced finite scalar outputs\n"
    logs["loss_config_summary.log"] = json.dumps(asdict(LossConfig.hybrid()), indent=2) + "\n"
    for filename, content in logs.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    return output_dir


__all__ = [
    "ChannelWiseSpatialKL", "DGBudgetDistance", "DGLoss", "DGLossOutput",
    "DGTargetDistance", "DGTargetRepresentationDistance", "InferenceReconstructionLoss", "InfoNCELoss", "LossConfig",
    "RepresentationDistance", "SparsePenalty", "TermWeightConfig",
    "prepare_dgloss", "write_loss_report",
]


if __name__ == "__main__":
    print(f"Wrote loss smoke artifacts: {_self_test()}")

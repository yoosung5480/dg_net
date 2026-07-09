"""Independent configuration and scheduling for every DGNet loss term."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi
from typing import Literal


ScheduleKind = Literal["none", "linear", "cosine"]
RepresentationDistanceKind = Literal["cosine", "infonce"]


@dataclass(frozen=True)
class TermWeightConfig:
    """Base coefficient and multiplicative schedule for one loss term."""

    weight: float = 1.0
    schedule: ScheduleKind = "none"
    start: float = 1.0
    end: float = 1.0
    total_steps: int = 10000

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("TermWeightConfig.weight must be non-negative.")
        if self.schedule not in ("none", "linear", "cosine"):
            raise ValueError("schedule must be one of: none, linear, cosine.")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive.")

    def effective_weight(self, step: int = 0) -> float:
        if step < 0:
            raise ValueError("step must be non-negative.")
        if self.weight == 0.0:
            return 0.0
        if self.schedule == "none":
            return self.weight
        progress = min(max(step / self.total_steps, 0.0), 1.0)
        if self.schedule == "cosine":
            factor = self.end + 0.5 * (self.start - self.end) * (1.0 + cos(pi * progress))
        else:
            factor = self.start + (self.end - self.start) * progress
        return self.weight * factor



def _term(weight: float = 0.0) -> TermWeightConfig:
    return TermWeightConfig(weight=weight)


@dataclass(frozen=True)
class LossConfig:
    """Tensor-only DGNet adversarial objective configuration."""

    inference_rec: TermWeightConfig = field(default_factory=lambda: _term(1.0))
    inference_rep: TermWeightConfig = field(default_factory=lambda: _term(0.0))
    dg_target: TermWeightConfig = field(default_factory=lambda: _term(1.0))
    dg_budget: TermWeightConfig = field(default_factory=lambda: _term(1.0))
    dg_kl: TermWeightConfig = field(default_factory=lambda: _term(0.0))
    dg_rep: TermWeightConfig = field(default_factory=lambda: _term(0.0))
    dg_sparse: TermWeightConfig = field(default_factory=lambda: _term(0.0))
    tau_deg: float = 0.05
    tau_budget: float = 0.01
    tau_rep: float = 0.5
    representation_distance: RepresentationDistanceKind = "cosine"
    infonce_temperature: float = 0.2
    eps: float = 1.0e-8
    inference_reduction: Literal["mean", "sum"] = "mean"
    dg_reduction: Literal["mean", "sum"] = "mean"

    def __post_init__(self) -> None:
        if self.tau_deg < 0.0 or self.tau_budget < 0.0:
            raise ValueError("tau_deg and tau_budget must be non-negative.")
        if not 0.0 < self.tau_rep <= 2.0:
            raise ValueError("tau_rep must be in the cosine-distance range (0, 2].")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        if self.representation_distance not in ("cosine", "infonce"):
            raise ValueError("representation_distance must be 'cosine' or 'infonce'.")
        if self.infonce_temperature <= 0.0:
            raise ValueError("infonce_temperature must be positive.")
        if self.inference_reduction not in ("mean", "sum") or self.dg_reduction not in ("mean", "sum"):
            raise ValueError("reductions must be 'mean' or 'sum'.")

    @classmethod
    def representation_only(cls, **kwargs: object) -> "LossConfig":
        return cls(
            inference_rec=_term(0.0), inference_rep=_term(1.0),
            dg_target=_term(0.0), dg_budget=_term(0.0), dg_kl=_term(0.0),
            dg_rep=_term(1.0), dg_sparse=_term(0.0), **kwargs,
        )

    @classmethod
    def reconstruction_only(cls, **kwargs: object) -> "LossConfig":
        return cls(
            inference_rec=_term(1.0), inference_rep=_term(0.0),
            dg_target=_term(1.0), dg_budget=_term(1.0), dg_kl=_term(0.01),
            dg_rep=_term(0.0), dg_sparse=_term(0.0), **kwargs,
        )

    @classmethod
    def hybrid(cls, **kwargs: object) -> "LossConfig":
        return cls(
            inference_rec=_term(1.0), inference_rep=_term(1.0),
            dg_target=_term(1.0), dg_budget=_term(1.0), dg_kl=_term(0.01),
            dg_rep=_term(1.0), dg_sparse=_term(0.0), **kwargs,
        )

    def weights_at(self, step: int = 0) -> dict[str, float]:
        return {
            name: getattr(self, name).effective_weight(step)
            for name in (
                "inference_rec", "inference_rep", "dg_target", "dg_budget",
                "dg_kl", "dg_rep", "dg_sparse",
            )
        }



def effective_weight(config: TermWeightConfig, step: int = 0) -> float:
    return config.effective_weight(step)


__all__ = [
    "LossConfig", "RepresentationDistanceKind", "ScheduleKind", "TermWeightConfig", "effective_weight",
]

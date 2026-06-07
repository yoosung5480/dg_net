"""Public tensor-only DGNet loss API with lazy module loading."""

from .loss_config import LossConfig, ScheduleKind, TermWeightConfig, effective_weight

_LOSS_EXPORTS = {
    "ChannelWiseSpatialKL", "DGBudgetDistance", "DGLoss", "DGLossOutput",
    "DGTargetDistance", "InferenceReconstructionLoss", "RepresentationDistance",
    "SparsePenalty", "prepare_dgloss", "write_loss_report",
}


def __getattr__(name: str):
    if name in _LOSS_EXPORTS:
        from . import loss as loss_module

        return getattr(loss_module, name)
    raise AttributeError(name)


__all__ = [
    *_LOSS_EXPORTS, "LossConfig", "ScheduleKind", "TermWeightConfig", "effective_weight",
]

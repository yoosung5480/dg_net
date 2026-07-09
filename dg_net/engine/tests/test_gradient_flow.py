import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from loss.loss import prepare_dgloss
from loss.loss_config import LossConfig
from model.dg_model import DGNet, DgNetConfig


MODEL_CONFIG = DgNetConfig(
    IMG_SIZE=16,
    PATCH_SIZE=8,
    EMBED_DIM=16,
    DEPTH=1,
    NUM_HEADS=4,
    DECODER_EMBED_DIM=16,
    DECODER_NUM_HEADS=4,
    PROJECTION_DIM=8,
    DG_ARCHITECT="HYBRID",
)
BATCH_SIZE = 4
LEARNING_RATE = 1.0e-3


@pytest.fixture
def images() -> torch.Tensor:
    torch.manual_seed(123)
    return torch.rand(BATCH_SIZE, 3, MODEL_CONFIG.IMG_SIZE, MODEL_CONFIG.IMG_SIZE)


@pytest.fixture
def base_model() -> DGNet:
    torch.manual_seed(0)
    return DGNet(copy.deepcopy(MODEL_CONFIG))


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _phase_i(model: DGNet) -> None:
    _set_requires_grad(model.degradation, False)
    _set_requires_grad(model.encoder, True)
    _set_requires_grad(model.decoder, True)
    _set_requires_grad(model.projection, True)


def _phase_m(model: DGNet) -> None:
    _set_requires_grad(model.degradation, True)
    _set_requires_grad(model.encoder, False)
    _set_requires_grad(model.decoder, False)
    _set_requires_grad(model.projection, False)


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def _module_changed(module: torch.nn.Module, before: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(before[name], parameter.detach()) for name, parameter in module.named_parameters())


def _finite_nonzero_grad_stats(module: torch.nn.Module) -> tuple[bool, float]:
    max_abs_grad = 0.0
    found = False
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        assert torch.isfinite(parameter.grad).all()
        current = float(parameter.grad.detach().abs().max().item())
        max_abs_grad = max(max_abs_grad, current)
        found = found or current > 0.0
    return found, max_abs_grad


def _all_grads_absent_or_zero(module: torch.nn.Module) -> bool:
    for parameter in module.parameters():
        if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0:
            return False
    return True


@pytest.mark.parametrize(
    ("phase", "loss_config", "expected_changes", "expected_loss"),
    [
        (
            "I",
            LossConfig.reconstruction_only(),
            {"degradation": False, "encoder": True, "decoder": True, "projection": False},
            "positive",
        ),
        (
            "I",
            LossConfig.representation_only(),
            {"degradation": False, "encoder": True, "decoder": False, "projection": True},
            "near_zero",
        ),
        (
            "I",
            LossConfig.representation_only(representation_distance="infonce"),
            {"degradation": False, "encoder": True, "decoder": False, "projection": True},
            "positive",
        ),
        (
            "M",
            LossConfig.reconstruction_only(),
            {"degradation": True, "encoder": False, "decoder": False, "projection": False},
            "positive",
        ),
        (
            "M",
            LossConfig.representation_only(),
            {"degradation": True, "encoder": False, "decoder": False, "projection": False},
            "positive",
        ),
        (
            "M",
            LossConfig.representation_only(representation_distance="infonce"),
            {"degradation": True, "encoder": False, "decoder": False, "projection": False},
            "positive",
        ),
    ],
    ids=[
        "phase_i_reconstruction",
        "phase_i_cosine_representation",
        "phase_i_infonce_representation",
        "phase_m_reconstruction_target",
        "phase_m_cosine_representation",
        "phase_m_infonce_representation",
    ],
)
def test_phase_update_isolation_matches_expected_objective_behavior(
    base_model: DGNet,
    images: torch.Tensor,
    phase: str,
    loss_config: LossConfig,
    expected_changes: dict[str, bool],
    expected_loss: str,
) -> None:
    model = copy.deepcopy(base_model)
    loss_fn = prepare_dgloss(loss_config)

    if phase == "I":
        _phase_i(model)
        optimizer = torch.optim.AdamW(
            list(model.encoder.parameters())
            + list(model.decoder.parameters())
            + list(model.projection.parameters()),
            lr=LEARNING_RATE,
        )
    else:
        _phase_m(model)
        optimizer = torch.optim.AdamW(model.degradation.parameters(), lr=LEARNING_RATE)

    before = {
        "degradation": _snapshot(model.degradation),
        "encoder": _snapshot(model.encoder),
        "decoder": _snapshot(model.decoder),
        "projection": _snapshot(model.projection),
    }

    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    values = loss_fn(outputs)
    loss = values.inference_loss if phase == "I" else values.degradation_loss

    assert torch.isfinite(loss)
    scalar_loss = float(loss.detach().item())
    if expected_loss == "near_zero":
        assert abs(scalar_loss) < 1.0e-6
    elif expected_loss == "positive":
        assert scalar_loss > 0.0
    else:
        assert scalar_loss < 0.0

    loss.backward()

    degradation_has_grad, degradation_max_abs_grad = _finite_nonzero_grad_stats(model.degradation)
    encoder_has_grad, encoder_max_abs_grad = _finite_nonzero_grad_stats(model.encoder)
    decoder_has_grad, decoder_max_abs_grad = _finite_nonzero_grad_stats(model.decoder)
    projection_has_grad, projection_max_abs_grad = _finite_nonzero_grad_stats(model.projection)

    if phase == "I":
        assert not degradation_has_grad
        assert degradation_max_abs_grad == 0.0
    else:
        assert degradation_has_grad
        assert degradation_max_abs_grad > 0.0
        assert _all_grads_absent_or_zero(model.encoder)
        assert _all_grads_absent_or_zero(model.decoder)
        assert _all_grads_absent_or_zero(model.projection)
        if loss_config.representation_distance == "infonce":
            assert encoder_max_abs_grad == 0.0
            assert decoder_max_abs_grad == 0.0
            assert projection_max_abs_grad == 0.0

    optimizer.step()

    changed = {
        "degradation": _module_changed(model.degradation, before["degradation"]),
        "encoder": _module_changed(model.encoder, before["encoder"]),
        "decoder": _module_changed(model.decoder, before["decoder"]),
        "projection": _module_changed(model.projection, before["projection"]),
    }
    assert changed == expected_changes

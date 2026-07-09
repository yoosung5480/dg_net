import dataclasses
import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

loss_module = importlib.import_module("loss.loss")
from loss import DGTargetRepresentationDistance, InfoNCELoss, LossConfig, TermWeightConfig, prepare_dgloss


EXPECTED_DGLOSSOUTPUT_FIELDS = [
    "inference_loss",
    "degradation_loss",
    "combined_loss",
    "inference_rec",
    "inference_rep",
    "dg_target",
    "dg_budget",
    "dg_kl",
    "dg_rep",
    "dg_sparse",
    "weighted_inference_rec",
    "weighted_inference_rep",
    "weighted_dg_target",
    "weighted_dg_budget",
    "weighted_dg_kl",
    "weighted_dg_rep",
    "weighted_dg_sparse",
    "effective_weights",
]


def make_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    original = torch.rand(4, 3, 8, 8)
    residual = torch.rand_like(original) * 0.1
    return {
        "original": original,
        "degraded": original - residual,
        "residual": residual,
        "reconstruction": original + torch.randn_like(original) * 0.05,
        "z_original": torch.randn(4, 8),
        "z_degraded": torch.randn(4, 8),
    }


def test_cosine_representation_distance_is_zero_for_identical_vectors():
    representations = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, -1.0]])

    distance = loss_module.RepresentationDistance()(representations, representations)

    assert distance.ndim == 0
    assert torch.isclose(distance, torch.zeros(()), atol=1e-7)


def test_dg_target_representation_distance_matches_per_image_threshold_error():
    original = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    degraded = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    target_error = DGTargetRepresentationDistance(tau_rep=0.25)(original, degraded)

    assert torch.isclose(target_error, torch.tensor(0.5))


@pytest.mark.parametrize("angle", [0.2, 1.2])
def test_minimizing_dg_target_representation_moves_distance_toward_tau(angle: float):
    original = torch.tensor([[1.0, 0.0]])
    degraded = torch.tensor([[torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))]], requires_grad=True)
    objective = DGTargetRepresentationDistance(tau_rep=0.5)
    optimizer = torch.optim.SGD([degraded], lr=0.1)

    before = objective(original, degraded).detach()
    objective(original, degraded).backward()
    optimizer.step()
    after = objective(original, degraded).detach()

    assert after < before


def test_infonce_prefers_matching_pairs_over_mismatched_pairs():
    objective = InfoNCELoss(temperature=0.2)
    z_original = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    matched = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    mismatched = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    matched_loss = objective(z_original, matched)
    mismatched_loss = objective(z_original, mismatched)

    assert matched_loss < mismatched_loss


def test_infonce_matches_exact_symmetric_local_formula():
    z_original = torch.tensor([[3.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float32)
    z_degraded = torch.tensor([[2.0, 0.0], [0.0, 4.0], [1.0, 1.0]], dtype=torch.float32)
    temperature = 0.3

    loss = InfoNCELoss(temperature=temperature)(z_original, z_degraded)

    normalized_original = F.normalize(z_original, dim=1)
    normalized_degraded = F.normalize(z_degraded, dim=1)
    logits = normalized_original @ normalized_degraded.T / temperature
    labels = torch.arange(z_original.shape[0])
    expected = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    assert torch.isclose(loss, expected)


def test_infonce_requires_local_batch_size_of_at_least_two():
    with pytest.raises(ValueError, match="at least 2"):
        InfoNCELoss()(torch.randn(1, 4), torch.randn(1, 4))


def test_infonce_temperature_must_be_positive():
    with pytest.raises(ValueError, match="infonce_temperature"):
        LossConfig(representation_distance="infonce", infonce_temperature=0.0)


@pytest.mark.parametrize("tau_rep", [0.0, -0.1, 2.1])
def test_tau_rep_must_be_a_reachable_positive_cosine_distance(tau_rep: float):
    with pytest.raises(ValueError, match="tau_rep"):
        LossConfig(tau_rep=tau_rep)


def test_baseline_presets_produce_finite_scalar_losses():
    for config in (
        LossConfig.representation_only(),
        LossConfig.representation_only(representation_distance="infonce"),
        LossConfig.reconstruction_only(),
        LossConfig.hybrid(),
        LossConfig.hybrid(representation_distance="infonce"),
    ):
        values = prepare_dgloss(config)(**make_inputs(), step=17)

        assert values.inference_loss.ndim == 0
        assert values.degradation_loss.ndim == 0
        assert values.combined_loss.ndim == 0
        assert torch.isfinite(values.inference_loss)
        assert torch.isfinite(values.degradation_loss)
        assert torch.isfinite(values.combined_loss)


def test_dg_target_rep_detaches_z_original_but_keeps_z_degraded_gradient():
    config = LossConfig(
        inference_rec=TermWeightConfig(weight=0.0),
        inference_rep=TermWeightConfig(weight=0.0),
        dg_target=TermWeightConfig(weight=0.0),
        dg_budget=TermWeightConfig(weight=0.0),
        dg_kl=TermWeightConfig(weight=0.0),
        dg_rep=TermWeightConfig(weight=1.0),
        dg_sparse=TermWeightConfig(weight=0.0),
        tau_rep=0.5,
    )
    inputs = make_inputs()
    inputs["z_original"] = torch.randn(4, 8, requires_grad=True)
    inputs["z_degraded"] = torch.randn(4, 8, requires_grad=True)

    values = prepare_dgloss(config)(**inputs)
    values.degradation_loss.backward()

    assert inputs["z_original"].grad is None
    assert inputs["z_degraded"].grad is not None
    assert torch.count_nonzero(inputs["z_degraded"].grad).item() > 0


def test_dg_target_rep_is_added_with_positive_weight():
    config = LossConfig(
        inference_rec=TermWeightConfig(weight=0.0),
        inference_rep=TermWeightConfig(weight=0.0),
        dg_target=TermWeightConfig(weight=0.0),
        dg_budget=TermWeightConfig(weight=0.0),
        dg_kl=TermWeightConfig(weight=0.0),
        dg_rep=TermWeightConfig(weight=2.0),
        dg_sparse=TermWeightConfig(weight=0.0),
        tau_rep=0.5,
    )

    values = prepare_dgloss(config)(**make_inputs())

    assert values.weighted_dg_rep > 0.0
    assert torch.isclose(values.degradation_loss, 2.0 * values.dg_rep)


def test_zero_weight_terms_have_zero_effective_weight_and_zero_weighted_terms():
    disabled = LossConfig(
        inference_rec=TermWeightConfig(weight=0.0),
        inference_rep=TermWeightConfig(weight=0.0),
        dg_target=TermWeightConfig(weight=0.0),
        dg_budget=TermWeightConfig(weight=0.0),
        dg_kl=TermWeightConfig(weight=0.0),
        dg_rep=TermWeightConfig(weight=0.0),
        dg_sparse=TermWeightConfig(weight=0.0),
    )

    values = prepare_dgloss(disabled)(**make_inputs(), step=23)

    assert all(weight == 0.0 for weight in values.effective_weights.values())
    for field_name in (
        "weighted_inference_rec",
        "weighted_inference_rep",
        "weighted_dg_target",
        "weighted_dg_budget",
        "weighted_dg_kl",
        "weighted_dg_rep",
        "weighted_dg_sparse",
    ):
        assert getattr(values, field_name).item() == 0.0


def test_linear_and_cosine_schedulers_hit_start_and_end_values():
    linear = TermWeightConfig(weight=2.0, schedule="linear", start=0.25, end=1.5, total_steps=20)
    cosine = TermWeightConfig(weight=3.0, schedule="cosine", start=0.5, end=1.25, total_steps=20)

    assert linear.effective_weight(0) == 0.5
    assert linear.effective_weight(20) == 3.0
    assert cosine.effective_weight(0) == 1.5
    assert cosine.effective_weight(20) == 3.75


def test_dglossoutput_fields_are_stable():
    field_names = [field.name for field in dataclasses.fields(loss_module.DGLossOutput)]

    assert field_names == EXPECTED_DGLOSSOUTPUT_FIELDS


def test_loss_package_exports_infonce_loss():
    package = importlib.import_module("loss")

    assert package.InfoNCELoss is loss_module.InfoNCELoss

import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

dg_model = importlib.import_module("model.dg_model")
DGNet = dg_model.DGNet
DgNetConfig = dg_model.DgNetConfig
compute_cross_patch_influence = dg_model.compute_cross_patch_influence
compute_patch_seam_ratio = dg_model.compute_patch_seam_ratio


def make_config(**overrides) -> DgNetConfig:
    values = dict(
        IMG_SIZE=32,
        PATCH_SIZE=8,
        EMBED_DIM=32,
        DEPTH=2,
        NUM_HEADS=4,
        DECODER_EMBED_DIM=32,
        DECODER_NUM_HEADS=4,
        PROJECTION_DIM=16,
        USE_POS_EMBED=True,
        POS_EMBED_INIT_ZERO=True,
    )
    values.update(overrides)
    return DgNetConfig(**values)


def deterministic_images() -> list[torch.Tensor]:
    lin = torch.linspace(0.0, 1.0, 32)
    grid_x = lin.unsqueeze(0).repeat(32, 1)
    grid_y = lin.unsqueeze(1).repeat(1, 32)

    horizontal = torch.stack([grid_x, grid_x, grid_x], dim=0)
    vertical = torch.stack([grid_y, grid_y, grid_y], dim=0)

    center_square = torch.zeros(3, 32, 32)
    center_square[:, 10:22, 10:22] = 1.0

    two_tone = torch.zeros(3, 32, 32)
    two_tone[:, :, :16] = 0.2
    two_tone[:, :, 16:] = 0.8

    return [image.unsqueeze(0) for image in (horizontal, vertical, center_square, two_tone)]


def seed_residual_head(model: DGNet) -> None:
    torch.manual_seed(123)
    with torch.no_grad():
        model.degradation.residual_head.weight.normal_(mean=0.0, std=0.02)
        model.degradation.residual_head.bias.normal_(mean=0.0, std=0.02)


def collect_metrics(architecture: str, refinement: bool = False) -> tuple[list[float], list[float]]:
    torch.manual_seed(0)
    model = DGNet(
        make_config(
            DG_ARCHITECT=architecture,
            DG_SPATIAL_REFINEMENT=refinement,
            DG_REFINEMENT_CHANNELS=12,
        )
    ).eval()
    seed_residual_head(model)

    cross_patch = []
    seam = []
    with torch.no_grad():
        for image in deterministic_images():
            cross_patch.append(
                compute_cross_patch_influence(model.degradation, image, patch_size=model.cfg.PATCH_SIZE)
            )
            seam.append(compute_patch_seam_ratio(model.degradation(image), patch_size=model.cfg.PATCH_SIZE))
    return cross_patch, seam


def test_vit_and_hybrid_cross_patch_influence_exceeds_globality_floor():
    for architecture in ("VIT", "HYBRID"):
        cross_patch, seam = collect_metrics(architecture)
        assert sum(cross_patch) / len(cross_patch) > 1e-4
        assert all(value >= 0.0 for value in cross_patch)
        assert all(value >= 0.0 for value in seam)


def test_cnn_globality_metrics_are_finite_without_threshold_requirement():
    cross_patch, seam = collect_metrics("CNN")

    assert all(torch.isfinite(torch.tensor(value)) for value in cross_patch)
    assert all(torch.isfinite(torch.tensor(value)) for value in seam)


def test_spatial_refinement_does_not_worsen_seam_ratio_on_deterministic_fixture():
    torch.manual_seed(0)
    off = DGNet(make_config(DG_ARCHITECT="HYBRID", DG_SPATIAL_REFINEMENT=False)).eval()
    on = DGNet(make_config(DG_ARCHITECT="HYBRID", DG_SPATIAL_REFINEMENT=True, DG_REFINEMENT_CHANNELS=12)).eval()
    off_state = off.state_dict()
    on.load_state_dict({**on.state_dict(), **{k: v for k, v in off_state.items() if k in on.state_dict()}}, strict=False)
    seed_residual_head(off)
    seed_residual_head(on)

    off_seam = []
    on_seam = []
    with torch.no_grad():
        for image in deterministic_images():
            off_seam.append(compute_patch_seam_ratio(off.degradation(image), patch_size=off.cfg.PATCH_SIZE))
            on_seam.append(compute_patch_seam_ratio(on.degradation(image), patch_size=on.cfg.PATCH_SIZE))

    assert sum(on_seam) / len(on_seam) <= sum(off_seam) / len(off_seam) + 1e-8

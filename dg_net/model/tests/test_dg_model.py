import importlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

dg_model = importlib.import_module("model.dg_model")
DGNet = dg_model.DGNet
DgNetConfig = dg_model.DgNetConfig
DgNetOutput = dg_model.DgNetOutput
save_sample_visualization = dg_model.save_sample_visualization


def make_config(**overrides) -> DgNetConfig:
    values = dict(
        IMG_SIZE=16,
        PATCH_SIZE=8,
        EMBED_DIM=16,
        DEPTH=1,
        NUM_HEADS=4,
        DECODER_EMBED_DIM=16,
        DECODER_NUM_HEADS=4,
        PROJECTION_DIM=8,
        USE_POS_EMBED=True,
        POS_EMBED_INIT_ZERO=True,
    )
    values.update(overrides)
    return DgNetConfig(**values)


def make_sample(batch_size: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch_size, 3, 16, 16)


@torch.no_grad()
def run_forward(architecture: str, sample: torch.Tensor, **config_overrides) -> DgNetOutput:
    model = DGNet(make_config(DG_ARCHITECT=architecture, **config_overrides)).eval()
    return model(sample)


def test_forward_shapes_match_expected_for_all_degradation_architectures():
    sample = make_sample()

    for architecture in ("VIT", "CNN", "HYBRID"):
        output = run_forward(architecture, sample)

        assert output.original.shape == sample.shape
        assert output.residual.shape == sample.shape
        assert output.degraded.shape == sample.shape
        assert output.reconstruction.shape == sample.shape
        assert output.original_reconstruction.shape == sample.shape
        assert output.representation.shape == (sample.shape[0], 8)
        assert output.degraded_representation.shape == (sample.shape[0], 8)


def test_initial_residual_is_zero_and_degraded_equals_original():
    sample = make_sample()

    for architecture in ("VIT", "CNN", "HYBRID"):
        output = run_forward(architecture, sample)
        assert torch.count_nonzero(output.residual) == 0
        assert torch.equal(output.residual, torch.zeros_like(output.residual))
        assert torch.equal(output.degraded, output.original)
        assert torch.equal(output.degraded, sample)


def test_spatial_refinement_preserves_shapes_and_zero_init_behavior():
    sample = make_sample()
    output = run_forward("HYBRID", sample, DG_SPATIAL_REFINEMENT=True, DG_REFINEMENT_CHANNELS=12)

    assert output.residual.shape == sample.shape
    assert output.degraded.shape == sample.shape
    assert torch.count_nonzero(output.residual) == 0
    assert torch.equal(output.degraded, sample)


def test_dgnet_output_keeps_existing_seven_fields():
    assert DgNetOutput._fields == (
        "original",
        "residual",
        "degraded",
        "reconstruction",
        "original_reconstruction",
        "representation",
        "degraded_representation",
    )
    assert len(DgNetOutput._fields) == 7


@torch.no_grad()
def test_save_load_round_trip_preserves_all_outputs(tmp_path):
    torch.manual_seed(0)
    cfg = make_config(DG_ARCHITECT="HYBRID", DG_SPATIAL_REFINEMENT=True, DG_REFINEMENT_CHANNELS=12)
    model = DGNet(cfg).eval()
    sample = make_sample()

    before = model(sample)
    saved_dir = model.save(tmp_path / "saved_dgnet")
    loaded = DGNet.load(saved_dir, map_location="cpu", verbose=False).eval()
    after = loaded(sample)

    assert saved_dir.is_dir()
    assert (saved_dir / "config.json").is_file()
    assert (saved_dir / "model.pt").is_file()

    for field in DgNetOutput._fields:
        assert torch.allclose(getattr(before, field), getattr(after, field))


def test_load_accepts_legacy_config_without_refinement_fields(tmp_path):
    torch.manual_seed(0)
    model = DGNet(make_config(DG_ARCHITECT="HYBRID")).eval()
    saved_dir = model.save(tmp_path / "legacy_dgnet")
    config_path = saved_dir / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload.pop("DG_SPATIAL_REFINEMENT")
    payload.pop("DG_REFINEMENT_CHANNELS")
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = DGNet.load(saved_dir, map_location="cpu", verbose=False)

    assert loaded.cfg.DG_SPATIAL_REFINEMENT is False
    assert loaded.cfg.DG_REFINEMENT_CHANNELS == 32


def test_config_validation_rejects_non_positive_refinement_channels():
    try:
        DgNetConfig(DG_REFINEMENT_CHANNELS=0).validate()
    except ValueError as error:
        assert "DG_REFINEMENT_CHANNELS" in str(error)
    else:
        raise AssertionError("Expected DG_REFINEMENT_CHANNELS validation failure")


def test_spatial_refinement_last_conv_receives_gradient():
    torch.manual_seed(0)
    model = DGNet(make_config(DG_ARCHITECT="HYBRID", DG_SPATIAL_REFINEMENT=True, DG_REFINEMENT_CHANNELS=12))
    with torch.no_grad():
        nn = model.degradation.spatial_refinement.layers
        nn[-1].weight.normal_(mean=0.0, std=0.02)
        nn[-1].bias.normal_(mean=0.0, std=0.02)
    sample = make_sample(batch_size=1)

    loss = model(sample).residual.square().mean()
    loss.backward()

    final_conv = model.degradation.spatial_refinement.layers[-1]
    assert final_conv.weight.grad is not None
    assert final_conv.bias.grad is not None
    assert torch.count_nonzero(final_conv.weight.grad) > 0
    assert torch.count_nonzero(final_conv.bias.grad) > 0


def test_save_sample_visualization_creates_png_and_metrics_sidecar(tmp_path):
    sample = make_sample()
    output = run_forward("HYBRID", sample)

    path = tmp_path / "sample.png"
    returned = save_sample_visualization(output, path)

    assert returned == path
    assert path.is_file()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0

    rendered = Image.open(path)
    assert rendered.size == (7 * 224, 224 + 28 + 84)

    metrics_path = tmp_path / "sample_metrics.txt"
    assert metrics_path.is_file()
    metrics_text = metrics_path.read_text(encoding="utf-8")
    assert "patch_seam_ratio" in metrics_text
    assert "cosine_similarity" in metrics_text

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.train_config import TrainConfig, build_arg_parser


def test_cli_round_trip_wires_infonce_spatial_refinement_and_imagenet_class_count() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--task",
            "dg_pretrain",
            "--dataset",
            "imagenet",
            "--representation-distance",
            "infonce",
            "--infonce-temperature",
            "0.3",
            "--dg-rep-target-weight",
            "0.7",
            "--tau-rep",
            "0.4",
            "--dg-spatial-refinement",
            "--dg-refinement-channels",
            "24",
        ]
    )

    cfg = TrainConfig.from_args(args)
    model_cfg = cfg.build_model_config()
    loss_cfg = cfg.build_loss_config()

    assert cfg.num_classes == 1000
    assert cfg.representation_distance == "infonce"
    assert cfg.infonce_temperature == 0.3
    assert cfg.dg_spatial_refinement is True
    assert cfg.dg_refinement_channels == 24
    assert model_cfg.DG_SPATIAL_REFINEMENT is True
    assert model_cfg.DG_REFINEMENT_CHANNELS == 24
    assert loss_cfg.representation_distance == "infonce"
    assert loss_cfg.infonce_temperature == 0.3
    assert loss_cfg.dg_rep.weight == 0.7
    assert loss_cfg.tau_rep == 0.4


def test_alias_dataset_names_keep_imagenet_class_count() -> None:
    for dataset_name in ("ImageNet", "imagenet", "imagefolder"):
        cfg = TrainConfig(dataset=dataset_name)
        assert cfg.num_classes == 1000


def test_nested_configs_reflect_new_engine_slice_fields() -> None:
    cfg = TrainConfig(
        task="dg_pretrain",
        dataset="ImageNet",
        split="train",
        val_split="validation",
        loss_mode="hybrid",
        representation_distance="infonce",
        infonce_temperature=0.15,
        dg_spatial_refinement=True,
        dg_refinement_channels=12,
    )

    train_data_cfg = cfg.build_data_config(train=True)
    eval_data_cfg = cfg.build_data_config(train=False)
    model_cfg = cfg.build_model_config()
    loss_cfg = cfg.build_loss_config()

    assert train_data_cfg.dataset == "ImageNet"
    assert train_data_cfg.mode == "ssl"
    assert train_data_cfg.split == "train"
    assert eval_data_cfg.split == "validation"
    assert model_cfg.DG_SPATIAL_REFINEMENT is True
    assert model_cfg.DG_REFINEMENT_CHANNELS == 12
    assert loss_cfg.representation_distance == "infonce"
    assert loss_cfg.infonce_temperature == 0.15


def test_write_json_serializes_model_and_loss_config_additions(tmp_path: Path) -> None:
    cfg = TrainConfig(
        task="dg_pretrain",
        dataset="imagefolder",
        run_id="engine-smoke",
        representation_distance="infonce",
        infonce_temperature=0.4,
        dg_spatial_refinement=True,
        dg_refinement_channels=20,
    )

    output_path = cfg.write_json(tmp_path / "config.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.is_file()
    assert payload["num_classes"] == 1000
    assert payload["loss_config"]["representation_distance"] == "infonce"
    assert payload["loss_config"]["infonce_temperature"] == 0.4
    assert payload["model_config"]["DG_SPATIAL_REFINEMENT"] is True
    assert payload["model_config"]["DG_REFINEMENT_CHANNELS"] == 20

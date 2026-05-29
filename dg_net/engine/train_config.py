"""Configuration assembly for DGNet engine training and evaluation scripts.

This module is the only shared configuration surface for the engine package.
It intentionally imports only public APIs from the sibling dataset/loss/model
packages and exposes helpers that executable scripts can reuse.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Optional

import numpy as np
import torch


ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import DataConfig  # noqa: E402
from loss.loss_config import LossConfig, ScheduleConfig  # noqa: E402
from model.dg_model import DgNetConfig  # noqa: E402


TASKS = ("dg_pretrain", "linear_probe", "knn_eval", "finetune", "eval")
LOOP_MODES = ("step", "epoch")
CLASS_COUNTS = {"STL10": 10, "CIFAR100": 100, "Flowers102": 102, "iNaturalist": 10000}


def _parse_tuple(value: str | Iterable[float] | None, *, length: int = 3) -> tuple[float, ...] | None:
    if value is None or isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(float(x) for x in value)
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"expected {length} comma-separated floats, got {value!r}")
    return tuple(float(p) for p in parts)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {help_text}")
    parser.set_defaults(**{dest: default})


@dataclass
class TrainConfig:
    """Single engine config for pretraining, transfer training, and eval."""

    task: str = "dg_pretrain"
    loop_mode: str = "step"
    total_steps: int = 100
    total_epochs: int = 1
    validate_every_steps: int = 50
    validate_every_epochs: int = 1
    save_every_steps: int = 100
    save_every_epochs: int = 1
    output_root: str = "output"
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    seed: int = 42
    deterministic: bool = False
    device: str = "auto"

    # Dataset
    dataset: str = "STL10"
    data_path: str = "/home/jeongyuseong/바탕화면/datasets"
    split: str = "train"
    val_split: Optional[str] = None
    download: bool = False
    classes: Optional[list[int]] = None
    max_samples: Optional[int] = None
    img_size: int = 32
    batch_size: int = 4
    eval_batch_size: Optional[int] = None
    num_workers: int = 0
    use_augmentation: bool = False
    crop_scale: tuple[float, float] = (0.08, 1.0)
    hflip_prob: float = 0.5
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ssl_num_views: int = 2
    drop_last: bool = True
    pin_memory: bool = False
    persistent_workers: bool = False

    # Model
    patch_size: int = 8
    embed_dim: int = 64
    depth: int = 2
    num_heads: int = 4
    encoder_embed_dim: Optional[int] = None
    encoder_depth: Optional[int] = None
    encoder_num_heads: Optional[int] = None
    decoder_embed_dim: int = 48
    decoder_depth: Optional[int] = None
    decoder_num_heads: int = 4
    dg_embed_dim: Optional[int] = None
    dg_depth: Optional[int] = None
    dg_num_heads: Optional[int] = None
    projection_dim: int = 32
    dg_architect: str = "HYBRID"
    cnn_architect: str = "CONVNEXT"
    clamp_degraded: bool = False
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    drop_path: float = 0.0

    # Loss
    alpha_inference: float = 1.0
    alpha_target: float = 1.0
    lambda_budget: float = 1.0
    lambda_reg: float = 0.1
    tau_deg: float = 0.25
    beta_mask: float = 0.15
    loss_reduction: str = "mean"

    # Optimizer
    lr_inference: float = 1.0e-3
    lr_degradation: float = 1.0e-3
    lr: float = 1.0e-3
    classifier_lr: float = 1.0e-3
    encoder_lr: float = 1.0e-4
    weight_decay: float = 1.0e-4

    # Classification / kNN
    num_classes: Optional[int] = None
    freeze_encoder: bool = True
    knn_k: int = 20
    temperature: float = 0.07
    topk: tuple[int, int] = (1, 5)

    # Checkpointing
    checkpoint: Optional[str] = None
    resume: Optional[str] = None
    map_location: str = "cpu"
    eval_max_batches: Optional[int] = 10

    # Runtime-populated paths
    output_dir: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.loop_mode not in LOOP_MODES:
            raise ValueError(f"loop_mode must be one of {LOOP_MODES}, got {self.loop_mode!r}")
        if self.total_steps < 0 or self.total_epochs < 0:
            raise ValueError("total_steps and total_epochs must be non-negative")
        if self.img_size != int(self.img_size) or self.patch_size != int(self.patch_size):
            raise ValueError("img_size and patch_size must be integers")
        if self.img_size % self.patch_size != 0:
            raise ValueError("DataConfig.img_size must be divisible by DgNetConfig.PATCH_SIZE")
        if self.clamp_degraded and (tuple(self.mean) != (0.0, 0.0, 0.0) or tuple(self.std) != (1.0, 1.0, 1.0)):
            raise ValueError("CLAMP_DEGRADED=True requires identity dataset normalization for pixel semantics")
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        if self.num_classes is None:
            self.num_classes = CLASS_COUNTS.get(self.dataset, 10)
        self.mean = tuple(float(x) for x in self.mean)  # type: ignore[assignment]
        self.std = tuple(float(x) for x in self.std)  # type: ignore[assignment]
        self.crop_scale = tuple(float(x) for x in self.crop_scale)  # type: ignore[assignment]

    @classmethod
    def from_args(cls, args: argparse.Namespace | dict[str, Any]) -> "TrainConfig":
        data = vars(args).copy() if isinstance(args, argparse.Namespace) else dict(args)
        for key in ("mean", "std"):
            if key in data and data[key] is not None:
                data[key] = _parse_tuple(data[key], length=3)
        if "crop_scale" in data and data["crop_scale"] is not None:
            data["crop_scale"] = _parse_tuple(data["crop_scale"], length=2)
        if isinstance(data.get("classes"), str):
            data["classes"] = [int(x) for x in data["classes"].split(",") if x.strip()]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__ and v is not None})

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def make_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"-{self.run_name}" if self.run_name else ""
        self.run_id = f"{self.task}-{stamp}{suffix}"
        return self.run_id

    def prepare_output_dir(self) -> Path:
        run_id = self.make_run_id()
        output_dir = Path(self.output_root).expanduser() / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(output_dir)
        return output_dir

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resolved_device"] = str(self.resolve_device())
        return data

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
        return output

    def build_data_config(self, *, split: str | None = None, mode: str | None = None, train: bool = True) -> DataConfig:
        mode = mode or ("ssl" if self.task == "dg_pretrain" else "classification")
        selected_split = split or (self.split if train else (self.val_split or self._default_val_split()))
        return DataConfig(
            dataset=self.dataset,
            data_path=self.data_path,
            mode=mode,
            split=selected_split,
            download=self.download,
            classes=self.classes,
            max_samples=self.max_samples,
            img_size=self.img_size,
            use_augmentation=self.use_augmentation if train else False,
            crop_scale=self.crop_scale,
            hflip_prob=self.hflip_prob,
            mean=self.mean,
            std=self.std,
            ssl_num_views=self.ssl_num_views,
            batch_size=self.batch_size if train else int(self.eval_batch_size or self.batch_size),
            num_workers=self.num_workers,
            shuffle=train,
            drop_last=self.drop_last if train else False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            seed=self.seed + (0 if train else 1000),
            output_dir=str(Path(self.output_dir or self.output_root) / "dataset_reports"),
            write_report=True,
        )

    def _default_val_split(self) -> str:
        if self.dataset in {"CIFAR100", "STL10"}:
            return "test"
        if self.dataset == "Flowers102":
            return "val"
        return "val"

    def build_loss_config(self) -> LossConfig:
        return LossConfig(
            alpha_inference=self.alpha_inference,
            alpha_target=self.alpha_target,
            lambda_budget=self.lambda_budget,
            lambda_reg=self.lambda_reg,
            tau_deg=self.tau_deg,
            beta_mask=self.beta_mask,
            reduction=self.loss_reduction,  # type: ignore[arg-type]
            alpha_inference_schedule=ScheduleConfig(),
            alpha_target_schedule=ScheduleConfig(),
            lambda_budget_schedule=ScheduleConfig(),
            lambda_reg_schedule=ScheduleConfig(),
        )

    def build_model_config(self) -> DgNetConfig:
        return DgNetConfig(
            IMG_SIZE=self.img_size,
            PATCH_SIZE=self.patch_size,
            EMBED_DIM=self.embed_dim,
            DEPTH=self.depth,
            NUM_HEADS=self.num_heads,
            ENCODER_EMBED_DIM=self.encoder_embed_dim,
            ENCODER_DEPTH=self.encoder_depth,
            ENCODER_NUM_HEADS=self.encoder_num_heads,
            DECODER_EMBED_DIM=self.decoder_embed_dim,
            DECODER_DEPTH=self.decoder_depth,
            DECODER_NUM_HEADS=self.decoder_num_heads,
            DG_EMBED_DIM=self.dg_embed_dim,
            DG_DEPTH=self.dg_depth,
            DG_NUM_HEADS=self.dg_num_heads,
            PROJECTION_DIM=self.projection_dim,
            DG_ARCHITECT=self.dg_architect,
            CNN_ARCHITECT=self.cnn_architect,
            CLAMP_DEGRADED=self.clamp_degraded,
            ATTN_DROPOUT=self.attn_dropout,
            PROJ_DROPOUT=self.proj_dropout,
            DROP_PATH=self.drop_path,
        )


def build_arg_parser(default_task: str = "dg_pretrain") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DGNet engine runner")
    parser.add_argument("--task", choices=TASKS, default=default_task)
    parser.add_argument("--loop-mode", choices=LOOP_MODES, default="step")
    parser.add_argument("--total-steps", type=int, default=100)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--validate-every-steps", type=int, default=50)
    parser.add_argument("--validate-every-epochs", type=int, default=1)
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    _add_bool_arg(parser, "deterministic", False, "deterministic backend settings")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--data-path", default="/home/jeongyuseong/바탕화면/datasets")
    parser.add_argument("--dataset", default="STL10")
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default=None)
    _add_bool_arg(parser, "download", False, "dataset download")
    parser.add_argument("--classes", default=None, help="comma-separated class ids")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    _add_bool_arg(parser, "use-augmentation", False, "data augmentation")
    parser.add_argument("--crop-scale", default="0.08,1.0")
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--mean", default="0.0,0.0,0.0")
    parser.add_argument("--std", default="1.0,1.0,1.0")
    parser.add_argument("--ssl-num-views", type=int, default=2)
    _add_bool_arg(parser, "drop-last", True, "drop last incomplete train batch")
    _add_bool_arg(parser, "pin-memory", False, "DataLoader pin memory")
    _add_bool_arg(parser, "persistent-workers", False, "persistent DataLoader workers")

    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--encoder-embed-dim", type=int, default=None)
    parser.add_argument("--encoder-depth", type=int, default=None)
    parser.add_argument("--encoder-num-heads", type=int, default=None)
    parser.add_argument("--decoder-embed-dim", type=int, default=48)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--decoder-num-heads", type=int, default=4)
    parser.add_argument("--dg-embed-dim", type=int, default=None)
    parser.add_argument("--dg-depth", type=int, default=None)
    parser.add_argument("--dg-num-heads", type=int, default=None)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--dg-architect", choices=("VIT", "CNN", "HYBRID"), default="HYBRID")
    parser.add_argument("--cnn-architect", choices=("RESNET", "CONVNEXT"), default="CONVNEXT")
    _add_bool_arg(parser, "clamp-degraded", False, "clamp degraded image to [0,1]")
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--proj-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0)

    parser.add_argument("--alpha-inference", type=float, default=1.0)
    parser.add_argument("--alpha-target", type=float, default=1.0)
    parser.add_argument("--lambda-budget", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--tau-deg", type=float, default=0.25)
    parser.add_argument("--beta-mask", type=float, default=0.15)
    parser.add_argument("--loss-reduction", choices=("mean", "sum"), default="mean")

    parser.add_argument("--lr-inference", type=float, default=1.0e-3)
    parser.add_argument("--lr-degradation", type=float, default=1.0e-3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--classifier-lr", type=float, default=1.0e-3)
    parser.add_argument("--encoder-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)

    parser.add_argument("--num-classes", type=int, default=None)
    _add_bool_arg(parser, "freeze-encoder", True, "freeze encoder for linear probing")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--eval-max-batches", type=int, default=10)
    return parser


def get_trainer(train_cfg: TrainConfig):
    """Create the trainer matching ``train_cfg.task`` using executable modules lazily."""

    if train_cfg.task == "dg_pretrain":
        from dg_train import DGTrainer

        return DGTrainer(train_cfg)
    if train_cfg.task == "linear_probe":
        from linear_probe_train import LinearProbeTrainer

        return LinearProbeTrainer(train_cfg)
    if train_cfg.task == "finetune":
        from finetune_train import FineTuneTrainer

        return FineTuneTrainer(train_cfg)
    if train_cfg.task == "knn_eval":
        from knn_eval import KNNEvaluator

        return KNNEvaluator(train_cfg)
    raise ValueError("task='eval' has no stateful trainer; use eval.py utilities or a concrete task")


# Backward-compatible typo alias requested by the harness instructions.
TrinConfig = TrainConfig


__all__ = [
    "TrainConfig",
    "TrinConfig",
    "TASKS",
    "LOOP_MODES",
    "build_arg_parser",
    "get_trainer",
]

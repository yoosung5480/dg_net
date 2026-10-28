"""DGNet self-supervised alternating pretraining loop."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import torch

ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import inspect_loader, prepare_dataloader  # noqa: E402
from loss.loss import prepare_dgloss  # noqa: E402
from model.dg_model import DGNet  # noqa: E402
from train_config import TrainConfig, build_arg_parser  # noqa: E402
from eval import (  # noqa: E402
    DG_COLUMNS,
    append_metrics_csv,
    batch_to_images,
    evaluate_dgnet,
    plot_accuracy_curve,
    plot_loss_curve,
    save_eval_artifacts,
)


class DGTrainer:
    """Alternating I/M optimizer trainer for DGNet pretraining."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.cfg.task = "dg_pretrain"
        self.cfg.seed_everything()
        self.device = cfg.resolve_device()
        self.output_dir = cfg.prepare_output_dir()
        self.csv_path = self.output_dir / "train.csv"
        self.logger = self._make_logger()
        self.start_time = time.time()
        self.global_step = 0
        self.epoch = 0
        self.best_validation_score = -math.inf

        self.cfg.write_json(self.output_dir / "config.json")
        self.logger.info("DGNet pretraining started")
        self.logger.info("device=%s torch=%s cuda_available=%s", self.device, torch.__version__, torch.cuda.is_available())
        self.logger.info("config=%s", json.dumps(self.cfg.to_dict(), ensure_ascii=False, default=str))

        self.train_loader = prepare_dataloader(cfg.build_data_config(mode="ssl", train=True))
        self.val_loader = prepare_dataloader(cfg.build_data_config(mode="ssl", train=False))
        self.logger.info("train_loader: %s", inspect_loader(self.train_loader, mode="ssl"))
        self.logger.info("val_loader: %s", inspect_loader(self.val_loader, mode="ssl"))

        self.model = DGNet(cfg.build_model_config()).to(self.device)
        if cfg.checkpoint:
            self.model = DGNet.load(cfg.checkpoint, map_location=self.device, verbose=False).to(self.device)
            self._check_model_config()
            self.logger.info("loaded pretrained DGNet from %s", cfg.checkpoint)
        self.loss_fn = prepare_dgloss(cfg.build_loss_config()).to(self.device)
        self.optimizer_i = torch.optim.AdamW(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + list(self.model.projection.parameters()),
            lr=cfg.lr_inference,
            weight_decay=cfg.weight_decay,
        )
        self.optimizer_m = torch.optim.AdamW(
            self.model.degradation.parameters(), lr=cfg.lr_degradation, weight_decay=cfg.weight_decay
        )
        if cfg.resume:
            self._load_resume(cfg.resume)

        first_batch = next(iter(self.train_loader))
        self.fixed_sample = batch_to_images(first_batch, self.device).detach()
        self.save_checkpoint("epoch000000" if self.cfg.loop_mode == "epoch" else "step000000", step=0, epoch=0)

    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"dg_train.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(self.output_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def _check_model_config(self) -> None:
        expected = self.cfg.build_model_config()
        if self.model.cfg.IMG_SIZE != expected.IMG_SIZE or self.model.cfg.PATCH_SIZE != expected.PATCH_SIZE:
            raise ValueError(
                "checkpoint model config mismatch: "
                f"checkpoint IMG/PATCH={(self.model.cfg.IMG_SIZE, self.model.cfg.PATCH_SIZE)} "
                f"requested={(expected.IMG_SIZE, expected.PATCH_SIZE)}"
            )

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _phase_i(self) -> None:
        self._set_requires_grad(self.model.degradation, False)
        self._set_requires_grad(self.model.encoder, True)
        self._set_requires_grad(self.model.decoder, True)
        self._set_requires_grad(self.model.projection, True)

    def _phase_m(self) -> None:
        self._set_requires_grad(self.model.degradation, True)
        self._set_requires_grad(self.model.encoder, False)
        self._set_requires_grad(self.model.decoder, False)
        self._set_requires_grad(self.model.projection, False)

    def train_step(self, batch: Any) -> dict[str, float]:
        images = batch_to_images(batch, self.device)
        self.model.train()

        self._phase_i()
        self.optimizer_i.zero_grad(set_to_none=True)
        values_i = self.loss_fn(self.model(images), step=self.global_step)
        values_i.inference_loss.backward()
        self.optimizer_i.step()

        self._phase_m()
        self.optimizer_m.zero_grad(set_to_none=True)
        values_m = self.loss_fn(self.model(images), step=self.global_step)
        values_m.degradation_loss.backward()
        self.optimizer_m.step()

        # Leave model fully trainable for checkpoints/eval state restoration.
        self._set_requires_grad(self.model.degradation, True)
        self._set_requires_grad(self.model.encoder, True)
        self._set_requires_grad(self.model.decoder, True)
        self._set_requires_grad(self.model.projection, True)

        metrics = values_m.metrics()
        metrics.update(
            {
                "train_loss": float(values_i.inference_loss.detach().item() + values_m.degradation_loss.detach().item()),
                "inference_loss": float(values_i.inference_loss.detach().item()),
                "degradation_loss": float(values_m.degradation_loss.detach().item()),
                "mask_budget_distance": metrics.get("budget_distance", math.nan),
            }
        )
        return metrics

    def validate(self) -> dict[str, float]:
        metrics = evaluate_dgnet(
            self.model,
            self.loss_fn,
            self.val_loader,
            self.device,
            step=self.global_step,
            max_batches=self.cfg.eval_max_batches,
        )
        score = metrics.get("validation_score", math.nan)
        if math.isfinite(score) and score > self.best_validation_score:
            self.best_validation_score = score
        return metrics

    def _base_row(self, *, phase: str, split: str = "train") -> dict[str, Any]:
        return {
            "run_id": self.cfg.run_id,
            "task": self.cfg.task,
            "loop_mode": self.cfg.loop_mode,
            "epoch": self.epoch,
            "step": self.global_step,
            "split": split,
            "phase": phase,
            "lr_inference": self.optimizer_i.param_groups[0]["lr"],
            "lr_degradation": self.optimizer_m.param_groups[0]["lr"],
            "elapsed_sec": time.time() - self.start_time,
        }

    def _record(self, row: dict[str, Any]) -> None:
        append_metrics_csv(self.csv_path, row, DG_COLUMNS)

    def _make_tag(self, *, step: int | None = None, epoch: int | None = None) -> str:
        if self.cfg.loop_mode == "epoch" or epoch is not None:
            return f"epoch{int(self.epoch if epoch is None else epoch):06d}"
        return f"step{int(self.global_step if step is None else step):06d}"

    def save_checkpoint(self, tag: str, *, step: int | None = None, epoch: int | None = None) -> Path:
        tag_dir = self.output_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(tag_dir)
        self.cfg.write_json(tag_dir / "train_config.json")
        # Keep config.json present after model.save; train_config is separate to avoid breaking DGNet.load.
        torch.save(
            {
                "global_step": self.global_step if step is None else step,
                "epoch": self.epoch if epoch is None else epoch,
                "train_config": self.cfg.to_dict(),
                "optimizer_i": self.optimizer_i.state_dict(),
                "optimizer_m": self.optimizer_m.state_dict(),
                "best_validation_score": self.best_validation_score,
            },
            tag_dir / "checkpoint.pt",
        )
        save_eval_artifacts(
            self.model,
            self.loss_fn,
            self.fixed_sample,
            tag_dir,
            tag,
            step=self.global_step if step is None else step,
            write_model_log=True,
        )
        self.logger.info("checkpoint saved: %s", tag_dir)
        return tag_dir

    def _load_resume(self, resume: str) -> None:
        path = Path(resume)
        checkpoint_path = path / "checkpoint.pt" if path.is_dir() else path
        payload = torch.load(checkpoint_path, map_location=self.device)
        model_dir = checkpoint_path.parent
        if (model_dir / "model.pt").exists() and (model_dir / "config.json").exists():
            self.model = DGNet.load(model_dir, map_location=self.device, verbose=False).to(self.device)
        self.optimizer_i.load_state_dict(payload["optimizer_i"])
        self.optimizer_m.load_state_dict(payload["optimizer_m"])
        self.global_step = int(payload.get("global_step", 0))
        self.epoch = int(payload.get("epoch", 0))
        self.best_validation_score = float(payload.get("best_validation_score", -math.inf))
        self.logger.info("resumed training from %s at step=%d epoch=%d", checkpoint_path, self.global_step, self.epoch)

    def _should_validate_step(self) -> bool:
        return self.cfg.validate_every_steps > 0 and self.global_step % self.cfg.validate_every_steps == 0

    def _should_save_step(self) -> bool:
        return self.cfg.save_every_steps > 0 and self.global_step % self.cfg.save_every_steps == 0

    def _after_validation(self, train_metrics: dict[str, float] | None, phase: str) -> dict[str, float]:
        metrics = self.validate()
        row = self._base_row(phase=phase, split="validation")
        if train_metrics:
            row.update({k: v for k, v in train_metrics.items() if k in DG_COLUMNS})
        row.update(metrics)
        # CSV column aliases expected by agent.md.
        row["mask_budget_distance"] = row.get("mask_budget_distance", row.get("budget_distance", math.nan))
        row["alpha_inference"] = row.get("alpha_inference", math.nan)
        row["alpha_target"] = row.get("alpha_target", math.nan)
        row["lambda_budget"] = row.get("lambda_budget", math.nan)
        row["lambda_reg"] = row.get("lambda_reg", math.nan)
        self._record(row)
        self.logger.info(
            "validation step=%d epoch=%d loss=%.6f score=%.6f",
            self.global_step,
            self.epoch,
            row.get("validation_loss", math.nan),
            row.get("validation_score", math.nan),
        )
        plot_loss_curve(self.csv_path, self.output_dir / "loss_curve.png")
        plot_accuracy_curve(self.csv_path, self.output_dir / "accuracy.png")
        return metrics

    def run(self) -> None:
        try:
            if self.cfg.loop_mode == "step":
                self._run_step_mode()
            else:
                self._run_epoch_mode()
            plot_loss_curve(self.csv_path, self.output_dir / "loss_curve.png")
            plot_accuracy_curve(self.csv_path, self.output_dir / "accuracy.png")
            self.logger.info("DGNet pretraining completed: output_dir=%s", self.output_dir)
        except Exception:
            self.logger.error("training failed at step=%d epoch=%d\n%s", self.global_step, self.epoch, traceback.format_exc())
            raise

    def _run_step_mode(self) -> None:
        iterator = iter(self.train_loader)
        while self.global_step < self.cfg.total_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                batch = next(iterator)
                self.epoch += 1
            self.global_step += 1
            metrics = self.train_step(batch)
            row = self._base_row(phase="I/M", split="train")
            row.update({k: v for k, v in metrics.items() if k in set(DG_COLUMNS) | {"budget_distance"}})
            row["mask_budget_distance"] = row.get("mask_budget_distance", row.get("budget_distance", math.nan))
            self._record(row)
            if self._should_validate_step():
                self._after_validation(metrics, phase="validation")
            if self._should_save_step():
                self.save_checkpoint(self._make_tag(step=self.global_step), step=self.global_step, epoch=self.epoch)

        if self.global_step and (self.cfg.save_every_steps <= 0 or self.global_step % self.cfg.save_every_steps != 0):
            self.save_checkpoint(self._make_tag(step=self.global_step), step=self.global_step, epoch=self.epoch)
        if self.global_step and (self.cfg.validate_every_steps <= 0 or self.global_step % self.cfg.validate_every_steps != 0):
            self._after_validation(None, phase="final_validation")

    def _run_epoch_mode(self) -> None:
        for epoch in range(self.epoch + 1, self.cfg.total_epochs + 1):
            self.epoch = epoch
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            last_metrics: dict[str, float] | None = None
            for batch in self.train_loader:
                self.global_step += 1
                last_metrics = self.train_step(batch)
                row = self._base_row(phase="I/M", split="train")
                row.update({k: v for k, v in last_metrics.items() if k in set(DG_COLUMNS) | {"budget_distance"}})
                row["mask_budget_distance"] = row.get("mask_budget_distance", row.get("budget_distance", math.nan))
                self._record(row)
            if self.cfg.validate_every_epochs > 0 and epoch % self.cfg.validate_every_epochs == 0:
                self._after_validation(last_metrics, phase="validation")
            if self.cfg.save_every_epochs > 0 and epoch % self.cfg.save_every_epochs == 0:
                self.save_checkpoint(self._make_tag(epoch=epoch), step=self.global_step, epoch=epoch)

        if self.cfg.total_epochs and (self.cfg.save_every_epochs <= 0 or self.cfg.total_epochs % self.cfg.save_every_epochs != 0):
            self.save_checkpoint(self._make_tag(epoch=self.cfg.total_epochs), step=self.global_step, epoch=self.cfg.total_epochs)
        if self.cfg.total_epochs and (self.cfg.validate_every_epochs <= 0 or self.cfg.total_epochs % self.cfg.validate_every_epochs != 0):
            self._after_validation(None, phase="final_validation")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser(default_task="dg_pretrain")
    args = parser.parse_args(argv)
    cfg = TrainConfig.from_args(args)
    cfg.task = "dg_pretrain"
    DGTrainer(cfg).run()


if __name__ == "__main__":
    main()

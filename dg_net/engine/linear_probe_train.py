"""Linear probe training on frozen DGNet representations."""

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
import torch.nn.functional as F

ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import inspect_loader, prepare_dataloader  # noqa: E402
from model.dg_model import DGNet  # noqa: E402
from train_config import TrainConfig, build_arg_parser  # noqa: E402
from eval import (  # noqa: E402
    CLASS_COLUMNS,
    append_metrics_csv,
    batch_to_images,
    batch_to_xy,
    evaluate_classifier,
    extract_features,
    plot_accuracy_curve,
    plot_loss_curve,
    save_eval_artifacts,
)


class LinearProbeTrainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.cfg.task = "linear_probe"
        self.cfg.seed_everything()
        self.device = cfg.resolve_device()
        self.output_dir = cfg.prepare_output_dir()
        self.csv_path = self.output_dir / "train.csv"
        self.logger = self._make_logger()
        self.start_time = time.time()
        self.global_step = 0
        self.epoch = 0
        self.best_accuracy = -math.inf
        self.cfg.write_json(self.output_dir / "config.json")
        self.logger.info("Linear probe started config=%s", json.dumps(self.cfg.to_dict(), ensure_ascii=False, default=str))

        self.train_loader = prepare_dataloader(cfg.build_data_config(mode="classification", train=True))
        self.val_loader = prepare_dataloader(cfg.build_data_config(mode="classification", train=False))
        self.logger.info("train_loader: %s", inspect_loader(self.train_loader, mode="classification"))
        self.logger.info("val_loader: %s", inspect_loader(self.val_loader, mode="classification"))

        self.model = self._load_or_create_model().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.classifier = torch.nn.Linear(self.model.cfg.PROJECTION_DIM, int(cfg.num_classes or 10)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=cfg.classifier_lr, weight_decay=cfg.weight_decay)
        if cfg.resume:
            self._load_resume(cfg.resume)
        first_batch = next(iter(self.train_loader))
        self.fixed_sample = batch_to_images(first_batch, self.device).detach()
        self.save_checkpoint("epoch000000" if self.cfg.loop_mode == "epoch" else "step000000")

    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"linear_probe.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(self.output_dir / "run.log", encoding="utf-8")
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def _load_or_create_model(self) -> DGNet:
        if self.cfg.checkpoint:
            return DGNet.load(self.cfg.checkpoint, map_location=self.device, verbose=False)
        return DGNet(self.cfg.build_model_config())

    def _load_resume(self, resume: str) -> None:
        path = Path(resume)
        ckpt = path / "checkpoint.pt" if path.is_dir() else path
        payload = torch.load(ckpt, map_location=self.device)
        self.classifier.load_state_dict(payload["classifier"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))
        self.epoch = int(payload.get("epoch", 0))
        self.best_accuracy = float(payload.get("best_accuracy", -math.inf))
        self.logger.info("resumed from %s", ckpt)

    def train_step(self, batch: Any) -> dict[str, float]:
        images, labels = batch_to_xy(batch, self.device)
        self.classifier.train()
        with torch.no_grad():
            features = extract_features(self.model, images)
        logits = self.classifier(features)
        loss = F.cross_entropy(logits, labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        pred = logits.argmax(dim=1)
        accuracy = pred.eq(labels).float().mean().item()
        return {"train_loss": loss.item(), "accuracy": accuracy, "top1": accuracy, "validation_score": accuracy}

    def validate(self) -> dict[str, float]:
        metrics = evaluate_classifier(self.model, self.classifier, self.val_loader, self.device, max_batches=self.cfg.eval_max_batches)
        acc = metrics.get("accuracy", math.nan)
        if math.isfinite(acc) and acc > self.best_accuracy:
            self.best_accuracy = acc
        return metrics

    def _base_row(self, split: str) -> dict[str, Any]:
        return {
            "run_id": self.cfg.run_id,
            "task": self.cfg.task,
            "loop_mode": self.cfg.loop_mode,
            "epoch": self.epoch,
            "step": self.global_step,
            "split": split,
            "lr": self.optimizer.param_groups[0]["lr"],
            "elapsed_sec": time.time() - self.start_time,
        }

    def save_checkpoint(self, tag: str) -> Path:
        tag_dir = self.output_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(tag_dir)
        self.cfg.write_json(tag_dir / "train_config.json")
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "train_config": self.cfg.to_dict(),
                "classifier": self.classifier.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_accuracy": self.best_accuracy,
            },
            tag_dir / "checkpoint.pt",
        )
        save_eval_artifacts(self.model, None, self.fixed_sample, tag_dir, tag, step=self.global_step)
        self.logger.info("checkpoint saved: %s", tag_dir)
        return tag_dir

    def _tag(self) -> str:
        return f"epoch{self.epoch:06d}" if self.cfg.loop_mode == "epoch" else f"step{self.global_step:06d}"

    def _validate_and_record(self, train_metrics: dict[str, float] | None = None) -> None:
        metrics = self.validate()
        row = self._base_row("validation")
        if train_metrics:
            row.update(train_metrics)
        row.update(metrics)
        append_metrics_csv(self.csv_path, row, CLASS_COLUMNS)
        plot_loss_curve(self.csv_path, self.output_dir / "loss_curve.png")
        plot_accuracy_curve(self.csv_path, self.output_dir / "accuracy.png")
        self.logger.info("validation step=%d epoch=%d acc=%.6f loss=%.6f", self.global_step, self.epoch, row.get("accuracy", math.nan), row.get("validation_loss", math.nan))

    def run(self) -> None:
        try:
            if self.cfg.loop_mode == "step":
                self._run_step_mode()
            else:
                self._run_epoch_mode()
            plot_loss_curve(self.csv_path, self.output_dir / "loss_curve.png")
            plot_accuracy_curve(self.csv_path, self.output_dir / "accuracy.png")
            self.logger.info("Linear probe completed output_dir=%s", self.output_dir)
        except Exception:
            self.logger.error("linear probe failed\n%s", traceback.format_exc())
            raise

    def _record_train(self, metrics: dict[str, float]) -> None:
        row = self._base_row("train")
        row.update(metrics)
        append_metrics_csv(self.csv_path, row, CLASS_COLUMNS)

    def _run_step_mode(self) -> None:
        iterator = iter(self.train_loader)
        last = None
        while self.global_step < self.cfg.total_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                batch = next(iterator)
                self.epoch += 1
            self.global_step += 1
            last = self.train_step(batch)
            self._record_train(last)
            if self.cfg.validate_every_steps > 0 and self.global_step % self.cfg.validate_every_steps == 0:
                self._validate_and_record(last)
            if self.cfg.save_every_steps > 0 and self.global_step % self.cfg.save_every_steps == 0:
                self.save_checkpoint(self._tag())
        if self.cfg.validate_every_steps <= 0 or self.global_step % self.cfg.validate_every_steps != 0:
            self._validate_and_record(last)
        if self.cfg.save_every_steps <= 0 or self.global_step % self.cfg.save_every_steps != 0:
            self.save_checkpoint(self._tag())

    def _run_epoch_mode(self) -> None:
        last = None
        for epoch in range(self.epoch + 1, self.cfg.total_epochs + 1):
            self.epoch = epoch
            for batch in self.train_loader:
                self.global_step += 1
                last = self.train_step(batch)
                self._record_train(last)
            if self.cfg.validate_every_epochs > 0 and epoch % self.cfg.validate_every_epochs == 0:
                self._validate_and_record(last)
            if self.cfg.save_every_epochs > 0 and epoch % self.cfg.save_every_epochs == 0:
                self.save_checkpoint(self._tag())
        if self.cfg.validate_every_epochs <= 0 or self.cfg.total_epochs % self.cfg.validate_every_epochs != 0:
            self._validate_and_record(last)
        if self.cfg.save_every_epochs <= 0 or self.cfg.total_epochs % self.cfg.save_every_epochs != 0:
            self.save_checkpoint(self._tag())


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser(default_task="linear_probe")
    args = parser.parse_args(argv)
    cfg = TrainConfig.from_args(args)
    cfg.task = "linear_probe"
    LinearProbeTrainer(cfg).run()


if __name__ == "__main__":
    main()

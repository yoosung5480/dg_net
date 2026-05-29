"""kNN evaluation for DGNet representation quality."""

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
from eval import CLASS_COLUMNS, append_metrics_csv, batch_to_xy, extract_features, plot_accuracy_curve, plot_loss_curve  # noqa: E402


class KNNEvaluator:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.cfg.task = "knn_eval"
        self.cfg.seed_everything()
        self.device = cfg.resolve_device()
        self.output_dir = cfg.prepare_output_dir()
        self.csv_path = self.output_dir / "train.csv"
        self.logger = self._make_logger()
        self.start_time = time.time()
        self.cfg.write_json(self.output_dir / "config.json")
        self.logger.info("kNN eval started config=%s", json.dumps(self.cfg.to_dict(), ensure_ascii=False, default=str))
        self.train_loader = prepare_dataloader(cfg.build_data_config(mode="classification", train=True))
        self.val_loader = prepare_dataloader(cfg.build_data_config(mode="classification", train=False))
        self.logger.info("bank_loader: %s", inspect_loader(self.train_loader, mode="classification"))
        self.logger.info("query_loader: %s", inspect_loader(self.val_loader, mode="classification"))
        self.model = self._load_or_create_model().to(self.device).eval()

    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"knn_eval.{id(self)}")
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

    @torch.no_grad()
    def _features(self, loader: Any) -> tuple[torch.Tensor, torch.Tensor]:
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for batch_index, batch in enumerate(loader):
            if self.cfg.eval_max_batches is not None and batch_index >= self.cfg.eval_max_batches:
                break
            images, target = batch_to_xy(batch, self.device)
            feat = F.normalize(extract_features(self.model, images), dim=1)
            features.append(feat.cpu())
            labels.append(target.cpu())
        if not features:
            raise RuntimeError("no features extracted for kNN evaluation")
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        bank_features, bank_labels = self._features(self.train_loader)
        query_features, query_labels = self._features(self.val_loader)
        k = min(int(self.cfg.knn_k), bank_features.shape[0])
        correct = 0
        total = 0
        num_classes = int(self.cfg.num_classes or max(int(bank_labels.max().item()) + 1, 1))
        for start in range(0, query_features.shape[0], 256):
            q = query_features[start : start + 256]
            labels = query_labels[start : start + 256]
            similarities = q @ bank_features.T
            weights, indices = similarities.topk(k, dim=1)
            neighbor_labels = bank_labels[indices]
            votes = torch.zeros(q.shape[0], num_classes, dtype=torch.float32)
            weights = torch.exp(weights / max(float(self.cfg.temperature), 1.0e-8)).cpu()
            votes.scatter_add_(1, neighbor_labels.clamp_max(num_classes - 1), weights)
            pred = votes.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.numel()
        accuracy = correct / total if total else math.nan
        return {
            "validation_loss": math.nan,
            "validation_score": accuracy,
            "accuracy": accuracy,
            "top1": accuracy,
            "top5": math.nan,
        }

    def run(self) -> None:
        try:
            metrics = self.evaluate()
            row = {
                "run_id": self.cfg.run_id,
                "task": self.cfg.task,
                "loop_mode": self.cfg.loop_mode,
                "epoch": 0,
                "step": 0,
                "split": "knn_eval",
                "train_loss": "",
                "lr": "",
                "elapsed_sec": time.time() - self.start_time,
                **metrics,
            }
            append_metrics_csv(self.csv_path, row, CLASS_COLUMNS)
            plot_loss_curve(self.csv_path, self.output_dir / "loss_curve.png")
            plot_accuracy_curve(self.csv_path, self.output_dir / "accuracy.png")
            (self.output_dir / "knn_eval.log").write_text(
                "\n".join([f"{k}: {v}" for k, v in metrics.items()]) + "\n", encoding="utf-8"
            )
            self.logger.info("kNN evaluation completed accuracy=%.6f output_dir=%s", metrics.get("accuracy", math.nan), self.output_dir)
        except Exception:
            self.logger.error("kNN eval failed\n%s", traceback.format_exc())
            raise


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser(default_task="knn_eval")
    args = parser.parse_args(argv)
    cfg = TrainConfig.from_args(args)
    cfg.task = "knn_eval"
    KNNEvaluator(cfg).run()


if __name__ == "__main__":
    main()

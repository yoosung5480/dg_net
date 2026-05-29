"""Shared evaluation, metrics, plotting, and artifact helpers for DGNet engine."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.dg_model import DGNet, save_sample_visualization, write_model_report  # noqa: E402
from loss.loss import write_loss_report  # noqa: E402


DG_COLUMNS = [
    "run_id", "task", "loop_mode", "epoch", "step", "split", "phase",
    "train_loss", "validation_loss", "validation_score", "inference_loss",
    "degradation_loss", "reconstruction_distance", "target_distance",
    "mask_budget_distance", "damage_kl", "alpha_inference", "alpha_target",
    "lambda_budget", "lambda_reg", "lr_inference", "lr_degradation", "elapsed_sec",
]

CLASS_COLUMNS = [
    "run_id", "task", "loop_mode", "epoch", "step", "split", "train_loss",
    "validation_loss", "validation_score", "accuracy", "top1", "top5", "lr", "elapsed_sec",
]


def batch_to_images(batch: Any, device: torch.device | str) -> torch.Tensor:
    """Select the actual image tensor from either SSL ``(views, labels)`` or classification batch."""

    if isinstance(batch, (tuple, list)):
        first = batch[0]
        if isinstance(first, (tuple, list)):
            images = first[0]
        else:
            images = first
    else:
        images = batch
    return images.to(device, non_blocking=True)


def batch_to_xy(batch: Any, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("classification batch must be (images, labels)")
    images, labels = batch[0], batch[1]
    if isinstance(images, (tuple, list)):
        images = images[0]
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True).long()


def scalar(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_or_nan(value: Any) -> float:
    out = scalar(value)
    return out if math.isfinite(out) else math.nan


@torch.no_grad()
def evaluate_dgnet(
    model: DGNet,
    loss_fn: Any,
    loader: Iterable[Any],
    device: torch.device | str,
    *,
    step: int | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Return validation loss/score and DGNet loss diagnostics averaged over a loader."""

    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch_to_images(batch, device)
        output = model(images)
        values = loss_fn(output, step=step or 0)
        metrics = values.metrics()
        metrics["validation_loss"] = metrics.get("combined_loss", math.nan)
        rec = metrics.get("reconstruction_distance", math.nan)
        budget = metrics.get("budget_distance", math.nan)
        target = metrics.get("target_distance", math.nan)
        score_denominator = 1.0 + finite_or_nan(rec) + finite_or_nan(budget) + finite_or_nan(target)
        metrics["validation_score"] = 1.0 / score_denominator if math.isfinite(score_denominator) else math.nan
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + finite_or_nan(value)
        count += 1
    if was_training:
        model.train()
    if count == 0:
        return {"validation_loss": math.nan, "validation_score": math.nan}
    averaged = {key: value / count for key, value in totals.items()}
    # Normalize public CSV naming.
    if "budget_distance" in averaged:
        averaged["mask_budget_distance"] = averaged["budget_distance"]
    averaged.setdefault("validation_loss", averaged.get("combined_loss", math.nan))
    averaged.setdefault("validation_score", math.nan)
    return averaged


@torch.no_grad()
def save_eval_artifacts(
    model: DGNet,
    loss_fn: Any | None,
    sample: torch.Tensor,
    output_dir: str | Path,
    tag: str,
    *,
    step: int = 0,
    write_model_log: bool = True,
) -> Path:
    """Save fixed-sample DGNet visualization and optional model/loss reports."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    output = model(sample)
    image_path = save_sample_visualization(output, directory / f"{tag}.png")
    if write_model_log:
        write_model_report(model, sample, directory)
    if loss_fn is not None:
        values = loss_fn(output, step=step)
        write_loss_report(values, directory / "loss.log", heading=f"DGNet loss report {tag}")
    if was_training:
        model.train()
    return image_path


def append_metrics_csv(csv_path: str | Path, row: Mapping[str, Any], columns: list[str] | None = None) -> Path:
    """Append one metric row, rewriting the header if new columns appear."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: ("" if value is None else value) for key, value in row.items()}
    existing_rows: list[dict[str, str]] = []
    existing_fields: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
    fieldnames: list[str] = []
    for source in (columns or [], existing_fields, list(normalized.keys())):
        for key in source:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for old in existing_rows:
            writer.writerow({key: old.get(key, "") for key in fieldnames})
        writer.writerow({key: normalized.get(key, "") for key in fieldnames})
    return path


def _read_numeric_series(csv_path: str | Path, y_candidates: list[str]) -> tuple[list[float], dict[str, list[float]]]:
    path = Path(csv_path)
    if not path.exists():
        return [], {}
    xs: list[float] = []
    series: dict[str, list[float]] = {name: [] for name in y_candidates}
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for index, row in enumerate(reader):
            x_value = row.get("step") or row.get("epoch") or index
            try:
                x = float(x_value)
            except (TypeError, ValueError):
                x = float(index)
            appended = False
            values: dict[str, float] = {}
            for name in y_candidates:
                raw = row.get(name, "")
                try:
                    y = float(raw)
                except (TypeError, ValueError):
                    y = math.nan
                values[name] = y
                appended = appended or math.isfinite(y)
            if appended:
                xs.append(x)
                for name, y in values.items():
                    series[name].append(y)
    return xs, {k: v for k, v in series.items() if any(math.isfinite(x) for x in v)}


def _write_fallback_png(output_path: str | Path, title: str, lines: list[str]) -> Path:
    from PIL import Image, ImageDraw

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (900, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), title, fill="black")
    y = 60
    for line in lines[:24]:
        draw.text((20, y), line, fill="black")
        y += 18
    image.save(path)
    return path


def plot_curve(csv_path: str | Path, output_path: str | Path, y_candidates: list[str], title: str) -> Path:
    xs, series = _read_numeric_series(csv_path, y_candidates)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not xs or not series:
        return _write_fallback_png(path, title, ["No finite numeric data available yet."])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.5))
        for name, ys in series.items():
            plt.plot(xs[: len(ys)], ys, marker="o", linewidth=1.5, label=name)
        plt.xlabel("step/epoch")
        plt.ylabel("value")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path
    except Exception as exc:  # pragma: no cover - fallback depends on local matplotlib/PIL state
        lines = [f"plot fallback: {type(exc).__name__}: {exc}"]
        for name, ys in series.items():
            finite = [y for y in ys if math.isfinite(y)]
            if finite:
                lines.append(f"{name}: first={finite[0]:.6g}, last={finite[-1]:.6g}, n={len(finite)}")
        return _write_fallback_png(path, title, lines)


def plot_loss_curve(csv_path: str | Path, output_path: str | Path) -> Path:
    return plot_curve(
        csv_path,
        output_path,
        ["train_loss", "validation_loss", "inference_loss", "degradation_loss", "reconstruction_distance"],
        "Loss curves",
    )


def plot_accuracy_curve(csv_path: str | Path, output_path: str | Path) -> Path:
    return plot_curve(csv_path, output_path, ["accuracy", "top1", "top5", "validation_score"], "Accuracy / score curves")


def extract_features(model: DGNet, images: torch.Tensor) -> torch.Tensor:
    """Return DGNet representation for classification transfer tasks."""

    output = model(images)
    return output.representation


@torch.no_grad()
def evaluate_classifier(
    model: DGNet,
    classifier: torch.nn.Module,
    loader: Iterable[Any],
    device: torch.device | str,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    was_model_training = model.training
    was_head_training = classifier.training
    model.eval()
    classifier.eval()
    total_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels = batch_to_xy(batch, device)
        logits = classifier(extract_features(model, images))
        loss = F.cross_entropy(logits, labels)
        batch_size = labels.numel()
        total_loss += loss.item() * batch_size
        total += batch_size
        max_k = min(5, logits.shape[1])
        _, pred = logits.topk(max_k, dim=1)
        correct = pred.eq(labels.view(-1, 1))
        top1 += correct[:, :1].any(dim=1).sum().item()
        top5 += correct.any(dim=1).sum().item()
    if was_model_training:
        model.train()
    if was_head_training:
        classifier.train()
    if total == 0:
        return {"validation_loss": math.nan, "accuracy": math.nan, "top1": math.nan, "top5": math.nan, "validation_score": math.nan}
    accuracy = top1 / total
    return {
        "validation_loss": total_loss / total,
        "accuracy": accuracy,
        "top1": accuracy,
        "top5": top5 / total,
        "validation_score": accuracy,
    }


__all__ = [
    "DG_COLUMNS",
    "CLASS_COLUMNS",
    "append_metrics_csv",
    "batch_to_images",
    "batch_to_xy",
    "evaluate_classifier",
    "evaluate_dgnet",
    "extract_features",
    "plot_accuracy_curve",
    "plot_loss_curve",
    "save_eval_artifacts",
]

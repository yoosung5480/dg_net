import os
import json
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.models.base_model import BaseModel
from src.models.masking_learner import DegradationLearnerNet
from src.models.auto_encoder.model_auto_encoder import MaskedAutoencoderViT
from src.models.auto_encoder.model_auto_encoder import MAEConfig
from src.config.DG_config import TrainConfig
from src.dataset.cls_datasets import prepare_dataloader


# =========================================================
# 1. Classifier Config / Model
# =========================================================
@dataclass(frozen=True)
class ClassifierConfig:
    ae_config: MAEConfig
    embed_dim: int
    num_classes: int
    pooling: str = "mean"   # "mean" | "cls"

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "classifier_config.json", "w") as f:
            json.dump(
                {
                    "ae_config": asdict(self.ae_config),
                    "embed_dim": self.embed_dim,
                    "num_classes": self.num_classes,
                    "pooling": self.pooling,
                },
                f,
                indent=4,
            )

    @classmethod
    def load(cls, path: str):
        path = Path(path)
        with open(path / "classifier_config.json", "r") as f:
            cfg_dict = json.load(f)

        return cls(
            ae_config=MAEConfig(**cfg_dict["ae_config"]),
            embed_dim=cfg_dict["embed_dim"],
            num_classes=cfg_dict["num_classes"],
            pooling=cfg_dict.get("pooling", "mean"),
        )


class Classifier(BaseModel):
    ConfigClass = ClassifierConfig

    def __init__(self, cfg: ClassifierConfig):
        super().__init__(cfg)
        self.encoder = MaskedAutoencoderViT(cfg.ae_config)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

    def forward(self, x):
        # forward_encoder return:
        # latent, mask, ids_restore
        z = self.encoder.forward_encoder(x, 0.0)[0]   # [B, N, D] or [B, 1+N, D]

        if self.cfg.pooling == "cls":
            # cls token pooling
            feat = z[:, 0]
        else:
            # mean pooling
            feat = z.mean(dim=1)

        out = self.head(feat)
        return out


# =========================================================
# 2. Utils
# =========================================================
def _ensure_dir(path: Optional[str]):
    if path is not None:
        os.makedirs(path, exist_ok=True)


def _move_batch_to_device(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def _set_classifier_trainable(model: Classifier, train_cfg: TrainConfig):
    """
    분류 실험에서는
    - encoder 내부에서 encoder / decoder 일부만 켤 수 있게 유지
    - classifier head는 항상 학습
    """
    set_requires_grad(model.encoder, False)

    enc_flag = getattr(train_cfg.param_grad, "train_encoder", True)
    dec_flag = getattr(train_cfg.param_grad, "train_decoder", False)

    encoder_like_names = [
        "patch_embed",
        "blocks",
        "norm",
        "cls_token",
        "pos_embed",
    ]
    decoder_like_names = [
        "decoder_embed",
        "mask_token",
        "decoder_pos_embed",
        "decoder_blocks",
        "decoder_norm",
        "decoder_pred",
    ]

    if enc_flag:
        for name in encoder_like_names:
            if hasattr(model.encoder, name):
                target = getattr(model.encoder, name)
                if isinstance(target, torch.nn.Parameter):
                    target.requires_grad = True
                else:
                    set_requires_grad(target, True)

    if dec_flag:
        for name in decoder_like_names:
            if hasattr(model.encoder, name):
                target = getattr(model.encoder, name)
                if isinstance(target, torch.nn.Parameter):
                    target.requires_grad = True
                else:
                    set_requires_grad(target, True)

    # head는 항상 학습
    set_requires_grad(model.head, True)


def _collect_trainable_parameters(module: torch.nn.Module):
    return [p for p in module.parameters() if p.requires_grad]


def _build_single_optimizer(
    name: str,
    params,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
):
    name = name.lower()

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {name}")


def _infer_val_data_path(train_cfg: TrainConfig) -> str:
    """
    기존 DataConfig만 최대한 활용해서 val path 추론
    """
    data_path = train_cfg.data.data_path
    dataset_name = train_cfg.data.dataset.lower()

    if dataset_name == "imagenet":
        p = Path(data_path)

        # /.../train -> /.../val
        if p.name == "train":
            candidate = p.parent / "val"
            if candidate.exists():
                return str(candidate)

        # /.../ImageNet -> /.../ImageNet/val
        candidate = p / "val"
        if candidate.exists():
            return str(candidate)

    # STL10 / custom은 기본적으로 같은 root 사용
    return data_path


def _build_train_val_loaders(train_cfg: TrainConfig):
    train_data_cfg = copy.deepcopy(train_cfg.data)
    val_data_cfg = copy.deepcopy(train_cfg.data)

    # ----- train -----
    train_data_cfg.task = "classification"
    train_data_cfg.is_train = True
    train_data_cfg.shuffle = True
    train_data_cfg.drop_last = True

    # ----- val -----
    val_data_cfg.task = "classification"
    val_data_cfg.is_train = False
    val_data_cfg.shuffle = False
    val_data_cfg.drop_last = False

    dataset_name = train_cfg.data.dataset.lower()

    if dataset_name == "stl10":
        train_data_cfg.split = "train"
        val_data_cfg.split = "test"
        val_data_cfg.data_path = train_cfg.data.data_path

    elif dataset_name == "imagenet":
        train_data_cfg.split = "train"
        val_data_cfg.split = "val"
        val_data_cfg.data_path = _infer_val_data_path(train_cfg)

    elif dataset_name == "custom":
        # custom은 기존 API상 val path가 따로 없을 수 있음.
        # 없으면 같은 경로 + eval transform으로 사용.
        train_data_cfg.split = "train"
        val_data_cfg.split = "val"
        val_data_cfg.data_path = getattr(train_cfg.data, "val_data_path", train_cfg.data.data_path)

    else:
        raise ValueError(f"Unknown dataset: {train_cfg.data.dataset}")

    train_loader = prepare_dataloader(train_data_cfg)
    val_loader = prepare_dataloader(val_data_cfg)

    return train_loader, val_loader


# =========================================================
# 3. Pretrained Encoder Loading
# =========================================================
def _extract_encoder_state_from_checkpoint(state_dict: Dict[str, torch.Tensor]):
    """
    checkpoint 전체 state_dict에서 ae_net/encoder 부분만 뽑아낸다.
    """
    new_state = {}

    for k, v in state_dict.items():
        # combined DG model saved as ae_net.xxx
        if k.startswith("ae_net."):
            new_key = k[len("ae_net."):]
            new_state[new_key] = v
        # classifier saved as encoder.xxx
        elif k.startswith("encoder."):
            new_key = k[len("encoder."):]
            new_state[new_key] = v

    return new_state


def _try_load_pretrained_encoder(classifier: Classifier, train_cfg: TrainConfig):

    if train_cfg.checkpoint_path is None:
        print("[WARN] checkpoint_path is None → random init")
        return

    ckpt_dir = Path(train_cfg.checkpoint_path)
    ckpt_file = ckpt_dir / f"{train_cfg.checkpoint_name}.pt"

    if not ckpt_file.exists():
        print(f"[WARN] checkpoint not found: {ckpt_file}")
        return

    print(f"[INFO] Loading checkpoint: {ckpt_file}")

    state_dict = torch.load(ckpt_file, map_location="cpu")

    # DGLearnerNet 형태일 경우 → ae_net 추출
    enc_state = _extract_encoder_state_from_checkpoint(state_dict)

    if len(enc_state) == 0:
        # fallback (encoder-only 저장된 경우)
        try:
            missing, unexpected = classifier.encoder.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded encoder directly. missing={len(missing)}, unexpected={len(unexpected)}")
            return
        except Exception as e:
            print(f"[WARN] direct load failed: {e}")
            return

    missing, unexpected = classifier.encoder.load_state_dict(enc_state, strict=False)

    print(f"[INFO] Pretrained encoder loaded. missing={len(missing)}, unexpected={len(unexpected)}")

# =========================================================
# 4. Logging / Plot / Checkpoint
# =========================================================
def _save_history(history: dict, save_path: str):
    _ensure_dir(save_path)

    with open(os.path.join(save_path, "history.json"), "w") as f:
        json.dump(history, f, indent=4)


def _plot_curves(history: dict, save_path: str):
    _ensure_dir(save_path)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

    # acc curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train / Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "acc_curve.png"))
    plt.close()


def _save_runtime_checkpoint(
    model: Classifier,
    optimizer,
    history: dict,
    save_path: str,
    name: str,
    epoch: int,
    best_val_acc: float,
):
    _ensure_dir(save_path)

    # BaseModel save
    model.save(save_path, name=name)

    # runtime states
    torch.save(
        {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        os.path.join(save_path, f"{name}_runtime.pt"),
    )


# =========================================================
# 5. Evaluation
# =========================================================
@torch.no_grad()
def evaluate_classification(model: Classifier, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        x, y = _move_batch_to_device(batch, device)

        logits = model(x)
        loss = criterion(logits, y)

        pred = logits.argmax(dim=1)
        correct = (pred == y).sum().item()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += correct
        total_count += bs

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)

    return avg_loss, avg_acc


# =========================================================
# 6. Main Train
# =========================================================
def train_classification(train_cfg: TrainConfig):
    """
    TrainConfig만으로 동작하는 classification training entry.

    필수 가정:
    - train_cfg.data.task = "classification"
    - train_cfg.data.num_classes 설정
    - train_cfg.model.ae_config 존재
    - (권장) train_cfg.pretrained_ckpt_path = ".../model.pt" or checkpoint directory
    """
    device = train_cfg.device
    save_path = train_cfg.save_path
    _ensure_dir(save_path)

    # -------------------------
    # data
    # -------------------------
    train_loader, val_loader = _build_train_val_loaders(train_cfg)

    # -------------------------
    # model
    # -------------------------
    clf_cfg = ClassifierConfig(
        ae_config=train_cfg.model.ae_config,
        embed_dim=train_cfg.model.ae_config.EMBED_DIM,
        num_classes=train_cfg.data.num_classes,
        pooling=getattr(train_cfg.data, "pooling", "mean"),
    )

    model = Classifier(clf_cfg).to(device)

    # pretrained encoder 로드
    _try_load_pretrained_encoder(model, train_cfg)

   # grad policy
    _set_classifier_trainable(model, train_cfg)

    # optimizer (핵심)
    if train_cfg.param_grad.train_encoder:
        # finetune
        params = _collect_trainable_parameters(model)
    else:
        # linear probe
        params = model.head.parameters()

    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg.optimizer.lr_ae,
        weight_decay=train_cfg.optimizer.weight_decay_ae
    )

    print("\n[DEBUG] Trainable params:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)

    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # history
    # -------------------------
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    global_step = 0
    stop_training = False

    # -------------------------
    # train loop
    # -------------------------
    for epoch in range(train_cfg.epochs):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for i, batch in enumerate(train_loader):
            x, y = _move_batch_to_device(batch, device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            correct = (pred == y).sum().item()

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_correct += correct
            running_count += bs

            global_step += 1

            if global_step % train_cfg.log_interval == 0:
                print(
                    f"[Train] epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} "
                    f"acc={(correct / bs):.4f}"
                )

            if train_cfg.mode == "step" and train_cfg.max_iters is not None:
                if global_step >= train_cfg.max_iters:
                    stop_training = True
                    break

        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)

        val_loss, val_acc = evaluate_classification(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_runtime_checkpoint(
                model=model,
                optimizer=optimizer,
                history=history,
                save_path=save_path,
                name="best_classifier",
                epoch=epoch,
                best_val_acc=best_val_acc,
            )

        # periodic save
        if (epoch + 1) % max(1, getattr(train_cfg, "save_interval", 1)) == 0:
            _save_runtime_checkpoint(
                model=model,
                optimizer=optimizer,
                history=history,
                save_path=save_path,
                name=f"epoch_{epoch+1}_classifier",
                epoch=epoch,
                best_val_acc=best_val_acc,
            )

        # always save last
        # _save_runtime_checkpoint(
        #     model=model,
        #     optimizer=optimizer,
        #     history=history,
        #     save_path=save_path,
        #     name="last_classifier",
        #     epoch=epoch,
        #     best_val_acc=best_val_acc,
        # )

        _save_history(history, save_path)
        _plot_curves(history, save_path)

        if stop_training:
            break

    print(f"[INFO] Training finished. best_val_acc={best_val_acc:.4f}")

    return model, history
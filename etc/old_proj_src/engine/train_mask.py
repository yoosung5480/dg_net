import os
import json
from dataclasses import asdict
from typing import Any, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

from src.config.DG_config import TrainConfig
from src.dataset.datasets import prepare_dataloader
from src.loss.DG_learn_loss import build_dg_learn_loss
from src.models.masking_learner import DegradationLearnerNet


################### utils ##############################

def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _ensure_dir(path: Optional[str]):
    if path is not None:
        os.makedirs(path, exist_ok=True)


def _save_config(save_path, ae_loss_fn, dg_loss_fn, train_cfg):
    os.makedirs(save_path, exist_ok=True)

    cfg_dict = {
        "train_cfg": asdict(train_cfg) if hasattr(train_cfg, "__dataclass_fields__") else str(train_cfg),
        "loss_cfg": {
            "distance": asdict(ae_loss_fn.cfg.distance),
            "regluation": asdict(ae_loss_fn.cfg.regluation),
            "reduction": ae_loss_fn.cfg.reduction,
        },
    }

    save_file = os.path.join(save_path, "train_config.json")

    with open(save_file, "w") as f:
        json.dump(cfg_dict, f, indent=4)

    print(f"[INFO] Saved config → {save_file}")


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


# ParamGradConfig 확장용. 기존 set_requires_grad는 그대로 유지.
def _set_ae_trainable(model, train_cfg: TrainConfig, flag: bool):
    base_model = _unwrap_model(model)

    set_requires_grad(base_model.ae_net, False)

    if not flag:
        return

    enc_flag = getattr(train_cfg.param_grad, "train_encoder", True)
    dec_flag = getattr(train_cfg.param_grad, "train_decoder", True)

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
            if hasattr(base_model.ae_net, name):
                target = getattr(base_model.ae_net, name)
                if isinstance(target, torch.nn.Parameter):
                    target.requires_grad = True
                else:
                    set_requires_grad(target, True)

    if dec_flag:
        for name in decoder_like_names:
            if hasattr(base_model.ae_net, name):
                target = getattr(base_model.ae_net, name)
                if isinstance(target, torch.nn.Parameter):
                    target.requires_grad = True
                else:
                    set_requires_grad(target, True)



def _set_dg_trainable(model, train_cfg: TrainConfig, flag: bool):
    base_model = _unwrap_model(model)
    real_flag = flag and getattr(train_cfg.param_grad, "train_degradation", True)
    set_requires_grad(base_model.dg_net, real_flag)



def _collect_trainable_parameters(module: torch.nn.Module):
    return [p for p in module.parameters() if p.requires_grad]



def _build_single_optimizer(name: str, params, lr: float, weight_decay: float, betas: Tuple[float, float]):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")



def build_optimizers(model, cfg: TrainConfig):
    base_model = _unwrap_model(model)

    _set_ae_trainable(base_model, cfg, True)
    _set_dg_trainable(base_model, cfg, True)

    ae_params = _collect_trainable_parameters(base_model.ae_net)
    dg_params = _collect_trainable_parameters(base_model.dg_net)

    if len(ae_params) == 0:
        raise ValueError("No trainable AE parameters found. Check ParamGradConfig.")
    if len(dg_params) == 0:
        raise ValueError("No trainable DG parameters found. Check ParamGradConfig.")

    opt_cfg = cfg.optimizer

    opt_ae = _build_single_optimizer(
        opt_cfg.ae_optimizer,
        ae_params,
        lr=opt_cfg.lr_ae,
        weight_decay=opt_cfg.weight_decay_ae,
        betas=opt_cfg.betas_ae,
    )

    opt_dg = _build_single_optimizer(
        opt_cfg.dg_optimizer,
        dg_params,
        lr=opt_cfg.lr_dg,
        weight_decay=opt_cfg.weight_decay_dg,
        betas=opt_cfg.betas_dg,
    )

    return opt_ae, opt_dg


def save_sample_visualization(model, x, save_path, step):
    os.makedirs(save_path, exist_ok=True)

    raw_model = _unwrap_model(model)
    raw_model.eval()

    with torch.no_grad():
        out = raw_model(x).to_dict()

        pred = out["pred"]
        x_dg = out["x_dg"]
        x = out["x"]
        z_x = out["z_x"]
        z_x_dg = out["z_x_dg"]

        # recon (AE output)
        recon = raw_model.ae_net.unpatchify(pred)

        n = min(4, x.shape[0])

        # -----------------------------
        # helper
        # -----------------------------
        def to_img(t):
            t = t.detach().cpu()
            t = t.permute(1, 2, 0).numpy()
            return np.clip(t, 0, 1)

        def to_gray_map(t):
            # [C,H,W] -> [H,W]
            return t.mean(dim=0).detach().cpu().numpy()

        def normalize_map(m):
            m = m - m.min()
            if m.max() > 0:
                m = m / m.max()
            return m

        # -----------------------------
        # cosine similarity
        # -----------------------------
        cos_sim = F.cosine_similarity(z_x, z_x_dg, dim=-1)  # [B]
        cos_dist = 1 - cos_sim

        # -----------------------------
        # 저장용 개별 figure들
        # -----------------------------
        fig_orig = plt.figure(figsize=(4*n, 3))
        fig_diff = plt.figure(figsize=(4*n, 3))
        fig_recon_diff = plt.figure(figsize=(4*n, 3))

        # -----------------------------
        # 통합 figure
        # -----------------------------
        fig_all = plt.figure(figsize=(4*n, 10))

        for i in range(n):
            xi = x[i]
            xdg = x_dg[i]
            ri = recon[i]

            # -------------------------
            # diff maps
            # -------------------------
            diff = torch.abs(xi - xdg)                # |x - x_dg|
            recon_diff = torch.abs(xdg - ri)          # |x_dg - recon|

            diff_map = normalize_map(to_gray_map(diff))
            recon_diff_map = normalize_map(to_gray_map(recon_diff))

            # -------------------------
            # stats
            # -------------------------
            diff_mean = diff.mean().item()
            diff_std = diff.std().item()

            recon_mean = recon_diff.mean().item()
            recon_std = recon_diff.std().item()

            cos_val = cos_sim[i].item()
            cos_d = cos_dist[i].item()

            stat_text = (
                f"diff μ={diff_mean:.4f}, σ={diff_std:.4f}\n"
                f"recon μ={recon_mean:.4f}, σ={recon_std:.4f}\n"
                f"cos={cos_val:.4f}, dist={cos_d:.4f}"
            )

            # =========================
            # 1. orig figure
            # =========================
            plt.figure(fig_orig.number)
            plt.subplot(1, n, i+1)
            plt.imshow(to_img(xi))
            plt.title("Original")
            plt.axis("off")

            # =========================
            # 2. diff map
            # =========================
            plt.figure(fig_diff.number)
            plt.subplot(1, n, i+1)
            plt.imshow(diff_map, cmap='jet')
            plt.title("|Δ(x)|")
            plt.axis("off")

            # =========================
            # 3. recon diff map
            # =========================
            plt.figure(fig_recon_diff.number)
            plt.subplot(1, n, i+1)
            plt.imshow(recon_diff_map, cmap='jet')
            plt.title("|dg(x) - recon|")
            plt.axis("off")

            # =========================
            # 4. 통합 figure
            # =========================
            # row1: original
            plt.figure(fig_all.number)
            plt.subplot(5, n, i+1)
            plt.imshow(to_img(xi))
            plt.title("Original")
            plt.axis("off")

            # row2: degraded
            plt.subplot(5, n, n+i+1)
            plt.imshow(to_img(xdg))
            plt.title("Masked : x - dg(x)")
            plt.axis("off")

            # row3: recon
            plt.subplot(5, n, 2*n+i+1)
            plt.imshow(to_img(ri))
            plt.title("Recon")
            plt.axis("off")

            # row4: diff
            plt.subplot(5, n, 3*n+i+1)
            plt.imshow(diff_map, cmap='jet')
            plt.title("dg(x) : x - (x - dg(x)) ")
            plt.axis("off")

            # row5: recon diff + stats
            plt.subplot(5, n, 4*n+i+1)
            plt.imshow(recon_diff_map, cmap='jet')
            plt.title("Recon Error")
            plt.xlabel(stat_text, fontsize=8)
            plt.axis("off")

        # -----------------------------
        # save all
        # -----------------------------
        plt.figure(fig_orig.number)
        plt.tight_layout()
        plt.savefig(f"{save_path}/step_orig_{step}.png")
        plt.close(fig_orig)

        plt.figure(fig_diff.number)
        plt.tight_layout()
        plt.savefig(f"{save_path}/step_difference_map_{step}.png")
        plt.close(fig_diff)

        plt.figure(fig_recon_diff.number)
        plt.tight_layout()
        plt.savefig(f"{save_path}/step_recon_difference_map_{step}.png")
        plt.close(fig_recon_diff)

        plt.figure(fig_all.number)
        plt.tight_layout()
        plt.savefig(f"{save_path}/step_{step}.png")
        plt.close(fig_all)

def _move_batch_to_device(batch: Any, device: torch.device):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Empty batch received from dataloader.")
        return batch[0].to(device)
    if isinstance(batch, dict):
        if "image" in batch:
            return batch["image"].to(device)
        first_value = next(iter(batch.values()))
        return first_value.to(device)
    return batch.to(device)



def _save_checkpoint(model, save_path: str, name: str = "dg_learner"):
    raw_model = _unwrap_model(model)
    os.makedirs(save_path, exist_ok=True)

    if hasattr(raw_model, "save"):
        raw_model.save(save_path, name=name)
        print(f"[INFO] Saved checkpoint → {os.path.join(save_path, name + '.pt')}")
        return

    ckpt_path = os.path.join(save_path, f"{name}.pt")
    torch.save(raw_model.state_dict(), ckpt_path)
    print(f"[INFO] Saved checkpoint → {ckpt_path}")



def _maybe_load_model(train_cfg: TrainConfig):
    checkpoint_path = getattr(train_cfg, "checkpoint_path", None)
    checkpoint_name = getattr(train_cfg, "checkpoint_name", "dg_learner")

    if checkpoint_path is not None and hasattr(DegradationLearnerNet, "load"):
        if os.path.isdir(checkpoint_path):
            print(f"[INFO] Loading model from → {checkpoint_path}")
            return DegradationLearnerNet.load(checkpoint_path, name=checkpoint_name)

    return DegradationLearnerNet(cfg=train_cfg.model)



def _build_dataloader(train_cfg: TrainConfig):
    loader = prepare_dataloader(train_cfg.data)

    sampler = None
    if hasattr(loader, "sampler"):
        sampler = loader.sampler

    return loader, sampler



def _apply_ddp_if_needed(model, train_cfg: TrainConfig):
    device = train_cfg.device

    if not getattr(train_cfg, "is_distributed", False):
        return model

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return model

    if device == "cuda":
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[train_cfg.local_rank],
            output_device=train_cfg.local_rank,
            find_unused_parameters=train_cfg.find_unused_parameters,
            broadcast_buffers=train_cfg.broadcast_buffers,
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=train_cfg.find_unused_parameters,
            broadcast_buffers=train_cfg.broadcast_buffers,
        )

    return model



def _is_main_process(train_cfg: TrainConfig) -> bool:
    return train_cfg.is_main_process


def _count_and_report_params(model):
    raw_model = _unwrap_model(model)

    def count_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    total_all, trainable_all = count_params(raw_model)
    total_dg, trainable_dg = count_params(raw_model.dg_net)
    total_enc, trainable_enc = count_params(raw_model.ae_net.blocks)
    total_dec, trainable_dec = count_params(raw_model.ae_net.decoder_blocks)

    def ratio(x):
        return x / total_all if total_all > 0 else 0.0

    print("\n========== 🔍 Model Parameter Report ==========")
    print(f"[Total Model]        total: {total_all:,} | trainable: {trainable_all:,}")

    print(f"[DG Net]             total: {total_dg:,} ({ratio(total_dg)*100:.2f}%)"
          f" | trainable: {trainable_dg:,}")

    print(f"[AE Encoder]         total: {total_enc:,} ({ratio(total_enc)*100:.2f}%)"
          f" | trainable: {trainable_enc:,}")

    print(f"[AE Decoder]         total: {total_dec:,} ({ratio(total_dec)*100:.2f}%)"
          f" | trainable: {trainable_dec:,}")

    frozen_all = total_all - trainable_all

    print("\n[Trainable vs Frozen]")
    print(f"Trainable: {trainable_all:,} ({trainable_all/total_all*100:.2f}%)")
    print(f"Frozen:    {frozen_all:,} ({frozen_all/total_all*100:.2f}%)")

    print("=============================================\n")

    raw_model._param_reported = True

############################################ train ###########################################

def train_step(
    model,
    batch,
    ae_loss_fn,
    dg_loss_fn,
    opt_ae,
    opt_dg,
    device,
    train_cfg: TrainConfig,
):
    # =========================
    # 🔥 최초 1회 파라미터 리포트
    # =========================
    raw_model = _unwrap_model(model)
    if not hasattr(raw_model, "_param_reported"):
        _count_and_report_params(raw_model)

    x = _move_batch_to_device(batch, device)

    # =========================
    # 1️⃣ AE step
    # =========================
    _set_dg_trainable(raw_model, train_cfg, False)
    _set_ae_trainable(raw_model, train_cfg, True)

    raw_model.ae_net.train()
    raw_model.dg_net.eval()

    opt_ae.zero_grad()

    out = raw_model(x).to_dict()

    ae_loss, ae_detail = ae_loss_fn(
        pred=out["pred"],
        z_x=out["z_x"],
        z_x_dg=out["z_x_dg"].detach(),
        x=out["x"],
    )

    ae_loss.backward()
    if train_cfg.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(_collect_trainable_parameters(raw_model.ae_net), train_cfg.grad_clip_norm)
    opt_ae.step()

    # =========================
    # 2️⃣ Degradation step
    # =========================
    _set_dg_trainable(raw_model, train_cfg, True)
    _set_ae_trainable(raw_model, train_cfg, False)

    raw_model.dg_net.train()
    raw_model.ae_net.eval()

    opt_dg.zero_grad()

    out = raw_model(x).to_dict()

    dg_loss, dg_detail = dg_loss_fn(
        pred=out["pred"],
        z_x=out["z_x"].detach(),
        z_x_dg=out["z_x_dg"],
        x=out["x"],
        x_dg=out["x_dg"],
        z_mask=out["z_mask"]
    )

    dg_loss.backward()
    if train_cfg.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(_collect_trainable_parameters(raw_model.dg_net), train_cfg.grad_clip_norm)
    opt_dg.step()

    return ae_loss.item(), dg_loss.item(), ae_detail, dg_detail

def train(train_cfg: TrainConfig, loss_hook=None):

    train_cfg.detect_device()
    train_cfg.detect_ddp()

    device = torch.device(train_cfg.device)

    # =========================
    # model
    # =========================
    model = _maybe_load_model(train_cfg)
    model.to(device)

    model = _apply_ddp_if_needed(model, train_cfg)

    # 🔥 DDP 안전하게 optimizer 대상 분리
    real_model = model.module if hasattr(model, "module") else model

    # =========================
    # dataloader
    # =========================
    loader, sampler = _build_dataloader(train_cfg)

    # =========================
    # optimizer
    # =========================
    opt_ae, opt_dg = build_optimizers(real_model, train_cfg)

    # =========================
    # sample (visualization용)
    # =========================
    dataset = loader.dataset
    idxs = [0, 1, 2, 3]

    sample_x = torch.stack([dataset[i][0] for i in idxs]).to(device)

    # =========================
    # save dir
    # =========================
    if train_cfg.save_path is not None and _is_main_process(train_cfg):
        _ensure_dir(train_cfg.save_path)

    # =========================
    # train loop
    # =========================
    step = 0

    for epoch in range(train_cfg.epochs):

        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        for batch in loader:

            # =========================
            # schedule
            # =========================
            progress = train_cfg.get_progress(
                epoch=epoch + 1,
                global_step=step
            )

            scheduled_loss_cfg = train_cfg.loss.build_scheduled_loss_config()

            ae_loss_fn, dg_loss_fn = build_dg_learn_loss(scheduled_loss_cfg)

            # =========================
            # train step
            # =========================
            ae_loss, dg_loss, ae_detail, dg_detail = train_step(
                model,
                batch,
                ae_loss_fn,
                dg_loss_fn,
                opt_ae,
                opt_dg,
                device,
                train_cfg,
            )

            # =========================
            # hook
            # =========================
            if loss_hook is not None:
                loss_hook(ae_loss, dg_loss)

            # =========================
            # logging
            # =========================
            if step % train_cfg.log_interval == 0 and _is_main_process(train_cfg):
                print(f"[step {step}] AE: {ae_loss:.4f} | DG: {dg_loss:.4f}")

            # =========================
            # visualization / save
            # =========================
            if train_cfg.save_path is not None and _is_main_process(train_cfg):
                # print(" save to  :", train_cfg.save_path)
                # 첫 step
                if step == 0:
                    save_sample_visualization(model, sample_x, train_cfg.save_path, step)

                # 중간 step
                elif train_cfg.iter_show is not None and step % train_cfg.iter_show == 0:
                    save_sample_visualization(model, sample_x, train_cfg.save_path, step)

                save_interval = train_cfg.save_interval

                if (
                    save_interval is not None
                    and save_interval > 0
                    and step > 0
                    and step % save_interval == 0
                ):
                    _save_checkpoint(model, train_cfg.save_path, name=f"dg_learner_step_{step}")
                    _save_config(train_cfg.save_path, ae_loss_fn, dg_loss_fn, train_cfg)
                    
            elif train_cfg.save_path is None:
                print(" save path error, current save_path : ", train_cfg.save_path)

            step += 1

            # =========================
            # 종료 조건 (step)
            # =========================
            if train_cfg.max_iters is not None and step >= train_cfg.max_iters:

                if _is_main_process(train_cfg):
                    print("Reached max_iters")

                    if train_cfg.save_path is not None:
                        save_sample_visualization(model, sample_x, train_cfg.save_path, step)
                        _save_checkpoint(model, train_cfg.save_path, name="dg_learner")
                        _save_config(train_cfg.save_path, ae_loss_fn, dg_loss_fn, train_cfg)

                return model

    # =========================
    # epoch 종료 저장
    # =========================
    if train_cfg.save_path is not None and _is_main_process(train_cfg):
        save_sample_visualization(model, sample_x, train_cfg.save_path, step)
        _save_checkpoint(model, train_cfg.save_path, name="dg_learner")
        _save_config(train_cfg.save_path, ae_loss_fn, dg_loss_fn, train_cfg)

    return model


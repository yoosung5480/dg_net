'''
### 호출 관계 (상속관계 아님.)

TrainConfig             # 내부 요소들에 대한 동작 정의 및 규칙
    - DataConfig        # 데이터로더에 대한 책임
    - LossConfig        # loss함수 생성에 대한 책임
        - DistanceConfig
        - RegluationConfig
    - DGlearnerConfig       # 모델 불러오기,저장등 모델생성에 대한 책임
        - DegradationConfig
        - MAEConfig
'''
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Tuple, Literal

import copy
import json
import math
import os
from pathlib import Path

try:
    import torch
except Exception:
    torch = None



######################################## dataset config ########################################
@dataclass
class DataConfig:
    # -------------------------
    # 분류를 위해서만 쓰이는 하이퍼 파라미터, DGnet훈련시엔 아예 안쓰이는 필드들임.
    # -------------------------
    task: Literal["dg", "classification"] = "dg"
    is_train: Optional[bool] = None   # split 대신 명확하게
    num_classes: Optional[int] = None  # optional (log용)
    # -------------------------
    # dataset
    # -------------------------
    dataset: str = "imagenet"   # "imagenet", "stl10", "custom"
    data_path: str = "/data/imagenet/train"

    split: str = "train"
    classes: Optional[List[int]] = None
    max_samples: Optional[int] = None

    # -------------------------
    # transform
    # -------------------------
    img_size: int = 224

    use_augmentation: bool = True
    crop_scale: Tuple[float, float] = (0.08, 1.0)
    hflip_prob: float = 0.5

    color_jitter: Optional[Tuple[float, float, float, float]] = None

    gaussian_blur: bool = False
    blur_prob: float = 0.5

    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float]  = (0.229, 0.224, 0.225)

    # -------------------------
    # dataloader
    # -------------------------
    batch_size: int = 64
    num_workers: int = 4

    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True

    # reproducibility
    seed: int = 42

######################################## loss config ########################################
@dataclass
class ScalarSchedulerConfig:
    enabled: bool = False
    mode: Literal["linear", "cosine", "sine"] = "linear"
    start: float = 0.0
    end: float = 0.0

    def compute_schedule_values(self, progress: float) -> float:
        """
        progress: 0.0 ~ 1.0
        """
        if not self.enabled:
            return self.start

        p = max(0.0, min(1.0, progress))

        if self.mode == "linear":
            ratio = p
        elif self.mode == "cosine":
            ratio = 0.5 * (1.0 - math.cos(math.pi * p))
        elif self.mode == "sine":
            ratio = math.sin(math.pi * 0.5 * p)
        else:
            raise ValueError(f"Unknown scheduler mode: {self.mode}")

        return self.start + (self.end - self.start) * ratio


@dataclass
class DistanceScheduleConfig:
    alpha_r: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)
    alpha_p: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)

    beta_r: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)
    beta_p: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)
    beta_m: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)

    mask_threshold: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)
    recon_target: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)


@dataclass
class RegluationScheduleConfig:
    lambda_deg: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)
    lambda_rep: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)

    lambda_tv: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)
    lambda_sparsity: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)

    rep_threshold: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)
    rep_exp_k: ScalarSchedulerConfig = field(default_factory=ScalarSchedulerConfig)


@dataclass
class LossScheduleConfig:
    distance: DistanceScheduleConfig = field(default_factory=DistanceScheduleConfig)
    regluation: RegluationScheduleConfig = field(default_factory=RegluationScheduleConfig)


@dataclass
class DistanceConfig:
    # AE loss weights
    alpha_r: float = 1.0
    alpha_p: float = 1.0

    # Degradation loss weights
    beta_r: float = 1.0
    beta_p: float = 1.0
    beta_m: float = 0.0

    # 🔥 추가: recon mode 선택
    ae_recon_mode: str = "D_p"        # "D_p" | "D_p_hat"
    dg_recon_mode: str = "D_p_hat"    # "D_p" | "D_p_hat"

    # Representation distance
    eps: float = 1e-8

    # Reconstruction distance
    patch_size: int = 16
    in_chans: int = 3
    use_norm_pix_loss: bool = False

    # Mask / degradation distance
    mask_threshold: float = 0.25
    recon_target: float = 0.1   # D^_p target


@dataclass
class RegluationConfig:
    # outer weights for degradation / representation regulation
    lambda_deg: float = 0.0
    lambda_rep: float = 0.0

    # inner weights for degradation regulation
    lambda_tv: float = 0.0
    lambda_sparsity: float = 0.0

    # representation regulation parameters
    rep_threshold: float = 0.3
    rep_exp_k: float = 10.0

    # numerical stability
    eps: float = 1e-8


@dataclass
class LossConfig:
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    regluation: RegluationConfig = field(default_factory=RegluationConfig)

    schedule: LossScheduleConfig = field(default_factory=LossScheduleConfig)

    reduction: str = "mean"
    return_details: bool = True

    _progress: float = field(default=0.0, init=False, repr=False)

    def set_progress(self, progress: float):
        self._progress = max(0.0, min(1.0, progress))

    def build_scheduled_loss_config(self) -> "LossConfig":
        cfg = copy.deepcopy(self)
        progress = getattr(self, "_progress", 0.0)

        # distance
        ds = cfg.schedule.distance
        dc = cfg.distance

        if ds.alpha_r.enabled:
            dc.alpha_r = ds.alpha_r.compute_schedule_values(progress)
        if ds.alpha_p.enabled:
            dc.alpha_p = ds.alpha_p.compute_schedule_values(progress)
        if ds.beta_r.enabled:
            dc.beta_r = ds.beta_r.compute_schedule_values(progress)
        if ds.beta_p.enabled:
            dc.beta_p = ds.beta_p.compute_schedule_values(progress)
        if ds.beta_m.enabled:
            dc.beta_m = ds.beta_m.compute_schedule_values(progress)
        if ds.mask_threshold.enabled:
            dc.mask_threshold = ds.mask_threshold.compute_schedule_values(progress)
        if ds.recon_target.enabled:
            dc.recon_target = ds.recon_target.compute_schedule_values(progress)

        # regulation
        rs = cfg.schedule.regluation
        rc = cfg.regluation

        if rs.lambda_deg.enabled:
            rc.lambda_deg = rs.lambda_deg.compute_schedule_values(progress)
        if rs.lambda_rep.enabled:
            rc.lambda_rep = rs.lambda_rep.compute_schedule_values(progress)
        if rs.lambda_tv.enabled:
            rc.lambda_tv = rs.lambda_tv.compute_schedule_values(progress)
        if rs.lambda_sparsity.enabled:
            rc.lambda_sparsity = rs.lambda_sparsity.compute_schedule_values(progress)
        if rs.rep_threshold.enabled:
            rc.rep_threshold = rs.rep_threshold.compute_schedule_values(progress)
        if rs.rep_exp_k.enabled:
            rc.rep_exp_k = rs.rep_exp_k.compute_schedule_values(progress)

        cfg._progress = progress
        return cfg


######################################## dataset config ########################################
@dataclass(frozen=True)
class BaseConfig:
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "model_config.json", "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str):
        path = Path(path)
        with open(path / "model_config.json", "r") as f:
            cfg_dict = json.load(f)
        return cls(**cfg_dict)

####### 
@dataclass(frozen=True)
class DegradationConfig(BaseConfig):
    # ---- input spec ----
    IMG_SIZE: int = 224
    PATCH_SIZE: int = 16
    IN_CHANS: int = 3

    # ---- architecture ----
    EMBED_DIM: int = 512
    DEPTH: int = 6
    NUM_HEADS: int = 8
    MLP_RATIO: float = 4.0

    # ---- positional embedding ----
    USE_POS_EMBED: bool = True
    POS_EMBED_INIT_ZERO: bool = True

    # ---- output control ----
    CLAMP_OUTPUT: bool = False

    # ---- optional advanced (확장 대비) ----
    ATTN_DROPOUT: float = 0.0
    PROJ_DROPOUT: float = 0.0
    CNN_KERNEL_SIZE: int = 3
    CNN_DROPOUT: float = 0.0
    FFN_DROPOUT: float = 0.0


@dataclass(frozen=True)
class MAEConfig(BaseConfig):
    # -------------------------------------------------
    # backbone (encoder)
    # -------------------------------------------------
    IMG_SIZE: int = 224
    PATCH_SIZE: int = 16
    IN_CHANS: int = 3

    EMBED_DIM: int = 1024
    DEPTH: int = 24
    NUM_HEADS: int = 16

    MLP_RATIO: float = 4.0

    # -------------------------------------------------
    # decoder
    # -------------------------------------------------
    DECODER_EMBED_DIM: int = 512
    DECODER_DEPTH: int = 8
    DECODER_NUM_HEADS: int = 16

    # -------------------------------------------------
    # loss / training
    # -------------------------------------------------
    NORM_PIX_LOSS: bool = False

    # -------------------------------------------------
    # optional (확장용)
    # -------------------------------------------------
    NAME: str = "mae_vit_large_patch16"


@dataclass(frozen=True)
class DGlearnerConfig(BaseConfig):
    degradation_config: DegradationConfig
    ae_config: MAEConfig

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "model_config.json", "w") as f:
            json.dump({
                "degradation_config": asdict(self.degradation_config),
                "ae_config": asdict(self.ae_config),
            }, f, indent=4)

    @classmethod
    def load(cls, path: str):
        path = Path(path)
        with open(path / "model_config.json", "r") as f:
            cfg_dict = json.load(f)

        return cls(
            degradation_config=DegradationConfig(**cfg_dict["degradation_config"]),
            ae_config=MAEConfig(**cfg_dict["ae_config"]),
        )
    




######################################## train config ########################################
'''         
model.dg_net                    # degradatoin 블럭 호출 api.
model.ae_net.blocks             # encoder 블럭 호출 api. 
model.ae_net.decoder_blocks     # decoder 블럭 호출 api


# TODO
- 컨셉: 훈련루프 생성을 위해선 util패키지와, "TrainConfig만"으로 훈련 코드 작성이 돼야함

### DDP dataloader설정
1. config 패키지의 __init__에서 이거 호출시, 자동으로 DDP 관련 세팅 세주기.
2. 그 외 사람이 지정할 ddp관련 파라미터 TrainConfig에 세팅

---
### ParamGradConfig
일단 이대로 유지 

---
### (중요) schedular 설정
loss실험 결과에 대한 마크다운 실험결과를 토대로, 실험이 진행됨에 따라서
1. step의 횟수가 훈련 종료 codition일때, steps의 횟수에 따라 증가 or 감소 or 주기함수
2. epoch의 횟수가 훈련 종료 codition일때, epoch의 횟수에 따라 증가 or 감소 or 주기함수


---
### TrainConfig 아래 포함관계 (계층관계가 아니라, 호출관계)
-> 즉, TrainConfig만 잇어도 훈련세팅이 다 돼야함. 

추후 TrainConfig를 호출하고, exp_config.py에 TrainConfig를 왕창생성해서, main_train.py에서는 exp_config 파일만 가지고도 훈련이 진행되게한다.

TrainConfig             # 내부 요소들에 대한 동작 정의 및 규칙
    - DataConfig        # 데이터로더에 대한 책임
    - LossConfig        # loss함수 생성에 대한 책임
        - DistanceConfig
        - RegluationConfig
    - DGlearnerConfig       # 모델 불러오기,저장등 모델생성에 대한 책임
        - DegradationConfig
        - MAEConfig
'''

# -----------------------------
# External dependency configs
# -----------------------------

@dataclass
class ParamGradConfig:
    train_encoder: bool = True
    train_decoder: bool = True
    train_degradation: bool = True


@dataclass
class OptimizerConfig:
    ae_optimizer: str = "adamw"
    dg_optimizer: str = "adamw"

    lr_ae: float = 1e-4
    lr_dg: float = 1e-4

    weight_decay_ae: float = 0.05
    weight_decay_dg: float = 0.05

    betas_ae: tuple[float, float] = (0.9, 0.999)
    betas_dg: tuple[float, float] = (0.9, 0.999)


# -----------------------------
# TrainConfig (FLAT ONLY)
# -----------------------------
@dataclass
class ClassificationConfig:
    num_classes: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_split: Optional[str] = None
    eval_data_path: Optional[str] = None
    save_name: str = "classifier"


@dataclass
class TrainConfig:
    # ===== legacy task type =====
    task: Literal["dg", "classification"] = "dg"
    clsc: ClassificationConfig = field(default_factory=ClassificationConfig)

    # ===== checkpoint =====
    checkpoint_path: Optional[str] = None
    checkpoint_name: str = "dg_learner"

    # ===== train control =====
    mode: Literal["epoch", "step", "classification"] = "step"
    epochs: int = 1
    max_iters: Optional[int] = 10000
    val_interval: Optional[int] = None

    # ===== runtime =====
    device: str = "auto"
    amp: bool = True
    grad_clip_norm: Optional[float] = 1.0

    seed: int = 42
    deterministic: bool = False

    log_interval: int = 10
    iter_show: Optional[int] = None
    save_path: Optional[str] = None
    save_interval: int = 1000

    # ===== ddp =====
    use_ddp: bool = False
    backend: str = "nccl"
    find_unused_parameters: bool = False
    broadcast_buffers: bool = False

    # runtime-detected
    is_distributed: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_main_process: bool = True

    # ===== optimizer shortcut (optional but useful) =====
    # lr_ae: float = 1e-4   # 제거
    # lr_dg: float = 1e-4   # 제거

    # -----------------------------
    # External modules (KEEP INNER)
    # -----------------------------
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    model: DGlearnerConfig = field(
        default_factory=lambda: DGlearnerConfig(
            degradation_config=DegradationConfig(),
            ae_config=MAEConfig(),
        )
    )

    param_grad: ParamGradConfig = field(default_factory=ParamGradConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # -----------------------------
    # utils
    # -----------------------------
    def detect_device(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        return self.device

    def detect_ddp(self):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        self.is_distributed = self.use_ddp and self.world_size > 1
        self.is_main_process = self.rank == 0

        if self.device != "cuda" and self.backend == "nccl":
            self.backend = "gloo"

        return self.is_distributed

    def get_progress(self, epoch: int, global_step: int) -> float:
        if self.mode == "step":
            if not self.max_iters:
                raise ValueError("step mode requires max_iters")
            progress = min(global_step / self.max_iters, 1.0)

        elif self.mode == "epoch":
            if self.epochs <= 0:
                raise ValueError("epoch mode requires epochs")
            progress = min(epoch / self.epochs, 1.0)

        elif self.mode == "classification":
            if self.max_iters is not None:
                progress = min(global_step / max(self.max_iters, 1), 1.0)
            elif self.epochs > 0:
                progress = min(epoch / self.epochs, 1.0)
            else:
                progress = 0.0

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.loss.set_progress(progress)
        return progress

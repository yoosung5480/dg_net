from src.models.degradation.model_masking import DegradationNet, DegradationConfig, sample_show
import torch
from src.models.auto_encoder.model_auto_encoder import MaskedAutoencoderViT, MAEConfig

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import json
from typing import Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict 
from contextlib import contextmanager

from src.config.DG_config import DGlearnerConfig
# from src.models.base_config import BaseConfig
from src.models.base_model import BaseModel


# -----------------------------
# Config
# -----------------------------
# @dataclass(frozen=True)
# class DGlearnerConfig(BaseConfig):
#     degradation_config: DegradationConfig
#     ae_config: MAEConfig

#     def save(self, path: str):
#         path = Path(path)
#         path.mkdir(parents=True, exist_ok=True)

#         with open(path / "model_config.json", "w") as f:
#             json.dump({
#                 "degradation_config": asdict(self.degradation_config),
#                 "ae_config": asdict(self.ae_config),
#             }, f, indent=4)

#     @classmethod
#     def load(cls, path: str):
#         path = Path(path)
#         with open(path / "model_config.json", "r") as f:
#             cfg_dict = json.load(f)

#         return cls(
#             degradation_config=DegradationConfig(**cfg_dict["degradation_config"]),
#             ae_config=MAEConfig(**cfg_dict["ae_config"]),
#         )

# -----------------------------
# Output
# -----------------------------
@dataclass
class DGOutput:
    pred_dg_x: torch.Tensor     # degradated x에 대한 복원
    z_x: torch.Tensor           # 원본x에 대한 subspace vector representation
    z_x_dg: torch.Tensor        # degradated x에 대한 subspace vector representation
    x_dg: torch.Tensor          # x에 대한 복원 
    x: torch.Tensor             # 원본 이미지 x
    z_mask : torch.Tensor       # degradated 이미지에 대한 latent vector

    def to_dict(self):
        return {
            "pred": self.pred_dg_x,
            "z_x": self.z_x,
            "z_x_dg": self.z_x_dg,
            "x_dg": self.x_dg,
            "x": self.x,
            "z_mask" : self.z_mask
        }

class DegradationLearnerNet(BaseModel):
    ''' 
    ### ae_config 예시
    ~~~python
    ae_cfg = MAEConfig(
        EMBED_DIM=768,
        DEPTH=12,
        NUM_HEADS=12,
    )
    ~~~


    ### degradation_config 예시
    ~~~pygthon
    degradation_cfg = DegradationConfig(
        IMG_SIZE=224,
        EMBED_DIM=384,
        DEPTH=4,
    )
    ~~~

    ## build 예시
    ~~~python
    
    ~~~
    '''
    ConfigClass = DGlearnerConfig
    def __init__(self, cfg: DGlearnerConfig):
        super().__init__(cfg)
        self.dg_net = DegradationNet(cfg=cfg.degradation_config)
        self.ae_net = MaskedAutoencoderViT(cfg=cfg.ae_config)

    @contextmanager
    def _freeze_ae_encoder(self):
        params = []

        # encoder 구성요소만 가져오기
        for p in self.ae_net.patch_embed.parameters():
            params.append(p)
        for p in self.ae_net.blocks.parameters():
            params.append(p)
        for p in self.ae_net.norm.parameters():
            params.append(p)

        params.append(self.ae_net.cls_token)

        # 기존 상태 저장
        old_flags = [p.requires_grad for p in params]

        # 끄기
        for p in params:
            p.requires_grad = False

        try:
            yield
        finally:
            # 복구
            for p, flag in zip(params, old_flags):
                p.requires_grad = flag

    def forward(self, x):
        '''
        ### 입력
        - x : 원본이미지 텐서 형태로 변환 형태 [B,3,H,W]

        ### 출력
        변경점. dg_net는 이제 이미지 전체 degradation하지 않고 노이즈를 만든다.
        - x         : 원본 입력 이미지 텐서
        - x_dg      : 손상시킨 x, x - x_dg
        - z_x       : 손상시킨 x에 대한 latent vector
        - z_x_dg    : 원본 입력 이미지에 대한 latent vector, encoder(x-dg_net(x))
        - pred_dg_x : 복원된 손상된 x
        - z_mask    : encoder(dg_net(x))
        '''
        x_dg = self.dg_net(x)
        m = x - x_dg

        # reconstruction
        pred_dg_x, _ = self.ae_net(m, mask_ratio=0.0)

        # 기준 latent (완전 고정)
        with torch.no_grad():
            latent_x = self.ae_net.forward_encoder(x, 0.0)[0]

        # DG용 latent (encoder weight만 freeze)
        with self._freeze_ae_encoder():
            latent_x_dg = self.ae_net.forward_encoder(m, 0.0)[0]
            latent_z_mask = self.ae_net.forward_encoder(x_dg, 0.0)[0]

        z_x = latent_x.mean(dim=1)
        z_x_dg = latent_x_dg.mean(dim=1)
        z_mask = latent_z_mask.mean(dim=1)

        return DGOutput(
            pred_dg_x=pred_dg_x,
            z_x=z_x,
            z_x_dg=z_x_dg,
            x_dg=m,
            x=x,
            z_mask=z_mask
        )
    # def forward(self, x):
        
    #     x_dg = self.dg_net(x)
    #     m = x - x_dg
    #     # reconstruction
    #     pred_dg_x, _ = self.ae_net(m, mask_ratio=0.0)

    #     # latent
    #     with torch.no_grad():
    #         latent_x = self.ae_net.forward_encoder(x, 0.0)[0]
    #     latent_x_dg = self.ae_net.forward_encoder(m, 0.0)[0]
    #     latent_z_mask = self.ae_net.forward_encoder(x_dg, 0.0)[0]

    #     z_x = latent_x.mean(dim=1)
    #     z_x_dg = latent_x_dg.mean(dim=1)
    #     z_mask = latent_z_mask.mean(dim=1)

    #     return DGOutput(
    #         pred_dg_x=pred_dg_x,
    #         z_x=z_x,
    #         z_x_dg=z_x_dg,
    #         x_dg=m,
    #         x=x,
    #         z_mask=z_mask                     # 추가
    #     )
    

    def sample_show(self, x, n=4):
        self.eval()
        with torch.no_grad():
            out: DGOutput = self.forward(x)

            pred = out.pred_dg_x
            x_dg = out.x_dg
            x = out.x

            recon = self.ae_net.unpatchify(pred)

            n = min(n, x.shape[0])

            plt.figure(figsize=(4*n, 6))

            for i in range(n):
                def to_img(t):
                    t = t.detach().cpu()
                    t = t.permute(1, 2, 0).numpy()
                    return np.clip(t, 0, 1)

                x_img = to_img(x[i])
                dg_img = to_img(x_dg[i])
                recon_img = to_img(recon[i])

                # row 1: original
                plt.subplot(3, n, i+1)
                plt.imshow(x_img)
                plt.title("Original")
                plt.axis("off")

                # row 2: degraded
                plt.subplot(3, n, n+i+1)
                plt.imshow(dg_img)
                plt.title("Degraded")
                plt.axis("off")

                # row 3: recon
                plt.subplot(3, n, 2*n+i+1)
                plt.imshow(recon_img)
                plt.title("Recon")
                plt.axis("off")

            plt.tight_layout()
            plt.show()

'''
### 해야할일 ###
1. 클래쓰 메서드
2. 함수 - sample
'''

    
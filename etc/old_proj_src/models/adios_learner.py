# models.adios_learner.py
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
from src.models.base_model import BaseModel

@dataclass
class AdiosOutput:
    z_x: torch.Tensor           # E(x)
    z_x_dg: torch.Tensor        # E(x - M(x))
    dg_raw: torch.Tensor        # M(x)
    x: torch.Tensor             # x

    def to_dict(self):
        {
            "z_x": self.z_x,
            "z_x_dg": self.z_x_dg,
            "dg_raw": self.dg_raw,
            "x": self.x,
        }

class AdiosLearnerNet(BaseModel):
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
    def _freeze_dg(self):
        params = []
        for p in self.dg_net.parameters():
            params.append(p)
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
        """
        ADIOS forward
        """

        # -------------------------
        # mask 생성
        # -------------------------
        dg_raw = self.dg_net(x)
        x_m = x  - dg_raw           # 이게 다름. 내 프레임워크의 설계는 이럼.
        # 헷갈릴점. 

        # -------------------------
        # latent
        # -------------------------
        # DG의 그래디언트는 끄고, 인코더 학습용
        with self._freeze_dg(): 
            latent_x = self.ae_net.forward_encoder(x, 0.0)[0]

        # 인코더의 그래디언트는 끄고, DG 학습용
        with self._freeze_ae_encoder():
            latent_x_m = self.ae_net.forward_encoder(x_m, 0.0)[0]

        z_x = latent_x.mean(dim=1)
        z_x_dg = latent_x_m.mean(dim=1)

        return AdiosOutput(
            z_x = z_x,
            z_x_dg = z_x_dg,
            dg_raw = dg_raw,
            x = x
        )

    def sample_show(self, x, n=4):
        self.eval()
        with torch.no_grad():
            out: AdiosOutput = self.forward(x)
            x_dg = out.x_dg
            x = out.x


            n = min(n, x.shape[0])

            plt.figure(figsize=(4*n, 6))

            for i in range(n):
                def to_img(t):
                    t = t.detach().cpu()
                    t = t.permute(1, 2, 0).numpy()
                    return np.clip(t, 0, 1)

                x_img = to_img(x[i])
                dg_img = to_img(x_dg[i])

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


            plt.tight_layout()
            plt.show()

'''
### 해야할일 ###
1. 클래쓰 메서드
2. 함수 - sample
'''

    
from src.models.degradation.model_degradation import DegradationNet, DegradationConfig, sample_show
import torch
from src.models.auto_encoder.model_auto_encoder import MaskedAutoencoderViT, MAEConfig

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn


from typing import Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict 

from src.config.DG_config import DGlearnerConfig
# from src.models.base_config import BaseConfig
from src.models.base_model import BaseModel
import json


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

    def to_dict(self):
        return {
            "pred": self.pred_dg_x,
            "z_x": self.z_x,
            "z_x_dg": self.z_x_dg,
            "x_dg": self.x_dg,
            "x": self.x,
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

    def forward(self, x):
        '''
        ### 입력
        - x : 원본이미지 텐서 형태로 변환 형태 [B,3,H,W]

        ### 출력
        - x         : 원본 입력 이미지 텐서
        - x_dg      : 손상시킨 x
        - z_x       : 손상시킨 x에 대한 latent vector
        - z_x_dg    : 원본 입력 이미지에 대한 latent vector
        - pred_dg_x : 복원된 손상된 x
        '''
        x_dg = self.dg_net(x)

        # reconstruction
        pred_dg_x, mask = self.ae_net(x_dg, mask_ratio=0.0)

        # latent
        latent_x = self.ae_net.forward_encoder(x, 0.0)[0]
        latent_x_dg = self.ae_net.forward_encoder(x_dg, 0.0)[0]

        z_x = latent_x.mean(dim=1)
        z_x_dg = latent_x_dg.mean(dim=1)

        return DGOutput(
            pred_dg_x=pred_dg_x,
            z_x=z_x,
            z_x_dg=z_x_dg,
            x_dg=x_dg,
            x=x
        )
    

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

    
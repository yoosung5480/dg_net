from typing import Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# from src.models.base_config import BaseConfig
from src.config.DG_config import BaseConfig, DegradationConfig
from src.models.base_model import BaseModel


# @dataclass(frozen=True)
# class DegradationConfig(BaseConfig):
#     # ---- input spec ----
#     IMG_SIZE: int = 224
#     PATCH_SIZE: int = 16
#     IN_CHANS: int = 3

#     # ---- architecture ----
#     EMBED_DIM: int = 512
#     DEPTH: int = 6
#     NUM_HEADS: int = 8
#     MLP_RATIO: float = 4.0

#     # ---- positional embedding ----
#     USE_POS_EMBED: bool = True
#     POS_EMBED_INIT_ZERO: bool = True

#     # ---- output control ----
#     CLAMP_OUTPUT: bool = False

#     # ---- optional advanced (확장 대비) ----
#     ATTN_DROPOUT: float = 0.0
#     PROJ_DROPOUT: float = 0.0
#     CNN_KERNEL_SIZE: int = 3
#     CNN_DROPOUT: float = 0.0
#     FFN_DROPOUT: float = 0.0


# =========================================================
# 1. FeedForward
# =========================================================
class FeedForward(nn.Module):
    """
    Transformer FFN
    [B, N, D] -> [B, N, hidden] -> [B, N, D]
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)      # [B, N, hidden_dim]
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)      # [B, N, dim]
        x = self.drop2(x)
        return x


# =========================================================
# 2. CNN Branch
# =========================================================
class TokenCNNBranch(nn.Module):
    """
    Token sequence [B, N, C]
    -> reshape -> [B, C, H, W]
    -> depthwise conv
    -> pointwise conv
    -> GELU
    -> dropout
    -> reshape back -> [B, N, C]
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
            bias=True,
        )
        self.pointwise = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> torch.Tensor:
        b, n, c = x.shape
        h, w = grid_size

        if n != h * w:
            raise ValueError(
                f"Token count mismatch: n={n}, but h*w={h*w}. "
                f"Received grid_size={grid_size}."
            )

        # [B, N, C] -> [B, C, H, W]
        x = x.transpose(1, 2).reshape(b, c, h, w)

        x = self.depthwise(x)   # [B, C, H, W]
        x = self.pointwise(x)   # [B, C, H, W]
        x = self.act(x)
        x = self.dropout(x)

        # [B, C, H, W] -> [B, N, C]
        x = x.flatten(2).transpose(1, 2)
        return x


# =========================================================
# 3. Hybrid Transformer Layer
# =========================================================
class HybridTransformerLayer(nn.Module):
    """
    순서:
    1. LayerNorm
    2. MHSA
    3. Residual
    4. LayerNorm
    5. CNN Block
    6. Residual
    7. LayerNorm
    8. FFN
    9. Residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_proj_drop = nn.Dropout(proj_dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.cnn_branch = TokenCNNBranch(
            dim=dim,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=ffn_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Tuple[int, int],
    ) -> torch.Tensor:
        # 1~3. LN -> MHSA -> Residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.attn_proj_drop(attn_out)

        # 4~6. LN -> CNN -> Residual
        x_norm = self.norm2(x)
        cnn_out = self.cnn_branch(x_norm, grid_size=grid_size)
        x = x + cnn_out

        # 7~9. LN -> FFN -> Residual
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


# =========================================================
# 4. Patch Embedding
# =========================================================
class PatchEmbed(nn.Module):
    """
    Conv2d(kernel=patch, stride=patch)로
    patchify + linear projection 동시 수행
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 512,
    ) -> None:
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})."
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_dim = patch_size * patch_size * in_chans

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(
                f"Input image size must be ({self.img_size}, {self.img_size}), "
                f"but got ({h}, {w})."
            )

        x = self.proj(x)                  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


# =========================================================
# 5. Residual Patch Head
# =========================================================
class ResidualPatchHead(nn.Module):
    """
    latent tokens [B, N, D] -> patch residual [B, N, patch_dim]
    마지막 linear를 zero-init 해서 초기엔 delta=0
    """

    def __init__(
        self,
        embed_dim: int,
        patch_dim: int,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(embed_dim, patch_dim),
        )

        last_linear = self.head[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# =========================================================
# 6. DegradationNet
# =========================================================
class DegradationNet(BaseModel):
    ConfigClass = DegradationConfig
    """
    ### 전체 구조
    image
    -> patchify + linear proj
    -> hybrid blocks x L
    -> residual head
    -> out_patch = input_patch + delta
    -> unpatchify

    ### 입출력
    input : tensor [B, C, H, W]
    return : tensor [B, C, H, W]
    """

    def __init__(self, cfg: DegradationConfig) -> None:
        super().__init__(cfg)

        self.cfg = cfg

        # =============================
        # basic
        # =============================
        self.img_size = cfg.IMG_SIZE
        self.patch_size = cfg.PATCH_SIZE
        self.in_chans = cfg.IN_CHANS
        self.embed_dim = cfg.EMBED_DIM
        self.depth = cfg.DEPTH
        self.clamp_output = cfg.CLAMP_OUTPUT

        # =============================
        # Patch Embedding
        # =============================
        self.patch_embed = PatchEmbed(
            img_size=cfg.IMG_SIZE,
            patch_size=cfg.PATCH_SIZE,
            in_chans=cfg.IN_CHANS,
            embed_dim=cfg.EMBED_DIM,
        )

        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = self.patch_embed.patch_dim

        # =============================
        # Positional Embedding
        # =============================
        self.use_pos_embed = cfg.USE_POS_EMBED
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, cfg.EMBED_DIM)
            )
            if not cfg.POS_EMBED_INIT_ZERO:
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        # =============================
        # Blocks
        # =============================
        self.blocks = nn.ModuleList([
            HybridTransformerLayer(
                dim=cfg.EMBED_DIM,
                num_heads=cfg.NUM_HEADS,
                mlp_ratio=cfg.MLP_RATIO,
                attn_dropout=cfg.ATTN_DROPOUT,
                proj_dropout=cfg.PROJ_DROPOUT,
                cnn_kernel_size=cfg.CNN_KERNEL_SIZE,
                cnn_dropout=cfg.CNN_DROPOUT,
                ffn_dropout=cfg.FFN_DROPOUT,
            )
            for _ in range(cfg.DEPTH)
        ])

        # =============================
        # Residual Head
        # =============================
        self.residual_head = ResidualPatchHead(
            embed_dim=cfg.EMBED_DIM,
            patch_dim=self.patch_dim,
        )

    def patchify_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """
        image -> raw patch pixels
        x: [B, C, H, W]
        return: [B, N, patch_dim]
        """
        b, c, h, w = x.shape
        p = self.patch_size

        if h != self.img_size or w != self.img_size:
            raise ValueError(
                f"Input image size must be ({self.img_size}, {self.img_size}), "
                f"but got ({h}, {w})."
            )

        gh, gw = h // p, w // p
        x = x.reshape(b, c, gh, p, gw, p)
        x = x.permute(0, 2, 4, 3, 5, 1)   # [B, gh, gw, p, p, C]
        x = x.reshape(b, gh * gw, p * p * c)
        return x

    def unpatchify_pixels(self, patches: torch.Tensor) -> torch.Tensor:
        """
        raw patch pixels -> image
        patches: [B, N, patch_dim]
        return: [B, C, H, W]
        """
        b, n, d = patches.shape
        p = self.patch_size
        c = self.in_chans
        gh, gw = self.grid_size

        if n != gh * gw:
            raise ValueError(f"Expected {gh * gw} patches, got {n}.")
        if d != p * p * c:
            raise ValueError(f"Expected patch_dim={p * p * c}, got {d}.")

        x = patches.reshape(b, gh, gw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)   # [B, C, gh, p, gw, p]
        x = x.reshape(b, c, gh * p, gw * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_delta: bool = False,
        return_patches: bool = False,
    ):
        """
        x: [B, C, H, W]

        return:
            degraded_img
            optionally delta_patches, out_patches
        """
        # 원본 patch

        # latent tokens
        tokens = self.patch_embed(x)             # [B, N, D]

        if self.use_pos_embed and self.pos_embed is not None:
            tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens, grid_size=self.grid_size)

        # residual patch
        delta_patches = self.residual_head(tokens)  # [B, N, patch_dim]

        # identical init: 초기에 delta = 0              ##### 변경점 : input_patches를 더이상 더하지 않음. 즉, zero init이지 이젠 identical init은 아니다.
        out_patches = delta_patches
        degraded_img = self.unpatchify_pixels(out_patches)

        if self.clamp_output:
            degraded_img = torch.clamp(degraded_img, 0.0, 1.0)

        outputs = [degraded_img]

        if return_delta:
            outputs.append(delta_patches)

        if return_patches:
            outputs.append(out_patches)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)



def sample_show(model, test_sample_path, device):
    '''
    util은 개판이지만 뭐 일단 해치움
    '''
    # -----------------------------------------------------
    # 이미지 로드
    # -----------------------------------------------------
    pil_img, x = load_image_as_tensor(
        image_path=test_sample_path,
        img_size=224,
    )
    if device=="cuda":
        x = x.to(device)

    # -----------------------------------------------------
    # 네트워크 통과
    # -----------------------------------------------------
    with torch.no_grad():
        y, delta = model(x, return_delta=True)

    # -----------------------------------------------------
    # numpy 변환
    # -----------------------------------------------------
    input_img = tensor_to_numpy_image(x)    # [H,W,C]
    output_img = tensor_to_numpy_image(y)   # [H,W,C]

    # 초기 상태에서는 거의 동일해야 정상
    diff_img = np.abs(output_img - input_img)
    diff_vis = diff_img / (diff_img.max() + 1e-8)

    output_img = np.clip(output_img, 0.0, 1.0)

    # -----------------------------------------------------
    # 출력
    # -----------------------------------------------------
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(input_img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("DegradationNet Output (Initial)")
    plt.imshow(output_img)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Absolute Difference")
    plt.imshow(diff_vis)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # 수치 확인
    # -----------------------------------------------------
    mean_abs_diff = (y - x).abs().mean().item()

    print("========================================")
    print("Input shape      :", x.shape)        # [1,3,224,224]
    print("Output shape     :", y.shape)        # [1,3,224,224]
    print("Delta shape      :", delta.shape)    # [1,196,768]
    print("Num parameters   :", sum(p.numel() for p in model.parameters()))
    print("Mean |y - x|     :", mean_abs_diff)
    print("Delta min/max    :", delta.min().item(), delta.max().item())
    print("========================================")


# =========================================================
# 7. Utility: image load / tensor convert
# =========================================================
def load_image_as_tensor(
    image_path: str,
    img_size: int = 224,
) -> Tuple[Image.Image, torch.Tensor]:
    """
    image_path -> PIL image, tensor [1,3,H,W]
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))

    img_np = np.array(img).astype(np.float32) / 255.0  # [H,W,C]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]

    return img, img_tensor


def tensor_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """
    [1,C,H,W] or [C,H,W] -> [H,W,C]
    """
    if x.dim() == 4:
        x = x.squeeze(0)
    x = x.detach().cpu().permute(1, 2, 0).numpy()
    return x


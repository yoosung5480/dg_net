"""Config-driven DGNet model assembly.

DGNet contains:

* ``M``: a selectable ViT/CNN/Hybrid degradation generator producing an
  image-shaped residual with a zero-initialized final layer.
* ``I``: a fixed ViT encoder-decoder reconstructing the original image from
  ``x_degraded = x - M(x)``.

The three selectable blocks are the only local fan-in dependencies.  Running
this module creates construction, save/load, and sample-inference artifacts in
``output/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw
import torch
from torch import nn

try:  # package import
    from .cnn import TokenCNNBlock
    from .hybrid import HybridBlock
    from .vit import ViTBlock
except ImportError:  # direct ``python dg_model.py`` invocation
    from cnn import TokenCNNBlock
    from hybrid import HybridBlock
    from vit import ViTBlock


@dataclass(frozen=True)
class DgNetConfig:
    """Single configuration surface for architecture construction and output."""

    IMG_SIZE: int = 224
    PATCH_SIZE: int = 16
    IN_CHANS: int = 3

    # Shared defaults; component overrides permit parameter-size experiments.
    EMBED_DIM: int = 512
    DEPTH: int = 6
    NUM_HEADS: int = 8
    MLP_RATIO: float = 4.0

    ENCODER_ARCHITECT: str = "VIT"
    DECODER_ARCHITECT: str = "VIT"
    DG_ARCHITECT: str = "HYBRID"  # "VIT", "CNN", or "HYBRID"
    CNN_ARCHITECT: str = "CONVNEXT"  # degradation CNN variant: "RESNET"/"CONVNEXT"

    ENCODER_EMBED_DIM: int | None = None
    ENCODER_DEPTH: int | None = None
    ENCODER_NUM_HEADS: int | None = None
    DECODER_EMBED_DIM: int | None = None
    DECODER_DEPTH: int | None = None
    DECODER_NUM_HEADS: int | None = None
    DG_EMBED_DIM: int | None = None
    DG_DEPTH: int | None = None
    DG_NUM_HEADS: int | None = None
    PROJECTION_DIM: int = 128

    USE_POS_EMBED: bool = True
    POS_EMBED_INIT_ZERO: bool = True
    CLAMP_DEGRADED: bool = False

    ATTN_DROPOUT: float = 0.0
    PROJ_DROPOUT: float = 0.0
    DROP_PATH: float = 0.0
    CNN_KERNEL_SIZE: int = 3
    CNN_DROPOUT: float = 0.0
    FFN_DROPOUT: float = 0.0

    def encoder_dim(self) -> int:
        return self.ENCODER_EMBED_DIM or self.EMBED_DIM

    def decoder_dim(self) -> int:
        return self.DECODER_EMBED_DIM or self.EMBED_DIM

    def dg_dim(self) -> int:
        return self.DG_EMBED_DIM or self.EMBED_DIM

    def encoder_depth(self) -> int:
        return self.ENCODER_DEPTH or self.DEPTH

    def decoder_depth(self) -> int:
        return self.DECODER_DEPTH or self.DEPTH

    def dg_depth(self) -> int:
        return self.DG_DEPTH or self.DEPTH

    def encoder_heads(self) -> int:
        return self.ENCODER_NUM_HEADS or self.NUM_HEADS

    def decoder_heads(self) -> int:
        return self.DECODER_NUM_HEADS or self.NUM_HEADS

    def dg_heads(self) -> int:
        return self.DG_NUM_HEADS or self.NUM_HEADS

    def validate(self) -> None:
        if self.IMG_SIZE <= 0 or self.PATCH_SIZE <= 0 or self.IMG_SIZE % self.PATCH_SIZE:
            raise ValueError("IMG_SIZE must be positive and divisible by PATCH_SIZE.")
        if self.IN_CHANS <= 0 or self.PROJECTION_DIM <= 0:
            raise ValueError("IN_CHANS and PROJECTION_DIM must be positive.")
        if self.ENCODER_ARCHITECT.upper() != "VIT" or self.DECODER_ARCHITECT.upper() != "VIT":
            raise ValueError("DGNet keeps ENCODER_ARCHITECT and DECODER_ARCHITECT fixed to VIT.")
        if self.DG_ARCHITECT.upper() not in {"VIT", "CNN", "HYBRID"}:
            raise ValueError("DG_ARCHITECT must be VIT, CNN, or HYBRID.")
        components = [
            ("encoder", self.encoder_dim(), self.encoder_depth(), self.encoder_heads()),
            ("decoder", self.decoder_dim(), self.decoder_depth(), self.decoder_heads()),
        ]
        if self.DG_ARCHITECT.upper() != "CNN":
            components.append(("degradation", self.dg_dim(), self.dg_depth(), self.dg_heads()))
        elif self.dg_dim() <= 0 or self.dg_depth() <= 0:
            raise ValueError("Invalid degradation CNN dimension/depth settings.")
        for name, dim, depth, heads in components:
            if dim <= 0 or depth <= 0 or heads <= 0 or dim % heads:
                raise ValueError(f"Invalid {name} dimension/depth/head settings: {(dim, depth, heads)}.")


class PatchEmbed(nn.Module):
    """Patchify and project an image to tokens using a strided convolution."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (self.in_chans, self.img_size, self.img_size):
            raise ValueError(
                f"Expected input [B, {self.in_chans}, {self.img_size}, {self.img_size}], "
                f"got {tuple(x.shape)}."
            )
        return self.proj(x).flatten(2).transpose(1, 2)


def unpatchify(
    patches: torch.Tensor, patch_size: int, in_chans: int, grid_size: Tuple[int, int]
) -> torch.Tensor:
    """Convert pixel patches ``[B, N, P*P*C]`` back to BCHW images."""

    batch, tokens, patch_dim = patches.shape
    height, width = grid_size
    expected_dim = patch_size * patch_size * in_chans
    if tokens != height * width or patch_dim != expected_dim:
        raise ValueError(
            f"Expected patches [B, {height * width}, {expected_dim}], got {tuple(patches.shape)}."
        )
    x = patches.reshape(batch, height, width, patch_size, patch_size, in_chans)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(batch, in_chans, height * patch_size, width * patch_size)


class TokenBlockStack(nn.Module):
    """Create and execute configurable token blocks with one consistent API."""

    def __init__(
        self,
        architecture: str,
        dim: int,
        depth: int,
        heads: int,
        cfg: DgNetConfig,
    ) -> None:
        super().__init__()
        self.architecture = architecture.upper()
        blocks: list[nn.Module] = []
        for _ in range(depth):
            if self.architecture == "VIT":
                blocks.append(
                    ViTBlock(
                        dim=dim,
                        num_heads=heads,
                        mlp_ratio=cfg.MLP_RATIO,
                        drop=cfg.PROJ_DROPOUT,
                        attn_drop=cfg.ATTN_DROPOUT,
                        drop_path=cfg.DROP_PATH,
                    )
                )
            elif self.architecture == "HYBRID":
                blocks.append(
                    HybridBlock(
                        dim=dim,
                        num_heads=heads,
                        mlp_ratio=cfg.MLP_RATIO,
                        attn_dropout=cfg.ATTN_DROPOUT,
                        proj_dropout=cfg.PROJ_DROPOUT,
                        cnn_kernel_size=cfg.CNN_KERNEL_SIZE,
                        cnn_dropout=cfg.CNN_DROPOUT,
                        ffn_dropout=cfg.FFN_DROPOUT,
                    )
                )
            elif self.architecture == "CNN":
                blocks.append(
                    TokenCNNBlock(
                        dim=dim,
                        architecture=cfg.CNN_ARCHITECT,
                        kernel_size=cfg.CNN_KERNEL_SIZE,
                        drop_path=cfg.DROP_PATH,
                    )
                )
            else:
                raise ValueError(f"Unsupported architecture: {architecture!r}.")
        self.blocks = nn.ModuleList(blocks)

    def forward(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens) if self.architecture == "VIT" else block(tokens, grid_size)
        return tokens


def _make_positional_embedding(
    num_patches: int, embed_dim: int, enabled: bool, initialize_zero: bool
) -> nn.Parameter | None:
    if not enabled:
        return None
    parameter = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
    if not initialize_zero:
        nn.init.trunc_normal_(parameter, std=0.02)
    return parameter


class ViTEncoder(nn.Module):
    """Image encoder fixed to patch-level ViT blocks."""

    def __init__(self, cfg: DgNetConfig) -> None:
        super().__init__()
        dim = cfg.encoder_dim()
        self.patch_embed = PatchEmbed(cfg.IMG_SIZE, cfg.PATCH_SIZE, cfg.IN_CHANS, dim)
        self.pos_embed = _make_positional_embedding(
            self.patch_embed.num_patches, dim, cfg.USE_POS_EMBED, cfg.POS_EMBED_INIT_ZERO
        )
        self.blocks = TokenBlockStack("VIT", dim, cfg.encoder_depth(), cfg.encoder_heads(), cfg)
        self.norm = nn.LayerNorm(dim)

    @property
    def grid_size(self) -> Tuple[int, int]:
        return self.patch_embed.grid_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(image)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        return self.norm(self.blocks(tokens, self.grid_size))


class ViTDecoder(nn.Module):
    """MAE-style patch decoder mapping encoded tokens to reconstructed pixels."""

    def __init__(self, cfg: DgNetConfig) -> None:
        super().__init__()
        encoder_dim, decoder_dim = cfg.encoder_dim(), cfg.decoder_dim()
        grid = (cfg.IMG_SIZE // cfg.PATCH_SIZE, cfg.IMG_SIZE // cfg.PATCH_SIZE)
        num_patches = grid[0] * grid[1]
        self.grid_size = grid
        self.patch_size = cfg.PATCH_SIZE
        self.in_chans = cfg.IN_CHANS
        self.embed = nn.Linear(encoder_dim, decoder_dim)
        self.pos_embed = _make_positional_embedding(
            num_patches, decoder_dim, cfg.USE_POS_EMBED, cfg.POS_EMBED_INIT_ZERO
        )
        self.blocks = TokenBlockStack("VIT", decoder_dim, cfg.decoder_depth(), cfg.decoder_heads(), cfg)
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, cfg.PATCH_SIZE * cfg.PATCH_SIZE * cfg.IN_CHANS)

    def forward(self, encoded_tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(encoded_tokens)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        predicted_patches = self.pred(self.norm(self.blocks(tokens, self.grid_size)))
        return unpatchify(predicted_patches, self.patch_size, self.in_chans, self.grid_size)


class DegradationGenerator(nn.Module):
    """Selectable residual generator ``M`` with a zero-initialized output head."""

    def __init__(self, cfg: DgNetConfig) -> None:
        super().__init__()
        dim = cfg.dg_dim()
        self.patch_size = cfg.PATCH_SIZE
        self.in_chans = cfg.IN_CHANS
        self.patch_embed = PatchEmbed(cfg.IMG_SIZE, cfg.PATCH_SIZE, cfg.IN_CHANS, dim)
        self.pos_embed = _make_positional_embedding(
            self.patch_embed.num_patches, dim, cfg.USE_POS_EMBED, cfg.POS_EMBED_INIT_ZERO
        )
        self.blocks = TokenBlockStack(cfg.DG_ARCHITECT, dim, cfg.dg_depth(), cfg.dg_heads(), cfg)
        self.norm = nn.LayerNorm(dim)
        self.residual_head = nn.Linear(dim, cfg.PATCH_SIZE * cfg.PATCH_SIZE * cfg.IN_CHANS)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @property
    def grid_size(self) -> Tuple[int, int]:
        return self.patch_embed.grid_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(image)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens, self.grid_size)
        residual_patches = self.residual_head(self.norm(tokens))
        return unpatchify(residual_patches, self.patch_size, self.in_chans, self.grid_size)


@dataclass
class DgNetOutput:
    """Named forward values used by adversarial reconstruction training."""

    original: torch.Tensor
    residual: torch.Tensor
    degraded: torch.Tensor
    reconstruction: torch.Tensor
    original_reconstruction: torch.Tensor
    representation: torch.Tensor
    degraded_representation: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "original": self.original,
            "residual": self.residual,
            "degraded": self.degraded,
            "reconstruction": self.reconstruction,
            "original_reconstruction": self.original_reconstruction,
            "representation": self.representation,
            "degraded_representation": self.degraded_representation,
        }


class DGNet(nn.Module):
    """Adversarial degradation/reconstruction model assembled from ``DgNetConfig``."""

    ConfigClass = DgNetConfig

    def __init__(self, cfg: DgNetConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.degradation = DegradationGenerator(cfg)
        self.encoder = ViTEncoder(cfg)
        self.decoder = ViTDecoder(cfg)
        self.projection = nn.Sequential(
            nn.Linear(cfg.encoder_dim(), cfg.PROJECTION_DIM),
            nn.GELU(),
            nn.Linear(cfg.PROJECTION_DIM, cfg.PROJECTION_DIM),
        )

    def forward(self, image: torch.Tensor) -> DgNetOutput:
        residual = self.degradation(image)
        degraded = image - residual
        if self.cfg.CLAMP_DEGRADED:
            degraded = degraded.clamp(0.0, 1.0)
        original_tokens = self.encoder(image)
        degraded_tokens = self.encoder(degraded)
        representation = self.projection(original_tokens.mean(dim=1))
        degraded_representation = self.projection(degraded_tokens.mean(dim=1))
        return DgNetOutput(
            original=image,
            residual=residual,
            degraded=degraded,
            reconstruction=self.decoder(degraded_tokens),
            original_reconstruction=self.decoder(original_tokens),
            representation=representation,
            degraded_representation=degraded_representation,
        )

    def save(self, save_path: str | Path) -> Path:
        """Save config JSON and weights in one portable model directory."""

        directory = Path(save_path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps(asdict(self.cfg), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        torch.save(self.state_dict(), directory / "model.pt")
        return directory

    @classmethod
    def load(
        cls, save_path: str | Path, map_location: str | torch.device = "cpu", verbose: bool = True
    ) -> "DGNet":
        """Construct from saved config and load weights."""

        directory = Path(save_path)
        config = DgNetConfig(**json.loads((directory / "config.json").read_text(encoding="utf-8")))
        model = cls(config)
        state = torch.load(directory / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        if verbose:
            print(f"DGNet load successful: {directory}")
            print(model.architecture_summary())
        return model

    def architecture_summary(self) -> str:
        lines = [
            (
                f"DGNet encoder={self.cfg.ENCODER_ARCHITECT}, "
                f"decoder={self.cfg.DECODER_ARCHITECT}, degradation={self.cfg.DG_ARCHITECT}"
            )
        ]
        for name, component in (
            ("encoder", self.encoder),
            ("decoder", self.decoder),
            ("degradation", self.degradation),
            ("projection", self.projection),
            ("total", self),
        ):
            parameters, size_mib = parameter_summary(component)
            lines.append(f"{name}: parameters={parameters:,}, parameter_size_mib={size_mib:.4f}")
        lines.extend(["", "encoder_structure:", str(self.encoder), "", "decoder_structure:", str(self.decoder)])
        lines.extend(["", "degradation_structure:", str(self.degradation)])
        return "\n".join(lines)


# Capitalization aliases for project code that prefers DgNet/DGNetConfig.
DgNet = DGNet
DGNetConfig = DgNetConfig


def parameter_summary(module: nn.Module) -> tuple[int, float]:
    parameters = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    bytes_used = sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return parameters, bytes_used / (1024**2)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255)
        .byte()
        .numpy()
    )
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)


def save_sample_visualization(output: DgNetOutput, output_path: str | Path) -> Path:
    """Write one input/degraded/reconstructed sample comparison image."""

    panels = [
        ("original input", output.original[0]),
        ("degraded x - M(x)", output.degraded[0]),
        ("decoder reconstruction", output.reconstruction[0]),
    ]
    rendered: list[Image.Image] = []
    for title, tensor in panels:
        image = _tensor_to_pil(tensor).resize((224, 224))
        panel = Image.new("RGB", (224, 250), "white")
        panel.paste(image, (0, 26))
        ImageDraw.Draw(panel).text((6, 6), title, fill="black")
        rendered.append(panel)
    comparison = Image.new("RGB", (len(rendered) * 224, 250), "white")
    for index, panel in enumerate(rendered):
        comparison.paste(panel, (index * 224, 0))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(path)
    return path


def write_model_report(
    model: DGNet, sample_batch: torch.Tensor, output_dir: str | Path | None = None
) -> Path:
    """Run a debug batch and persist requested architecture/sample artifacts."""

    directory = (
        Path(output_dir) if output_dir else Path(__file__).resolve().parent / "output"
    )
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        output = model(sample_batch)
    report_path = directory / "dg_model.log"
    report = [
        "DGNet construction and sample inference successful.",
        model.architecture_summary(),
        "",
        f"input_shape: {tuple(sample_batch.shape)}",
        f"residual_M_shape: {tuple(output.residual.shape)}",
        f"degraded_shape: {tuple(output.degraded.shape)}",
        f"reconstruction_shape: {tuple(output.reconstruction.shape)}",
        f"representation_shape: {tuple(output.representation.shape)}",
        f"initial_residual_max_abs: {output.residual.abs().max().item():.8f}",
        f"initial_degraded_equals_input: {torch.equal(output.degraded, sample_batch)}",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    save_sample_visualization(output, directory / "dg_model_sample.png")
    return report_path


def _self_test() -> Path:
    """Exercise all degradation choices, zero-init, reports, and save/load."""

    directory = Path(__file__).resolve().parent / "output"
    base = DgNetConfig(
        IMG_SIZE=32,
        PATCH_SIZE=8,
        EMBED_DIM=64,
        DEPTH=2,
        NUM_HEADS=4,
        DECODER_EMBED_DIM=48,
        DECODER_NUM_HEADS=4,
        PROJECTION_DIM=32,
    )
    torch.manual_seed(0)
    sample = torch.rand(2, 3, 32, 32)
    verification: list[str] = []
    for architecture in ("VIT", "CNN", "HYBRID"):
        model = DGNet(replace(base, DG_ARCHITECT=architecture)).eval()
        with torch.no_grad():
            output = model(sample)
        assert output.residual.shape == sample.shape
        assert torch.count_nonzero(output.residual) == 0
        assert torch.equal(output.degraded, sample)
        parameters, size_mib = parameter_summary(model.degradation)
        verification.extend(
            [
                (
                    f"{architecture}: zero residual and shape verification passed; "
                    f"parameters={parameters:,}, parameter_size_mib={size_mib:.4f}"
                ),
                str(model.degradation),
                "",
            ]
        )

    model = DGNet(base).eval()
    report_path = write_model_report(model, sample, directory)
    saved = model.save(directory / "saved_dgnet")
    loaded = DGNet.load(saved, verbose=False).eval()
    with torch.no_grad():
        before, after = model(sample), loaded(sample)
    assert torch.equal(before.residual, after.residual)
    assert torch.allclose(before.reconstruction, after.reconstruction)
    with report_path.open("a", encoding="utf-8") as report:
        report.write("\narchitecture_variants:\n" + "\n".join(verification) + "\n")
        report.write("model_load_success: True\nsave_load_round_trip: passed\n")
        report.write(f"saved_directory: {saved}\n")
    return report_path


if __name__ == "__main__":
    print(f"Wrote DGNet verification artifacts: {_self_test()}")

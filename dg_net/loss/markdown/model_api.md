## dgnet 모델 사용 api
```python
from dg_model import DgNetConfig, DGNet, write_model_report
from pathlib import Path

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
model = DGNet(base)

saved = model.save(directory / "saved_dgnet")   # 저장
loaded = DGNet.load(saved, verbose=False)       # 불러오기
```

### DgNetConfig
```python
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
```


참고로 모델 save, load 시구조는 아래와 같다.
```text
directory/
    config.json     # 모델 아키텍쳐 재현정보
    model.pt        # 실제 모델의 가중치 파일
```

## dg_model.py 테스트 샘플 코드 api
```python


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

```
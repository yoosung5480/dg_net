# Model API

```python
import torch
from model.dg_model import DGNet, DgNetConfig
model = DGNet(DgNetConfig(IMG_SIZE=32, PATCH_SIZE=8, EMBED_DIM=64, DEPTH=2, NUM_HEADS=4))
out = model(torch.rand(2, 3, 32, 32))
```

공개 API는 `DGNet`, `DgNetConfig`, `DgNetOutput`, `parameter_summary`,
`save_sample_visualization`, `write_model_report`다. output은 `original`,
`residual`, `degraded`, `reconstruction`, `original_reconstruction`,
`representation`, `degraded_representation`을 제공한다.

`DG_ARCHITECT`는 `VIT/CNN/HYBRID` ablation을 선택한다. residual head는 zero-init
되므로 최초 residual은 0이다. `model.save(path)`와 `DGNet.load(path)`는
`config.json + model.pt` round trip을 제공한다.

```bash
python -m model.dg_model
# model/output/dg_model.log, dg_model_sample.png, saved_dgnet/
```

DDP checkpoint는 wrapper가 아닌 `ddp_model.module.save(path)`로 저장해야 한다.
훈련 샘플은 checkpoint 디렉터리의 PNG에 기록된다.

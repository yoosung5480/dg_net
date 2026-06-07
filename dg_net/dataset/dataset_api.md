# Dataset API

`DataConfig`와 `prepare_dataloader`가 SSL/classification loader를 만든다.

```python
from dataset import DataConfig, prepare_dataloader
loader = prepare_dataloader(DataConfig(dataset="STL10", mode="ssl", batch_size=4))
views, labels = next(iter(loader))  # views[0]: [B,C,H,W]
```

분류 mode는 `(images, labels)`를 반환한다. STL10 SSL은 항상 unlabeled split을
사용하며 CIFAR100/Flowers102/iNaturalist는 transfer classification용이다.
`Synthetic`는 데이터 파일이 없는 CI/smoke 전용이다.

주요 필드: `data_path`, `dataset`, `mode`, `split`, `img_size`, `mean/std`,
`batch_size`, `max_samples`, `distributed`, `rank`, `world_size`.
`CLAMP_DEGRADED=True`이면 pixel 의미를 보존하도록 `mean=0`, `std=1`을 사용한다.

```python
cfg = DataConfig(dataset="STL10", mode="ssl", distributed=True, rank=0, world_size=2)
loader = prepare_dataloader(cfg)
for epoch in range(10):
    loader.sampler.set_epoch(epoch)
```

성공 보고서는 `output_dir/*.txt`에 저장된다. 실제 연결은
`engine/dg_train.py`와 `scripts/run_dgnet_pipeline_*.bash`를 사용한다.

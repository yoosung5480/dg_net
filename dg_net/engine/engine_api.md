# Engine API

## One-shot 실행

```bash
CUDA_DEVICE=0 DATA_PATH=/data PRETRAIN_EPOCHS=100 LINEAR_EPOCHS=20 \
  bash scripts/run_dgnet_pipeline_single_gpu.bash

CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 DATA_PATH=/data \
  bash scripts/run_dgnet_pipeline_ddp.bash
```

두 script는 pretrain 최신 `epoch*/step*` checkpoint를 찾아 linear probe에 전달한다.
개별 실행과 resume:

```bash
python engine/dg_train.py --loss-mode hybrid --loop-mode step --total-steps 100
python engine/dg_train.py --resume output/run/step000100 --total-steps 200
python engine/linear_probe_train.py --checkpoint output/run/step000200
```

`TrainConfig`는 model/data/loss/optimizer/runtime/DDP 필드를 조립하고
`build_model_config`, `build_data_config`, `build_loss_config`를 제공한다.
DDP는 torchrun의 `LOCAL_RANK/RANK/WORLD_SIZE`, NCCL, DistributedSampler,
`sampler.set_epoch`, DDP wrapper를 사용하며 logging/checkpoint/artifact는 rank 0만 쓴다.

checkpoint에는 `model.pt`, model `config.json`, `train_config.json`,
`checkpoint.pt`가 있으며 engine state는 optimizer I/M, scheduler I/M,
`global_step`, `epoch`, best score를 포함한다.

run root에는 `config.json`, `train.csv`, `run.log`, `loss_curve.png`,
`accuracy.png`가 기록된다. CSV에는 raw/weighted loss terms, effective weights,
두 phase objective와 learning rate가 기록된다.

# DGNet Engine API

`dg_net/engine`는 `dataset`, `model`, `loss` 공개 API를 조립해 DGNet 사전학습, 표현 평가, linear probe, fine-tuning을 실행하는 훈련 루프 패키지이다.

## 공개 모듈

| 파일 | 역할 |
| --- | --- |
| `train_config.py` | 모든 실행 스크립트가 공유하는 `TrainConfig`, CLI parser, trainer factory |
| `eval.py` | validation, CSV 기록, curve plot, sample artifact 저장 유틸리티 |
| `dg_train.py` | DGNet self-supervised alternating pretraining executable |
| `linear_probe_train.py` | frozen DGNet representation 위 linear classifier 학습 |
| `finetune_train.py` | DGNet encoder/projection + classifier fine-tuning |
| `knn_eval.py` | DGNet representation kNN 평가 |

엔진 실행 파일끼리는 서로 import하지 않는다. 공유 표면은 `train_config.py`와 `eval.py`뿐이다.

## 가장 짧은 DGNet pretrain smoke

```bash
python dg_train.py \
  --loop-mode step \
  --total-steps 100 \
  --validate-every-steps 50 \
  --save-every-steps 100 \
  --dataset STL10 \
  --data-path /home/jeongyuseong/바탕화면/datasets \
  --img-size 32 \
  --batch-size 4 \
  --num-workers 0 \
  --max-samples 400 \
  --no-use-augmentation \
  --output-root output
```

생성 산출물:

```text
output/{run_id}/
├── config.json
├── train.csv
├── loss_curve.png
├── accuracy.png        # DG pretrain에서는 validation_score curve
├── run.log
├── step000000/ 또는 epoch000000/
└── step000100/ 또는 epoch000001/
    ├── model.pt
    ├── config.json     # DGNet model config; DGNet.load() 호환
    ├── train_config.json
    ├── checkpoint.pt   # optimizer/global step 등 engine state
    ├── {tag}.png
    ├── dg_model.log
    └── loss.log
```

## `TrainConfig`

```python
from train_config import TrainConfig, get_trainer

cfg = TrainConfig(
    task="dg_pretrain",
    loop_mode="step",
    total_steps=100,
    dataset="STL10",
    img_size=32,
    batch_size=4,
)
trainer = get_trainer(cfg)
trainer.run()
```

핵심 메서드:

- `build_data_config(split=None, mode=None, train=True) -> dataset.DataConfig`
- `build_loss_config() -> loss.loss_config.LossConfig`
- `build_model_config() -> model.dg_model.DgNetConfig`
- `resolve_device() -> torch.device` (`auto`는 CUDA 가능 시 CUDA, 아니면 CPU)
- `prepare_output_dir() -> Path`
- `write_json(path) -> Path`

주요 CLI 인자는 모든 executable에서 동일하게 지원한다.

```text
--task {dg_pretrain,linear_probe,knn_eval,finetune,eval}
--loop-mode {step,epoch}
--total-steps / --total-epochs
--validate-every-steps / --validate-every-epochs
--save-every-steps / --save-every-epochs
--output-root --run-name --seed --device
--data-path --dataset --img-size --batch-size --num-workers --max-samples
--use-augmentation / --no-use-augmentation
--checkpoint --resume
```
그 외 모델 훈련 주요 외부선언 가능 인자들은 `dg_net/engine/train_config.py`를 보고 참고한다.

## DGNet pretraining loop

`DGTrainer`는 한 batch에 대해 두 optimizer phase를 분리한다.

1. **I phase**: `model.degradation` freeze, `encoder/decoder/projection` train
   - objective: `values.inference_loss`
2. **M phase**: `model.degradation` train, `encoder/decoder/projection` freeze
   - objective: `values.degradation_loss`

`values.combined_loss`는 CSV/log smoke metric으로만 사용하며 단일 optimizer objective로 쓰지 않는다.

## 평가 유틸리티

```python
from eval import evaluate_dgnet, save_eval_artifacts, append_metrics_csv

metrics = evaluate_dgnet(model, loss_fn, loader, device, step=global_step)
save_eval_artifacts(model, loss_fn, fixed_sample, "output/run/step000100", "step000100")
append_metrics_csv("output/run/train.csv", metrics)
```

- `evaluate_dgnet`: validation loss, reconstruction distance, target distance, budget distance, KL, validation score 반환
- `evaluate_classifier`: classification loss, top1/top5 accuracy 반환
- `plot_loss_curve`: `train.csv`에서 `loss_curve.png` 재생성
- `plot_accuracy_curve`: `accuracy.png` 또는 DG validation score curve 생성

## Transfer tasks

### Linear probe

```bash
python linear_probe_train.py \
  --checkpoint output/{pretrain_run}/step000100 \
  --dataset STL10 \
  --loop-mode step \
  --total-steps 1000
```

- DGNet 전체를 freeze한다.
- `output.representation` 위에 `Linear(PROJECTION_DIM, num_classes)`만 학습한다.
- `train.csv`, `loss_curve.png`, `accuracy.png`, checkpoint를 저장한다.

### Fine-tune

```bash
python finetune_train.py \
  --checkpoint output/{pretrain_run}/step000100 \
  --dataset STL10 \
  --loop-mode epoch \
  --total-epochs 10
```

- 기본 정책은 encoder/projection + classifier 학습, degradation/decoder freeze이다.
- classifier learning rate는 `--classifier-lr`, encoder learning rate는 `--encoder-lr`로 분리한다.

### kNN eval

```bash
python knn_eval.py \
  --checkpoint output/{pretrain_run}/step000100 \
  --dataset STL10 \
  --knn-k 20
```

- classification train split으로 feature bank를 만든다.
- validation/test split query에 대해 cosine similarity kNN accuracy를 계산한다.
- `train.csv`, `accuracy.png`, `knn_eval.log`를 저장한다.

## CSV columns

DGNet pretrain 최소 컬럼:

```text
run_id,task,loop_mode,epoch,step,split,phase,train_loss,validation_loss,validation_score,
inference_loss,degradation_loss,reconstruction_distance,target_distance,mask_budget_distance,damage_kl,
alpha_inference,alpha_target,lambda_budget,lambda_reg,lr_inference,lr_degradation,elapsed_sec
```

Classification/kNN 최소 컬럼:

```text
run_id,task,loop_mode,epoch,step,split,train_loss,validation_loss,validation_score,accuracy,top1,top5,lr,elapsed_sec
```

없는 값은 빈 문자열 또는 `nan`으로 기록되며 plot 함수는 이를 건너뛴다.

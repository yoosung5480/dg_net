# DGNet harness

이 디렉토리는 새 DGNet 훈련 프레임워크 구현을 위한 agent용 하네스 지침서이다.

## 현재 가상환경
environment: pretext_task_gen
python: 3.10.20
torch: 2.2.2+cu121
torchvision: 0.17.2+cu121
cuda_available: True


## 기존 프레임워크 참고용 api모음집
```text
version2_
version1_api/
├── dataset_api.md
├── model_api.md
├── engine_api.bash
├── loss_api.bash
```

## 포함 파일

```text
version2_
harness/
├── agent.md
├── package_api_requirements.md
├── run_dgnet_pipeline_single_gpu.bash
├── run_dgnet_pipeline_ddp.bash
└── smoke_loss_scheduler.bash
```

## 적용 대상

1. 새 loss objective
2. term-wise scheduler
3. representation/reconstruction/hybrid ablation
4. DDP multi-GPU pretraining
5. pretrain -> linear probe one-shot execution
6. 각 패키지별 api.md 갱신

## 사용법

`agent.md`를 구현 agent의 최상위 지침으로 사용한다.

bash script는 `src/scripts/`에 복사해서 사용한다.

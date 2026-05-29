# DGNet small-scale experiment plan

이 실험 묶음은 ADIOS 재현용 200 epoch 세팅으로 넘어가기 전, `dg_net` 프레임워크만으로 collapse 여부와 사전학습 표현의 전이 가능성을 확인하기 위한 10 epoch 소규모 검증이다. 모든 스크립트는 `pretrain -> linear probe`를 연속 실행하고, 산출물은 `dg_net/output` 아래에 저장한다.

## 실행 스크립트

| Case | Script | 목적 | dg_model | 동일 조건 | Output |
| --- | --- | --- | --- | --- | --- |
| case1 | `hybrid_dgnet_trian.bash` | baseline. MAE/DINO 비교 전 DGNet 기본 성능과 collapse 확인 | `HYBRID` | batch 16, SSL 10 epochs, linear probe 10 epochs, STL10, 동일 encoder/decoder/loss/optimizer | `output/case1-hybrid-dgnet-pretrain-*`, `output/case1-hybrid-dgnet-linear-probe-*` |
| case2.1 | `vit_dgnet_trian.bash` | case1 best 대비 degradation architecture ablation | `VIT` | case1과 동일. `--dg-architect`만 변경 | `output/case2_1-vit-dgnet-pretrain-*`, `output/case2_1-vit-dgnet-linear-probe-*` |
| case2.2 | `cnn_dgnet_trian.bash` | case1 best 대비 degradation architecture ablation | `CNN` / `CONVNEXT` | case1과 동일. `--dg-architect`만 변경 | `output/case2_2-cnn-dgnet-pretrain-*`, `output/case2_2-cnn-dgnet-linear-probe-*` |

## 공통 데이터/런타임 설정

| Field | Value | 이유 |
| --- | ---: | --- |
| dataset | `STL10` | ADIOS/SSL baseline 비교 대상으로 유지 |
| image size | `32` | 현재 engine smoke와 맞춘 소규모 collapse 검증용 해상도 |
| batch size | `16` | TODO 요구사항 |
| SSL epochs | `10` | TODO 요구사항 |
| linear probe epochs | `10` | TODO 요구사항 |
| max samples | `5000` | 빠른 collapse/학습 정상성 확인용. 전체 데이터 사용 시 `MAX_SAMPLES=0`으로 실행 |
| output root | `output` | TODO 요구사항의 `dg_net/output` |
| scheduler | off | 현재 engine의 `ScheduleConfig()` 기본값 사용 |
| seed | `42` | 세 case 비교 재현성 |

## 권장 모델 하이퍼파라미터

| Component | Field | Value |
| --- | --- | ---: |
| shared | `patch_size` | `4` |
| shared | `embed_dim`, `depth`, `num_heads` | `96`, `2`, `4` |
| encoder | `encoder_embed_dim`, `encoder_depth`, `encoder_num_heads` | `128`, `4`, `4` |
| decoder | `decoder_embed_dim`, `decoder_depth`, `decoder_num_heads` | `96`, `2`, `4` |
| dg_model | `dg_embed_dim`, `dg_depth`, `dg_num_heads` | `96`, `2`, `4` |
| projection | `projection_dim` | `128` |
| regularization | `attn_dropout`, `proj_dropout`, `drop_path` | `0.0`, `0.0`, `0.0` |

이 설정의 HYBRID 기준 trainable parameter는 대략 `dg_model 260k : encoder 808k : decoder 247k`로, 요구한 `1 : 3 : 1` 비율에 가깝다. VIT/CNN ablation은 dg_model block 종류만 바꾸므로 encoder 중심 조건은 유지된다.

## 권장 loss/optimizer 하이퍼파라미터

| Field | Value | 이유 |
| --- | ---: | --- |
| `alpha_inference` | `1.0` | inference reconstruction objective 기준 계수 |
| `alpha_target` | `1.0` | degradation target matching 기준 계수 |
| `lambda_budget` | `2.0` | 작은 실험에서 residual budget 이탈을 더 강하게 억제 |
| `lambda_reg` | `0.1` | brightness-damage collapse 방지 항 유지 |
| `tau_deg` | `0.25` | 10 epoch 안정성 검증용. `0.5`는 초기 소규모 실험에서는 과한 degradation 목표가 될 수 있음 |
| `beta_mask` | `0.15` | 평균 residual magnitude를 낮게 시작해 collapse를 먼저 확인 |
| `loss_reduction` | `mean` | batch/해상도 변화에 덜 민감 |
| `lr_inference` | `3e-4` | encoder/decoder AdamW 기본 학습률 |
| `lr_degradation` | `1e-4` | adversarial degradation 쪽을 더 천천히 업데이트 |
| `classifier_lr` | `1e-3` | frozen representation 위 linear classifier 학습 |
| `weight_decay` | `5e-2` pretrain, `1e-4` linear probe | transformer pretrain regularization과 probe 안정성 분리 |

## 실행 예시

```bash
cd /home/jeongyuseong/바탕화면/SSL/DGnet_proj/version1/dg_net
bash hybrid_dgnet_trian.bash
bash vit_dgnet_trian.bash
bash cnn_dgnet_trian.bash
```

전체 STL10 split으로 돌릴 때는 다음처럼 `MAX_SAMPLES=0`을 지정한다.

```bash
MAX_SAMPLES=0 bash hybrid_dgnet_trian.bash
```

# DGNet small-scale experiment plan

이 실험 묶음은 ADIOS 재현용 200 epoch 세팅으로 넘어가기 전, `dg_net` 프레임워크만으로 collapse 여부와 사전학습 표현의 전이 가능성을 확인하기 위한 10 epoch 소규모 검증이다. 모든 스크립트는 `pretrain -> linear probe`를 연속 실행하고, 산출물은 `dg_net/output` 아래에 저장한다.

## 실행 스크립트

| Case | Script | 목적 | dg_model | 동일 조건 | Output |
| --- | --- | --- | --- | --- | --- |
| case1 | `hybrid_dgnet_trian.bash` | baseline. MAE/DINO 비교 전 DGNet 기본 성능과 collapse 확인 | `HYBRID` | batch 16, SSL 10 epochs, linear probe 10 epochs, STL10, 동일 encoder/decoder/loss/optimizer | `output/case1-hybrid-dgnet-pretrain-*`, `output/case1-hybrid-dgnet-linear-probe-*` |
| case2.1 | `vit_dgnet_trian.bash` | case1 best 대비 degradation architecture ablation | `VIT` | case1과 동일. `--dg-architect`만 변경 | `output/case2_1-vit-dgnet-pretrain-*`, `output/case2_1-vit-dgnet-linear-probe-*` |
| case2.2 | `cnn_dgnet_trian.bash` | case1 best 대비 degradation architecture ablation | `CNN` / `CONVNEXT` | case1과 동일. `--dg-architect`만 변경 | `output/case2_2-cnn-dgnet-pretrain-*`, `output/case2_2-cnn-dgnet-linear-probe-*` |


이 설정의 HYBRID 기준 trainable parameter는 대략 `dg_model 260k : encoder 808k : decoder 247k`로, 요구한 `1 : 3 : 1` 비율에 가깝다. VIT/CNN ablation은 dg_model block 종류만 바꾸므로 encoder 중심 조건은 유지된다.

## 권장 loss/optimizer 하이퍼파라미터

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

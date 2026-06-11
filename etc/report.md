# DGNet 구현 현황 및 실패 원인 분석

## 1. 결론 요약

- [확인됨] 현재와 구형 degradation generator는 모두 이미지를 patch token으로 바꾸고, token별 pixel patch를 출력한 뒤 unpatchify한다. 구형이 전체 이미지를 단일 flatten vector로 출력했다는 근거는 없다.
- [확인됨] 현재 VIT/HYBRID generator에는 self-attention이 있어 patch 간 전역 정보 교환이 가능하다. CNN-only의 교환 범위는 patch-grid convolution receptive field로 제한된다.
- [확인됨] 현재 representation objective는 cosine distance만 지원하며 InfoNCE는 없다.
- [확인됨] engine은 I phase에서 degradation을 freeze하고 inference network를 갱신하며, M phase에서 inference network를 freeze하고 degradation만 갱신한다. M phase forward에 `no_grad`가 없어 frozen inference 연산을 통과하는 gradient graph가 유지된다.
- [추론] 실험 실패 원인은 “전역 정보 교환 부재” 하나로 확정할 수 없다. objective 선택 제한, patch 출력 경계 seam, 실제 gradient 크기/학습 동역학을 분리 진단해야 한다.

## 2. 현재/구형 architecture tensor flow 비교

고정 runtime config: `IMG_SIZE=32, PATCH_SIZE=8, EMBED_DIM=64, DEPTH=2, NUM_HEADS=4, DECODER_EMBED_DIM=48, DECODER_NUM_HEADS=4, PROJECTION_DIM=32, DG_ARCHITECT=HYBRID`, batch 2.

| 단계 | 현재 구현 | 구형 구현 |
|---|---|---|
| input image | [확인됨] `[2,3,32,32]` | [확인됨] `x [B,C,H,W]` |
| patch embedding | [확인됨] `[2,16,64]`, strided Conv2d | [확인됨] `[B,N,D]`, strided Conv2d |
| DG blocks output | [확인됨] `[2,16,64]`, selectable VIT/CNN/HYBRID | [확인됨] `[B,N,D]`, hybrid blocks |
| DG output head | [확인됨] `[2,16,192]` token별 patch residual | [확인됨] `[B,N,patch_dim]` token별 delta patch |
| residual/degraded output | [확인됨] residual `[2,3,32,32]`; `degraded=x-residual` | [확인됨] raw input patches와 delta patches를 조합 후 unpatchify한 degraded image |
| encoder tokens | [확인됨] `[2,16,64]` | [확인됨] MAE encoder latent `[B,N,D]` |
| pooled representation | [확인됨] `[2,64]` mean pooling | [확인됨] latent token mean |
| projection output | [확인됨] `[2,32]` MLP projection | [확인됨] 별도 projection 없이 pooled representation 사용 |
| decoder patch prediction | [확인됨] `[2,16,192]` | [확인됨] MAE patch prediction |
| reconstruction image | [확인됨] `[2,3,32,32]`, decoder 내부 unpatchify | [확인됨] learner visualization에서 MAE prediction을 unpatchify |

## 3. 현재 loss 비교와 objective 차이

- [확인됨] `RepresentationDistance`는 `mean(1-cosine_similarity(z_original,z_degraded))`이다.
- [확인됨] inference loss는 reconstruction 및 representation weighted sum을 최소화한다.
- [확인됨] degradation loss는 target/budget을 최소화하고 KL 및 representation distance에 음의 weight를 적용해 최대화한다.
- [확인됨] term별 weight와 linear/cosine scheduler가 있다.
- [확인됨] symmetric local in-batch InfoNCE 및 temperature config는 없다.
- [추론] cosine은 sample 간 negatives를 직접 사용하지 않으므로, 동일 batch의 다른 sample을 구분하는 InfoNCE와 학습 신호가 다르다.

## 4. I/M phase gradient flow 분석

| Phase | degradation | encoder | decoder | projection | 의도 |
|---|---|---|---|---|---|
| I | freeze | train | train | train | reconstruction/representation objective 최소화 |
| M | train | freeze | freeze | freeze | degradation objective 최소화; 음수 representation term으로 distance 최대화 |

- [확인됨] `_phase_i()`와 `_phase_m()`가 parameter `requires_grad`를 위 표대로 전환한다.
- [확인됨] M phase에도 전체 model forward를 수행하며 `torch.no_grad()`를 사용하지 않는다.
- [추론] 따라서 frozen encoder/projection의 parameter gradient는 막되, `z_degraded`에서 degradation까지의 입력 gradient는 유지된다.
- [미확인] 각 objective에서 실제 finite non-zero gradient와 parameter update isolation은 Phase 1 회귀 테스트로 확인해야 한다.

## 5. 패치 독립성/전역성 진단 결과

- [확인됨] VIT/HYBRID는 multi-head self-attention을 포함하므로 token 간 전역 의존 경로가 존재한다.
- [확인됨] CNN-only는 patch grid에서 convolution을 적용하므로 depth/kernel에 따른 국소 receptive field를 가진다.
- [미확인] deterministic fixture 기반 VIT/HYBRID cross-patch influence ratio.
- [미확인] VIT/HYBRID residual의 patch seam ratio와 spatial refinement 권장 여부.

## 6. 확인된 구현 문제

1. [확인됨] representation objective가 cosine으로 고정되어 config/CLI 이름과 달리 선택지가 없다.
2. [확인됨] ImageNet ImageFolder loader와 1000-class engine mapping이 없다.
3. [확인됨] sample visualization은 original/degraded/reconstruction 3개 panel만 제공한다.
4. [확인됨] model/loss/engine에 정식 pytest 회귀 테스트가 없다.
5. [확인됨] optional full-image spatial refinement head가 없다.
6. [확인됨] canonical API 문서 외 중복 문서와 오래된 compatibility script가 존재한다.

## 7. 설계 차이

- [확인됨] 현재 model은 generator/encoder/decoder/projection을 단일 config와 `DgNetOutput` 7개 필드로 조립한다.
- [확인됨] 구형 learner는 degradation net과 MAE를 조립하며 projection MLP 없이 pooled MAE latent를 representation으로 사용한다.
- [확인됨] 현재 generator는 `residual`을 반환하고 DGNet에서 `x-residual`을 계산한다. 구형 generator는 내부에서 degraded image를 직접 조합한다.
- [추론] 현재 구조는 tensor 계약과 역할 분리가 더 명확하지만 patch residual head의 경계 artifact 가능성은 별도 진단이 필요하다.

## 8. 미확인 가설과 한계

- [미확인] cosine 대비 InfoNCE의 실제 학습 품질 우월성. 2-step smoke는 실행 가능성만 검증한다.
- [미확인] 실제 checkpoint에서 시각적 patch seam이 반복되는지 여부.
- [미확인] CUDA 2-GPU DDP 실행 가능 여부.
- [미확인] 장기 학습 안정성 및 downstream 성능.
- [확인됨] 구형 코드는 `src.*` import 전제를 가져 현재 위치에서 직접 실행하지 않고 정적 분석했다.

## 9. 리팩터링 결정

정리 범위는 `harness/agent.md`가 허용한 `dg_net` 파일과 필수 신규 테스트/문서로 제한한다.

1. 기존 동작을 model/loss/gradient-flow 회귀 테스트로 먼저 고정한다.
2. dead/stale surfaces는 삭제 대신 compatibility 안내/wrapper로 정리한다.
3. 중복 objective 선택 로직 없이 `DGLoss` 내부에서 cosine/InfoNCE를 선택한다.
4. M representation reference는 `z_original.detach()`로 명시한다.
5. spatial refinement는 default-off additive config로만 추가한다.
6. 기존 subset utility를 재사용해 ImageNet ImageFolder를 추가한다.
7. visualization/API/script를 실제 구현과 일치시킨다.
8. 새 dependency와 public API 삭제/이름 변경은 하지 않는다.

## 10. 실행한 검증 명령과 결과

| 명령 | 종료 코드 | 핵심 결과 |
|---|---:|---|
| `python -m pytest dg_net/dataset/tests/test_dataset.py -q` | 0 | [확인됨] 8 passed |
| `python -m compileall -q dg_net` | 0 | [확인됨] compile 성공 |
| `PYTHONPATH=dg_net python -m loss.loss` | 0 | [확인됨] loss smoke artifact 생성 |
| `python dg_net/model/dg_model.py` | 0 | [확인됨] model verification artifact 생성 |
| 고정 config runtime shape probe | 0 | [확인됨] 2절 현재 tensor shape 측정 |


## 11. 구현 변경 내역

- [확인됨] `DGLoss`에 symmetric local in-batch InfoNCE를 추가하고 `LossConfig`/CLI의 `representation_distance`로 cosine과 선택 가능하게 연결했다. default temperature는 `0.2`이며 batch size 2 미만은 실패한다.
- [확인됨] M representation objective는 `z_original.detach()`를 reference로 사용하며 frozen inference 연산을 통과한 gradient가 degradation에 도달한다.
- [확인됨] `DgNetConfig`에 default-off spatial refinement와 channel config를 추가했고, 구형 config JSON load 호환성을 유지했다.
- [확인됨] ImageNet/ImageFolder SSL·classification loader, train/val root dispatch, alias, subset/filter, distributed sampler를 추가했다.
- [확인됨] visualization을 7-panel과 residual/reconstruction/cosine/seam 통계 sidecar로 확장했다.
- [확인됨] canonical package API 문서와 single/DDP script, legacy wrapper를 실제 CLI에 맞게 정리했다.

## 12. Cosine/InfoNCE 비교 결과

- [확인됨] cosine과 InfoNCE 모두 loss unit test, backward, I/M gradient isolation, CPU Synthetic 2-step pretrain smoke를 통과했다.
- [확인됨] InfoNCE는 `z_original @ z_degraded.T / temperature`의 forward/reverse cross entropy 평균이며 negatives는 process-local batch만 사용한다.
- [미확인] 2-step smoke만으로 어느 objective의 학습 품질 또는 downstream 성능이 우월한지는 판단할 수 없다.

## 13. 전역성 및 seam metric 결과

고정 deterministic fixture 평균:

| Architecture | cross-patch ratio 평균 | seam ratio 평균 |
|---|---:|---:|
| VIT | 0.04922224 | 0.97692741 |
| CNN | 0.00000416 | 1.07000786 |
| HYBRID | 0.13038552 | 1.08366182 |

- [확인됨] VIT/HYBRID 모두 cross-patch ratio가 `1e-4`를 초과하여 전역 의존 경로가 동작한다.
- [확인됨] VIT/HYBRID seam ratio가 `1.25` 이하이므로 이번 진단에서는 spatial refinement 사용을 필수 권장하지 않는다. default-off 유지가 권장된다.
- [확인됨] zero-init refinement on/off seam ratio는 동일했으며 non-regression test를 통과했다.

## 14. Gradient flow 검증 결과

- [확인됨] cosine/InfoNCE 모두 I phase에서 degradation이 unchanged이며 inference 대상 module만 갱신된다.
- [확인됨] cosine/InfoNCE 모두 M phase에서 encoder/decoder/projection은 unchanged이고 degradation parameter에 finite non-zero gradient와 update가 발생한다.
- [확인됨] 전체 관련 pytest 기준 `45 passed`이다.

## 15. ImageNet loader 검증 결과

- [확인됨] 임시 ImageFolder fixture로 classification train/val dispatch, SSL multi-view, root fallback, class filter/max_samples, distributed sampler, `ImageNet`/`imagenet`/`imagefolder` alias를 검증했다.
- [확인됨] engine의 ImageNet 기본 class count는 1000으로 연결되었다.

## 16. Single/DDP/Transfer smoke 결과

- [확인됨] CPU Synthetic cosine pretrain 2-step: 완료, `output/harness_refactor/cosine-pretrain/step000002` 생성.
- [확인됨] CPU Synthetic InfoNCE pretrain 2-step: 완료, `output/harness_refactor/infonce-pretrain/step000002` 생성.
- [확인됨] cosine checkpoint 기반 linear probe 1-step, finetune 1-step, kNN smoke가 모두 완료되었다.
- [확인됨] 각 pretrain checkpoint에 model/config/checkpoint/CSV/loss log/visualization PNG 및 metrics artifact가 생성되었다.
- [미확인] CUDA device가 1개뿐이므로 2-GPU DDP runtime smoke는 실행하지 않았다. DDP script syntax는 검증했다.

## 17. 남은 위험과 미검증 항목

- [미확인] 장기 학습 안정성, 실제 ImageNet 규모 처리 성능, cosine 대비 InfoNCE 성능 우월성.
- [미확인] 실제 학습 checkpoint에서 반복적인 patch seam이 시각적으로 나타나는지 여부.
- [미확인] multi-node 또는 2-GPU DDP runtime 동작. 현재 환경은 CUDA 1개이다.
- [확인됨] 사용자가 작업 전 만들어 둔 unrelated 삭제/추가 변경은 되돌리지 않았다.

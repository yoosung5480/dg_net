# package_api_requirements.md

## 1. 작성해야 할 API 문서

각 패키지는 반드시 독립적인 `api.md`를 가진다.

```text
src/model/model_api.md
src/loss/loss_api.md
src/dataset/dataset_api.md
src/engine/engine_api.md
```

---

## 2. 공통 작성 규칙

모든 API 문서는 다음을 포함한다.

1. 패키지의 역할
2. 공개 API 목록
3. 가장 짧은 실행 예제
4. Config 필드 설명
5. smoke test 예제
6. output artifact 설명
7. train script 연결 예제
8. DDP 또는 distributed 사용 시 주의사항

---

## 3. model_api.md 필수 내용

`model_api.md`는 다음을 포함한다.

1. `DGNet`, `DgNetConfig` 생성 예제
2. forward output field 설명
3. `DG_ARCHITECT={VIT,CNN,HYBRID}` ablation 예제
4. zero-init residual head 확인 예제
5. save/load round trip 예제
6. parameter summary 예제
7. sample visualization 예제
8. DDP wrapping 전/후 checkpoint 저장 주의사항

---

## 4. loss_api.md 필수 내용

`loss_api.md`는 다음을 포함한다.

1. `TermWeightConfig` 정의와 사용법
2. `LossConfig` 정의와 preset
3. representation only 예제
4. reconstruction only 예제
5. hybrid 예제
6. term-wise scheduler 예제
7. term off 예제
8. DGLoss 입력 tensor 계약
9. DGLossOutput field 설명
10. alternating training에서 `inference_loss`, `degradation_loss`를 따로 쓰는 예제

---

## 5. dataset_api.md 필수 내용

`dataset_api.md`는 다음을 포함한다.

1. `DataConfig` 사용법
2. SSL loader 반환 형식
3. classification loader 반환 형식
4. STL10 unlabeled split 사용법
5. transfer dataset 사용법
6. `distributed=True`일 때 sampler 설정
7. `sampler.set_epoch(epoch)` 호출 위치
8. `mean/std`와 `CLAMP_DEGRADED` 관계

---

## 6. engine_api.md 필수 내용

`engine_api.md`는 다음을 포함한다.

1. single GPU DGNet pretrain 실행 예제
2. DDP DGNet pretrain 실행 예제
3. pretrain 후 linear probe 실행 예제
4. bash one-shot pipeline 예제
5. checkpoint 구조
6. resume 예제
7. rank 0 logging/checkpoint 규칙
8. CSV columns
9. loss term logging columns
10. transfer task 실행 규칙

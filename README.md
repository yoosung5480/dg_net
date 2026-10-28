# DGNet

이미지별 손상 생성기와 복원 모델을 교대 학습하는 시각 자기지도학습 연구 프로젝트다. 고정된 무작위 마스킹 대신 학습 가능한 손상을 사용해 인코더의 표현 품질을 높이는 것을 목표로 한다.

## 현재 구현

- 손상 생성기 `M`이 잔차 `r = M(x)`를 생성하고, ViT 인코더·디코더가 `x - r`에서 원본 이미지를 복원한다. 생성기는 ViT, CNN, Hybrid 중 선택한다.
- I-step은 생성기를 고정하고 복원 모델을 학습한다. M-step은 복원 모델의 파라미터를 고정하고 생성기를 학습한다.
- 현재 손실은 복원 오차, 목표 복원 난이도와의 절댓값 차이, 목표 평균 절대 잔차와의 차이, 음의 KL 규제 항으로 구성된다. 표현 일치 손실은 포함되지 않는다.
- linear probe, fine-tuning, k-NN 평가 경로가 있다. 평가에 사용하는 `representation`은 인코더 토큰 평균을 projection MLP에 통과시킨 값이다.

## 구조

| 경로 | 내용 |
| --- | --- |
| [model/](model/) | DGNet 및 ViT·CNN·Hybrid 블록. `model_masking.py`는 초기 구현 |
| [dataset/](dataset/) | 데이터 설정과 로더. SSL 모드는 STL10만 지원 |
| [loss/](loss/) | 손실 항과 계수 스케줄 |
| [engine/](engine/) | 사전학습·평가 루프와 공통 실행 설정 |
| [26_06_04_exp_scripts/](26_06_04_exp_scripts/) | KL 규제 계수 비교 실험 구성 |
| [26_06_05_exp_scripts/](26_06_05_exp_scripts/) | budget·규제 항 제거 실험 구성 |
| [requirements.txt](requirements.txt) | CUDA 12.1용 PyTorch 2.2.2 등을 지정한 의존성 목록 |

## 사용법 안내

훈련·평가 진입 파일은 `engine/dg_train.py`, `engine/linear_probe_train.py`, `engine/finetune_train.py`, `engine/knn_eval.py`다. 데이터 위치는 `--data-path`, 출력 위치는 `--output-root`, 전이 평가에 사용할 사전학습 체크포인트는 `--checkpoint`로 지정한다.

[Engine API](engine/api.md)의 실행 예시는 `engine/`을 작업 디렉터리로 하는 상대 경로를 사용한다. 데이터 준비와 설정은 [Dataset API](dataset/api.md), 모델 생성·저장은 [Model API](model/api.md), 손실 설정은 [Loss API](loss/api.md)를 참고한다. 현재 목적함수의 절댓값 기반 설명은 [DGNet Loss Objective](26_06_04_exp_scripts/dg_net_goal.md)에 정리되어 있다.

## 현재 결과

학습·전이 평가 코드와 비교 실험 설정이 포함되어 있다. 남아 있는 `engine/output_smoke/` 자료는 데이터 로더 보고서다. 표현 성능 개선이나 비교 방법 대비 우월성을 확인할 학습·평가 결과는 제공본에서 확인되지 않았다. 실행 환경별 재현 여부와 표현 성능 개선은 별도 검증이 필요하다.

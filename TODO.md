# 실험 리스트
해당 프레임워크에선 DGnet모델의 적대적 복원 학습을 통해서 인코더를 훈련시켜서 제공해야한다.

ADIOS논문 재현을 위해서 epoch=200을 맞추고 나머지 하이퍼 파라미터도 맞추기전에, 기본적으로 collpase와 실제로 훈련이 제대로 되고있는지 검증하기 위해서 다른 프레임워크 없이 dg_net 프레임워크만 소규모로 작게 돌려볼것이다.

# environment
environment: pretext_task_gen
python: 3.10.20
torch: 2.2.2+cu121
torchvision: 0.17.2+cu121
cuda_available: True


## 해야할일
각 실험에 대한 배시스크립트를 정의해야하고, experiment.md에 테이블로 각 베시스크립트의 실험정보에 대해서 테이블로 설명해야한다.
각 배시 스크립트는 아래와같다.
`hybrid_dgnet_trian.bash`
`vit_dgnet_trian.bash`
`cnn_dgnet_trian.bash`
`experiment.md`

그리고 현재 하이퍼 파라미터를 못찾고 있어서 너가 추천해줘야한다. 
그냥 너가 하이퍼파라미터를 잘 채워넣으면된다.
아키텍쳐에서 중요한 요구사항은 네트워크 크기 비율이 `dg_net : encoder : decoder = 1 : 3 : 1` 정도의 크기가 됐으면한다는 것이다. 결국엔 이 프로젝트에서 중요한건 인코더를 훈련 잘시키는것이기 때문이다.

# 참고할 api 파일 모음
1. `api/engine_api.md` 이 경로에 실제 훈련 루프를 정의한 스크립트들에 대한 사용 설명서가 있다.
2. `dg_net/engine/train_config.py` 여기에 실제 argparse를 통해서 어떤 하이퍼 파라미터를 지정할수있는지 볼 수 있다.
3. `api/` 디렉토리에서 필요하면 내부를 읽어보고 필요한 Config에 대한 정보를 가져가라.

#  실험리스트
- case1. baseline 훈련 : 목표 - 동일조건 MAE, DINO보다 우수해야함, dg_model은 hybrid로해야함
- case2. dg_model에 대한 비교실험, case1의 best와 모든 하이퍼 파라미터 동일로 맞추기, dg_model의 아키텍쳐만 변경
    - 2.1 ViT
    - 2.2 CNN
-> 총 3개의 케이스 훈련

각 케이스는 linearProbe까지 전이 학습을 마쳐야한다. 즉 각 스크립트는 pretrain을 하고, pretrain결과를 토대로 다시 전이학습하고 그 결과까지 반환해얀한다.



## 훈련세팅
**output 디렉토리 경로 `dg_net/output`** 

### 데이터셋 
- 배치사이즈 : 16
- epochs : 10 (SSL, linearProbe 둘 다)

### DGnet 아키텍쳐
- encoder : ViT
    - 블럭수 :
    - n_dim :
    - ...
    
- decoder : ViT
    - 블럭수 :
    - n_dim : 
    - ...

- dg_model : Hybrid
    - 블럭수 :
    - n_dim : 
    - ...

### 옵티마이저 하이퍼 파라미터
모든 스케듈러는 "끈다"

| 필드 | 기본값 | 의미 |
| --- | ---: | --- |
| `alpha_inference` | `1.0` | inference encoder/decoder가 최소화하는 `D_rec` 계수 |
| `alpha_target` | `1.0` | degradation model의 목표 난이도 matching 항 계수 |
| `lambda_budget` | `1.0` | residual 평균 크기를 `beta_mask`로 맞추는 항 계수 |
| `lambda_reg` | `0.1` | uniform brightness-damage collapse를 막는 KL 계수; loss에는 음수 부호로 들어간다 |
| `tau_deg` | `0.5` | degradation model이 만들 목표 reconstruction error |
| `beta_mask` | `0.5` | `mean(abs(M(x)))`의 목표 damage budget |
| `eps` | `1e-8` | zero-initialized residual에서도 KL을 finite하게 유지하는 안정화 값 |
| `reduction` | `"mean"` | reconstruction/KL scalar reduction: `"mean"` 또는 `"sum"` |


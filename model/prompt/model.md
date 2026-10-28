# 해야할일
아래의 3개의 스크립트를 구현해야한다.
1. `cnn.py`
2. `vit.py`
3. `hybrid.py`

## 참고 실험사항

### DG_net의 구성요소
1. I모델 AutoEncoder (encoder : Vit - decoder : Vit ,고정)
2. M모델 degradation generator : **아키텍쳐 실험요소**
    1. baseline - vit 모델
    2. cnn
    3. hybrid - 우리 실험의 노벨티


# 구현조건
해당 모듈들은 **독립**모듈이여야 한다. 외부 fan-in 없이 단일스크립트로 완성 가능해야한다. 해당 모듈들에서는 딥러닝 네트워크 전체가 아니라, 딥러닝 네트워크의 구성요소인 **블럭단위** 구조를 정의하는 파트이다. 여기서 정의된 블럭에 따라서 `dg_net.py`에서 정해진 아키텍쳐에서 원하는 파트만 cnn, vit, hybrid중 골라서 교체해가며 실험 할 것이다.

그리고 각각 모델에 대한 생성자와 생성 코드 예제 및 주석을 통해서 모델 생성 test가 가능한 구조여야한다. 

검증요소는 아래와 같다.
1. 파라미터 개수, 모델크기 출력
2. 블럭및 내부구조 출력

이 단계에선 **config**를 통한 제어를 하지 않는다. 

### cnn.py
- resnet구조와 conxnet구조 호출가능해야함
- SSL훈련을 위해서 pretrain되지 않은 모델블럭 호출
+ `output/cnn.log`에 정상 작동 로그 출력

### vit.py
- vision transformer블럭을 호출해야함.
- SSL훈련을 위해서 pretrain되지 않은 모델블럭 호출
- MAE논문 구현 코드의 Vit 블럭을 따라서 맞춰야함
- 인코더와 디코더는 ViT만 사용할것임으로 DINO, MAE, ADIOS논문과 비교해야할 baseline 모델이다. 재현가능성과 다른 훈련 프레임워크와의 통일성에 포커스를 줘야함
+ `output/vit.log`에 정상 작동 로그 출력

### hybrid.py
- 내가 정의한 노벨티를 위한 아키텍쳐
- DG_net에서 M모델 (degradation generator)용으로 고안한 모델
+ `output/hybrid.log`에 정상 작동 로그 출력

#### 아키텍쳐
| 순서 | 구성 | 설명 |
| --- | --- | --- |
| 1 | LayerNorm | pre-norm |
| 2 | MHSA | global relation |
| 3 | Residual | x + attn |
| 4 | LayerNorm |  |
| 5 | CNN Block | local degradation |
| 6 | Residual | x + cnn |
| 7 | LayerNorm |  |
| 8 | FFN | channel mixing |
| 9 | Residual | x + ffn |

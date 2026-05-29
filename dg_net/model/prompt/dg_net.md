# 훈련 목표사항
`train_DGnet.md` 
해당모델은 적대적 학습모델 네트워크다. 복원 모델 I와 Degradation 모델 M간의 적대적 학습을 통해서
1. I 모델은 이미지의 중요한 특징을 이해하는 encoder representation 성능향상이 목표
2. M 모델은 I모델이 이미지 특징을 잘 훈련하기 위한 질좋은 문제를 생성해줘야한다. 

## 요구사항

1. M모델의 마지막 FC층의 zeroinit을 통한 identical 생성 init
2. 패치별 복원 가능하게 하는 ViT 인코더-디코더
3. Config기반으로 모델 생성및 제어
4. 모델의 save, load 기능
    - save : 모델의 가중치 파일 *.pt파일을 config.json과 함께 save_path에 디렉토리로 저장
    - load : 모델의 config.json를 통해서 아키텍쳐 읽어와서 생성하고, *.pt의 가중치를 load하기
5. 모델 로드 성공 출력기능
    - 각 아키텍쳐 구조 보고
    - encoder, decoder, degradtation net의 파라미터 개수 및 크기 비교출력
    - 데이터로더 한 배치에 대해서 sample 추론 결과를 이미지로 출력해주는 디버깅용 유틸함수.
        - 원본 입력 이미지
        - degradtation net 출력결과
        - decoder 출력결과

요구사항의 테스트가 `output/`에 남아있어야한다. 개발자가 그 디렉토리 결과를 보고 본인이 원한 결과가 맞는지에 대해서 판단할것이다.
`model_masking.py`는 예전에 구현했던 코드이다. 참고하면 도움이 될것이다.

### Fan-out, Fan-in 규칙

아래 3개 모듈만 fan-in가능하다. 
패키지 단위로써는 완전 독립적이여야한다.
1. `cnn.py`
2. `vit.py`
3. `hybrid.py`

DgNetConfig를 통해서 각 모듈의 블럭을 가져와서 생성해서 현재 `dg_model.py`에서는 조립을 하는 형태이다.


### Config 

아래 예시를 똑같이 따를 필요는 없다. 하지만 아키텍쳐 생성이 DgNetConfig 만으로 생성돼야한다.
ENCODER_ARCHITECT, DECODER_ARCHITECT, DG_ARCHITECT를 지정하고 각각의 모델 크기 및 구조를 다 DgNetConfig만으로 제어해야한다.

```python
@dataclass(frozen=True)
class DgNetConfig(BaseConfig):
    # ---- input spec ----
    IMG_SIZE: int = 224
    PATCH_SIZE: int = 16
    IN_CHANS: int = 3
    # ---- architecture ----
    EMBED_DIM: int = 512
    DEPTH: int = 6
    NUM_HEADS: int = 8
    MLP_RATIO: float = 4.0
    ENCODER_ARCHITECT = "VIT"
    DECODER_ARCHITECT = "VIT"
    DG_ARCHITECT = "VIT"
    ...
    # ---- positional embedding ----
    USE_POS_EMBED: bool = True
    POS_EMBED_INIT_ZERO: bool = True

    # ---- output control ----
    CLAMP_OUTPUT: bool = False

    # ---- optional advanced (확장 대비) ----
    ATTN_DROPOUT: float = 0.0
    PROJ_DROPOUT: float = 0.0
    CNN_KERNEL_SIZE: int = 3
    CNN_DROPOUT: float = 0.0
    FFN_DROPOUT: float = 0.0
```


### 상세설명
원본이미지 x : shape[batch_size, 3, 224, 224]
M모델출력 m : shape[batch_size, 3, 224, 224]
훼손이미지 생성 : dg = x - m 
중요한건 M모델출력이 처음엔 zero init이여야한다. 그럼으로 처음엔 dg(degraded)이미지 결과가 원본이미지와 동일 (identical이여야한다.)

또한 우리는 인코더와 디코더는 ViT모델로 고정할것이다.
그 이유는 패치별 prediction이라는 그 특징이 가장 강력한 특징이기 때문이다. 

#### 텐서 충간출력 로그 예시
''' 
pred     :  torch.Size([batch_size, 196, 768])
z_x      :  torch.Size([batch_size, 768])
z_mask   :  torch.Size([batch_size, 768])
z_x_dg   :  torch.Size([batch_size, 768])
x_dg     :  torch.Size([batch_size, 3, 224, 224])
x        :  torch.Size([batch_size, 3, 224, 224])
'''

### 


# 전체 네트워크 구성


## 모델의정의
- M : degradation network   Hybrid Transformer - 우리 훈련목표
- I : inference network     Encoder-Decoder (AutoEncoder)
    - I_{Encoder}       : tiny encoder (보통 이게 훈련목표)
    - I_{Decoder}       : pixel-wise reconstruction network
    - I_{prejection}    : subspace projection linear network

## 출력 값 정의
x : 이미지원본
m : x - M(x)
z : I_{prejection}(I_{Encoder}(x))
z^ : I_{prejection}(I_{Encoder}(M(x)))
x_{recon} : I_{Decoder}(I_{Encoder}(x))
x^__{recon} : I_{Decoder}(I_{Encoder}(M(x)))



## Loss
### 1. Inference Network의 목표
Inference Network(AE)는  
원본 입력과 degradation 입력이 **가능한 한 같은 표현과 같은 복원 결과**를 내도록 학습된다.

---
### 2. Masking / Degradation Network의 목표
Degradation Network(M)는  
입력을 완전히 붕괴시키는 것이 아니라,  
**제한된 범위 안에서 AE가 풀기 어려운 입력**을 생성하도록 학습된다.



## 파라미터의 크기
M:1,  I_{Encoder}:1,  I_{Decoder}:0.2~5 , I_{prejection} 은 아주 얕은 projection층

## Degradation 모델의 아키텍쳐

## Custom. Hybrid-Transformer layer

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


### degradtion 전체 아키텍쳐
~~~
patchify → [196,768]

→ linear proj → [196,D]

→ Hybrid Block × L (L=4~8)

→ residual head → [196,768]

→ out = input + delta
~~~

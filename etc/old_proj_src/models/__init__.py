''' 

## 현재 model 패키지 구성
- model/
        - base_model.py
        - auto_encoder
            model_auto_encoder.py   # MAE 논문 기반의 auto_encoder
            pos_embed.py            # 포지셔널 임베딩위한 모듈
        - degradation               # hybridTransformer 아키텍쳐 - 훈련 목표
            degradatinNet.py
        - DGnet.py                  # degradation, encoder, decoder, projection 층을 갖는 전체 네트워크

        
## 모든 model객체에 대해서 해야할일

### model 패키지 리펙토링 해야할일
1. model 패키지안에 있는 이 프레임워크에 대한 모델에 대한 최상위 부모모델 객체 정의 - model
2. 모든 config에 대해서 최상위 부모 데이터 클래쓰 객체정의 (내 프레임워크의 모든 모델은 본인만을위한 config가있따.) - model_config
3. 모든 모델의 모델 save, load, build를 만들어야한다. 
    - 모든 모델의 본인의 conifg로 빌드가 돼야한다.
    - save(save_path) : save_path에 모델의 *.pt와 모델의 config를 저장한다. save_path는 디렉토리고, 안에는 *pt, config.json이 함께있다.
    - save(checkpoint_path) : checkpoint_path에서 config파일을 읽고 모델을 빌드한다. 그리고 내부 가중치를 *.pt로 채운다.

## 저장형식 예시
save_dir/
    - model_config.json
    - dg_learner.pt

## 배경
우리는 프레임워크상 dg_learner로 불러올수도 있지만, auto_encoder, degradation 등등을 따로 떼서 저장하고 불러올일이 많을것이다. 
그래서 이런 아키텍쳐가 필요할듯하다.

## 현재 코드
### degradation_learner.py


degradation network   Hybrid Transformer - 우리 훈련목표
inference network     Encoder-Decoder (AutoEncoder)
    - I_{Encoder}       : tiny encoder (보통 이게 훈련목표)
    - I_{Decoder}       : pixel-wise reconstruction network
    - I_{prejection}    : subspace projection linear network

아키텍쳐 상세.

기본적인 요구사항
1. forward() (필수)
2. sample_show() (필수)
    I_{i} in I model에 대해서, 각각의 입력에 대한 텐서값 출력, 시각화 결과를 봐야함.
    I, M 에대해서도 같은 동작 필요함
    DGnet : I&M에 대해서도 전체 같은동작 필요

### 입출력 변수 정리
x -> degradationNet -> m
x -> encoder -> q
m -> encoder -> q^
q -> decoder -> x_r
q^ -> decoder -> x^_r
q -> prejection -> z_r
q^ -> prejection -> z^_r

### 입출력 텐서 형태 정리
x : [B, W, H, C]
m : [B, W, h, C]
q : [B, 1024 or etc]
z : [B, 512 or etc]


'''



''' 

DGnet.inference.sample_show()
    ## 총 4개의 결과
    - 1. x -> encoder-decoder.show()
    - 2. x -> encoder-prejection.show()
    - 3. x -> degradationNet-encoder-decoder.show()
    - 4. x -> degradationNet-encoder-prejection.show()

-> 아래의 하위 4개 멤서드 사용
- DGnet.degradationNet.sample_show()    
- DGnet.inference.encoder.sample_show()
- DGnet.inference.decoder.sample_show()
- DGnet.inference.prejection.sample_show()


'''
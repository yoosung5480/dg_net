'''
## 프로젝트 디렉토리
~~~python
    loss/
        loss_config.py
        DG_learn_loss.py        # degradation_loss, ae_loss 객체 2개를 LossConfig를 통해서 정의
        distance/
            - total_distrance.py
            - representation_distance.py
            - degradation_distance.py
            - reconstruction_distance.py
        regluation/
            - degradation_regluation.py
            - representation_regluation.py
        loss_test.ipynb             # 지금 너가 구현한 코드 테스트용 주피터 노트북
~~~

## 패키지간 규약
모든 loss는 배치기준으로 처리된다. 
loss_config.py 안에는 LossConfig, DistanceConfig, RegluationConfig 가 포함돼있다.
각각 하이퍼 파라미터가 앞에 달려있는 상수 alpha, beta, lambda가 있다. 그런데 해당 수치가 0.0일경우는 아예 동작하지 않도록 해야한다.
어떤 하이퍼 파라미터를 얼마나 쓰고, 또 쓸지 말지 모두다 실험의 영역이기 때문이다.


## 모델의정의
- M : degradation network   Hybrid Transformer - 우리 훈련목표
- I : inference network     Encoder-Decoder (AutoEncoder)
    - I_{Encoder}       : tiny encoder (보통 이게 훈련목표)
    - I_{Decoder}       : pixel-wise reconstruction network
    - I_{prejection}    : subspace projection linear network

## 출력 값 정의
x : 이미지원본                                     # [B, 3, 224, 224]
m : M(x)                                        # [B, 3, 224, 224]
z : I_{prejection}(I_{Encoder}(x))              # 
z^ : I_{prejection}(I_{Encoder}(M(x)))          #
x_{recon} : I_{Decoder}(I_{Encoder}(x))         #

## 실제 DegradationLearnerNet 출력값 (patch_size=16, patch_num=14 사실상 고정임)
(x_{recon})pred     :  torch.Size([batch_size, patch_num*patch_num, 3*patch_size*patch_size])
(z)z_x      :  torch.Size([batch_size, n_dim])
(z^)z_x_dg   :  torch.Size([batch_size, n_dim])
(m)x_dg     :  torch.Size([batch_size, 3, 224, 224])
(x)x        :  torch.Size([batch_size, 3, 224, 224])


## Distance 거리의 정의 2개.
1. D_{representaion} = 1-cosine_similarity(z, z^)
2. D_{pixel} = (x_{recon} - x)^2
4. D_{mask} = ((x - t ) - m)^2               # threshold 있음.

### about penalty..
M : identical network로 init돼서 점점 degradation강도를 적대적으로 높일것이다. 완전노이즈나 blank가 안나오게 규제해야한다. 현재 penalty는 Adios논문을 그대로 적용했다. 이에 의해서 내부 pixel sum이 원본과 너무 같아도, 너무 달라도 차이가 나게 했다. 하지만 단순 summation이라서 나는 내부원소 차로 했다. 왜냐하면 지금 논문의 단순 pixel-wise 합은 서로 상쇄되는값에 의한 0or1 수렴 구분이 안된다. (왜 논문 모델이 성적냈는지 모르겠네)

## Loss
Loss_{I} : (minimize distance)
    = alpha_r * D_{r} + alpha_p * D_{p}

Loss_{M} : (maximize distance)
     = -(beta_r * D_{r} + beta_p * D_{p} + lambda_{deg} * R_{deg}) 
        + beta_m * D_{mask} + lambda_{rep} * R_{rep}

## Regulation
R_{deg} = lambda_{tv} * TotalVariation(x, m) + lambda_{sparsity} * sparsity(x, m)       # 마스캉과 원본의 픽셀단위 제약, 자체로써 opt 목표
R_{rep} = ReLU(D_{r} - t) * exp(k * (D_{r} - t))                                        # representaion 결과에 대한 유사도 제약, t넘으면 큰 패널티 +
                                          # threshold 기반 마스킹제약. 
 


'''
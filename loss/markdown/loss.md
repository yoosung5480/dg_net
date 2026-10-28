# 해야할일
1. loss.py를 구현한다.
2. loss_config.py를 구현한다.
3. api.md 를 작성한다.

# 프로젝트 목표
`train_DGnet.md`를 읽고 프로젝트 목표에 대해서 이해한다.

## 프로젝트 환경
environment: pretext_task_gen
python: 3.10.20
torch: 2.2.2+cu121
torchvision: 0.17.2+cu121
cuda_available: True


## 전체 프로젝트 디렉토리
```text
root/
    src/
        - model/
            - output/           
                *.log           # 실행결과에 대한 로그파일, 개발자가 보고 요구사항 정합성 검사
            - cnn.py            # cnn 기반 pretrain 모델과 vanila 모델 (ResNet, ConvNeXt)
            - vit.py            # Vit 기반 pretrain 모델과 vanila 모델 (ViT, Swin)
            - hybrid.py         # cnn + vit 기반 특수 아키텍쳐 모델 
            - dg_net_model.py   # Hybrid 모델 기반 + zero initialize 기능추가
            - base_model.py     # Vit only기반 + zero initialize 없는 바닐라 모델
        - loss/  <- (현재 root)               ## TODO ##
            - output/           
                *.log           # 실행결과에 대한 로그파일, 개발자가 보고 요구사항 정합성 검사
            - loss.py           # 하이퍼 파리미터값만으로 단항식 추가 또는 제거 또는 스케듈러 사용여부 제어가 가능한 형태로 optimizer를 생성할수 있어야함
            - loss_config.py
        - dataset/
            - output/           
                *.log                   # 실행결과에 대한 로그파일, 개발자가 보고 요구사항 정합성 검사
            - pretrain_datasets.py      # self supervised learning을 위한 형태의 데이터셋 로드
            - cls_datasets.py           # tranfer task, 분류 데이터셋용 데이터로더 및 데이터셋 정의
        engine/
            - pretrain_train.py          # SSL pretraining
            - classification_train.py    # supervised classification: scratch, linear probe, fine-tuning 전부 담당
            - knn_eval.py                # k-NN은 학습이라기보다 평가라서 train보다 eval이 맞음
            - evaluate.py                # 공통 metric/eval loop, 선택사항
        - util/ 
            - # 근데 이건 나중에 생각하자. 구현 목표에만 집중                                  
        test_train.py
        classification_main.py
```

## 사용가능 api
`model_api.md`, `dataset.md`에는 각각 모델 아키텍쳐 로드와 데이터셋로더 호출 api가 있다. 해당 api를 사용해서 loss.py 스크립트 검증을 위한 세팅으로 활용한다.

## loss.py

### Inference model loss

$$
\boxed{
\mathcal{L}_{I}
=
\left\|
I_{dec}(I_{enc}(x - M(x))) - x
\right\|_2^2
}
$$



### Degradation model loss

$$
\boxed{
\mathcal{L}_{M}
=
D_{target}(I, M)
+
\lambda_{budget}
D_{budget}(M)
-
\lambda_{reg}
D_{KL}(P(x) \| U)
}
$$

이를 풀어 쓰면 다음과 같다.

$$
\boxed{
\mathcal{L}_{M}
=
\left(
\left\|
I_{dec}(I_{enc}(x - M(x))) - x
\right\|_2^2
-
\tau_{deg}
\right)^2
+
\lambda_{budget}
\left|
\mathrm{mean}(|M(x)|) - \beta_{mask}
\right|
-
\lambda_{reg}
\sum_{i,j}
P_{ij}(x)
\log
\frac{
P_{ij}(x)
}{
U_{ij}
}
}
$$

## 요구사항

1. 모든 단항식은 따로 함수나 객체로 묶어서 개발자가 보고 이해할수잇게 가독성있게 해야한다.
2. 모든 단항식에는 제어를 위한 하이퍼 파라미터가 앞에 붙어야한다. 예를들어 `D_{target}`에도 `alpha_{target} = 1.0`이라는 하이퍼 파라미터가 붙어있다 이해해야한다.
3. 모든 단항식 앞의 하이퍼파라미터는 `0.0`으로 지정시 **그래디언트가 아예 안흐르게 꺼져야한다**
4. 모든 단항식 앞의 하이퍼파라미터는 스케쥴링 방식을 on-off로 지원한다. 스케쥴링 방식은 `linear`, `cosine`중에서 선택할수 있어야한다.
5. 위와같은 모든 하이퍼 파라미터는 `LossConfig`객체만을 통해서 제어되고 생성된다. 또한 LossConfig의 각 필드에 대해서 개발자가 이해할수있는 아주 상세한 주석이 필요하다.
```
loss_cfg = LossConfig()
loss = prepare_dgloss(loss_cfg) # 훈련시 사용할 optimizer
```
6. `output/`디렉토리에 검증사항이 있어야한다. 검증사항은 아래와같다.
    1. `model_api.md`, `dataset.md`를 통해서 기본세팅, `batch_size=4`, `mode='ssl`로한다. 기본적으로 step갯수를 100개로 하드코딩해서 코드가 딱 실행됏을때의 샘플이미지 결고와, 100번 뒤의 샘플이미지 결과를 출력해서 각각 저장한다.
    2. loss 객체를 각각 Config 하이퍼파라미터로 제어가능한지, 단항식 끄기-켜기, 스케쥴러적용 여부 검증해서 *.log로 남겨놓는다.
7. 바로 사용자가 갖다 붙여쓸수있는 참고용 api모음집과 케이스별 사례를 준비한 `api.md`를 작성한다.

## 구현규칙
loss.py와 loss_conifg.py는 외부 의존성없이 완벽하게 작동하는 독립모듈이여야한다. 해당 패키지의 fan_out은 `engine`패키지에만 적용된다.
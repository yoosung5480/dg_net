# DGNet Experiment Overview: Adversarial Problem Generator for Self-Supervised Learning

## 1. 연구 목표

본 실험의 핵심 목표는 SSL(Self-Supervised Learning)에서 사람이 고정적으로 설계한 augmentation이나 masking rule 대신, **적대적으로 학습되는 문제생성기(problem generator)** 를 도입하는 것이다.

기존 SSL에서는 보통 다음과 같은 방식으로 pretext problem을 만든다.

$$
\text{random crop}
$$

$$
\text{color jitter}
$$

$$
\text{random masking}
$$

$$
\text{block masking}
$$

$$
\text{fixed ratio masking}
$$

이 방식들은 대부분 사람이 미리 정한 변환 규칙에 기반한다. 반면 본 실험에서는 Degradation Network인 DGNet이 입력 이미지마다 직접 degradation residual을 생성한다.

$$
r = DGNet(x)
$$

$$
x_{\text{dg}} = x - r
$$

즉 DGNet은 단순한 noise generator가 아니라, SSL 학습자가 풀어야 할 문제를 생성하는 **adversarial problem generator**이다.

본 연구의 중심 가설은 다음과 같다.

$$
\boxed{
\text{좋은 SSL은 좋은 learner만의 문제가 아니라, 좋은 problem generator의 문제이기도 하다.}
}
$$

---

## 2. 연구 의의와 Novelty

본 실험의 novelty는 크게 세 가지 축으로 정리된다.

### 2.1 Adversarial Problem Generator for SSL

기존 SSL은 사람이 정의한 augmentation policy나 random masking rule을 사용한다. 하지만 본 연구에서는 DGNet이 입력 이미지와 현재 Inference model의 상태를 바탕으로, 학습에 유용한 degraded view를 생성한다.

즉 고정된 view 생성이 아니라, 학습 가능한 문제생성기를 도입한다.

$$
DGNet = \text{learnable SSL problem generator}
$$

DGNet이 좋은 문제를 만든다면, 작은 데이터셋에서도 더 다양하고 질 좋은 SSL pretext problem을 생성할 수 있다.

가설은 다음과 같다.

$$
\boxed{
\text{좋은 문제생성기는 작은 데이터셋에서도 효과적인 SSL augmentation source가 될 수 있다.}
}
$$

작은 데이터셋에서는 데이터 수 자체가 적기 때문에 view diversity가 부족하다.

$$
|\mathcal{D}| \text{ small}
\Rightarrow
\text{limited view diversity}
$$

DGNet은 입력 이미지별로 다른 degradation을 생성함으로써, 작은 데이터셋에서도 더 많은 informative view를 만들 수 있다.

---

### 2.2 Pixel Reconstruction vs Representation vs Hybrid Adversarial Signal

기존 adversarial SSL 계열은 보통 representation space에서 적대적 학습을 수행한다.

예를 들어 ADIOS는 다음과 같은 구조를 갖는다.

$$
\min_I \max_M
D_{\text{rep}}(I(x), I(x_m))
$$

여기서 mask generator는 representation을 흔드는 view를 만든다.

반면 본 실험은 pixel reconstruction 기반 adversarial problem generation을 중심에 둔다.

$$
x_{\text{dg}} = x - DGNet(x)
$$

$$
\hat{x}_{\text{dg}} = I(x_{\text{dg}})
$$

$$
\hat{x}_{\text{dg}} \rightarrow x
$$

즉 DGNet은 복원하기 적절히 어려운 이미지를 생성하고, Inference model은 그 이미지를 복원하면서 representation을 학습한다.

비교할 학습 신호는 세 가지이다.

1. Representation only
2. Reconstruction only
3. Representation + Reconstruction hybrid

이 비교를 통해 다음 질문을 검증한다.

$$
\boxed{
\text{SSL용 문제생성기는 어떤 adversarial signal로 훈련해야 가장 효과적인가?}
}
$$

---

### 2.3 Problem Generator-specific Hybrid Architecture

기존 mask generator 또는 degradation generator는 보통 shallow CNN, plain CNN, 또는 ViT 구조를 사용한다.

본 실험에서는 문제생성기라는 역할에 맞춘 특수 Hybrid Transformer 구조를 제안한다.

DGNet은 일반적인 classifier도 아니고, autoencoder도 아니다. DGNet은 다음 능력을 동시에 가져야 한다.

$$
\text{local texture / edge / low-level corruption control}
$$

$$
\text{global semantic / object-level importance estimation}
$$

CNN은 local pattern에 강하고, ViT는 global relation에 강하다. 따라서 DGNet은 CNN과 ViT의 장점을 결합한 hybrid architecture가 적합하다는 가설을 둔다.

$$
\boxed{
\text{Hybrid DGNet can produce higher-quality adversarial SSL problems than shallow CNN or plain ViT generators.}
}
$$

---

## 3. 전체 모델 구조

전체 모델은 두 개의 주요 네트워크로 구성된다.

- $I$: Inference Network / AutoEncoder
- $M$: Degradation Network / DGNet

기본 입력은 다음과 같다.

$$
x \in \mathbb{R}^{B \times C \times H \times W}
$$

DGNet은 degradation residual을 생성한다.

$$
r = M(x) = DGNet(x)
$$

Degraded image는 다음과 같이 생성된다.

$$
x_{\text{dg}} = x - r
$$

Inference model은 degraded image를 입력받아 원본 이미지를 복원한다.

$$
\hat{x}_{\text{dg}} = I_{\text{dec}}(I_{\text{enc}}(x_{\text{dg}}))
$$

원본과 degraded image의 representation은 다음과 같다.

$$
z_x = I_{\text{proj}}(I_{\text{enc}}(x))
$$

$$
z_{\text{dg}} = I_{\text{proj}}(I_{\text{enc}}(x_{\text{dg}}))
$$

최소 forward 출력은 다음을 포함해야 한다.

```text
x             : original image
r             : degradation residual
x_dg          : degraded image
z_x           : original representation
z_dg          : degraded representation
x_recon       : reconstruction from original image
x_dg_recon    : reconstruction from degraded image
```

---

## 4. 모델별 역할

### 4.1 Inference Network \(I\)

Inference Network는 문제를 푸는 모델이다.

구성은 다음과 같다.

- $I_{\text{enc}}$: image encoder
- $I_{\text{dec}}$: pixel-wise reconstruction decoder
- $I_{\text{proj}}$: representation projection head

Inference Network의 목표는 다음과 같다.

1. degraded image에서 원본 이미지를 복원한다.
2. 원본 image와 degraded image의 representation을 가깝게 만든다.
3. 최종적으로 encoder가 이미지의 중요한 특징을 이해하도록 학습한다.

즉 $I$는 단순 복원 모델이 아니라 SSL representation learner이다.

---

### 4.2 Degradation Network \(M\) / DGNet

DGNet은 문제를 생성하는 모델이다.

DGNet은 입력을 완전히 붕괴시키면 안 된다. 대신 제한된 범위 안에서 Inference Network가 풀기 어려운 문제를 생성해야 한다.

DGNet의 목표는 다음과 같다.

1. 너무 쉬운 문제를 만들지 않는다.
2. 너무 불가능한 문제를 만들지 않는다.
3. 원본 이미지의 중요한 정보를 적절히 제거한다.
4. 작은 데이터셋에서도 질 좋은 SSL view를 생성한다.

DGNet은 다음과 같은 문제생성기로 해석된다.

$$
\boxed{
M = \text{adversarial SSL problem generator}
}
$$

---

## 5. 최종 목표함수 개요

본 프레임워크는 모든 단항식 앞에 독립 계수를 둔다. 계수를 $0$으로 두면 해당 항을 끌 수 있다.

전체 목표함수는 다음 세 실험군을 모두 표현할 수 있어야 한다.

1. representation only
2. reconstruction only
3. representation + reconstruction hybrid

---

## 6. 공통 정의

기본 정의는 다음과 같다.

$$
x \in \mathbb{R}^{B \times C \times H \times W}
$$

$$
r = DGNet(x)
$$

$$
x_{\text{dg}} = x - r
$$

$$
\hat{x}_{\text{dg}} = I_{\text{dec}}(I_{\text{enc}}(x_{\text{dg}}))
$$

$$
z_x = I_{\text{proj}}(I_{\text{enc}}(x))
$$

$$
z_{\text{dg}} = I_{\text{proj}}(I_{\text{enc}}(x_{\text{dg}}))
$$

---

## 7. Inference Network 목표함수

Inference Network는 reconstruction loss와 representation loss를 가질 수 있다.

$$
\mathcal{L}_{I}
=
\alpha_{\text{rec}}^{I}
D_{\text{rec}}^{I}
+
\alpha_{\text{rep}}^{I}
D_{\text{rep}}^{I}
$$

최적화 방향은 다음과 같다.

$$
\min_I \mathcal{L}_{I}
$$

---

### 7.1 Inference reconstruction loss

Inference reconstruction loss는 batch mean MSE를 사용한다.

$$
D_{\text{rec}}^{I}
=
\frac{1}{BCHW}
\sum_{b=1}^{B}
\sum_{c=1}^{C}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
(\hat{x}_{\text{dg},bchw}-x_{bchw})^2
$$

이는 다음과 동치다.

$$
D_{\text{rec}}^{I}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left[
\frac{1}{CHW}
\sum_{c,h,w}
(\hat{x}_{\text{dg},bchw}-x_{bchw})^2
\right]
$$

따라서 Inference Network 기준에서는 이미지별 MSE를 먼저 구한 뒤 batch 평균하든, 전체 $BCHW$에 대해 한 번에 mean을 하든 결과가 같다.

---

### 7.2 Inference representation loss

Representation distance는 원본 image와 degraded image의 representation 사이에서 계산한다.

$$
D_{\text{rep}}^{I}
=
\frac{1}{B}
\sum_{b=1}^{B}
D_{\text{rep}}^{(b)}
$$

cosine distance를 사용하면 다음과 같다.

$$
D_{\text{rep}}^{(b)}
=
1
-
\frac{
\langle z_{x,b}, z_{\text{dg},b} \rangle
}{
\|z_{x,b}\|_2 \|z_{\text{dg},b}\|_2 + \epsilon
}
$$

Inference Network는 representation distance를 줄인다.

$$
D_{\text{rep}}^{I} \downarrow
$$

의미는 다음과 같다.

$$
I_{\text{enc}}(x)
\approx
I_{\text{enc}}(x_{\text{dg}})
$$

즉 Inference Network는 degradation에도 변하지 않는 representation을 학습한다.

---

## 8. DGNet 목표함수

DGNet은 adversarial problem generator이다.

DGNet의 전체 목적함수는 다음과 같다.

$$
\mathcal{L}_{DG}
=
\alpha_{\text{target}}^{DG}
D_{\text{target}}^{DG}
+
\lambda_{\text{budget}}^{DG}
D_{\text{budget}}^{DG}
-
\lambda_{\text{KL}}^{DG}
D_{\text{KL}}^{DG}
-
\alpha_{\text{rep}}^{DG}
D_{\text{rep}}^{DG}
+
\lambda_{\text{sparse}}^{DG}
D_{\text{sparse}}^{DG}
$$

최적화 방향은 다음과 같다.

$$
\min_{DGNet}
\mathcal{L}_{DG}
$$

여기서 representation term 앞에 음수가 붙는 이유는 DGNet이 representation distance를 키워야 하기 때문이다.

$$
\min_{DGNet}
\left(
-
\alpha_{\text{rep}}^{DG}
D_{\text{rep}}^{DG}
\right)
\equiv
\max_{DGNet}
D_{\text{rep}}^{DG}
$$

---

### 8.1 DGNet reconstruction target distance

DGNet 기준 reconstruction distance는 이미지별로 계산한다.

$$
D_{\text{rec}}^{DG,(b)}
=
\frac{1}{CHW}
\sum_{c=1}^{C}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
(\hat{x}_{\text{dg},bchw}-x_{bchw})^2
$$

DGNet target distance는 각 이미지별 reconstruction distance와 target threshold 사이의 차이를 계산한 뒤 batch 평균한다.

$$
D_{\text{target}}^{DG}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left|
D_{\text{rec}}^{DG,(b)}
-
\tau_{\text{deg}}
\right|
$$

의미는 다음과 같다.

$$
D_{\text{rec}}^{DG,(b)}
\rightarrow
\tau_{\text{deg}}
$$

즉 DGNet은 reconstruction error를 무한히 키우는 것이 아니라, 각 이미지마다 정해진 난이도에 가까운 문제를 생성한다.

---

### 8.2 DGNet budget distance

DGNet이 생성한 residual energy는 이미지별로 계산한다.

$$
D_{\text{budget-energy}}^{DG,(b)}
=
\frac{1}{CHW}
\sum_{c=1}^{C}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
r_{bchw}^{2}
$$

DGNet budget distance는 다음과 같다.

$$
D_{\text{budget}}^{DG}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left|
D_{\text{budget-energy}}^{DG,(b)}
-
\tau_{\text{budget}}
\right|
$$

의미는 다음과 같다.

$$
D_{\text{budget-energy}}^{DG,(b)}
\rightarrow
\tau_{\text{budget}}
$$

즉 DGNet은 각 이미지마다 사용할 수 있는 degradation energy 총량을 제한받는다.

기존처럼 batch 전체를 먼저 평균내면 다음 문제가 생긴다.

$$
\left|
\operatorname{mean}_{b,c,h,w}(\cdot)
-
\tau
\right|
$$

이 경우 어떤 이미지는 많이 망가지고, 어떤 이미지는 거의 안 망가져도 batch 평균만 맞으면 된다.

최종 목표함수는 다음 구조를 사용한다.

$$
\operatorname{mean}_{b}
\left(
\left|
\operatorname{mean}_{c,h,w}(\cdot)
-
\tau
\right|
\right)
$$

즉 각 이미지마다 budget을 독립적으로 맞춰야 한다.

---

### 8.3 DGNet channel-wise spatial KL

DGNet damage KL은 residual의 channel 평균을 먼저 내지 않는다.

각 이미지 $b$, 각 채널 $c$마다 독립적인 spatial damage distribution을 만들고, 이를 uniform spatial distribution과 비교한다.

채널별 spatial damage map은 다음과 같다.

$$
d_{bc,hw}
=
|r_{bchw}|
$$

이를 spatial dimension에 대해 flatten한다.

$$
d_{bc}
\in
\mathbb{R}^{HW}
$$

채널별 damage probability는 다음과 같다.

$$
p_{bc,i}
=
\frac{d_{bc,i}}
{\sum_{j=1}^{HW} d_{bc,j} + \epsilon}
$$

uniform spatial probability는 다음과 같다.

$$
u_i
=
\frac{1}{HW}
$$

채널별 이미지별 KL divergence는 다음과 같다.

$$
D_{\text{KL}}^{(b,c)}
=
\sum_{i=1}^{HW}
p_{bc,i}
\log
\frac{p_{bc,i}}{u_i}
$$

DGNet 기준 channel-wise spatial KL은 이미지별, 채널별 KL divergence를 평균한 값이다.

$$
D_{\text{KL}}^{DG}
=
\frac{1}{BC}
\sum_{b=1}^{B}
\sum_{c=1}^{C}
D_{\text{KL}}^{(b,c)}
$$

DGNet loss에는 이 항이 음수로 들어간다.

$$
-
\lambda_{\text{KL}}^{DG}
D_{\text{KL}}^{DG}
$$

따라서 DGNet을 minimize하면 $D_{\text{KL}}^{DG}$는 커지는 방향으로 작동한다.

이 항의 의미는 uniform한 전체 밝기 감소 같은 손상을 막고, 특정 spatial region에 집중된 degradation을 유도하는 것이다.

다만 이 KL은 semantic mask를 직접 보장하지 않는다. 역할은 shortcut 방지용 보조항이다.

---

### 8.4 DGNet representation loss

Representation distance는 원본 image와 degraded image의 representation 사이에서 계산한다.

$$
D_{\text{rep}}^{DG}
=
\frac{1}{B}
\sum_{b=1}^{B}
D_{\text{rep}}^{(b)}
$$

cosine distance를 사용하면 다음과 같다.

$$
D_{\text{rep}}^{(b)}
=
1
-
\frac{
\langle z_{x,b}, z_{\text{dg},b} \rangle
}{
\|z_{x,b}\|_2 \|z_{\text{dg},b}\|_2 + \epsilon
}
$$

DGNet은 이 값을 키우는 방향으로 학습된다.

$$
D_{\text{rep}}^{DG} \uparrow
$$

하지만 전체 DGNet objective는 minimization 형태로 작성하기 때문에 loss에는 음수로 들어간다.

$$
-
\alpha_{\text{rep}}^{DG}
D_{\text{rep}}^{DG}
$$

이는 ADIOS 계열의 representation adversarial objective와 대응된다.

$$
\min_I D_{\text{rep}}
$$

$$
\max_{DGNet} D_{\text{rep}}
$$

---

### 8.5 DGNet sparsity penalty

ADIOS의 sparsity penalty는 mask가 all-zero 또는 all-one으로 붕괴되는 것을 막기 위한 항이다.

DGNet이 명시적인 mask $m$을 출력한다고 가정하면 이미지별 sparsity penalty는 다음과 같다.

$$
p^{(b)}
=
\sin
\left(
\frac{\pi}{HW}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
m_{bhw}
\right)^{-1}
$$

batch 평균 sparsity penalty는 다음과 같다.

$$
D_{\text{sparse}}^{DG}
=
\frac{1}{B}
\sum_{b=1}^{B}
p^{(b)}
$$

DGNet loss에는 다음과 같이 들어간다.

$$
+
\lambda_{\text{sparse}}^{DG}
D_{\text{sparse}}^{DG}
$$

다만 현재 DGNet이 명시적인 binary 또는 soft mask가 아니라 residual $r=DGNet(x)$만 생성하는 구조라면, 이 sparsity penalty는 직접 적용하지 않는다.

이 경우 다음과 같이 둔다.

$$
\lambda_{\text{sparse}}^{DG}=0
$$

---

## 9. 실험 설정별 objective

### 9.1 Representation only

이 설정은 ADIOS에 가장 가까운 baseline이다.

Inference Network objective는 다음과 같다.

$$
\mathcal{L}_{I}
=
\alpha_{\text{rep}}^{I}
D_{\text{rep}}^{I}
$$

DGNet objective는 다음과 같다.

$$
\mathcal{L}_{DG}
=
-
\alpha_{\text{rep}}^{DG}
D_{\text{rep}}^{DG}
+
\lambda_{\text{sparse}}^{DG}
D_{\text{sparse}}^{DG}
$$

목적은 representation-level adversarial masking 또는 degradation이 SSL 성능에 미치는 영향을 확인하는 것이다.

---

### 9.2 Reconstruction only

이 설정은 본 실험의 핵심 novelty에 가장 가깝다.

Inference Network objective는 다음과 같다.

$$
\mathcal{L}_{I}
=
\alpha_{\text{rec}}^{I}
D_{\text{rec}}^{I}
$$

DGNet objective는 다음과 같다.

$$
\mathcal{L}_{DG}
=
\alpha_{\text{target}}^{DG}
D_{\text{target}}^{DG}
+
\lambda_{\text{budget}}^{DG}
D_{\text{budget}}^{DG}
-
\lambda_{\text{KL}}^{DG}
D_{\text{KL}}^{DG}
$$

목적은 pixel-level adversarial reconstruction game이 SSL representation learning에 도움이 되는지 검증하는 것이다.

이 실험이 성공하면 다음 주장이 가능하다.

$$
\boxed{
\text{Pixel-level adversarial problem generation can be stabilized by image-wise difficulty and budget constraints.}
}
$$

---

### 9.3 Representation + Reconstruction hybrid

이 설정은 representation-level adversarial signal과 reconstruction-level adversarial signal을 함께 사용한다.

Inference Network objective는 다음과 같다.

$$
\mathcal{L}_{I}
=
\alpha_{\text{rec}}^{I}
D_{\text{rec}}^{I}
+
\alpha_{\text{rep}}^{I}
D_{\text{rep}}^{I}
$$

DGNet objective는 다음과 같다.

$$
\mathcal{L}_{DG}
=
\alpha_{\text{target}}^{DG}
D_{\text{target}}^{DG}
+
\lambda_{\text{budget}}^{DG}
D_{\text{budget}}^{DG}
-
\lambda_{\text{KL}}^{DG}
D_{\text{KL}}^{DG}
-
\alpha_{\text{rep}}^{DG}
D_{\text{rep}}^{DG}
+
\lambda_{\text{sparse}}^{DG}
D_{\text{sparse}}^{DG}
$$

목적은 representation 관점에서도 어렵고, pixel reconstruction 관점에서도 적절히 어려운 문제를 생성하는 hybrid problem generator가 더 좋은 SSL 성능을 내는지 확인하는 것이다.

---

## 10. DGNet 아키텍처 실험

DGNet은 문제생성기 전용 architecture를 사용한다.

비교할 DGNet architecture는 다음과 같다.

1. shallow CNN generator
2. plain ViT generator
3. custom Hybrid Transformer generator

DGNet Hybrid Block은 다음 구조를 따른다.

| 순서 | 구성 | 역할 |
| --- | --- | --- |
| 1 | LayerNorm | pre-norm |
| 2 | MHSA | global relation |
| 3 | Residual | x + attn |
| 4 | LayerNorm | pre-norm |
| 5 | CNN Block | local degradation |
| 6 | Residual | x + cnn |
| 7 | LayerNorm | pre-norm |
| 8 | FFN | channel mixing |
| 9 | Residual | x + ffn |

DGNet 전체 구조는 다음과 같다.

```text
image
→ patchify
→ patch tokens [B, N, patch_dim]
→ linear projection [B, N, D]
→ Hybrid Block × L
→ residual head [B, N, patch_dim]
→ unpatchify
→ residual image r [B, C, H, W]
```

예를 들어 $224 \times 224$ 이미지와 patch size $16$을 사용할 경우:

$$
N
=
\frac{224}{16}
\times
\frac{224}{16}
=
196
$$

patch dimension은 다음과 같다.

$$
patch\_dim
=
16 \times 16 \times 3
=
768
$$

따라서 patch token shape은 다음과 같다.

```text
[B, 196, 768]
```

DGNet 출력은 원본 이미지와 같은 shape이어야 한다.

```text
r: [B, 3, 224, 224]
```

---

## 11. DGNet zero initialization

DGNet의 마지막 출력층은 zero initialization 되어야 한다.

초기 조건은 다음과 같다.

$$
DGNet(x) \approx 0
$$

따라서 degraded image는 다음과 같다.

$$
x_{\text{dg}}
=
x - DGNet(x)
\approx
x
$$

이는 적대적 학습 초기에 DGNet이 이미지를 과도하게 망가뜨려 Inference Network의 학습을 붕괴시키는 것을 방지한다.

테스트에서는 다음이 확인되어야 한다.

```text
mean(abs(DGNet(x))) ≈ 0
mean(abs(x_dg - x)) ≈ 0
```

---

## 12. Config 기반 구현 요구사항

모든 아키텍처는 `DgNetConfig`만으로 생성 가능해야 한다.

`DgNetConfig`는 다음을 제어해야 한다.

- input image size
- patch size
- input channel count
- encoder architecture
- decoder architecture
- DGNet architecture
- embedding dimension
- depth
- number of attention heads
- MLP ratio
- dropout
- positional embedding
- output clamp 여부
- CNN kernel size
- hybrid block 설정
- projection dimension
- zero-init 여부

핵심 원칙은 다음과 같다.

```text
DgNetConfig만으로 전체 모델 구조가 재현 가능해야 한다.
```

---

## 13. Fan-out / Fan-in 규칙

다음 세 모듈만 fan-in 가능하다.

```text
cnn.py
vit.py
hybrid.py
```

각 모듈의 역할은 다음과 같다.

```text
cnn.py      → CNN blocks
vit.py      → ViT encoder/decoder blocks
hybrid.py   → Hybrid Transformer blocks
dg_model.py → config 기반 모델 조립
```

`dg_model.py`는 직접 CNN, ViT, Hybrid block을 모두 구현하면 안 된다. `dg_model.py`는 config를 읽고 각 모듈의 block을 조립하는 역할만 수행한다.

---

## 14. Save / Load 요구사항

모델은 반드시 config와 weight를 함께 저장해야 한다.

저장 구조는 다음과 같다.

```text
save_path/
├── config.json
├── model.pt
└── summary.txt
```

load 과정은 다음 순서를 따른다.

1. `config.json`을 읽는다.
2. `DgNetConfig`를 복원한다.
3. config 기반으로 모델 아키텍처를 생성한다.
4. `model.pt`의 weight를 load한다.
5. load 성공 메시지를 출력한다.

load 성공 시 다음 정보가 출력되어야 한다.

```text
[Load Success]
- config path
- weight path
- encoder architecture
- decoder architecture
- degradation architecture
- total parameter count
- encoder parameter count
- decoder parameter count
- degradation net parameter count
```

---

## 15. 테스트 및 output 요구사항

구현 요구사항의 테스트 결과는 반드시 `output/` 디렉토리에 남아야 한다.

필수 output 구조는 다음과 같다.

```text
output/
├── model_summary.txt
├── load_test_summary.txt
├── tensor_shape_log.txt
├── zero_init_check.txt
└── sample_inference/
    ├── original.png
    ├── residual.png
    ├── degraded.png
    └── reconstruction.png
```

필수 테스트는 다음과 같다.

### 15.1 Config 기반 모델 생성 테스트

- `DgNetConfig`만으로 모델 생성 가능해야 한다.
- encoder, decoder, DGNet architecture가 config 값에 따라 생성되어야 한다.
- 결과는 `output/model_summary.txt`에 저장한다.

### 15.2 Forward shape 테스트

한 batch 입력에 대해 다음 shape이 맞아야 한다.

```text
x          : [B, 3, H, W]
r          : [B, 3, H, W]
x_dg       : [B, 3, H, W]
z_x        : [B, projection_dim]
z_dg       : [B, projection_dim]
x_recon    : [B, 3, H, W]
x_dg_recon : [B, 3, H, W]
```

결과는 `output/tensor_shape_log.txt`에 저장한다.

### 15.3 Zero-init 테스트

확인 항목은 다음과 같다.

```text
mean(abs(DGNet(x)))
mean(abs(x_dg - x))
max(abs(DGNet(x)))
```

결과는 `output/zero_init_check.txt`에 저장한다.

기대 결과는 다음과 같다.

```text
mean(abs(DGNet(x))) ≈ 0
mean(abs(x_dg - x)) ≈ 0
```

### 15.4 Save / Load 테스트

다음 순서를 테스트한다.

1. 모델 생성
2. save
3. 새 모델 load
4. 동일 입력에 대해 forward
5. save 전후 출력 차이 비교

확인 항목은 다음과 같다.

```text
max_abs_diff_before_after_load
mean_abs_diff_before_after_load
```

결과는 `output/load_test_summary.txt`에 저장한다.

### 15.5 Sample inference 이미지 저장 테스트

dataloader 한 batch에 대해 다음 이미지를 저장한다.

```text
output/sample_inference/original.png
output/sample_inference/residual.png
output/sample_inference/degraded.png
output/sample_inference/reconstruction.png
```

초기 zero-init 상태에서는 다음이 기대된다.

```text
degraded image ≈ original image
residual image ≈ black image
```

---

## 16. 평가 지표와 분석 관점

최종 실험에서는 단순히 pretraining loss만 보는 것이 아니라, 다음을 함께 확인해야 한다.

### 16.1 Representation quality

- linear probing accuracy
- fine-tuning accuracy
- k-NN evaluation if needed

### 16.2 Problem generator quality

- residual visualization
- degraded image visualization
- reconstruction visualization
- image-wise reconstruction distance distribution
- image-wise budget energy distribution
- channel-wise spatial KL value

### 16.3 Small-data efficiency

작은 데이터셋에서 다음을 비교한다.

- fixed augmentation baseline
- random masking baseline
- representation-only DGNet
- reconstruction-only DGNet
- hybrid DGNet

핵심 질문은 다음과 같다.

$$
\boxed{
\text{DGNet이 작은 데이터셋에서 더 질 좋은 SSL problem을 만들어 성능을 올리는가?}
}
$$

---

## 17. 최종 실험 축 정리

전체 실험은 세 축으로 구성된다.

| 축 | 질문 | 비교 |
| --- | --- | --- |
| Problem generation | 문제생성기가 SSL 성능을 올리는가? | fixed/random aug vs DGNet-generated degradation |
| Training signal | 어떤 adversarial signal이 좋은 문제를 만드는가? | representation only / reconstruction only / hybrid |
| Architecture | 어떤 DGNet 구조가 좋은 문제를 만드는가? | shallow CNN / ViT / Hybrid DGNet |

---

## 18. 최종 주장 후보

본 실험이 성공할 경우 주장할 수 있는 핵심은 다음과 같다.

$$
\boxed{
\text{SSL 성능은 learner뿐 아니라 problem generator의 품질에도 크게 의존한다.}
}
$$

$$
\boxed{
\text{Adversarially learned problem generators can produce informative views for small-data SSL.}
}
$$

$$
\boxed{
\text{Pixel-level adversarial reconstruction can be stabilized using image-wise difficulty and budget constraints.}
}
$$

$$
\boxed{
\text{Hybrid CNN-ViT problem generators can provide better degradation problems than shallow CNN or plain ViT generators.}
}
$$

---

## 19. 한 문장 요약

본 연구는 SSL에서 사람이 설계한 augmentation 대신, 적대적으로 학습되는 DGNet 문제생성기를 도입하고, 이 문제생성기의 학습 신호와 아키텍처가 작은 데이터셋에서의 representation learning 성능에 미치는 영향을 분석한다.

영문 요약은 다음과 같다.

```text
This work studies adversarial problem generation for self-supervised learning, where a learnable DGNet generates informative degraded views, and investigates how generator objectives and architectures affect SSL performance, especially under small-data regimes.
```

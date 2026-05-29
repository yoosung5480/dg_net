# 4. DGnet: 복원학습 기반 adversarial degradation baseline

DGnet은 ADIOS와 구조적으로 비슷하지만, 핵심 차이는 **representation matching이 아니라 reconstruction distance를 제어하는 복원 기반 adversarial degradation**이라는 점이다.

ADIOS는 다음과 같이 representation distance를 사용한다.

$$
x_m = x \odot M(x)
$$

$$
\mathcal{L}_{ADIOS}
=
D_{rep}(I(x), I(x_m))
$$

반면 DGnet은 다음과 같이 degradation residual을 제거한 이미지를 복원 입력으로 사용한다.

$$
m = M(x)
$$

$$
x_{deg} = x - m
$$

$$
\hat{x}
=
I_{dec}(I_{enc}(x_{deg}))
$$

$$
\mathcal{L}_{DGnet}
=
D_{rec}(\hat{x}, x)
$$

---

## 4.1 입력 변환

DGnet의 degradation model은 입력 이미지 \(x\)에 대해 image-shaped residual을 생성한다.

$$
m = M(x)
$$

$$
m \in \mathbb{R}^{H \times W \times C}
$$

여기서 \(m\)은 최종 degraded image가 아니라, 원본 이미지에서 제거할 residual이다.

따라서 degraded image는 다음과 같이 정의된다.

$$
x_{deg}
=
x - M(x)
$$

안정성을 위해 clipping을 사용할 경우 다음과 같이 쓸 수 있다.

$$
x_{deg}
=
\mathrm{clip}(x - M(x), 0, 1)
$$

---

## 4.2 Inference model

DGnet의 inference model \(I\)는 encoder와 decoder로 구성된다.

$$
I = I_{enc} + I_{dec}
$$

encoder는 degraded image로부터 latent representation을 추출한다.

$$
z_{deg}
=
I_{enc}(x_{deg})
$$

decoder는 latent representation으로부터 원본 이미지를 복원한다.

$$
\hat{x}
=
I_{dec}(z_{deg})
$$

따라서 전체 복원 과정은 다음과 같다.

$$
\hat{x}
=
I_{dec}(I_{enc}(x - M(x)))
$$

---

## 4.3 Reconstruction distance

복원 거리 \(D_{rec}\)는 복원 결과 \(\hat{x}\)와 원본 이미지 \(x\) 사이의 거리로 정의한다.

$$
D_{rec}(I, M)
=
\left\|
\hat{x} - x
\right\|_2^2
$$

즉,

$$
D_{rec}(I, M)
=
\left\|
I_{dec}(I_{enc}(x - M(x))) - x
\right\|_2^2
$$

MAE-style patch reconstruction을 사용할 경우 다음과 같이 쓸 수 있다.

$$
D_{rec}(I, M)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
x_i - \hat{x}_i
\right\|_2^2
$$

---

## 4.4 두 개의 threshold

DGnet에서는 두 개의 threshold를 구분해서 제어한다.

첫 번째는 \(M(x)\) 자체에 대한 threshold이다. 이는 degradation residual의 크기를 제한하는 예산이다.

두 번째는 \(x - M(x)\)로 인해 발생하는 복원 난이도에 대한 threshold이다. 이는 degraded image가 원본 의미를 얼마나 해치도록 만들 것인지를 제어하는 목표 난이도이다.

---

## 4.5 Mask budget threshold

첫 번째 threshold는 **mask budget threshold**로 정의한다.

$$
\beta_{mask}
$$

이는 \(M(x)\)가 사용할 수 있는 총 파괴 예산을 의미한다. MAE에서 mask ratio를 0.75로 두는 것처럼, DGnet에서는 \(M(x)\)의 평균 residual magnitude가 일정 수준을 넘지 않도록 제어한다.

degradation budget distance는 다음과 같이 정의한다.

$$
D_{budget}(M)
=
\left|
\mathrm{mean}(|M(x)|) - \beta_{mask}
\right|
$$

여기서:

$$
\mathrm{mean}(|M(x)|)
$$

은 \(M(x)\)가 전체 이미지에서 평균적으로 얼마나 많은 residual을 제거하려 하는지를 나타낸다.

따라서 \(M\)은 다음 조건을 만족하도록 유도된다.

$$
\mathrm{mean}(|M(x)|)
\rightarrow
\beta_{mask}
$$

즉, \(M\)은 무한히 이미지를 망가뜨릴 수 없고, 정해진 파괴 예산 안에서만 degradation을 생성해야 한다.

---

## 4.6 Degradation target threshold

두 번째 threshold는 **degradation target threshold**로 정의한다.

$$
\tau_{deg}
$$

이는 \(x - M(x)\)가 inference model \(I\)에게 얼마나 어려운 복원 문제를 만들 것인지에 대한 목표값이다.

중요한 점은 \(M\)의 목표가 복원 거리를 무한히 키우는 것이 아니라는 점이다.

잘못된 목표는 다음과 같다.

$$
M:
D_{rec}(I, M)
\rightarrow
\infty
$$

이 방식은 유클리디언 거리 기반 loss에서 위험하다. 단순히 \(-D_{rec}\)를 minimize하면 \(M\)은 이미지를 무한히 망가뜨리거나, 전체 밝기 감소 같은 trivial solution으로 갈 수 있다.

따라서 DGnet에서 \(M\)의 목표는 다음과 같이 정의한다.

$$
M:
D_{rec}(I, M)
\rightarrow
\tau_{deg}
$$

이를 위해 target distance matching loss를 사용한다.

$$
D_{target}(I, M)
=
\left(
D_{rec}(I, M) - \tau_{deg}
\right)^2
$$

즉,

$$
D_{target}(I, M)
=
\left(
\left\|
I_{dec}(I_{enc}(x - M(x))) - x
\right\|_2^2
-
\tau_{deg}
\right)^2
$$

이 항은 \(M\)이 복원 오차를 무한히 키우는 것이 아니라, 목표 난이도 \(\tau_{deg}\) 근처로 맞추도록 만든다.

---

## 4.7 Brightness-reduction regularization

mask budget threshold만으로는 전체 밝기 감소를 막을 수 없다.

예를 들어 다음과 같은 해가 가능하다.

$$
M(x)
=
\beta_{mask} \cdot \mathbf{1}
$$

$$
x_{deg}
=
x - \beta_{mask}
$$

이는 budget은 만족하지만, 모든 픽셀을 균등하게 어둡게 만드는 trivial solution이다.

이를 막기 위해 damage map을 정의한다.

$$
A(x)
=
\mathrm{mean}_{c}(|M(x)|)
$$

$$
A(x)
\in
\mathbb{R}^{H \times W}
$$

damage map을 확률분포로 정규화한다.

$$
P_{ij}(x)
=
\frac{
A_{ij}(x)
}{
\sum_{i,j} A_{ij}(x) + \epsilon
}
$$

uniform distribution은 다음과 같다.

$$
U_{ij}
=
\frac{1}{HW}
$$

전체 밝기 감소는 damage가 모든 위치에 균등하게 퍼지는 경우이므로, \(P(x)\)가 \(U\)에 가까운 상태로 해석할 수 있다.

따라서 \(P(x)\)가 uniform distribution에서 멀어지도록 KL divergence를 사용한다.

$$
D_{KL}(P(x) \| U)
=
\sum_{i,j}
P_{ij}(x)
\log
\frac{
P_{ij}(x)
}{
U_{ij}
}
$$

\(M\)이 minimize하는 loss에서는 이 항을 음수로 넣는다.

$$
\mathcal{L}_{reg}(M, x)
=
-
D_{KL}(P(x) \| U)
$$

즉, \(M\)은 \(D_{KL}(P(x) \| U)\)를 크게 만들도록 유도된다.

이는 damage distribution이 uniform하게 퍼지는 것을 막고, 전체 밝기 감소 collapse를 방지하기 위한 항이다.

---

## 4.8 Inference model loss

Inference model \(I\)의 목표는 degraded image를 보고 원본 이미지를 복원하는 것이다.

따라서 \(I\)의 loss는 reconstruction distance 자체이다.

$$
\boxed{
\mathcal{L}_{I}
=
D_{rec}(I, M)
}
$$

즉,

$$
\boxed{
\mathcal{L}_{I}
=
\left\|
I_{dec}(I_{enc}(x - M(x))) - x
\right\|_2^2
}
$$

따라서 \(I\)는 다음 방향으로 학습된다.

$$
I:
D_{rec}(I, M)
\rightarrow
0
$$

---

## 4.9 Degradation model loss

Degradation model \(M\)은 세 가지 목표를 가진다.

첫째, 복원 난이도를 목표 threshold \(\tau_{deg}\)에 맞춘다.

$$
D_{rec}(I, M)
\rightarrow
\tau_{deg}
$$

둘째, degradation residual의 크기를 mask budget threshold \(\beta_{mask}\)에 맞춘다.

$$
\mathrm{mean}(|M(x)|)
\rightarrow
\beta_{mask}
$$

셋째, damage distribution이 uniform distribution이 되는 것을 피한다.

$$
P(x)
\not\rightarrow
U
$$

따라서 \(M\)의 loss는 다음과 같다.

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
D_{rec}(I, M) - \tau_{deg}
\right)^2
+
\lambda_{budget}
\left|
\mathrm{mean}(|M(x)|) - \beta_{mask}
\right|
-
\lambda_{reg}
D_{KL}(P(x) \| U)
}
$$

최종적으로 \(D_{rec}\)까지 모두 풀어 쓰면 다음과 같다.

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

---

## 4.10 최종 해석

DGnet의 핵심은 \(M\)이 복원 오차를 무한히 키우는 것이 아니라, 정해진 난이도의 복원 문제를 만드는 것이다.

Inference model은 다음 목표를 가진다.

$$
\boxed{
I:
D_{rec}(I, M)
\rightarrow
0
}
$$

Degradation model은 다음 목표를 가진다.

$$
\boxed{
M:
D_{rec}(I, M)
\rightarrow
\tau_{deg}
}
$$

동시에 \(M\)은 정해진 파괴 예산을 만족해야 한다.

$$
\boxed{
M:
\mathrm{mean}(|M(x)|)
\rightarrow
\beta_{mask}
}
$$

또한 전체 밝기 감소 collapse를 피해야 한다.

$$
\boxed{
M:
P_{damage}(x)
\not\rightarrow
U
}
$$

따라서 DGnet은 다음과 같이 요약된다.

$$
\boxed{
\mathcal{L}_{I}
=
D_{rec}(I, M)
}
$$

$$
\boxed{
\mathcal{L}_{M}
=
\left(
D_{rec}(I, M) - \tau_{deg}
\right)^2
+
\lambda_{budget}
\left|
\mathrm{mean}(|M(x)|) - \beta_{mask}
\right|
-
\lambda_{reg}
D_{KL}(P(x) \| U)
}
$$

한 줄로 정리하면 다음과 같다.

DGnet은 \(M(x)\)의 파괴 가능 예산을 \(\beta_{mask}\)로 제한하고, \(x - M(x)\)가 만드는 복원 난이도를 \(\tau_{deg}\)로 제어하며, \(-D_{KL}(P(x)\|U)\)를 통해 전체 밝기 감소 collapse를 방지하는 복원 기반 adversarial degradation framework이다.
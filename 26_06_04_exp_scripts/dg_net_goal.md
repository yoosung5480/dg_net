# DGNet Loss Objective 정리

이 문서는 현재 `loss.py` 구현을 기준으로 DGNet의 목적함수를 수식적으로 정리한 것이다.

---

## 1. 기본 기호

`DGLoss.from_tensors()`는 세 개의 텐서를 입력으로 받는다.

$$
x = \text{original}
$$

$$
\hat{x} = \text{reconstruction}
$$

$$
r = \text{residual} = M(x)
$$

여기서 $r$은 degradation network가 생성한 degradation residual이다.

입력 텐서의 shape은 다음과 같이 가정한다.

$$
x, \hat{x}, r \in \mathbb{R}^{B \times C \times H \times W}
$$

---

## 2. Reconstruction Distance

복원 오차는 reconstruction $\hat{x}$와 original image $x$ 사이의 squared error로 정의한다.

픽셀별 squared error는 다음과 같다.

$$
e_{b,c,h,w}
=
(\hat{x}_{b,c,h,w} - x_{b,c,h,w})^2
$$

reduction이 `mean`이면 reconstruction distance는 다음과 같다.

$$
D_{\text{rec}}
=
\frac{1}{BCHW}
\sum_{b=1}^{B}
\sum_{c=1}^{C}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
(\hat{x}_{b,c,h,w} - x_{b,c,h,w})^2
$$

reduction이 `sum`이면 reconstruction distance는 다음과 같다.

$$
D_{\text{rec}}
=
\sum_{b=1}^{B}
\sum_{c=1}^{C}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
(\hat{x}_{b,c,h,w} - x_{b,c,h,w})^2
$$

따라서 일반적으로 다음과 같이 쓸 수 있다.

$$
D_{\text{rec}}
=
\operatorname{reduce}
\left(
(\hat{x} - x)^2
\right)
$$

---

## 3. Inference Objective

Inference model $I$의 목표는 degraded image로부터 원본 이미지를 복원하는 것이다.

따라서 inference loss는 reconstruction distance를 최소화한다.

$$
L_{\text{inference}}
=
\alpha_{\text{inference}}
D_{\text{rec}}
$$

step $t$에서 effective weight를 사용할 경우 다음과 같다.

$$
L_{\text{inference}}^{(t)}
=
\alpha_{\text{inference}}^{(t)}
D_{\text{rec}}
$$

즉 inference model은 다음 방향으로 학습된다.

$$
I:
D_{\text{rec}}
\rightarrow
0
$$

---

## 4. Degradation Target Distance

Degradation model $M$의 목표는 reconstruction distance를 무한히 키우는 것이 아니다.

잘못된 목표는 다음과 같다.

$$
M:
D_{\text{rec}}
\rightarrow
\infty
$$

이 방식은 유클리디언 거리 기반 loss에서 위험하다. 단순히 $-D_{\text{rec}}$를 minimize하면 $M$은 이미지를 무한히 망가뜨리거나 trivial solution으로 갈 수 있다.

따라서 DGNet에서는 degradation target threshold를 정의한다.

$$
\tau_{\text{deg}}
$$

$M$의 목표는 reconstruction distance를 이 target threshold 근처로 맞추는 것이다.

$$
M:
D_{\text{rec}}
\rightarrow
\tau_{\text{deg}}
$$

현재 target distance는 절댓값 기반으로 정의된다.

$$
D_{\text{target}}
=
\left|
D_{\text{rec}} - \tau_{\text{deg}}
\right|
$$

즉 다음 두 경우 모두 penalty가 발생한다.

$$
D_{\text{rec}} < \tau_{\text{deg}}
\Rightarrow
D_{\text{target}}
=
\tau_{\text{deg}} - D_{\text{rec}}
$$

$$
D_{\text{rec}} > \tau_{\text{deg}}
\Rightarrow
D_{\text{target}}
=
D_{\text{rec}} - \tau_{\text{deg}}
$$

weighted target term은 다음과 같다.

$$
L_{\text{target}}^{w}
=
\alpha_{\text{target}}
\left|
D_{\text{rec}} - \tau_{\text{deg}}
\right|
$$

step $t$에서는 다음과 같다.

$$
L_{\text{target}}^{(t)}
=
\alpha_{\text{target}}^{(t)}
\left|
D_{\text{rec}} - \tau_{\text{deg}}
\right|
$$

---

## 5. Mask Budget Distance

Mask budget threshold는 residual $r=M(x)$가 사용할 수 있는 평균 파괴 예산을 의미한다.

$$
\beta_{\text{mask}}
$$

먼저 residual magnitude 평균을 계산한다.

$$
\mu_r
=
\frac{1}{BCHW}
\sum_{b=1}^{B}
\sum_{c=1}^{C}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
|r_{b,c,h,w}|
$$

mask budget distance는 다음과 같이 정의된다.

$$
D_{\text{budget}}
=
\left|
\mu_r - \beta_{\text{mask}}
\right|
$$

즉,

$$
D_{\text{budget}}
=
\left|
\operatorname{mean}(|r|)
-
\beta_{\text{mask}}
\right|
$$

weighted budget term은 다음과 같다.

$$
L_{\text{budget}}^{w}
=
\lambda_{\text{budget}}
\left|
\operatorname{mean}(|r|)
-
\beta_{\text{mask}}
\right|
$$

step $t$에서는 다음과 같다.

$$
L_{\text{budget}}^{(t)}
=
\lambda_{\text{budget}}^{(t)}
\left|
\operatorname{mean}(|r|)
-
\beta_{\text{mask}}
\right|
$$

따라서 degradation model은 다음 조건을 만족하도록 유도된다.

$$
M:
\operatorname{mean}(|r|)
\rightarrow
\beta_{\text{mask}}
$$

---

## 6. Damage Distribution

전체 밝기 감소 같은 trivial solution을 방지하기 위해 residual로부터 spatial damage distribution을 정의한다.

먼저 channel-wise absolute residual damage map을 계산한다.

$$
a_{b,h,w}
=
\frac{1}{C}
\sum_{c=1}^{C}
|r_{b,c,h,w}|
$$

$$
a \in \mathbb{R}^{B \times H \times W}
$$

각 이미지의 spatial damage map을 flatten한다.

$$
a_b \in \mathbb{R}^{HW}
$$

각 이미지별 damage probability distribution은 다음과 같다.

$$
p_{b,i}
=
\frac{
a_{b,i}
}{
\sum_{j=1}^{HW} a_{b,j} + \epsilon
}
$$

여기서,

$$
i \in \{1, \dots, HW\}
$$

uniform distribution은 다음과 같다.

$$
u_i
=
\frac{1}{HW}
$$

---

## 7. Damage KL Divergence

각 이미지별 KL divergence는 다음과 같다.

$$
D_{\text{KL}}^{(b)}
=
\sum_{i=1}^{HW}
p_{b,i}
\log
\frac{p_{b,i}}{u_i}
$$

$p_{b,i}=0$인 위치는 0 contribution으로 처리한다.

$$
p_{b,i} = 0
\Rightarrow
p_{b,i}
\log
\frac{p_{b,i}}{u_i}
= 0
$$

따라서 구현 관점에서는 다음과 같이 쓸 수 있다.

$$
D_{\text{KL}}^{(b)}
=
\sum_{i=1}^{HW}
\mathbf{1}[p_{b,i}>0]
p_{b,i}
\left(
\log(\max(p_{b,i}, \epsilon))
-
\log(u_i)
\right)
$$

reduction이 `mean`이면 다음과 같다.

$$
D_{\text{KL}}
=
\frac{1}{B}
\sum_{b=1}^{B}
D_{\text{KL}}^{(b)}
$$

reduction이 `sum`이면 다음과 같다.

$$
D_{\text{KL}}
=
\sum_{b=1}^{B}
D_{\text{KL}}^{(b)}
$$

regularization term은 음수 부호를 가진다.

$$
L_{\text{reg}}^{w}
=
-
\lambda_{\text{reg}}
D_{\text{KL}}
$$

step $t$에서는 다음과 같다.

$$
L_{\text{reg}}^{(t)}
=
-
\lambda_{\text{reg}}^{(t)}
D_{\text{KL}}
$$

즉 degradation model이 minimize할 때 $D_{\text{KL}}$을 키우도록 유도된다.

$$
M:
P_{\text{damage}}
\not\rightarrow
U
$$

---

## 8. Degradation Objective

Degradation loss는 target term, budget term, KL regularization term으로 구성된다.

$$
L_{\text{degradation}}
=
L_{\text{target}}^{w}
+
L_{\text{budget}}^{w}
+
L_{\text{reg}}^{w}
$$

따라서 다음과 같다.

$$
L_{\text{degradation}}
=
\alpha_{\text{target}}
D_{\text{target}}
+
\lambda_{\text{budget}}
D_{\text{budget}}
-
\lambda_{\text{reg}}
D_{\text{KL}}
$$

이를 풀어 쓰면 다음과 같다.

$$
L_{\text{degradation}}
=
\alpha_{\text{target}}
\left|
D_{\text{rec}} - \tau_{\text{deg}}
\right|
+
\lambda_{\text{budget}}
\left|
\operatorname{mean}(|r|)
-
\beta_{\text{mask}}
\right|
-
\lambda_{\text{reg}}
D_{\text{KL}}
$$

step $t$에서의 degradation loss는 다음과 같다.

$$
L_{\text{degradation}}^{(t)}
=
\alpha_{\text{target}}^{(t)}
\left|
D_{\text{rec}} - \tau_{\text{deg}}
\right|
+
\lambda_{\text{budget}}^{(t)}
\left|
\operatorname{mean}(|r|)
-
\beta_{\text{mask}}
\right|
-
\lambda_{\text{reg}}^{(t)}
D_{\text{KL}}
$$

---

## 9. Combined Loss

보고용 combined loss는 다음과 같다.

$$
L_{\text{combined}}
=
L_{\text{inference}}
+
L_{\text{degradation}}
$$

즉,

$$
L_{\text{combined}}
=
\alpha_{\text{inference}}D_{\text{rec}}
+
\alpha_{\text{target}}
\left|
D_{\text{rec}} - \tau_{\text{deg}}
\right|
+
\lambda_{\text{budget}}
\left|
\operatorname{mean}(|r|)
-
\beta_{\text{mask}}
\right|
-
\lambda_{\text{reg}}
D_{\text{KL}}
$$

step $t$에서는 다음과 같다.

$$
L_{\text{combined}}^{(t)}
=
\alpha_{\text{inference}}^{(t)}
D_{\text{rec}}
+
\alpha_{\text{target}}^{(t)}
\left|
D_{\text{rec}} - \tau_{\text{deg}}
\right|
+
\lambda_{\text{budget}}^{(t)}
\left|
\operatorname{mean}(|r|)
-
\beta_{\text{mask}}
\right|
-
\lambda_{\text{reg}}^{(t)}
D_{\text{KL}}
$$

단, 실제 학습에서는 $L_{\text{combined}}$를 직접 하나의 optimizer로 최적화하지 않고, inference objective와 degradation objective를 분리해서 최적화한다.

---

## 10. Mean / Sum Scale

reconstruction distance에서 `mean` reduction을 쓰면 다음과 같다.

$$
D_{\text{rec}}^{mean}
=
\frac{1}{BCHW}
\sum_{b,c,h,w}
(\hat{x}_{b,c,h,w} - x_{b,c,h,w})^2
$$

이때 $\tau_{\text{deg}}$는 pixel-wise mean squared error scale을 가져야 한다.

$$
\tau_{\text{deg}}
\approx
\text{pixel-wise mean squared error target}
$$

reconstruction distance에서 `sum` reduction을 쓰면 다음과 같다.

$$
D_{\text{rec}}^{sum}
=
\sum_{b,c,h,w}
(\hat{x}_{b,c,h,w} - x_{b,c,h,w})^2
$$

이때 같은 reconstruction error 수준을 표현하려면 다음 관계를 만족해야 한다.

$$
\tau_{\text{deg}}^{sum}
=
\tau_{\text{deg}}^{mean}
\times B \times C \times H \times W
$$

KL divergence에서 `mean` reduction은 다음과 같다.

$$
D_{\text{KL}}^{mean}
=
\frac{1}{B}
\sum_{b=1}^{B}
D_{\text{KL}}^{(b)}
$$

KL divergence에서 `sum` reduction은 다음과 같다.

$$
D_{\text{KL}}^{sum}
=
\sum_{b=1}^{B}
D_{\text{KL}}^{(b)}
$$

즉 KL의 `sum`은 전체 $BCHW$ 원소 합이 아니라, per-image KL을 batch dimension에 대해 합산하는 것이다.

---

## 11. 최종 목적함수 요약

Inference objective는 다음과 같다.

$$
L_{\text{inference}}^{(t)}
=
\alpha_{\text{inference}}^{(t)}
\operatorname{reduce}
\left(
(\hat{x} - x)^2
\right)
$$

Degradation objective는 다음과 같다.

$$
L_{\text{degradation}}^{(t)}
=
\alpha_{\text{target}}^{(t)}
\left|
D_{\text{rec}} - \tau_{\text{deg}}
\right|
+
\lambda_{\text{budget}}^{(t)}
\left|
\operatorname{mean}(|r|)
-
\beta_{\text{mask}}
\right|
-
\lambda_{\text{reg}}^{(t)}
KL(P_{\text{damage}} \| U)
$$

여기서,

$$
D_{\text{rec}}
=
\operatorname{reduce}
\left(
(\hat{x} - x)^2
\right)
$$

$$
P_{\text{damage}}(i)
=
\frac{
\frac{1}{C}\sum_{c=1}^{C}|r_{c,i}|
}{
\sum_{j=1}^{HW}
\frac{1}{C}\sum_{c=1}^{C}|r_{c,j}|
+
\epsilon
}
$$

$$
U(i)
=
\frac{1}{HW}
$$

$$
KL(P_{\text{damage}} \| U)
=
\operatorname{reduce}_{batch}
\left[
\sum_{i=1}^{HW}
P_{\text{damage}}(i)
\log
\frac{P_{\text{damage}}(i)}{U(i)}
\right]
$$

최종적으로 현재 loss 구조는 다음과 같다.

$$
L_{\text{combined}}
=
\alpha_{\text{inference}}D_{\text{rec}}
+
\alpha_{\text{target}}
\left|
D_{\text{rec}}-\tau_{\text{deg}}
\right|
+
\lambda_{\text{budget}}
\left|
\operatorname{mean}(|r|)
-
\beta_{\text{mask}}
\right|
-
\lambda_{\text{reg}}
KL(P_{\text{damage}}\|U)
$$

실제 학습에서는 다음 두 objective를 분리해서 사용한다.

$$
L_{\text{inference}}
=
\alpha_{\text{inference}}D_{\text{rec}}
$$

$$
L_{\text{degradation}}
=
\alpha_{\text{target}}
\left|
D_{\text{rec}}-\tau_{\text{deg}}
\right|
+
\lambda_{\text{budget}}
\left|
\operatorname{mean}(|r|)
-
\beta_{\text{mask}}
\right|
-
\lambda_{\text{reg}}
KL(P_{\text{damage}}\|U)
$$
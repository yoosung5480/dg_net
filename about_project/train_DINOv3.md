# 2. DINOv3: 대조/증류학습 baseline

DINO 계열에서 `M`은 mask라기보다 **view 생성 함수**, 즉 augmentation operator다. DINO는 student-teacher/self-distillation 구조이고, teacher output을 student가 맞춘다. 기존 DINO는 teacher를 EMA로 업데이트하고, cross-entropy로 student distribution이 teacher distribution을 맞추게 한다. 

DINOv3는 이 DINOv2 계열 SSL recipe를 scale-up하고, long training에서 dense feature가 망가지는 문제를 줄이기 위해 **Gram anchoring**을 추가한 형태로 보는 게 맞다. DINOv3 technical report는 DINOv2-style algorithm을 scale에서 사용하고, dense feature degradation을 Gram anchoring으로 완화한다고 설명한다. ([arXiv][1])

## 입력 변환

두 개 이상의 augmentation view를 만든다.

```text
v_1 = M_1(x)
v_2 = M_2(x)
```

여기서:

```text
M_k = random crop, color jitter, blur, solarization 등
```

DINOv3/DiNO 계열에서 `M`은 학습되는 adversarial model이 아니라 stochastic augmentation pipeline이다.

## I 모델

student:

```text
p_s(v) = softmax(g_s(I_enc^s(v)) / τ_s)
```

teacher:

```text
p_t(v) = softmax((g_t(I_enc^t(v)) - c) / τ_t)
```

여기서 `c`는 centering term, `τ_s`, `τ_t`는 temperature다.

teacher는 gradient로 학습하지 않고 EMA로 업데이트한다.

```text
θ_t ← λ θ_t + (1 - λ) θ_s
```

## DINO global loss

두 view에 대해 teacher-student cross entropy:

```text
L_DINO
=
H(sg[p_t(v_1)], p_s(v_2))
+
H(sg[p_t(v_2)], p_s(v_1))
```

cross entropy는:

```text
H(p_t, p_s)
=
- ∑_k p_t^{(k)} log p_s^{(k)}
```

최적화:

```text
θ_s* = argmin_{θ_s} L_DINO
θ_t는 EMA update
```

## DINOv3 Gram anchoring 추가항

DINOv3의 핵심 추가는 patch-level feature의 pairwise similarity 구조를 보존하는 Gram loss다. 논문은 student patch feature와 Gram teacher patch feature의 Gram matrix를 맞추는 방식으로 설명한다. ([arXiv][1])

patch feature:

```text
F_s(x) ∈ R^{N × d}
F_g(x) ∈ R^{N × d}
```

row-wise normalized라면:

```text
G_s(x) = F_s(x) F_s(x)^T
G_g(x) = F_g(x) F_g(x)^T
```

Gram anchoring loss:

```text
L_Gram
=
|| G_s(x) - sg[G_g(x)] ||_F^2
```

최종 DINOv3-style pretrain objective:

```text
L_DINOv3
=
L_DINO
+
λ_iBOT L_iBOT
+
λ_KoLeo L_KoLeo
+
λ_Gram L_Gram
```

실험 baseline을 단순화하면, 네 비교실험에서는 보통 이렇게 잡으면 된다.

```text
L_DINOv3 ≈ L_DINO + λ_Gram L_Gram
```

## 네 세계관 기준 해석

```text
I = student/teacher encoder
I_dec = 없음
M = random augmentation/view generator
M은 학습되지 않음
목표는 서로 다른 view의 representation 일치
```

즉 DINOv3는:

```text
min_I  representation_matching_loss(I(M_1(x)), I_teacher(M_2(x)))
```
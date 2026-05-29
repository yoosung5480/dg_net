# ADIOS



# 3. ADIOS: 대조학습 기반 adversarial masking baseline

ADIOS는 네 세계관과 가장 가까운 기존 adversarial baseline이다. ADIOS는 `I`와 `M`을 명시적으로 둔다. 논문에서 `M(x)`는 image-sized mask이고, masked image는 Hadamard product로 만든다. `I`는 원본 이미지와 masked image의 representation distance를 줄이고, `M`은 그 distance를 키우는 min-max 구조다. 

## 입력 변환

ADIOS의 mask model:

```text
m = M(x),    m ∈ [0, 1]^{H×W}
```

masked image:

```text
x_m = x ⊙ m
```

논문 표기상 `⊙`는 Hadamard product다.

## I 모델

ADIOS의 핵심 버전은 encoder-only다.

```text
z   = I(x)
z_m = I(x_m)
```

representation distance:

```text
D(z, z_m)
```

예를 들어 cosine distance라면:

```text
D(z, z_m)
=
1 - cos(z, z_m)
```

## 기본 min-max objective

논문은 다음 구조를 사용한다.

```text
I*, M*
=
argmin_I max_M L(x; I, M)
```

encoder-only objective:

```text
L_ENC(x; I, M)
=
D(I(x), I(x ⊙ M(x)))
```

따라서:

```text
I* = argmin_I D(I(x), I(x ⊙ M(x)))
M* = argmax_M D(I(x), I(x ⊙ M(x)))
```

의미:

```text
I는 원본과 masked image의 representation을 같게 만들려 함
M은 representation이 달라지도록 중요한 부분을 가리려 함
```

## ADIOS + contrastive SSL 형태

ADIOS는 SimCLR/BYOL/SimSiam 같은 SSL objective에 붙을 수 있다. 논문은 SimCLR-ADIOS objective에서 한쪽 augmented view를 mask한 representation으로 대체한다. 

두 augmentation:

```text
x^A = A(x)
x^B = B(x)
```

mask:

```text
m^A = M(x^A)
x^{A,m} = x^A ⊙ m^A
```

representation:

```text
z_i^{A,m} = I(x_i^A ⊙ M(x_i^A))
z_i^B     = I(x_i^B)
```

contrastive objective:

```text
L_ADIOS-contrastive
=
- log
  exp(sim(z_i^{A,m}, z_i^B) / τ)
  /
  ∑_{j}
  exp(sim(z_i^{A,m}, z_j^B) / τ)
```

min-max:

```text
min_I max_M L_ADIOS-contrastive
```

주의할 점은 `I` 입장에서 loss를 줄인다는 것은 positive pair를 가깝게 하는 것이고, `M` 입장에서는 masked view가 원본/다른 view와 멀어지도록 어려운 mask를 만든다는 뜻이다.

## 네 세계관 기준 해석

```text
I = encoder only
I_dec = 없음
M = adversarial mask generator
x_m = x ⊙ M(x)
목표 = representation invariance
```

즉 ADIOS는:

```text
min_I max_M  representation_distance(I(x), I(x ⊙ M(x)))
```

---


### Representation loss

ADIOS 방식의 representation loss는 inference network `I`와 degradation network `M`의 min-max objective로 정의된다.

$$
I^*, M^*
=
\arg\min_I \max_M
\frac{1}{N}
\sum_{n=1}^{N}
\mathcal{L}^{(n)}(x; I, M)
$$

여기서 `M`은 입력 이미지 `x`에 대해 `N`개의 mask를 생성한다.

$$
\{m^{(1)}, m^{(2)}, \dots, m^{(N)}\}
$$

각 `L^{(n)}(x; I, M)`은 원본 view와 `n`번째 masked/degraded view 사이에서 계산되는 SSL representation loss이다.

---

### Sparsity penalty

ADIOS는 mask가 전부 0이 되거나 전부 1이 되는 trivial solution을 막기 위해 sparsity penalty를 사용한다.

$$
p_n
=
\sin
\left(
\frac{\pi}{hw}
\sum_{i=1}^{h}
\sum_{j=1}^{w}
m_{ij}^{(n)}
\right)^{-1}
$$

이 penalty는 `m^{(n)}`이 all-zero 또는 all-one에 가까워질수록 커진다.

---

### Final representation objective

최종 representation objective는 SSL representation loss와 sparsity penalty를 함께 사용한다.

$$
I^*, M^*
=
\arg\min_I \max_M
\frac{1}{N}
\sum_{n=1}^{N}
\left(
\mathcal{L}^{(n)}(x; I, M)
-
\lambda p_n
\right)
$$

여기서 `lambda`는 sparsity penalty의 강도를 조절하는 계수이다.

Inference network `I`는 `L^{(n)}`을 최소화하는 방향으로 학습된다.

Degradation network `M`은 `L^{(n)}`을 최대화하되, `-lambda * p_n` 항에 의해 degenerate mask를 만들지 않도록 제한된다.
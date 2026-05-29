
# 1. MAE: 복원학습 baseline

MAE에서 `M`은 neural network가 아니라 **random masking operator**다. 입력 이미지 `x`를 patch sequence로 나눈 뒤 일부 patch를 제거한다. MAE 논문은 random patch masking, asymmetric encoder-decoder, masked patch pixel reconstruction을 핵심으로 둔다. 특히 encoder는 visible patch만 보고, decoder가 mask token을 포함한 전체 sequence를 복원한다. 

## 입력 변환

이미지를 patchify한다.

```text
x -> {x_i}_{i=1}^{N}
```

random mask index set:

```text
Ω_m ⊂ {1, ..., N}
Ω_v = {1, ..., N} \ Ω_m
```

MAE의 degradation operator:

```text
M_MAE(x) = Ω_m
```

encoder 입력:

```text
x_v = {x_i | i ∈ Ω_v}
```

## I 모델

```text
z = I_enc(x_v)
```

decoder는 encoder output과 mask token을 함께 받는다.

```text
x_hat = I_dec(z, mask_tokens, Ω_m)
```

## loss

MAE는 복원 loss를 **masked patch에 대해서만** 계산한다. 논문도 pixel-space MSE를 masked patches에 대해서만 계산한다고 명시한다. 

```text
L_MAE(I)
=
(1 / |Ω_m|)
∑_{i ∈ Ω_m}
|| x_i - x_hat_i ||_2^2
```

최적화:

```text
I* = argmin_I L_MAE(I)
```

## 네 세계관 기준 해석

```text
I = I_enc + I_dec
M = random mask sampler
M은 학습되지 않음
목표는 x_visible -> x_original 복원
```

즉 MAE는:

```text
min_I  reconstruction_loss(I(x without random patches), x)
```

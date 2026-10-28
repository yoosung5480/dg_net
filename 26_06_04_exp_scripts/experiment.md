## 이번실험 목표
- 규제항 $KL(P_{\text{damage}} \| U)$ 이 주는 영향에 댇해서 대조실험

## 목적함수
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



## 실험 Case
### Case1.
- 규제항 단항식 계수 0.01로 설정



```bash
LOSS_ARGS=(
  --alpha-inference 1.0
  --alpha-target 1.0
  --lambda-budget 1.0
  --lambda-reg 0.01
  --tau-deg 0.5
  --beta-mask 0.5
  --loss-reduction mean
)
```


### Case2.
- 규제항 단항식 계수 0.05로 설정

```bash
LOSS_ARGS=(
  --alpha-inference 1.0
  --alpha-target 1.0
  --lambda-budget 1.0
  --lambda-reg 0.01
  --tau-deg 0.5
  --beta-mask 0.5
  --loss-reduction mean
)
```


### Case3.
- 규제항을 아예 제거

```bash
LOSS_ARGS=(
  --alpha-inference 1.0
  --alpha-target 1.0
  --lambda-budget 1.0
  --lambda-reg 0.00
  --tau-deg 0.5
  --beta-mask 0.5
  --loss-reduction mean
)
```

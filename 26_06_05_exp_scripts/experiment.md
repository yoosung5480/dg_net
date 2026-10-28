## 이번실험 목표
- budget 제한이 오히려 마스킹 결과를 안좋게 만들지 않는가? 에 대한 목표

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

=
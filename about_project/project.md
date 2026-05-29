# 비교실험
---

# pretrain 훈련
크게 4가지 지원해야함

## Self Supervised learning baselines
1. MAE      - 복원학습모델 베이스라인
2. DINOv3   - 대조학습모델 베이스라인

## Adversarial Train baseline
1. ADIOS    - 대조 학습 기반 적대적  
2. DGnet    - 복원 학습 기반 적대적 (내 노벨티)

## 아키텍쳐
- ViT only 기반 (baseline)
- CNN 기반 (Conxnet)
- Hybrid 기반 (내 노벨티)

---
# 전이학습
1. linearProbe 분류 task 성능
2. fine tuning
3. KNN (representation 성능 측정)

---
# 실험순서.
ViT only기반으로 pretrain 4개방식 비교
MAE, DINOv3, ADIOS, DGnet 에대해서 각각 비교해야함.



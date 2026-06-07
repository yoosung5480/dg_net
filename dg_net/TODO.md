# 실험 리스트
해당 프레임워크에선 DGnet모델의 적대적 복원 학습을 통해서 인코더를 훈련시켜서 제공해야한다.

ADIOS논문 재현을 위해서 epoch=200을 맞추고 나머지 하이퍼 파라미터도 맞추기전에, 기본적으로 collpase와 실제로 훈련이 제대로 되고있는지 검증하기 위해서 다른 프레임워크 없이 dg_net 프레임워크만 소규모로 작게 돌려볼것이다.

# environment
environment: pretext_task_gen
python: 3.10.20
torch: 2.2.2+cu121
torchvision: 0.17.2+cu121
cuda_available: True


## 해야할일
1. 현재 프레임워크는 DDP를 통한 병렬 GPU작업이 지원이 안된다. 이것을 해결하라.
2. 현재 프로젝트의 목표함수가 수정됐다. 

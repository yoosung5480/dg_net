# 목표사항
1. dataset.py 구현
2. dataset_config.py 구현
3. **Config를 입력으로 받으면, 데이터 로더를 반환하는 하나의 완결된 독립함수**
### 성공조건 코드
```python
cfg = DataConfig()
loader = prepare_dataloader(cfg)
```

# *_train.py 공통요구사항
데이터셋 경로 : `/home/jeongyuseong/바탕화면/datasets`

###  input
Config를 통해서 모든것이 생성되고 제어돼야한다.


### Config 예시
아래처럼 config를 확정지을 필요는없다. 
1. 데이터셋 선택과 
2. 모드선택 (분류인지, SSL훈련인지), 
3. 배치사이즈 선택
4. 데이터 증강 제어


```python
@dataclass
class DataConfig:
    dataset: str = "STL10"   # "STL10", "CIFAR100", "Flowers102", "iNaturalist"
    data_path: str = "/data/imagenet/train"

    split: str = "train"
    classes: Optional[List[int]] = None
    max_samples: Optional[int] = None

    # -------------------------
    # transform
    # -------------------------
    img_size: int = 224
    use_augmentation: bool = True
    crop_scale: Tuple[float, float] = (0.08, 1.0)
    hflip_prob: float = 0.5
    color_jitter: Optional[Tuple[float, float, float, float]] = None
    gaussian_blur: bool = False
    blur_prob: float = 0.5
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float]  = (0.229, 0.224, 0.225)

    # -------------------------
    # dataloader
    # -------------------------
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True

    # reproducibility
    seed: int = 42
```

## 데이터셋
1. STL10 : pretrain & 분류, 가장중요함
2. CIFAR100 : 분류 전이 학습용
3. Flowers102 : 분류 전이 학습용
4. iNaturalist : 분류 전이 학습용

## 구현요구사항
1. config를 통해서 SSL용 데이터셋으론 STL10의 unlabeled 데이터셋로드하기
2. config를 통해서 각데이터셋에 대해서 구별해서 로드하기 
3. DDP등의 설정으로 다중 GPU지정이 지원되게 하기

`dataset/output/`에 각 데이터셋에 대해 데이터로더 로드 성공내용 출력돼야함
1. 로드된 데이터셋 정보 (데이터셋 개수, 디렉토리 경로, 훈련데이터셋 종류(분류인지, SSL인지)) 
2. 배치 샘플 출력결과 
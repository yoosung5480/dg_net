## 데이터로더 생성 api
```python
cfg = make_config(tmp_path, dataset="CIFAR100", mode="ssl", write_report=False)
dataloader = prepare_dataloader(cfg)
```

### DataConfig 파라미터 설명
```python

DataConfig 사용 설명
====================

가장 짧은 실행 형태
-------------------
``cfg = DataConfig(...)`` 로 모든 데이터 입력/증강/배치 조건을 결정한 뒤,
``loader = prepare_dataloader(cfg)`` 한 번으로 학습에 사용할 DataLoader를 만든다.
이 패키지는 SSL 사전학습과 지도 분류학습을 같은 설정 객체로 다루기 위해 만들어졌다.

데이터셋, mode, split 조합
--------------------------
* ``dataset="STL10"``
  - ``mode="ssl"``: 라벨을 사용하지 않는 사전학습이다. ``split`` 값과 무관하게
    torchvision STL10의 ``"unlabeled"`` split(100,000개 무라벨 이미지)을 읽는다.
    한 이미지에서 ``ssl_num_views``개의 독립 증강 view를 반환한다.
  - ``mode="classification"``: 라벨 기반 학습/평가이다.
    ``split="train"``(5,000개 labeled train) 또는 ``split="test"``(8,000개 test)를 사용한다.
* ``dataset="CIFAR100"`` (분류 전이학습용, ``mode="classification"``만 허용)
  - ``split="train"``: 100개 class의 학습 이미지 50,000개.
  - ``split="test"``: 평가 이미지 10,000개. CIFAR100에는 ``"val"`` split이 없다.
* ``dataset="Flowers102"`` (분류 전이학습용, ``mode="classification"``만 허용)
  - ``split="train"``, ``split="val"``/``"validation"``, ``split="test"``를 지원한다.
    torchvision의 공식 train/validation/test partition을 그대로 사용한다.
* ``dataset="iNaturalist"`` (분류 전이학습용, ``mode="classification"``만 허용)
  - ``split="train"``은 기본적으로 torchvision ``version="2021_train"``을 사용한다.
  - ``split="val"``/``"validation"``은 ``version="2021_valid"``를 사용한다.
  - 다른 torchvision 제공 release를 사용하려면 ``inaturalist_version``을 직접 지정한다.

필드별 사용법
-------------
* ``data_path``: 데이터 저장 상위 디렉터리. 본 프로젝트의 공통 경로
  ``/home/jeongyuseong/바탕화면/datasets``를 넣으면 STL10의 경우
  ``stl10/train/stl10_binary`` 또는 평가용 하위 경로를 자동 탐색한다.
* ``download``: torchvision에 없는 파일을 다운로드할지 결정한다. 서버/오프라인
  환경에서는 이미 준비된 데이터와 ``False`` 사용을 권장한다.
* ``classes``: 분류 dataset에서 사용할 정수 class id만 선택한다. ``None``이면 전체 class.
  STL10 SSL unlabeled pretraining에는 class filtering을 사용하지 않는 것이 일반적이다.
* ``max_samples``: 빠른 smoke test 또는 소규모 실험을 위한 최대 샘플 수.
  ``seed``에 의해 동일 subset이 재현된다.

* ``img_size``: transform 이후 모델 입력의 정사각형 한 변 픽셀 수.
* ``use_augmentation``: ``True``이면 random resized crop/horizontal flip 및 선택 증강을,
  ``False``이면 deterministic resize를 적용한다.
* ``crop_scale``: random resized crop의 최소/최대 면적 비율.
* ``hflip_prob``: horizontal flip 확률이며 0이면 flip을 생략한다.
* ``color_jitter``: ``(brightness, contrast, saturation, hue)`` 튜플이며
  ``None``이면 color jitter를 생략한다.
* ``gaussian_blur`` / ``blur_prob``: blur 증강 사용 여부 및 적용 확률.
* ``mean`` / ``std``: ToTensor 뒤 channel-wise normalization 통계.
* ``ssl_num_views``: SSL 모드에서 같은 원본으로부터 생성할 augmented view 개수.
  contrastive/multi-view 학습을 위해 2 이상이어야 한다.

* ``batch_size``: 한 optimizer step에서 loader가 묶어 반환하는 sample 수.
* ``num_workers``: DataLoader subprocess 개수. 노트북/간단 확인은 0이 안전하다.
* ``shuffle``: 매 epoch sample 순서를 섞을지 결정한다. DDP 사용 시 sampler에 전달된다.
* ``drop_last``: 마지막 불완전 batch를 버릴지 결정한다. SSL batch 통계 안정화에는
  ``True``, 전체 평가에는 ``False``가 일반적이다.
* ``pin_memory``: CUDA 전송을 빠르게 하기 위한 pinned host memory 요청이다.
  실제 CUDA 사용 가능 시에만 DataLoader에 활성화된다.
* ``persistent_workers``: epoch 간 worker 유지 여부. ``True``이면
  반드시 ``num_workers > 0``이어야 한다.
* ``seed``: subset 선택, shuffle, worker randomness의 재현성 seed.

* ``distributed``: DDP/multi-GPU에서 ``DistributedSampler``를 사용할지 결정한다.
* ``rank`` / ``world_size``: torch.distributed가 아직 초기화되지 않은 smoke test 또는
  launcher 외 실행에서 sampler를 만들기 위한 process index/전체 process 수.
  일반 DDP 런타임에서는 초기화된 process group 정보를 자동으로 사용한다.

* ``output_dir``: 성공한 로더 구성과 첫 batch tensor shape를 저장하는 디렉터리.
* ``write_report``: ``True``이면 ``prepare_dataloader`` 실행 직후 텍스트 보고서를 쓴다.

실행 예시의 의도
---------------
프로젝트 루트의 ``test.py``는 실제 설치된 STL10 unlabeled 파일을 이용하는 가장 작은
SSL 확인 스크립트이다. 첫 번째 논리 줄에서 config를 만들고, 두 번째 논리 줄에서
``prepare_dataloader``를 호출하여 ``output/stl10_ssl_unlabeled.txt``를 생성한다.

```

~~~python
root/
    src/
        - model/
            - base_model.py
            - auto_encoder
                model_auto_encoder.py   # MAE 논문 기반의 auto_encoder
                pos_embed.py            # 포지셔널 임베딩위한 모듈
            - degradation               # hybridTransformer 아키텍쳐 - 훈련 목표
                model_degradation.py
                model_masking.py
            - DGnet.py                  # degradation, encoder, decoder, projection 층을 갖는 전체 네트워크
            - 
        - loss/     
            distance/
                - total_distrance.py
                - representation_distance.py
                - degradation_distance.py
                - reconstruction_distance.py
            regluation/
                - degradation_regluation.py
                - representation_regluation.py
            loss_test.ipynb             # 지금 너가 구현한 코드 테스트용 주피터 노트북
        - dataset/
            - datasets.py
            - cls_datasets.py      # TODO : 새로 필요없으면 구현 안해도됨
            - MAE_datasets.py               # TODO : 새로 필요없으면 구현 안해도됨
        - engine/
            - train.py
            - mask_train.py
            - MAE_train.py
            - classification_train.py         # TODO : 필수 구현목표 , loss가 CE로 쉽기때문에 loss모델은 없다. 다만 이 또한 TrainConfig만으로 다 작동하는 api 형태여야한다.
        - util/ 
            - # 근데 이건 나중에 생각하자. 구현 목표에만 집중
        - config/
                - DG_config.py
                - exp_configs.py                                        #  지금 해야할일
        
        test_train.py
        classification_main.py
~~~

## 
이 프레임워크는 실험단계로써, 메모리 최적화는 안돼있다.
지금은 아이디어 점검으로써 더 중요한 시기임으로, 메모리 최적화는 안돼있을거다.


이미지텐서 -> model_auto_encoder -> 

# loss
- loss/DG_learn_loss.py
내부의 distance, regluation에서 원하는 규제함수에 대한 값반환을 담당한다. 그렇다면 언제 어떤값을 쓸건지에 대해서는 모두다 이 모듈에 달려있다. 예를들어 현재 RepresentationRegluation의 값을 우리가 원하는 방향으로 반환할려면, RepresentationRegluation 에서는 우리가 원하는 패널티의 구현을 맡고, DG_learn_loss에는 거기에 해당하는 인자를 넣어주는 로직의 역할일 해야한다.

## distance
- loss/distance/
### RepresentationDistance
- loss/distance/representation_distance.py
~~~python
class RepresentationDistance(nn.Module):
    """
    D_representation = 1 - cosine_similarity(z, z_dg)

    Input:
        z     : [B, D]
        z_dg  : [B, D]

    Output:
        [B]
    """

    def __init__(self, cfg: DistanceConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, z: torch.Tensor, z_dg: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z_dg.ndim != 2:
            raise ValueError(
                f"RepresentationDistance expects 2D tensors [B, D], "
                f"but got z={tuple(z.shape)}, z_dg={tuple(z_dg.shape)}"
            )
        if z.shape != z_dg.shape:
            raise ValueError(
                f"Shape mismatch: z={tuple(z.shape)} vs z_dg={tuple(z_dg.shape)}"
            )

        z = F.normalize(z, dim=-1, eps=self.cfg.eps)
        z_dg = F.normalize(z_dg, dim=-1, eps=self.cfg.eps)

        cos = torch.sum(z * z_dg, dim=-1)   # [B]
        d_rep = 1.0 - cos                   # [B]
        return d_rep
~~~

## regluation
- loss/distance/representation_regluation
~~~python

class RepresentationRegluation(nn.Module):
    """
    R_rep = ReLU(D_r - t) * exp(k * (D_r - t))

    D_r must be batch-wise tensor [B].
    """

    def __init__(self, cfg: RegluationConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, d_rep: torch.Tensor) -> torch.Tensor:
        if d_rep.ndim != 1:
            raise ValueError(f"Expected d_rep [B], got {tuple(d_rep.shape)}")

        diff = d_rep - self.cfg.rep_threshold
        relu = torch.relu(diff)
        penalty = relu * torch.exp(self.cfg.rep_exp_k * relu)
        return penalty
~~~

# engine/train_mask.py
~~~python
############################################ train ###########################################

def train_step(
    model,
    batch,
    ae_loss_fn,
    dg_loss_fn,
    opt_ae,
    opt_dg,
    device,
    train_cfg: TrainConfig,
):
    # =========================
    # 🔥 최초 1회 파라미터 리포트
    # =========================
    raw_model = _unwrap_model(model)
    if not hasattr(raw_model, "_param_reported"):
        _count_and_report_params(raw_model)

    x = _move_batch_to_device(batch, device)

    # =========================
    # 1️⃣ AE step
    # =========================
    _set_dg_trainable(raw_model, train_cfg, False)
    _set_ae_trainable(raw_model, train_cfg, True)

    raw_model.ae_net.train()
    raw_model.dg_net.eval()

    opt_ae.zero_grad()

    out = raw_model(x).to_dict()

    ae_loss, ae_detail = ae_loss_fn(
        pred=out["pred"],
        z_x=out["z_x"],
        z_x_dg=out["z_x_dg"].detach(),
        x=out["x"],
    )

    ae_loss.backward()
    if train_cfg.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(_collect_trainable_parameters(raw_model.ae_net), train_cfg.grad_clip_norm)
    opt_ae.step()

    # =========================
    # 2️⃣ Degradation step
    # =========================
    _set_dg_trainable(raw_model, train_cfg, True)
    _set_ae_trainable(raw_model, train_cfg, False)

    raw_model.dg_net.train()
    raw_model.ae_net.eval()

    opt_dg.zero_grad()

    out = raw_model(x).to_dict()

    dg_loss, dg_detail = dg_loss_fn(
        pred=out["pred"],
        z_x=out["z_x"].detach(),
        z_x_dg=out["z_x_dg"],
        x=out["x"],
        x_dg=out["x_dg"],
        z_mask=out["z_mask"]
    )

    dg_loss.backward()
    if train_cfg.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(_collect_trainable_parameters(raw_model.dg_net), train_cfg.grad_clip_norm)
    opt_dg.step()

    return ae_loss.item(), dg_loss.item(), ae_detail, dg_detail
~~~


# DegradationLearnerNet
- models/masking_learner.py
직접적인 아키텍쳐의 모든 구현과 선언은 내부 모델들에서 이미 다 돼있다.
- self.dg_net = DegradationNet(cfg=cfg.degradation_config)
- self.ae_net = MaskedAutoencoderViT(cfg=cfg.ae_config)

이 모듈은 반드시 결과를 DGOutput으로 반환해야한다.
그래디언트 설정이나 우리 실험 로직에 따른 입출력 반환은 모두 이 모듈에서만 제어된다. 내부 DegradationNet, MaskedAutoencoderViT 에서는 그냥 원래 정해진 구현에 충실한다. 즉, 훈련 변경사항 예를들어 그래디언트 정책이나 반환형식 변환등의 모든 훈련 변경사항은 DegradationLearnerNet 안에서 이뤄져야한다.

DegradationLearnerNet 만이 engine/*train.py 와소통한다.

~~~python 
def forward(self, x):
        '''
        ### 입력
        - x : 원본이미지 텐서 형태로 변환 형태 [B,3,H,W]

        ### 출력
        변경점. dg_net는 이제 이미지 전체 degradation하지 않고 노이즈를 만든다.
        - x         : 원본 입력 이미지 텐서
        - x_dg      : 손상시킨 x, x - x_dg
        - z_x       : 손상시킨 x에 대한 latent vector
        - z_x_dg    : 원본 입력 이미지에 대한 latent vector, encoder(x-dg_net(x))
        - pred_dg_x : 복원된 손상된 x
        - z_mask    : encoder(dg_net(x))
        '''
    x_dg = self.dg_net(x)
    m = x - x_dg

    # reconstruction
    pred_dg_x, _ = self.ae_net(m, mask_ratio=0.0)

    # 기준 latent (완전 고정)
    with torch.no_grad():
        latent_x = self.ae_net.forward_encoder(x, 0.0)[0]

    # DG용 latent (encoder weight만 freeze)
    with self._freeze_ae_encoder():
        latent_x_dg = self.ae_net.forward_encoder(m, 0.0)[0]
        latent_z_mask = self.ae_net.forward_encoder(x_dg, 0.0)[0]

    z_x = latent_x.mean(dim=1)
    z_x_dg = latent_x_dg.mean(dim=1)
    z_mask = latent_z_mask.mean(dim=1)
    return DGOutput(
            pred_dg_x=pred_dg_x,
            z_x=z_x,
            z_x_dg=z_x_dg,
            x_dg=m,
            x=x,
            z_mask=z_mask
        )

''' 
pred     :  torch.Size([1, 196, 768])
z_x      :  torch.Size([1, 768])
z_mask   :  torch.Size([1, 768])
z_x_dg   :  torch.Size([1, 768])
x_dg     :  torch.Size([1, 3, 224, 224])
x        :  torch.Size([1, 3, 224, 224])
'''
~~~



## DegradationNet 
- models/degradation/model_degradation.py or model_mask.py
둘다 사용 api는 똑같음. 

### 전체 forward api
~~~python
def forward(
        self,
        x: torch.Tensor,
        return_delta: bool = False,
        return_patches: bool = False,
    ):
        """
        x: [B, C, H, W]

        return:
            degraded_img
            optionally delta_patches, out_patches
        """
    ...
    outputs = [degraded_img]
    return tuple(outputs)
~~~

## MaskedAutoencoderViT 
- models/auto_encoder/model_audo_encoder.py
model : MaskedAutoencoderViT 기준

### 인코더
해당 모델의 MAE논문 모델을 그래도 가져옴. 우리는 이제 랜덤마스킹 적용안해서 뒤에 _, ids_restore는 안씀. 하지만 api유지해야해서 그냥 저렇게 반환하게 냅둠.

~~~python
model.forward_encoder(x) = z_x, _, ids_restore
~~~

### 디코더
입력은 latentvector를 받아서, 복원된 이미지를 반환함.

~~~python
model.forward_decoder(x) = x
~~~

### 전체 forward
~~~python
# mask_ratio는 이제 안쓰임. 그런데 api 유지하기 위해 냅둠
def forward(self, imgs, mask_ratio=0.75):
    latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    pred = self.forward_decoder(latent, ids_restore)
    return pred, mask
~~~

pred     :  torch.Size([1, 196, 768])
z_x      :  torch.Size([1, 768])
z_x_dg   :  torch.Size([1, 768])
x_dg     :  torch.Size([1, 3, 224, 224])
x        :  torch.Size([1, 3, 224, 224])


# 해야할일
우리는 
1. loss에서
    1. representation_regluation.py 의 구현을 ADIOS 패널티에 맞게바꾸고
    2. DG_learn_loss.py 의 구현을 우리가 원하는 형태의 loss에 맞게 다시 만든다. 입출력 형식은 무조건 LossConfig로 동일한 DG_learn_loss.py와 동일하지만 반환값을 딕셔너리로 받아서 쓸수있다. 우린 ADIOS_loss.py를 만들것이다.
2. DegradationNet 에서 gradinet를 정상 동작할수 있게 제어해야한다. 우린 결론적으로 forward함수 "만"다른, ADIOS_learner.py를 구현할것이다. 
3. ADIOS_train.py를 구현할것이다. 모든 로직과 중간디버깅 및 모든로직이 동일하지만, train_step에서 loss업데이트만 다를것이다.

즉, 이미 구현돼있는 모듈을 복붙해서 그대로 api를 대부분 유지하며, 필요한 부분만 재정의한 모듈을 몇개 만들것이다.
일단 우린 **ADIOS_loss.py와 ADIOS_learner.py** 모듈을 만드는데 집중할것이다. 

- 참고사항
우린 exp_config에서 모든 훈련 하이퍼파라미터를 정의하고 실행한다. 
그럼으로, train_mask에서 보다시피 우린 실험 시작전에 아래를 통해서 필요한 모델은 끌 수 있다.
해당 파라미터 활상화 로직은 *train.py에 있다.
~~~python
@dataclass
class ParamGradConfig:
    train_encoder: bool = True
    train_decoder: bool = True
    train_degradation: bool = True
~~~

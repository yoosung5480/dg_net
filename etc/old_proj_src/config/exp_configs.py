import argparse

from src.config.DG_config import TrainConfig, DGlearnerConfig, DegradationConfig, MAEConfig

# =========================
# 실험 정의
# =========================

def step_base_cfg():
    cfg = TrainConfig()

    cfg.data.dataset = "stl10"
    cfg.data.data_path = "/home/jeongyuseong/바탕화면/datasets/stl10/train"
    cfg.data.split = "unlabeled"
    # cfg.data.classes = [0,1,2]    # imageNet만 가능 ㅋㅎ

    # ===== control (FLAT으로 변경) =====
    cfg.mode = "step"
    cfg.max_iters = 10000
    cfg.epochs = 20

    # ===== runtime (이미 flat 사용 중이라 OK) =====
    cfg.iter_show = 100
    cfg.save_path = "./outputs/gradient_fix_0411_0556"
    cfg.save_interval = 1000
    cfg.log_interval = 100

    # =========================
    # model config
    # =========================
    ''' 
    ========== 🔍 Model Parameter Report ==========
    [Total Model]        total: 26,752,768 | trainable: 26,752,768
    [DG Net]             total: 12,227,712 (45.71%) | trainable: 12,227,712
    [AE Encoder]         total: 10,646,784 (39.80%) | trainable: 10,646,784
    [AE Decoder]         total: 3,159,040 (11.81%) | trainable: 3,159,040
    '''
    cfg.model = DGlearnerConfig(
        degradation_config=DegradationConfig(
            IMG_SIZE=224,
            PATCH_SIZE=16,
            IN_CHANS=3,
            EMBED_DIM=384,
            DEPTH=6,
            NUM_HEADS=8,
        ),
        ae_config=MAEConfig(
            IMG_SIZE=224,
            PATCH_SIZE=16,
            IN_CHANS=3,
            EMBED_DIM=384,
            DEPTH=6,
            NUM_HEADS=6,
            DECODER_EMBED_DIM=256,
            DECODER_DEPTH=4,
            DECODER_NUM_HEADS=4,
        )
    )

    # ===== optimizer =====
    cfg.optimizer.lr_ae = 3e-4
    cfg.optimizer.lr_dg = 3e-4

    cfg.data.batch_size = 64

    # =========================
    # distance
    # =========================
    cfg.loss.distance.alpha_r = 0.0
    cfg.loss.distance.alpha_p = 1.0

    cfg.loss.distance.beta_r = 0.1
    cfg.loss.distance.beta_p = 0.1
    cfg.loss.distance.beta_m = 0.05

    cfg.loss.distance.mask_threshold = 0.75
    cfg.loss.distance.recon_target = 0.25   

    cfg.loss.distance.ae_recon_mode = "D_p"
    cfg.loss.distance.dg_recon_mode = "D_p_hat"

    # =========================
    # regulation
    # =========================
    cfg.loss.regluation.lambda_deg = 0.0
    cfg.loss.regluation.lambda_tv = 0.0
    cfg.loss.regluation.lambda_sparsity = 0.0
    cfg.loss.regluation.lambda_rep = 0.0
    cfg.loss.regluation.rep_threshold = 0.0
    cfg.loss.regluation.rep_exp_k = 3.0

    # =========================
    # scheduler (변경 없음)
    # =========================
    cfg.loss.schedule.distance.alpha_r.enabled = False
    cfg.loss.schedule.distance.alpha_r.mode = "cosine"
    cfg.loss.schedule.distance.alpha_r.start = 0.0
    cfg.loss.schedule.distance.alpha_r.end = 0.0

    cfg.loss.schedule.distance.alpha_p.enabled = False
    cfg.loss.schedule.distance.alpha_p.mode = "cosine"
    cfg.loss.schedule.distance.alpha_p.start = 1.0
    cfg.loss.schedule.distance.alpha_p.end = 1.0

    cfg.loss.schedule.distance.beta_r.enabled = True
    cfg.loss.schedule.distance.beta_r.mode = "cosine"
    cfg.loss.schedule.distance.beta_r.start = 0.1
    cfg.loss.schedule.distance.beta_r.end = 0.1

    cfg.loss.schedule.distance.beta_p.enabled = True
    cfg.loss.schedule.distance.beta_p.mode = "cosine"
    cfg.loss.schedule.distance.beta_p.start = 0.1
    cfg.loss.schedule.distance.beta_p.end = 0.1

    cfg.loss.schedule.distance.beta_m.enabled = True
    cfg.loss.schedule.distance.beta_m.mode = "cosine"
    cfg.loss.schedule.distance.beta_m.start = 0.05
    cfg.loss.schedule.distance.beta_m.end = 0.05

    cfg.loss.schedule.distance.mask_threshold.enabled = False
    cfg.loss.schedule.distance.mask_threshold.mode = "cosine"
    cfg.loss.schedule.distance.mask_threshold.start = 0.75
    cfg.loss.schedule.distance.mask_threshold.end = 0.75

    cfg.loss.schedule.distance.recon_target.enabled = False
    cfg.loss.schedule.distance.recon_target.mode = "cosine"
    cfg.loss.schedule.distance.recon_target.start = 0.25
    cfg.loss.schedule.distance.recon_target.end = 0.25

    cfg.loss.schedule.regluation.lambda_deg.enabled = False
    cfg.loss.schedule.regluation.lambda_deg.mode = "cosine"
    cfg.loss.schedule.regluation.lambda_deg.start = 0.0
    cfg.loss.schedule.regluation.lambda_deg.end = 0.0

    cfg.loss.schedule.regluation.lambda_tv.enabled = False
    cfg.loss.schedule.regluation.lambda_tv.mode = "cosine"
    cfg.loss.schedule.regluation.lambda_tv.start = 0.0
    cfg.loss.schedule.regluation.lambda_tv.end = 0.0

    cfg.loss.schedule.regluation.lambda_sparsity.enabled = False
    cfg.loss.schedule.regluation.lambda_sparsity.mode = "cosine"
    cfg.loss.schedule.regluation.lambda_sparsity.start = 0.0
    cfg.loss.schedule.regluation.lambda_sparsity.end = 0.0

    cfg.loss.schedule.regluation.lambda_rep.enabled = False
    cfg.loss.schedule.regluation.lambda_rep.mode = "cosine"
    cfg.loss.schedule.regluation.lambda_rep.start = 0.0
    cfg.loss.schedule.regluation.lambda_rep.end = 0.0

    cfg.loss.schedule.regluation.rep_threshold.enabled = False
    cfg.loss.schedule.regluation.rep_threshold.mode = "cosine"
    cfg.loss.schedule.regluation.rep_threshold.start = 0.0
    cfg.loss.schedule.regluation.rep_threshold.end = 0.0

    cfg.loss.schedule.regluation.rep_exp_k.enabled = False
    cfg.loss.schedule.regluation.rep_exp_k.mode = "cosine"
    cfg.loss.schedule.regluation.rep_exp_k.start = 3.0
    cfg.loss.schedule.regluation.rep_exp_k.end = 3.0

    return cfg


def only_advarsarial_loss_no_scheduled_target_threshold():
    cfg = step_base_cfg()
    cfg.save_path = "./outputs/only_advarsarial_loss_no_scheduled_target_threshold"

    cfg.loss.schedule.distance.beta_p.enabled = False
    cfg.loss.schedule.distance.beta_p.mode = "cosine"
    cfg.loss.schedule.distance.beta_p.start = 0.1
    cfg.loss.schedule.distance.beta_p.end = 0.1

    cfg.loss.schedule.distance.recon_target.enabled = False
    cfg.loss.schedule.distance.recon_target.mode = "cosine"
    cfg.loss.schedule.distance.recon_target.start = 1.0
    cfg.loss.schedule.distance.recon_target.end = 1.0

    cfg.loss.distance.beta_r = 0.0
    cfg.loss.schedule.distance.beta_r.enabled = False
    cfg.loss.schedule.distance.beta_r.mode = "cosine"
    cfg.loss.schedule.distance.beta_r.start = 0.0
    cfg.loss.schedule.distance.beta_r.end = 0.0

    cfg.loss.distance.beta_m = 0.00
    cfg.loss.schedule.distance.mask_threshold.enabled = False
    cfg.loss.schedule.distance.mask_threshold.mode = "cosine"
    cfg.loss.schedule.distance.mask_threshold.start = 0.0
    cfg.loss.schedule.distance.mask_threshold.end = 0.0

    cfg.loss.schedule.distance.beta_m.enabled = False
    cfg.loss.schedule.distance.beta_m.mode = "cosine"
    cfg.loss.schedule.distance.beta_m.start = 0.0
    cfg.loss.schedule.distance.beta_m.end = 0.0
    return cfg

def only_advarsarial_loss_no_scheduled_target_threshold_25():
    cfg = only_advarsarial_loss_no_scheduled_target_threshold()
    cfg.loss.distance.recon_target = 0.25   
    cfg.save_path = "./outputs/only_advarsarial_loss_no_scheduled_target_threshold_25"
    cfg.loss.schedule.distance.recon_target.enabled = False
    cfg.loss.schedule.distance.recon_target.mode = "cosine"
    cfg.loss.schedule.distance.recon_target.start = 0.25
    cfg.loss.schedule.distance.recon_target.end = 0.25
    return cfg

def only_advarsarial_loss_no_scheduled_target_threshold_50():
    cfg = only_advarsarial_loss_no_scheduled_target_threshold()
    cfg.loss.distance.recon_target = 0.5   
    cfg.save_path = "./outputs/only_advarsarial_loss_no_scheduled_target_threshold_50"
    cfg.loss.schedule.distance.recon_target.enabled = False
    cfg.loss.schedule.distance.recon_target.mode = "cosine"
    cfg.loss.schedule.distance.recon_target.start = 0.5
    cfg.loss.schedule.distance.recon_target.end = 0.5
    return cfg

def only_advarsarial_loss_no_scheduled_target_threshold_75():
    cfg = only_advarsarial_loss_no_scheduled_target_threshold()
    cfg.loss.distance.recon_target = 0.75   
    cfg.save_path = "./outputs/only_advarsarial_loss_no_scheduled_target_threshold_75"
    cfg.loss.schedule.distance.recon_target.enabled = False
    cfg.loss.schedule.distance.recon_target.mode = "cosine"
    cfg.loss.schedule.distance.recon_target.start = 0.75
    cfg.loss.schedule.distance.recon_target.end = 0.75
    return cfg


def only_advarsarial_loss_scheduled_target_threshold():
    cfg = step_base_cfg()
    cfg.save_path = "./outputs/only_advarsarial_loss_scheduled_target_threshold"

    cfg.loss.schedule.distance.beta_p.enabled = False
    cfg.loss.schedule.distance.beta_p.mode = "cosine"
    cfg.loss.schedule.distance.beta_p.start = 0.1
    cfg.loss.schedule.distance.beta_p.end = 0.1

    cfg.loss.schedule.distance.recon_target.enabled = True
    cfg.loss.schedule.distance.recon_target.mode = "cosine"
    cfg.loss.schedule.distance.recon_target.start = 0.01
    cfg.loss.schedule.distance.recon_target.end = 1.0

    cfg.loss.distance.beta_r = 0.0
    cfg.loss.schedule.distance.beta_r.enabled = False
    cfg.loss.schedule.distance.beta_r.mode = "cosine"
    cfg.loss.schedule.distance.beta_r.start = 0.0
    cfg.loss.schedule.distance.beta_r.end = 0.0

    cfg.loss.distance.beta_m = 0.00
    cfg.loss.schedule.distance.mask_threshold.enabled = False
    cfg.loss.schedule.distance.mask_threshold.mode = "cosine"
    cfg.loss.schedule.distance.mask_threshold.start = 0.0
    cfg.loss.schedule.distance.mask_threshold.end = 0.0

    cfg.loss.schedule.distance.beta_m.enabled = True
    cfg.loss.schedule.distance.beta_m.mode = "cosine"
    cfg.loss.schedule.distance.beta_m.start = 0.0
    cfg.loss.schedule.distance.beta_m.end = 0.0
    return cfg


def step_no_mask_distance():
    cfg = step_base_cfg()
    cfg.save_path = "./outputs/no_mask_distance_0411_1412"

    cfg.loss.distance.beta_p = 0.1
    cfg.loss.schedule.distance.beta_p.enabled = True
    cfg.loss.schedule.distance.beta_p.mode = "cosine"
    cfg.loss.schedule.distance.beta_p.start = 0.1
    cfg.loss.schedule.distance.beta_p.end = 0.1


    cfg.loss.distance.beta_r = 0.0
    cfg.loss.schedule.distance.beta_r.enabled = True
    cfg.loss.schedule.distance.beta_r.mode = "cosine"
    cfg.loss.schedule.distance.beta_r.start = 0.0
    cfg.loss.schedule.distance.beta_r.end = 0.0

    cfg.loss.distance.beta_m = 0.00
    cfg.loss.schedule.distance.mask_threshold.enabled = False
    cfg.loss.schedule.distance.mask_threshold.mode = "cosine"
    cfg.loss.schedule.distance.mask_threshold.start = 0.0
    cfg.loss.schedule.distance.mask_threshold.end = 0.0

    cfg.loss.schedule.distance.beta_m.enabled = True
    cfg.loss.schedule.distance.beta_m.mode = "cosine"
    cfg.loss.schedule.distance.beta_m.start = 0.0
    cfg.loss.schedule.distance.beta_m.end = 0.0
    return cfg



def epochs_base_cfg():
    cfg = step_base_cfg()

    # ===== control (FLAT) =====
    cfg.mode = "epoch"
    cfg.epochs = 20
    cfg.max_iters = None

    return cfg



# =========================
# registry
# =========================

EXP_REGISTRY = {
    "step_base": step_base_cfg,
    "step_no_mask_distance": step_no_mask_distance,
    "epoch_base" : epochs_base_cfg,
    "only_advarsarial_loss_no_scheduled_target_threshold": only_advarsarial_loss_no_scheduled_target_threshold,
    "only_advarsarial_loss_scheduled_target_threshold" : only_advarsarial_loss_scheduled_target_threshold,
    "only_advarsarial_loss_no_scheduled_target_threshold_25" : only_advarsarial_loss_no_scheduled_target_threshold_25,
    "only_advarsarial_loss_no_scheduled_target_threshold_50" : only_advarsarial_loss_no_scheduled_target_threshold_50,
    "only_advarsarial_loss_no_scheduled_target_threshold_75" : only_advarsarial_loss_no_scheduled_target_threshold_75,
}


# =========================
# argparse
# =========================

def build_train_cfg(args):
    if args.exp_name not in EXP_REGISTRY:
        raise ValueError(f"Unknown exp: {args.exp_name}")

    cfg = EXP_REGISTRY[args.exp_name]()
    return cfg


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="step_base")
    return parser

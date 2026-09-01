"""
CAT-GS configuration.

CAT-GS: Balanced Multimodal Learning via Calibrated Gating and Fusion Surgery.

All experiment settings live in this file. The typical workflow is:
    1. Set `dataset`, `exp_name`, and the data paths below.
    2. Train the unimodal teachers:   `exp_name = 'aT'` / `'vT'`  ->  python teacher.py
    3. Point `TEACHER_WEIGHTS` at the saved teacher checkpoints.
    4. Train the multimodal student:  `exp_name = 'aT+aTF+vT+vTF-to-mS'`  ->  python student.py

Symbols in comments refer to the paper:
    tau_low / tau_high  : regime thresholds on the reliability margin Delta (Eq. 5-6)
    E_w, p_drop         : warm-up dropout epochs and probability (Sec. 3.4, regime 1)
    lambda_bias         : weak-modality bias factor (Eq. 8)
    epsilon             : gating floor in soft regimes (Sec. 3.4)
    beta                : EMA momentum for teacher reliability (Eq. 4)
    beta_g              : EMA momentum for the gradient-norm budget (Eq. 9)
    gamma_cap           : cap on gradient-budget renormalization (Eq. 10)
    delta               : numerical-stability constant (Eq. 10, 12)
"""

import os
import torch

# =====================================================================
# 1. Experiment selection
# =====================================================================

DEBUG = False  # True = 1 epoch on 10 samples (smoke test)

# Random seed. The paper reports mean +/- std over seeds {42, 123, 999}.
random_seed = 42

# What to train:
#   'aT'                  -> audio  teacher (unimodal)
#   'vT'                  -> visual teacher (unimodal)
#   'aT+aTF+vT+vTF-to-mS' -> multimodal student distilled from both teachers
#                            (teacher logits + teacher features), the paper's default
#   'aTF+vTF-to-mS'       -> student with feature distillation only
exp_name = 'aT+aTF+vT+vTF-to-mS'

# Backward-pass controller:
#   'Normal' -> Joint-Train baseline (no modulation)
#   'OGM' / 'OGM-GE' -> Peng et al., CVPR 2022
#   'G2D'    -> confidence-gated distillation baseline
#   'CAT-GS' -> ours
modulation = 'CAT-GS'

# Benchmark: 'CREMAD' | 'AVMNIST' | 'VGGSound' | 'AVE' | 'CGMNIST'
dataset = 'CREMAD'

# Fusion head for the multimodal student (Table "fusion strategies"):
#   'late' (paper default / best) | 'concat' | 'sum' | 'film' | 'gated'
fusion_method_choice = 'late'

# =====================================================================
# 2. Paths (edit these for your machine)
# =====================================================================

DATA_DIR = './data'          # root that holds one folder per dataset
CKPT_DIR = './checkpoints'   # where teacher/student checkpoints are written
SCORES_DIR = './scores'      # per-step CSV logs (gating regime, fusion cosine, ...)

if dataset == 'CREMAD':
    train_path = f'{DATA_DIR}/CREMAD/train.csv'
    test_path = f'{DATA_DIR}/CREMAD/test.csv'
    visual_path = f'{DATA_DIR}/CREMAD/Image-01-FPS'   # frames extracted at 1 fps
    audio_path = f'{DATA_DIR}/CREMAD/AudioWAV'
elif dataset == 'VGGSound':
    vggsound_csv = f'{DATA_DIR}/VGGSound/vggsound.csv'
    vggsound_video_root = f'{DATA_DIR}/VGGSound/video/frames'
    vggsound_audio_root = f'{DATA_DIR}/VGGSound/audio'
elif dataset == 'AVMNIST':
    data_root = f'{DATA_DIR}/AVMNIST'
elif dataset == 'CGMNIST':
    data_root = f'{DATA_DIR}/Mnist'
elif dataset == 'AVE':
    data_root = f'{DATA_DIR}/AVE_Dataset'
    visual_path = f'{DATA_DIR}/AVE_Dataset'                 # holds Image-XX-FPS-SE folders
    audio_path = f'{DATA_DIR}/AVE_Dataset/Audio-1004-SE'    # precomputed spectrogram .pkl
    train_txt = f'{DATA_DIR}/AVE_Dataset/trainSet.txt'
    test_txt = f'{DATA_DIR}/AVE_Dataset/testSet.txt'
    val_txt = f'{DATA_DIR}/AVE_Dataset/valSet.txt'

# Sequence-feature benchmarks trained via train_multimodal.py (not `dataset` above):
# UR-FUNNY (tri-modal humor) and CMU-MOSI (sentiment) with Transformer encoders.
urfunny_data_dir = f'{DATA_DIR}/URFUNNY'            # converted pickles (see dataset/URFunnyDataset.py)
mosi_data_path = f'{DATA_DIR}/MOSI/mosi_raw.pkl'    # MultiBench-format pickle

# Frozen unimodal teacher checkpoints used by the student (fill in after step 2).
TEACHER_WEIGHTS = {
    'CREMAD': {
        'audio': f'{CKPT_DIR}/CREMAD/teacher/best_audio_teacher.pth',
        'visual': f'{CKPT_DIR}/CREMAD/teacher/best_visual_teacher.pth',
    },
    'AVMNIST': {
        'audio': f'{CKPT_DIR}/AVMNIST/teacher/best_audio_teacher.pth',
        'visual': f'{CKPT_DIR}/AVMNIST/teacher/best_visual_teacher.pth',
    },
    'VGGSound': {
        'audio': f'{CKPT_DIR}/VGGSound/teacher/best_audio_teacher.pth',
        'visual': f'{CKPT_DIR}/VGGSound/teacher/best_visual_teacher.pth',
    },
    'AVE': {
        'audio': f'{CKPT_DIR}/AVE/teacher/best_audio_teacher.pth',
        'visual': f'{CKPT_DIR}/AVE/teacher/best_visual_teacher.pth',
    },
    'CGMNIST': {
        'audio': f'{CKPT_DIR}/CGMNIST/teacher/best_gray_teacher.pth',
        'visual': f'{CKPT_DIR}/CGMNIST/teacher/best_color_teacher.pth',
    },
}

# Resume an interrupted run (checkpoints are saved periodically during training).
resume_training = False
resume_checkpoint_path = None  # e.g. f'{CKPT_DIR}/CREMAD/student/checkpoint_epoch_100.pth'

# =====================================================================
# 3. Optimization (paper Sec. "Training Details")
# =====================================================================

batch_size = 16
epochs = 400
optimiser = 'sgd'        # 'sgd' (momentum 0.9) or 'adam'
learning_rate = 1e-3
lr_decay_step = 200      # StepLR: decay at epoch 200
lr_decay_ratio = 0.1
weight_decay = 1e-4

# =====================================================================
# 4. Distillation objective (Eq. 1-2)
# =====================================================================

ce_loss_weight = 1.0            # lambda_CE
logit_audio_loss_weight = 1.0   # lambda_KD^a  (KL on teacher logits)
logit_video_loss_weight = 1.0   # lambda_KD^v
audio_feature_loss_weight = 1.0  # lambda_feat^a (MSE on teacher features)
video_feature_loss_weight = 1.0  # lambda_feat^v
temp_audio = 1                  # KD temperature tau_a
temp_video = 1                  # KD temperature tau_v

# =====================================================================
# 5. Modulation window (shared by OGM-GE / G2D / CAT-GS)
# =====================================================================

alpha_modulation = 1.0   # OGM-GE strength; always 1.0 for G2D / CAT-GS
modulation_starts = 0    # first epoch with gradient modulation
modulation_ends = 150    # last epoch with gradient modulation

# =====================================================================
# 6. CAT-GS controller
# =====================================================================
# Fixed across all datasets in the paper (Sec. "Training Details"):
#   beta = beta_g = 0.9, gamma_cap = 1.5, epsilon = 0.1,
#   lambda_bias = 0.2, E_w = 5, p_drop = 0.6.
# Only (tau_low, tau_high) are tuned once per dataset; a practical recipe is
# in the paper's threshold-sensitivity analysis: log the reliability margin
# Delta over a 3-5 epoch pilot, set tau_low near its 40th percentile and
# tau_high near its 80th percentile, keeping a gap of at least 0.05.

# Regime thresholds (tau_low, tau_high) per dataset; fallback is (0.05, 0.15).
CATGS_THRESHOLDS = {
    'CREMAD':   (0.05, 0.15),
    'AVMNIST':  (0.05, 0.15),
    'VGGSound': (0.05, 0.15),
    'AVE':      (0.05, 0.15),
    'CGMNIST':  (0.05, 0.15),
}
catgs_tau_low, catgs_tau_high = CATGS_THRESHOLDS.get(dataset, (0.05, 0.15))

catgs_warmup_epochs = 5             # E_w   : warm-up dropout epochs
catgs_dropout_prob = 0.6            # p_drop: modality-dropout probability in warm-up
catgs_weak_bias = 0.2               # lambda_bias in [0.1, 0.3]
catgs_eps_floor = 0.1               # epsilon: minimum gating weight in soft regimes
catgs_reliability_ema_beta = 0.9    # beta  : EMA momentum for teacher reliability (Eq. 4)
catgs_gradnorm_ema_beta = 0.9       # beta_g: EMA momentum for gradient budget (Eq. 9)
catgs_gradnorm_cap = 1.5            # gamma_cap: renormalization cap (Eq. 10)
catgs_delta = 1e-8                  # delta : numerical-stability constant

# --- Component switches (ablation Table "component-wise contribution") ---
use_calibrated_T = True       # temperature-scaled teacher confidence (Eq. 3)
use_ema = True                # EMA smoothing of reliabilities (Eq. 4)
use_weak_bias = True          # weak-biased blending in the soft regime (Eq. 7-8)
use_gradnorm_realloc = True   # gradient-budget reallocation (Eq. 9-10)
use_pcgrad = True             # fusion-only PCGrad surgery (Eq. 11-13)

# --- Reliability calibration temperatures (T_m in Eq. 3; paper default 1.0) ---
T_cal_audio = 1.0
T_cal_video = 1.0

# --- Teacher-miscalibration stress test (Table "teacher miscalibration") ---
# Extra temperature mismatch applied to teacher logits BEFORE CAT-GS reads them.
#   1.0 = calibrated baseline (R0)      <1.0 = overconfident teacher
#   >1.0 = underconfident teacher       e.g. R1: (0.5, 1.0), R2: (2.0, 1.0), R3: (2.0, 0.5)
catgs_teacher_mismatch_T_audio = 1.0
catgs_teacher_mismatch_T_video = 1.0

# =====================================================================
# 7. Data dimensions
# =====================================================================

img_width = 224
img_height = 224
spectrogram_width = 512
spectrogram_height = 128
fps = 1                # frames per second extracted for CREMA-D / AVE
use_video_frames = 3   # VGGSound frames per clip
num_frame = 10         # AVE frames per clip

# =====================================================================
# 8. Derived settings -- no edits needed below this line
# =====================================================================

# Role and modality follow from exp_name.
if exp_name == 'aT':
    modality = 'audio'
elif exp_name == 'vT':
    modality = 'visual'
elif exp_name == 'mT' or 'mS' in exp_name:
    modality = 'multimodal'
else:
    raise NotImplementedError(f"Incorrect experiment name {exp_name}")

role = 'teacher' if exp_name in ('aT', 'vT', 'mT') else 'student'
fusion_method = fusion_method_choice if modality == 'multimodal' else None

if DEBUG:
    epochs = 1

# Teacher checkpoints for the current dataset (used by student.py).
if role == 'student':
    if 'aT' in exp_name:
        audio_teacher_weights = TEACHER_WEIGHTS[dataset]['audio']
    if 'vT' in exp_name:
        video_teacher_weights = TEACHER_WEIGHTS[dataset]['visual']

# Run identifier used for checkpoints, logs, and score files.
if modulation == 'CAT-GS':
    modulation_string = (
        f"{modulation}_tauL{catgs_tau_low}_tauH{catgs_tau_high}"
        f"_warmup{catgs_warmup_epochs}_drop{catgs_dropout_prob}"
        f"_bias{catgs_weak_bias}_eps{catgs_eps_floor}_cap{catgs_gradnorm_cap}"
        f"_tmisA{catgs_teacher_mismatch_T_audio}_tmisV{catgs_teacher_mismatch_T_video}"
    )
elif modulation == 'Normal':
    modulation_string = modulation
else:
    modulation_string = f"{modulation}_alpha_{alpha_modulation}_modS_{modulation_starts}_modE_{modulation_ends}"

modulation_string = (
    f"{modulation_string}_lr_{learning_rate}_lr-dstep_{lr_decay_step}"
    f"_lr-dratio_{lr_decay_ratio}_bt_{batch_size}_{optimiser}_wd_{weight_decay}"
)

# Checkpoint locations.
ckpt_path = f'{CKPT_DIR}/{dataset}/{role}'
os.makedirs(ckpt_path, exist_ok=True)

if modality == 'multimodal':
    ckpt_string = f'best_model_s{random_seed}_{exp_name}_{modulation_string}_{fusion_method}_{dataset}_{role}'
else:
    ckpt_string = f'best_model_s{random_seed}_{exp_name}_{dataset}_{role}'
if DEBUG:
    ckpt_string = 'DEBUG_' + ckpt_string
best_model_path = os.path.join(ckpt_path, ckpt_string + '.pth')

# Score / diagnostics CSVs (gating regime, alphas, fusion-gradient cosine, ...).
scores_path = f'{SCORES_DIR}/{dataset}'
os.makedirs(scores_path, exist_ok=True)
filename = f'scores_{dataset}_{modulation_string}.csv'

# KD-temperature tag used in student checkpoint names.
if role == 'student':
    if 'aT' in exp_name and 'vT' in exp_name:
        temperature = f"ta{temp_audio}_tv{temp_video}"
    elif 'aT' in exp_name:
        temperature = f"ta{temp_audio}"
    elif 'vT' in exp_name:
        temperature = f"tv{temp_video}"
    else:
        temperature = ''

# Device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    gpu_ids = list(range(torch.cuda.device_count()))
    device_count = len(gpu_ids)
else:
    gpu_ids = None
    device_count = 1


def get_hparams():
    """All relevant hyperparameters as a flat dict (for TensorBoard logging)."""
    hparams = {
        'exp_name': exp_name,
        'modulation': modulation,
        'dataset': dataset,
        'role': role,
        'modality': modality,
        'fusion_method': str(fusion_method),
        'batch_size': batch_size,
        'epochs': epochs,
        'optimiser': optimiser,
        'learning_rate': learning_rate,
        'lr_decay_step': lr_decay_step,
        'lr_decay_ratio': lr_decay_ratio,
        'weight_decay': weight_decay,
        'random_seed': random_seed,
        'modulation_starts': modulation_starts,
        'modulation_ends': modulation_ends,
        'device': str(device),
        'best_model_path': str(best_model_path),
    }
    if modulation == 'CAT-GS':
        hparams.update({
            'catgs_tau_low': catgs_tau_low,
            'catgs_tau_high': catgs_tau_high,
            'catgs_warmup_epochs': catgs_warmup_epochs,
            'catgs_dropout_prob': catgs_dropout_prob,
            'catgs_weak_bias': catgs_weak_bias,
            'catgs_eps_floor': catgs_eps_floor,
            'catgs_reliability_ema_beta': catgs_reliability_ema_beta,
            'catgs_gradnorm_ema_beta': catgs_gradnorm_ema_beta,
            'catgs_gradnorm_cap': catgs_gradnorm_cap,
            'use_calibrated_T': use_calibrated_T,
            'use_ema': use_ema,
            'use_weak_bias': use_weak_bias,
            'use_gradnorm_realloc': use_gradnorm_realloc,
            'use_pcgrad': use_pcgrad,
            'T_cal_audio': T_cal_audio,
            'T_cal_video': T_cal_video,
            'catgs_teacher_mismatch_T_audio': catgs_teacher_mismatch_T_audio,
            'catgs_teacher_mismatch_T_video': catgs_teacher_mismatch_T_video,
        })
    if role == 'student':
        hparams.update({
            'ce_loss_weight': ce_loss_weight,
            'logit_audio_loss_weight': logit_audio_loss_weight,
            'audio_feature_loss_weight': audio_feature_loss_weight,
            'logit_video_loss_weight': logit_video_loss_weight,
            'video_feature_loss_weight': video_feature_loss_weight,
            'temp_audio': temp_audio,
            'temp_video': temp_video,
        })
    return hparams


def print_hparams():
    print("\nHyperparameters:")
    for key, value in get_hparams().items():
        print(f"  {key}: {value}")
    print()


print_hparams()

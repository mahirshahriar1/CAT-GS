# CAT-GS: Balanced Multimodal Learning via Calibrated Gating and Fusion Surgery

Official PyTorch implementation of **CAT-GS**, an optimization-stage controller for balanced multimodal learning. It operates entirely between `loss.backward()` and `optimizer.step()` — no changes to architectures, fusion modules, or task losses — combining calibrated reliability estimation, margin-thresholded adaptive gating, gradient-budget reallocation, and fusion-only PCGrad.

## Environment

Python 3.11, PyTorch 2.0.1 / Torchvision 0.15.2 (CUDA 11.7)

```bash
pip install torch torchvision numpy scipy librosa pillow tqdm tensorboard
```

## Datasets

Place datasets under `./data/` (layouts in [data/README.md](data/README.md)), or edit the paths in `config.py`. Supported: CREMA-D, AV-MNIST, VGGSound, AVE, CG-MNIST, UR-FUNNY ([raw pickles](https://github.com/ROC-HCI/UR-FUNNY), convert with `python -m dataset.URFunnyDataset`), and CMU-MOSI ([MultiBench](https://github.com/pliang279/MultiBench) `mosi_raw.pkl`).

## Training

All settings live in `config.py` (dataset, seed, fusion head, and every CAT-GS hyperparameter from the paper). Then:

```bash
# 1. Unimodal teachers (exp_name = 'aT', then 'vT')
python teacher.py

# 2. Multimodal student with CAT-GS
#    (exp_name = 'aT+aTF+vT+vTF-to-mS', point TEACHER_WEIGHTS at the teacher checkpoints)
python student.py
```

Set `modulation` to `'Normal'`, `'OGM-GE'`, or `'G2D'` for the baselines.

**UR-FUNNY / CMU-MOSI** (Transformer encoders, M-modality CAT-GS):

```bash
python train_multimodal.py --dataset urfunny --stage teacher --modalities audio visual text
python train_multimodal.py --dataset urfunny --stage student --modalities audio visual text
python train_multimodal.py --dataset mosi    --stage student --modalities visual text
```

Paper defaults: `β = β_g = 0.9`, `γ_cap = 1.5`, `ε = 0.1`, `λ_bias = 0.2`, `E_w = 5`, `p_drop = 0.6` fixed across datasets; only `(τ_low, τ_high)` are tuned per dataset (default `(0.05, 0.15)`). Results are averaged over seeds 42, 123, 999. Checkpoints go to `./checkpoints/`, controller diagnostics to `./scores/`.

## Citation

```bibtex
@article{tamim2026cat,
  title={CAT-GS: Balanced Multimodal Learning via Calibrated Gating and Fusion Surgery},
  author={Tamim, Mahir Shahriar and Khan, Sharjil and Alim, Md Samiul and Khan, Tanvir Ahmed and Rahman, Shafin and Mohammed, Nabeel},
  journal={arXiv preprint arXiv:2608.24947},
  year={2026}
}
```

## License

[MIT](LICENSE). Builds on the pipelines of [G²D](https://github.com/rAIson-Lab/G2D) and [OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022).

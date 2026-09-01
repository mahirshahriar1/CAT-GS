"""
CAT-GS training on the sequence-feature benchmarks: UR-FUNNY and CMU-MOSI.

These datasets use pre-extracted per-frame/word features and lightweight
Transformer encoders (models/transformer_model.py) instead of the ResNet
pipeline in teacher.py / student.py. CAT-GS itself is the same controller,
generalized to M modalities (utils/catgs_multimodal.py).

Usage (paper settings):
    # 1. Train the unimodal teachers
    python train_multimodal.py --dataset urfunny --stage teacher --modalities audio visual text
    python train_multimodal.py --dataset mosi    --stage teacher --modalities visual audio text

    # 2. Train the multimodal student with CAT-GS
    python train_multimodal.py --dataset urfunny --stage student --modalities audio visual text
    python train_multimodal.py --dataset urfunny --stage student --modalities audio text
    python train_multimodal.py --dataset mosi    --stage student --modalities visual text
    python train_multimodal.py --dataset mosi    --stage student --modalities visual audio text

Hyperparameters default to config.py (shared CAT-GS settings) and the
transformer settings below (hidden 768, 4 layers, 8 heads, dropout 0.1).
"""

import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from dataset.URFunnyDataset import URFunnyDataset, collate_multimodal
from dataset.MOSIDataset import MOSIDataset
from models.transformer_model import MultimodalTransformerClassifier
from utils.catgs_multimodal import (CATGSGateM, evaluate_model,
                                    train_student_epoch, train_teacher_epoch)
from utils.utils import setup_seed


def build_datasets(args):
    if args.dataset == 'urfunny':
        data_dir = args.data_path or getattr(config, 'urfunny_data_dir', './data/URFUNNY')
        make = lambda split: URFunnyDataset(data_dir, split=split, modalities=args.modalities)
    else:  # mosi
        data_path = args.data_path or getattr(config, 'mosi_data_path', './data/MOSI/mosi_raw.pkl')
        make = lambda split: MOSIDataset(data_path, split=split, modalities=args.modalities)
    return make('train'), make('dev'), make('test')


def make_model(modalities, feature_dims, args, device):
    return MultimodalTransformerClassifier({
        'modalities': modalities,
        'feature_dims': feature_dims,
        'num_classes': 2,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'fusion_type': 'late',
    }).to(device)


def loop(train_ds, dev_ds, test_ds, args):
    device = config.device
    save_dir = os.path.join(config.CKPT_DIR, args.dataset.upper(), args.stage)
    os.makedirs(save_dir, exist_ok=True)
    tag = '-'.join(m[0] for m in args.modalities).upper()  # e.g. A-V-T

    loaders = {
        'train': DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_multimodal),
        'dev': DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_multimodal),
        'test': DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_multimodal),
    }

    if args.stage == 'teacher':
        # Train one unimodal teacher per requested modality.
        for mod in args.modalities:
            print(f"\n=== Training {mod} teacher on {args.dataset} ===")
            model = make_model([mod], train_ds.feature_dims, args, device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
            best_acc, best_path = 0.0, os.path.join(save_dir, f'best_{mod}_teacher.pth')
            for epoch in range(args.epochs):
                tr_loss, tr_acc = train_teacher_epoch(model, loaders['train'], optimizer, device)
                dev_loss, dev_acc = evaluate_model(model, loaders['dev'], device)
                print(f"[{mod}] epoch {epoch + 1}: train {tr_acc:.2f}% | dev {dev_acc:.2f}%")
                if dev_acc >= best_acc:
                    best_acc = dev_acc
                    torch.save({'model_state_dict': model.state_dict(),
                                'feature_dims': train_ds.feature_dims,
                                'dev_accuracy': dev_acc, 'epoch': epoch}, best_path)
            model.load_state_dict(torch.load(best_path)['model_state_dict'])
            _, test_acc = evaluate_model(model, loaders['test'], device)
            print(f"[{mod}] best dev {best_acc:.2f}% | test {test_acc:.2f}% -> {best_path}")
        return

    # --- Student with CAT-GS ------------------------------------------------
    teacher_dir = args.teacher_dir or os.path.join(config.CKPT_DIR, args.dataset.upper(), 'teacher')
    teachers = {}
    for mod in args.modalities:
        path = os.path.join(teacher_dir, f'best_{mod}_teacher.pth')
        teacher = make_model([mod], train_ds.feature_dims, args, device)
        teacher.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])
        teacher.eval()
        teachers[mod] = teacher
        print(f"Loaded {mod} teacher from {path}")

    student = make_model(args.modalities, train_ds.feature_dims, args, device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    catgs_gate = CATGSGateM(
        modalities=args.modalities,
        tau_low=config.catgs_tau_low,
        tau_high=config.catgs_tau_high,
        warmup_epochs=config.catgs_warmup_epochs,
        dropout_prob=config.catgs_dropout_prob,
        weak_bias=config.catgs_weak_bias,
        eps_floor=config.catgs_eps_floor,
        reliability_ema_beta=config.catgs_reliability_ema_beta,
        gradnorm_ema_beta=config.catgs_gradnorm_ema_beta,
        gradnorm_cap=config.catgs_gradnorm_cap,
        delta=config.catgs_delta,
        use_calibrated_T=config.use_calibrated_T,
        use_ema=config.use_ema,
        use_weak_bias=config.use_weak_bias,
        use_gradnorm_realloc=config.use_gradnorm_realloc,
        use_pcgrad=config.use_pcgrad and not args.no_pcgrad,
    )

    train_config = {
        'feat_loss_weight': config.audio_feature_loss_weight,
        'logit_loss_weight': config.logit_audio_loss_weight,
        'temperature': config.temp_audio,
        'use_catgs': not args.no_catgs,
        'modulation_starts': config.modulation_starts,
        'modulation_ends': config.modulation_ends,
    }

    best_acc = 0.0
    best_path = os.path.join(save_dir, f'best_student_{tag}_s{config.random_seed}.pth')
    for epoch in range(args.epochs):
        metrics = train_student_epoch(student, teachers, loaders['train'], optimizer,
                                      device, train_config, catgs_gate, epoch=epoch)
        dev_loss, dev_acc = evaluate_model(student, loaders['dev'], device)
        print(f"epoch {epoch + 1}: train {metrics['accuracy']:.2f}% "
              f"(task {metrics['task_loss']:.3f}, feat {metrics['feat_loss']:.3f}, "
              f"logit {metrics['logit_loss']:.3f}) | dev {dev_acc:.2f}% "
              f"| regimes {metrics['catgs_regimes']}")
        if dev_acc >= best_acc:
            best_acc = dev_acc
            torch.save({'model_state_dict': student.state_dict(),
                        'feature_dims': train_ds.feature_dims,
                        'modalities': args.modalities,
                        'dev_accuracy': dev_acc, 'epoch': epoch,
                        'catgs_history': catgs_gate.history}, best_path)

    student.load_state_dict(torch.load(best_path)['model_state_dict'])
    _, test_acc = evaluate_model(student, loaders['test'], device)
    print(f"\nBest dev {best_acc:.2f}% | test {test_acc:.2f}% -> {best_path}")


def main():
    parser = argparse.ArgumentParser(description='CAT-GS on UR-FUNNY / CMU-MOSI')
    parser.add_argument('--dataset', choices=['urfunny', 'mosi'], required=True)
    parser.add_argument('--stage', choices=['teacher', 'student'], required=True)
    parser.add_argument('--modalities', nargs='+', required=True,
                        choices=['audio', 'visual', 'text'],
                        help="e.g. --modalities audio visual text")
    parser.add_argument('--data_path', default=None,
                        help='override the dataset path from config.py')
    parser.add_argument('--teacher_dir', default=None,
                        help='directory holding best_<mod>_teacher.pth files')
    parser.add_argument('--epochs', type=int, default=None,
                        help='default: 300 for teachers, 100 for the student')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no_catgs', action='store_true',
                        help='Joint-Train baseline (distillation without CAT-GS)')
    parser.add_argument('--no_pcgrad', action='store_true',
                        help='disable fusion-only PCGrad in the controller')
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 300 if args.stage == 'teacher' else 100

    setup_seed(config.random_seed)
    train_ds, dev_ds, test_ds = build_datasets(args)
    loop(train_ds, dev_ds, test_ds, args)


if __name__ == '__main__':
    main()

"""
CAT-GS for M >= 2 modalities (paper Sec. "Extension to M Modalities").

Used for the sequence-feature benchmarks (UR-FUNNY A-V / A-T / V-T / A-V-T,
CMU-MOSI V-T / V-A-T) with the Transformer models in
`models/transformer_model.py`. The margin generalizes to the max-min spread
of the smoothed reliabilities, Delta_M = max_m p_m - min_m p_m; the same
regime policy applies, and fusion-only PCGrad becomes sequential pairwise
projection over the modality-specific fusion gradients.

Call `CATGSGateM.step(...)` AFTER loss.backward() and BEFORE optimizer.step().
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CATGSGateM:
    """CAT-GS controller for an arbitrary set of modalities."""

    def __init__(self,
                 modalities,
                 tau_low=0.05,
                 tau_high=0.15,
                 warmup_epochs=5,
                 dropout_prob=0.6,
                 weak_bias=0.2,
                 eps_floor=0.1,
                 reliability_ema_beta=0.9,
                 gradnorm_ema_beta=0.9,
                 gradnorm_cap=1.5,
                 delta=1e-8,
                 use_calibrated_T=True,
                 use_ema=True,
                 use_weak_bias=True,
                 use_gradnorm_realloc=True,
                 use_pcgrad=True,
                 calibration_T=None):
        self.modalities = list(modalities)
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.warmup_epochs = warmup_epochs
        self.dropout_prob = dropout_prob
        self.weak_bias = weak_bias
        self.eps_floor = eps_floor
        self.reliability_ema_beta = reliability_ema_beta
        self.gradnorm_ema_beta = gradnorm_ema_beta
        self.gradnorm_cap = gradnorm_cap
        self.delta = delta
        self.use_calibrated_T = use_calibrated_T
        self.use_ema = use_ema
        self.use_weak_bias = use_weak_bias
        self.use_gradnorm_realloc = use_gradnorm_realloc
        self.use_pcgrad = use_pcgrad
        self.T = calibration_T or {mod: 1.0 for mod in self.modalities}

        self.ema_conf = {mod: None for mod in self.modalities}
        self.ema_gradnorm = {mod: None for mod in self.modalities}
        self.history = []

    # ------------------------------------------------------------------
    # Reliability estimation (Eq. 3-4)
    # ------------------------------------------------------------------

    def _compute_confidence(self, teacher_logits, labels, modality):
        T = self.T.get(modality, 1.0) if self.use_calibrated_T else 1.0
        probs = F.softmax(teacher_logits / T, dim=1)
        labels = labels.long()
        gt_probs = probs[torch.arange(labels.size(0), device=labels.device), labels]
        return gt_probs.mean().item()

    @staticmethod
    def _update_ema(current, ema, beta):
        return current if ema is None else beta * ema + (1 - beta) * current

    def compute_modality_scores(self, teachers, batch, labels):
        """Teacher reliabilities {mod: p_m}, EMA-smoothed into self.ema_conf."""
        scores = {}
        with torch.no_grad():
            for modality, teacher in teachers.items():
                single_mod_batch = {
                    modality: batch[modality],
                    f'{modality}_mask': batch.get(f'{modality}_mask'),
                    'label': labels,
                }
                conf = self._compute_confidence(teacher(single_mod_batch), labels, modality)
                scores[modality] = conf
                if self.use_ema:
                    self.ema_conf[modality] = self._update_ema(
                        conf, self.ema_conf[modality], self.reliability_ema_beta)
                else:
                    self.ema_conf[modality] = conf
        return scores

    # ------------------------------------------------------------------
    # Regime selection and gating (Eq. 5-8, generalized margin Delta_M)
    # ------------------------------------------------------------------

    def compute_gating_coefficients(self, epoch, modality_scores):
        confs = {mod: self.ema_conf[mod] for mod in modality_scores.keys()}
        conf_values = list(confs.values())
        margin = max(conf_values) - min(conf_values)

        sorted_mods = sorted(confs.items(), key=lambda kv: kv[1])
        weakest_mod = sorted_mods[0][0]

        alphas = {mod: 1.0 for mod in confs}

        if epoch < self.warmup_epochs and margin < self.tau_low:
            # Regime 1: warm-up dropout — keep one random modality with prob p_drop.
            regime = 'warmup_dropout'
            if random.random() < self.dropout_prob:
                keep_mod = random.choice(list(confs.keys()))
                alphas = {mod: 1.0 if mod == keep_mod else 0.0 for mod in confs}

        elif margin > self.tau_high:
            # Regime 2: clear dominance — train only the weakest modality.
            regime = 'dominance'
            alphas = {mod: 1.0 if mod == weakest_mod else 0.0 for mod in confs}

        elif self.tau_low <= margin <= self.tau_high:
            # Regime 3: close reliabilities — floored proportional split with weak bias.
            regime = 'weak_bias' if self.use_weak_bias else 'proportional'
            total_conf = max(sum(conf_values), 1e-6)
            for mod in confs:
                alpha = max(confs[mod] / total_conf, self.eps_floor)
                if self.use_weak_bias:
                    if confs[mod] == min(conf_values):
                        alpha *= (1 + self.weak_bias)
                    elif confs[mod] == max(conf_values):
                        alpha *= (1 - self.weak_bias)
                alphas[mod] = alpha
            total_alpha = sum(alphas.values())
            alphas = {mod: a / total_alpha for mod, a in alphas.items()}

        else:
            regime = 'proportional'
            total_conf = max(sum(conf_values), 1e-6)
            alphas = {mod: max(confs[mod] / total_conf, self.eps_floor) for mod in confs}

        return regime, alphas, margin

    # ------------------------------------------------------------------
    # Gradient modulation and budget reallocation (Eq. 9-10)
    # ------------------------------------------------------------------

    def _compute_gradnorm(self, student_model, modality):
        norm = 0.0
        if modality in student_model.encoders:
            for param in student_model.encoders[modality].parameters():
                if param.grad is not None:
                    norm += param.grad.norm().item() ** 2
        return float(np.sqrt(norm))

    def _renorm_gradients(self, student_model, modality, target_norm):
        current_norm = self._compute_gradnorm(student_model, modality)
        if current_norm < self.delta:
            return
        scale = min(target_norm / current_norm, self.gradnorm_cap)
        for param in student_model.encoders[modality].parameters():
            if param.grad is not None:
                param.grad.mul_(scale)

    def apply_gradient_modulation(self, student_model, alphas):
        for modality, alpha in alphas.items():
            if modality in student_model.encoders:
                for param in student_model.encoders[modality].parameters():
                    if param.grad is not None:
                        param.grad.mul_(alpha)

    def apply_gradient_budget_reallocation(self, student_model, alphas):
        for modality in alphas:
            current = self._compute_gradnorm(student_model, modality)
            self.ema_gradnorm[modality] = self._update_ema(
                current, self.ema_gradnorm[modality], self.gradnorm_ema_beta)

        if any(alpha == 0.0 for alpha in alphas.values()):
            active_mods = [m for m, a in alphas.items() if a > 0]
            for active_mod in active_mods:
                if self.ema_gradnorm[active_mod] is not None:
                    self._renorm_gradients(student_model, active_mod,
                                           self.ema_gradnorm[active_mod])

    # ------------------------------------------------------------------
    # Fusion-only PCGrad (Eq. 11-13, sequential pairwise projections)
    # ------------------------------------------------------------------

    def _fusion_pcgrad(self, student_model, modality_features):
        """Pairwise-project conflicting modality-specific fusion gradients on
        the final classifier layer, then overwrite its .grad with their sum."""
        fusion_param = student_model.classifier[-1].weight
        if fusion_param.grad is None:
            return None

        grads = {}
        for mod in student_model.modalities:
            out_m = student_model.modality_logits(modality_features, mod)
            g = torch.autograd.grad(
                outputs=out_m,
                inputs=fusion_param,
                grad_outputs=torch.ones_like(out_m),
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g is None:
                return None
            grads[mod] = g.clone()

        # PCGrad: for each conflicting pair, project g_i off g_j's direction.
        mods = list(grads.keys())
        n_conflicts = 0
        for i in mods:
            for j in mods:
                if i == j:
                    continue
                gi, gj = grads[i].flatten(), grads[j].flatten()
                dot = torch.dot(gi, gj)
                if dot < 0:
                    n_conflicts += 1
                    grads[i] = grads[i] - (dot / (gj.norm() ** 2 + self.delta)) * grads[j]

        fusion_param.grad.copy_(sum(grads.values()))
        return n_conflicts

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, teachers, student_model, batch, labels, epoch,
             modulation_starts=0, modulation_ends=150, modality_features=None):
        """Full CAT-GS update. Call after loss.backward(), before optimizer.step().

        Args:
            teachers: {mod: frozen unimodal teacher model}
            student_model: MultimodalTransformerClassifier (multi-modality)
            batch: collated batch dict
            labels: (B,) ground-truth labels
            epoch: current epoch
            modality_features: per-modality student features from the forward
                pass (required for fusion-only PCGrad; pass the second output
                of `student_model(batch, return_features=True)`)

        Returns:
            (regime, scores, alphas, margin)
        """
        scores = self.compute_modality_scores(teachers, batch, labels)
        regime, alphas, margin = self.compute_gating_coefficients(epoch, scores)

        if modulation_starts <= epoch <= modulation_ends:
            self.apply_gradient_modulation(student_model, alphas)
            if self.use_gradnorm_realloc:
                self.apply_gradient_budget_reallocation(student_model, alphas)
            if self.use_pcgrad and modality_features is not None:
                self._fusion_pcgrad(student_model, modality_features)

        self.history.append({
            'epoch': epoch, 'regime': regime, 'margin': margin,
            'scores': dict(scores), 'ema_conf': dict(self.ema_conf),
            'alphas': dict(alphas),
        })
        return regime, scores, alphas, margin


# ----------------------------------------------------------------------
# Distillation losses (Eq. 1-2) and epoch loops
# ----------------------------------------------------------------------

def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)


def feature_loss(student_features, teacher_features):
    return F.mse_loss(student_features, teacher_features)


def train_teacher_epoch(model, dataloader, optimizer, device):
    """One supervised epoch for a unimodal teacher. Returns (loss, accuracy)."""
    from tqdm import tqdm
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(dataloader, desc="Training teacher"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch)
        loss = F.cross_entropy(logits, batch['label'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += logits.argmax(1).eq(batch['label']).sum().item()
        total += batch['label'].size(0)
    return total_loss / len(dataloader), 100.0 * correct / total


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Returns (loss, accuracy) on a dataloader."""
    from tqdm import tqdm
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch)
        total_loss += F.cross_entropy(logits, batch['label']).item()
        correct += logits.argmax(1).eq(batch['label']).sum().item()
        total += batch['label'].size(0)
    return total_loss / len(dataloader), 100.0 * correct / total


def train_student_epoch(student_model, teachers, dataloader, optimizer, device,
                        train_config, catgs_gate, epoch=0):
    """One CAT-GS student epoch with teacher distillation.

    train_config keys: 'feat_loss_weight', 'logit_loss_weight', 'temperature',
    'use_catgs', 'modulation_starts', 'modulation_ends'.
    """
    from tqdm import tqdm
    student_model.train()
    for teacher in teachers.values():
        teacher.eval()

    criterion = nn.CrossEntropyLoss()
    totals = {'loss': 0.0, 'task_loss': 0.0, 'feat_loss': 0.0, 'logit_loss': 0.0}
    correct, total = 0, 0
    regime_counts = {}

    for batch in tqdm(dataloader, desc=f"Training student (epoch {epoch + 1})"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        labels = batch['label']

        student_logits, student_features = student_model(batch, return_features=True)
        task_loss = criterion(student_logits, labels)

        # Frozen-teacher targets.
        teacher_logits, teacher_feats = {}, {}
        with torch.no_grad():
            for mod, teacher in teachers.items():
                single_mod_batch = {
                    mod: batch[mod],
                    f'{mod}_mask': batch.get(f'{mod}_mask'),
                    'label': labels,
                }
                t_logits, t_feats = teacher(single_mod_batch, return_features=True)
                teacher_logits[mod] = t_logits
                teacher_feats[mod] = t_feats[mod]

        feat_l = sum(feature_loss(student_features[m], teacher_feats[m])
                     for m in student_model.modalities) / len(student_model.modalities)
        logit_l = sum(distillation_loss(student_logits, teacher_logits[m],
                                        train_config.get('temperature', 1.0))
                      for m in student_model.modalities) / len(student_model.modalities)

        loss = (task_loss
                + train_config.get('feat_loss_weight', 1.0) * feat_l
                + train_config.get('logit_loss_weight', 1.0) * logit_l)

        optimizer.zero_grad()
        loss.backward(retain_graph=train_config.get('use_catgs', True))

        if train_config.get('use_catgs', True):
            regime, _, _, _ = catgs_gate.step(
                teachers=teachers,
                student_model=student_model,
                batch=batch,
                labels=labels,
                epoch=epoch,
                modulation_starts=train_config.get('modulation_starts', 0),
                modulation_ends=train_config.get('modulation_ends', 150),
                modality_features=student_features,
            )
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        optimizer.step()

        totals['loss'] += loss.item()
        totals['task_loss'] += task_loss.item()
        totals['feat_loss'] += feat_l.item()
        totals['logit_loss'] += logit_l.item()
        correct += student_logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    metrics = {k: v / len(dataloader) for k, v in totals.items()}
    metrics['accuracy'] = 100.0 * correct / total
    metrics['catgs_regimes'] = regime_counts
    return metrics

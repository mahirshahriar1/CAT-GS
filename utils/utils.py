import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os


def _ensure_csv_header(path: str, header_line: str):
    """Create file (or write header) if missing/empty."""
    try:
        needs_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        if needs_header:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w') as f:
                f.write(header_line.rstrip('\n') + '\n')
    except Exception:
        # Never break training due to logging
        pass


def _get_scores_dir() -> str:
    # Prefer configured absolute path; otherwise fall back to local ./scores
    return getattr(config, 'scores_path', os.path.join('.', 'scores'))


def _fusion_params(model):
    """Return fusion-module parameters to measure conflict on."""
    if not hasattr(model, 'fusion_module'):
        return []
    params = [p for p in model.fusion_module.parameters() if p.requires_grad]
    return params


def _flatten_grads(grads):
    if grads is None:
        return None
    flat = []
    for g in grads:
        if g is None:
            continue
        flat.append(g.reshape(-1))
    if len(flat) == 0:
        return None
    return torch.cat(flat)


def fusion_grad_conflict_cos(model, out_a, out_v, labels):
    """Compute cosine similarity between modality-specific fusion gradients.

    Definition matches paper: g_f^m = d L_m / d theta_f, where L_m is unimodal loss
    and theta_f are fusion parameters.

    Returns: float cosine similarity
    """
    params = _fusion_params(model)
    if len(params) == 0:
        return 0.0

    # Unimodal losses (classification)
    labels = labels.long()
    loss_a = F.cross_entropy(out_a, labels)
    loss_v = F.cross_entropy(out_v, labels)

    grads_a = torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=True)
    grads_v = torch.autograd.grad(loss_v, params, retain_graph=True, allow_unused=True)
    ga = _flatten_grads(grads_a)
    gv = _flatten_grads(grads_v)
    if ga is None or gv is None:
        return 0.0
    cos = torch.nn.functional.cosine_similarity(ga, gv, dim=0)
    return float(cos.item())


def log_fusion_conflict_joint(epoch: int, batch_idx: int, model, out_a, out_v, labels, tag: str = 'joint'):
    """Log fusion gradient cosine for Joint-Train (and can be reused elsewhere)."""
    # Keep joint logs alongside other score logs.
    # If config.scores_path is not set (e.g., modulation='Normal'), fall back to ./scores/<dataset>.
    scores_dir = _default_scores_dir()
    dataset = getattr(config, 'dataset', 'dataset')
    path = os.path.join(scores_dir, f"conflict_{dataset}_{tag}.csv")
    _ensure_csv_header(path, 'epoch,batch,fusion_cos')
    try:
        cos = fusion_grad_conflict_cos(model, out_a, out_v, labels)
        with open(path, 'a') as f:
            f.write(f"{epoch},{batch_idx},{cos:.6f}\n")
    except Exception:
        pass


def _default_scores_dir():
    # Use config.scores_path when available, else a sensible local default.
    # This lets Joint-Train (modulation='Normal') still log without changing config.py.
    dataset = getattr(config, 'dataset', 'run')
    return getattr(config, 'scores_path', os.path.join('scores', str(dataset)))


def log_conflict_cos(epoch, fusion_cos, method):
    """Append a single conflict-dynamics datapoint.

    Writes to: <scores_dir>/conflict_<dataset>_<method>.csv
    Columns: epoch,fusion_cos
    """
    if fusion_cos is None:
        return
    scores_dir = _default_scores_dir()
    os.makedirs(scores_dir, exist_ok=True)
    dataset = getattr(config, 'dataset', 'run')
    path = os.path.join(scores_dir, f"conflict_{dataset}_{method}.csv")
    try:
        if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            with open(path, 'w') as f:
                f.write('epoch,fusion_cos\n')
        with open(path, 'a') as f:
            f.write(f"{epoch},{fusion_cos:.6f}\n")
    except Exception:
        # Never break training due to logging
        pass


def fusion_grad_cosine(model, out_a, out_v, apply_pcgrad=False):
    """Cosine similarity between modality-specific fusion-layer gradients.

    If apply_pcgrad=True and cosine<0, projects grad_a to remove conflict and
    overwrites fusion-layer .grad with the projected sum (PCGrad surgery).

    Returns:
        cosine (float) or None if fusion layer or grads are unavailable.
    """
    fusion_param = None
    for name, p in model.named_parameters():
        if 'fc_out' in name:
            fusion_param = p
            break
    if fusion_param is None:
        return None

    try:
        grad_a = torch.autograd.grad(
            outputs=out_a,
            inputs=fusion_param,
            grad_outputs=torch.ones_like(out_a),
            retain_graph=True,
            allow_unused=True
        )[0]
        grad_v = torch.autograd.grad(
            outputs=out_v,
            inputs=fusion_param,
            grad_outputs=torch.ones_like(out_v),
            retain_graph=True,
            allow_unused=True
        )[0]
    except Exception:
        return None

    if grad_a is None or grad_v is None:
        return None

    ga = grad_a.flatten()
    gv = grad_v.flatten()
    cos = torch.nn.functional.cosine_similarity(ga, gv, dim=0)

    if apply_pcgrad and cos.item() < 0:
        denom = (gv.norm() ** 2) + 1e-8
        proj = (torch.dot(ga, gv) / denom) * grad_v
        grad_a_proj = grad_a - proj

        # Apply fusion-layer PCGrad surgery
        if fusion_param.grad is not None:
            fusion_param.grad.copy_(grad_a_proj + grad_v)
        else:
            fusion_param.grad = (grad_a_proj + grad_v)

        # Return cosine after surgery (conflict reduced)
        cos_post = torch.nn.functional.cosine_similarity(grad_a_proj.flatten(), gv, dim=0)
        return cos_post.item()

    return cos.item()


def log_conflict_from_outputs(epoch, model, out_a, out_v, method, apply_pcgrad=False):
    cos = fusion_grad_cosine(model, out_a, out_v, apply_pcgrad=apply_pcgrad)
    log_conflict_cos(epoch, cos, method)
    return cos


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # # Additional settings for cross-GPU reproducibility
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True, warn_only=True)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        

def unimodal_outputs(model, af, vf):
    fusion_module = model.fusion_module

    if config.fusion_method == 'sum':
        # SumFusion: Linear transformations on each modality separately, then add
        out_a = (torch.mm(af, torch.transpose(fusion_module.fc_x.weight, 0, 1)) +
                 fusion_module.fc_x.bias / 2)
        out_v = (torch.mm(vf, torch.transpose(fusion_module.fc_y.weight, 0, 1)) +
                 fusion_module.fc_y.bias / 2)

    elif config.fusion_method == 'late':
        # LateFusion: Similar to SumFusion but with averaging logits
        out_a = (torch.mm(af, torch.transpose(fusion_module.fc_x.weight, 0, 1)) +
                 fusion_module.fc_x.bias)
        out_v = (torch.mm(vf, torch.transpose(fusion_module.fc_y.weight, 0, 1)) +
                 fusion_module.fc_y.bias)

    elif config.fusion_method == 'concat':
        # ConcatFusion: Split weights in fc_out for each modality
        weight_size = fusion_module.fc_out.weight.size(1)
        out_a = (torch.mm(af, torch.transpose(fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                 + fusion_module.fc_out.bias / 2)
        out_v = (torch.mm(vf, torch.transpose(fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                 + fusion_module.fc_out.bias / 2)

    elif config.fusion_method == 'film':
        # FiLM: Use conditioning with gamma and beta from fc for each modality
        gamma_a, beta_a = torch.split(fusion_module.fc(af), fusion_module.dim, 1)
        out_a = gamma_a * af + beta_a
        out_a = torch.mm(out_a, torch.transpose(fusion_module.fc_out.weight, 0, 1)) + fusion_module.fc_out.bias / 2

        gamma_v, beta_v = torch.split(fusion_module.fc(vf), fusion_module.dim, 1)
        out_v = gamma_v * vf + beta_v
        out_v = torch.mm(out_v, torch.transpose(fusion_module.fc_out.weight, 0, 1)) + fusion_module.fc_out.bias / 2
    
    elif config.fusion_method == 'gated':
        out_a = torch.mm(af, torch.transpose(fusion_module.fc_out.weight, 0, 1)) + fusion_module.fc_out.bias / 2
        out_v = torch.mm(vf, torch.transpose(fusion_module.fc_out.weight, 0, 1)) + fusion_module.fc_out.bias / 2

    return out_a, out_v



def evaluate(model, device, test_loader, criterion):
    """
    Evaluate the model using the provided data loader.
    Handles different modalities based on the configuration.
    """
    from tqdm import tqdm
    
    model.eval()
    
    total_loss = 0
    total_loss_a = 0
    total_loss_v = 0
    
    total_correct = 0
    total_correct_a = 0
    total_correct_v = 0
    
    total_samples = 0
    
    with torch.no_grad():
        # Add progress bar for evaluation
        test_loader_tqdm = tqdm(test_loader, desc='Evaluating', unit='batch')
        
        for data in test_loader_tqdm:
            audio, video, labels = data
            audio, video, labels = audio.to(device), video.to(device), labels.to(device)
            
            # Handle different dataset formats
            if config.dataset == 'AVMNIST':
                audio = audio.float()
                audio = audio.unsqueeze(1).float()
                video = video.float()
            elif config.dataset == 'CGMNIST':
                # For CGMNIST: gray images already have shape [batch, 28, 28], need [batch, 1, 28, 28]
                # colored images already have shape [batch, 3, 28, 28]
                audio = audio.float()  # Gray images (modal1)
                if len(audio.shape) == 3:  # [batch, 28, 28]
                    audio = audio.unsqueeze(1)  # Add channel dimension -> [batch, 1, 28, 28]
                video = video.float()  # Colored images (modal2), already [batch, 3, 28, 28]
            else:
                audio = audio.unsqueeze(1).float()
                video = video.float()
            
            if config.modality == 'multimodal':

                af, vf, outputs = model(audio, video, return_features=True)
                
                out_a, out_v = unimodal_outputs(model, af, vf)

                loss_a = criterion(out_a, labels) 
                loss_v = criterion(out_v, labels)
                
                total_loss_a += loss_a.item()
                total_loss_v += loss_v.item() 
                
                total_correct_a += calculate_accuracy(out_a, labels)
                total_correct_v += calculate_accuracy(out_v, labels)  
                
            else:
                if config.modality == 'audio':
                    outputs = model(audio, None)
                    
                if config.modality == 'visual':
                    outputs = model(None, video)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            total_correct += calculate_accuracy(outputs, labels)
            total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * (total_correct / total_samples)
    
    if config.modality == 'multimodal':
        avg_loss_a = total_loss_a / len(test_loader)
        avg_loss_v = total_loss_v / len(test_loader)
        accuracy_a = 100 * (total_correct_a / total_samples)
        accuracy_v = 100 * (total_correct_v / total_samples)
        return avg_loss_a, avg_loss_v, avg_loss, accuracy_a, accuracy_v, accuracy
    
    return avg_loss, accuracy


def calculate_accuracy(outputs, labels):
    outputs = nn.Softmax(dim=1)(outputs)
    _, preds = torch.max(outputs.data, 1)
    return (preds == labels).sum().item()



def update_model_with_OGM_GE(out_a, out_v, labels, model, epoch):
    'OGM_GE implementation from the CVPR 2022 Paper: "Balanced Multimodal Learning via On-the-fly Gradient Modulation"'
    
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    
    # Convert labels to long for proper indexing
    labels = labels.long()
    
    # Modulation starts here !
    score_v = sum([softmax(out_v)[i][labels[i]] for i in range(out_v.size(0))])
    score_a = sum([softmax(out_a)[i][labels[i]] for i in range(out_a.size(0))])

    ratio_v = score_v / score_a
    ratio_a = 1 / ratio_v
    
    """
    Below is the Eq.(10) in our CVPR paper:
            1 - tanh(alpha * rho_t_u), if rho_t_u > 1
    k_t_u =
            1,                         else
    coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
    """
    
    if ratio_v > 1:
        coeff_v = 1 - tanh(config.alpha_modulation * relu(ratio_v))
        coeff_a = 1
    else:
        coeff_a = 1 - tanh(config.alpha_modulation * relu(ratio_a))
        coeff_v = 1
     
    if config.modulation_starts <= epoch <= config.modulation_ends:
        for name, parms in model.named_parameters():
            layer = str(name).split('.')[0]
            # print(f"layer: {layer}")

            if 'audio' in layer and len(parms.grad.size()) == 4:
                if config.modulation == 'OGM-GE':
                    # print("Doing OGM-GE Modulation:")  # For debugging
                    parms.grad = parms.grad * coeff_a + \
                                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif config.modulation == 'OGM':
                    # print("Doing OGM Modulation:")  # For debugging
                    parms.grad *= coeff_a

            if 'visual' in layer and len(parms.grad.size()) == 4:
                if config.modulation == 'OGM-GE':  
                    # print("Doing OGM-GE Modulation:")  # For debugging
                    parms.grad = parms.grad * coeff_v + \
                                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                elif config.modulation == 'OGM':
                    # print("Doing OGM Modulation:")  # For debugging
                    parms.grad *= coeff_v
    else:
        # print("Not doing Modulation.")
        pass  
    
    return ratio_v, coeff_a, coeff_v



def modulate_gradients_based_on_scores_from_two_modalities(teacher_out_a, teacher_out_v, out_a, out_v, labels, modulation_type, model, epoch):
    """
    Modulates gradients using G2D with simple confidence-based gating per batch.
    
    Suppresses stronger modality, trains weaker modality.
    Decision made independently per batch based on teacher scores.
    
    Args:
        teacher_out_a: output of audio teacher
        teacher_out_v: output of visual teacher
        out_a: output of audio modality of student
        out_v: output of visual modality of student
        labels: labels 
        modulation_type: should be 'G2D'
        model: model
        epoch: current epoch
    
    Returns:
        ratio: 0 (placeholder for compatibility)
        coeff_a: coefficient applied to audio gradients
        coeff_v: coefficient applied to visual gradients
    """
    
    softmax = nn.Softmax(dim=1)
    
    # Step 1: Compute confidence scores from teachers (Equation 5 in paper)
    # ρ_t_m = (1/|B_m|) * Σ Softmax(l_t_m(x_i))[y_i]
    score_teacher_a = sum([softmax(teacher_out_a)[i][labels[i]] for i in range(teacher_out_a.size(0))]) / teacher_out_a.size(0)
    score_teacher_v = sum([softmax(teacher_out_v)[i][labels[i]] for i in range(teacher_out_v.size(0))]) / teacher_out_v.size(0)
    
    # Also compute student scores for logging
    score_a = sum([softmax(out_a)[i][labels[i]] for i in range(out_a.size(0))]) / out_a.size(0)
    score_v = sum([softmax(out_v)[i][labels[i]] for i in range(out_v.size(0))]) / out_v.size(0)
    
    # Step 2: Rank modalities by confidence
    # Lower score = weaker modality = should be prioritized
    teacher_scores = {
        'audio': score_teacher_a.item(),
        'visual': score_teacher_v.item()
    }
    
    # Sort by score (ascending) to get ranking: weakest modality first
    ranked_modalities = sorted(teacher_scores.keys(), key=lambda x: teacher_scores[x])
    weakest_modality = ranked_modalities[0]
    strongest_modality = ranked_modalities[1]
    
    # Initialize coefficients
    coeff_a = 1.0
    coeff_v = 1.0
    phase = "G2D: Confidence-based gating"
    
    # Simple per-batch confidence-based gating
    # Suppress stronger modality, train weaker modality
    if weakest_modality == 'audio':
        coeff_a = 1.0  # Train audio (weaker)
        coeff_v = 0.0  # Suppress visual (stronger)
        phase = "G2D: Train Audio (weaker)"
    else:
        coeff_a = 0.0  # Suppress audio (stronger)
        coeff_v = 1.0  # Train visual (weaker)
        phase = "G2D: Train Visual (weaker)"
    
    # Step 3: Apply gradient modulation
    if config.modulation_starts <= epoch <= config.modulation_ends:
        for name, parms in model.named_parameters():                       
            layer = str(name).split('.')[0]  
            
            if 'audio' in layer and len(parms.data.size()) == 4:                 
                parms.grad = (parms.grad * coeff_a)
                    
            if 'visual' in layer and len(parms.data.size()) == 4: 
                parms.grad = (parms.grad * coeff_v)
    
    # Fusion-layer conflict cosine (paper metric)
    fusion_cos = fusion_grad_conflict_cos(model, out_a, out_v, labels)

    # Log scores, coefficients, phase + fusion_cos to CSV
    csv_path = os.path.join(_get_scores_dir(), getattr(config, 'filename', 'scores.csv'))
    _ensure_csv_header(
        csv_path,
        'epoch,coeff_a,coeff_v,score_teacher_a,score_teacher_v,score_a,score_v,weakest_modality,phase,fusion_cos'
    )
    with open(csv_path, 'a') as f:
        f.write(
            f"{epoch},{coeff_a},{coeff_v},{score_teacher_a:.4f},{score_teacher_v:.4f},"
            f"{score_a:.4f},{score_v:.4f},{weakest_modality},{phase},{fusion_cos:.6f}\n"
        )
    
    return 0, coeff_a, coeff_v




# --- CAT-GS: Calibrated, Adaptive, Thresholded Gating + Fusion Surgery ---------
class CATGSGate:
    """
    CAT-GS: Combines calibrated margin-thresholded gating, weak-biased scaling, gradient budget reallocation, and fusion-layer PCGrad.
    """
    def __init__(self,
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
                 T_a=1.0,
                 T_v=1.0):
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
        self.Ta = T_a
        self.Tv = T_v
        self.ema_pa = None
        self.ema_pv = None
        self.ema_gradnorm_audio = None
        self.ema_gradnorm_visual = None
        # logging
        self.log_path = os.path.join(getattr(config, 'scores_path', '.'), f"catgs_{getattr(config,'filename','stats.csv')}")
        os.makedirs(getattr(config, 'scores_path', '.'), exist_ok=True)

        _ensure_csv_header(self.log_path, 'epoch,regime,pa,pv,margin,alpha_a,alpha_v,gradnorm_audio,gradnorm_visual,fusion_cos')

    @torch.no_grad()
    def _gt_conf(self, logits, labels, T=1.0):
        """Batch-mean probability assigned to the ground-truth label (Eq. 3)."""
        labels = labels.long()
        probs = torch.softmax(logits / T, dim=1)
        idx = torch.arange(labels.size(0), device=labels.device)
        return probs[idx, labels].mean().item()

    def _ema(self, x, ema, beta):
        return x if ema is None else beta * ema + (1 - beta) * x

    def _gradnorm(self, model, modality):
        norm = 0.0
        for name, p in model.named_parameters():
            if p.grad is None or p.grad.ndim != 4:
                continue
            layer = name.split('.')[0]
            if modality == 'audio' and 'audio' in layer:
                norm += p.grad.norm().item()
            elif modality == 'visual' and 'visual' in layer:
                norm += p.grad.norm().item()
        return norm

    def _renorm(self, model, modality, target_norm):
        # Rescale all conv grads in modality so total norm matches target_norm (with cap)
        current_norm = self._gradnorm(model, modality)
        if current_norm < 1e-8:
            return
        scale = min(target_norm / current_norm, self.gradnorm_cap)
        for name, p in model.named_parameters():
            if p.grad is None or p.grad.ndim != 4:
                continue
            layer = name.split('.')[0]
            if modality == 'audio' and 'audio' in layer:
                p.grad.mul_(scale)
            elif modality == 'visual' and 'visual' in layer:
                p.grad.mul_(scale)

    def _fusion_pcgrad(self, model, out_a, out_v, labels):
        # Apply PCGrad to fusion layer weights (e.g., fc_out)
        fusion_layer = None
        for name, p in model.named_parameters():
            if 'fc_out' in name and p.grad is not None:
                fusion_layer = p
                break
        if fusion_layer is None:
            return None
        # Compute fusion grads for audio and visual heads
        grad_a = torch.autograd.grad(
            outputs=out_a,
            inputs=fusion_layer,
            grad_outputs=torch.ones_like(out_a),
            retain_graph=True,
            allow_unused=True
        )[0]
        grad_v = torch.autograd.grad(
            outputs=out_v,
            inputs=fusion_layer,
            grad_outputs=torch.ones_like(out_v),
            retain_graph=True,
            allow_unused=True
        )[0]
        if grad_a is None or grad_v is None:
            return None
        # PCGrad: if cosine similarity < 0, project grad_a onto normal plane of grad_v
        cos = torch.nn.functional.cosine_similarity(grad_a.flatten(), grad_v.flatten(), dim=0)
        if cos < 0:
            grad_a_proj = grad_a - (torch.dot(grad_a.flatten(), grad_v.flatten()) / (grad_v.flatten().norm()**2 + self.delta)) * grad_v
            fusion_layer.grad.copy_(grad_a_proj + grad_v)
        else:
            fusion_layer.grad.copy_(grad_a + grad_v)
        return cos.item()

    def step(self, teacher_out_a, teacher_out_v, out_a, out_v, labels, model, epoch):
        """
        Main CAT-GS logic with margin-thresholded adaptive gating.
        
        Returns: regime, pa, pv, margin, alpha_a, alpha_v, gradnorms, fusion_cos
        """
        
        # 1. Compute teacher confidences
        Ta = self.Ta if self.use_calibrated_T else 1.0
        Tv = self.Tv if self.use_calibrated_T else 1.0
        pa = self._gt_conf(teacher_out_a, labels, Ta)
        pv = self._gt_conf(teacher_out_v, labels, Tv)
        
        # Apply EMA smoothing if enabled (Eq. 4, momentum beta)
        if self.use_ema:
            self.ema_pa = self._ema(pa, self.ema_pa, self.reliability_ema_beta)
            self.ema_pv = self._ema(pv, self.ema_pv, self.reliability_ema_beta)
            pa_ema = self.ema_pa
            pv_ema = self.ema_pv
        else:
            pa_ema = pa
            pv_ema = pv
        
        margin = abs(pa_ema - pv_ema)

        # 2. Decide regime and compute alpha values using CAT-GS margin-thresholded gating
        regime = None
        alpha_a, alpha_v = 1.0, 1.0
        
        if epoch < self.warmup_epochs and margin < self.tau_low:
            # Aggressive modality dropout (warm-up)
            regime = 'warmup_dropout'
            if random.random() < self.dropout_prob:
                # Drop audio
                alpha_a, alpha_v = 0.0, 1.0
            else:
                # Drop visual
                alpha_a, alpha_v = 1.0, 0.0
        elif margin > self.tau_high:
            # Clear dominance: G2D mode
            regime = 'g2d'
            if pa_ema > pv_ema:
                alpha_a, alpha_v = 0.0, 1.0
            else:
                alpha_a, alpha_v = 1.0, 0.0
        elif self.tau_low <= margin <= self.tau_high:
            # Close: weak-biased split (only if enabled)
            regime = 'weak_bias' if self.use_weak_bias else 'proportional'
            denom = max(pa_ema + pv_ema, 1e-6)
            alpha_a = max(pa_ema / denom, self.eps_floor)
            alpha_v = max(pv_ema / denom, self.eps_floor)
            
            # Apply weak-bias amplification only if enabled
            if self.use_weak_bias:
                # Identify weak/strong
                if pa_ema < pv_ema:
                    alpha_a = alpha_a * (1 + self.weak_bias)
                    alpha_v = alpha_v * (1 - self.weak_bias)
                else:
                    alpha_v = alpha_v * (1 + self.weak_bias)
                    alpha_a = alpha_a * (1 - self.weak_bias)
            
            # Re-normalize
            total = alpha_a + alpha_v
            alpha_a /= total
            alpha_v /= total
        else:
            regime = 'default'
            denom = max(pa_ema + pv_ema, 1e-6)
            alpha_a = max(pa_ema / denom, self.eps_floor)
            alpha_v = max(pv_ema / denom, self.eps_floor)

        # 3. Apply gating to conv grads (only within modulation window)
        if config.modulation_starts <= epoch <= config.modulation_ends:
            for name, p in model.named_parameters():
                if p.grad is None or p.grad.ndim != 4:
                    continue
                layer = name.split('.')[0]
                if 'audio' in layer:
                    p.grad.mul_(alpha_a)
                elif 'visual' in layer:
                    p.grad.mul_(alpha_v)
        # else: skip gradient modulation outside window (epochs > modulation_ends)

        # 4. Gradient budget reallocation (also respect modulation window)
        if config.modulation_starts <= epoch <= config.modulation_ends and self.use_gradnorm_realloc:
            gradnorm_audio = self._gradnorm(model, 'audio')
            gradnorm_visual = self._gradnorm(model, 'visual')
            self.ema_gradnorm_audio = self._ema(gradnorm_audio, self.ema_gradnorm_audio, self.gradnorm_ema_beta)
            self.ema_gradnorm_visual = self._ema(gradnorm_visual, self.ema_gradnorm_visual, self.gradnorm_ema_beta)
            # If one side is zeroed, renorm the other to match baseline
            if alpha_a == 0.0 and gradnorm_visual > 0:
                self._renorm(model, 'visual', self.ema_gradnorm_visual)
            elif alpha_v == 0.0 and gradnorm_audio > 0:
                self._renorm(model, 'audio', self.ema_gradnorm_audio)
        else:
            # Still compute gradient norms for logging, but don't apply reallocation
            gradnorm_audio = self._gradnorm(model, 'audio')
            gradnorm_visual = self._gradnorm(model, 'visual')

        # 5. Fusion conflict cosine (paper metric) + optional PCGrad surgery
        fusion_cos = fusion_grad_conflict_cos(model, out_a, out_v, labels)
        if config.modulation_starts <= epoch <= config.modulation_ends and self.use_pcgrad:
            # Keep your existing fusion surgery behavior (only affects optimizer update)
            self._fusion_pcgrad(model, out_a, out_v, labels)

        # 6. Logging
        try:
            with open(self.log_path, 'a') as f:
                f.write(f"{epoch},{regime},{pa:.4f},{pv:.4f},{margin:.4f},{alpha_a:.3f},{alpha_v:.3f},{gradnorm_audio:.2f},{gradnorm_visual:.2f},{fusion_cos:.6f}\n")
        except Exception:
            pass

        return regime, pa, pv, margin, alpha_a, alpha_v, gradnorm_audio, gradnorm_visual, fusion_cos

# Singleton helper
_CAT_GS = None

def modulate_gradients_catgs(teacher_out_a, teacher_out_v, out_a, out_v, labels, model, epoch):
    """
    CAT-GS: Calibrated, Adaptive, Thresholded Gating + fusion surgery.
    Call this AFTER loss.backward() and BEFORE optimizer.step().
    """
    global _CAT_GS
    if _CAT_GS is None:
        _CAT_GS = CATGSGate(
            tau_low=getattr(config, 'catgs_tau_low', 0.05),
            tau_high=getattr(config, 'catgs_tau_high', 0.15),
            warmup_epochs=getattr(config, 'catgs_warmup_epochs', 5),
            dropout_prob=getattr(config, 'catgs_dropout_prob', 0.6),
            weak_bias=getattr(config, 'catgs_weak_bias', 0.2),
            eps_floor=getattr(config, 'catgs_eps_floor', 0.1),
            reliability_ema_beta=getattr(config, 'catgs_reliability_ema_beta', 0.9),
            gradnorm_ema_beta=getattr(config, 'catgs_gradnorm_ema_beta', 0.9),
            gradnorm_cap=getattr(config, 'catgs_gradnorm_cap', 1.5),
            delta=getattr(config, 'catgs_delta', 1e-8),
            use_calibrated_T=getattr(config, 'use_calibrated_T', True),
            use_ema=getattr(config, 'use_ema', True),
            use_weak_bias=getattr(config, 'use_weak_bias', True),
            use_gradnorm_realloc=getattr(config, 'use_gradnorm_realloc', True),
            use_pcgrad=getattr(config, 'use_pcgrad', True),
            T_a=getattr(config, 'T_cal_audio', 1.0),
            T_v=getattr(config, 'T_cal_video', 1.0)
        )

    # Teacher-dependence stress test: simulate miscalibrated / mismatched teachers
    # by sharpening/softening teacher logits before reliability estimation.
    # T_mismatch < 1 => sharper (overconfident); T_mismatch > 1 => softer (underconfident).
    tmis_a = float(getattr(config, 'catgs_teacher_mismatch_T_audio', 1.0) or 1.0)
    tmis_v = float(getattr(config, 'catgs_teacher_mismatch_T_video', 1.0) or 1.0)
    if tmis_a <= 0:
        tmis_a = 1.0
    if tmis_v <= 0:
        tmis_v = 1.0
    if tmis_a != 1.0:
        teacher_out_a = teacher_out_a / tmis_a
    if tmis_v != 1.0:
        teacher_out_v = teacher_out_v / tmis_v

    return _CAT_GS.step(teacher_out_a, teacher_out_v, out_a, out_v, labels, model, epoch)
# --- end CAT-GS ----------------------------------------------------------------


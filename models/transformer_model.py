"""
Lightweight Transformer encoders and multimodal classifier for the
sequence-feature benchmarks (UR-FUNNY, CMU-MOSI).

Each modality gets its own Transformer encoder over pre-extracted frame/word
features; features are mean-pooled and fused by concatenation ('late') or
summation before a shared MLP classification head. A model built with a
single modality serves as a unimodal teacher.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class ModalityTransformerEncoder(nn.Module):
    """Transformer encoder for a single modality's feature sequence."""

    def __init__(self, input_dim, hidden_dim=768, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_dim = hidden_dim

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) boolean; True for real data, False for padding
        Returns:
            (batch, hidden_dim) masked-mean-pooled features
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            features = (x * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        else:
            features = x.mean(dim=1)
        return features


class MultimodalTransformerClassifier(nn.Module):
    """Multimodal classifier over any subset of modalities.

    Built with one modality it acts as a unimodal teacher; with several it is
    the multimodal student. Batches are dicts as produced by
    `dataset.URFunnyDataset.collate_multimodal`.
    """

    def __init__(self, model_config):
        super().__init__()
        self.modalities = list(model_config['modalities'])
        self.num_classes = model_config['num_classes']
        self.hidden_dim = model_config['hidden_dim']
        self.fusion_type = model_config.get('fusion_type', 'late')

        self.encoders = nn.ModuleDict({
            mod: ModalityTransformerEncoder(
                input_dim=model_config['feature_dims'][mod],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                num_heads=model_config['num_heads'],
                dropout=model_config['dropout'],
            )
            for mod in self.modalities
        })

        if self.fusion_type == 'late':
            fusion_dim = len(self.modalities) * self.hidden_dim  # concatenation
        else:
            fusion_dim = self.hidden_dim  # summation
        self.fusion_dim = fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(model_config['dropout']),
            nn.Linear(fusion_dim // 2, self.num_classes),
        )

    def encode(self, batch):
        """Per-modality pooled features: {mod: (batch, hidden_dim)}."""
        return {
            mod: self.encoders[mod](batch[mod], batch.get(f'{mod}_mask'))
            for mod in self.modalities
        }

    def fuse(self, modality_features):
        if self.fusion_type == 'late':
            return torch.cat([modality_features[mod] for mod in self.modalities], dim=1)
        return sum(modality_features.values())

    def modality_logits(self, modality_features, mod):
        """Logits from a single modality with the other fusion slots zeroed.

        Used by fusion-only PCGrad to form modality-specific fusion gradients.
        """
        if self.fusion_type == 'late':
            parts = [
                modality_features[m] if m == mod else torch.zeros_like(modality_features[m])
                for m in self.modalities
            ]
            fused = torch.cat(parts, dim=1)
        else:
            fused = modality_features[mod]
        return self.classifier(fused)

    def forward(self, batch, return_features=False):
        modality_features = self.encode(batch)
        logits = self.classifier(self.fuse(modality_features))
        if return_features:
            return logits, modality_features
        return logits

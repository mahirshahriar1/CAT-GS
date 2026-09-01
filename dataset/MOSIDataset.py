"""
CMU-MOSI dataset (sentiment; modalities: visual, audio, text).

Expects the standard MultiBench / CMU-MultimodalSDK processed pickle
(`mosi_raw.pkl`), available via the MultiBench repository
(https://github.com/pliang279/MultiBench):

    {'train'|'valid'|'test': {
        'vision': (N, 50, 20)  Facet visual features,
        'audio' : (N, 50, 5)   COVAREP acoustic features,
        'text'  : (N, 50, 300) GloVe word embeddings,
        'labels': (N, 1)       continuous sentiment in [-3, 3],
        'id'    : sample ids }}

Following standard practice (and MultiBench), sentiment is binarized as
label > 0 -> positive (1), label <= 0 -> negative (0).

Samples are returned in the same dict format as URFunnyDataset
({'audio', 'visual', 'text', 'label'}), so the same `collate_multimodal`,
models, and training code apply. The paper's CMU-MOSI experiments use the
V--T and V--A--T modality combinations.
"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

# Map our modality names to the keys used inside the MultiBench pickle.
_KEY_MAP = {'audio': 'audio', 'visual': 'vision', 'text': 'text'}


class MOSIDataset(Dataset):
    """CMU-MOSI binary sentiment classification (2 classes)."""

    def __init__(self, data_path, split='train', modalities=('visual', 'text'),
                 zero_pad_mask=True):
        """
        Args:
            data_path: path to mosi_raw.pkl
            split: 'train', 'valid'/'dev', or 'test'
            modalities: subset of ('audio', 'visual', 'text')
            zero_pad_mask: mask out all-zero (padded) timesteps in the
                pre-aligned sequences
        """
        split = 'valid' if split == 'dev' else split
        assert split in ('train', 'valid', 'test')
        self.split = split
        self.modalities = list(modalities)
        self.zero_pad_mask = zero_pad_mask

        with open(data_path, 'rb') as f:
            alldata = pickle.load(f)
        data = alldata[split]

        self.features = {}
        for mod in self.modalities:
            feats = np.asarray(data[_KEY_MAP[mod]], dtype=np.float32)
            # Replace any NaN/inf left over from feature extraction.
            self.features[mod] = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        raw_labels = np.asarray(data['labels']).reshape(-1)
        self.labels = (raw_labels > 0).astype(np.int64)

        self.feature_dims = {mod: self.features[mod].shape[2] for mod in self.modalities}
        print(f"[CMU-MOSI] {split}: {len(self.labels)} samples "
              f"(pos: {int(self.labels.sum())}, neg: {int((1 - self.labels).sum())}), "
              f"feature dims {self.feature_dims}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        for mod in self.modalities:
            seq = torch.from_numpy(self.features[mod][idx])  # (L, D), pre-aligned
            if self.zero_pad_mask:
                # Trim trailing all-zero timesteps so collate_multimodal masks them.
                nonzero = (seq.abs().sum(dim=1) > 0).nonzero()
                if len(nonzero) > 0:
                    seq = seq[: int(nonzero[-1]) + 1]
                else:
                    seq = seq[:1]
            sample[mod] = seq
        return sample

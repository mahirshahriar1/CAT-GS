"""
UR-FUNNY dataset (tri-modal humor detection: audio, visual, text).

Expects the converted per-split pickle files produced by `convert_urfunny()`:
    <data_dir>/audio_features_{split}.pkl    list of (seq_len, 81)  COVAREP
    <data_dir>/visual_features_{split}.pkl   list of (seq_len, 371) OpenFace
    <data_dir>/text_features_{split}.pkl     list of (seq_len, 300) GloVe
    <data_dir>/labels_{split}.pkl            list of {0, 1}

The raw UR-FUNNY V2 pickles (data_folds.pkl, language_sdk.pkl,
covarep_features_sdk.pkl, openface_features_sdk.pkl, humor_label_sdk.pkl,
word_embedding_list.pkl) are distributed by the authors at
https://github.com/ROC-HCI/UR-FUNNY — run this file as a script to convert:

    python -m dataset.URFunnyDataset --input_dir data/URFUNNY/raw --output_dir data/URFUNNY
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


class URFunnyDataset(Dataset):
    """UR-FUNNY dataset for multimodal humor detection (2 classes)."""

    def __init__(self, data_dir, split='train', modalities=('audio', 'visual', 'text')):
        """
        Args:
            data_dir: directory containing the converted pickle files
            split: 'train', 'dev', or 'test'
            modalities: subset of ('audio', 'visual', 'text')
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = list(modalities)

        self.data = {}
        for mod in self.modalities:
            fname = {'audio': 'audio_features', 'visual': 'visual_features', 'text': 'text_features'}[mod]
            self.data[mod] = load_pickle(self.data_dir / f'{fname}_{split}.pkl')

        self.labels = load_pickle(self.data_dir / f'labels_{split}.pkl')

        self.feature_dims = {mod: self.data[mod][0].shape[1] for mod in self.modalities}
        print(f"[UR-FUNNY] {split}: {len(self.labels)} samples, feature dims {self.feature_dims}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'label': torch.tensor(int(self.labels[idx]), dtype=torch.long)}
        for mod in self.modalities:
            sample[mod] = torch.FloatTensor(np.asarray(self.data[mod][idx]))
        return sample


def collate_multimodal(batch):
    """Pad variable-length sequences to the batch max length; add boolean masks.

    Returns a dict: {mod: (B, L, D) float, f'{mod}_mask': (B, L) bool, 'label': (B,)}.
    Shared by URFunnyDataset and MOSIDataset.
    """
    modalities = [k for k in batch[0].keys() if k != 'label']
    collated = {'label': torch.stack([sample['label'] for sample in batch])}

    for mod in modalities:
        max_len = max(sample[mod].size(0) for sample in batch)
        feat_dim = batch[0][mod].size(1)
        padded, masks = [], []
        for sample in batch:
            seq = sample[mod]
            seq_len = seq.size(0)
            if seq_len < max_len:
                seq = torch.cat([seq, torch.zeros(max_len - seq_len, feat_dim)], dim=0)
            padded.append(seq)
            masks.append(torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)]))
        collated[mod] = torch.stack(padded)
        collated[f'{mod}_mask'] = torch.stack(masks).bool()

    return collated


def convert_urfunny(input_dir, output_dir, use_context=True, max_seq_len=None):
    """Convert the raw UR-FUNNY V2 pickles into per-split feature files.

    Args:
        input_dir: directory with the six original UR-FUNNY pickle files
        output_dir: destination for the converted files
        use_context: prepend context-sentence features to the punchline
        max_seq_len: optional truncation length
    """
    from tqdm import tqdm

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original UR-FUNNY pickle files...")
    data_folds = load_pickle(input_dir / 'data_folds.pkl')
    language_sdk = load_pickle(input_dir / 'language_sdk.pkl')
    covarep_sdk = load_pickle(input_dir / 'covarep_features_sdk.pkl')
    openface_sdk = load_pickle(input_dir / 'openface_features_sdk.pkl')
    humor_labels = load_pickle(input_dir / 'humor_label_sdk.pkl')
    word_embeddings = load_pickle(input_dir / 'word_embedding_list.pkl')

    for split_name in ['train', 'dev', 'test']:
        audio_features, visual_features, text_features, labels = [], [], [], []

        for video_id in tqdm(data_folds[split_name], desc=f"Converting {split_name}"):
            punchline_word_ids = np.array(language_sdk[video_id]['punchline_embedding_indexes'])
            punchline_covarep = np.array(covarep_sdk[video_id]['punchline_features'])
            punchline_openface = np.array(openface_sdk[video_id]['punchline_features'])
            punchline_text = np.array([word_embeddings[int(i)] for i in punchline_word_ids])

            if use_context and len(language_sdk[video_id]['context_embedding_indexes']) > 0:
                ctx_word_ids = np.concatenate(language_sdk[video_id]['context_embedding_indexes'])
                ctx_covarep = np.concatenate(covarep_sdk[video_id]['context_features'])
                ctx_openface = np.concatenate(openface_sdk[video_id]['context_features'])
                ctx_text = np.array([word_embeddings[int(i)] for i in ctx_word_ids])
                text_seq = np.vstack([ctx_text, punchline_text])
                audio_seq = np.vstack([ctx_covarep, punchline_covarep])
                visual_seq = np.vstack([ctx_openface, punchline_openface])
            else:
                text_seq, audio_seq, visual_seq = punchline_text, punchline_covarep, punchline_openface

            if max_seq_len is not None:
                text_seq = text_seq[:max_seq_len]
                audio_seq = audio_seq[:max_seq_len]
                visual_seq = visual_seq[:max_seq_len]

            text_features.append(text_seq)
            audio_features.append(audio_seq)
            visual_features.append(visual_seq)
            labels.append(humor_labels[video_id])

        save_pickle(audio_features, output_dir / f'audio_features_{split_name}.pkl')
        save_pickle(visual_features, output_dir / f'visual_features_{split_name}.pkl')
        save_pickle(text_features, output_dir / f'text_features_{split_name}.pkl')
        save_pickle(labels, output_dir / f'labels_{split_name}.pkl')
        print(f"Saved {len(labels)} {split_name} samples "
              f"(humor: {sum(labels)}, non-humor: {len(labels) - sum(labels)})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert raw UR-FUNNY V2 pickles')
    parser.add_argument('--input_dir', required=True, help='directory with the six raw UR-FUNNY pickles')
    parser.add_argument('--output_dir', required=True, help='destination for converted per-split files')
    parser.add_argument('--no_context', action='store_true', help='use punchline features only')
    parser.add_argument('--max_seq_len', type=int, default=None)
    args = parser.parse_args()
    convert_urfunny(args.input_dir, args.output_dir,
                    use_context=not args.no_context, max_seq_len=args.max_seq_len)

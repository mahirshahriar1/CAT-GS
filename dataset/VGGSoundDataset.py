# vggsound_dataset_images_first.py
import csv, os, glob, random
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Your config.py should define: use_video_frames (int), batch_size, etc.
import config

class VGGSound(Dataset):
    """
    VGGSound loader with optional audio dependency.
    - If require_audio=False: index samples even if audio is missing.
    - If load_audio=False: __getitem__ returns a zeros spectrogram placeholder.

    Frames layout assumed:
      <video_root>/<split>/Image-01-FPS/<youtube_id>_<timestamp>/

    Audio (if used) searched in:
      <audio_root>/<split>/<youtube_id>_<timestamp>.wav|.mp3
      <audio_root>/<youtube_id>_<timestamp>.wav|.mp3
    """

    def __init__(
        self,
        mode='train',
        csv_path=None,
        video_root=None,
        audio_root=None,
        fps_folder='Image-01-FPS',
        allow_any_fps_folder=True,
        allow_no_fps_folder=True,
        use_video_frames=None,
        min_frames_required=None,
        require_audio=False,    # <-- make audio optional for indexing
        load_audio=False,       # <-- skip audio load in __getitem__
        show_progress=True,
        log_first_n_issues=20
    ):
        assert mode in ('train', 'test')
        self.mode = mode
        # Fall back to the paths defined in config.py
        self.csv_path = csv_path or getattr(config, 'vggsound_csv', './data/VGGSound/vggsound.csv')
        self.video_root = video_root or getattr(config, 'vggsound_video_root', './data/VGGSound/video/frames')
        self.audio_root = audio_root or getattr(config, 'vggsound_audio_root', './data/VGGSound/audio')
        self.fps_folder = fps_folder
        self.allow_any_fps_folder = allow_any_fps_folder
        self.allow_no_fps_folder = allow_no_fps_folder
        self.use_video_frames = use_video_frames or getattr(config, 'use_video_frames', 8)
        self.min_frames_required = (min_frames_required
                                    if min_frames_required is not None
                                    else self.use_video_frames)
        self.require_audio = require_audio
        self.load_audio = load_audio
        self.show_progress = show_progress
        self.log_first_n_issues = log_first_n_issues

        self.video, self.audio, self.label = [], [], []
        self.classes = []
        self.skipped_samples = 0  # Track skipped samples due to corrupted images

        total_rows = None
        if self.show_progress:
            with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
                total_rows = max(0, sum(1 for _ in f) - 1)

        skip_reasons = Counter()
        reason_examples = defaultdict(list)

        with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # header

            iterator = reader
            if self.show_progress and _HAS_TQDM:
                iterator = tqdm(iterator, total=total_rows, desc="VGGSound: scanning CSV", unit="row")

            for row in iterator:
                try:
                    yid = row[0].strip()
                    ts_raw = row[1].strip()
                    try:
                        ts_int = int(float(ts_raw))
                    except:
                        ts_int = int(ts_raw)
                    ts6 = f"{ts_int:06d}"
                    ts5 = f"{ts_int:05d}"

                    label_str = row[2].strip()
                    split_raw = row[3].strip().lower()
                    split = 'train' if split_raw.startswith('train') else ('test' if split_raw.startswith('test') else split_raw)

                    if label_str not in self.classes:
                        self.classes.append(label_str)

                    if split != self.mode:
                        continue

                    vdir = self._resolve_video_dir(yid, (ts6, ts5), split)
                    if vdir is None:
                        r = "video_dir_not_found"
                        skip_reasons[r] += 1
                        if len(reason_examples[r]) < self.log_first_n_issues:
                            reason_examples[r].append(f"{yid} {ts6}/{ts5} split={split}")
                        continue

                    frames = self._safe_listdir(vdir)
                    if len(frames) < self.min_frames_required:
                        r = f"not_enough_frames(<{self.min_frames_required})"
                        skip_reasons[r] += 1
                        if len(reason_examples[r]) < self.log_first_n_issues:
                            reason_examples[r].append(f"{vdir} has {len(frames)} frames")
                        continue

                    afile = self._resolve_audio_file(yid, (ts6, ts5), split)

                    # If audio required but missing -> skip; else accept with placeholder
                    if self.require_audio and afile is None:
                        r = "audio_not_found"
                        skip_reasons[r] += 1
                        if len(reason_examples[r]) < self.log_first_n_issues:
                            reason_examples[r].append(f"{yid} {ts6}/{ts5} split={split}")
                        continue

                    self.video.append(vdir)
                    self.audio.append(afile)  # may be None
                    self.label.append(label_str)

                except Exception as e:
                    r = f"row_error:{type(e).__name__}"
                    skip_reasons[r] += 1
                    if len(reason_examples[r]) < self.log_first_n_issues:
                        reason_examples[r].append(f"row={row} err={e}")

        self.classes = sorted(set(self.classes))
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.label = [class_to_idx[c] for c in self.label]

        if self.show_progress:
            print(f"[{self.mode}] samples={len(self.video)} | classes={len(self.classes)} "
                  f"| require_audio={self.require_audio} load_audio={self.load_audio}")
            if skip_reasons:
                print("Skip summary:")
                for k, v in skip_reasons.most_common():
                    print(f"  - {k}: {v}")
                print(f"Examples (up to {self.log_first_n_issues} per reason):")
                for r, exs in reason_examples.items():
                    for ex in exs:
                        print(f"    * {r}: {ex}")

        if len(self.video) == 0:
            raise RuntimeError(
                "VGGSound built 0 samples. With audio optional, this means frame paths/padding/folders still mismatched.\n"
                f"  video_root={self.video_root}\n  audio_root={self.audio_root}\n  fps_folder={self.fps_folder}"
            )

        # image transforms
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    # -------- helpers --------
    def _safe_listdir(self, p):
        try:
            return sorted([x for x in os.listdir(p) if not x.startswith('.')])
        except Exception:
            return []

    def _resolve_video_dir(self, yid, ts_candidates, split):
        # 1) exact FPS folder: Image-01-FPS
        for ts in ts_candidates:
            p1 = os.path.join(self.video_root, split, self.fps_folder, f'{yid}_{ts}')
            if os.path.isdir(p1):
                return p1
        # 2) any FPS folder
        if self.allow_any_fps_folder:
            for ts in ts_candidates:
                pattern = os.path.join(self.video_root, split, '*FPS*', f'{yid}_{ts}')
                for h in glob.glob(pattern):
                    if os.path.isdir(h):
                        return h
        # 3) no FPS folder
        if self.allow_no_fps_folder:
            for ts in ts_candidates:
                p3 = os.path.join(self.video_root, split, f'{yid}_{ts}')
                if os.path.isdir(p3):
                    return p3
        return None

    def _resolve_audio_file(self, yid, ts_candidates, split):
        for ts in ts_candidates:
            for ext in ('.wav', '.mp3'):
                p = os.path.join(self.audio_root, split, f'{yid}_{ts}{ext}')
                if os.path.isfile(p): return p
            for ext in ('.wav', '.mp3'):
                p = os.path.join(self.audio_root, f'{yid}_{ts}{ext}')
                if os.path.isfile(p): return p
        return None

    # -------- dataset API --------
    def __len__(self):
        return len(self.video)
    
    def get_skip_count(self):
        """Return the number of samples skipped due to corrupted images."""
        return self.skipped_samples

    def __getitem__(self, idx):
        # --- images only (always) ---
        image_files = self._safe_listdir(self.video[idx])
        replace_flag = len(image_files) < self.use_video_frames
        indices = np.random.choice(len(image_files), size=self.use_video_frames, replace=replace_flag)
        images = torch.zeros((self.use_video_frames, 3, 224, 224))
        
        # Try to load all required images
        for i, j in enumerate(indices):
            try:
                img_path = os.path.join(self.video[idx], image_files[j])
                img = Image.open(img_path).convert('RGB')
                images[i] = self.transform(img)
            except Exception as e:
                # If any image fails, skip this entire sample
                self.skipped_samples += 1
                if self.skipped_samples <= 10:  # Log first 10 skips
                    print(f"[Skip #{self.skipped_samples}] Corrupted image: {img_path}")
                elif self.skipped_samples == 11:
                    print(f"[Skip #11+] Further skips will not be printed individually...")
                # Return next sample
                return self.__getitem__((idx + 1) % len(self.video))
        
        images = torch.permute(images, (1, 0, 2, 3))  # (C, T, H, W)

        # --- audio (optional) ---
        if self.load_audio and self.audio[idx] is not None:
            import librosa  # import here to avoid dependency if unused
            sample, rate = librosa.load(self.audio[idx], sr=16000, mono=True)
            while len(sample) / rate < 10.0:
                sample = np.tile(sample, 2)
            start = random.randint(0, rate * 5)
            new_sample = sample[start:start + rate * 5]
            new_sample = np.clip(new_sample, -1.0, 1.0)
            spect = librosa.stft(new_sample, n_fft=256, hop_length=128)
            spect = np.log(np.abs(spect) + 1e-7)
        else:
            # zeros placeholder: shape similar to (freq, time)
            # n_fft=256 -> freq bins=129; for 5s @16k with hop 128 -> ~626 frames
            spect = np.zeros((129, 626), dtype=np.float32)

        label = self.label[idx]
        return spect, images, label

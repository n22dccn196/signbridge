# =============================================================
# dataset.py — BSLDataset: load .npy features, split, augment
# =============================================================
"""
File naming conventions:
  Regular: {episode}_{word}_{t_center:.2f}.npy
  Augmented: AUG_{word}_{aug_name}_{uid}.npy

Split strategy:
  - Group regular files by episode → split episodes (not files)
    to prevent data leakage between train/val/test
  - AUG_ files always go into train only
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict

from config import (
    NPY_DIR, CLASSES, CLASS2IDX, NUM_CLASSES,
    N_FRAMES, N_LANDMARKS, N_COORDS,
    VAL_RATIO, TEST_RATIO, SEED,
    BATCH_SIZE,
    ROOTREL_CLIP_MIN, ROOTREL_CLIP_MAX,
    MAX_SAMPLES_PER_CLASS,
)
import math
import config as _active_config

WLASL_LABEL_ALIASES = getattr(_active_config, "WLASL_LABEL_ALIASES", {})
SMALL_DATASET_THRESHOLD = 500


# ─── Helper ───────────────────────────────────────────────────

def _parse_filename(fn: str):
    """Return (episode, word, t_center) or None if unparseable.
    Handles prefixes: MANUAL_, AUTO_, AUG_, WLASL_
    """
    fn = fn[:-4]  # strip .npy

    # Strip hybrid prefixes (MANUAL_ or AUTO_)
    if fn.startswith("MANUAL_"):
        fn = fn[7:]  # len("MANUAL_") = 7
    elif fn.startswith("AUTO_"):
        fn = fn[5:]  # len("AUTO_") = 5

    # WLASL format: WLASL_{split}_{word}_{video_id}
    if fn.startswith("WLASL_"):
        rest = fn[6:]  # strip "WLASL_"
        # split is train/val/test
        for split in ("train", "val", "test"):
            prefix = split + "_"
            if rest.startswith(prefix):
                remainder = rest[len(prefix):]
                # remainder = {word}_{video_id}
                parts = remainder.rsplit("_", 1)
                if len(parts) == 2:
                    word, vid_id = parts
                    word = WLASL_LABEL_ALIASES.get(word, word)
                    if word in CLASS2IDX:
                        return (f"WLASL_{split}", word, None)
                break
        return None

    if fn.startswith("AUG_"):
        # AUG_{word}_{aug_name}_{uid}
        parts = fn[4:].split("_", 1)
        word = parts[0]
        return ("AUG", word, None)
    # Regular: {episode}_{word}_{t_center}
    parts = fn.rsplit("_", 2)
    if len(parts) != 3:
        return None
    episode, word, t_str = parts
    if word not in CLASS2IDX:
        return None
    try:
        t = float(t_str)
    except ValueError:
        return None
    return (episode, word, t)


def _episode_split(episode_word_files, val_ratio, test_ratio, seed):
    """
    Split by unique episodes to avoid leakage.

    Returns: train_files, val_files, test_files
    """
    rng = random.Random(seed)

    # group files by (episode, word) → each episode can be in only one split
    ep_to_files = defaultdict(list)
    for f in episode_word_files:
        info = _parse_filename(os.path.basename(f))
        if info is None:
            continue
        ep, word, t = info
        ep_to_files[ep].append(f)

    episodes = sorted(ep_to_files.keys())
    rng.shuffle(episodes)

    n = len(episodes)
    n_val  = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))

    test_eps  = set(episodes[:n_test])
    val_eps   = set(episodes[n_test:n_test + n_val])
    train_eps = set(episodes[n_test + n_val:])

    train_f = [f for ep in train_eps for f in ep_to_files[ep]]
    val_f   = [f for ep in val_eps   for f in ep_to_files[ep]]
    test_f  = [f for ep in test_eps  for f in ep_to_files[ep]]

    return train_f, val_f, test_f


def _stratified_split(files, val_ratio, test_ratio, seed):
    """
    Stratified file-level split: ensures every class has samples in all splits.
    Used for small datasets where episode-based splitting would leave classes empty.

    Returns: train_files, val_files, test_files
    """
    rng = random.Random(seed)

    # Group by class
    class_files = defaultdict(list)
    for f in files:
        info = _parse_filename(os.path.basename(f))
        if info is None:
            continue
        _, word, _ = info
        if word in CLASS2IDX:
            class_files[word].append(f)

    train_f, val_f, test_f = [], [], []

    for word in sorted(class_files.keys()):
        flist = class_files[word][:]
        rng.shuffle(flist)
        n = len(flist)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        # Ensure at least 1 in train
        if n_test + n_val >= n:
            n_test = max(1, n // 3)
            n_val = max(1, n // 3)
            if n_test + n_val >= n:
                n_test = 1
                n_val = min(1, n - 2)

        test_f.extend(flist[:n_test])
        val_f.extend(flist[n_test:n_test + n_val])
        train_f.extend(flist[n_test + n_val:])

    return train_f, val_f, test_f


def _wlasl_split(files):
    """
    Split WLASL files using the split embedded in their filenames.
    WLASL files are named: WLASL_{split}_{word}_{video_id}.npy

    Returns: train_files, val_files, test_files
    """
    train_f, val_f, test_f = [], [], []
    for f in files:
        bn = os.path.basename(f)
        info = _parse_filename(bn)
        if info is None:
            continue
        episode, word, _ = info
        if word not in CLASS2IDX:
            continue
        if episode == "WLASL_train":
            train_f.append(f)
        elif episode == "WLASL_val":
            val_f.append(f)
        elif episode == "WLASL_test":
            test_f.append(f)
    return train_f, val_f, test_f


def build_splits(npy_dir=NPY_DIR,
                 val_ratio=VAL_RATIO,
                 test_ratio=TEST_RATIO,
                 seed=SEED,
                 max_per_class=MAX_SAMPLES_PER_CLASS):
    """
    Scan npy_dir, split into train/val/test.
    AUG_ files go to train only.
    If max_per_class is set, caps training samples per class
    (prioritizing MANUAL_ files over AUTO_ files).
    Returns dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    all_files = [
        os.path.join(npy_dir, f)
        for f in os.listdir(npy_dir)
        if f.endswith(".npy") and not f.startswith(".tmp.")
    ]

    regular = [f for f in all_files if not os.path.basename(f).startswith("AUG_")]
    aug     = [f for f in all_files if os.path.basename(f).startswith("AUG_")]

    # Filter to known classes
    regular = [f for f in regular if _parse_filename(os.path.basename(f)) is not None]
    aug     = [f for f in aug     if _parse_filename(os.path.basename(f)) is not None
               and _parse_filename(os.path.basename(f))[1] in CLASS2IDX]

    # Detect WLASL data: if any files start with WLASL_, use embedded splits
    has_wlasl = any(os.path.basename(f).startswith("WLASL_") for f in regular)

    if has_wlasl:
        print(f"[Dataset] WLASL data detected ({len(regular)} files) -> using embedded splits")
        train_r, val_r, test_r = _wlasl_split(regular)
    elif len(regular) < SMALL_DATASET_THRESHOLD:
        print(f"[Dataset] Small dataset ({len(regular)} files) → stratified file-level split")
        train_r, val_r, test_r = _stratified_split(regular, val_ratio, test_ratio, seed)
    else:
        train_r, val_r, test_r = _episode_split(regular, val_ratio, test_ratio, seed)
    # AUG only in train
    train_files = train_r + aug
    val_files   = val_r
    test_files  = test_r

    # ── Anti-bias: cap training samples per class ──
    if max_per_class is not None and max_per_class > 0:
        rng = random.Random(seed)
        # Group train files by class
        class_files = defaultdict(list)
        for f in train_files:
            info = _parse_filename(os.path.basename(f))
            if info and info[1] in CLASS2IDX:
                class_files[info[1]].append(f)

        capped_train = []
        capped_count = 0
        for word in sorted(class_files.keys()):
            files = class_files[word]
            if len(files) <= max_per_class:
                capped_train.extend(files)
            else:
                # Prioritize MANUAL_ files over AUTO_ files
                manual = [f for f in files if os.path.basename(f).startswith("MANUAL_")]
                auto   = [f for f in files if not os.path.basename(f).startswith("MANUAL_")]
                rng.shuffle(auto)
                selected = manual + auto[:max(0, max_per_class - len(manual))]
                capped_train.extend(selected[:max_per_class])
                capped_count += len(files) - max_per_class

        if capped_count > 0:
            print(f"[Dataset] Anti-bias: capped {capped_count} excess training samples "
                  f"(max {max_per_class}/class)")
        train_files = capped_train

    print(f"[Dataset] train={len(train_files):>6}  val={len(val_files):>5}  test={len(test_files):>5}")
    return {"train": train_files, "val": val_files, "test": test_files}


# ─── Class-Adaptive Skeletal Augmentor ────────────────────────

class SkeletalAugmentor:
    """
    Class-adaptive augmentation: minority classes get heavier transforms.

    Tiers (based on training class counts):
      heavy   (n < 30):  high probabilities, strong transforms
      moderate (30 ≤ n < 100): medium probabilities
      light   (n ≥ 100): baseline probabilities (original defaults)

    New transforms beyond Plan A:
      - rotation: ±15° around root (simulates signer tilt)
      - hand_jitter: independent noise on hand landmarks (33-74)
      - temporal_mask: zero-out random 3-5 frame windows (occlusion)
    """

    # Probability tables:  (hflip, scale, time_shift, noise, speed, dropout, rotation, hand_jitter, temporal_mask)
    TIERS = {
        "heavy":    (0.50, 0.70, 0.50, 0.70, 0.50, 0.40, 0.40, 0.50, 0.30),
        "moderate": (0.40, 0.60, 0.40, 0.60, 0.40, 0.30, 0.25, 0.35, 0.20),
        "light":    (0.30, 0.50, 0.30, 0.50, 0.30, 0.20, 0.10, 0.15, 0.10),
    }

    @staticmethod
    def tier_for_count(n: int) -> str:
        if n < 30:
            return "heavy"
        elif n < 100:
            return "moderate"
        else:
            return "light"

    @staticmethod
    def _rotation(x, max_deg=15.0):
        """Rotate all landmarks around root (0,0) by ±max_deg degrees."""
        angle = torch.empty(1).uniform_(-max_deg, max_deg).item()
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        out = x.clone()
        nonzero = (x.abs().sum(dim=-1) > 0)  # (T, 75)
        # Apply 2D rotation
        rx = x[:, :, 0] * cos_a - x[:, :, 1] * sin_a
        ry = x[:, :, 0] * sin_a + x[:, :, 1] * cos_a
        out[:, :, 0] = torch.where(nonzero, rx, x[:, :, 0])
        out[:, :, 1] = torch.where(nonzero, ry, x[:, :, 1])
        return out

    @staticmethod
    def _hand_jitter(x, sigma=0.008):
        """Independent Gaussian noise on hand landmarks (indices 33-74)."""
        out = x.clone()
        hand_noise = torch.randn(x.shape[0], 42, 2) * sigma  # 42 hand landmarks
        nonzero = (x[:, 33:75].abs().sum(dim=-1) > 0).unsqueeze(-1)  # (T, 42, 1)
        out[:, 33:75] = x[:, 33:75] + hand_noise * nonzero.float()
        return out

    @staticmethod
    def _temporal_mask(x, min_len=3, max_len=5):
        """Zero out a random contiguous window of 3-5 frames (occlusion sim)."""
        T = x.shape[0]
        win_len = random.randint(min_len, max_len)
        if win_len >= T:
            return x
        start = random.randint(0, T - win_len)
        out = x.clone()
        out[start:start + win_len] = 0.0
        return out

    def augment(self, x, label: int, class_counts: np.ndarray):
        """
        Apply class-adaptive augmentation pipeline.
        x: tensor (T, 75, 2), label: int, class_counts: array of per-class counts.
        """
        n = int(class_counts[label]) if label < len(class_counts) else 100
        tier = self.tier_for_count(n)
        probs = self.TIERS[tier]
        p_hflip, p_scale, p_tshift, p_noise, p_speed, p_drop, p_rot, p_hjit, p_tmask = probs

        if random.random() < p_hflip:
            x = BSLDataset._hflip(x)
        if random.random() < p_scale:
            x = BSLDataset._scale_jitter(x)
        if random.random() < p_tshift:
            x = BSLDataset._time_shift(x)
        if random.random() < p_noise:
            x = BSLDataset._noise(x)
        if random.random() < p_speed:
            x = BSLDataset._speed_perturb(x)
        if random.random() < p_drop:
            x = BSLDataset._landmark_dropout(x)
        if random.random() < p_rot:
            x = self._rotation(x)
        if random.random() < p_hjit:
            x = self._hand_jitter(x)
        if random.random() < p_tmask:
            x = self._temporal_mask(x)

        return x


# Global augmentor instance (stateless, safe to share)
_skeletal_augmentor = SkeletalAugmentor()


# ─── Dataset ──────────────────────────────────────────────────

class BSLDataset(Dataset):
    """
    Loads (50, 75, 2) .npy skeleton windows.
    Returns (tensor [50, 150], label int).
    """

    def __init__(self, files: list, augment: bool = False,
                 class_counts: np.ndarray | None = None):
        self.augment = augment
        self._use_adaptive = class_counts is not None
        self._ext_class_counts = class_counts  # from full training set
        # Build (path, label) list, skip unknown words
        self.samples = []
        for f in files:
            info = _parse_filename(os.path.basename(f))
            if info is None:
                continue
            _, word, _ = info
            if word not in CLASS2IDX:
                continue
            self.samples.append((f, CLASS2IDX[word]))

        # Class counts for weighted sampler
        self._class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _, lbl in self.samples:
            self._class_counts[lbl] += 1

    # ── augmentation ops (Plan A — webcam-robust) ──────────────
    @staticmethod
    def _hflip(x):
        """Horizontal flip: negate X coords (root-relative, centered at 0)."""
        out = x.clone()
        out[:, :, 0] = -x[:, :, 0]   # root-relative: flip = negate X
        return out

    @staticmethod
    def _scale_jitter(x):
        """Random scale ±5% + translation ±3% (simulates camera distance/position)."""
        sx = 1.0 + torch.empty(1).uniform_(-0.05, 0.05).item()
        sy = 1.0 + torch.empty(1).uniform_(-0.05, 0.05).item()
        tx = torch.empty(1).uniform_(-0.03, 0.03).item()
        ty = torch.empty(1).uniform_(-0.03, 0.03).item()
        out = x.clone()
        # Only apply to non-zero landmarks
        nonzero = (x.abs().sum(dim=-1) > 0).unsqueeze(-1)  # (T, 75, 1)
        out[:, :, 0] = torch.where(nonzero.squeeze(-1), x[:, :, 0] * sx + tx, x[:, :, 0])
        out[:, :, 1] = torch.where(nonzero.squeeze(-1), x[:, :, 1] * sy + ty, x[:, :, 1])
        return out

    @staticmethod
    def _time_shift(x):
        """Random temporal shift ±3 frames."""
        shift = random.randint(-3, 3)
        return torch.roll(x, shift, dims=0)

    @staticmethod
    def _noise(x):
        """Gaussian noise ±0.5% of shoulder width (simulates webcam jitter)."""
        noise = torch.randn_like(x) * 0.005
        # Only apply to non-zero landmarks
        nonzero = (x.abs().sum(dim=-1) > 0).unsqueeze(-1)
        return x + noise * nonzero.float()

    @staticmethod
    def _speed_perturb(x):
        """Random speed change: stretch or compress temporally by ±15%."""
        T, L, C = x.shape  # (50, 75, 2)
        factor = 1.0 + torch.empty(1).uniform_(-0.15, 0.15).item()
        new_T = max(10, int(T * factor))
        # Reshape to (1, L*C, T) for 1D interpolation
        x_flat = x.reshape(T, L * C).permute(1, 0).unsqueeze(0)  # (1, 150, T)
        x_rs = torch.nn.functional.interpolate(
            x_flat, size=new_T, mode='linear', align_corners=True
        ).squeeze(0).permute(1, 0).reshape(new_T, L, C)  # (new_T, 75, 2)
        # Pad or crop back to T
        if x_rs.shape[0] >= T:
            start = (x_rs.shape[0] - T) // 2
            x_rs = x_rs[start:start + T]
        else:
            pad_n = T - x_rs.shape[0]
            pad = x_rs[-1:].expand(pad_n, -1, -1)
            x_rs = torch.cat([x_rs, pad], dim=0)
        return x_rs

    @staticmethod
    def _landmark_dropout(x, drop_prob=0.05):
        """Randomly zero out some landmarks (simulates occlusion)."""
        out = x.clone()
        mask = torch.rand(x.shape[0], x.shape[1]) > drop_prob  # (T, 75)
        mask = mask.unsqueeze(-1).expand_as(x)  # (T, 75, 2)
        return out * mask.float()

    def _augment(self, x):
        """Apply augmentation pipeline (Plan A — webcam-robust)."""
        if random.random() < 0.3:
            x = self._hflip(x)
        if random.random() < 0.5:
            x = self._scale_jitter(x)
        if random.random() < 0.3:
            x = self._time_shift(x)
        if random.random() < 0.5:
            x = self._noise(x)
        if random.random() < 0.3:
            x = self._speed_perturb(x)
        if random.random() < 0.2:
            x = self._landmark_dropout(x)
        return x

    # ── Dataset API ───────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)          # (T, 75, 2) or (T, 150)

        # Handle pre-flattened data (e.g. WLASL: shape (T, 150))
        if arr.ndim == 2 and arr.shape[-1] == N_LANDMARKS * N_COORDS:
            arr = arr.reshape(arr.shape[0], N_LANDMARKS, N_COORDS)  # → (T, 75, 2)

        # Enforce exactly N_FRAMES frames
        T = arr.shape[0]
        if T > N_FRAMES:
            arr = arr[:N_FRAMES]                        # truncate
        elif T < N_FRAMES:
            pad = np.tile(arr[[-1]], (N_FRAMES - T, 1, 1))  # repeat last frame
            arr = np.concatenate([arr, pad], axis=0)    # (N_FRAMES, 75, 2)

        # Root-relative normalization: data is already root-relative (pre-computed),
        # just clip extreme outliers from noisy detections
        arr = np.clip(arr, ROOTREL_CLIP_MIN, ROOTREL_CLIP_MAX)

        x = torch.from_numpy(arr)                       # (N_FRAMES, 75, 2)
        if self.augment:
            if self._use_adaptive:
                x = _skeletal_augmentor.augment(x, label, self._ext_class_counts)
            else:
                x = self._augment(x)
        x = x.reshape(N_FRAMES, N_LANDMARKS * N_COORDS) # (50, 150)
        return x, label

    def get_weights(self):
        """Sample weights for WeightedRandomSampler.
        Uses sqrt-inverse-frequency: w = 1/sqrt(n).
        Proven to outperform true inverse-frequency (1/n) which is too aggressive
        and destroys accuracy on large classes. Sqrt provides moderate rebalancing.
        """
        weights = np.zeros(len(self.samples), dtype=np.float32)
        class_w = 1.0 / np.sqrt(self._class_counts + 1.0)  # sqrt-inverse-frequency
        for i, (_, lbl) in enumerate(self.samples):
            weights[i] = class_w[lbl]
        return weights

    def print_class_distribution(self, split_name: str = "train"):
        """Print class distribution and sampler weight summary."""
        from config import CLASSES
        print(f"\n  [{split_name}] Class distribution ({len(self.samples)} samples):")
        class_w = 1.0 / (self._class_counts + 1.0)
        for i, cls in enumerate(CLASSES):
            n = int(self._class_counts[i])
            w = class_w[i]
            bar = '#' * min(50, max(1, n // 5))
            print(f"    {cls:>15}: {n:>4} samples  w={w:.4f}  {bar}")
        print(f"  Max/min ratio: {self._class_counts.max()}/{max(1,self._class_counts[self._class_counts>0].min())} "
              f"= {self._class_counts.max()/max(1,self._class_counts[self._class_counts>0].min()):.1f}x")


# ─── DataLoader factory ───────────────────────────────────────

def get_loaders(npy_dir=NPY_DIR,
                batch_size=BATCH_SIZE,
                val_ratio=VAL_RATIO,
                test_ratio=TEST_RATIO,
                seed=SEED,
                num_workers=0,
                use_adaptive_aug=True):
    """
    Returns: (train_loader, val_loader, test_loader)
    Train uses WeightedRandomSampler for class balance.
    use_adaptive_aug: if True, uses SkeletalAugmentor; if False, uses Plan A.
    """
    splits = build_splits(npy_dir, val_ratio, test_ratio, seed)

    # Build train dataset first to get class counts for adaptive augmentation
    train_ds = BSLDataset(splits["train"], augment=False)  # temp, no augment yet
    class_counts = train_ds._class_counts.copy()

    # Rebuild with adaptive augmentation using class counts
    train_ds = BSLDataset(splits["train"], augment=True,
                          class_counts=class_counts if use_adaptive_aug else None)
    val_ds   = BSLDataset(splits["val"],   augment=False)
    test_ds  = BSLDataset(splits["test"],  augment=False)

    # Weighted sampler for train
    weights   = train_ds.get_weights()
    sampler   = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ─── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    train_l, val_l, test_l = get_loaders(batch_size=8)
    x, y = next(iter(train_l))
    print(f"Batch shape: {x.shape}  labels: {y[:4]}")
    print(f"X range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Train batches: {len(train_l)}  Val: {len(val_l)}  Test: {len(test_l)}")

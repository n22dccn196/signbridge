# =============================================================
# inference_antibias.py — Real-time anti-bias for camera inference
# =============================================================
"""
Phase 4 — Real-time Anti-bias.

Three mechanisms to combat bias during live webcam inference:

1. **Logit penalization**: Scale down logits for overrepresented classes
   so the model doesn't default to "good" or "different" on ambiguous frames.

2. **Dynamic confidence thresholds**: Require higher confidence for common
   classes, lower for rare classes.  Prevents majority-class hallucination.

3. **Movement filter**: Reject predictions during transition frames where
   hand movement is too low (signer is resting or transitioning between signs).

Usage:
    from inference_antibias import AntibiaFilter

    # At init (once):
    filt = AntibiaFilter.from_config()

    # Per prediction (in Predictor.predict):
    probs = model(x)   # raw probabilities from ensemble/model
    label, conf = filt.apply(probs, frame_buffer)
"""
import numpy as np
from typing import Optional


class AntibiaFilter:
    """
    Post-inference anti-bias filter for real-time BSL recognition.

    Parameters
    ----------
    class_counts : np.ndarray
        Training class counts (length = num_classes).
    num_classes : int
        Number of classes.
    idx2class : dict
        Mapping from class index to class name.
    logit_penalty_strength : float
        How strongly to penalize overrepresented classes (default 1.0).
    min_confidence : float
        Global minimum confidence below which we reject (default 0.35).
    movement_threshold : float
        Minimum hand landmark variance to accept a prediction (default 0.002).
    movement_window : int
        Number of trailing frames to compute movement over (default 5).
    """

    def __init__(
        self,
        class_counts: np.ndarray,
        num_classes: int,
        idx2class: dict,
        logit_penalty_strength: float = 1.0,
        min_confidence: float = 0.35,
        movement_threshold: float = 0.002,
        movement_window: int = 5,
    ):
        self.num_classes = num_classes
        self.idx2class = idx2class
        self.min_confidence = min_confidence
        self.movement_threshold = movement_threshold
        self.movement_window = movement_window

        counts = np.array(class_counts, dtype=np.float64)
        counts = np.maximum(counts, 1.0)  # avoid div-by-zero

        # ── 1. Logit penalty factors ──
        # penalty_c = sqrt(count_c / median(counts)), clamped [1.0, 3.0]
        # Divide raw probs by this to penalize overrepresented classes
        median_n = float(np.median(counts[counts > 0]))
        self.penalty = np.sqrt(counts / median_n)
        self.penalty = np.clip(self.penalty, 1.0, 3.0).astype(np.float32)
        # Scale by strength
        self.penalty = 1.0 + (self.penalty - 1.0) * logit_penalty_strength

        # ── 2. Dynamic confidence thresholds ──
        # thresh_c = 0.45 + 0.40 * (count_c / max(counts)), clamped [0.40, 0.90]
        # Common classes need HIGH confidence, rare classes need LOW confidence
        max_n = float(counts.max())
        self.thresholds = 0.45 + 0.40 * (counts / max_n)
        self.thresholds = np.clip(self.thresholds, 0.40, 0.90).astype(np.float32)

        # Stats for logging
        self._n_calls = 0
        self._n_penalized = 0
        self._n_movement_rejected = 0
        self._n_threshold_rejected = 0
        self._n_accepted = 0

    @classmethod
    def from_config(cls, dataset: Optional[str] = None, **kwargs):
        """Build from config + actual training data class counts."""
        import os
        import config_selector
        # OLD (kept for reference): defaulting to "wlasl" forced 100-class
        # anti-bias when running WLASL-2000 wrappers.
        # cfg = config_selector.get_config(dataset)
        if dataset is None:
            active = config_selector.active_dataset()
            dataset = active if active is not None else "manual"
        cfg = config_selector.get_config(dataset)

        # Get class counts from training data
        npy_dir = cfg.NPY_DIR
        counts = np.zeros(cfg.NUM_CLASSES, dtype=np.int64)
        if os.path.isdir(npy_dir):
            for fn in os.listdir(npy_dir):
                if not fn.endswith(".npy"):
                    continue
                # Skip augmented files — use only original counts
                # so antibias correctly penalizes overrepresented classes
                if fn.startswith("AUG_"):
                    continue
                # Parse class from filename
                name = fn[:-4]
                for prefix in ("MANUAL_", "AUTO_"):
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                        break
                parts = name.rsplit("_", 2)
                word = parts[1] if len(parts) == 3 else None
                if word and word in cfg.CLASS2IDX:
                    counts[cfg.CLASS2IDX[word]] += 1

        if counts.sum() == 0:
            print("[AntibiaFilter] WARNING: No training files found, using uniform counts")
            counts = np.ones(cfg.NUM_CLASSES, dtype=np.int64) * 50

        print(f"[AntibiaFilter] Loaded class counts from {npy_dir}")
        print(f"  min={counts.min()} max={counts.max()} median={int(np.median(counts))}")

        # Load per-class calibrated thresholds if available (from Phase 2F)
        import json
        calib_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "runs_manual", "per_class_thresholds.json"
        )
        calibrated_thresholds = None
        if os.path.exists(calib_path):
            try:
                calibrated_thresholds = json.load(open(calib_path))
                print(f"[AntibiaFilter] Loaded calibrated thresholds from {calib_path}")
            except Exception:
                pass

        instance = cls(
            class_counts=counts,
            num_classes=cfg.NUM_CLASSES,
            idx2class=cfg.IDX2CLASS,
            **kwargs,
        )

        # Override thresholds with calibrated values
        if calibrated_thresholds:
            for cls_name, thresh in calibrated_thresholds.items():
                if cls_name in cfg.CLASS2IDX:
                    idx = cfg.CLASS2IDX[cls_name]
                    instance.thresholds[idx] = float(thresh)
            print(f"[AntibiaFilter] Applied calibrated per-class thresholds")

        return instance

    # ── Movement detection ─────────────────────────────────────

    @staticmethod
    def compute_hand_movement(frames: list, window: int = 5) -> float:
        """
        Compute hand landmark variance over the last `window` frames.
        frames: list of (75, 2) arrays.
        Returns: mean variance of hand landmarks (indices 33-74).
        """
        if len(frames) < 2:
            return 0.0
        recent = frames[-window:]
        arr = np.stack(recent, axis=0)  # (W, 75, 2)
        hands = arr[:, 33:75, :]        # (W, 42, 2)
        # Only consider non-zero landmarks
        nonzero = np.any(hands != 0, axis=(0, 2))  # (42,) — landmarks present in any frame
        if nonzero.sum() < 5:
            return 0.0
        active = hands[:, nonzero, :]   # (W, active, 2)
        # Frame-to-frame displacement variance
        if active.shape[0] < 2:
            return 0.0
        deltas = np.diff(active, axis=0)  # (W-1, active, 2)
        return float(np.var(deltas))

    # ── Main apply method ────────────────────────────────────

    def apply(
        self,
        probs: np.ndarray,
        frame_buffer: Optional[list] = None,
    ) -> tuple:
        """
        Apply anti-bias filtering to raw model probabilities.

        Parameters
        ----------
        probs : np.ndarray
            Raw probability vector (num_classes,) from model/ensemble.
        frame_buffer : list of (75,2) arrays, optional
            Recent keypoint frames for movement detection.

        Returns
        -------
        (label, confidence, was_filtered) : tuple
            label: str — predicted class name (or "" if rejected)
            confidence: float — adjusted confidence
            was_filtered: bool — True if the prediction was modified/rejected
        """
        self._n_calls += 1
        was_filtered = False

        # ── Step 1: Movement filter ──
        if frame_buffer is not None and len(frame_buffer) >= 2:
            movement = self.compute_hand_movement(frame_buffer, self.movement_window)
            if movement < self.movement_threshold:
                self._n_movement_rejected += 1
                return ("", 0.0, True)

        # ── Step 2: Logit penalization ──
        # Divide probs by penalty, then re-normalize
        adjusted = probs / self.penalty
        prob_sum = adjusted.sum()
        if prob_sum > 0:
            adjusted = adjusted / prob_sum
        else:
            adjusted = probs.copy()

        # Check if penalization changed the top prediction
        orig_top = int(probs.argmax())
        new_top = int(adjusted.argmax())
        if orig_top != new_top:
            self._n_penalized += 1
            was_filtered = True

        # ── Step 3: Dynamic confidence threshold ──
        top_idx = int(adjusted.argmax())
        top_conf = float(adjusted[top_idx])
        required = float(self.thresholds[top_idx])

        if top_conf < required:
            # Try second-best
            sorted_idx = np.argsort(adjusted)[::-1]
            for candidate in sorted_idx[:3]:
                c_conf = float(adjusted[candidate])
                c_req = float(self.thresholds[candidate])
                if c_conf >= c_req and c_conf >= self.min_confidence:
                    self._n_threshold_rejected += 1
                    label = self.idx2class.get(int(candidate), "?")
                    return (label, c_conf, True)
            # No candidate passes
            self._n_threshold_rejected += 1
            return ("", 0.0, True)

        if top_conf < self.min_confidence:
            self._n_threshold_rejected += 1
            return ("", 0.0, True)

        self._n_accepted += 1
        label = self.idx2class.get(top_idx, "?")
        return (label, top_conf, was_filtered)

    # ── Diagnostics ────────────────────────────────────────────

    def stats(self) -> dict:
        """Return filtering statistics."""
        return {
            "total_calls": self._n_calls,
            "accepted": self._n_accepted,
            "penalized_changed": self._n_penalized,
            "movement_rejected": self._n_movement_rejected,
            "threshold_rejected": self._n_threshold_rejected,
            "accept_rate": self._n_accepted / max(1, self._n_calls),
        }

    def describe(self) -> str:
        """Print human-readable description of filter parameters."""
        lines = ["[AntibiaFilter] Configuration:"]
        lines.append(f"  Movement threshold: {self.movement_threshold}")
        lines.append(f"  Min confidence: {self.min_confidence}")
        lines.append(f"  Penalty range: [{self.penalty.min():.2f}, {self.penalty.max():.2f}]")
        lines.append(f"  Threshold range: [{self.thresholds.min():.2f}, {self.thresholds.max():.2f}]")
        lines.append(f"  Per-class penalties & thresholds:")
        for i in range(self.num_classes):
            name = self.idx2class.get(i, f"cls{i}")
            lines.append(f"    {name:>15}: penalty={self.penalty[i]:.3f}  "
                         f"thresh={self.thresholds[i]:.3f}")
        return "\n".join(lines)


# ─── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    filt = AntibiaFilter.from_config("manual")
    print(filt.describe())
    # Dummy test
    probs = np.random.dirichlet(np.ones(filt.num_classes))
    label, conf, filtered = filt.apply(probs, None)
    print(f"\nDummy test: label={label}, conf={conf:.3f}, filtered={filtered}")
    print(f"Stats: {filt.stats()}")

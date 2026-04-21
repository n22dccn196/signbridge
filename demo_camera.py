# =============================================================
# demo_camera.py — Real-time BSL recognition via webcam (Phase 2)
# =============================================================
"""
Phase 2 features:
  - Sign Spotting (B1): heuristic velocity-based sign boundary detection
  - Ensemble (F): multi-model soft/hard voting
  - Temporal smoothing: majority vote over last N predictions
  - Sentence log: accumulated recognized words

Modes:
  --mode sliding    Original Phase 1: continuous sliding-window prediction
  --mode spotting   Phase 2: detect sign → classify only when spotted
  --mode hybrid     Both: spotting triggers classify, sliding as fallback

Usage:
    # Phase 2 — sign spotting + ensemble:
    python demo_camera.py --mode spotting --ensemble

    # Phase 1 — sliding window (backwards compat):
    python demo_camera.py --mode sliding --model hybrid

    # Hybrid mode with ensemble:
    python demo_camera.py --mode hybrid --ensemble

Controls:
    Q / ESC — quit
    R       — reset buffer & spotter & sentence
    S       — save current skeleton frame
    M       — cycle mode (sliding → spotting → hybrid)
    SPACE   — pause/unpause
"""

import cv2
import numpy as np
import torch
import logging
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic    # will raise if broken
    mp_draw     = mp.solutions.drawing_utils
except Exception:
    mp = None
    mp_holistic = None
    mp_draw     = None
import collections
import time
import argparse
import os
import sys
from pathlib import Path

# ─── Diagnostic Logger ──────────────────────────────────────
_diag_logger = logging.getLogger("camera_diag")
_diag_logger.setLevel(logging.INFO)
_diag_fh = logging.FileHandler("camera_diagnostics.log", mode="w", encoding="utf-8")
_diag_fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
_diag_logger.addHandler(_diag_fh)
_diag_logger.propagate = False

sys.path.insert(0, str(Path(__file__).parent))

# ─── Early config selection (before importing config) ─────────
# Parse --manual / --wlasl flag early so config_selector activates
# before any `from config import ...` statement.
if "--manual" in sys.argv:
    import config_selector
    config_selector.activate("manual")
elif "--wlasl" in sys.argv:
    import config_selector
    config_selector.activate("wlasl")

from config import (
    CLASSES, IDX2CLASS, NUM_CLASSES, INPUT_SIZE,
    N_FRAMES, N_LANDMARKS,
    CAM_BUFFER_SIZE, CAM_CONFIDENCE, PREDICT_EVERY_N, TRAIN_FPS,
    CHECKPOINT_DIR,
    SPOT_VELOCITY_THRESHOLD,
    ROOTREL_CLIP_MIN, ROOTREL_CLIP_MAX,
    ROOTREL_IDX_LEFT_SHOULDER, ROOTREL_IDX_RIGHT_SHOULDER,
    ROOTREL_FALLBACK_SHOULDER_W, ROOTREL_MIN_SHOULDER_W,
)
from models import get_model
from sign_spotter import SignSpotter
from ensemble import Ensemble
from sentence_builder import SentenceBuilder
try:
    from inference_antibias import AntibiaFilter
    _HAS_ANTIBIAS = True
except ImportError:
    _HAS_ANTIBIAS = False


def extract_keypoints(results) -> np.ndarray:
    """
    Extract 75 landmarks from MediaPipe Holistic results.
    Returns (75, 2) float32 array — values in [0, 1].
    Missing landmarks → (0, 0).
    """
    kps = np.zeros((N_LANDMARKS, 2), dtype=np.float32)

    # Pose (0–32): 33 landmarks
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:33]):
            kps[i] = [lm.x, lm.y]

    # Left hand (33–53): 21 landmarks
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            kps[33 + i] = [lm.x, lm.y]

    # Right hand (54–74): 21 landmarks
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            kps[54 + i] = [lm.x, lm.y]

    return kps


def detection_quality(kps: np.ndarray) -> dict:
    """
    Assess quality of a single keypoint frame (75, 2).
    Returns dict with:
      - has_pose:  bool  (any pose landmarks detected)
      - has_left:  bool  (left hand detected)
      - has_right: bool  (right hand detected)
      - hand_count: int  (0, 1, or 2 hands)
      - nonzero_ratio: float  (fraction of non-zero landmarks)
    """
    pose_nonzero  = np.any(kps[:33] != 0, axis=1).sum()
    left_nonzero  = np.any(kps[33:54] != 0, axis=1).sum()
    right_nonzero = np.any(kps[54:75] != 0, axis=1).sum()
    total_nonzero = np.any(kps != 0, axis=1).sum()

    return {
        "has_pose":  pose_nonzero > 5,
        "has_left":  left_nonzero > 10,
        "has_right": right_nonzero > 10,
        "hand_count": int(left_nonzero > 10) + int(right_nonzero > 10),
        "nonzero_ratio": total_nonzero / 75.0,
    }


def buffer_has_hands(buffer, min_hand_frames: int = 10) -> bool:
    """
    Check if the frame buffer contains enough frames with at least one hand.
    Prevents prediction on zero/garbage input.
    """
    hand_frames = 0
    for kps in buffer:
        left  = np.any(kps[33:54] != 0, axis=1).sum()
        right = np.any(kps[54:75] != 0, axis=1).sum()
        if left > 10 or right > 10:
            hand_frames += 1
    return hand_frames >= min_hand_frames


def filter_transition_frames(frames: list, var_threshold: float = 0.0001) -> list:
    """
    Transition Frame Filtering: reject frames where MediaPipe landmarks
    flicker (near-zero spatial variance = frozen/missing detection).
    Replaces bad frames with nearest valid neighbour.
    """
    if len(frames) < 3:
        return frames
    filtered = list(frames)
    for i in range(len(filtered)):
        kps = filtered[i]
        nz = kps[np.any(kps != 0, axis=1)]
        if len(nz) < 5 or nz.var() < var_threshold:
            # Replace with nearest valid frame (forward then backward)
            replaced = False
            for j in range(1, len(filtered)):
                for k in [i + j, i - j]:
                    if 0 <= k < len(filtered):
                        nz2 = filtered[k][np.any(filtered[k] != 0, axis=1)]
                        if len(nz2) >= 5 and nz2.var() >= var_threshold:
                            filtered[i] = filtered[k].copy()
                            replaced = True
                            break
                if replaced:
                    break
    return filtered


def resample_fps(frames: list,
                 source_fps: float,
                 target_fps: float = TRAIN_FPS) -> list:
    """
    Resample a list of keypoint frames from source_fps to target_fps.

    Training data was recorded at TRAIN_FPS (25).  If the webcam runs at
    a higher rate (e.g. 30 fps), the same number of frames covers LESS
    real-world time.  This function re-indexes the frames so that the
    output represents the same temporal duration at target_fps.

    Example: 60 frames @ 30 fps = 2.0 s → 50 frames @ 25 fps = 2.0 s
    """
    n = len(frames)
    if n == 0 or abs(source_fps - target_fps) < 0.5:
        return frames          # same fps or empty — no-op
    duration = n / source_fps  # seconds of real time
    target_n = max(1, int(round(duration * target_fps)))
    indices  = np.linspace(0, n - 1, target_n).astype(int)
    return [frames[i] for i in indices]


def preprocess_frames(frames: list) -> torch.Tensor:
    """
    Convert list of (75, 2) arrays → model input tensor (1, 50, 150).
    Applies root-relative normalization (same as training pre-compute):
      1. Root = mid-point of shoulders (landmarks 11, 12)
      2. Scale = shoulder width
      3. Subtract root, divide by scale for each non-zero landmark
      4. Clip to [ROOTREL_CLIP_MIN, ROOTREL_CLIP_MAX]
    """
    n = len(frames)
    if n == 0:
        return torch.zeros(1, N_FRAMES, INPUT_SIZE)

    arr = np.stack(frames, axis=0).astype(np.float32)   # (n, 75, 2)

    # Pad or center-crop to N_FRAMES
    if n >= N_FRAMES:
        start = (n - N_FRAMES) // 2
        arr = arr[start:start + N_FRAMES]
    else:
        pad = np.tile(arr[-1:], (N_FRAMES - n, 1, 1))
        arr = np.concatenate([arr, pad], axis=0)

    # ── Root-relative normalization ──
    T = arr.shape[0]
    l_sh = arr[:, ROOTREL_IDX_LEFT_SHOULDER, :]     # (T, 2)
    r_sh = arr[:, ROOTREL_IDX_RIGHT_SHOULDER, :]    # (T, 2)

    l_present = np.any(l_sh != 0, axis=1)
    r_present = np.any(r_sh != 0, axis=1)
    both = l_present & r_present

    roots  = np.zeros((T, 2), dtype=np.float32)
    scales = np.full(T, ROOTREL_FALLBACK_SHOULDER_W, dtype=np.float32)

    if both.any():
        vi = np.where(both)[0]
        roots[vi] = (l_sh[vi] + r_sh[vi]) / 2.0
        scales[vi] = np.linalg.norm(l_sh[vi] - r_sh[vi], axis=1)

        # Forward/backward fill for missing frames
        if not both.all():
            last = -1
            for t in range(T):
                if both[t]:
                    last = t
                elif last >= 0:
                    roots[t] = roots[last]
                    scales[t] = scales[last]
            first = vi[0]
            if first > 0:
                for t in range(first):
                    roots[t] = roots[first]
                    scales[t] = scales[first]

    scales = np.maximum(scales, ROOTREL_MIN_SHOULDER_W)

    # Normalize each frame
    out = np.zeros_like(arr)
    for t in range(T):
        nonzero = np.any(arr[t] != 0, axis=1)
        if nonzero.any():
            out[t, nonzero] = (arr[t, nonzero] - roots[t]) / scales[t]

    # Clip extreme outliers
    out = np.clip(out, ROOTREL_CLIP_MIN, ROOTREL_CLIP_MAX)

    out = out.reshape(N_FRAMES, INPUT_SIZE)
    return torch.from_numpy(out).unsqueeze(0).float()


# ─── Temporal smoother ────────────────────────────────────────

class TemporalSmoother:
    """
    Keeps sliding window of recent predictions and returns
    the most confident stable prediction.
    """

    def __init__(self, window: int = 5, min_confidence: float = 0.55):
        self.window = window
        self.min_confidence = min_confidence
        self.history = collections.deque(maxlen=window)

    def update(self, label: str, confidence: float):
        self.history.append((label, confidence))

    def get_smoothed(self) -> tuple:
        """Returns (label, confidence) or ("", 0.0) if not stable."""
        if not self.history:
            return ("", 0.0)

        # Count votes
        votes = {}
        for label, conf in self.history:
            if label not in votes:
                votes[label] = []
            votes[label].append(conf)

        # Find most frequent with highest avg confidence
        best_label = ""
        best_score = 0.0
        for label, confs in votes.items():
            score = len(confs) / self.window * np.mean(confs)
            if score > best_score:
                best_score = score
                best_label = label

        if best_score < self.min_confidence:
            return ("", 0.0)

        avg_conf = np.mean(votes[best_label])
        return (best_label, float(avg_conf))

    def clear(self):
        self.history.clear()


# ─── Drawing helpers ──────────────────────────────────────────

def draw_prediction_bar(frame, probs: np.ndarray, top_k: int = 5):
    """Draw top-K prediction bars on right side of frame."""
    h, w = frame.shape[:2]
    bar_w = min(220, w // 3)
    top_indices = np.argsort(probs)[::-1][:top_k]

    y_start = 30
    for rank, idx in enumerate(top_indices):
        cls   = CLASSES[idx]
        prob  = probs[idx]
        y     = y_start + rank * 36

        # Background
        cv2.rectangle(frame, (w - bar_w - 10, y - 18),
                      (w - 10, y + 10), (30, 30, 30), -1)

        # Bar fill (proportional)
        bar_fill = int((bar_w - 80) * prob)
        color = (0, 220, 0) if rank == 0 else (100, 180, 100)
        cv2.rectangle(frame, (w - bar_w + 70, y - 8),
                      (w - bar_w + 70 + bar_fill, y + 2), color, -1)

        # Text
        cv2.putText(frame, f"{cls:>12}", (w - bar_w - 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"{prob*100:4.1f}%", (w - 60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)


def draw_skeleton_overlay(frame, results):
    """Draw MediaPipe skeleton overlay with custom colors."""
    if mp_draw is None or mp_holistic is None:
        return   # mediapipe unavailable
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2),
        )
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(200, 80, 0),  thickness=2),
        )
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 100, 255), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 80, 200),  thickness=2),
        )


def draw_spotter_state(frame, spotter: SignSpotter):
    """Draw sign spotting state indicator."""
    h, w = frame.shape[:2]
    debug = spotter.get_debug_info()

    state = debug["state"]
    vel   = debug["smoothed_vel"]
    slen  = debug["sign_len"]

    # State indicator — top-right corner
    state_colors = {
        "IDLE":     (128, 128, 128),  # gray
        "SIGNING":  (0, 0, 255),      # red (attention!)
        "COOLDOWN": (0, 180, 255),    # orange
    }
    color = state_colors.get(state, (128, 128, 128))

    # State badge
    cv2.rectangle(frame, (w - 160, 0), (w, 24), color, -1)
    cv2.putText(frame, f"{state}", (w - 155, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Velocity meter (vertical bar on left edge)
    meter_h = min(h - 80, 200)
    meter_x = 5
    meter_y = 60
    cv2.rectangle(frame, (meter_x, meter_y),
                  (meter_x + 12, meter_y + meter_h), (40, 40, 40), -1)

    # Fill based on velocity (capped at 0.05)
    fill_ratio = min(1.0, vel / 0.05)
    fill_h = int(meter_h * fill_ratio)
    if fill_h > 0:
        bar_color = (0, 255, 0) if state != "SIGNING" else (0, 0, 255)
        cv2.rectangle(frame, (meter_x, meter_y + meter_h - fill_h),
                      (meter_x + 12, meter_y + meter_h), bar_color, -1)

    # Threshold line
    thresh_y = meter_y + meter_h - int(meter_h * min(1.0, spotter.velocity_threshold / 0.05))
    cv2.line(frame, (meter_x, thresh_y), (meter_x + 14, thresh_y), (0, 255, 255), 2)

    # Sign length indicator (when signing)
    if state == "SIGNING" and slen > 0:
        pct = min(1.0, slen / spotter.max_sign_frames)
        bar_len = int(100 * pct)
        cv2.rectangle(frame, (25, meter_y), (25 + bar_len, meter_y + 10), (0, 0, 255), -1)
        cv2.putText(frame, f"{slen}f", (25 + bar_len + 4, meter_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def draw_sentence_log(frame, sentence: list, max_display: int = 8):
    """Draw recognized sentence at bottom of frame."""
    h, w = frame.shape[:2]

    if not sentence:
        return

    text = " ".join(sentence[-max_display:])
    if len(sentence) > max_display:
        text = "... " + text

    # Background bar
    cv2.rectangle(frame, (0, h - 60), (w, h - 35), (20, 20, 20), -1)
    cv2.putText(frame, text, (10, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)


def draw_hud(frame, prediction: str, confidence: float,
             fps: float, mode: str, n_models: int,
             is_spotting_result: bool = False,
             confidence_threshold: float = 0.55):
    """Draw heads-up display info on frame."""
    h, w = frame.shape[:2]

    # Top-left: current prediction (large)
    if prediction and confidence > 0.3:
        bg_color = (0, 80, 0)
        if is_spotting_result:
            bg_color = (0, 0, 120)   # darker red = spotted sign
        cv2.rectangle(frame, (0, 0), (380, 55), bg_color, -1)
        label_text = f"{prediction.upper()}  {confidence*100:.0f}%"
        cv2.putText(frame, label_text,
                    (10, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 0), 2)

    # Bottom-left: info bar
    cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)
    info = f"Mode:{mode}  FPS:{fps:4.1f}  Models:{n_models}  MinConf:{confidence_threshold:.0%}"
    cv2.putText(frame, info, (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Controls hint (top-center)
    cv2.putText(frame, "Q:quit R:reset M:mode S:save SPACE:pause",
                (150, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)


# ─── Sign spotter visualizer (Fig. 1-style heatmap panel) ────

class SpotterVisualizer:
    """
    Renders a Fig.1-style scrolling velocity heatmap below the camera feed.

    4 rows: body / face / left hand / right hand.
    Each column = 1 frame.
    Pixel colour ∝ velocity magnitude (viridis-like colormap: dark purple → bright yellow).
    Yellow overlay on columns where the spotter is in SIGNING state.
    Row labels on the right edge (matching Fig. 1).
    """

    # Viridis colormap keypoints in BGR (5 stops: 0.0 → 0.25 → 0.5 → 0.75 → 1.0)
    _CMAP = np.array([
        [ 84,   1,  68],   # 0.00 — dark purple
        [139,  82,  59],   # 0.25 — deep blue
        [140, 145,  33],   # 0.50 — teal
        [ 98, 201,  94],   # 0.75 — green-yellow
        [ 37, 231, 253],   # 1.00 — bright yellow
    ], dtype=np.float32)

    def __init__(
        self,
        history_frames: int = 300,   # ~10 s at 30 fps
        row_height:     int = 28,
        row_gap:        int = 3,
        label_w:        int = 48,
    ):
        self.history_frames = history_frames
        self.row_height = row_height
        self.row_gap    = row_gap
        self.label_w    = label_w

        self._rows   = ['body', 'face', 'lh', 'rh']
        self._labels = ['body', 'face', 'left', 'right']

        # Adaptive per-channel normalization (EMA of peak)
        self._max_vel: dict = {r: 0.015 for r in self._rows}
        self._ema_alpha = 0.97

        # Ring buffer: one entry per frame
        self._buf: collections.deque = collections.deque(maxlen=history_frames)

    @property
    def panel_height(self) -> int:
        return self.row_gap + len(self._rows) * (self.row_height + self.row_gap)

    def update(self, channel_vels: dict, is_signing: bool):
        """
        Call once per camera frame.
        channel_vels — dict with keys 'body', 'face', 'lh', 'rh'
        is_signing   — True when spotter state == SIGNING
        """
        self._buf.append({
            'body': channel_vels.get('body', 0.0),
            'face': channel_vels.get('face', 0.0),
            'lh':   channel_vels.get('lh',   0.0),
            'rh':   channel_vels.get('rh',   0.0),
            'signing': is_signing,
        })
        # Adapt per-channel normalizer
        for r in self._rows:
            v = channel_vels.get(r, 0.0)
            if v > self._max_vel[r]:
                self._max_vel[r] = v          # instant tracking on rising edge
            else:
                self._max_vel[r] = (
                    self._ema_alpha * self._max_vel[r]
                    + (1 - self._ema_alpha) * v
                )
            self._max_vel[r] = max(self._max_vel[r], 0.004)  # floor

    @classmethod
    def _viridis(cls, t: np.ndarray) -> np.ndarray:
        """Vectorised viridis-like interpolation. t∈[0,1] → (N,3) uint8 BGR."""
        t = np.clip(t, 0.0, 1.0)
        pos = t * 4.0                              # position within 5 stops
        lo  = np.floor(pos).astype(int)
        hi  = np.minimum(lo + 1, 4)
        frac= (pos - lo)[:, None]
        return (cls._CMAP[lo] + frac * (cls._CMAP[hi] - cls._CMAP[lo])).astype(np.uint8)

    def render(self, width: int) -> np.ndarray:
        """
        Render and return BGR numpy array of shape (panel_height, width, 3).
        Call this once per frame and vstack with the camera image.
        """
        panel = np.full((self.panel_height, width, 3), 15, dtype=np.uint8)

        history = list(self._buf)
        n = len(history)
        if n == 0:
            return panel

        plot_w = width - self.label_w
        # Pre-compute x pixel bounds for every frame column
        xs      = (np.arange(n)     * plot_w / n).astype(int) + self.label_w
        xs_next = (np.arange(1,n+1) * plot_w / n).astype(int) + self.label_w
        xs_next = np.maximum(xs + 1, xs_next)                  # at least 1px wide

        signing_mask = np.array([f['signing'] for f in history], dtype=bool)

        for ri, (row_key, label) in enumerate(zip(self._rows, self._labels)):
            y0 = self.row_gap + ri * (self.row_height + self.row_gap)
            y1 = y0 + self.row_height
            y_mid = (y0 + y1) // 2

            max_vel = self._max_vel[row_key]

            # ── Velocity heatmap (vectorised) ──────────────────────────
            vals = np.array([f[row_key] for f in history], dtype=np.float32)
            t_all = np.clip(vals / max_vel, 0.0, 1.0)
            colors = self._viridis(t_all)              # (n, 3)

            for fi in range(n):
                x0c = xs[fi];  x1c = min(xs_next[fi], width)
                panel[y0:y1, x0c:x1c] = colors[fi]

            # ── Velocity threshold hairline (yellow dash, lh/rh only) ──
            if row_key in ('lh', 'rh'):
                thresh_y = y1 - 1 - int(
                    (self.row_height - 2) * min(1.0, SPOT_VELOCITY_THRESHOLD / max_vel)
                )
                if y0 <= thresh_y < y1:
                    panel[thresh_y, self.label_w:width:4] = (0, 230, 230)   # cyan dashes

            # ── Yellow signing span overlay ─────────────────────────────
            if np.any(signing_mask):
                for fi in np.where(signing_mask)[0]:
                    x0c = xs[fi];  x1c = min(xs_next[fi], width)
                    region = panel[y0:y1, x0c:x1c].astype(np.float32)
                    panel[y0:y1, x0c:x1c] = (
                        region * 0.30 + np.array([0, 215, 255], np.float32) * 0.70
                    ).astype(np.uint8)

            # ── Row label (right edge, matching Fig. 1) ────────────────
            cv2.putText(
                panel, label,
                (width - self.label_w + 3, y_mid + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA,
            )
            # Thin separator line
            cv2.line(panel, (self.label_w - 1, y0),
                     (self.label_w - 1, y1), (40, 40, 40), 1)

        # ── Time axis annotation ────────────────────────────────────────
        cv2.putText(
            panel, f"← {n} frames",
            (self.label_w + 4, self.panel_height - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (70, 70, 70), 1,
        )

        return panel


# ─── Inference wrapper ────────────────────────────────────────

class Predictor:
    """Wraps single-model or ensemble inference with optional anti-bias filtering."""

    def __init__(self, use_ensemble: bool = False, model_name: str = "hybrid",
                 checkpoint: str | None = None, use_antibias: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ensemble = use_ensemble
        self.ckpt_num_classes = NUM_CLASSES
        self.antibias = None

        # Initialize anti-bias filter
        if use_antibias and _HAS_ANTIBIAS:
            try:
                self.antibias = AntibiaFilter.from_config()
                print(f"[Predictor] Anti-bias filter active")
            except Exception as e:
                print(f"[Predictor] Anti-bias filter unavailable: {e}")
                self.antibias = None

        if use_ensemble:
            self.ensemble = Ensemble.from_config()
            self.n_models = self.ensemble.num_models
        else:
            ckpt_path = checkpoint or os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
            print(f"[Predictor] Loading '{model_name}' from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            # Support both checkpoint key formats
            state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))
            mkw = ckpt.get("model_kwargs", {})

            # Detect checkpoint num_classes
            ckpt_nc = ckpt.get("num_classes", None)
            if ckpt_nc:
                self.ckpt_num_classes = ckpt_nc
            else:
                for key, val in state_dict.items():
                    if key.endswith(".weight") and val.ndim == 2 and val.shape[0] in (21, 31, 36, 56):
                        self.ckpt_num_classes = val.shape[0]
            if self.ckpt_num_classes != NUM_CLASSES:
                print(f"[Predictor] Checkpoint has {self.ckpt_num_classes} classes, "
                      f"config has {NUM_CLASSES}. Adapting...")

            self.model = get_model(model_name, num_classes=self.ckpt_num_classes, **mkw).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.n_models = 1

        print(f"[Predictor] Ready. Device={self.device}, Models={self.n_models}")

    @torch.no_grad()
    def predict(self, frames: list) -> tuple:
        """
        Predict from frame list.
        Returns (probs_array, label_str, confidence_float)
        probs_array always has NUM_CLASSES elements.
        """
        # Transition frame filtering: clean flickering MediaPipe detections
        frames = filter_transition_frames(frames)
        x = preprocess_frames(frames).to(self.device)

        if self.use_ensemble:
            probs = self.ensemble.predict_probs(x)
        else:
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # Pad if checkpoint had fewer classes
            if self.ckpt_num_classes < NUM_CLASSES:
                padded = np.zeros(NUM_CLASSES, dtype=probs.dtype)
                padded[:self.ckpt_num_classes] = probs
                probs = padded

        # Apply anti-bias filter if available
        # if self.antibias is not None:
        #     label, conf, was_filtered = self.antibias.apply(probs, frames)
        #     if label == "":
        #         # Rejected — return probs but with empty label
        #         top_idx = probs.argmax()
        #         return probs, "", conf
        #     return probs, label, conf

        # top_idx = probs.argmax()
        # return probs, IDX2CLASS[int(top_idx)], float(probs[top_idx])

        if self.antibias is not None:
            penalty = getattr(self.antibias, "penalty", None)
            if penalty is not None and len(penalty) == len(probs):
                label, conf, was_filtered = self.antibias.apply(probs, frames)
                if label == "":
                    return probs, "", conf
                return probs, label, conf
            else:
                print(
                    f"[Predictor] Anti-bias skipped: penalty len="
                    f"{len(penalty) if penalty is not None else 'None'} "
                    f"!= probs len={len(probs)}"
                )        
        top_idx = probs.argmax()
        return probs, IDX2CLASS[int(top_idx)], float(probs[top_idx])

# ─── Main demo loop ───────────────────────────────────────────

MODES = ["sliding", "spotting", "hybrid"]


def run_demo(
    model_name: str = "hybrid",
    checkpoint: str | None = None,
    cam_id: int = 0,
    mode: str = "spotting",
    use_ensemble: bool = True,
    show_skeleton: bool = True,
    show_bars: bool = True,
    confidence_threshold: float = 0.55,
    use_antibias: bool = True,
):
    # Load predictor
    predictor = Predictor(
        use_ensemble=use_ensemble,
        model_name=model_name,
        checkpoint=checkpoint,
        use_antibias=use_antibias,
    )

    if mp_holistic is None:
        print("[ERR] MediaPipe not available. "
              "Install with: pip install mediapipe")
        return

    # Sign spotter (fps_ratio set after webcam FPS detection below)
    spotter = SignSpotter()

    # Fig.1-style velocity heatmap panel
    visualizer = SpotterVisualizer()

    # Temporal smoother
    smoother = TemporalSmoother(window=5, min_confidence=confidence_threshold)

    # Camera
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[ERR] Cannot open camera {cam_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    # ── FPS normalization ──
    # Detect actual webcam FPS and expand buffer so it covers the same
    # real-time duration as training data (N_FRAMES / TRAIN_FPS = 2.0 s).
    webcam_fps = cap.get(cv2.CAP_PROP_FPS)
    if webcam_fps <= 0 or webcam_fps > 120:
        webcam_fps = 30.0          # safe default
    fps_ratio = TRAIN_FPS / webcam_fps  # e.g. 25/30 = 0.833
    webcam_buffer_size = max(N_FRAMES, int(round(N_FRAMES / fps_ratio)))  # 60 @ 30fps
    spotter.fps_ratio = fps_ratio  # enable FPS resampling in sign spotter
    print(f"[Demo] Webcam FPS: {webcam_fps:.1f} → buffer {webcam_buffer_size} frames "
          f"({webcam_buffer_size/webcam_fps:.2f}s ≈ {N_FRAMES/TRAIN_FPS:.1f}s training)")
    print(f"[Demo] FPS normalization: {webcam_fps:.0f}fps → {TRAIN_FPS}fps "
          f"(ratio={fps_ratio:.3f})")

    # State
    buffer       = collections.deque(maxlen=webcam_buffer_size)
    frame_count  = 0
    prediction   = ""
    confidence   = 0.0
    probs        = np.zeros(NUM_CLASSES)
    fps_t0       = time.time()
    fps_val      = 0.0
    sentence     = []          # accumulated recognized words (raw)
    sb           = SentenceBuilder(
        stability_count=2, cooldown_sec=1.0,
        confidence_threshold=confidence_threshold,
    )
    is_spotted   = False       # last prediction came from spotter
    paused       = False
    mode_idx     = MODES.index(mode) if mode in MODES else 1

    # Anti-repetition guard: consecutive same-class predictions need higher confidence
    _last_spotted_labels = collections.deque(maxlen=5)
    REPEAT_CONF_PENALTY  = 0.10   # +10% per consecutive repeat

    # Runtime diagnostics
    _diag_kps_x_min, _diag_kps_x_max = 999.0, -999.0
    _diag_kps_y_min, _diag_kps_y_max = 999.0, -999.0
    _diag_predictions = 0
    _diag_rejected    = 0

    print(f"[Demo] Mode: {MODES[mode_idx]} | Ensemble: {use_ensemble}")
    print(f"[Demo] Confidence threshold: {confidence_threshold:.2f}")
    print(f"[Demo] Anti-repetition: +{REPEAT_CONF_PENALTY:.0%} per consecutive same-class prediction")
    print(f"[Demo] MediaPipe processes ORIGINAL frame (no flip) → correct L/R hand assignment")
    print(f"[Demo] Display is mirrored for natural interaction")
    print(f"[Demo] Press Q/ESC=quit, R=reset, M=cycle mode, S=save, SPACE=pause")

    with mp_holistic.Holistic(
        min_detection_confidence=CAM_CONFIDENCE,
        min_tracking_confidence=CAM_CONFIDENCE,
        model_complexity=1,
    ) as holistic:

        while True:
            ret, frame_raw = cap.read()
            if not ret:
                break

            # CRITICAL: Process ORIGINAL (non-flipped) frame through MediaPipe!
            # cv2.flip before MediaPipe causes left/right hand landmark SWAP,
            # which destroys accuracy (67.9% → 8.5% in testing).
            # Training videos have the same camera-facing orientation as the
            # unflipped webcam frame → matching hand L/R assignment.

            if paused:
                display = cv2.flip(frame_raw, 1)   # mirror for display only
                cv2.putText(display, "PAUSED", (250, 240),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow("BSL Recognition — Phase 2", display)
                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key in (ord("q"), 27):
                    break
                continue

            frame_count += 1
            current_mode = MODES[mode_idx]

            # FPS calculation
            if frame_count % 30 == 0:
                fps_val = 30 / max(0.001, time.time() - fps_t0)
                fps_t0  = time.time()

            # MediaPipe inference on ORIGINAL (non-flipped) frame
            rgb = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # Extract keypoints from non-flipped frame (correct L/R hand assignment)
            kps = extract_keypoints(results)
            buffer.append(kps)

            # ── Runtime diagnostics (first 300 frames = ~10s) ──
            if frame_count <= 300:
                nz = kps[np.any(kps != 0, axis=1)]
                if len(nz) > 0:
                    _diag_kps_x_min = min(_diag_kps_x_min, nz[:, 0].min())
                    _diag_kps_x_max = max(_diag_kps_x_max, nz[:, 0].max())
                    _diag_kps_y_min = min(_diag_kps_y_min, nz[:, 1].min())
                    _diag_kps_y_max = max(_diag_kps_y_max, nz[:, 1].max())
                if frame_count == 300:
                    print(f"[Diag] Keypoint ranges after 300 frames:")
                    print(f"  X: [{_diag_kps_x_min:.4f}, {_diag_kps_x_max:.4f}]")
                    print(f"  Y: [{_diag_kps_y_min:.4f}, {_diag_kps_y_max:.4f}]")
                    if _diag_kps_y_max < 1.01:
                        print(f"  NOTE: Y max < 1.01 (webcam upper body only).")
                        print(f"        This is expected. Training Y goes up to ~2.3 (full body).")

            # ── SIGN SPOTTING (modes: spotting, hybrid) ──
            # Always feed spotter (for visualizer channel velocities);
            # only *use* spotted results in the relevant modes.
            spotted_result_raw = spotter.feed(kps)
            debug_info         = spotter.get_debug_info()
            visualizer.update(debug_info["channel_vels"],
                              debug_info["state"] == "SIGNING")

            # ── Detection quality for current frame ──
            det_q = detection_quality(kps)
            _hands_status = f"{'L' if det_q['has_left'] else '-'}{'R' if det_q['has_right'] else '-'} ({det_q['hand_count']})"

            spotted_result = None
            if current_mode in ("spotting", "hybrid"):
                spotted_result = spotted_result_raw

                if spotted_result is not None:
                    frames, meta = spotted_result
                    _diag_predictions += 1
                    # Guard: only predict if frames contain hand data
                    if buffer_has_hands(frames, min_hand_frames=5):
                        _t0 = time.perf_counter()
                        probs, label, conf = predictor.predict(frames)
                        _lat_ms = (time.perf_counter() - _t0) * 1000
                        _top3i = np.argsort(probs)[::-1][:3]
                        _top3s = ", ".join(f"{IDX2CLASS[i]}: {probs[i]*100:.1f}%" for i in _top3i)
                        _diag = f"[DIAG] FPS:{fps_val:5.1f} | Lat:{_lat_ms:6.1f}ms | Hands:{_hands_status} | Top3: [{_top3s}]"
                        print(_diag)
                        _diag_logger.info(_diag)

                        # Anti-repetition: require higher confidence for
                        # consecutive predictions of the same class
                        repeat_count = sum(1 for prev in _last_spotted_labels if prev == label)
                        effective_threshold = confidence_threshold + repeat_count * REPEAT_CONF_PENALTY

                        if conf >= effective_threshold:
                            prediction = label
                            confidence = conf
                            is_spotted = True
                            emitted = sb.feed(label, conf)
                            if emitted:
                                sentence.append(label)
                            _last_spotted_labels.append(label)
                            extra = f" (repeat#{repeat_count+1})" if repeat_count > 0 else ""
                            print(f"  [SPOT] #{meta['sign_index']}: "
                                  f"{label.upper()} ({conf*100:.1f}%) "
                                  f"[{meta['raw_length']}f]{extra}")
                        else:
                            # Reject uncertain or repeated prediction
                            _diag_rejected += 1
                            reason = "UNCERTAIN"
                            if repeat_count > 0 and conf >= confidence_threshold:
                                reason = f"REPEAT×{repeat_count} (need {effective_threshold*100:.0f}%)"
                            print(f"  [SPOT] #{meta['sign_index']}: "
                                  f"??? {reason} ({label} {conf*100:.1f}% < {effective_threshold*100:.0f}%) "
                                  f"[{meta['raw_length']}f] — REJECTED")
                    else:
                        _diag_rejected += 1
                        _diag = f"[DIAG] FPS:{fps_val:5.1f} | Lat:  N/A  | Hands:{_hands_status} | Top3: [NO HANDS — skipped]"
                        print(_diag)
                        _diag_logger.info(_diag)
                        print(f"  [SPOT] #{meta['sign_index']}: "
                              f"--- NO HANDS [{meta['raw_length']}f] — SKIPPED")

            # ── SLIDING WINDOW (modes: sliding, hybrid) ──
            if current_mode in ("sliding", "hybrid"):
                if len(buffer) == webcam_buffer_size and frame_count % PREDICT_EVERY_N == 0:
                    # Only do sliding prediction if not currently mid-sign (hybrid mode)
                    if current_mode == "sliding" or not spotter.is_signing:
                        # Guard: skip prediction if buffer lacks hand data
                        if not buffer_has_hands(buffer, min_hand_frames=10):
                            _diag = f"[DIAG] FPS:{fps_val:5.1f} | Lat:  N/A  | Hands:{_hands_status} | Top3: [NO HANDS — skipped]"
                            print(_diag)
                            _diag_logger.info(_diag)
                        else:
                            # FPS normalization: resample buffer to TRAIN_FPS
                            buf_frames = resample_fps(list(buffer), webcam_fps, TRAIN_FPS)
                            _t0 = time.perf_counter()
                            probs_s, label_s, conf_s = predictor.predict(buf_frames)
                            _lat_ms = (time.perf_counter() - _t0) * 1000
                            _top3i = np.argsort(probs_s)[::-1][:3]
                            _top3s = ", ".join(f"{IDX2CLASS[i]}: {probs_s[i]*100:.1f}%" for i in _top3i)
                            _diag = f"[DIAG] FPS:{fps_val:5.1f} | Lat:{_lat_ms:6.1f}ms | Hands:{_hands_status} | Top3: [{_top3s}]"
                            print(_diag)
                            _diag_logger.info(_diag)
                            smoother.update(label_s, conf_s)
                            sm_label, sm_conf = smoother.get_smoothed()

                            if sm_conf >= confidence_threshold:
                                # In hybrid mode, only update if spotter hasn't produced
                                # a recent result
                                if current_mode == "sliding" or not is_spotted:
                                    probs      = probs_s
                                    prediction = sm_label
                                    confidence = sm_conf
                                    is_spotted = False

            # Decay spotted flag after some frames
            if is_spotted and frame_count % 60 == 0:
                is_spotted = False

            # ── DRAW ──
            # Draw skeleton overlay on ORIGINAL (non-flipped) frame,
            # then flip for mirror display. This preserves correct
            # skeleton positions when the entire image is flipped.
            draw_frame = frame_raw.copy()

            if show_skeleton:
                draw_skeleton_overlay(draw_frame, results)

            if current_mode in ("spotting", "hybrid"):
                draw_spotter_state(draw_frame, spotter)

            if show_bars and prediction:
                draw_prediction_bar(draw_frame, probs)

            # Detection quality indicator (top-right area)
            h_f, w_f = draw_frame.shape[:2]
            hand_str = f"Hands:{det_q['hand_count']}"
            hand_color = (0, 255, 0) if det_q['hand_count'] > 0 else (0, 0, 255)
            cv2.putText(draw_frame, hand_str, (w_f - 100, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, hand_color, 1)

            draw_sentence_log(draw_frame, sb.sentence)
            draw_hud(draw_frame, prediction, confidence,
                     fps_val, current_mode, predictor.n_models,
                     is_spotting_result=is_spotted,
                     confidence_threshold=confidence_threshold)

            # Flip for mirror display (natural interaction)
            display = cv2.flip(draw_frame, 1)

            # ── Velocity heatmap panel (Fig.1-style) ──────────────────
            viz_panel = visualizer.render(display.shape[1])
            display   = np.vstack([display, viz_panel])

            cv2.imshow("BSL Recognition — Phase 2", display)

            # ── KEYBOARD ──
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                buffer.clear()
                spotter.reset()
                smoother.clear()
                sentence.clear()
                sb.clear()
                visualizer._buf.clear()
                prediction = ""
                confidence = 0.0
                probs      = np.zeros(NUM_CLASSES)
                is_spotted = False
                print("[Demo] Full reset")
            elif key == ord("m"):
                mode_idx = (mode_idx + 1) % len(MODES)
                spotter.reset()
                smoother.clear()
                print(f"[Demo] Mode → {MODES[mode_idx]}")
            elif key == ord("s"):
                ts = int(time.time())
                cv2.imwrite(f"frame_{ts}.png", display)
                if len(buffer) > 0:
                    np.save(f"skeleton_{ts}.npy", np.stack(list(buffer), axis=0))
                print(f"[Demo] Saved frame_{ts}.png")
            elif key == ord(" "):
                paused = True

    cap.release()
    cv2.destroyAllWindows()

    if sb.sentence:
        print(f"\n[Demo] Recognized sentence: {sb.text}")
        print(f"[Demo] SentenceBuilder stats: {sb.stats()}")
    
    # Runtime diagnostics summary
    print(f"\n[Demo] Session summary:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Predictions attempted: {_diag_predictions}")
    print(f"  Predictions rejected (low confidence/no hands): {_diag_rejected}")
    print(f"  Predictions accepted: {_diag_predictions - _diag_rejected}")
    print(f"  Keypoint X range: [{_diag_kps_x_min:.4f}, {_diag_kps_x_max:.4f}]")
    print(f"  Keypoint Y range: [{_diag_kps_y_min:.4f}, {_diag_kps_y_max:.4f}]")
    if _diag_kps_y_max < 1.01:
        print(f"  WARNING: Y-range limited to [0,1] — lower body not detected/below frame")
    print(f"  Confidence threshold: {confidence_threshold:.2f}")
    print(f"  Diagnostic log written to: camera_diagnostics.log")
    _diag_logger.info(f"Session ended. Frames: {frame_count}, Predictions: {_diag_predictions}, Rejected: {_diag_rejected}")
    print("[Demo] Done.")


def detect_available_cameras(max_index: int = 10) -> list:
    """Detect available camera indices."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def main():

    p = argparse.ArgumentParser(description="BSL Real-time Camera Demo (Phase 2)")
    p.add_argument("--model",      default="hybrid",
                   choices=["bilstm","transformer","stgcn","tcn","hybrid"])
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt checkpoint (default: checkpoints/{model}_best.pt)")
    
    # Detect available cameras
    available_cameras = detect_available_cameras()
    p.add_argument("--camera",     type=int, default=0,
                   choices=available_cameras if available_cameras else [0],
                   help=f"Camera index. Available: {available_cameras}")
    p.add_argument("--mode",       default="spotting",
                   choices=["sliding", "spotting", "hybrid"],
                   help="Detection mode: sliding (Phase 1), spotting (Phase 2), hybrid (both)")
    p.add_argument("--ensemble",   action="store_true",
                   help="Use ensemble of all trained models (from config.py)")
    p.add_argument("--confidence", type=float, default=0.55,
                   help="Minimum confidence threshold for predictions (default: 0.55)")
    p.add_argument("--no-skeleton",action="store_true")
    p.add_argument("--no-bars",    action="store_true")
    p.add_argument("--no-antibias",action="store_true",
                   help="Disable real-time anti-bias filter")
    p.add_argument("--fast",       action="store_true",
                   help="Use fast 2-model ensemble (TCN+BiLSTM, ~2.8x faster)")
    p.add_argument("--manual",     action="store_true",
                   help="Use manual-annotations-only model (21 classes)")
    p.add_argument("--wlasl",      action="store_true",
                   help="Use WLASL-100 ASL model (100 classes)")
    args = p.parse_args()
    
    # Interactive menu if no args provided
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("BSL Real-time Camera Demo (Phase 2) — Configuration Menu")
        print("="*60)
        
        # Model selection
        models = ["bilstm", "transformer", "stgcn", "tcn", "hybrid"]
        print("\n[1] Select Model:")
        for i, m in enumerate(models, 1):
            print(f"    {i}. {m}")
        model_choice = input("Enter choice (1-5, default=5 hybrid): ").strip() or "5"
        model_idx = min(max(int(model_choice) - 1, 0), len(models) - 1)
        args.model = models[model_idx]
        
        # Mode selection
        modes = ["sliding", "spotting", "hybrid"]
        print("\n[2] Select Detection Mode:")
        for i, m in enumerate(modes, 1):
            print(f"    {i}. {m} ({['Phase 1: continuous', 'Phase 2: sign spotting', 'Both modes'][i-1]})")
        mode_choice = input("Enter choice (1-3, default=2 spotting): ").strip() or "2"
        mode_idx = min(max(int(mode_choice) - 1, 0), len(modes) - 1)
        args.mode = modes[mode_idx]
        
        # Camera selection
        print(f"\n[3] Select Camera:")
        if available_cameras:
            for i, cam_id in enumerate(available_cameras, 1):
                print(f"    {i}. Camera {cam_id}")
            cam_choice = input(f"Enter choice (1-{len(available_cameras)}, default=1): ").strip() or "1"
            cam_idx = min(max(int(cam_choice) - 1, 0), len(available_cameras) - 1)
            args.camera = available_cameras[cam_idx]
        else:
            print("    No cameras detected, using default (0)")
            args.camera = 0
        
        # Ensemble selection
        print(f"\n[4] Use Ensemble?")
        print(f"    1. Yes (multi-model voting)")
        print(f"    2. No (single model: {args.model})")
        ens_choice = input("Enter choice (1-2, default=2): ").strip() or "2"
        args.ensemble = (ens_choice.strip() == "1")
        
        # # Manual selection
        # print(f"\n[5] Use Manual-Annotations-Only Model (21 classes)?")
        # print(f"    1. Yes (only signs with manual annotations)")
        # print(f"    2. No (all signs, including those with auto annotations)")
        # manual_choice = input("Enter choice (1-2, default=2): ").strip() or "2"
        # args.manual = (manual_choice.strip() == "1")

        # Confidence threshold
        print(f"\n[6] Confidence Threshold (recommended: 0.55):")
        conf_str = input("Enter threshold 0.35-1.0 (default=0.55): ").strip()
        args.confidence = float(conf_str) if conf_str else 0.55
        args.confidence = max(0.35, min(1.0, args.confidence))
        if args.confidence < 0.45:
            print(f"    WARNING: Low threshold ({args.confidence:.2f}) may produce noisy predictions.")
        
        # Display options
        print(f"\n[7] Display Options:")
        skel_str = input("Show skeleton overlay? (Y/n, default=Y): ").strip().lower()
        args.no_skeleton = (skel_str == "n")
        bars_str = input("Show probability bars? (Y/n, default=Y): ").strip().lower()
        args.no_bars = (bars_str == "n")
        
        # Wasl selection
        print(f"\n[8] Use WLASL-100 ASL Model (100 classes)?")
        print(f"    1. Yes (model trained on WLASL-100)")
        print(f"    2. No (BSL model with {NUM_CLASSES} classes)")
        wlasl_choice = input("Enter choice (1-2, default=2): ").strip() or "2"
        args.wlasl = (wlasl_choice.strip() == "1")



        # Anti-bias selection
        print(f"\n[10] Use Real-time Anti-Bias Filter?")
        print(f"    1. Yes (filters implausible predictions based on training data)")
        print(f"    2. No (disable anti-bias)")
        antibias_choice = input("Enter choice (1-2, default=1): ").strip() or "1"
        args.no_antibias = (antibias_choice.strip() == "2")

    

        

        # Summary
        print("\n" + "="*60)
        print("Configuration Summary:")
        print(f"  Model:        {args.model}")
        print(f"  Mode:         {args.mode}")
        print(f"  Camera:       {args.camera}")
        print(f"  Ensemble:     {args.ensemble}")
        print(f"  Confidence:   {args.confidence:.2f}")
        print(f"  Skeleton:     {not args.no_skeleton}")
        print(f"  Bars:         {not args.no_bars}")
        print("="*60 + "\n")

    # Validate checkpoint exists for single-model mode
    if not args.ensemble and not getattr(args, 'fast', False):
        ckpt = args.checkpoint or os.path.join(CHECKPOINT_DIR, f"{args.model}_best.pt")
        if not os.path.exists(ckpt):
            print(f"[ERR] Checkpoint not found: {ckpt}")
            print(f"      Train first: python train.py --model {args.model}")
            sys.exit(1)
        args.checkpoint = ckpt

    # Dynamic Thresholding: scale confidence for high class counts
    # 100 classes → lower default threshold since chance is 1% vs 5% (21 classes)
    if args.confidence == 0.55 and NUM_CLASSES >= 50:
        args.confidence = max(0.25, 0.55 * (21.0 / NUM_CLASSES) ** 0.3)
        print(f"[Demo] Dynamic threshold: {args.confidence:.2f} "
              f"(auto-scaled for {NUM_CLASSES} classes)")

    # WLASL optimized ensemble weights: inject from result JSON if available
    if getattr(args, 'wlasl', False) and args.ensemble:
        import json as _json
        _wlasl_ens_path = os.path.join(os.path.dirname(__file__),
                                       "runs_wlasl", "ensemble_wlasl_optimized.json")
        if os.path.exists(_wlasl_ens_path):
            import config as _cfg
            _wlasl_ens = _json.load(open(_wlasl_ens_path))
            _cfg.ENSEMBLE_WEIGHTS = _wlasl_ens["optimized_weights_list"]
            print(f"[Demo] WLASL optimized weights: {_cfg.ENSEMBLE_WEIGHTS}")

    # Fast 2-model ensemble: override config temporarily
    if getattr(args, 'fast', False):
        args.ensemble = True
        import json as _json
        _fast_cfg_path = os.path.join(os.path.dirname(__file__), "runs_manual", "fast_ensemble_result.json")
        if os.path.exists(_fast_cfg_path):
            _fast = _json.load(open(_fast_cfg_path))
            from config import ENSEMBLE_MODELS as _EM, ENSEMBLE_WEIGHTS as _EW
            # Monkey-patch config for fast ensemble
            import config as _cfg
            _cfg.ENSEMBLE_MODELS = _fast["models"]
            _cfg.ENSEMBLE_WEIGHTS = _fast["weights"]
            print(f"[Demo] FAST mode: {_fast['models']} weights={_fast['weights']}")
        else:
            print(f"[Demo] FAST mode requested but no config found — using default ensemble")

    run_demo(
        model_name=args.model,
        checkpoint=args.checkpoint,
        cam_id=args.camera,
        mode=args.mode,
        use_ensemble=args.ensemble,
        show_skeleton=not args.no_skeleton,
        show_bars=not args.no_bars,
        confidence_threshold=args.confidence,
        use_antibias=not getattr(args, 'no_antibias', False),
    )


if __name__ == "__main__":
    main()

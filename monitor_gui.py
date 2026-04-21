# =============================================================
# monitor_gui.py — Real-time BSL Sign Language Monitor GUI
# =============================================================
"""
Real-time monitoring GUI with:
  - Live camera feed + MediaPipe skeleton overlay
  - Scrolling velocity heatmap (body / face / left hand / right hand)
    inspired by optical-flow norm representation from sign language research
  - Yellow markers for detected sign spans
  - Sign spotter state machine visualization
  - Classification results + sentence log

Controls:
    Q / ESC — quit
    SPACE   — pause/unpause
    R       — reset sign spotter + sentence log
    M       — cycle mode (sliding → spotting → hybrid)
    S       — toggle skeleton overlay
    H       — toggle heatmap panel
    +/-     — adjust velocity threshold

Usage:
    python monitor_gui.py                    # default hybrid mode
    python monitor_gui.py --mode spotting    # sign spotting mode
    python monitor_gui.py --model hybrid     # specific model
    python monitor_gui.py --ensemble         # use ensemble if available
    python monitor_gui.py --camera 0         # camera index
    python monitor_gui.py --heatmap-height 200  # heatmap panel height
"""

import cv2
import numpy as np
import torch
import time
import argparse
import os
import sys
from pathlib import Path
from collections import deque

# Suppress TF/mediapipe warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[WARN] mediapipe not installed. Camera skeleton overlay disabled.")

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CLASSES, IDX2CLASS, NUM_CLASSES, INPUT_SIZE,
    N_FRAMES, N_LANDMARKS, CHECKPOINT_DIR,
    CAM_BUFFER_SIZE, CAM_CONFIDENCE, PREDICT_EVERY_N,
    SPOT_VELOCITY_THRESHOLD, SPOT_IDLE_THRESHOLD,
)
from sign_spotter import SignSpotter, SpotterState
from models import get_model

# Try ensemble
try:
    from ensemble import Ensemble
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False


# ═══════════════════════════════════════════════════════════════
#  HEATMAP VISUALIZATION (Optical-flow norm style)
# ═══════════════════════════════════════════════════════════════

class VelocityHeatmap:
    """
    Scrolling heatmap showing per-channel velocity over time.

    Channels (rows):
      body  — upper body (shoulders, elbows, hips)
      face  — face/head landmarks
      lh    — left hand
      rh    — right hand

    Yellow markers overlay when a sign is detected.
    Inspired by: "Optical-flow norm representation of BSL conversation"
    """

    def __init__(self, width=640, channel_height=30, history_frames=300):
        """
        Args:
            width: pixel width of heatmap panel
            channel_height: pixel height per channel row
            history_frames: number of frames to keep in history (x-axis)
        """
        self.width = width
        self.ch_h = channel_height
        self.n_channels = 4
        self.history_frames = history_frames

        # Channel labels
        self.channel_names = ["body", "face", "left", "right"]
        self.channel_colors = [
            (255, 200, 100),   # body: light blue
            (200, 100, 255),   # face: purple
            (100, 255, 100),   # left hand: green
            (100, 200, 255),   # right hand: orange
        ]

        # Velocity buffers per channel (circular)
        self.vel_buffers = {
            "body": deque(maxlen=history_frames),
            "face": deque(maxlen=history_frames),
            "lh":   deque(maxlen=history_frames),
            "rh":   deque(maxlen=history_frames),
        }
        self.vel_keys = ["body", "face", "lh", "rh"]

        # Sign span markers: list of (start_frame_idx, end_frame_idx, label)
        self.sign_spans = deque(maxlen=200)
        self.frame_count = 0

        # Spotter state history for coloring background
        self.state_history = deque(maxlen=history_frames)

        # Max velocity seen (for adaptive normalization)
        self.vel_max = 0.05   # initial; adapts over time

    def update(self, channel_vels: dict, state: SpotterState):
        """Push one frame of velocities."""
        self.frame_count += 1
        for key in self.vel_keys:
            v = channel_vels.get(key, 0.0)
            self.vel_buffers[key].append(v)
            self.vel_max = max(self.vel_max, v * 1.2)
        self.state_history.append(state)

    def mark_sign(self, label: str, duration_frames: int):
        """Mark a detected sign span at current position."""
        end = self.frame_count
        start = end - duration_frames
        self.sign_spans.append((start, end, label))

    def render(self) -> np.ndarray:
        """
        Render heatmap as BGR image.

        Layout:
          Each channel is a horizontal strip (ch_h pixels tall).
          X-axis = time (most recent on right).
          Color intensity = velocity magnitude (dark=low, bright=high).
          Yellow overlay bars = detected sign spans.
          Right side: channel labels.
        """
        label_w = 55  # pixels reserved for labels
        hmap_w = self.width - label_w
        total_h = self.n_channels * self.ch_h + 12  # +12 for bottom time axis
        canvas = np.zeros((total_h, self.width, 3), dtype=np.uint8)

        n = len(self.vel_buffers["body"])
        if n == 0:
            return canvas

        for ci, key in enumerate(self.vel_keys):
            buf = list(self.vel_buffers[key])
            y_start = ci * self.ch_h
            y_end = y_start + self.ch_h

            # Create 1D velocity signal and stretch to hmap_w
            vel_arr = np.array(buf, dtype=np.float32)
            # Normalize to [0, 1]
            normed = np.clip(vel_arr / max(self.vel_max, 1e-6), 0, 1)

            # Stretch to hmap_w pixels
            if len(normed) < hmap_w:
                # Pad left with zeros
                padded = np.zeros(hmap_w, dtype=np.float32)
                padded[hmap_w - len(normed):] = normed
                normed = padded
            else:
                # Take the last hmap_w samples
                normed = normed[-hmap_w:]

            # Create column for this channel: use inferno-like colormap
            # Reshape to 2D for colormap: (1, hmap_w) -> (ch_h, hmap_w)
            strip_1d = (normed * 255).astype(np.uint8)
            strip_2d = np.tile(strip_1d, (self.ch_h, 1))  # (ch_h, hmap_w)

            # Apply colormap (COLORMAP_INFERNO for research-style look)
            strip_colored = cv2.applyColorMap(strip_2d, cv2.COLORMAP_INFERNO)

            # Place into canvas
            canvas[y_start:y_end, :hmap_w] = strip_colored

            # Channel separator line
            if ci > 0:
                cv2.line(canvas, (0, y_start), (self.width, y_start), (80, 80, 80), 1)

            # Channel label on right side
            label_y = y_start + self.ch_h // 2 + 5
            cv2.putText(canvas, self.channel_names[ci], (hmap_w + 5, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.channel_colors[ci], 1)

        # Overlay sign span markers (yellow bars)
        for (s_start, s_end, label) in self.sign_spans:
            # Convert frame indices to pixel positions
            visible_start = self.frame_count - n
            if s_end < visible_start:
                continue  # span is no longer in visible window

            px_start = int((s_start - visible_start) / max(n, 1) * hmap_w)
            px_end = int((s_end - visible_start) / max(n, 1) * hmap_w)
            px_start = max(0, min(px_start, hmap_w))
            px_end = max(0, min(px_end, hmap_w))

            if px_end - px_start < 2:
                px_end = px_start + 2

            # Draw yellow bars on each channel
            for ci in range(self.n_channels):
                y_start = ci * self.ch_h
                y_end = y_start + self.ch_h
                # Semi-transparent yellow overlay
                overlay = canvas[y_start:y_end, px_start:px_end].copy()
                yellow = np.full_like(overlay, (0, 255, 255))  # yellow BGR
                cv2.addWeighted(overlay, 0.5, yellow, 0.5, 0,
                                canvas[y_start:y_end, px_start:px_end])

                # Bottom border of yellow mark
                cv2.line(canvas, (px_start, y_end - 1), (px_end, y_end - 1),
                         (0, 220, 220), 1)

            # Label above the span
            if px_end - px_start > 10:
                cv2.putText(canvas, label[:8], (px_start + 2, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # Overlay spotter state coloring on bottom edge
        state_strip_h = 4
        state_y = self.n_channels * self.ch_h
        states = list(self.state_history)
        if states:
            state_arr = np.zeros(hmap_w, dtype=np.uint8)
            if len(states) < hmap_w:
                mapped = np.zeros(len(states), dtype=np.uint8)
                for i, s in enumerate(states):
                    if s == SpotterState.SIGNING:
                        mapped[i] = 2
                    elif s == SpotterState.COOLDOWN:
                        mapped[i] = 1
                padded = np.zeros(hmap_w, dtype=np.uint8)
                padded[hmap_w - len(mapped):] = mapped
                state_arr = padded
            else:
                for i, s in enumerate(states[-hmap_w:]):
                    if s == SpotterState.SIGNING:
                        state_arr[i] = 2
                    elif s == SpotterState.COOLDOWN:
                        state_arr[i] = 1

            for x in range(hmap_w):
                if state_arr[x] == 2:  # SIGNING
                    canvas[state_y:state_y + state_strip_h, x] = (0, 200, 0)
                elif state_arr[x] == 1:  # COOLDOWN
                    canvas[state_y:state_y + state_strip_h, x] = (0, 140, 200)

        # Time axis labels at bottom
        time_y = total_h - 2
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            px = int(frac * hmap_w)
            frame_ago = int((1.0 - frac) * n)
            label = f"-{frame_ago}f" if frame_ago > 0 else "now"
            cv2.putText(canvas, label, (max(0, px - 15), time_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1)

        return canvas


# ═══════════════════════════════════════════════════════════════
#  PREDICTOR (single/ensemble model inference)
# ═══════════════════════════════════════════════════════════════

class Predictor:
    """Wraps model inference with 31→56 class adaptation."""

    def __init__(self, model_name="hybrid", use_ensemble=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ensemble = use_ensemble
        self.ckpt_num_classes = NUM_CLASSES

        if use_ensemble and HAS_ENSEMBLE:
            self.ensemble = Ensemble.from_config(device=self.device)
            self.model = None
            print(f"[Predictor] Ensemble mode: {self.ensemble.loaded_model_names}")
        else:
            self.use_ensemble = False
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                # Support both checkpoint formats
                sd = ckpt.get("model_state_dict", ckpt.get("model", {}))
                # Auto-detect num_classes: find the LAST 2D weight in classifier/fc (output layer)
                for k, v in sd.items():
                    if ("classifier" in k or "fc" in k) and "weight" in k and hasattr(v, 'dim') and v.dim() == 2:
                        self.ckpt_num_classes = v.shape[0]  # keep overwriting → last one wins
                if self.ckpt_num_classes != NUM_CLASSES:
                    print(f"[Predictor] Checkpoint has {self.ckpt_num_classes} classes, config has {NUM_CLASSES}. Adapting...")
                self.model = get_model(model_name, num_classes=self.ckpt_num_classes).to(self.device)
                self.model.load_state_dict(sd, strict=False)
                self.model.eval()
                print(f"[Predictor] Loaded {model_name} ({self.ckpt_num_classes} classes)")
            else:
                print(f"[Predictor] No checkpoint at {ckpt_path}. Random weights.")
                self.model = get_model(model_name, num_classes=NUM_CLASSES).to(self.device)
                self.model.eval()
            self.ensemble = None

    @torch.no_grad()
    def predict(self, frames_list):
        """
        Predict from list of keypoint frames.
        Returns (probs, label, confidence)
        """
        x = self._preprocess(frames_list)
        if x is None:
            return np.zeros(NUM_CLASSES), "", 0.0

        x = x.to(self.device)

        if self.use_ensemble:
            probs = self.ensemble.predict_probs(x)
        else:
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            if self.ckpt_num_classes < NUM_CLASSES:
                padded = np.zeros(NUM_CLASSES, dtype=probs.dtype)
                padded[:self.ckpt_num_classes] = probs
                probs = padded

        top_idx = int(probs.argmax())
        label = IDX2CLASS.get(top_idx, f"?{top_idx}")
        conf = float(probs[top_idx])
        return probs, label, conf

    def _preprocess(self, frames_list):
        """Convert list of (75,2) frames to tensor (1, N_FRAMES, 150)."""
        if not frames_list:
            return None
        n = len(frames_list)
        if n >= N_FRAMES:
            start = (n - N_FRAMES) // 2
            frames_list = frames_list[start:start + N_FRAMES]
        else:
            frames_list = frames_list + [frames_list[-1]] * (N_FRAMES - n)

        arr = np.array([f.flatten() for f in frames_list], dtype=np.float32)  # (50, 150)
        return torch.from_numpy(arr).unsqueeze(0)  # (1, 50, 150)


# ═══════════════════════════════════════════════════════════════
#  TEMPORAL SMOOTHER
# ═══════════════════════════════════════════════════════════════

class TemporalSmoother:
    """Smooth predictions over a sliding window."""

    def __init__(self, window=5, min_confidence=0.35):
        self.window = window
        self.min_confidence = min_confidence
        self.history = deque(maxlen=window)

    def update(self, probs):
        self.history.append(probs.copy())

    def get_smoothed(self):
        if not self.history:
            return "", 0.0
        avg = np.mean(list(self.history), axis=0)
        idx = int(avg.argmax())
        conf = float(avg[idx])
        if conf < self.min_confidence:
            return "", 0.0
        return IDX2CLASS.get(idx, "?"), conf


# ═══════════════════════════════════════════════════════════════
#  KEYPOINT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_keypoints(results) -> np.ndarray:
    """Extract 75 landmarks (33 pose + 21 LH + 21 RH) → (75, 2)."""
    pose = np.zeros((33, 2), dtype=np.float32)
    lh = np.zeros((21, 2), dtype=np.float32)
    rh = np.zeros((21, 2), dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i] = [lm.x, lm.y]
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i] = [lm.x, lm.y]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i] = [lm.x, lm.y]

    return np.concatenate([pose, lh, rh], axis=0)  # (75, 2)


# ═══════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ═══════════════════════════════════════════════════════════════

def draw_skeleton(frame, results):
    """Draw MediaPipe skeleton on frame."""
    if not HAS_MEDIAPIPE:
        return
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                               mp_holistic.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=2),
                               mp_draw.DrawingSpec(color=(80, 44, 121), thickness=1))
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                               mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=2),
                               mp_draw.DrawingSpec(color=(121, 44, 250), thickness=1))
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                               mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=2),
                               mp_draw.DrawingSpec(color=(245, 66, 230), thickness=1))


def draw_spotter_badge(frame, spotter, x=10, y=10):
    """Draw spotter state badge + velocity meter."""
    debug = spotter.get_debug_info()
    state = debug["state"]
    vel = debug["smoothed_vel"]

    # State badge
    colors = {"IDLE": (120, 120, 120), "SIGNING": (0, 200, 0), "COOLDOWN": (0, 140, 200)}
    color = colors.get(state, (120, 120, 120))
    cv2.rectangle(frame, (x, y), (x + 100, y + 22), color, -1)
    cv2.putText(frame, state, (x + 5, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Velocity meter (horizontal bar)
    meter_x = x + 105
    meter_w = 120
    meter_h = 16
    meter_y = y + 3

    # Background
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h),
                  (50, 50, 50), -1)

    # Thresholds
    thresh_px = int(SPOT_VELOCITY_THRESHOLD / 0.1 * meter_w)
    idle_px = int(SPOT_IDLE_THRESHOLD / 0.1 * meter_w)
    cv2.line(frame, (meter_x + thresh_px, meter_y), (meter_x + thresh_px, meter_y + meter_h),
             (0, 0, 200), 1)
    cv2.line(frame, (meter_x + idle_px, meter_y), (meter_x + idle_px, meter_y + meter_h),
             (200, 200, 0), 1)

    # Velocity bar
    vel_px = int(min(vel / 0.1, 1.0) * meter_w)
    bar_color = (0, 200, 0) if vel > SPOT_VELOCITY_THRESHOLD else (150, 150, 150)
    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + vel_px, meter_y + meter_h),
                  bar_color, -1)

    # Velocity text
    cv2.putText(frame, f"vel={vel:.4f}", (meter_x + meter_w + 5, meter_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Sign count
    cv2.putText(frame, f"signs: {debug['sign_count']}", (x, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if debug["sign_len"] > 0:
        cv2.putText(frame, f"len: {debug['sign_len']}", (x + 80, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def draw_prediction(frame, label, confidence, y_offset=70):
    """Draw prediction label + confidence bar on frame."""
    h, w = frame.shape[:2]

    if not label:
        return

    # Background bar
    bar_h = 50
    bar_y = h - bar_h - y_offset
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, bar_y + bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Label
    color = (0, 255, 0) if confidence > 0.6 else (0, 200, 255) if confidence > 0.3 else (100, 100, 255)
    cv2.putText(frame, f"{label}", (10, bar_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"{confidence:.1%}", (w - 100, bar_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Confidence bar
    bar_w = int(confidence * (w - 220))
    cv2.rectangle(frame, (160, bar_y + 15), (160 + bar_w, bar_y + 35), color, -1)


def draw_sentence_log(frame, sentences, y_offset=10):
    """Draw accumulated sentence at bottom of frame."""
    h, w = frame.shape[:2]
    if not sentences:
        return

    text = " ".join(sentences[-10:])  # last 10 words
    # Background
    cv2.rectangle(frame, (0, h - 30 - y_offset), (w, h - y_offset), (20, 20, 20), -1)
    cv2.putText(frame, text, (10, h - 12 - y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)


def draw_mode_info(frame, mode, fps, show_skeleton, show_heatmap):
    """Draw mode + FPS info at top-right."""
    h, w = frame.shape[:2]
    info_lines = [
        f"Mode: {mode}",
        f"FPS: {fps:.0f}",
        f"Skel: {'ON' if show_skeleton else 'OFF'}",
        f"HMap: {'ON' if show_heatmap else 'OFF'}",
    ]
    for i, line in enumerate(info_lines):
        y = 18 + i * 18
        cv2.putText(frame, line, (w - 130, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


# ═══════════════════════════════════════════════════════════════
#  MAIN APPLICATION LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BSL Monitor GUI")
    parser.add_argument("--mode", default="hybrid",
                        choices=["sliding", "spotting", "hybrid"])
    parser.add_argument("--model", default="hybrid",
                        choices=["bilstm", "transformer", "stgcn", "tcn", "hybrid"])
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=CAM_CONFIDENCE)
    parser.add_argument("--heatmap-height", type=int, default=30,
                        help="Height per heatmap channel row (default 30px)")
    parser.add_argument("--heatmap-history", type=int, default=300,
                        help="Number of history frames for heatmap (default 300)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if not HAS_MEDIAPIPE:
        print("[ERROR] mediapipe is required. Install: pip install mediapipe")
        return

    # ─── Initialize components ────────────────────────────
    print("[Init] Loading model...")
    predictor = Predictor(args.model, use_ensemble=args.ensemble)

    spotter = SignSpotter()
    heatmap = VelocityHeatmap(
        width=args.width,
        channel_height=args.heatmap_height,
        history_frames=args.heatmap_history,
    )
    smoother = TemporalSmoother(window=5, min_confidence=args.confidence)

    # State
    mode = args.mode
    modes = ["sliding", "spotting", "hybrid"]
    paused = False
    show_skeleton = True
    show_heatmap = True
    sentence_log = []
    current_label = ""
    current_conf = 0.0
    frame_buffer = deque(maxlen=CAM_BUFFER_SIZE)
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    fps_counter = 0

    # Window name
    win_name = "BSL Monitor"

    # ─── Camera + MediaPipe ───────────────────────────────
    print(f"[Init] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    print(f"[Ready] Mode={mode} | Model={args.model} | Press H for help")
    print(f"  Controls: Q=quit  SPACE=pause  M=mode  S=skeleton  H=heatmap  R=reset  +/-=threshold")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Camera read failed")
                    break

                frame = cv2.flip(frame, 1)  # Mirror
                frame_count += 1

                # ─── MediaPipe processing ─────────────────
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                # ─── Extract keypoints ────────────────────
                kps = extract_keypoints(results)  # (75, 2)
                frame_buffer.append(kps)

                # ─── Sign Spotter ─────────────────────────
                spotter_result = spotter.feed(kps)
                debug = spotter.get_debug_info()
                channel_vels = debug["channel_vels"]

                # Update heatmap
                heatmap.update(channel_vels, spotter.state)

                # ─── Classification ───────────────────────
                if mode in ("spotting", "hybrid"):
                    if spotter_result is not None:
                        sign_frames, meta = spotter_result
                        probs, label, conf = predictor.predict(sign_frames)
                        if conf >= args.confidence:
                            current_label = label
                            current_conf = conf
                            heatmap.mark_sign(label, meta.get("raw_length", 30))
                            if not sentence_log or sentence_log[-1] != label:
                                sentence_log.append(label)
                            smoother.update(probs)

                if mode in ("sliding", "hybrid"):
                    if frame_count % PREDICT_EVERY_N == 0 and len(frame_buffer) >= N_FRAMES // 2:
                        frames_list = list(frame_buffer)
                        probs, label, conf = predictor.predict(frames_list)
                        smoother.update(probs)
                        sm_label, sm_conf = smoother.get_smoothed()
                        if sm_conf >= args.confidence:
                            # In hybrid mode, only use sliding if no recent spotting result
                            if mode == "sliding" or (mode == "hybrid" and current_conf < args.confidence):
                                current_label = sm_label
                                current_conf = sm_conf

                # ─── Draw skeleton ────────────────────────
                if show_skeleton:
                    draw_skeleton(frame, results)

                # ─── Draw UI overlays ─────────────────────
                draw_spotter_badge(frame, spotter)
                draw_prediction(frame, current_label, current_conf, y_offset=10 if not show_heatmap else 10)
                draw_sentence_log(frame, sentence_log, y_offset=50 if not show_heatmap else 50)

                # FPS
                fps_counter += 1
                if time.time() - fps_timer > 1.0:
                    fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()

                draw_mode_info(frame, mode, fps, show_skeleton, show_heatmap)

                # ─── Compose final display ────────────────
                if show_heatmap:
                    hmap_img = heatmap.render()
                    # Resize heatmap to match frame width
                    if hmap_img.shape[1] != frame.shape[1]:
                        hmap_img = cv2.resize(hmap_img, (frame.shape[1], hmap_img.shape[0]))
                    # Stack vertically: camera on top, heatmap on bottom
                    display = np.vstack([frame, hmap_img])
                else:
                    display = frame

            else:
                # Paused — show last frame with PAUSED overlay
                if 'display' not in locals():
                    continue
                cv2.putText(display, "PAUSED", (display.shape[1] // 2 - 60, display.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # ─── Show window ──────────────────────────────
            cv2.imshow(win_name, display)

            # ─── Handle keyboard ──────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
            elif key in (ord('m'), ord('M')):  # Cycle mode
                idx = modes.index(mode)
                mode = modes[(idx + 1) % len(modes)]
                print(f"[Mode] → {mode}")
            elif key in (ord('s'), ord('S')):  # Toggle skeleton
                show_skeleton = not show_skeleton
            elif key in (ord('h'), ord('H')):  # Toggle heatmap
                show_heatmap = not show_heatmap
            elif key in (ord('r'), ord('R')):  # Reset
                spotter.reset()
                sentence_log.clear()
                current_label = ""
                current_conf = 0.0
                frame_buffer.clear()
                print("[Reset] Spotter + sentence log cleared")
            elif key == ord('+') or key == ord('='):  # Increase threshold
                spotter.velocity_threshold = min(0.2, spotter.velocity_threshold + 0.002)
                print(f"[Threshold] → {spotter.velocity_threshold:.4f}")
            elif key == ord('-'):  # Decrease threshold
                spotter.velocity_threshold = max(0.001, spotter.velocity_threshold - 0.002)
                print(f"[Threshold] → {spotter.velocity_threshold:.4f}")

            # Decay current label confidence over time
            if current_conf > 0:
                current_conf *= 0.995  # slow decay
                if current_conf < 0.1:
                    current_label = ""
                    current_conf = 0.0

    except KeyboardInterrupt:
        print("\n[Interrupt] Stopping...")
    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()
        print("[Done] Monitor closed.")


if __name__ == "__main__":
    main()

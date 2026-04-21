# =============================================================
# sign_spotter.py — Heuristic sign boundary detection (Phase 2)
# =============================================================
"""
Detects when a person starts and stops signing using hand velocity.

State machine:
    IDLE  →  SIGNING  →  COOLDOWN  →  IDLE
         ↑ velocity > threshold    ↑ velocity < idle_thresh for N frames

When a sign window completes, emits the captured keypoint frames
for classification.

Usage:
    spotter = SignSpotter()
    for each camera frame:
        kps = extract_keypoints(...)          # (75, 2)
        result = spotter.feed(kps)
        if result is not None:
            frames, meta = result             # frames: list[(75,2)], meta: dict
            # → classify frames
"""

import numpy as np
from collections import deque
from enum import Enum, auto
from config import (
    N_FRAMES, N_LANDMARKS,
    SPOT_VELOCITY_THRESHOLD,
    SPOT_MIN_SIGN_FRAMES,
    SPOT_MAX_SIGN_FRAMES,
    SPOT_COOLDOWN_FRAMES,
    SPOT_PRE_BUFFER,
    SPOT_POST_BUFFER,
    SPOT_SMOOTH_WINDOW,
    SPOT_IDLE_THRESHOLD,
)


class SpotterState(Enum):
    IDLE     = auto()
    SIGNING  = auto()
    COOLDOWN = auto()


class SignSpotter:
    """
    Heuristic sign spotter using hand landmark velocity.

    Key idea:
      - Track wrist + hand center velocity frame-to-frame
      - High velocity → hands are moving → signing
      - Sustained low velocity → sign ended

    Parameters come from config.py (SPOT_*).
    """

    def __init__(
        self,
        velocity_threshold: float = SPOT_VELOCITY_THRESHOLD,
        min_sign_frames:    int   = SPOT_MIN_SIGN_FRAMES,
        max_sign_frames:    int   = SPOT_MAX_SIGN_FRAMES,
        cooldown_frames:    int   = SPOT_COOLDOWN_FRAMES,
        pre_buffer:         int   = SPOT_PRE_BUFFER,
        post_buffer:        int   = SPOT_POST_BUFFER,
        smooth_window:      int   = SPOT_SMOOTH_WINDOW,
        idle_threshold:     float = SPOT_IDLE_THRESHOLD,
        fps_ratio:          float = 1.0,
    ):
        self.velocity_threshold = velocity_threshold
        self.min_sign_frames    = min_sign_frames
        self.max_sign_frames    = max_sign_frames
        self.cooldown_frames    = cooldown_frames
        self.pre_buffer         = pre_buffer
        self.post_buffer        = post_buffer
        self.smooth_window      = smooth_window
        self.idle_threshold     = idle_threshold
        self.fps_ratio          = fps_ratio  # target_fps / source_fps for resampling

        # State
        self.state        = SpotterState.IDLE
        self.prev_kps     = None
        self.pre_ring     = deque(maxlen=pre_buffer)   # frames before onset
        self.sign_frames  = []                          # frames during sign
        self.idle_counter = 0                           # consecutive idle frames
        self.cooldown_counter = 0
        self.vel_history  = deque(maxlen=smooth_window) # for smoothing
        self.frame_idx    = 0
        self.sign_count   = 0                           # total signs detected
        self._channel_vels = {'body': 0.0, 'face': 0.0, 'lh': 0.0, 'rh': 0.0}

    def reset(self):
        """Reset spotter state (e.g., on user press R)."""
        self.state        = SpotterState.IDLE
        self.prev_kps     = None
        self.pre_ring.clear()
        self.sign_frames  = []
        self.idle_counter = 0
        self.cooldown_counter = 0
        self.vel_history.clear()
        self.frame_idx    = 0

    # ─── Per-channel velocity (for visualizer) ────────────────
    def _compute_channel_velocities(self, kps: np.ndarray) -> dict:
        """
        Compute per-channel hand/body/face velocities for visualization.

        Channels:
          body  — shoulder + elbow + hip landmarks (11-14, 23-24)
          face  — facial / head landmarks (0-10)
          lh    — left hand center (33-53)
          rh    — right hand center (54-74)

        Returns: dict with keys 'body', 'face', 'lh', 'rh' (floats).
        """
        if self.prev_kps is None:
            return {'body': 0.0, 'face': 0.0, 'lh': 0.0, 'rh': 0.0}

        def _center_vel(indices):
            curr = kps[indices]
            prev = self.prev_kps[indices]
            valid = np.any(curr != 0, axis=1) & np.any(prev != 0, axis=1)
            if not np.any(valid):
                return 0.0
            return float(np.linalg.norm(
                curr[valid].mean(axis=0) - prev[valid].mean(axis=0)))

        face_vel = _center_vel(np.arange(0, 11))                     # nose/eyes/ears
        body_vel = _center_vel(np.array([11, 12, 13, 14, 23, 24]))   # shoulder/elbow/hip

        lh_curr = kps[33:54]; lh_prev = self.prev_kps[33:54]
        lh_vel = float(np.linalg.norm(
            lh_curr.mean(axis=0) - lh_prev.mean(axis=0))
        ) if (np.any(lh_curr) and np.any(lh_prev)) else 0.0

        rh_curr = kps[54:75]; rh_prev = self.prev_kps[54:75]
        rh_vel = float(np.linalg.norm(
            rh_curr.mean(axis=0) - rh_prev.mean(axis=0))
        ) if (np.any(rh_curr) and np.any(rh_prev)) else 0.0

        return {'body': body_vel, 'face': face_vel, 'lh': lh_vel, 'rh': rh_vel}

    def _compute_hand_velocity(self, kps: np.ndarray) -> float:
        """
        Compute max hand velocity between current and previous frame.

        Uses:
          - Left hand center (mean of landmarks 33–53)
          - Right hand center (mean of landmarks 54–74)
          - Pose wrists (landmarks 15, 16)

        Returns: float — max displacement (normalized [0, ~1]).
        """
        if self.prev_kps is None:
            return 0.0

        vel = 0.0

        # Left hand center
        lh_curr = kps[33:54]   # (21, 2)
        lh_prev = self.prev_kps[33:54]
        if np.any(lh_curr) and np.any(lh_prev):   # not all zeros
            lc = lh_curr.mean(axis=0)
            lp = lh_prev.mean(axis=0)
            vel = max(vel, np.linalg.norm(lc - lp))

        # Right hand center
        rh_curr = kps[54:75]   # (21, 2)
        rh_prev = self.prev_kps[54:75]
        if np.any(rh_curr) and np.any(rh_prev):
            rc = rh_curr.mean(axis=0)
            rp = rh_prev.mean(axis=0)
            vel = max(vel, np.linalg.norm(rc - rp))

        # Wrists from pose (more stable when hand detection flickers)
        for idx in [15, 16]:   # left wrist, right wrist
            if np.any(kps[idx]) and np.any(self.prev_kps[idx]):
                wvel = np.linalg.norm(kps[idx] - self.prev_kps[idx])
                vel = max(vel, wvel)

        return float(vel)

    def _smoothed_velocity(self, raw_vel: float) -> float:
        """Moving average of velocity."""
        self.vel_history.append(raw_vel)
        if len(self.vel_history) == 0:
            return raw_vel
        return float(np.mean(self.vel_history))

    def _finalize_sign(self, fps_ratio: float = 1.0) -> tuple | None:
        """
        Pad/trim captured sign to N_FRAMES and return.
        Returns (frames_list, meta_dict) or None if too short.

        fps_ratio: target_fps / source_fps (e.g. 25/30 = 0.833).
                   When != 1.0, resamples the raw frames to match
                   training data temporal scale BEFORE center-crop.
        """
        frames = list(self.pre_ring) + self.sign_frames
        n = len(frames)

        if n < self.min_sign_frames:
            # Too short — discard
            self.sign_frames = []
            return None

        self.sign_count += 1
        raw_length = n

        # FPS normalization: resample to training FPS *before* center-crop
        if fps_ratio != 1.0 and n > 1:
            target_n = max(1, int(round(n * fps_ratio)))
            indices  = np.linspace(0, n - 1, target_n).astype(int)
            frames   = [frames[i] for i in indices]
            n = len(frames)

        # Pad or trim to N_FRAMES
        if n >= N_FRAMES:
            # Center-crop to N_FRAMES
            start = (n - N_FRAMES) // 2
            frames = frames[start:start + N_FRAMES]
        else:
            # Pad by repeating last frame
            pad_count = N_FRAMES - n
            frames = frames + [frames[-1]] * pad_count

        meta = {
            "raw_length":  raw_length,
            "sign_index":  self.sign_count,
            "frame_idx":   self.frame_idx,
        }
        self.sign_frames = []
        return (frames, meta)

    def feed(self, kps: np.ndarray) -> tuple | None:
        """
        Feed one frame of keypoints (75, 2) into the spotter.

        Returns:
            None              — no sign detected yet
            (frames, meta)    — a complete sign window detected
                frames: list of np.ndarray (each (75, 2)), length=N_FRAMES
                meta: dict with metadata
        """
        self.frame_idx += 1
        result = None

        raw_vel  = self._compute_hand_velocity(kps)
        velocity = self._smoothed_velocity(raw_vel)
        self._channel_vels = self._compute_channel_velocities(kps)
        self.prev_kps = kps.copy()

        if self.state == SpotterState.IDLE:
            self.pre_ring.append(kps.copy())

            if velocity > self.velocity_threshold:
                # Transition: IDLE → SIGNING
                self.state = SpotterState.SIGNING
                self.sign_frames = [kps.copy()]
                self.idle_counter = 0
            # else: stay IDLE

        elif self.state == SpotterState.SIGNING:
            self.sign_frames.append(kps.copy())

            if velocity < self.idle_threshold:
                self.idle_counter += 1
            else:
                self.idle_counter = 0

            total_len = len(self.pre_ring) + len(self.sign_frames)

            # End conditions
            if self.idle_counter >= self.post_buffer:
                # Hands stopped → sign ended
                result = self._finalize_sign(self.fps_ratio)
                self.state = SpotterState.COOLDOWN
                self.cooldown_counter = 0

            elif total_len >= self.max_sign_frames:
                # Force cut (sign too long)
                result = self._finalize_sign(self.fps_ratio)
                self.state = SpotterState.COOLDOWN
                self.cooldown_counter = 0

        elif self.state == SpotterState.COOLDOWN:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.cooldown_frames:
                self.state = SpotterState.IDLE
                self.pre_ring.clear()
                self.idle_counter = 0

        return result

    @property
    def is_signing(self) -> bool:
        return self.state == SpotterState.SIGNING

    @property
    def current_sign_length(self) -> int:
        if self.state == SpotterState.SIGNING:
            return len(self.pre_ring) + len(self.sign_frames)
        return 0

    def get_debug_info(self) -> dict:
        """Return debug info for visualization."""
        return {
            "state":        self.state.name,
            "velocity":     float(self.vel_history[-1]) if self.vel_history else 0.0,
            "smoothed_vel": float(np.mean(self.vel_history)) if self.vel_history else 0.0,
            "sign_len":     self.current_sign_length,
            "sign_count":   self.sign_count,
            "idle_counter": self.idle_counter,
            "channel_vels": dict(self._channel_vels),   # body/face/lh/rh per-frame
        }

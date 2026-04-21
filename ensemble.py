# =============================================================
# ensemble.py — Multi-model ensemble inference (Phase 2)
# =============================================================
"""
Load multiple trained models, run inference on each,
combine predictions via soft voting (probability average)
or hard voting (majority class).

Usage:
    ens = Ensemble.from_config()           # loads ENSEMBLE_MODELS from config
    ens = Ensemble(["hybrid", "transformer"], weights=[0.6, 0.4])

    probs = ens.predict(input_tensor)      # (NUM_CLASSES,) numpy
    label = ens.predict_label(input_tensor) # str
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from models import get_model


def _cfg():
    """Resolve config at runtime so config_selector.activate() is respected."""
    return sys.modules.get('config', __import__('config'))


class Ensemble:
    """
    Multi-model ensemble for sign classification.

    Supports:
      - soft voting: weighted average of class probabilities
      - hard voting: majority vote of argmax predictions
    """

    def __init__(
        self,
        model_names: list[str] | None = None,
        weights: list[float] | None = None,
        mode: str = "soft",
        device: torch.device | None = None,
    ):
        cfg = _cfg()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mode = mode
        self.models = []
        self.num_classes = cfg.NUM_CLASSES
        self.idx2class = cfg.IDX2CLASS
        self.model_names = model_names or cfg.ENSEMBLE_MODELS
        self.weights = weights or cfg.ENSEMBLE_WEIGHTS

        # Normalize weights
        w_sum = sum(self.weights)
        self.weights = [w / w_sum for w in self.weights]

        # Pad weights if fewer than models
        while len(self.weights) < len(self.model_names):
            self.weights.append(1.0 / len(self.model_names))

        # Load each model — checkpoint resolution order:
        # 1. HardMine (_hardmine.pt) if ENSEMBLE_USE_HARDMINE is True
        # 2. KD (_kd.pt) if ENSEMBLE_USE_KD is True
        # 3. Best (_best.pt) as fallback
        ckpt_dir = cfg.CHECKPOINT_DIR
        print(f"[Ensemble] CHECKPOINT_DIR = {ckpt_dir}")
        use_kd = getattr(cfg, 'ENSEMBLE_USE_KD', False)
        use_hm = getattr(cfg, 'ENSEMBLE_USE_HARDMINE', False)
        for name in self.model_names:
            hm_path   = os.path.join(ckpt_dir, f"{name}_hardmine.pt")
            kd_path   = os.path.join(ckpt_dir, f"{name}_kd.pt")
            best_path = os.path.join(ckpt_dir, f"{name}_best.pt")
            if use_hm and os.path.exists(hm_path):
                ckpt_path = hm_path
            elif use_kd and os.path.exists(kd_path):
                ckpt_path = kd_path
            elif os.path.exists(best_path):
                ckpt_path = best_path
            else:
                print(f"[Ensemble] WARNING: no checkpoint for '{name}' — skipping")
                continue

            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            # Detect num_classes from checkpoint — find last classifier 2D weight
            sd = ckpt.get("model_state_dict", ckpt.get("model", {}))
            ckpt_num_classes = self.num_classes
            for key, val in sd.items():
                if ("classifier" in key or "fc" in key) and key.endswith(".weight") and val.ndim == 2:
                    ckpt_num_classes = val.shape[0]  # last one wins
            if ckpt_num_classes != self.num_classes:
                print(f"[Ensemble] WARNING: '{name}' trained with {ckpt_num_classes} classes "
                      f"but config has {self.num_classes}. Loading with strict=False.")

            # Use model_kwargs from checkpoint to match architecture exactly
            mkw = ckpt.get("model_kwargs", {})
            model = get_model(name, num_classes=ckpt_num_classes, **mkw).to(self.device)
            model.load_state_dict(sd)
            model.eval()
            self.models.append((name, model, ckpt_num_classes))
            print(f"[Ensemble] Loaded '{name}' from {ckpt_path} ({ckpt_num_classes} classes, kwargs={mkw})")

        if not self.models:
            raise RuntimeError(
                "[Ensemble] No models loaded! Train at least one model first.\n"
                "  python train.py --model hybrid"
            )

        # Rebalance weights for successfully loaded models
        loaded_names = [n for n, _, _ in self.models]
        loaded_weights = []
        for i, name in enumerate(self.model_names):
            if name in loaded_names:
                loaded_weights.append(self.weights[i] if i < len(self.weights) else 1.0)
        w_sum = sum(loaded_weights) or 1.0
        self.weights = [w / w_sum for w in loaded_weights]

        print(f"[Ensemble] {len(self.models)} models, mode={self.mode}, weights={self.weights}")

    @classmethod
    def from_config(cls, device=None):
        """Create ensemble from config.py settings."""
        cfg = _cfg()
        return cls(
            model_names=cfg.ENSEMBLE_MODELS,
            weights=cfg.ENSEMBLE_WEIGHTS,
            mode=cfg.ENSEMBLE_MODE,
            device=device,
        )

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> np.ndarray:
        """
        Run ensemble inference.

        Args:
            x: input tensor (1, T, F) or (B, T, F)

        Returns:
            probs: numpy array (NUM_CLASSES,) — averaged class probabilities
                   (for batch=1; otherwise (B, NUM_CLASSES))
        """
        x = x.to(self.device)
        all_probs = []

        for name, model, ckpt_nc in self.models:
            logits = model(x)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()

            # If model has fewer classes than config, pad to num_classes
            if ckpt_nc < self.num_classes:
                if probs.ndim == 1:
                    padded = np.zeros(self.num_classes, dtype=probs.dtype)
                    padded[:ckpt_nc] = probs
                    probs = padded
                else:
                    padded = np.zeros((probs.shape[0], self.num_classes), dtype=probs.dtype)
                    padded[:, :ckpt_nc] = probs
                    probs = padded
            elif ckpt_nc > self.num_classes:
                probs = probs[..., :self.num_classes]

            all_probs.append(probs)

        if self.mode == "soft":
            # Weighted average of probabilities
            combined = np.zeros_like(all_probs[0])
            for prob, w in zip(all_probs, self.weights):
                combined += w * prob
            result = combined
        else:
            # Hard voting: each model votes for argmax
            votes = np.zeros_like(all_probs[0])
            for prob, w in zip(all_probs, self.weights):
                idx = prob.argmax(axis=-1)
                for b in range(prob.shape[0]):
                    votes[b, idx[b]] += w
            result = votes / votes.sum(axis=-1, keepdims=True)

        # Squeeze batch dim if single sample
        if result.shape[0] == 1:
            result = result[0]

        return result

    def predict_label(self, x: torch.Tensor) -> tuple[str, float]:
        """
        Predict single class label.

        Returns:
            (class_name, confidence)
        """
        probs   = self.predict_probs(x)
        top_idx = probs.argmax()
        return self.idx2class[int(top_idx)], float(probs[top_idx])

    def predict_topk(self, x: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
        """
        Predict top-K classes.

        Returns:
            list of (class_name, probability)
        """
        probs = self.predict_probs(x)
        top_k = np.argsort(probs)[::-1][:k]
        return [(self.idx2class[int(i)], float(probs[i])) for i in top_k]

    @property
    def num_models(self) -> int:
        return len(self.models)

    @property
    def loaded_model_names(self) -> list[str]:
        return [n for n, _, _ in self.models]

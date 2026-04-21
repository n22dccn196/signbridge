# =============================================================
# config_selector.py — Soft re-routing between Auto and Manual configs
# =============================================================
"""
Phase 1 — Soft Re-routing.

Provides transparent config switching so that dataset.py, models/,
ensemble.py, and demo_camera.py all pick up the correct class list,
paths, and hyperparameters without per-file monkey-patching.

Usage:
    import config_selector
    config_selector.activate("manual")   # or "auto"
    config_selector.activate("wlasl100") # 100-class WLASL
    config_selector.activate("wlasl2000")# 2000-class WLASL
    # Now `import config` returns the correct module everywhere
"""
import sys
import importlib
from typing import Optional


# ─── Registry ─────────────────────────────────────────────────
_CONFIGS = {
    "auto":   "config",          # 36 classes, features_npy_rootrel
    "manual": "config_manual",   # 21 classes, features_npy_manual
    "wlasl":  "config_wlasl100", # backward-compat alias -> WLASL-100
    "wlasl100": "config_wlasl100",
    "wlasl2000": "config_wlasl2000",
}
_active = None  # tracks which dataset config is live


def get_config(dataset: str = "manual"):
    """
    Import and return the config module for a dataset WITHOUT
    injecting into sys.modules.  Safe for read-only inspection.
    """
    key = dataset.lower()
    if key not in _CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(_CONFIGS)}")
    return importlib.import_module(_CONFIGS[key])


def activate(dataset: str = "manual"):
    """
    Inject the chosen config into ``sys.modules['config']`` so that
    every downstream ``from config import …`` picks up the right values.

    Call this ONCE at the top of your entry-point script (train, demo, etc.)
    before importing dataset.py or models/.

    Returns the activated config module.
    """
    global _active
    key = dataset.lower()
    if key not in _CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(_CONFIGS)}")

    mod = importlib.import_module(_CONFIGS[key])
    sys.modules["config"] = mod
    _active = key
    print(f"[config_selector] Activated '{key}' config  "
          f"({mod.NUM_CLASSES} classes, {mod.NPY_DIR})")
    return mod


# OLD (Python 3.10+ only): def active_dataset() -> str | None:
def active_dataset() -> Optional[str]:
    """Return the name of the currently activated dataset, or None."""
    return _active


def is_manual() -> bool:
    """Convenience: True if the manual config is active."""
    return _active == "manual"


def is_auto() -> bool:
    """Convenience: True if the auto config is active."""
    return _active == "auto"

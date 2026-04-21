# =============================================================
# models/__init__.py — Export all 5 model options
# =============================================================
from .option1_bilstm      import SignBiLSTM
from .option2_transformer import SignTransformer
from .option3_stgcn       import SignSTGCN
from .option4_tcn         import SignTCN
from .option5_hybrid      import SignHybrid

MODEL_REGISTRY = {
    "bilstm":      SignBiLSTM,
    "transformer": SignTransformer,
    "stgcn":       SignSTGCN,
    "tcn":         SignTCN,
    "hybrid":      SignHybrid,      # Option 5 — default
}

def get_model(name: str = "hybrid", **kwargs):
    """Factory: get_model('hybrid', num_classes=31)"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)

__all__ = ["SignBiLSTM", "SignTransformer", "SignSTGCN", "SignTCN", "SignHybrid", "get_model"]

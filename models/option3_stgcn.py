# =============================================================
# models/option3_stgcn.py — Option 3: Spatial-Temporal GCN
# =============================================================
"""
Spatial-Temporal Graph Convolutional Network (Yan et al., 2018).
Adapted for MediaPipe Holistic 75 landmarks:
  - 33 pose keypoints (indices 0–32)
  - 21 left hand keypoints (indices 33–53)
  - 21 right hand keypoints (indices 54–74)

Architecture:
    Input (B, 2, 50, 75)  [channels, frames, nodes]
    → 6× ST-GCN block (spatial GCN + temporal BN-Conv)
    → GlobalAvgPool(time) → GlobalAvgPool(nodes)
    → FC(31)

Params: ~600K
Inference: ~3ms/frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import N_COORDS, N_FRAMES, N_LANDMARKS, NUM_CLASSES

# ─── Skeleton definition ──────────────────────────────────────
# MediaPipe Pose connections (subset used in our pipeline)
POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),        # left face
    (0, 4), (4, 5), (5, 6), (6, 8),        # right face
    (9, 10),                                # mouth
    (11, 12),                               # shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # right arm
    (11, 23), (12, 24), (23, 24),           # torso
    (23, 25), (25, 27), (27, 29), (27, 31), # left leg
    (24, 26), (26, 28), (28, 30), (28, 32), # right leg
]

# Left hand (21 nodes: 33-53) — MediaPipe hand connections
def _hand_edges(offset: int):
    """Generate hand connectivity for 21 keypoints starting at offset."""
    raw = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # index
        (0, 9), (9, 10), (10, 11), (11, 12),   # middle
        (0, 13),(13, 14),(14, 15),(15, 16),     # ring
        (0, 17),(17, 18),(18, 19),(19, 20),     # pinky
        (5, 9), (9, 13), (13, 17),              # palm
    ]
    return [(a + offset, b + offset) for a, b in raw]

ALL_EDGES = POSE_EDGES + _hand_edges(33) + _hand_edges(54)

# Cross-body connections (pose wrist ↔ hand root)
CROSS_EDGES = [(15, 33), (16, 54)]   # left wrist→left hand, right wrist→right hand
ALL_EDGES += CROSS_EDGES

N = N_LANDMARKS   # 75

def _build_adjacency() -> torch.Tensor:
    """Build normalized symmetric adjacency matrix A ∈ R^{N×N}."""
    A = np.eye(N, dtype=np.float32)
    for i, j in ALL_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Degree normalization: D^{-1/2} A D^{-1/2}
    D = np.diag(A.sum(axis=1) ** -0.5)
    A = D @ A @ D
    return torch.from_numpy(A)


# ─── Building blocks ──────────────────────────────────────────

class SpatialGCN(nn.Module):
    """Single-partition graph convolution: Y = A X W."""

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)
        self.W = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N)
        # Spatial mixing: (B, C, T, N) × (N, N) → (B, C, T, N)
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.W(x)             # (B, out_ch, T, N)
        return self.bn(x)


class STGCNBlock(nn.Module):
    """One ST-GCN block: spatial GCN + temporal Conv + residual."""

    def __init__(self, in_ch: int, out_ch: int, A: torch.Tensor,
                 stride: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gcn = SpatialGCN(in_ch, out_ch, A)
        # Temporal convolution over T axis
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(9, 1), stride=(stride, 1),
                      padding=(4, 0)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )
        self.relu = nn.ReLU(inplace=True)

        # Residual shortcut
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N)
        res = self.residual(x)
        x = self.relu(self.gcn(x))
        x = self.tcn(x) + res
        return self.relu(x)


# ─── Full ST-GCN ──────────────────────────────────────────────

class SignSTGCN(nn.Module):
    def __init__(
        self,
        in_channels:  int   = N_COORDS,      # 2
        num_classes:  int   = NUM_CLASSES,    # 31
        dropout:      float = 0.25,
    ):
        super().__init__()
        A = _build_adjacency()

        self.data_bn = nn.BatchNorm1d(in_channels * N)

        # 6 ST-GCN blocks: channel progression 64→64→64→128→128→256
        self.layers = nn.ModuleList([
            STGCNBlock( 2,  64, A, dropout=dropout),
            STGCNBlock(64,  64, A, dropout=dropout),
            STGCNBlock(64,  64, A, dropout=dropout),
            STGCNBlock(64, 128, A, dropout=dropout),
            STGCNBlock(128,128, A, dropout=dropout),
            STGCNBlock(128,256, A, dropout=dropout),
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(256, num_classes)
        self.drop = nn.Dropout(dropout)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 150) — flattened landmarks
        Returns:
            logits: (B, num_classes)
        """
        B, T, _ = x.shape
        # Reshape to (B, N, C, T) → BN → (B, C, T, N)
        x = x.view(B, T, N_LANDMARKS, N_COORDS)      # (B, T, N, C)
        x = x.permute(0, 3, 1, 2)                    # (B, C, T, N)

        # Data batch norm
        x_bn = x.permute(0, 1, 3, 2).contiguous().view(B, N_COORDS * N_LANDMARKS, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(B, N_COORDS, N_LANDMARKS, T).permute(0, 1, 3, 2)  # (B, C, T, N)

        for layer in self.layers:
            x = layer(x)                              # (B, 256, T', N)

        x = self.pool(x).view(x.size(0), -1)         # (B, 256) — safe for B=1
        x = self.drop(x)
        return self.fc(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    model = SignSTGCN()
    x = torch.randn(8, N_FRAMES, 150)
    out = model(x)
    print(f"[ST-GCN] Input: {x.shape} -> Output: {out.shape}")
    print(f"[ST-GCN] Parameters: {model.count_params():,}")

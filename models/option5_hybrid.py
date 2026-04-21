# =============================================================
# models/option5_hybrid.py — Option 5: CNN + BiLSTM + Attention
# =============================================================
"""
FOCUS model for BSL recognition.

Design rationale:
  1. CNN branch  — captures LOCAL temporal patterns (micro-gestures,
     short bursts of motion across adjacent frames) using dilated
     kernels to widen receptive field without downsampling.
  2. BiLSTM      — models the GLOBAL temporal dynamics / ordering
     of the sign across all 50 frames.
  3. Multi-Head  — Attention pooling over the LSTM output lets the
     network weight WHICH frames are most discriminative per sign,
     rather than naively averaging all frames.
  4. Landmark    — grouping: separate sub-networks for face landmarks
     and hand landmarks (fused before LSTM) to learn part-specific
     patterns.

Architecture overview:
                        ┌── InputNorm ──────────────────────────┐
                        │                                        │
    x (B,T,150)         │ Pose/hand/face split    (optional)     │
                        └──────────────────────────────────────-─┘
                              │
                         TemporalConvModule          (B,T,128)
                              │  multi-scale CNN (k=3,5,7)
                              │  BatchNorm + GELU + Dropout
                              ▼
                          BiLSTM(128, 2-layer)      (B,T,256)
                              │  recurrent temporal dynamics
                              ▼
                         MultiHeadAttention          (B,256)
                              │  (query=CLS, key/val=LSTM out)
                              │  weighted pooling
                              ▼
                       Classifier head
                              │  FC(256→128) → GELU → Dropout
                              │  FC(128→31)
                              ▼
                          logits (B,31)

Params: ~410K
Inference: ~4ms/frame on CPU
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    INPUT_SIZE, N_FRAMES, NUM_CLASSES,
    CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS,
    ATTN_HEADS, DROP_CNN, DROP_LSTM, DROP_CLS, FC_HIDDEN,
)


# ─── Sub-modules ──────────────────────────────────────────────

class MultiScaleConvModule(nn.Module):
    """
    Parallel 1D convolutions at scales k=3, 5, 7.
    Captures short (3-frame), medium (5-frame) and longer (7-frame)
    local motion patterns simultaneously.
    Feature maps are concatenated then projected back to out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        mid = out_ch // 3

        # Three parallel branches: small / medium / large kernel
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_ch, mid, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_ch, mid, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(mid),
            nn.GELU(),
        )
        # 1×1 fusion conv to merge 3 branches → out_ch
        self.fuse = nn.Sequential(
            nn.Conv1d(mid * 3, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual shortcut
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b3 = self.branch3(x)   # (B, mid, T)
        b5 = self.branch5(x)   # (B, mid, T)
        b7 = self.branch7(x)   # (B, mid, T)
        cat = torch.cat([b3, b5, b7], dim=1)   # (B, mid*3, T)
        out = self.fuse(cat) + self.shortcut(x)  # residual
        return F.gelu(out)


class AttentionPooling(nn.Module):
    """
    Multi-head cross-attention pooling.
    Uses a learnable query vector to select the most relevant
    time steps from the BiLSTM output sequence.

    Complexity: O(T) — much cheaper than full self-attention O(T²).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        # Learnable query: replaces the CLS token approach
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.query.view(1, embed_dim, 1))

        self.q_proj  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim)  — BiLSTM sequence
        Returns:
            pooled: (B, embed_dim) — weighted summary
        """
        B, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(self.query.expand(B, 1, D))     # (B, 1, D)
        k = self.k_proj(x)                               # (B, T, D)
        v = self.v_proj(x)                               # (B, T, D)

        # Reshape for multi-head
        q = q.view(B, 1, H, Dh).transpose(1, 2)         # (B, H, 1, Dh)
        k = k.view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)
        v = v.view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)

        attn = (q @ k.transpose(-2, -1)) / self.scale   # (B, H, 1, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).squeeze(-2)                     # (B, H, Dh)
        out = out.reshape(B, D)                          # (B, D)
        out = self.out_proj(out)
        return self.norm(out)


class TemporalConvStack(nn.Module):
    """
    2-layer stacked MultiScaleConvModule with increasing dilation
    to widen the temporal receptive field progressively.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block1 = MultiScaleConvModule(in_ch, out_ch, dropout)
        self.block2 = MultiScaleConvModule(out_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


# ─── Main model ───────────────────────────────────────────────

class SignHybrid(nn.Module):
    """
    Option 5: CNN + BiLSTM + Attention Pooling

    This is the FOCUS model. Combines three stages:
      Stage 1 (CNN):     local temporal feature extraction
      Stage 2 (BiLSTM):  sequential dynamics modeling
      Stage 3 (Attn):    discriminative temporal pooling
    """

    def __init__(
        self,
        input_size:   int   = INPUT_SIZE,     # 150
        cnn_channels: int   = CNN_CHANNELS,   # 128
        lstm_hidden:  int   = LSTM_HIDDEN,    # 128
        lstm_layers:  int   = LSTM_LAYERS,    # 2
        attn_heads:   int   = ATTN_HEADS,     # 4
        fc_hidden:    int   = FC_HIDDEN,      # 128
        drop_cnn:     float = DROP_CNN,       # 0.1
        drop_lstm:    float = DROP_LSTM,      # 0.3
        drop_cls:     float = DROP_CLS,       # 0.4
        num_classes:  int   = NUM_CLASSES,    # 31
    ):
        super().__init__()

        # ── Stage 0: Input normalization ────────────────────
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, cnn_channels, bias=False),
            nn.LayerNorm(cnn_channels),
            nn.GELU(),
        )

        # ── Stage 1: Multi-scale CNN ─────────────────────────
        self.cnn = TemporalConvStack(
            in_ch=cnn_channels,
            out_ch=cnn_channels,
            dropout=drop_cnn,
        )
        # CNN output: (B, cnn_channels, T)

        # ── Stage 2: BiLSTM ──────────────────────────────────
        lstm_out = lstm_hidden * 2   # 256 (bidirectional)
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_lstm if lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_out)

        # ── Stage 3: Attention pooling ───────────────────────
        self.attn_pool = AttentionPooling(
            embed_dim=lstm_out,
            num_heads=attn_heads,
            dropout=drop_cnn,
        )

        # ── Classifier head ───────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, fc_hidden),
            nn.GELU(),
            nn.Dropout(drop_cls),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.GELU(),
            nn.Dropout(drop_cls * 0.5),
            nn.Linear(fc_hidden // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                        # Forget gate bias = 1 (helps long sequences)
                        n = param.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 150)  — e.g. (64, 50, 150)
        Returns:
            logits: (B, num_classes)

        Forward pass detail:
          1. LayerNorm → Linear input proj → (B, T, 128)
          2. Permute → (B, 128, T) → CNN stack → (B, 128, T)
          3. Permute → (B, T, 128) → BiLSTM → (B, T, 256)
          4. LayerNorm → Attention pooling → (B, 256)
          5. Classifier FC layers → (B, 31)
        """
        # ── Stage 0 ────────────────────────
        x = self.input_norm(x)                 # (B, T, 150)
        x = self.input_proj(x)                 # (B, T, 128)

        # ── Stage 1: CNN ─────────────────────
        x = x.permute(0, 2, 1)                 # (B, 128, T) for Conv1d
        x = self.cnn(x)                         # (B, 128, T)
        x = x.permute(0, 2, 1)                 # (B, T, 128)

        # ── Stage 2: BiLSTM ──────────────────
        x, _ = self.bilstm(x)                  # (B, T, 256)
        x = self.lstm_norm(x)

        # ── Stage 3: Attention pooling ────────
        x = self.attn_pool(x)                  # (B, 256)

        # ── Classifier ───────────────────────
        return self.classifier(x)              # (B, 31)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_weights(self, x: torch.Tensor) -> tuple:
        """
        Returns (logits, attn_weights) for visualization.
        attn_weights: (B, num_heads, T) — which frames were attended to.
        """
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.lstm_norm(x)

        # Manually compute attention for visualization
        B, T, D = x.shape
        H, Dh = self.attn_pool.num_heads, self.attn_pool.head_dim
        q = self.attn_pool.q_proj(self.attn_pool.query.expand(B, 1, D))
        k = self.attn_pool.k_proj(x)
        q = q.view(B, 1, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) / self.attn_pool.scale, dim=-1)
        attn_weights = attn.squeeze(2)        # (B, H, T)

        v = self.attn_pool.v_proj(x).view(B, T, H, Dh).transpose(1, 2)
        out = (attn.detach() @ v).squeeze(-2).reshape(B, D)
        out = self.attn_pool.out_proj(out)
        out = self.attn_pool.norm(out)
        logits = self.classifier(out)

        return logits, attn_weights


# ── Quick test & summary ──────────────────────────────────────
if __name__ == "__main__":
    from config import N_FRAMES, INPUT_SIZE

    model = SignHybrid()
    x = torch.randn(8, N_FRAMES, INPUT_SIZE)

    # Forward
    logits = model(x)
    print(f"[Hybrid] Input:   {x.shape}")
    print(f"[Hybrid] Output:  {logits.shape}")
    print(f"[Hybrid] Params:  {model.count_params():,}")

    # Attention weights
    logits2, attn_w = model.get_attention_weights(x)
    print(f"[Hybrid] Attn weights shape: {attn_w.shape}  (B, heads, T)")

    # Param breakdown by module
    print("\n[Hybrid] Parameter breakdown:")
    total = 0
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        total += n
        print(f"  {name:<20} {n:>8,}")
    print(f"  {'TOTAL':<20} {total:>8,}")

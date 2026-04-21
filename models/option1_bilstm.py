# =============================================================
# models/option1_bilstm.py — Option 1: Bidirectional LSTM
# =============================================================
"""
Architecture:
    Input (B, 50, 150)
    → LayerNorm
    → BiLSTM(192) × 2  [bidirectional, 2 layers]
    → LayerNorm
    → Dropout(0.3)
    → GlobalAvgPool over T
    → FC(128) → GELU → Dropout(0.4)
    → FC(num_classes)

Params: ~470K (upgraded from 231K to fix 28% collapse)
Inference: ~3ms/frame
"""

import torch
import torch.nn as nn
from config import INPUT_SIZE, N_FRAMES, NUM_CLASSES


class SignBiLSTM(nn.Module):
    def __init__(
        self,
        input_size:  int = INPUT_SIZE,
        hidden_size: int = 192,
        num_layers:  int = 2,
        fc_hidden:   int = 128,
        dropout:     float = 0.3,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # BiLSTM outputs hidden_size * 2 = 384
        lstm_out = hidden_size * 2

        self.output_norm = nn.LayerNorm(lstm_out)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 150)
        Returns:
            logits: (B, num_classes)
        """
        x = self.input_norm(x)               # (B, T, 150)
        out, _ = self.lstm(x)                # (B, T, 384)
        out = self.output_norm(out)          # LayerNorm stabilizes training
        out = self.dropout(out)
        pooled = out.mean(dim=1)             # (B, 384) — global avg pooling over time
        return self.classifier(pooled)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    model = SignBiLSTM()
    x = torch.randn(8, N_FRAMES, INPUT_SIZE)
    out = model(x)
    print(f"[BiLSTM] Input: {x.shape} -> Output: {out.shape}")
    print(f"[BiLSTM] Parameters: {model.count_params():,}")

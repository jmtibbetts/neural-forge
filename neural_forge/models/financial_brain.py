"""
NeuralForge - Financial Brain
LSTM + MLP hybrid for technical analysis signal generation.
Input  : sequence of OHLCV + indicators (window_size x features)
Output : BUY / HOLD / SELL probabilities + confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FinancialBrain(nn.Module):
    """
    Architecture:
        LSTM encoder  -> captures temporal patterns in candle sequences
        Attention     -> weights which timesteps matter most
        MLP head      -> maps encoded state to signal probabilities
    """

    def __init__(
        self,
        input_size: int = 12,       # number of features per timestep
        hidden_size: int = 128,     # LSTM hidden units
        num_layers: int = 2,        # stacked LSTM layers
        dropout: float = 0.3,
        num_classes: int = 3,       # BUY=0, HOLD=1, SELL=2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # --- LSTM encoder ---
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # --- Temporal attention ---
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # --- MLP classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "weight" in name and p.dim() >= 2:
                nn.init.kaiming_normal_(p)

    def forward(self, x):
        """
        x : (batch, seq_len, input_size)
        returns:
            logits      : (batch, num_classes)
            attn_weights: (batch, seq_len)  — for visualization
        """
        # LSTM
        lstm_out, _ = self.lstm(x)          # (B, T, H)

        # Attention over timesteps
        attn_scores  = self.attention(lstm_out)         # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)# (B, T, 1)
        context      = (attn_weights * lstm_out).sum(dim=1)  # (B, H)

        # Classify
        logits = self.classifier(context)   # (B, 3)

        return logits, attn_weights.squeeze(-1)

    def predict(self, x):
        """Single inference pass — returns label + confidence."""
        self.eval()
        with torch.no_grad():
            logits, attn = self.forward(x)
            probs  = F.softmax(logits, dim=-1)
            conf, label = probs.max(dim=-1)
        return label, conf, probs, attn

    @property
    def label_map(self):
        return {0: "BUY", 1: "HOLD", 2: "SELL"}

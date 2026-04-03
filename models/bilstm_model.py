"""
models/bilstm_model.py
-----------------------
Bidirectional LSTM (BiLSTM) classifier for sequence-level anomaly detection.

Architecture:
    Input  → BiLSTM (hidden_size, num_layers, bidirectional=True)
           → Dropout
           → Linear(hidden_size*2, 64) + ReLU
           → Linear(64, 1) + Sigmoid
    Output: anomaly probability per sequence (scalar in [0, 1])
"""

from __future__ import annotations

import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger("BiLSTMClassifier")


class BiLSTMClassifier(nn.Module):
    """Binary anomaly classifier built on a bi-directional LSTM.

    Args:
        input_size:  Number of input features per time-step.
        hidden_size: Number of LSTM units per direction.
        num_layers:  Number of stacked BiLSTM layers.
        dropout:     Dropout probability applied between LSTM outputs and FC.
    """

    def __init__(
        self,
        input_size:  int   = 162,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(p=dropout)

        # Classifier head
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()
        logger.info(
            "BiLSTMClassifier created — input=%d, hidden=%d×2, layers=%d",
            input_size, hidden_size, num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass – returns sigmoid anomaly probability.

        Args:
            x: Tensor of shape (batch, seq_len, input_size).

        Returns:
            Tensor of shape (batch, 1) – anomaly probability in [0, 1].
        """
        out, _ = self.bilstm(x)            # (B, T, H*2)
        out = out[:, -1, :]                # (B, H*2) – last time-step
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))     # (B, 64)
        out = self.sigmoid(self.fc2(out))  # (B, 1)
        return out

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass – returns raw logits (no sigmoid).

        Use this with BCEWithLogitsLoss during training for numerical stability.

        Args:
            x: Tensor of shape (batch, seq_len, input_size).

        Returns:
            Tensor of shape (batch, 1) – raw logits.
        """
        out, _ = self.bilstm(x)         # (B, T, H*2)
        out = out[:, -1, :]             # (B, H*2)
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))  # (B, 64)
        out = self.fc2(out)             # (B, 1) – raw logits
        return out

    def _init_weights(self):
        """Xavier initialisation for linear layers."""
        for name, p in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif "fc" in name and "weight" in name:
                nn.init.xavier_uniform_(p)

    def save(self, path: str):
        """Save model state dict to *path*."""
        torch.save(self.state_dict(), path)
        logger.info("Model saved -> %s", path)

    @classmethod
    def load(cls, path: str, config) -> "BiLSTMClassifier":
        """Load a previously saved checkpoint."""
        model = cls(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        logger.info("Model loaded <- %s", path)
        return model

    def parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

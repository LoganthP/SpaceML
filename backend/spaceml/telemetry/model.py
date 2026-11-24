from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .detection import WINDOW


class LSTMForecaster(nn.Module):
    """Two stacked LSTM layers (80 units, dropout 0.3) -> next telemetry value."""

    def __init__(self, n_inputs: int, hidden: int = 80, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.n_inputs = n_inputs
        self.lstm = nn.LSTM(n_inputs, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, F)
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


def make_windows(x: np.ndarray, window: int = WINDOW) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (N, window, F) inputs and (N,) next-step telemetry targets as zero-copy views."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    X = t.unfold(0, window, 1).permute(0, 2, 1)[:-1]  # windows ending before each target
    y = t[window:, 0]
    return X, y


@torch.no_grad()
def predict(model: LSTMForecaster, x: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    X, _ = make_windows(x)
    outs = [model(X[i:i + batch].contiguous()) for i in range(0, len(X), batch)]
    return torch.cat(outs).numpy() if outs else np.zeros(0, dtype=np.float32)


def save_checkpoint(model: LSTMForecaster, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_inputs": model.n_inputs, **meta}, path)


def load_checkpoint(path: Path) -> tuple[LSTMForecaster, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LSTMForecaster(ckpt["n_inputs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

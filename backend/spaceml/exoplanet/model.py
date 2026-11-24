"""Two-branch 1D CNN in the style of Shallue & Vanderburg (2018), sized for a laptop."""
from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int, k: int, pool: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.ReLU(),
        nn.Conv1d(c_out, c_out, k, padding=k // 2), nn.ReLU(),
        nn.MaxPool1d(pool, stride=2),
    )


class TwoBranchCNN(nn.Module):
    def __init__(self, global_len: int = 2001, local_len: int = 201, dropout: float = 0.3):
        super().__init__()
        self.global_branch = nn.Sequential(
            _block(1, 16, 5, 5), _block(16, 32, 5, 5), _block(32, 64, 5, 5),
            _block(64, 128, 5, 5), _block(128, 256, 5, 5))
        self.local_branch = nn.Sequential(_block(1, 16, 5, 7), _block(16, 32, 5, 7))
        with torch.no_grad():
            n = (self.global_branch(torch.zeros(1, 1, global_len)).numel()
                 + self.local_branch(torch.zeros(1, 1, local_len)).numel())
        self.head = nn.Sequential(
            nn.Linear(n, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1))

    def forward(self, g: torch.Tensor, l: torch.Tensor) -> torch.Tensor:  # (B, 2001), (B, 201)
        a = self.global_branch(g.unsqueeze(1)).flatten(1)
        b = self.local_branch(l.unsqueeze(1)).flatten(1)
        return self.head(torch.cat([a, b], 1)).squeeze(-1)  # logits

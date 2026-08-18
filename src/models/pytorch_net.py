import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    """
    Один скрытый слой с BatchNorm, Dropout и L2-регуляризацией (weight_decay в оптимизаторе).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
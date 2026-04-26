import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    """
    Linear -> BatchNorm -> ReLU -> Dropout -> Linear
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < num_layers:
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
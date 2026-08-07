import torch
import torch.nn as nn
from typing import *

class MLP(nn.Module):
    def __init__(self, channels: int, inner_channels: int, channels_out: Optional[int] = None, mlp_layer_num: int = 2):
        super().__init__()
        layers = []
        for i in range(mlp_layer_num - 1):
            layers.append(nn.Linear(channels if i == 0 else inner_channels, inner_channels))
            layers.append(nn.GELU(approximate="tanh"))
        layers.append(nn.Linear(inner_channels, channels if channels_out is None else channels_out))    
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
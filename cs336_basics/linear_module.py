import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from typing import Any
from torchtyping import TensorType
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features : int, out_features : int, device : torch.device | None = None, dtype : torch.dtype | None = None) -> None:
        super().__init__()
        std = torch.sqrt(torch.tensor(2/(in_features + out_features)))
        weights = torch.randn(out_features, in_features, dtype=dtype, device=device) * std.clamp(-3 * std, 3 * std)
        self.weights : TensorType["out_dim in_dim", float] = Parameter(weights)

    def forward(self, x : TensorType["*batch in_dim"]) -> TensorType["*batch out_dim"]:
        return einsum(x, self.weights, "... in_dim, out_dim in_dim -> ... out_dim")    
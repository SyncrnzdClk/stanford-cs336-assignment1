import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum
from .utils import SiLU
from .linear import Linear
class SwiGLU(nn.Module):
    def __init__(self, 
                 d_model : int, 
                 d_ff : int | None = None,
                 device : torch.device | None = None,
                 dtype : torch.dtype | None = None
                 ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = (round(8 * d_model / 3.0)/64) * 64
            
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        
    def forward(self,
                x : TensorType["... d_model", float]
                ) -> TensorType["... d_model", float]:
        y : TensorType["... d_ff"] = SiLU(self.w1(x)) * self.w3(x)
        return  self.w2(y)
    
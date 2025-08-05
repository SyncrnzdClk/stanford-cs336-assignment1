import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum
from .utils import SiLU

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
            
        std = torch.sqrt(torch.tensor(2/(d_ff+d_model)))
        w1_weight = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        nn.init.trunc_normal_(w1_weight, 0, std, a=-3, b=3)
        w2_weight = torch.empty(d_model, d_ff, device=device, dtype=dtype)
        nn.init.trunc_normal_(w2_weight, 0, std, a=-3, b=3)
        w3_weight = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        nn.init.trunc_normal_(w3_weight, 0, std, a=-3, b=3)
        self.w1_weight : TensorType["d_ff d_model", float] = Parameter(w1_weight)
        self.w2_weight : TensorType["d_model d_ff", float] = Parameter(w2_weight)
        self.w3_weight : TensorType["d_ff d_model", float] = Parameter(w3_weight)
        
    def forward(self,
                x : TensorType["... d_model", float]
                ) -> TensorType["... d_model", float]:
        y : TensorType["... d_ff"] = SiLU(einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")) * einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.w2_weight, y, "d_model d_ff, ... d_ff -> ... d_model")
    
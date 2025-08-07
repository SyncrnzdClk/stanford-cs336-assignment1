import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType

class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model : int,
                 eps : float = 1e-5, 
                 device : torch.device | None = None,
                 dtype : torch.dtype | None = None
                 ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = torch.tensor(eps).to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(torch.ones(d_model, dtype=dtype))
        
    def forward(self,
                x : TensorType["batch_size sequence_length d_model", float]
                ) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        
        rms = torch.sqrt(torch.sum(torch.square(x), dim=-1)/self.d_model + self.eps).unsqueeze(-1)
        rms_norm = self.weight * x / rms
        
        return rms_norm.to(in_dtype)
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum, rearrange

class RoPE(nn.Module):
    R_buffer : TensorType["max_seq_len d_k_half", float]
    
    def __init__(self,
                 theta : float,
                 d_k : int,
                 max_seq_len : int,
                 device : torch.device | None = None
                 ) -> None:
        super().__init__()
        # calculate Rotation matrix buffer
        dim_idx = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        theta_k = 1.0 / (theta ** (dim_idx / d_k))
        pos = torch.arange(0, max_seq_len, 1, dtype=torch.float32, device=device)
        angles = einsum(pos, theta_k, "max_seq_len, d_k_half -> max_seq_len d_k_half")
        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)
        R_buffer : TensorType["max_seq_len d_k_half dim1 dim2", float] = rearrange(torch.stack([cos_values, -sin_values, sin_values, cos_values], dim=-1), "max_seq_len d_k_half (dim1 dim2) -> max_seq_len d_k_half dim1 dim2", dim1=2, dim2=2)
        self.register_buffer("R_buffer", R_buffer, persistent=False)
        
    def forward(self, 
                x : TensorType["... seq_len d_k", float],
                token_positions : TensorType["... seq_len", float]
                ) -> TensorType["... seq_len d_k", float]:
        # slice the R_buffer
        R_buffer_batch : TensorType["... seq_len d_k_half dim1 dim2", float] = self.R_buffer[token_positions]
        x_rearranged = rearrange(x, "... seq_len (d_k_half dim) -> ... seq_len d_k_half dim", dim=2)
        return rearrange(einsum(R_buffer_batch, x_rearranged, "... seq_len d_k_half dim1 dim, ... seq_len d_k_half dim -> ... seq_len d_k_half dim1"), "... seq_len d_k_half dim1 -> ... seq_len (d_k_half dim1)")
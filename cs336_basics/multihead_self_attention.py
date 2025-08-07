import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum, reduce, rearrange
from .utils import scaled_dot_product_attention
from .rotary_positional_embedding import RoPE
from .linear import Linear
from math import sqrt

class MultiHeadSelfAttn(nn.Module):
    def __init__(self,
                 d_model : int, 
                 num_heads : int
                 ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model is not a multiple of num_heads"
        self.heads : int = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
    
    def forward(self, inputs : TensorType["... seq_len d_model", float], rope : RoPE | None = None, token_positions : TensorType["... seq_len", int] | None = None) -> TensorType["... seq_len d_model", float]:
        # perform multihead
        q_proj_res = self.q_proj(inputs) 
        k_proj_res = self.k_proj(inputs) 
        v_proj_res = self.v_proj(inputs) 
        q_proj_res : TensorType["... h seq_len d_k", float] = rearrange(q_proj_res, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.heads)
        k_proj_res : TensorType["... h seq_len d_k", float] = rearrange(k_proj_res, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.heads)
        if rope is not None and token_positions is not None:
            q_proj_res = rope(q_proj_res, token_positions)
            k_proj_res = rope(k_proj_res, token_positions)
        v_proj_res : TensorType["... h seq_len d_v", float] = rearrange(v_proj_res, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.heads)
        mask = torch.tril(torch.ones(q_proj_res.shape[-2], k_proj_res.shape[-2]), diagonal=0).bool()
        multihead : TensorType["... seq_len hd_v"] = rearrange(scaled_dot_product_attention(q_proj_res, k_proj_res, v_proj_res, mask), "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.output_proj(multihead)
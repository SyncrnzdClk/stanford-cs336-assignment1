import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum, reduce, rearrange
from .utils import scaled_dot_product_attention
from .rotary_positional_embedding import RoPE
from math import sqrt

class MultiHeadSelfAttn(nn.Module):
    def __init__(self,
                 d_model : int, 
                 num_heads : int
                 ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model is not a multiple of num_heads"
        self.heads : int = num_heads
        q_weights = nn.init.trunc_normal_(torch.empty(d_model, d_model), 0, sqrt(1/d_model), -3, 3)
        k_weights = nn.init.trunc_normal_(torch.empty(d_model, d_model), 0, sqrt(1/d_model), -3, 3)
        v_weights = nn.init.trunc_normal_(torch.empty(d_model, d_model), 0, sqrt(1/d_model), -3, 3)
        o_weights = nn.init.trunc_normal_(torch.empty(d_model, d_model), 0, sqrt(1/d_model), -3, 3)
        self.q_weights : TensorType["hd_k d_model", float] = Parameter(q_weights)
        self.k_weights : TensorType["hd_k d_model", float] = Parameter(k_weights)
        self.v_weights : TensorType["hd_v d_model", float] = Parameter(v_weights)
        self.o_weights : TensorType["d_model hd_v", float] = Parameter(o_weights)
    
    def forward(self, inputs : TensorType["... seq_len d_model", float], rope : RoPE | None = None, token_positions : TensorType["... seq_len", int] | None = None) -> TensorType["... seq_len d_model", float]:
        # perform multihead
        q_proj_res = einsum(self.q_weights, inputs, "hd_k d_model, ... seq_len d_model -> ... seq_len hd_k")
        k_proj_res = einsum(self.k_weights, inputs, "hd_k d_model, ... seq_len d_model -> ... seq_len hd_k")
        v_proj_res = einsum(self.v_weights, inputs, "hd_v d_model, ... seq_len d_model -> ... seq_len hd_v")
        q_proj_res : TensorType["... h seq_len d_k", float] = rearrange(q_proj_res, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.heads)
        k_proj_res : TensorType["... h seq_len d_k", float] = rearrange(k_proj_res, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.heads)
        if rope is not None and token_positions is not None:
            q_proj_res = rope(q_proj_res, token_positions)
            k_proj_res = rope(k_proj_res, token_positions)
        v_proj_res : TensorType["... h seq_len d_v", float] = rearrange(v_proj_res, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.heads)
        mask = torch.tril(torch.ones(q_proj_res.shape[-2], k_proj_res.shape[-2]), diagonal=0).bool()
        multihead : TensorType["... seq_len hd_v"] = rearrange(scaled_dot_product_attention(q_proj_res, k_proj_res, v_proj_res, mask), "... h seq_len d_v -> ... seq_len (h d_v)")
        return einsum(multihead, self.o_weights, "... seq_len hd_v, d_model hd_v -> ... seq_len d_model")
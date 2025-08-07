import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum, reduce
from math import sqrt
from .multihead_self_attention import MultiHeadSelfAttn
from .rmsnorm import RMSNorm
from .position_wise_feed_forward import SwiGLU
from .rotary_positional_embedding import RoPE

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model : int,
                 num_heads : int,
                 d_ff : int
                 ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttn(d_model, num_heads)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)
        
    def forward(self, x : TensorType["... seq_len d_model", float], rope : RoPE ) -> TensorType["... seq_len d_model", float]:
        # TODO: add rope operation
        y = x + self.attn(self.ln1(x))
        return y + self.ffn(self.ln2(y))
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum, reduce
from math import sqrt
from cs336_basics.embedding import Embedding
from cs336_basics.rotary_positional_embedding import RoPE
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.linear import Linear
from cs336_basics.utils import softmax

class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size : int,
                 context_length : int,
                 d_model : int,
                 num_layers : int,
                 num_heads : int,
                 d_ff : int,
                 rope_theta : float,
                 device: torch.device | None = None
                 ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.rope = RoPE(rope_theta, d_model//num_heads, context_length)
        self.device = device
        
    def forward(self,
                in_indices : TensorType["batch_size seq_len", int],
                ) -> TensorType["batch_size seq_len vocab_size", float]:
        assert in_indices.shape[-1] <= self.context_length, "sequence length exceeds the maximum context length"
        x : TensorType["batch_size seq_len d_model", float] = self.token_embeddings(in_indices)
        token_positions = torch.arange(x.shape[1])
        for layer in self.layers:
            x = layer(x, self.rope, token_positions)
        x = self.ln_final(x)
        x : TensorType["batch_size seq_len vocab_size", float] = self.lm_head(x)
        return x # unnormalized next-token prediction distribution
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"total params: {total_params:,}")
        print(f"trainable params: {trainable_params:,}")
        print(f"model size: {total_params / 1e9:.2f}B")
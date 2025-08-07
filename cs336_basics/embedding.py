import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings : int, 
                 embedding_dim : int, 
                 device : torch.device | None = None,
                 dtype : torch.dtype | None = None
                 ) -> None:
        super().__init__()
        weights = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(weights, mean=0, std=1, a=-3, b=3)
        self.weight : TensorType["num_embeddings d_model", float] = Parameter(weights)
        
    def forward(self, token_ids : TensorType["batch_size sequence_length", float]) -> TensorType["batch_size sequence_length d_model", float]:
        return self.weight[token_ids]
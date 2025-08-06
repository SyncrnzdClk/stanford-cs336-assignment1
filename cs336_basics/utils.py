import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum, reduce
from math import sqrt

def SiLU(input : TensorType["... in_features", float]) -> TensorType["... in_features", float]:
    return input * torch.sigmoid(input=input)

def softmax(input : TensorType["...", float], dim : int) -> TensorType["...", float]:
    max_element, _ = input.max(dim=dim, keepdim=True)
    stablized_input = input - max_element
    stablized_input_exp = torch.exp(stablized_input)
    stablized_input_exp_sum = stablized_input_exp.sum(dim=dim, keepdim=True)
    return stablized_input_exp / stablized_input_exp_sum

def scaled_dot_product_attention(Q : TensorType["... queries d_k", float],
                                 K : TensorType["... keys d_k", float],
                                 V : TensorType["... values d_v", float],
                                 mask : TensorType["... queries keys", float] | None = None
                                 ) -> TensorType["... queries d_v"]:
    if mask is not None:
        mask = torch.where(mask, 0.0, float('-inf'))
        attention_weights = softmax(einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(Q.shape[-1]) + mask, -1)
    else:
        attention_weights = softmax(einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(Q.shape[-1]), -1)
    attention_scores = einsum(attention_weights, V, "... queries seq_len, ... seq_len d_v -> ... queries d_v")
    return attention_scores
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum, reduce

def SiLU(input : TensorType["... in_features", float]) -> TensorType["... in_features", float]:
    return input * torch.sigmoid(input=input)

def softmax(input : TensorType["...", float], dim : int) -> TensorType["...", float]:
    max_element, _ = input.max(dim=dim, keepdim=True)
    stablized_input = input - max_element
    stablized_input_exp = torch.exp(stablized_input)
    stablized_input_exp_sum = stablized_input_exp.sum(dim=dim, keepdim=True)
    return stablized_input_exp / stablized_input_exp_sum
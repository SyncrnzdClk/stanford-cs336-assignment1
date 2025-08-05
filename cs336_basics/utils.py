import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum

def SiLU(input : TensorType["... in_features", float]) -> TensorType["... in_features", float]:
    return input * torch.sigmoid(input=input)
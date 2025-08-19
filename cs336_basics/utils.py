import torch
import torch.nn as nn
from torch.nn import Parameter
from torchtyping import TensorType
from einops import einsum, reduce
from math import sqrt, cos, pi
from typing import Iterable
import numpy as np
import numpy.typing as npt
import os
import typing

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

def cross_entropy(inputs : TensorType["... vocab_size", float],
                  targets : TensorType["... ", float]
                  ) -> TensorType["", float]:
    # for numerical stability, we cannot directly apply the softmax function we defined before
    # we should perform some transformations to eliminate some exps and logs
    max_element, _ = inputs.max(dim=-1, keepdim=True)
    stablized_input = inputs - max_element
    stablized_input_exp = torch.exp(stablized_input)
    stablized_input_exp_sum : TensorType["...", float] = stablized_input_exp.sum(dim=-1)
    target_scores : TensorType["...", float] = torch.gather(stablized_input, dim=1, index=targets.unsqueeze(1)).squeeze(1)
    return -(target_scores - torch.log(stablized_input_exp_sum)).sum() / torch.tensor(inputs.shape[:-1]).prod()

def lr_cosine_schedule(t, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if t < warmup_iters:
        learning_rate = t / warmup_iters * max_learning_rate
    elif t >= warmup_iters and t <= cosine_cycle_iters:
        learning_rate = min_learning_rate + 1/2 * (1 + cos(pi*(t-warmup_iters)/(cosine_cycle_iters-warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        learning_rate = min_learning_rate
    return learning_rate        
        
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps : float = 1e-6):
    with torch.no_grad():
        total_grads = torch.cat([param.grad.reshape(-1) for param in parameters if param.grad is not None])
        total_norm = torch.norm(total_grads, p=2)
        if total_norm > max_l2_norm:
            scale = max_l2_norm / (total_norm + eps)
            for param in parameters:
                if param.grad is not None:
                    param.grad *= scale
                    
def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    indices = np.random.randint(len(dataset) - context_length, size=batch_size)
    indices = indices[:, None] + np.arange(context_length)
    sample_input = dataset[indices]
    next_token_targets = dataset[indices+1]
    return torch.tensor(sample_input, device=device), torch.tensor(next_token_targets, device=device)

def save_checkpoint(
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    iteration : int,
    out : str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src : str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model : torch.nn.Module, 
                    optimizer : torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
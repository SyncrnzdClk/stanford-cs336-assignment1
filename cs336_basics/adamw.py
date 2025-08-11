import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math
from einops import einsum

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas" : betas, "eps": eps, "weight_decay" : weight_decay}
        super().__init__(params, defaults)
        
    def step(self, closure : Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta_1, beta_2 = group['betas']
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
        for p in group['params']:
            if p.grad is None:
                continue
            
            state = self.state[p]
            t = state.get("t", 1)
            m = state.get("m", torch.zeros_like(p.data))
            v = state.get("v", torch.zeros_like(p.data))
            grad = p.grad.data
            m = beta_1 * m + (1 - beta_1) * grad
            v = beta_2 * v + (1 - beta_2) * grad.pow(2)
            alpha_t = lr * math.sqrt(1-math.pow(beta_2, t)) / (1 - math.pow(beta_1, t))
            p.data -= alpha_t * m / (v.sqrt() + eps)
            p.data -= lr * weight_decay * p.data
            state['t'] = t + 1
            state['m'] = m
            state['v'] = v
        return loss
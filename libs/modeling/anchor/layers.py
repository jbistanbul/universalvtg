"""
Shared utility layers for the anchor module.

Contains reusable components like normalization and regularization layers.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Equivalent to T5LayerNorm and LlamaRMSNorm.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerScale(nn.Module):
    """
    Layer scaling with stochastic depth for residual connections.
    
    Multiplies residual by a per-channel scaling factor (zero-init style)
    before adding, combined with stochastic depth during training.
    
    Reference: https://arxiv.org/abs/2103.17239
    
    Args:
        n_channels: Number of channels/features
        pdrop: Drop path probability
        init_scale: Initial scale factor
    """
    def __init__(self, n_channels: int, pdrop: float = 0.1, init_scale: float = 1e-4):
        super().__init__()
        self.scale = nn.Parameter(init_scale * torch.ones((1, 1, n_channels)))
        self.pdrop = pdrop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(self.scale.to(x.dtype) * x, self.pdrop, self.training)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic Depth per sample.
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping a path
        training: Whether in training mode
        
    Returns:
        Output after applying stochastic depth
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    x = x.div(keep_prob) * mask.floor_()
    return x

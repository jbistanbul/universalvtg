"""
Pooling strategies for anchor token generation.

This module provides various pooling methods for extracting anchor tokens
from sequence blocks in the HieraMamba architecture.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorPooling(nn.Module):
    """
    Configurable pooling module for extracting anchor tokens from sequence blocks.
    
    Supports multiple pooling strategies:
    - 'mean': Average pooling (default, fastest)
    - 'max': Max pooling (good for sparse features)
    - 'attn': Attention pooling with learnable query (CLIP-style)
    - 'gated': Gated pooling that adaptively combines mean and max
    
    Args:
        stride: The stride used for downsampling (tokens per anchor block)
        method: Pooling method ('mean', 'max', 'attn', or 'gated')
        d_model: Embedding dimension (required for 'attn' and 'gated')
        nhead: Number of attention heads (for 'attn' method)
        dropout: Dropout probability for attention
    """
    def __init__(self,
                 stride: int,
                 method: str = "mean",
                 d_model: int = 0,
                 nhead: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.stride = stride
        self.method = method

        if method == "mean":
            self.pooler = MeanPooling(stride)
        elif method == "max":
            self.pooler = MaxPooling(stride)
        elif method == "attn":
            if d_model <= 0:
                raise ValueError("d_model must be provided for attention pooling")
            self.pooler = AttnPooling(stride, d_model, nhead, dropout)
        elif method == "gated":
            if d_model <= 0:
                raise ValueError("d_model must be provided for gated pooling")
            self.pooler = GatedPooling(stride, d_model)
        else:
            raise ValueError(f"Unknown pooling method {method!r}")

    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_blk: Input tensor of shape (B, D, L) or (B, D, num_blocks, stride)
        Returns:
            Pooled anchor tokens of shape (B, D, num_blocks)
        """
        return self.pooler(x_blk)


class MeanPooling(nn.Module):
    """Mean pooling along the temporal dimension."""
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
        
    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(x_blk, kernel_size=self.stride, stride=self.stride)


class MaxPooling(nn.Module):
    """Max pooling along the temporal dimension."""
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride
        
    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        return F.max_pool1d(x_blk, kernel_size=self.stride, stride=self.stride)


class GatedPooling(nn.Module):
    """
    Gated pooling that adaptively combines mean and max pooling.
    
    Learns channel-wise gates to dynamically blend mean and max pooling
    based on the input features.
    """
    def __init__(self, stride: int, d_model: int):
        super().__init__()
        self.stride = stride
        self.gate_proj = nn.Conv1d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        μ = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)
        m = F.max_pool1d(x, kernel_size=self.stride, stride=self.stride)
        cat = torch.cat([μ, m], dim=1)
        g = torch.sigmoid(self.gate_proj(cat))
        return g * m + (1.0 - g) * μ


class AttnPooling(nn.Module):
    """
    CLIP-style attention pooling with a learnable query vector.
    
    A single learnable query attends to all tokens in a sequence block
    to produce a pooled representation.
    """
    def __init__(self, stride: int, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.stride = stride
        self.pool_q = nn.Parameter(torch.randn(1, 1, d_model) / (d_model ** 0.5))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        B, D, L2 = x_blk.shape
        num_blocks = L2 // self.stride
        x_blk = x_blk.reshape(B, D, num_blocks, self.stride)
        kv = (
            x_blk
            .permute(3, 0, 2, 1)
            .reshape(self.stride, B * num_blocks, D)
        )
        q = self.pool_q.expand(-1, B * num_blocks, -1)
        out, _ = self.pool_attn(q, kv, kv)
        return (
            out
            .squeeze(0)
            .view(B, num_blocks, D)
            .permute(0, 2, 1)
        )


def downsample_mask(mask: torch.Tensor, stride: int = 2) -> torch.Tensor:
    """
    Downsample a boolean mask using max pooling with ceiling rounding.
    
    Args:
        mask: Boolean mask of shape (B, 1, L)
        stride: Downsampling factor
        
    Returns:
        Downsampled mask of shape (B, 1, ceil(L/stride))
    """
    mask_float = mask.float()
    downsampled = F.max_pool1d(mask_float, kernel_size=stride, stride=stride, ceil_mode=True)
    return downsampled.bool()

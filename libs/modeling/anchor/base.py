"""
Base classes for anchor-based sequence modeling blocks.

This module provides the foundational BaseAnchorBlock class that handles
anchor generation, interleaving, and extraction operations used by all
anchor block variants.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pooling import AnchorPooling, downsample_mask


class BaseAnchorBlock(nn.Module):
    """
    Base class for all anchor blocks containing common anchor generation operations.
    
    Provides:
    - Efficient anchor position caching
    - Robust anchor generation and interleaving
    - Consistent mask creation and sequence extraction
    - Configurable pooling methods for anchor generation
    
    Supported pooling methods:
    - "mean": Average pooling (default)
    - "max": Max pooling
    - "attn": Attention pooling (CLIP-style)
    - "gated": Adaptive mean/max blending
    
    All derived anchor blocks should inherit from this class.
    """
    def __init__(self, stride: int, d_model: int, pool_method: str = "mean", dropout: float = 0.1):
        """
        Args:
            stride: Stride for downsampling (tokens per anchor block)
            d_model: Model embedding dimension
            pool_method: Pooling method ('mean', 'max', 'attn', 'gated')
            dropout: Dropout probability
        """
        super().__init__()
        self.stride = stride
        self.d_model = d_model
        
        self.anchor_pooling = AnchorPooling(
            stride=stride, 
            method=pool_method, 
            d_model=d_model, 
            nhead=1, 
            dropout=dropout
        )
        
        # Cache for anchor positions to avoid recomputation
        self._anchor_positions_cache = {}
    
    def _get_anchor_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Cache and retrieve anchor positions."""
        cache_key = (seq_len, str(device))
        if cache_key not in self._anchor_positions_cache:
            self._anchor_positions_cache[cache_key] = torch.arange(
                0, seq_len, self.stride + 1, device=device
            )
        return self._anchor_positions_cache[cache_key]

    def _generate_and_interleave_anchors(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized method to interleave anchors with sequence tokens.
        
        Creates interleaved sequence: [anchor0, tok0, tok1, anchor1, tok2, tok3, ...]

        Args:
            x: Input tensor of shape (B, D, L)
            mask: Input mask of shape (B, 1, L)

        Returns:
            Tuple of:
            - x_combined: Interleaved anchors and sequences (B, combined_seq_len, D)
            - anchor_positions: Positions of anchors in combined sequence
            - expanded_mask: Mask for combined sequence (B, combined_seq_len, 1)
            - anchor_mask: Downsampled mask for anchors (B, 1, num_blocks)
        """
        B, D, L = x.shape
        s = self.stride
        device = x.device

        # Calculate padding for stride alignment
        num_blocks = (L + s - 1) // s
        needed_len = num_blocks * s
        pad_len = needed_len - L

        if pad_len > 0:
            x_padded = F.pad(x, (0, pad_len))
        else:
            x_padded = x

        # Reshape into blocks
        x_blocks = x_padded.reshape(B, D, num_blocks, s)

        # Generate anchors
        anchors = self.anchor_pooling(x_padded)
        
        # Vectorized interleaving
        anchors_expanded = anchors.unsqueeze(-1)  # (B, D, num_blocks, 1)
        mixed = torch.cat((anchors_expanded, x_blocks), dim=-1)  # (B, D, num_blocks, s+1)

        # Flatten to sequence
        x_combined = mixed.permute(0, 2, 3, 1).reshape(B, -1, D)

        # Get anchor positions
        anchor_positions = self._get_anchor_positions(x_combined.size(1), device)

        # Create masks
        anchor_mask = downsample_mask(mask, s)

        nonmasked_seq_len = int(mask.sum().item())
        nonmasked_anchor_len = int(anchor_mask.sum().item())
        nonmasked_len = nonmasked_seq_len + nonmasked_anchor_len

        expanded_mask = torch.zeros((B, x_combined.size(1), 1), dtype=torch.bool, device=device)
        expanded_mask[:, :nonmasked_len] = True

        return x_combined, anchor_positions, expanded_mask, anchor_mask

    def _extract_anchor_and_sequence_outputs(
        self,
        processed_features: torch.Tensor,
        anchor_positions: torch.Tensor,
        original_seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract anchors and sequence tokens from processed combined features.
        
        Args:
            processed_features: Processed features (B, combined_seq_len, D)
            anchor_positions: Positions of anchors
            original_seq_len: Original sequence length L
            
        Returns:
            Tuple of:
            - anchor_out: Anchor features (B, D, num_blocks)
            - seq_out: Sequence features (B, D, L)
        """
        B, combined_len, D = processed_features.shape
        s = self.stride
        step = s + 1

        # Reshape for efficient extraction
        num_blocks = combined_len // step
        mixed = processed_features.view(B, num_blocks, step, D)

        # Anchors at position 0 in every block
        anchor_out = mixed[:, :, 0, :].transpose(1, 2)

        # Sequence tokens at positions 1..s
        seq_tokens = mixed[:, :, 1:, :].reshape(B, num_blocks * s, D)
        seq_out = seq_tokens.transpose(1, 2)[..., :original_seq_len]

        return anchor_out, seq_out

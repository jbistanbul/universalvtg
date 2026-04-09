"""
Anchor-based sequence modeling module for HieraMamba.

This package provides anchor block implementations that combine global
sequence modeling (via Mamba/Hydra) with optional local attention.

Main components:
- AnchorMambaPoolingBlock: Standard global-local encoder
- AnchorMambaPoolingBlockGated: Gated fusion variant
- AnchorPooling: Configurable pooling strategies
"""

from .base import BaseAnchorBlock
from .blocks import (
    AnchorMambaPoolingBlock,
    AnchorMambaPoolingBlockGated,
    EnhancedAnchorBlock_Refined,
)
from .pooling import (
    AnchorPooling,
    MeanPooling,
    MaxPooling,
    GatedPooling,
    AttnPooling,
    downsample_mask,
)
from .layers import (
    RMSNorm,
    LayerScale,
    drop_path,
)

__all__ = [
    # Base
    "BaseAnchorBlock",
    # Blocks
    "AnchorMambaPoolingBlock", 
    "AnchorMambaPoolingBlockGated",
    "EnhancedAnchorBlock_Refined",
    # Pooling
    "AnchorPooling",
    "MeanPooling",
    "MaxPooling", 
    "GatedPooling",
    "AttnPooling",
    "downsample_mask",
    # Layers
    "RMSNorm",
    "LayerScale",
    "drop_path",
]

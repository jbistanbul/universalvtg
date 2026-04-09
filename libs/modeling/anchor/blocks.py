"""
Anchor block variants for hierarchical sequence modeling.

This module provides different anchor block implementations that inherit
from BaseAnchorBlock and use various encoder configurations.
"""

from typing import Tuple, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from hydra.modules.hydra import Hydra
from mamba_ssm import Mamba2

from .base import BaseAnchorBlock
from .layers import RMSNorm, LayerScale
from ..blocks import TransformerEncoder, SwiGLUFFN

# Set triton precision for older GPUs
if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    gpu_capability = torch.cuda.get_device_capability(gpu_id)
else:
    gpu_capability = None
if gpu_capability is not None and gpu_capability[0] < 8:
    os.environ["TRITON_F32_DEFAULT"] = "ieee"


class AnchorMambaPoolingBlock(BaseAnchorBlock):
    """
    Two-stage global-local encoder block for sequence modeling.
    
    Uses Hydra (bidirectional Mamba) for global encoding with optional
    local transformer encoder for fine-grained local attention.

    Input:
        x: Tensor of shape (B, D, L)
        mask: Tensor of shape (B, 1, L), optional boolean mask

    Output:
        anchor_out: (B, D, ceil(L/stride)) - downsampled anchors
        seq_out: (B, D, L) - same length as input
        anchor_mask: downsampled version of input mask
        mask: original input mask
    """
    def __init__(
        self, 
        stride: int, 
        d_model: int, 
        nhead: int = 4, 
        local_window_size: int = 5,
        dropout: float = 0.1, 
        ffn_ratio: int = 4,
        use_swiglu: bool = True,
        local_encode: bool = False, 
        pool_method: str = "mean",
        mamba_headdim: int = 64, 
        mamba_dstate: int = 64,
        mamba_expand: int = 2,
        mamba_dconv: int = 7,
        bidirectional: bool = True
    ):
        super().__init__(stride=stride, d_model=d_model, pool_method=pool_method, dropout=dropout)
        
        self.local_encode = local_encode

        # Global encoder (Hydra for bidirectional, Mamba2 for unidirectional)
        if bidirectional:
            self.global_encoder = Hydra(
                d_model=d_model, d_state=mamba_dstate, d_conv=mamba_dconv, 
                expand=mamba_expand, use_mem_eff_path=True, headdim=mamba_headdim
            )
        else:
            self.global_encoder = Mamba2(
                d_model=d_model, d_state=mamba_dstate, d_conv=4, 
                expand=mamba_expand, headdim=48
            )
            
        self.norm_global_in = RMSNorm(d_model)
        self.drop_path_global = LayerScale(d_model, dropout)

        self.norm_ffn_in = RMSNorm(d_model)
        self.drop_path_ffn = LayerScale(d_model, dropout)

        if local_encode:
            self.norm_local_in = RMSNorm(d_model)
            self.drop_path_local = LayerScale(d_model, dropout)
            self.local_encoder = TransformerEncoder(
                d_model, stride=1, window_size=local_window_size, n_heads=2
            )
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.ffn = SwiGLUFFN(d_model, dropout=dropout) if use_swiglu else nn.Sequential(
                nn.Linear(d_model, ffn_ratio * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_ratio * d_model, d_model),
                nn.Dropout(dropout)
            )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, L = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones((B, 1, L), dtype=torch.bool, device=device)

        x.masked_fill_(~mask, 0.)

        # Interleave anchors
        x_combined, anchor_pos, expanded_mask, anchor_mask = \
            self._generate_and_interleave_anchors(x, mask)

        # Global encoder with prenorm + residual
        g = self.norm_global_in(x_combined)
        g = self.global_encoder(g)
        g = g + self.drop_path_global(x_combined)
        g.masked_fill_(~expanded_mask, 0.)

        # Optional local encoder
        if self.local_encode:
            l = self.norm_local_in(g)
            l_t = l.transpose(1, 2).contiguous()
            encoder_mask = expanded_mask.transpose(1, 2)
            l_t = self.local_encoder(l_t, encoder_mask)[0]
            l = l_t.transpose(1, 2)
            g = g + self.drop_path_local(l)

        # FFN
        z = self.norm_ffn_in(g)
        z = self.ffn(z)
        z = g + self.drop_path_ffn(z)

        # Extract outputs
        anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(z, anchor_pos, L)
        return anchor_out, seq_out, anchor_mask, mask


class AnchorMambaPoolingBlockGated(BaseAnchorBlock):
    """
    Anchor Mamba Pooling Block with Gated Fusion.
    
    Key Features:
    1. Hierarchical gated fusion (gate1 always, gate2 with local encoder)
    2. Optional local encoder
    3. Final FFN always present
    4. Strong residual connections
    
    Architecture:
    When local_encode=True:
        input → global → gate1 → local → gate2 → ffn → output
    When local_encode=False:
        input → global → gate1 → ffn → output
    """
    def __init__(
        self, 
        stride: int = 2, 
        d_model: int = 384, 
        nhead: int = 4, 
        local_window_size: int = 5,
        dropout: float = 0.1, 
        ffn_ratio: int = 4,
        local_encode: bool = False,
        pool_method: str = "mean",
        local_encoder_type: str = "transformer",
        mamba_headdim: int = 64, 
        mamba_dstate: int = 64,
        mamba_expand: int = 2,
        mamba_dconv: int = 7,
        bidirectional: bool = True
    ):
        super().__init__(stride=stride, d_model=d_model, pool_method=pool_method, dropout=dropout)
        
        self.local_encode = local_encode
        
        # Global encoder
        if bidirectional:
            self.global_encoder = Hydra(
                d_model=d_model, d_state=mamba_dstate, d_conv=mamba_dconv,
                expand=mamba_expand, use_mem_eff_path=False, headdim=mamba_headdim
            )
        else:
            self.global_encoder = Mamba2(
                d_model=d_model, d_state=mamba_dstate, d_conv=4,
                expand=mamba_expand, headdim=48
            )
            
        self.norm_global = RMSNorm(d_model)
        self.drop_path_global = LayerScale(d_model, dropout)

        # First gate (always present)
        self.gate1 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        # Optional local encoder components
        if local_encode:
            self.gate2 = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.Sigmoid()
            )
            self.norm_local = RMSNorm(d_model)
            self.drop_path_local = LayerScale(d_model, dropout)
            self.local_encoder = TransformerEncoder(
                d_model, stride=1, window_size=local_window_size, n_heads=2
            )
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.ffn = SwiGLUFFN(d_model, dropout=dropout)

        self.norm_ffn = RMSNorm(d_model)
        self.drop_path_ffn = LayerScale(d_model, dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, L = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones((B, 1, L), dtype=torch.bool, device=device)

        x.masked_fill_(~mask, 0.)

        # Generate and interleave anchors
        x_combined, anchor_positions, expanded_mask, anchor_mask = \
            self._generate_and_interleave_anchors(x, mask)

        # Global encoding with residual
        global_out = self.global_encoder(self.norm_global(x_combined))
        global_out = self.drop_path_global(global_out) + x_combined
        global_out.mul_(expanded_mask)

        # First gated fusion
        gate1_weights = self.gate1(torch.cat([x_combined, global_out], dim=-1))
        fusion1_out = gate1_weights * global_out + (1 - gate1_weights) * x_combined

        # Optional local encoding and second fusion
        if self.local_encode:
            local_out = self.local_encoder(
                self.norm_local(fusion1_out).transpose(1, 2),
                expanded_mask.transpose(1, 2)
            )[0].transpose(1, 2)
            local_out = self.drop_path_local(local_out) + fusion1_out
            local_out.mul_(expanded_mask)

            gate2_weights = self.gate2(torch.cat([fusion1_out, local_out], dim=-1))
            fused = gate2_weights * local_out + (1 - gate2_weights) * fusion1_out
        else:
            fused = fusion1_out

        # Final FFN with residual
        ffn_out = self.ffn(self.norm_ffn(fused))
        final_out = self.drop_path_ffn(ffn_out) + fused

        # Extract outputs
        anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(
            final_out, anchor_positions, L
        )
        return anchor_out, seq_out, anchor_mask, mask


class EnhancedAnchorBlock_Refined(BaseAnchorBlock):
    """
    Enhanced Anchor Block - Refined Gating with Optional Enhancement
    
    Based on the insight that residual connections to gated features are critical,
    this block explores different ways to enhance gated features:
    1. Optional FFN refinement
    2. Lightweight enhancement layers
    3. Strong residual connections to preserve gated information
    
    Architecture:
    input → global → gate1 → local → gate2 → [optional_refinement] → output
                                     ↓                ↑
                                     └─── residual ───┘
    """
    def __init__(
        self, 
        stride: int, 
        d_model: int, 
        nhead: int = 4, 
        local_window_size: int = 19,
        dropout: float = 0.1, 
        ffn_ratio: int = 4,
        pool_method: str = "mean",
        local_encode: bool = True,
        local_encoder_type: str = "transformer",
        use_ffn_refinement: bool = True,
        refinement_type: str = "lightweight",  # Options: 'lightweight', 'full_ffn', 'conv_refinement'
        mamba_headdim: int = 64, 
        mamba_dstate: int = 64,
        mamba_expand: int = 2,
        mamba_dconv: int = 7,
        bidirectional: bool = True
    ):
        super().__init__(stride=stride, d_model=d_model, pool_method=pool_method, dropout=dropout)

        # Core processing components
        self.global_encoder = Hydra(
            d_model=d_model, d_state=mamba_dstate, d_conv=mamba_dconv, 
            expand=mamba_expand, use_mem_eff_path=True, headdim=mamba_headdim
        )
        
        self.local_encoder = TransformerEncoder(
            d_model, stride=1, window_size=local_window_size, n_heads=2
        )
        
        # Hierarchical gating mechanisms
        self.gate1 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        self.gate2 = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        
        # Refinement mechanism based on type
        self.use_ffn_refinement = use_ffn_refinement
        self.refinement_type = refinement_type
        
        if use_ffn_refinement:
            if refinement_type == "lightweight":
                self.refinement = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            elif refinement_type == "full_ffn":
                self.refinement = nn.Sequential(
                    nn.Linear(d_model, ffn_ratio * d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_ratio * d_model, d_model),
                    nn.Dropout(dropout)
                )
            elif refinement_type == "conv_refinement":
                self.refinement = nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(d_model, d_model, kernel_size=1),
                    nn.Dropout(dropout)
                )
        
        # Normalization layers
        self.norm_global = RMSNorm(d_model)
        self.norm_local = RMSNorm(d_model)
        if use_ffn_refinement:
            self.norm_refinement = RMSNorm(d_model)
        
        # Drop path layers
        self.drop_path_1 = LayerScale(d_model, dropout)
        self.drop_path_2 = LayerScale(d_model, dropout)
        if use_ffn_refinement:
            self.drop_path_3 = LayerScale(d_model, dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, L = x.shape
        device = x.device
        
        if mask is None:
            mask = torch.ones((B, 1, L), dtype=torch.bool, device=device)
        x = x * mask.to(x.dtype)

        # Use base class method for anchor generation and interleaving
        x_combined, anchor_positions, expanded_mask, anchor_mask = \
            self._generate_and_interleave_anchors(x, mask)
        
        # Store original input
        original_features = x_combined
        
        # Step 1: Global encoding
        global_norm = self.norm_global(x_combined)
        global_out = self.global_encoder(global_norm)
        global_out = x_combined + self.drop_path_1(global_out)
        global_out = global_out * expanded_mask
        
        # Step 2: First gated fusion (input + global)
        gate1_input = torch.cat([original_features, global_out], dim=-1)
        gate1_weights = self.gate1(gate1_input)
        fusion1_out = gate1_weights * global_out + (1 - gate1_weights) * original_features
        
        # Step 3: Local encoding (processes fusion1 result)
        local_norm = self.norm_local(fusion1_out)
        local_transposed = local_norm.transpose(1, 2)
        encoder_mask = expanded_mask.transpose(1, 2)
        local_out_transposed = self.local_encoder(local_transposed, encoder_mask)[0]
        local_out = local_out_transposed.transpose(1, 2)
        local_out = fusion1_out + self.drop_path_2(local_out)
        local_out = local_out * expanded_mask
        
        # Step 4: Second gated fusion (fusion1 result + local)
        gate2_input = torch.cat([fusion1_out, local_out], dim=-1)
        gate2_weights = self.gate2(gate2_input)
        gate2_out = gate2_weights * local_out + (1 - gate2_weights) * fusion1_out
        
        # Step 5: Optional refinement with strong residual connection
        if self.use_ffn_refinement:
            refinement_norm = self.norm_refinement(gate2_out)
            
            if self.refinement_type == "conv_refinement":
                refinement_input = refinement_norm.transpose(1, 2)
                refined_features = self.refinement(refinement_input).transpose(1, 2)
            else:
                refined_features = self.refinement(refinement_norm)
            
            final_out = gate2_out + self.drop_path_3(refined_features)
        else:
            final_out = gate2_out
        
        # Use base class method for output extraction
        anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(
            final_out, anchor_positions, L
        )
        
        return anchor_out, seq_out, anchor_mask, mask

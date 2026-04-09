from copy import deepcopy
import math

import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    sinusoid_encoding, MaskedConv1D, LayerNorm, TransformerEncoder, MaskedMaxPool1D
)

# Import all block types from refactored anchor module
from .anchor import *

backbones = dict()
def register_video_net(name):
    def decorator(module):
        backbones[name] = module
        return module
    return decorator

@register_video_net('hieramamba_backbone')
class HieraMambaBackbone(nn.Module):
    """
    HieraMambaBackbone: Combines convolutions with transformer encoder layers 
    to build a feature pyramid.
    
    video clip features
    -> [embedding convs x L1]
    -> [stem transformer x L2]
    -> [branch transformer x L3]
    -> latent video feature pyramid
    """
    def __init__(
        self,
        in_dim=256,             # video feature dimension
        embd_dim=384,           # embedding dimension
        max_seq_len=2304,        # max sequence length
        n_heads=4,            # number of attention heads for MHA
        mha_win_size=0,       # local window size for MHA (0 for global attention)
        stride=1,           # conv stride applied to the input features
        arch=(2, 0, 8),     # (#convs, #stem transformers, #branch transformers)
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        use_abs_pe=False,   # whether to apply absolute position encoding
        local_window_size=5, # whether to encode local features
        pool_method='mean',
        return_anchor=False,
        block_type="AnchorMambaPoolingBlock",  # block type from anchor_mamba module
        local_encoder_type="transformer",  # Options: 'transformer' or 'mamba'
        ffn_ratio=2,
        local_encode=False,
        local_encode_num_layers=0,
        mamba_headdim=64, 
        mamba_dstate=64,
        mamba_expand=2,
        mamba_dconv=7,
        bidirectional=True
    ):
        super().__init__()

        assert len(arch) == 3, '(embed convs, stem, branch)'
        assert stride & (stride - 1) == 0
        assert arch[0] >= int(math.log2(stride))
        self.max_seq_len = max_seq_len
        self.return_anchor = return_anchor
        local_encode_num_layers = arch[2] if local_encode_num_layers == 0 else local_encode_num_layers

        # embedding projection
        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # embedding convs
        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()
        for _ in range(arch[0]):
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=5 if stride > 1 else 3,
                    stride=2 if stride > 1 else 1,
                    padding=2 if stride > 1 else 1,
                    bias=False
                )
            )
            self.embd_norms.append(LayerNorm(embd_dim))
            stride = max(stride // 2, 1)

        # position encoding (c, t)
        if use_abs_pe:
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # stem transformers
        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(
                TransformerEncoder(
                    embd_dim,
                    stride=1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop
                )
            )

        # Get the block class from the anchor_mamba module
        BlockClass = globals().get(block_type, AnchorMambaPoolingBlock)

        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                BlockClass(
                    d_model=embd_dim,
                    stride=2,
                    nhead=n_heads,
                    local_window_size=local_window_size if idx < 5 else 0,  # Use local window size for first 5 layers
                    local_encode=local_encode if idx < local_encode_num_layers else False,
                    pool_method=pool_method,
                    local_encoder_type=local_encoder_type,
                    ffn_ratio=ffn_ratio,
                    mamba_headdim=mamba_headdim,
                    mamba_dstate=mamba_dstate,
                    mamba_expand=mamba_expand,
                    mamba_dconv=mamba_dconv,
                    bidirectional=bidirectional
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c1, t1)): video features.
            mask (bool tensor, (bs, t1)): video mask.
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)    # (bs, l) -> (bs, 1, l)

        # embedding projection
        x, _ = self.embd_fc(x, mask)

        # embedding convs
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)
            # x = F.relu(norm(x).clone(), inplace=False)

        # position encoding
        _, _, t = x.size()
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)
        else:
            x = x * mask.to(x.dtype)

        # stem layers
        for block in self.stem:
            x, mask = block(x, mask)

        # branch layers
        fpn, fpn_masks, anchor_fpn, anchor_fpn_masks = tuple(), tuple(), tuple(), tuple()
        for i, block in enumerate(self.branch):
            if i == 0:
                anchor_out, x_out, anchor_mask, x_mask = block(x, mask)
            else:
                anchor_out, x_out, anchor_mask, x_mask = block(anchor_out, anchor_mask)
            fpn += (x_out, )
            fpn_masks += (x_mask, )
            anchor_fpn += (anchor_out, )
            anchor_fpn_masks += (anchor_mask, )
        if self.return_anchor:
            return fpn, fpn_masks, anchor_fpn, anchor_fpn_masks
        else:
            return fpn, fpn_masks
        
@register_video_net('transformer')
class VideoTransformer(nn.Module):
    """
    A backbone that combines convolutions with transformer encoder layers 
    to build a feature pyramid.
    
    video clip features
    -> [embedding convs x L1]
    -> [stem transformer x L2]
    -> [branch transformer x L3]
    -> latent video feature pyramid
    """
    def __init__(
        self,
        in_dim=256,             # video feature dimension
        embd_dim=384,           # embedding dimension
        max_seq_len=2304,        # max sequence length
        n_heads=4,            # number of attention heads for MHA
        mha_win_size=19,       # local window size for MHA (0 for global attention)
        stride=1,           # conv stride applied to the input features
        arch=(2, 0, 8),     # (#convs, #stem transformers, #branch transformers)
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        use_abs_pe=True,   # whether to apply absolute position encoding
    ):
        super().__init__()

        assert len(arch) == 3, '(embed convs, stem, branch)'
        assert stride & (stride - 1) == 0
        assert arch[0] >= int(math.log2(stride))
        self.max_seq_len = max_seq_len

        # embedding projection
        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # embedding convs
        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()
        for _ in range(arch[0]):
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim, embd_dim,
                    kernel_size=5 if stride > 1 else 3,
                    stride=2 if stride > 1 else 1,
                    padding=2 if stride > 1 else 1,
                    bias=False
                )
            )
            self.embd_norms.append(LayerNorm(embd_dim))
            stride = max(stride // 2, 1)

        # position encoding (c, t)
        if use_abs_pe:
            pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # stem transformers
        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(
                TransformerEncoder(
                    embd_dim,
                    stride=1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop
                )
            )

        # branch transformers (for FPN)
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            cur_seq_len = max_seq_len // (2 ** (idx))
            self.branch.append(
                TransformerEncoder(
                    embd_dim,
                    stride=2 if idx > 0 else 1,
                    n_heads=n_heads,
                    window_size=mha_win_size if mha_win_size < cur_seq_len else 0,  # Use local window size if it fits
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c1, t1)): video features.
            mask (bool tensor, (bs, t1)): video mask.
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)    # (bs, l) -> (bs, 1, l)

        # embedding projection
        x, _ = self.embd_fc(x, mask)

        # embedding convs
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        # position encoding
        _, _, t = x.size()
        if self.pe is not None:
            pe = self.pe.to(x.dtype)
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
            x = x + pe[..., :t] * mask.to(x.dtype)

        # stem layers
        for block in self.stem:
            x, mask = block(x, mask)

        # branch layers
        fpn, fpn_masks = tuple(), tuple()
        for block in self.branch:
            x, mask = block(x, mask)
            fpn += (x, )
            fpn_masks += (mask, )

        return fpn, fpn_masks


def make_video_net(opt):
    opt = deepcopy(opt)
    return backbones[opt.pop('name')](**opt)
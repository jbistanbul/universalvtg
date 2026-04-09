import torch
import torch.nn as nn

from .fusion import make_fusion
from .head import make_head
from .text_net import make_text_net
from .video_net import make_video_net
from .blocks import MaskedConv1D, masked_max_pool1d

from copy import deepcopy

models = dict()
def register_models_net(name):
    def decorator(module):
        models[name] = module
        return module
    return decorator
    
@register_models_net('pt_transformer')
class PtTransformer(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()

        # backbones
        self.text_net = make_text_net(opt['text_net'])
        self.vid_net = make_video_net(opt['vid_net'])
        vid_net_param_trainable_count = sum(p.numel() for p in self.vid_net.parameters() if p.requires_grad)
        print(f"vid_net trainable parameter count: {vid_net_param_trainable_count}")
        vid_net_param_count = sum(p.numel() for p in self.vid_net.parameters())
        print(f"vid_net parameter count: {vid_net_param_count}")
        # fusion and prediction heads
        self.fusion = make_fusion(opt['fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fpn, fpn_masks = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks)
        return fpn_logits, fpn_offsets, fpn_masks
    
    def encode_text2(self, text, text_masks, text_size):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        text, text_masks = self.encode_text(text, text_masks)
        return text, text_masks
    
    def forward(self, vid, vid_masks, text, text_masks, text_size=None):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        fpn, fpn_masks = self.encode_video(vid, vid_masks)
        fpn_logits, fpn_offsets, fpn_masks = \
            self.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)

        return fpn_logits, fpn_offsets, fpn_masks


@register_models_net('hieramamba')
class HieraMamba(nn.Module):
    """
    HieraMamba: Model for single-stage sentence grounding with auxiliary loss
    """
    def __init__(self, opt):
        super().__init__()
        self.early_fusion = opt.get('early_fusion', True)
        self.opt = opt
        # backbones
        self.text_net = make_text_net(opt['text_net'])
        if self.early_fusion:
            vid_in_dim = opt['vid_net']['in_dim']  # raw video feature dimension (e.g., 2048 for concat)
            vid_embd_dim = opt['vid_net']['embd_dim']  # target embedding dimension (e.g., 512)
            opt['vid_net']['in_dim'] = vid_embd_dim

            # Pop dual-feature keys if present (not used by make_video_net)
            opt['vid_net'].pop('in_dim_a', None)
            opt['vid_net'].pop('in_dim_b', None)
            self.vid_net = make_video_net(opt['vid_net'])

            self.vid_proj = MaskedConv1D(vid_in_dim, vid_embd_dim, 1)
        else:
            opt['vid_net'].pop('in_dim_a', None)
            opt['vid_net'].pop('in_dim_b', None)
            self.vid_net = make_video_net(opt['vid_net'])
        # fusion and prediction heads
        self.fusion = make_fusion(opt['fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])

    def project_video(self, vid, vid_masks):
        """Project raw video features to embedding space."""
        vid, vid_masks = self.vid_proj(vid, vid_masks)
        return vid, vid_masks

    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_text2(self, text, text_masks, text_size):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        text, text_masks = self.encode_text(text, text_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks, anchor_fpn, anchor_fpn_mask = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks, anchor_fpn, anchor_fpn_mask

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fpn, fpn_masks_fusion = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        fpn_logits, _ = self.cls_head(fpn, fpn_masks_fusion)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks_fusion) # fpn: 16, 256, 2304 mask: 2, 1, 2304
        return fpn_logits, fpn_logits, fpn_offsets, fpn_masks


        
    def forward(self, vid, vid_masks, text, text_masks, text_size=None):
        if self.early_fusion:
            return self._forward_earlyfusion(vid, vid_masks, text, text_masks, text_size)
        else:
            return self._forward_regular(vid, vid_masks, text, text_masks, text_size)

    def _forward_regular(self, vid, vid_masks, text, text_masks, text_size=None):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_mask = self.encode_video(vid, vid_masks)
        fpn_logits, fpn_logits2, fpn_offsets, fpn_masks = \
            self.fuse_and_predict(fpn, sequence_fpn_masks, text, text_masks, text_size) # B, 1, 2304

        return fpn_logits, fpn_logits2, fpn_offsets, fpn_masks, fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_mask

    def _forward_earlyfusion(self, vid, vid_masks, text, text_masks, text_size=None):
        # pack text features
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat(
                [t[:k] for t, k in zip(text_masks, text_size)]
            )
        
        text, text_masks = self.encode_text(text, text_masks)
        
        # Project raw video features to embedding space before fusion
        vid_masks = vid_masks.unsqueeze(1)
        vid, vid_masks = self.project_video(vid, vid_masks)
        
        # early fusion 
        vid_fused, vid_masks_fused = self.fusion(vid, vid_masks, text, text_masks, text_size) # vid_fused: (b_query, c, t), vid_masks_fused: (b_query, 1, t)
        
        # continue with video encoding using the fused features
        fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_mask = self.encode_video(vid_fused, vid_masks_fused)
        fpn_logits, fpn_logits2, fpn_offsets, fpn_masks = \
            self.fuse_and_predict(fpn, sequence_fpn_masks, text, text_masks, text_size)

        return fpn_logits, fpn_logits2, fpn_offsets, fpn_masks, fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_mask

class BufferList(nn.Module):

    def __init__(self, buffers):
        super().__init__()

        for i, buf in enumerate(buffers):
            self.register_buffer(str(i), buf, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class PtGenerator(nn.Module):
    """
    A generator for candidate points from specified FPN levels.
    """
    def __init__(
        self,
        max_seq_len,        # max sequence length
        num_fpn_levels,     # number of feature pyramid levels
        regression_range=4, # normalized regression range
        sigma=1,            # controls overlap between adjacent levels
        use_offset=False,   # whether to align points at the middle of two tics
    ):
        super().__init__()

        self.num_fpn_levels = num_fpn_levels
        assert max_seq_len % 2 ** (self.num_fpn_levels - 1) == 0
        self.max_seq_len = max_seq_len

        # derive regression range for each pyramid level
        self.regression_range = ((0, regression_range), )
        assert sigma > 0 and sigma <= 1
        for l in range(1, self.num_fpn_levels):
            assert regression_range <= max_seq_len
            v_min = regression_range * sigma
            v_max = regression_range * 2
            if l == self.num_fpn_levels - 1:
                v_max = max(v_max, max_seq_len + 1)
            self.regression_range += ((v_min, v_max), )
            regression_range = v_max

        self.use_offset = use_offset

        # generate and buffer all candidate points
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        # tics on the input grid
        tics = torch.arange(0, self.max_seq_len, 1.0)

        points_list = tuple()
        for l in range(self.num_fpn_levels):
            stride = 2 ** l
            points = tics[::stride][:, None]                    # (t, 1)
            if self.use_offset:
                points += 0.5 * stride

            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 2)
            stride = torch.as_tensor(
                stride, dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 1)
            points = torch.cat((points, reg_range, stride), 1)  # (t, 4)
            points_list += (points, )

        return BufferList(points_list)

    def forward(self, fpn_n_points):
        """
        Args:
            fpn_n_points (int list [l]): number of points at specified levels.

        Returns:
            fpn_point (float tensor [l * (p, 4)]): candidate points from speficied levels.
        """
        assert len(fpn_n_points) == self.num_fpn_levels

        fpn_points = tuple()
        for n_pts, pts in zip(fpn_n_points, self.buffer_points):
            assert n_pts <= len(pts), (
                'number of requested points {:d} cannot exceed max number '
                'of buffered points {:d}'.format(n_pts, len(pts))
            )
            fpn_points += (pts[:n_pts], )

        return fpn_points
    
def make_models_net(opt):
    opt = deepcopy(opt)
    return models[opt['model_net'].pop('name')](opt['model'])
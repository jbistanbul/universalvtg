import os
import random

import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple

class Logger:

    def __init__(self, filepath):

        self.filepath = filepath

    def write(self, log_str):
        print(log_str)
        with open(self.filepath, 'a') as f:
            print(log_str, file=f)


class AverageMeter(object):

    def __init__(self):

        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.mean = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean = self.sum / self.count

    def item(self):
        return self.mean


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t > 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def fix_random_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ## NOTE: uncomment for CUDA >= 10.2
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # ## NOTE: uncomment for pytorch >= 1.8
    # torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)
    return rng


def iou(pred_segs, gt_segs):
    """
    Args:
        pred_segs (float tensor, (..., 2)): predicted segments.
        gt_segs (float tensor, (..., 2)): ground-truth segments.

    Returns:
        out (float tensor, (...)): intersection over union.
    """
    ps, pe = pred_segs[..., 0], pred_segs[..., 1]
    gs, ge = gt_segs[..., 0], gt_segs[..., 1]

    overlap = (torch.minimum(pe, ge) - torch.maximum(ps, gs)).clamp(min=0)
    union = (pe - ps) + (ge - gs) - overlap
    out = overlap / union
    return out

def generate_multiscale_gt_masks(gt: Tensor, scale_lengths: List[int]) -> Tensor:
    """
    Generate boolean ground truth masks at multiple temporal scales.
    
    Args:
        gt: Tensor of shape (bs, 2) where gt[:, 0] is start and gt[:, 1] is end
        scale_lengths: List of integers [int1, int2, ..., intn] representing 
                      temporal lengths at different scales
    
    Returns:
        Tuple of n boolean tensors, each of shape (bs, int_i), where True regions
        correspond to the scaled intervals from ceil(gt[:, 0]) to floor(gt[:, 1])
    """
    import math
    
    bs = gt.shape[0]
    device = gt.device
    
    # Use the first scale as reference for scaling ratios
    reference_length = scale_lengths[0]
    
    output_masks = []
    
    for scale_length in scale_lengths:
        # Create boolean mask for this scale
        mask = torch.zeros(bs, scale_length, dtype=torch.bool, device=device)
        
        # Calculate scaling factor
        scale_factor = scale_length / reference_length
        
        for b in range(bs):
            # Scale the gt coordinates to this temporal resolution
            start_scaled = gt[b, 0] * scale_factor
            end_scaled = gt[b, 1] * scale_factor
            
            # Apply ceil and floor as specified
            start_idx = math.ceil(start_scaled.item())
            end_idx = math.floor(end_scaled.item())
            
            # Clamp indices to valid range
            start_idx = max(0, min(start_idx, scale_length - 1))
            end_idx = max(0, min(end_idx, scale_length - 1))
            
            # Mark the region as True if valid interval
            if start_idx <= end_idx:
                mask[b, start_idx:end_idx + 1] = True
        
        output_masks.append(mask)
    
    return torch.cat(output_masks, dim=1) # (bs, T_l)


def generate_multiscale_gt_masks_contrastive(points, targets, center_sampling_radius=1.5):
    """
    Assign ground-truth labels and offsets to candidate points.

    Args:
        fpn_points (List[float tensor, (p, 4)]): candidate points.
            (coordinate (1), regression range (2), stride(1))
        targets (float tensor, (bs, 2)): ground-truth segments.

    Returns:
        labels (bool tensor, (bs, p)): ground-truth binary labels.
        offsets (float tensor, (bs, p, 2)): ground-truth offsets.
    """
    labels_list = tuple()
    for target in targets:
        # labels = annotate_points_per_video(points, target, center_sampling_radius)
        labels = annotate_points_per_video_span_aware(points, target, center_sampling_radius)
        labels_list += (labels, )

    labels = torch.stack(labels_list)
    return labels

def annotate_points_per_video(points, target, center_sampling_radius):
    """
    Args:
        points (float tensor, (p, 4)): candidate points from all levels.
            (coordinate (1), regression range (2), stride (1))
        target (float tensor, (2,)): ground-truth segment.

    Returns:
        labels (bool tensor, (p,)): ground-truth binary labels.
        offsets (float tensor, (p, 2)): ground-truth offsets.
    """

    # Center Sampling
    ctr = 0.5 * (target[0] + target[1])
    radius = points[:, 3] * center_sampling_radius
    t_min = (ctr - radius).clamp_(min=target[0])
    t_max = (ctr + radius).clamp_(max=target[1])
    # point distance to window boundaries
    pt2left = points[:, 0] - t_min  # (p,)
    pt2right = t_max - points[:, 0] # (p,)
    inside_window = torch.logical_and(pt2left > 0, pt2right > 0)

    labels = inside_window

    return labels

def annotate_points_per_video_span_aware(points, target, center_sampling_radius):
    """
    Corrected version that considers the temporal span each point represents.
    Only labels spans as positive if they are COMPLETELY contained within GT region.
    
    Args:
        points (float tensor, (p, 4)): candidate points from all levels.
            (coordinate (1), regression range (2), stride (1))
        target (float tensor, (2,)): ground-truth segment.
        center_sampling_radius (float): radius for center sampling
        overlap_threshold (float): minimum overlap ratio to be considered positive (1.0 = 100%)

    Returns:
        labels (bool tensor, (p,)): ground-truth binary labels.
    """
    # Each point represents a span [coordinate, coordinate + stride)
    span_starts = points[:, 0]  # coordinates
    span_ends = points[:, 0] + points[:, 3]  # coordinate + stride
    
    # Center sampling window
    ctr = 0.5 * (target[0] + target[1])
    radius = points[:, 3] * center_sampling_radius
    t_min = (ctr - radius).clamp_(min=target[0])
    t_max = (ctr + radius).clamp_(max=target[1])
    
    # For 100% overlap requirement, the entire span must be within center sampling window
    # AND the center sampling window is already clipped to GT bounds
    span_within_window = torch.logical_and(
        span_starts >= t_min,  # span start >= window start
        span_ends <= t_max     # span end <= window end
    )
    
    labels = span_within_window

    return labels
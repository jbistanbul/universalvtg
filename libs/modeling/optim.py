from bisect import bisect_right
from collections import Counter
import math
import warnings

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

from .blocks import MaskedConv1D, LayerNorm, Scale, LayerScale
from .anchor import RMSNorm, LayerScale

def make_optimizer(model, opt):
    decay, no_decay = set(), set()
    whitelist_modules = (nn.Linear, nn.Conv1d, MaskedConv1D)
    blacklist_modules = (LayerNorm, Scale, LayerScale, RMSNorm)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # Special handling for anchor_mamba AttnPooling parameters
            if ('anchor_pooling.pooler.pool_q' in fpn or
                'anchor_pooling.pooler.pool_attn.in_proj' in fpn or 
                'anchor_pooling.pooler.pool_attn.out_proj' in fpn):
                decay.add(fpn)
                continue
            # Special handling for cross_attention parameters (from EnhancedAnchorBlock_B)
            elif ('cross_attention.in_proj_weight' in fpn or
                  'cross_attention.out_proj.weight' in fpn):
                decay.add(fpn)
                continue
            elif ('cross_attention.in_proj_bias' in fpn or
                  'cross_attention.out_proj.bias' in fpn):
                no_decay.add(fpn)
                continue
            # Special handling for cross_attn parameters (from EnhancedAnchorBlock_Efficient)
            elif ('cross_attn.in_proj_weight' in fpn or
                  'cross_attn.out_proj.weight' in fpn):
                decay.add(fpn)
                continue
            elif ('cross_attn.in_proj_bias' in fpn or
                  'cross_attn.out_proj.bias' in fpn):
                no_decay.add(fpn)
                continue
            # Special handling for fusion_weights parameters (from EnhancedAnchorBlock_MultiScale)
            elif 'fusion_weights' in fpn:
                no_decay.add(fpn)  # These are learnable weights that shouldn't have weight decay
                continue
            # Special handling for memory_bank parameters (from EnhancedAnchorBlock_Memory)
            elif 'memory_bank' in fpn:
                decay.add(fpn)
                continue
            # Special handling for memory attention parameters (from EnhancedAnchorBlock_Memory)
            elif ('memory_attention.in_proj_weight' in fpn or
                  'memory_attention.out_proj.weight' in fpn or
                  'memory_update.in_proj_weight' in fpn or
                  'memory_update.out_proj.weight' in fpn):
                decay.add(fpn)
                continue
            elif ('memory_attention.in_proj_bias' in fpn or
                  'memory_attention.out_proj.bias' in fpn or
                  'memory_update.in_proj_bias' in fpn or
                  'memory_update.out_proj.bias' in fpn):
                no_decay.add(fpn)
                continue
            elif pn in ['A_log', 'D']: #or 'norm.weight' in pn:  # Hydra module special parameters
                decay.add(fpn)
            elif 'norm.weight' in pn:  
                no_decay.add(fpn)
            elif pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_modules):
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, blacklist_modules):
                no_decay.add(fpn)
            elif pn.endswith('bkgd_token'):
                no_decay.add(fpn)
            elif '.scale' in pn:  # For LayerScale2 parameters
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, (
        'parameters {:s} made it into both decay/no decay sets'
        ''.format(str(inter_params))
    )
    assert len(param_dict.keys() - union_params) == 0, (
        'parameters {:s} were not separated into either decay/no decay set'
        ''.format(str(param_dict.keys() - union_params))
    )

    # Build parameter groups without overlaps
    param_groups = []
    assigned_params = set()
    
    # Process each component in priority order
    components = [
        ('cls_head', 'cls_head_lr'),
        ('reg_head', 'reg_head_lr'), 
        ('fusion', 'fusion_lr'),
        ('text_net', 'text_net_lr'),
        ('vid_net', 'vid_net_lr'),
    ]
    
    for component, lr_key in components:
        # Get parameters for this component that haven't been assigned yet
        component_decay = [pn for pn in decay if component in pn and pn not in assigned_params]
        component_no_decay = [pn for pn in no_decay if component in pn and pn not in assigned_params]
        
        # Add decay group
        if component_decay:
            param_groups.append({
                'params': [param_dict[pn] for pn in sorted(component_decay)],
                'weight_decay': opt['weight_decay'],
                'lr': opt.get(lr_key, opt['lr']),
            })
            assigned_params.update(component_decay)
        
        # Add no_decay group
        if component_no_decay:
            param_groups.append({
                'params': [param_dict[pn] for pn in sorted(component_no_decay)],
                'weight_decay': 0.0,
                'lr': opt.get(lr_key, opt['lr']),
            })
            assigned_params.update(component_no_decay)
    
    # Handle any remaining unassigned parameters
    remaining_decay = [pn for pn in decay if pn not in assigned_params]
    remaining_no_decay = [pn for pn in no_decay if pn not in assigned_params]
    
    if remaining_decay:
        param_groups.append({
            'params': [param_dict[pn] for pn in sorted(remaining_decay)],
            'weight_decay': opt['weight_decay'],
            'lr': opt['lr'],
        })
    
    if remaining_no_decay:
        param_groups.append({
            'params': [param_dict[pn] for pn in sorted(remaining_no_decay)],
            'weight_decay': 0.0,
            'lr': opt['lr'],
        })

    # Fixed: Remove global lr and weight_decay to preserve per-group settings
    if opt['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=opt.get('momentum', 0.9)
        )
    elif opt['name'] == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            betas=(0.9, 0.999)
        )
    elif opt['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError(f"invalid optimizer: {opt['name']}")

    return optimizer


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr=0.0,
        eta_min=1e-8,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            # Fixed: Prevent division by zero and use consistent base
            if self.warmup_epochs <= 1:
                return self.base_lrs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            # Fixed: Prevent division by zero
            if self.warmup_epochs <= 1:
                return self.base_lrs
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


class LinearWarmupMultiStepLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a multi-step schedule that decays
    the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        milestones,
        warmup_start_lr=0.0,
        gamma=0.1,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            milestones (list): List of epoch indices. Must be increasing.
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.milestones = Counter(milestones)
        self.gamma = gamma

        super(LinearWarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            # Fixed: Prevent division by zero and use consistent base
            if self.warmup_epochs <= 1:
                return self.base_lrs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]
        elif (self.last_epoch - self.warmup_epochs) not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [
            group['lr'] * self.gamma ** self.milestones[self.last_epoch - self.warmup_epochs]
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            # Fixed: Prevent division by zero
            if self.warmup_epochs <= 1:
                return self.base_lrs
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        milestones = list(sorted(self.milestones.elements()))
        return [base_lr * self.gamma ** bisect_right(milestones, self.last_epoch - self.warmup_epochs)
                for base_lr in self.base_lrs]


def make_scheduler(optimizer, opt):
    warmup_itrs = opt.get('warmup_epochs', 0) * opt['itrs_per_epoch']

    if opt['name'] == 'cosine':
        max_itrs = warmup_itrs + opt['epochs'] * opt['itrs_per_epoch']
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_itrs,
            max_epochs=max_itrs
        )
    elif opt['name'] == 'multistep':
        step_itrs = [opt['itrs_per_epoch'] * s for s in opt['steps']]
        scheduler = LinearWarmupMultiStepLR(
            optimizer,
            warmup_epochs=warmup_itrs,
            milestones=step_itrs,
            gamma=opt.get('gamma', 0.1)
        )
    else:
        raise NotImplementedError(f"invalid scheduler: {opt['name']}")

    return scheduler

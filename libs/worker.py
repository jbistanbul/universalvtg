from collections import OrderedDict
from copy import deepcopy
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .data import make_dataset, make_dataloader
from .dist_utils import get_rank, get_world_size, barrier, all_gather, print0
from .modeling import (
    PtGenerator, PtTransformer, HieraMamba, sigmoid_focal_loss, ctr_giou_loss, ctr_diou_loss,
    make_optimizer, make_scheduler, MultiScaleMaskedContrastive, MultiScaleMaskedGTPointContrastive
)
from .nms import batched_nms
from .train_utils import Logger, AverageMeter, fix_random_seed, iou, time_str, generate_multiscale_gt_masks, generate_multiscale_gt_masks_contrastive

from .modeling.model import make_models_net
from torch.cuda.amp import autocast, GradScaler
import json

AUX_LOSS_REGISTRY = {
    "ds_contrastive": MultiScaleMaskedContrastive, 
    "gt_point_contrastive": MultiScaleMaskedGTPointContrastive,
}

class Trainer:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # build model and EMA
        # self.model = PtTransformer(opt['model']).cuda()
        self.model = make_models_net(opt).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.scaler = GradScaler()  # Initialize the gradient scaler for AMP
        # self.scaler = GradScaler('cuda') 
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(
            opt['train']['data'], num_epochs=self.num_epochs, is_training=True
        )
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.logger = self.tb_writer = self.loss_meters = self.timer = None

        # load model weights and training states
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()], find_unused_parameters=True)
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

    def run(self):
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            for data_list in self.dataloader:
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    self.scaler.unscale_(self.optimizer)  # Unscale gradients for clipping
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.scaler.step(self.optimizer)  # Use scaler for optimizer step
                self.scaler.update()  # Update the scaler
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
            self.checkpoint()
            barrier()
        print0("Training completed.")

    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = norm = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']

        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {'cls': cls_loss, 'reg': reg_loss, 'total': total_loss}

    def _microbatch_forward_backward(self, data_list, is_last=False):
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)

        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            with autocast():  # Enable autocasting for the forward pass
                fpn_logits, fpn_offsets, fpn_masks = \
                    self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                with autocast():  # Enable autocasting for the forward pass
                    fpn_logits, fpn_offsets, fpn_masks = \
                        self.model(vid, vid_masks, text, text_masks, text_size)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        fpn_logits = torch.cat(fpn_logits, dim=1)   # (bs, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1) # (bs, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)     # (bs, p)
        points = torch.cat(fpn_points)              # (p, 4)

        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)

        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()

        ## (2) classification loss on valid points
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()

        total_loss = cls_loss + self.loss_weight * reg_loss
        self.scaler.scale(total_loss).backward()  # Scale the loss for backward
        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
        }

    def _batchify_videos(self, vid_list):
        """
        Put video features and their masks in a batch.

        Args:
            vid_list (List[float tensor, (c1, t1)]): video features.

        Returns:
            vid (float tensor, (bs, c1, t1)): video feature sequences.
            vid_masks (bool tensor, (bs, t1)): video masks.
        """
        bs = len(vid_list)
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        vid = vid_list[0].new_full((bs, vid_dim, self.input_vid_len), 0.)
        for idx in range(bs):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(self.input_vid_len)[None] < vid_lens
        return vid, vid_masks

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks

    def _batchify(self, vid_list, text_list):
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return vid, vid_masks, text, text_masks, text_size

    def _annotate_points(self, points, targets):
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
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _annotate_points_per_video(self, points, target):
        """
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min  # (p,)
            pt2right = t_max - points[:, 0] # (p,)
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets

    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        state_ckpt = torch.load(state_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(model_ckpt['model'])
        self.model_ema.load_state_dict(model_ckpt['model_ema'])
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        if 'scaler' in state_ckpt:  # Load scaler state if available
            self.scaler.load_state_dict(state_ckpt['scaler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self):
        barrier()
        if get_rank() == 0:
            e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
            print0(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
            model_dir = os.path.join(self.opt['_root'], 'models')
            state_dir = os.path.join(self.opt['_root'], 'states')
            model_ckpt = {
                'model': self._unwrap(self.model).state_dict(),
                'model_ema': self.model_ema.state_dict(),
            }
            state_ckpt = {
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),  # Save the scaler state
                'epoch': self.epoch,
                'itr': self.itr,
            }
            torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
            torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
            if self.epoch in self.checkpoint_epochs:
                shutil.copyfile(
                    os.path.join(model_dir, 'last.pth'),
                    os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
                )
        barrier()
    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()

class TrainerOriginal:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 20))
        # rng = None
        # print('no seed set')
        # build model and EMA
        # self.model = PtTransformer(opt['model']).cuda()
        self.model = make_models_net(opt).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.ema_beta = opt['train'].get('ema_beta', 0.999)

        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = make_dataset(
            opt['train']['data'], num_epochs=self.num_epochs, is_training=True
        )
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.logger = self.tb_writer = self.loss_meters = self.timer = None

        # load model weights and training states
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()],) #find_unused_parameters=True)
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')

    def run(self):
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            for data_list in self.dataloader:
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
            self.checkpoint()
            barrier()
        print0("Training completed.")

    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = norm = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']

        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {'cls': cls_loss, 'reg': reg_loss, 'total': total_loss}

    def _microbatch_forward_backward(self, data_list, is_last=False):
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True) # (bs, c_v, t)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True) # (bs, num_queries, c_t, t)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)

        targets = torch.cat([d['target'] / self.vid_stride for d in data_list]) # (bs * num_queries, 2)
        targets = targets.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits, fpn_offsets, fpn_masks = \
                self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_offsets, fpn_masks = \
                    self.model(vid, vid_masks, text, text_masks, text_size)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        fpn_logits = torch.cat(fpn_logits, dim=1)   # (bs * num_queries, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1) # (bs * num_queries, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)     # (bs * num_queries, p)
        points = torch.cat(fpn_points)              # (p, 4)

        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)
        # gt_labels, gt_offsets = self._annotate_points_adaptive(points, targets)
        # gt_labels, gt_offsets = self._annotate_points_improved(points, targets)
        # gt_labels, gt_offsets = self._annotate_points_improved2(points, targets)
        
        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()

        ## (2) classification loss on valid points
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()

        total_loss = cls_loss + self.loss_weight * reg_loss
        total_loss.backward()
        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
        }

    def _batchify_videos(self, vid_list):
        """
        Put video features and their masks in a batch.

        Args:
            vid_list (List[float tensor, (c1, t1)]): video features.

        Returns:
            vid (float tensor, (bs, c1, t1)): video feature sequences.
            vid_masks (bool tensor, (bs, t1)): video masks.
        """
        bs = len(vid_list)
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        vid = vid_list[0].new_full((bs, vid_dim, self.input_vid_len), 0.)
        for idx in range(bs):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(self.input_vid_len)[None] < vid_lens
        return vid, vid_masks

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks

    def _batchify(self, vid_list, text_list):
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return vid, vid_masks, text, text_masks, text_size

    def _annotate_points(self, points, targets):
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
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _annotate_points_per_video(self, points, target):
        """
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min  # (p,)
            pt2right = t_max - points[:, 0] # (p,)
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets
    
    def _annotate_points_per_video_fine_scale_fix(self, points, target):
        """
        Conservative fine-scale fix that reduces regression loss.
        """
        pt2start = points[:, 0] - target[0]
        pt2end = target[1] - points[:, 0]
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]
        
        segment_length = target[1] - target[0]
        ctr = 0.5 * (target[0] + target[1])
        
        # (1) Standard center sampling
        if self.center_sampling == 'radius':
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0]
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)
        
        # (2) Conservative regression range adaptation
        max_reg_dist = torch.maximum(pt2start, pt2end)
        strides = points[:, 3]
        
        # More conservative approach: only minimal expansion for very short segments
        fine_scale_mask = strides <= 4
        very_short_segment = segment_length < 15  # Only help very short segments
        
        # Only expand upper bound, and only modestly
        adapted_reg_max = torch.where(
            fine_scale_mask & very_short_segment,
            points[:, 2] * 1.3,  # Only 30% expansion vs your 200%+ expansion
            points[:, 2]  # Keep original for others
        )
        
        # Keep original minimum to avoid too many low-quality positives
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1],  # Original minimum
            max_reg_dist < adapted_reg_max  # Slightly expanded maximum
        )
        
        # Additional quality filter for very short segments
        if segment_length < 15:
            # Only keep points with reasonable symmetry to reduce regression difficulty
            asymmetry_ratio = torch.abs(pt2start - pt2end) / torch.maximum(pt2start, pt2end)
            symmetry_filter = asymmetry_ratio < 0.8  # Allow up to 80% asymmetry
            inside_range = torch.logical_and(inside_range, symmetry_filter)
        
        labels = torch.logical_and(inside_window, inside_range)
        return labels, offsets
    
    def _annotate_points_improved2(self, points, targets):
        """
        Improved annotation using multi-strategy approach.
        
        Args:
            points (float tensor, (p, 4)): candidate points.
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video_fine_scale_fix(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets
    
    def _annotate_points_per_video_short_segments(self, points, target):
        """
        Improved annotation method for very short segments (e.g., length ~10 in videos of length ~900).
        Uses more conservative and targeted improvements.
        
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        # Calculate segment length and center
        segment_length = target[1] - target[0]
        ctr = 0.5 * (target[0] + target[1])

        # (1) Multi-scale aware center sampling
        if self.center_sampling == 'radius':
            base_radius = points[:, 3] * self.center_sampling_radius
            
            # Strategy 1: Scale-adaptive radius with conservative expansion
            stride_ratio = segment_length / points[:, 3]
            
            # Only boost for very fine scales where segment is smaller than 2x stride
            scale_boost = torch.where(
                stride_ratio < 1.0,  # Very short relative to stride
                torch.clamp(1.0 / torch.sqrt(stride_ratio), 1.0, 1.5),  # Conservative sqrt-based boost
                torch.ones_like(stride_ratio)
            )
            
            # Strategy 2: Ensure minimum effective radius but cap it
            min_effective_radius = torch.minimum(
                segment_length * 0.2,  # 20% of segment length
                points[:, 3] * 0.5     # Or half the stride, whichever is smaller
            )
            
            adaptive_radius = torch.maximum(base_radius * scale_boost, min_effective_radius)
            
            # Apply adaptive radius
            t_min = (ctr - adaptive_radius).clamp_(min=target[0])
            t_max = (ctr + adaptive_radius).clamp_(max=target[1])
            
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0]
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) Conservative regression range adaptation
        max_reg_dist = torch.maximum(pt2start, pt2end)
        
        # Strategy 3: Gradual regression range relaxation based on segment/stride ratio
        if segment_length < points[:, 3].min() * 2:  # Only for very short segments
            # Conservative expansion: only 25% relaxation
            relaxation_factor = 1.25
            expanded_reg_min = points[:, 1] / relaxation_factor
            expanded_reg_max = points[:, 2] * relaxation_factor
            
            # Use weighted combination of original and relaxed ranges
            weight = torch.minimum(segment_length / (points[:, 3] * 2), torch.tensor(1.0))
            final_reg_min = weight * points[:, 1] + (1 - weight) * expanded_reg_min
            final_reg_max = weight * points[:, 2] + (1 - weight) * expanded_reg_max
            
            inside_range = torch.logical_and(
                max_reg_dist >= final_reg_min, max_reg_dist < final_reg_max
            )
        else:
            # Normal regression range constraint
            inside_range = torch.logical_and(
                max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
            )

        # Strategy 4: Additional quality filtering for very short segments
        if segment_length < 20:
            # Prefer points closer to segment center for very short segments
            dist_to_center = torch.abs(points[:, 0] - ctr)
            center_weight = torch.exp(-dist_to_center / (segment_length * 0.5))
            
            # Only keep points with reasonable center alignment (top 80% by center weight)
            center_thresh = torch.quantile(center_weight[inside_window], 0.2) if inside_window.sum() > 5 else 0.0
            center_filter = center_weight >= center_thresh
            
            labels = torch.logical_and(
                torch.logical_and(inside_window, inside_range),
                center_filter
            )
        else:
            labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets

    def _annotate_points_ultra_short(self, points, target):
        """
        Specialized handling for ultra-short segments (< 5 time units).
        """
        pt2start = points[:, 0] - target[0]
        pt2end = target[1] - points[:, 0]
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]
        
        segment_length = target[1] - target[0]
        ctr = 0.5 * (target[0] + target[1])
        
        # Very tight center sampling
        if self.center_sampling == 'radius':
            # Use fixed small radius for ultra-short segments
            tight_radius = torch.minimum(
                segment_length * 0.3,  # 30% of segment
                points[:, 3] * 0.3     # 30% of stride
            )
            
            t_min = (ctr - tight_radius).clamp_(min=target[0])
            t_max = (ctr + tight_radius).clamp_(max=target[1])
            
            pt2left = points[:, 0] - t_min
            pt2right = t_max - points[:, 0]
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)
        
        # Strict regression range - only minimal relaxation
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1] * 0.8,  # Minimal relaxation
            max_reg_dist < points[:, 2] * 1.2
        )
        
        # Only keep the most relevant scale levels for ultra-short segments
        scale_filter = points[:, 3] <= segment_length * 2  # Only use fine scales
        
        labels = torch.logical_and(
            torch.logical_and(inside_window, inside_range),
            scale_filter
        )
        
        return labels, offsets

    def _annotate_points_multi_strategy(self, points, target):
        """
        Multi-strategy annotation that combines different approaches based on segment characteristics.
        """
        segment_length = target[1] - target[0]
        
        # Strategy selection based on segment length
        if segment_length < 5:
            # Ultra-short segments: very conservative, focus on highest quality points
            return self._annotate_points_ultra_short(points, target)
        elif segment_length < 20:
            # Short segments: use improved conservative method
            return self._annotate_points_per_video_short_segments(points, target)
        else:
            # Normal segments: use original method
            return self._annotate_points_per_video(points, target)

    def _annotate_points_improved(self, points, targets):
        """
        Improved annotation using multi-strategy approach.
        
        Args:
            points (float tensor, (p, 4)): candidate points.
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_multi_strategy(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        state_ckpt = torch.load(state_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(model_ckpt['model'])
        self.model_ema.load_state_dict(model_ckpt['model_ema'])
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self):
        if get_rank() != 0:
            return
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))
        if self.epoch in self.checkpoint_epochs:
            shutil.copyfile(
                os.path.join(model_dir, 'last.pth'),
                os.path.join(model_dir, f"{self.epoch:0{e}d}.pth")
            )

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()

class TrainerAuxiliary(TrainerOriginal):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        vid_embd_dim = opt['model']['vid_net']['embd_dim']
        self.ds_contrastive = opt['train']['loss_aux']['ds_contrast']['enable']
        self.gt_contrastive = opt['train']['loss_aux']['gt_contrast']['enable']
        self.early_fusion = opt['model'].get('early_fusion', True)
        self.use_mst = opt['model'].get('use_mst', False)
        if self.early_fusion and self.logger:
            self.logger.write("Early fusion enabled")

        if self.ds_contrastive:
            ds_loss_type = AUX_LOSS_REGISTRY[opt['train']['loss_aux']['ds_contrast'].get('type', 'ds_contrastive')]
            self.ds_contrastive_loss = ds_loss_type(opt['train']['loss_aux']['ds_contrast'], vid_embd_dim).cuda()
            self.ds_contrastive_weight = opt['train']['loss_aux']['ds_contrast']['weight']
        else: 
            self.ds_contrastive_weight = 0.0
        
        if self.gt_contrastive:
            gt_loss_type = AUX_LOSS_REGISTRY[opt['train']['loss_aux']['gt_contrast'].get('type', 'gt_point_contrastive')]
            self.gt_contrastive_loss = gt_loss_type(opt['train']['loss_aux']['gt_contrast'], vid_embd_dim).cuda()
            self.gt_contrastive_weight = opt['train']['loss_aux']['gt_contrast']['weight']
            self.loss_aux_gt_type = opt['train']['loss_aux']['gt_contrast'].get('gt_type', 'point')
            self.loss_aux_span_radius = opt['train']['loss_aux']['gt_contrast'].get('span_radius', self.center_sampling_radius)
            self.span_contr_gt = opt['train']['loss_aux']['gt_contrast'].get('span_contr_gt', False)       
        else:
            self.gt_contrastive_weight = 0.0

    def forward_backward(self, data_list):
        cls_loss = reg_loss = total_loss = norm = ds_contrast = gt_contrast = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']
            ds_contrast += loss_dict['ds_contrast']
            gt_contrast += loss_dict['gt_contrast']
        
        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {'cls': cls_loss, 'reg': reg_loss, 'ds_contrast': ds_contrast, 'gt_contrast': gt_contrast, 'total': total_loss}

    def _microbatch_forward_backward(self, data_list, is_last=False):
        # batch data
        vid, vid_masks, text, text_masks, text_size = self._batchify(
            vid_list=[d['vid'] for d in data_list], 
            text_list=[d['text'] for d in data_list]
        )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)

        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits, fpn_logits2, fpn_offsets, fpn_masks, fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_masks= \
                self.model(vid, vid_masks, text, text_masks, text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_logits2, fpn_offsets, fpn_masks, fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_masks= \
                    self.model(vid, vid_masks, text, text_masks, text_size)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        fpn_logits = torch.cat(fpn_logits, dim=1)   # (bs, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1) # (bs, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)     # (bs, p)
        points = torch.cat(fpn_points)              # (p, 4)

        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)

        if self.ds_contrastive:
            ds_contrastive_loss = self.ds_contrastive_loss(
                fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_masks
            ) / self.loss_norm * get_world_size()
            # normalize by number of queries if not early fusion
            if not self.early_fusion:
                ds_contrastive_loss = text_size.float().mean() * ds_contrastive_loss
        else:
            ds_contrastive_loss = torch.tensor(0.0).cuda()

        if self.gt_contrastive:
            gt_labels_split = gt_labels.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            fpn_masks_split = fpn_masks.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            # fpn_logits_split = fpn_logits.split(fpn_n_points, dim=1)

            # gt labels SPAN
            gt_labels_span = generate_multiscale_gt_masks(targets, fpn_n_points)
            gt_labels_span = gt_labels_span.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            
            # gt labels SPAN CONTRASTIVE
            gt_labels_span_contrastive = generate_multiscale_gt_masks_contrastive(points, targets, self.loss_aux_span_radius)
            gt_labels_span_contrastive = gt_labels_span_contrastive.split(fpn_n_points, dim=1) # (B*num_queries, T_l)
            
            # replace gt labels with contrastive gt labels that was sampled with configured radius
            if self.span_contr_gt:
                gt_labels_split = gt_labels_span_contrastive
            
            # Expand fpn from (bs, d, T_l) to (bs*num_queries, d, T_l) for non-early fusion
            if not self.early_fusion:
                fpn_expanded = tuple(torch.repeat_interleave(fpn_layer, text_size, dim=0) for fpn_layer in fpn)
                # fpn_expanded = fpn
            else:
                fpn_expanded = fpn
            
            gt_contrastive_loss = self.gt_contrastive_loss(
                fpn_expanded, fpn_masks_split, gt_labels_split, gt_labels_span
            ) / self.loss_norm * get_world_size()

            # not masking entire gt span
            # gt_contrastive_loss = self.gt_contrastive_loss(
            #     fpn, fpn_masks_split, gt_labels_split, gt_labels_split
            # ) / self.loss_norm * get_world_size()
        else:
            gt_contrastive_loss = torch.tensor(0.0).cuda()

        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()

        ## (2) classification loss on valid points
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()
        if self.use_mst:
            fpn_logits2 = torch.cat(fpn_logits2, dim=1) # (bs, p)
            cls_loss2 = self._calc_focal_loss(
                logits=fpn_logits2[fpn_masks], labels=gt_labels[fpn_masks]
            ) / self.loss_norm * get_world_size()
            cls_loss = (cls_loss + cls_loss2) / 2
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()

        total_loss = cls_loss + self.loss_weight * reg_loss + \
            self.ds_contrastive_weight * ds_contrastive_loss + \
            self.gt_contrastive_weight * gt_contrastive_loss
        total_loss.backward()

        return {
            'cls': cls_loss.detach(),
            'reg': reg_loss.detach(),
            'total': total_loss.detach(),
            'norm': norm.detach(),
            'ds_contrast': ds_contrastive_loss.detach(),
            'gt_contrast': gt_contrastive_loss.detach()
        }

    def log(self):
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        for k, v in self.loss_meters.items():
            if k == 'ds_contrast' or k == 'gt_contrast':
                log_str += f"{k} {float(v.item()):.6f} | "
            else:
                log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(k, v.item(), self.itr)
            v.reset()
        lr = self.scheduler.get_last_lr()[0]
        self.tb_writer.add_scalar('lr', lr, self.itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()


class TrainerAuxiliaryWithValidation(TrainerAuxiliary):
    """
    Extended TrainerAuxiliary with two additional features:
    1. Validation after every epoch with recall reporting and best model saving
    2. Support for loading from a pretrained checkpoint
    
    Configuration (in YAML):
        train:
          validation:
            enable: true                    # Enable validation after each epoch
            val_every: 1                    # Validate every N epochs
            patience: -1                    # Early stopping patience (-1 = disabled)
            metric: "R1@0.3"                # Metric to track for early stopping/best model
                                            # Options: "R1@0.3", "R1@0.5", "R5@0.3", "R5@0.5"
                                            # Or "avg"/"mean" for average of all R1 and R5 metrics
            batch_size: 1                   # Validation batch size (default: 1)
            num_workers: 0                  # Validation dataloader workers (default: 0)
            # Optional: Override eval data paths for validation
            anno_file: null                 # If null, uses eval.data.anno_file
            vid_feat_dir: null              # If null, uses eval.data.vid_feat_dir
            text_feat_dir: null             # If null, uses eval.data.text_feat_dir
          
          pretrain_checkpoint: null         # Path to pretrained checkpoint (optional)
    """

    def __init__(self, opt):
        # Check for pretrained checkpoint BEFORE calling super().__init__
        self.pretrain_checkpoint = opt['train'].get('pretrain_checkpoint', None)
        
        # Initialize parent class (TrainerAuxiliary)
        super().__init__(opt)
        
        # Load pretrained checkpoint if specified
        if self.pretrain_checkpoint:
            self._load_pretrain_checkpoint(self.pretrain_checkpoint)
        
        # Parse validation config
        val_config = opt['train'].get('validation', {})
        self.validation_enabled = val_config.get('enable', False)
        self.val_every = val_config.get('val_every', 1)
        self.patience = val_config.get('patience', -1)  # -1 means disabled
        self.metric_name = val_config.get('metric', 'R1@0.3')
        self._parse_metric_name(self.metric_name)
        
        if self.validation_enabled:
            if self.logger:
                self.logger.write(f"Validation enabled: every {self.val_every} epochs")
                self.logger.write(f"Tracking metric: {self.metric_name}")
                if self.patience > 0:
                    self.logger.write(f"Early stopping patience: {self.patience}")
            
            # Validation dataloader settings
            self.val_batch_size = val_config.get('batch_size', 1)
            self.val_num_workers = val_config.get('num_workers', 0)
            
            # Prepare validation dataset
            # Use eval.data as base config, only override if validation config provides non-null values
            val_data_config = deepcopy(opt['eval']['data'])
            if val_config.get('anno_file') is not None:
                val_data_config['anno_file'] = val_config['anno_file']
            if val_config.get('vid_feat_dir') is not None:
                val_data_config['vid_feat_dir'] = val_config['vid_feat_dir']
            if val_config.get('text_feat_dir') is not None:
                val_data_config['text_feat_dir'] = val_config['text_feat_dir']
            if val_config.get('ext_score_dir') is not None:
                val_data_config['ext_score_dir'] = val_config['ext_score_dir']
            
            self.val_dataset = make_dataset(val_data_config, is_training=False)
            rng = fix_random_seed(opt.get('seed', 2022))
            self.val_dataloader, _ = make_dataloader(
                self.val_dataset, is_training=False, generator=rng,
                batch_size=self.val_batch_size, num_workers=self.val_num_workers
            )
            if self.logger:
                self.logger.write(f"Validation batch_size: {self.val_batch_size}, num_workers: {self.val_num_workers}")
            
            # Validation hyperparameters
            self.ranks = opt['eval'].get('ranks', (1, 5))
            self.topk = max(self.ranks)
            self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
            self.batched_nms_val = lambda segs, scores: batched_nms(segs, scores, **opt['eval']['nms'])
            self.pre_nms_topk = opt['eval']['pre_nms_topk']
            self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
            self.seg_len_thresh = opt['eval']['seg_len_thresh']
            self.max_text_len_eval = opt['eval'].get('max_text_len', 24)
            self.batchify_text_queries = opt['eval'].get('batchify_text_queries', True)
            
            # Calculate min_chunk_size for sliding window evaluation
            num_fpn_levels = opt['model']['num_fpn_levels']
            mha_win_size = opt['model']['mha_win_size']
            ds_strides = [2 ** i for i in range(num_fpn_levels)]
            min_chunk_size = 1
            for idx in range(num_fpn_levels):
                stride = ds_strides[idx]
                if mha_win_size > 0:
                    stride *= (mha_win_size // 2) * 2
                min_chunk_size = max(min_chunk_size, stride)
            self.min_chunk_size = min_chunk_size
            
            # Sliding window parameters
            self.window_size = opt['eval'].get('window_size')
            self.window_stride = opt['eval'].get('window_stride')
            
            # Best model tracking
            self.best_metric = 0.0
            self.best_epoch = 0
            self.epochs_without_improvement = 0
            self.validation_history = {}
    
    def _parse_metric_name(self, metric):
        """Parse metric name like 'R1@0.3' into rank and IoU threshold.
        
        Special metric names:
            - "avg" or "mean": Average of all R1 and R5 metrics across all IoU thresholds
            - "mIoU": Mean IoU of top-1 prediction vs ground truth
        """
        self.use_avg_metric = False
        self.use_miou_metric = False
        
        if metric.lower() in ('avg', 'mean', 'average'):
            # Use average of all R1 and R5 metrics
            self.use_avg_metric = True
            self.track_rank = None
            self.track_iou = None
        elif metric.lower() == 'miou':
            # Use mean IoU of top-1 prediction vs ground truth
            self.use_miou_metric = True
            self.track_rank = None
            self.track_iou = None
        else:
            # Parse standard metric name like 'R1@0.3'
            parts = metric.replace("R", "").split("@")
            self.track_rank = int(parts[0])
            self.track_iou = float(parts[1])
    
    def _load_pretrain_checkpoint(self, checkpoint_path):
        """Load model weights from a pretrained checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrain checkpoint not found: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Unwrap DDP if needed — checkpoint keys don't have 'module.' prefix
        model = self._unwrap(self.model)
        
        # Load model weights (prefer model_ema if available, otherwise model)
        if 'model_ema' in ckpt:
            model.load_state_dict(ckpt['model_ema'])
            self.model_ema.load_state_dict(ckpt['model_ema'])
            print0(f"Loaded pretrained model (EMA weights) from: {checkpoint_path}")
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            self.model_ema.load_state_dict(ckpt['model'])
            print0(f"Loaded pretrained model from: {checkpoint_path}")
        else:
            # Assume the checkpoint is the state dict directly
            model.load_state_dict(ckpt)
            self.model_ema.load_state_dict(ckpt)
            print0(f"Loaded pretrained model (direct state dict) from: {checkpoint_path}")
        
        if get_rank() == 0:
            self.logger.write(f"Loaded pretrained checkpoint: {checkpoint_path}")
    
    def run(self):
        """Training loop with optional validation after each epoch."""
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            for data_list in self.dataloader:
                # run one optimization step
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            self.epoch += 1
            self.checkpoint()
            barrier()
            
            # Run validation after epoch if enabled
            if self.validation_enabled and self.epoch % self.val_every == 0:
                should_stop = self._run_validation_epoch()
                
                # Early stopping check (synchronized across ranks)
                if self.opt['_distributed']:
                    import torch.distributed as dist
                    should_stop_tensor = torch.tensor([int(should_stop)], device='cuda')
                    dist.broadcast(should_stop_tensor, src=0)
                    should_stop = bool(should_stop_tensor.item())
                
                if should_stop:
                    print0(f"Early stopping triggered at epoch {self.epoch}")
                    print0(f"Best {self.metric_name}: {self.best_metric:.4f} at epoch {self.best_epoch}")
                    break
        
        print0("Training completed.")
        if self.validation_enabled:
            print0(f"Best {self.metric_name}: {self.best_metric:.4f} at epoch {self.best_epoch}")
    
    def _run_validation_epoch(self):
        """Run validation and return whether to stop early."""
        if get_rank() == 0:
            val_metric, all_metrics = self._validate()
            
            # Track validation history (store all metrics)
            self.validation_history[self.epoch] = all_metrics
            self._save_validation_history()
            
            # Check for improvement (using tracked metric)
            if val_metric > self.best_metric:
                self.best_metric = val_metric
                self.best_epoch = self.epoch
                self.epochs_without_improvement = 0
                self._save_best_model()
                self.logger.write(f"  New best {self.metric_name}: {val_metric:.4f} at epoch {self.epoch}")
            else:
                self.epochs_without_improvement += 1
                self.logger.write(f"  No improvement for {self.epochs_without_improvement} validation(s)")
                self.logger.write(f"  Current: {val_metric:.4f}, Best: {self.best_metric:.4f} (epoch {self.best_epoch})")
            
            # Log to tensorboard (all metrics)
            if self.tb_writer:
                self.tb_writer.add_scalar(f'val/{self.metric_name}', val_metric, self.epoch)
                self.tb_writer.add_scalar('val/best_metric', self.best_metric, self.epoch)
                for metric_name, metric_val in all_metrics.items():
                    self.tb_writer.add_scalar(f'val/{metric_name}', metric_val, self.epoch)
            
            # Check early stopping
            if self.patience > 0 and self.epochs_without_improvement >= self.patience:
                # Clear CUDA cache before returning to training
                torch.cuda.empty_cache()
                return True
        
        barrier()
        # Clear CUDA cache to release memory used during validation
        torch.cuda.empty_cache()
        return False
    
    @torch.no_grad()
    def _validate(self):
        """Run validation and return the tracked metric and all metrics dict."""
        self.logger.write(f"\n=== Validation at epoch {self.epoch} ===")
        
        # Use EMA model for validation
        self.model_ema.eval()
        
        counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
        text_cnt = 0
        iou_sum = 0.0  # accumulate top-1 IoU for mIoU
        
        for data_list in self.val_dataloader:
            # Iterate over all videos in the batch (handles batch_size > 1)
            for data in data_list:
                results = self._predict_val(data)
                targets = data['segment']
                
                for result, target in zip(results, targets):
                    segs, scores = result['segments'], result['scores']
                    idx = scores.argsort(descending=True)
                    segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                    target = torch.as_tensor(target, dtype=torch.float)
                    target = target.expand(len(segs), -1)
                    
                    iou_topk = iou(segs, target)
                    iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                    counts += (iou_n[:, None] >= self.iou_threshs[None])
                    iou_sum += iou_topk[0].item()  # mIoU: top-1 IoU
                
                text_cnt += len(targets)
        
        # Calculate metrics
        metrics = counts / text_cnt
        miou = iou_sum / text_cnt
        
        # Build all metrics dict
        all_metrics = {}
        log_str = f"Validation results: "
        for i, rank in enumerate(self.ranks):
            for j, thresh in enumerate(self.iou_threshs):
                metric_val = metrics[i, j]
                metric_name = f"R{rank}@{thresh:.1f}"
                all_metrics[metric_name] = metric_val
                log_str += f"{metric_name}={metric_val * 100:.2f}% "
        
        # mIoU metric
        all_metrics['mIoU'] = miou
        log_str += f"mIoU={miou * 100:.2f}% "
        
        # Compute average of all recall metrics (for "avg" metric option)
        avg_metric = metrics.mean()
        all_metrics['avg'] = avg_metric
        log_str += f"avg={avg_metric * 100:.2f}%"
        
        self.logger.write(log_str)
        
        # Get the tracked metric value
        if self.use_miou_metric:
            # Use mean IoU of top-1 prediction vs ground truth
            tracked_metric = miou
        elif self.use_avg_metric:
            # Use average of all R1 and R5 metrics across all IoU thresholds
            tracked_metric = avg_metric
        else:
            # Use specific metric like R1@0.3
            rank_idx = self.ranks.index(self.track_rank) if self.track_rank in self.ranks else 0
            iou_idx = np.where(np.isclose(self.iou_threshs, self.track_iou))[0]
            iou_idx = iou_idx[0] if len(iou_idx) > 0 else 0
            tracked_metric = metrics[rank_idx, iou_idx]
        
        return tracked_metric, all_metrics
    
    def _batchify_text_for_eval(self, text_list):
        """Batch text queries for evaluation."""
        bs = len(text_list)
        n = max(len(t) for t in text_list)
        text_dim = text_list[0][0].size(0)
        text_lens = [[t.size(-1) for t in ts] for ts in text_list]
        max_text_len = min(max(max(lens) for lens in text_lens), self.max_text_len_eval)
        
        text = text_list[0][0].new_full((bs, n, text_dim, max_text_len), 0.)
        text_masks = torch.zeros(bs, n, max_text_len, dtype=torch.bool)
        text_size = torch.zeros(bs, dtype=torch.long)
        
        for i, (ts, lens) in enumerate(zip(text_list, text_lens)):
            text_size[i] = len(ts)
            for j, (t, l) in enumerate(zip(ts, lens)):
                actual_len = min(l, max_text_len)
                text[i, j, :, :actual_len] = t[:, :actual_len]
                text_masks[i, j, :actual_len] = True
                
        return text, text_masks, text_size
    
    def _predict_val(self, data):
        """Predict event segments for validation.

        Respects ``self.batchify_text_queries``:
          True  – all queries encoded together and forwarded in a single batched
                  pass per window (faster, more VRAM).
          False – each query encoded and forwarded independently, one complete
                  forward pass per query per window (matches single-query
                  inference path, lower VRAM).
        """
        # Parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens,)
        
        # Parse video
        vid = data['vid']
        vid_len = vid.size(-1)
        
        # Encode text queries
        if self.batchify_text_queries:
            text, text_masks, text_size = self._batchify_text_for_eval([tokens])
            text = text.cuda(non_blocking=True)
            text_masks = text_masks.cuda(non_blocking=True)
            text_size = text_size.cuda(non_blocking=True)
            text_encoded, text_masks_encoded = self.model_ema.encode_text2(text, text_masks, text_size)
        else:
            # Encode each query separately; full forward pass is also per-query (see window loop below)
            text_list, text_mask_list = [], []
            for token in tokens:
                t = token[None].cuda(non_blocking=True)
                t_mask = t.new_full((1, 1, t.size(-1)), 1, dtype=torch.bool).cuda(non_blocking=True)
                t_enc, t_mask_enc = self.model_ema.encode_text(t, t_mask)
                text_list.append(t_enc)
                text_mask_list.append(t_mask_enc)

        # External scores
        ext_scores = data.get('ext_scores')
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]
        
        # Sliding window evaluation
        window_size = min(self.window_size or vid_len, vid_len, self.max_vid_len)
        window_stride = self.window_stride or window_size
        
        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = [], [], []
        
        idx = 0
        while idx <= max(n, 0):
            windows.append(vid[..., idx:idx + window_size])
            window_offsets.append(idx)
            if ext_scores is not None:
                window_ext_scores.append(ext_scores[..., idx:idx + window_size])
            else:
                window_ext_scores.append(None)
            idx += window_stride
            if n <= 0:
                break
        
        if n > 0 and n % window_stride > 0:
            windows.append(vid[..., -window_size:])
            window_offsets.append(n)
            if ext_scores is not None:
                window_ext_scores.append(ext_scores[..., -window_size:])
            else:
                window_ext_scores.append(None)
        
        # Calculate input_vid_len for padding
        stride = self.min_chunk_size * self.vid_stride
        input_vid_len = (window_size + (stride - 1)) // stride * stride
        
        segs_list, scores_list = [], []
        
        for window, window_offset, window_ext in zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)

            window_segs_list, window_scores_list = [], []

            if self.batchify_text_queries:
                # Single batched forward pass for all queries
                if self.early_fusion:
                    window_proj, window_mask_proj = self.model_ema.vid_proj(window, window_mask)
                    window_fused, window_mask_fused = self.model_ema.fusion(
                        window_proj, window_mask_proj, text_encoded, text_masks_encoded, text_size
                    )
                else:
                    window_fused, window_mask_fused = window, window_mask
                
                fpn, fpn_masks, _, _ = self.model_ema.encode_video(window_fused, window_mask_fused)
                
                if self.use_mst:
                    fpn_logits, _, fpn_offsets, _ = self.model_ema.fuse_and_predict_mst(
                        fpn, fpn_masks, text_encoded, text_masks_encoded, text_size
                    )
                else:
                    fpn_logits, _, fpn_offsets, _ = self.model_ema.fuse_and_predict(
                        fpn, fpn_masks, text_encoded, text_masks_encoded, text_size
                    )
                
                fpn_n_points = [m.size(-1) for m in fpn_masks]
                fpn_points = self.pt_gen(fpn_n_points)
                fpn_masks_squeezed = [m.squeeze(1) for m in fpn_masks]
                
                for query_idx in range(len(tokens)):
                    query_logits = tuple(layer[query_idx:query_idx+1] for layer in fpn_logits)
                    query_offsets = tuple(layer[query_idx:query_idx+1] for layer in fpn_offsets)
                    
                    segs, scores = self._collect_segments_val(
                        fpn_points, query_logits, query_offsets, fpn_masks_squeezed,
                        window_ext[query_idx] if window_ext is not None else None
                    )
                    segs = segs + window_offset / self.vid_stride
                    window_segs_list.append(segs.cpu())
                    window_scores_list.append(scores.cpu())
            else:
                # One complete forward pass per query
                for q_idx, (text_enc, text_mask_enc) in enumerate(zip(text_list, text_mask_list)):
                    if self.early_fusion:
                        w, wm = self.model_ema.vid_proj(window, window_mask)
                        w, wm = self.model_ema.fusion(w, wm, text_enc, text_mask_enc)
                    else:
                        w, wm = window, window_mask

                    fpn, fpn_masks, _, _ = self.model_ema.encode_video(w, wm)

                    if self.use_mst:
                        fpn_logits, _, fpn_offsets, _ = self.model_ema.fuse_and_predict_mst(
                            fpn, fpn_masks, text_enc, text_mask_enc
                        )
                    else:
                        fpn_logits, _, fpn_offsets, _ = self.model_ema.fuse_and_predict(
                            fpn, fpn_masks, text_enc, text_mask_enc
                        )

                    fpn_n_points = [m.size(-1) for m in fpn_masks]
                    fpn_points = self.pt_gen(fpn_n_points)
                    fpn_masks_squeezed = [m.squeeze(1) for m in fpn_masks]

                    query_logits = tuple(layer[0:1] for layer in fpn_logits)
                    query_offsets = tuple(layer[0:1] for layer in fpn_offsets)

                    segs, scores = self._collect_segments_val(
                        fpn_points, query_logits, query_offsets, fpn_masks_squeezed,
                        window_ext[q_idx] if window_ext is not None else None
                    )
                    segs = segs + window_offset / self.vid_stride
                    window_segs_list.append(segs.cpu())
                    window_scores_list.append(scores.cpu())

            segs_list.append(window_segs_list)
            scores_list.append(window_scores_list)
        
        # Combine results from all windows
        combined_segs = [torch.cat([w[i] for w in segs_list]) for i in range(len(tokens))]
        combined_scores = [torch.cat([w[i] for w in scores_list]) for i in range(len(tokens))]
        
        results = []
        for segs, scores in zip(combined_segs, combined_scores):
            # Top-k selection
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]
            
            # NMS
            segs, scores = self.batched_nms_val(segs[idx].float().cpu(), scores[idx].float().cpu())
            
            # Convert to timestamps
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']
                
                segs = segs * self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)
            
            results.append({'segments': segs, 'scores': scores})
        
        return results
    
    def _collect_segments_val(self, fpn_points, fpn_logits, fpn_offsets, fpn_masks, ext_scores):
        """Collect and NMS segments from FPN outputs for validation."""
        points = torch.cat(fpn_points)
        logits = torch.cat(fpn_logits, dim=1)[0]
        offsets = torch.cat(fpn_offsets, dim=1)[0]
        masks = torch.cat(fpn_masks, dim=1)[0]
        
        # Filter by mask and confidence
        scores = logits.sigmoid()
        valid = masks & (scores > self.pre_nms_thresh)
        
        points = points[valid]
        scores = scores[valid]
        offsets = offsets[valid]
        
        # Top-k selection
        if len(scores) > self.pre_nms_topk:
            topk_idx = scores.topk(self.pre_nms_topk).indices
            points = points[topk_idx]
            scores = scores[topk_idx]
            offsets = offsets[topk_idx]
        
        # Convert offsets to segments
        segs = torch.stack([
            points[:, 0] - offsets[:, 0] * points[:, 3],
            points[:, 0] + offsets[:, 1] * points[:, 3]
        ], dim=-1)
        
        # Filter short segments
        seg_lens = segs[:, 1] - segs[:, 0]
        valid = seg_lens > self.seg_len_thresh
        segs = segs[valid]
        scores = scores[valid]
        
        return segs, scores
    
    def _save_best_model(self):
        """Save best model checkpoint."""
        if get_rank() != 0:
            return
        
        model_dir = os.path.join(self.opt['_root'], 'models')
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'metric_name': self.metric_name,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'best.pth'))
        self.logger.write(f"  Saved best model at epoch {self.epoch}")
    
    def _save_validation_history(self):
        """Save validation history to JSON file with all metrics."""
        if get_rank() != 0:
            return
        
        history_path = os.path.join(self.opt['_root'], 'validation_history.json')
        
        # Convert history: each epoch has a dict of all metrics
        history_converted = {}
        for epoch, metrics in self.validation_history.items():
            if isinstance(metrics, dict):
                # New format: dict of all metrics
                history_converted[str(epoch)] = {k: float(v) for k, v in metrics.items()}
            else:
                # Legacy format: single metric value (backwards compatibility)
                history_converted[str(epoch)] = {self.metric_name: float(metrics)}
        
        history_data = {
            'tracked_metric': self.metric_name,
            'best_metric': float(self.best_metric),
            'best_epoch': self.best_epoch,
            'history': history_converted
        }
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)


class TrainerMultiDataset(TrainerAuxiliary):
    """
    Multi-dataset trainer that trains on multiple datasets simultaneously and
    validates on each dataset separately, reporting metrics per dataset.
    
    Features:
    - Training: Combines multiple datasets using ConcatDataset
    - Validation: Separate validation on each dataset with per-dataset metrics
    - Best model selection based on primary dataset or average metric
    - Support for pretrained checkpoint loading
    
    Configuration (in YAML):
        train:
          datasets:  # List of training datasets
            - name: ego4d
              data:
                anno_file: /path/to/ego4d.json
                vid_feat_dir: /path/to/ego4d_vid
                text_feat_dir: /path/to/ego4d_text
                ext_score_dir: null
                clip_size: 15
                clip_stride: 15
                max_vid_len: 4608
                max_text_len: 1
                max_num_text: 8
                crop_ratio: [0.9, 1.0]
            - name: tacos
              data:
                anno_file: /path/to/tacos.json
                vid_feat_dir: /path/to/tacos_vid
                text_feat_dir: /path/to/tacos_text
                ext_score_dir: null
                clip_size: 15
                clip_stride: 15
                max_vid_len: 2304
                max_text_len: 1
                max_num_text: 4
                crop_ratio: [0.9, 1.0]
          
          batch_size: 4
          num_workers: 4
          # ... other training params
          
          validation:
            enable: true
            val_every: 1
            patience: -1
            metric: "R1@0.3"
            batch_size: 1
            num_workers: 0
          
          pretrain_checkpoint: null
        
        eval:
          datasets:  # List of validation datasets
            - name: ego4d
              data:
                split: val
                anno_file: /path/to/ego4d.json
                vid_feat_dir: /path/to/ego4d_vid
                text_feat_dir: /path/to/ego4d_text
                ext_score_dir: null
                max_vid_len: 4608
                max_text_len: 1
                clip_size: 15
                clip_stride: 15
            - name: tacos
              data:
                split: test
                anno_file: /path/to/tacos.json
                vid_feat_dir: /path/to/tacos_vid
                text_feat_dir: /path/to/tacos_text
                ext_score_dir: null
                max_vid_len: 2304
                max_text_len: 1
                clip_size: 15
                clip_stride: 15
          
          primary_dataset: ego4d  # or "average" for averaging across datasets
          ranks: [1, 5]
          iou_threshs: [0.3, 0.5]
          # ... other eval params (nms, etc.)
    """

    def __init__(self, opt):
        # Store pretrain checkpoint before super().__init__
        self.pretrain_checkpoint = opt['train'].get('pretrain_checkpoint', None)
        
        # Check if we're using multi-dataset config
        self.multi_dataset_mode = 'datasets' in opt['train']
        
        if self.multi_dataset_mode:
            # IMPORTANT: build train datasets under a fixed seed so every rank
            # constructs identical grouped samples (VideoCentricDataset grouping
            # uses random ops during dataset initialization).
            fix_random_seed(opt.get('seed', 2022))
            # Multi-dataset training: need to create combined dataset before super().__init__
            self._setup_multi_dataset_training(opt)
        
        # Initialize parent class (TrainerAuxiliary)
        # This will use self.dataset if we've set it up
        if not self.multi_dataset_mode:
            super().__init__(opt)
        else:
            # Custom initialization for multi-dataset
            self._init_multi_dataset(opt)
        
        # Load pretrained checkpoint if specified
        if self.pretrain_checkpoint:
            self._load_pretrain_checkpoint(self.pretrain_checkpoint)
        
        # Setup validation
        self._setup_validation(opt)
    
    def _setup_multi_dataset_training(self, opt):
        """Setup training datasets and combine them."""
        from torch.utils.data import ConcatDataset
        
        self.train_dataset_names = []
        train_datasets = []
        self.train_dataset_lengths = []
        
        num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        
        for ds_config in opt['train']['datasets']:
            ds_name = ds_config['name']
            ds_data_config = deepcopy(ds_config['data'])
            ds_data_config['name'] = ds_data_config.get('name', 'video_centric')
            ds_data_config['split'] = ds_data_config.get('split', 'train')
            
            dataset = make_dataset(ds_data_config, num_epochs=num_epochs, is_training=True)
            train_datasets.append(dataset)
            self.train_dataset_names.append(ds_name)
            self.train_dataset_lengths.append(len(dataset))
            print0(f"  Loaded training dataset: {ds_name} ({len(dataset)} samples)")
        
        # Combine datasets
        self.combined_dataset = ConcatDataset(train_datasets)
        self.individual_train_datasets = train_datasets
        print0(f"Combined training dataset: {len(self.combined_dataset)} total samples")

    def _build_multi_dataset_sample_weights(self, opt):
        """
        Build per-sample weights for ConcatDataset based on:
            train.dataset_weights: {dataset_name: group_weight}

        Dataset names are matched against train.datasets[*].name.
        """
        weight_cfg = opt['train'].get('dataset_weights')
        if not weight_cfg:
            return None, None
        if not isinstance(weight_cfg, dict):
            raise ValueError("train.dataset_weights must be a dict if provided")
        unknown_names = sorted(set(weight_cfg.keys()) - set(self.train_dataset_names))
        if unknown_names:
            print0(
                "Warning: train.dataset_weights contains unknown dataset names: "
                f"{unknown_names}. They will be ignored."
            )

        if not hasattr(self, 'train_dataset_lengths') or not self.train_dataset_lengths:
            return None, None

        group_weights = []
        per_sample_weights = []
        report_rows = []

        for ds_name, ds_len in zip(self.train_dataset_names, self.train_dataset_lengths):
            group_weight = float(weight_cfg.get(ds_name, 1.0))
            if group_weight < 0:
                raise ValueError(
                    f"train.dataset_weights['{ds_name}'] must be >= 0, got {group_weight}"
                )
            group_weights.append(group_weight)
            if ds_len <= 0:
                raise ValueError(f"Dataset '{ds_name}' has invalid length {ds_len}")
            per_weight = group_weight / ds_len if group_weight > 0 else 0.0
            per_sample_weights.extend([per_weight] * ds_len)
            report_rows.append((ds_name, ds_len, group_weight))

        group_weight_sum = sum(group_weights)
        if group_weight_sum <= 0:
            raise ValueError("Sum of train.dataset_weights must be > 0")

        sampling_report = OrderedDict()
        for ds_name, ds_len, group_weight in report_rows:
            sampling_report[ds_name] = {
                'size': ds_len,
                'group_weight': group_weight,
                'expected_prob': group_weight / group_weight_sum,
            }

        return torch.as_tensor(per_sample_weights, dtype=torch.double), sampling_report
    
    def _init_multi_dataset(self, opt):
        """Initialize trainer with combined multi-dataset."""
        self.opt = opt
        
        # Set random seed
        rng = fix_random_seed(opt.get('seed', 2022))
        
        # Build model and EMA
        self.model = make_models_net(opt).cuda()
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.ema_beta = opt['train'].get('ema_beta', 0.999)
        
        # Use combined dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.dataset = self.combined_dataset
        
        # Create a wrapper to handle set_epoch for ConcatDataset
        class ConcatDatasetWrapper:
            def __init__(self, concat_dataset, individual_datasets):
                self.concat_dataset = concat_dataset
                self.individual_datasets = individual_datasets
            
            def __len__(self):
                return len(self.concat_dataset)
            
            def __getitem__(self, idx):
                return self.concat_dataset[idx]
            
            def set_epoch(self, epoch):
                for ds in self.individual_datasets:
                    if hasattr(ds, 'set_epoch'):
                        ds.set_epoch(epoch)
        
        self.dataset = ConcatDatasetWrapper(self.combined_dataset, self.individual_train_datasets)
        self.dataset_sample_weights, self.dataset_sampling_report = \
            self._build_multi_dataset_sample_weights(opt)
        
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataloader, self.sampler = make_dataloader(
            self.combined_dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank(),
            sample_weights=self.dataset_sample_weights,
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0
        
        # Build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')
        
        # Build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1,))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'train.txt'))
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tb'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
            if self.dataset_sampling_report:
                self.logger.write("Dataset-weighted sampling enabled:")
                for ds_name, item in self.dataset_sampling_report.items():
                    self.logger.write(
                        f"  - {ds_name}: size={item['size']}, "
                        f"group_weight={item['group_weight']:.6g}, "
                        f"expected_prob={item['expected_prob'] * 100:.2f}%"
                    )
        else:
            self.logger = self.tb_writer = self.loss_meters = self.timer = None
        
        # Load checkpoint if resuming
        if opt['_resume']:
            self.load()
            barrier()
        
        # Set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()])
            self._ema_init()
        
        # Register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride
        
        # Register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']
        
        # Register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')
        
        # Initialize auxiliary losses (from TrainerAuxiliary)
        vid_embd_dim = opt['model']['vid_net']['embd_dim']
        self.ds_contrastive = opt['train']['loss_aux']['ds_contrast']['enable']
        self.gt_contrastive = opt['train']['loss_aux']['gt_contrast']['enable']
        self.early_fusion = opt['model'].get('early_fusion', True)
        self.use_mst = opt['model'].get('use_mst', False)
        
        if self.early_fusion and self.logger:
            self.logger.write("Early fusion enabled")
        
        if self.ds_contrastive:
            ds_loss_type = AUX_LOSS_REGISTRY[opt['train']['loss_aux']['ds_contrast'].get('type', 'ds_contrastive')]
            self.ds_contrastive_loss = ds_loss_type(opt['train']['loss_aux']['ds_contrast'], vid_embd_dim).cuda()
            self.ds_contrastive_weight = opt['train']['loss_aux']['ds_contrast']['weight']
        else:
            self.ds_contrastive_weight = 0.0
        
        if self.gt_contrastive:
            gt_loss_type = AUX_LOSS_REGISTRY[opt['train']['loss_aux']['gt_contrast'].get('type', 'gt_point_contrastive')]
            self.gt_contrastive_loss = gt_loss_type(opt['train']['loss_aux']['gt_contrast'], vid_embd_dim).cuda()
            self.gt_contrastive_weight = opt['train']['loss_aux']['gt_contrast']['weight']
            self.loss_aux_gt_type = opt['train']['loss_aux']['gt_contrast'].get('gt_type', 'point')
            self.loss_aux_span_radius = opt['train']['loss_aux']['gt_contrast'].get('span_radius', self.center_sampling_radius)
            self.span_contr_gt = opt['train']['loss_aux']['gt_contrast'].get('span_contr_gt', False)
        else:
            self.gt_contrastive_weight = 0.0
        
        # Log dataset info
        if self.logger:
            self.logger.write(f"Multi-dataset training enabled:")
            for ds_name in self.train_dataset_names:
                self.logger.write(f"  - {ds_name}")
            self.logger.write(f"Combined dataset size: {len(self.combined_dataset)}")
    
    def _setup_validation(self, opt):
        """Setup validation datasets and dataloaders."""
        val_config = opt['train'].get('validation', {})
        self.validation_enabled = val_config.get('enable', False)
        self.val_every = val_config.get('val_every', 1)
        self.patience = val_config.get('patience', -1)
        self.metric_name = val_config.get('metric', 'R1@0.3')
        self._parse_metric_name(self.metric_name)
        
        if not self.validation_enabled:
            return
        
        if self.logger:
            self.logger.write(f"Validation enabled: every {self.val_every} epochs")
            self.logger.write(f"Tracking metric: {self.metric_name}")
            if self.patience > 0:
                self.logger.write(f"Early stopping patience: {self.patience}")
        
        # Validation settings
        self.val_batch_size = val_config.get('batch_size', 1)
        self.val_num_workers = val_config.get('num_workers', 0)
        
        # Check for multi-dataset validation
        self.multi_val_mode = 'datasets' in opt['eval']
        
        if self.multi_val_mode:
            # Multiple validation datasets
            self.val_dataloaders = OrderedDict()
            self.val_dataset_names = []
            self.val_iou_threshs = {}           # Per-dataset iou_threshs (falls back to global)
            self.val_batchify_text_queries = {}  # Per-dataset batchify_text_queries (falls back to global)
            
            rng = fix_random_seed(opt.get('seed', 2022))
            
            global_iou_threshs = opt['eval'].get('iou_threshs', (0.3, 0.5))
            global_batchify = opt['eval'].get('batchify_text_queries', True)
            
            for ds_config in opt['eval']['datasets']:
                ds_name = ds_config['name']
                ds_data_config = deepcopy(ds_config['data'])
                ds_data_config['name'] = ds_data_config.get('name', 'video_centric')
                
                val_dataset = make_dataset(ds_data_config, is_training=False)
                val_dataloader, _ = make_dataloader(
                    val_dataset, is_training=False, generator=rng,
                    batch_size=self.val_batch_size, num_workers=self.val_num_workers
                )
                
                self.val_dataloaders[ds_name] = val_dataloader
                self.val_dataset_names.append(ds_name)

                # Per-dataset iou_threshs: use dataset-specific if provided, else global
                ds_iou_threshs = ds_config.get('iou_threshs', global_iou_threshs)
                self.val_iou_threshs[ds_name] = np.array(ds_iou_threshs)

                # Per-dataset batchify_text_queries: use dataset-specific if provided, else global
                ds_batchify = ds_config.get('batchify_text_queries', global_batchify)
                self.val_batchify_text_queries[ds_name] = ds_batchify

                if self.logger:
                    self.logger.write(
                        f"  Loaded validation dataset: {ds_name} "
                        f"({len(val_dataset)} samples, iou_threshs={list(ds_iou_threshs)}, "
                        f"batchify_text_queries={ds_batchify})"
                    )
            
            # Primary dataset for best model selection
            self.primary_dataset = opt['eval'].get('primary_dataset', 'average')
            if self.logger:
                self.logger.write(f"Primary dataset for best model: {self.primary_dataset}")
        else:
            # Single validation dataset (backwards compatible)
            val_data_config = deepcopy(opt['eval']['data'])
            rng = fix_random_seed(opt.get('seed', 2022))
            
            self.val_dataset = make_dataset(val_data_config, is_training=False)
            self.val_dataloader, _ = make_dataloader(
                self.val_dataset, is_training=False, generator=rng,
                batch_size=self.val_batch_size, num_workers=self.val_num_workers
            )
            self.primary_dataset = 'single'
        
        # Validation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.batched_nms_val = lambda segs, scores: batched_nms(segs, scores, **opt['eval']['nms'])
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']
        self.max_text_len_eval = opt['eval'].get('max_text_len', 24)
        self.batchify_text_queries = opt['eval'].get('batchify_text_queries', True)
        
        # Calculate min_chunk_size for sliding window evaluation
        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        self.min_chunk_size = min_chunk_size
        
        # Sliding window parameters
        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')
        
        # Best model tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.validation_history = {}
    
    def _parse_metric_name(self, metric):
        """Parse metric name like 'R1@0.3' into rank and IoU threshold.
        
        Special metric names:
            - "avg" or "mean": Average of all R1 and R5 metrics across all IoU thresholds
            - "mIoU": Mean IoU of top-1 prediction vs ground truth
        """
        self.use_avg_metric = False
        self.use_miou_metric = False
        
        if metric.lower() in ('avg', 'mean', 'average'):
            # Use average of all R1 and R5 metrics
            self.use_avg_metric = True
            self.track_rank = None
            self.track_iou = None
        elif metric.lower() == 'miou':
            # Use mean IoU of top-1 prediction vs ground truth
            self.use_miou_metric = True
            self.track_rank = None
            self.track_iou = None
        else:
            # Parse standard metric name like 'R1@0.3'
            parts = metric.replace("R", "").split("@")
            self.track_rank = int(parts[0])
            self.track_iou = float(parts[1])
    
    def _load_pretrain_checkpoint(self, checkpoint_path):
        """Load model weights from a pretrained checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrain checkpoint not found: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Unwrap DDP if needed — checkpoint keys don't have 'module.' prefix
        model = self._unwrap(self.model)
        
        if 'model_ema' in ckpt:
            model.load_state_dict(ckpt['model_ema'])
            self.model_ema.load_state_dict(ckpt['model_ema'])
            print0(f"Loaded pretrained model (EMA weights) from: {checkpoint_path}")
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            self.model_ema.load_state_dict(ckpt['model'])
            print0(f"Loaded pretrained model from: {checkpoint_path}")
        else:
            model.load_state_dict(ckpt)
            self.model_ema.load_state_dict(ckpt)
            print0(f"Loaded pretrained model (direct state dict) from: {checkpoint_path}")
        
        if get_rank() == 0:
            self.logger.write(f"Loaded pretrained checkpoint: {checkpoint_path}")
    
    def run(self):
        """Training loop with multi-dataset validation after each epoch."""
        print0("=" * 70)
        print0("MULTI-DATASET TRAINING")
        print0("=" * 70)
        if self.multi_dataset_mode:
            print0(f"Training datasets: {', '.join(self.train_dataset_names)}")
        if self.validation_enabled and self.multi_val_mode:
            print0(f"Validation datasets: {', '.join(self.val_dataset_names)}")
            print0(f"Primary dataset: {self.primary_dataset}")
        print0("=" * 70)
        
        print0("Training started.")
        while self.epoch < self.num_epochs:
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            
            for data_list in self.dataloader:
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                
                if get_rank() == 0:
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log()
            
            self.epoch += 1
            self.checkpoint()
            barrier()
            
            # Run validation after epoch if enabled
            if self.validation_enabled and self.epoch % self.val_every == 0:
                should_stop = self._run_validation_epoch()
                
                # Early stopping check (synchronized across ranks)
                if self.opt['_distributed']:
                    import torch.distributed as dist
                    should_stop_tensor = torch.tensor([int(should_stop)], device='cuda')
                    dist.broadcast(should_stop_tensor, src=0)
                    should_stop = bool(should_stop_tensor.item())
                
                if should_stop:
                    print0(f"Early stopping triggered at epoch {self.epoch}")
                    print0(f"Best {self.metric_name}: {self.best_metric:.4f} at epoch {self.best_epoch}")
                    break
        
        print0("Training completed.")
        if self.validation_enabled:
            print0(f"Best {self.metric_name}: {self.best_metric:.4f} at epoch {self.best_epoch}")
    
    def _run_validation_epoch(self):
        """Run validation on all datasets and return whether to stop early."""
        if get_rank() == 0:
            if self.multi_val_mode:
                val_metric, all_metrics = self._validate_multi_dataset()
            else:
                val_metric, all_metrics = self._validate_single_dataset(self.val_dataloader, "single")
            
            # Track validation history
            self.validation_history[self.epoch] = all_metrics
            self._save_validation_history()
            
            # Check for improvement
            if val_metric > self.best_metric:
                self.best_metric = val_metric
                self.best_epoch = self.epoch
                self.epochs_without_improvement = 0
                self._save_best_model()
                self.logger.write(f"  New best {self.metric_name}: {val_metric:.4f} at epoch {self.epoch}")
            else:
                self.epochs_without_improvement += 1
                self.logger.write(f"  No improvement for {self.epochs_without_improvement} validation(s)")
                self.logger.write(f"  Current: {val_metric:.4f}, Best: {self.best_metric:.4f} (epoch {self.best_epoch})")
            
            # Log to tensorboard
            if self.tb_writer:
                self.tb_writer.add_scalar(f'val/{self.metric_name}', val_metric, self.epoch)
                self.tb_writer.add_scalar('val/best_metric', self.best_metric, self.epoch)
                for metric_name, metric_val in all_metrics.items():
                    self.tb_writer.add_scalar(f'val/{metric_name}', metric_val, self.epoch)
            
            # Check early stopping
            if self.patience > 0 and self.epochs_without_improvement >= self.patience:
                torch.cuda.empty_cache()
                return True
        
        barrier()
        torch.cuda.empty_cache()
        return False
    
    @torch.no_grad()
    def _validate_multi_dataset(self):
        """Validate on all datasets and return primary metric and all metrics."""
        self.logger.write(f"\n{'='*60}")
        self.logger.write(f"=== Multi-Dataset Validation at epoch {self.epoch} ===")
        self.logger.write(f"{'='*60}")
        
        all_metrics = {}
        dataset_tracked_metrics = {}
        
        for ds_name, val_dataloader in self.val_dataloaders.items():
            tracked_metric, ds_metrics = self._validate_single_dataset(val_dataloader, ds_name)
            
            # Store with dataset prefix
            for metric_name, metric_val in ds_metrics.items():
                all_metrics[f"{ds_name}/{metric_name}"] = metric_val
            
            dataset_tracked_metrics[ds_name] = tracked_metric
        
        # Calculate primary metric for best model selection
        if self.primary_dataset == 'average':
            primary_metric = np.mean(list(dataset_tracked_metrics.values()))
            self.logger.write(f"\nAverage {self.metric_name}: {primary_metric * 100:.2f}%")
        else:
            primary_metric = dataset_tracked_metrics.get(self.primary_dataset, 0.0)
            self.logger.write(f"\nPrimary ({self.primary_dataset}) {self.metric_name}: {primary_metric * 100:.2f}%")
        
        all_metrics['primary_metric'] = primary_metric
        
        return primary_metric, all_metrics
    
    @torch.no_grad()
    def _validate_single_dataset(self, val_dataloader, ds_name):
        """Validate on a single dataset and return tracked metric and all metrics."""
        self.logger.write(f"\n--- {ds_name} ---")
        
        self.model_ema.eval()
        
        # Use per-dataset iou_threshs if available, else fall back to global
        ds_iou_threshs = (
            self.val_iou_threshs[ds_name]
            if hasattr(self, 'val_iou_threshs') and ds_name in self.val_iou_threshs
            else self.iou_threshs
        )

        # Use per-dataset batchify_text_queries if available, else fall back to global
        ds_batchify = (
            self.val_batchify_text_queries[ds_name]
            if hasattr(self, 'val_batchify_text_queries') and ds_name in self.val_batchify_text_queries
            else self.batchify_text_queries
        )
        
        counts = np.zeros((len(self.ranks), len(ds_iou_threshs)))
        text_cnt = 0
        iou_sum = 0.0  # accumulate top-1 IoU for mIoU
        
        for data_list in val_dataloader:
            for data in data_list:
                results = self._predict_val(data, batchify_text_queries=ds_batchify)
                targets = data['segment']
                
                for result, target in zip(results, targets):
                    segs, scores = result['segments'], result['scores']
                    idx = scores.argsort(descending=True)
                    segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                    target = torch.as_tensor(target, dtype=torch.float)
                    target = target.expand(len(segs), -1)
                    
                    iou_topk = iou(segs, target)
                    iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                    counts += (iou_n[:, None] >= ds_iou_threshs[None])
                    
                    # mIoU: IoU between top-1 prediction and ground truth
                    iou_sum += iou_topk[0].item()
                
                text_cnt += len(targets)
        
        # Calculate metrics
        metrics = counts / text_cnt
        miou = iou_sum / text_cnt
        
        # Build metrics dict
        all_metrics = {}
        log_str = f"  Results: "
        for i, rank in enumerate(self.ranks):
            for j, thresh in enumerate(ds_iou_threshs):
                metric_val = metrics[i, j]
                metric_name = f"R{rank}@{thresh:.1f}"
                all_metrics[metric_name] = metric_val
                log_str += f"{metric_name}={metric_val * 100:.2f}% "
        
        # mIoU metric
        all_metrics['mIoU'] = miou
        log_str += f"mIoU={miou * 100:.2f}% "
        
        # Compute average of all recall metrics (for "avg" metric option)
        avg_metric = metrics.mean()
        all_metrics['avg'] = avg_metric
        log_str += f"avg={avg_metric * 100:.2f}%"
        
        self.logger.write(log_str)
        
        # Get tracked metric
        if self.use_miou_metric:
            # Use mean IoU of top-1 prediction vs ground truth
            tracked_metric = miou
        elif self.use_avg_metric:
            # Use average of all R1 and R5 metrics across all IoU thresholds
            tracked_metric = avg_metric
        else:
            # Use specific metric like R1@0.3
            rank_idx = self.ranks.index(self.track_rank) if self.track_rank in self.ranks else 0
            iou_idx = np.where(np.isclose(ds_iou_threshs, self.track_iou))[0]
            iou_idx = iou_idx[0] if len(iou_idx) > 0 else 0
            tracked_metric = metrics[rank_idx, iou_idx]
        
        return tracked_metric, all_metrics
    
    def _batchify_text_for_eval(self, text_list):
        """Batch text queries for evaluation."""
        bs = len(text_list)
        n = max(len(t) for t in text_list)
        text_dim = text_list[0][0].size(0)
        text_lens = [[t.size(-1) for t in ts] for ts in text_list]
        max_text_len = min(max(max(lens) for lens in text_lens), self.max_text_len_eval)
        
        text = text_list[0][0].new_full((bs, n, text_dim, max_text_len), 0.)
        text_masks = torch.zeros(bs, n, max_text_len, dtype=torch.bool)
        text_size = torch.zeros(bs, dtype=torch.long)
        
        for i, (ts, lens) in enumerate(zip(text_list, text_lens)):
            text_size[i] = len(ts)
            for j, (t, l) in enumerate(zip(ts, lens)):
                actual_len = min(l, max_text_len)
                text[i, j, :, :actual_len] = t[:, :actual_len]
                text_masks[i, j, :actual_len] = True
        
        return text, text_masks, text_size
    
    def _predict_val(self, data, batchify_text_queries=None):
        """Predict event segments for validation.

        Args:
            data: data dict from the dataloader.
            batchify_text_queries: if True, all text queries for a video are
                encoded and forwarded together (faster, more VRAM).  If False,
                each query is encoded and forwarded individually (slower but
                matches the single-query inference path).  Defaults to
                ``self.batchify_text_queries``.
        """
        if batchify_text_queries is None:
            batchify_text_queries = self.batchify_text_queries

        # Parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens,)
        
        # Parse video
        vid = data['vid']
        vid_len = vid.size(-1)
        
        # Encode text queries
        if batchify_text_queries:
            # All queries encoded together in a single batched call
            text, text_masks, text_size = self._batchify_text_for_eval([tokens])
            text = text.cuda(non_blocking=True)
            text_masks = text_masks.cuda(non_blocking=True)
            text_size = text_size.cuda(non_blocking=True)
            text_encoded, text_masks_encoded = self.model_ema.encode_text2(text, text_masks, text_size)
        else:
            # Each query encoded separately (matches single-query inference path)
            text_list, text_mask_list = [], []
            for token in tokens:
                t = token[None].cuda(non_blocking=True)
                t_mask = t.new_full((1, 1, t.size(-1)), 1, dtype=torch.bool).cuda(non_blocking=True)
                t_enc, t_mask_enc = self.model_ema.encode_text(t, t_mask)
                text_list.append(t_enc)
                text_mask_list.append(t_mask_enc)

        # External scores
        ext_scores = data.get('ext_scores')
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]
        
        # Sliding window evaluation
        window_size = min(self.window_size or vid_len, vid_len, self.max_vid_len)
        window_stride = self.window_stride or window_size
        
        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = [], [], []
        
        idx = 0
        while idx <= max(n, 0):
            windows.append(vid[..., idx:idx + window_size])
            window_offsets.append(idx)
            if ext_scores is not None:
                window_ext_scores.append(ext_scores[..., idx:idx + window_size])
            else:
                window_ext_scores.append(None)
            idx += window_stride
            if n <= 0:
                break
        
        if n > 0 and n % window_stride > 0:
            windows.append(vid[..., -window_size:])
            window_offsets.append(n)
            if ext_scores is not None:
                window_ext_scores.append(ext_scores[..., -window_size:])
            else:
                window_ext_scores.append(None)
        
        # Calculate input_vid_len for padding
        stride = self.min_chunk_size * self.vid_stride
        input_vid_len = (window_size + (stride - 1)) // stride * stride
        
        segs_list, scores_list = [], []
        
        for window, window_offset, window_ext in zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)

            window_segs_list, window_scores_list = [], []

            if batchify_text_queries:
                # Forward pass with all queries batched together
                if self.early_fusion:
                    window_proj, window_mask_proj = self.model_ema.vid_proj(window, window_mask)
                    window_fused, window_mask_fused = self.model_ema.fusion(
                        window_proj, window_mask_proj, text_encoded, text_masks_encoded, text_size
                    )
                else:
                    window_fused, window_mask_fused = window, window_mask
                
                fpn, fpn_masks, _, _ = self.model_ema.encode_video(window_fused, window_mask_fused)
                
                if self.use_mst:
                    fpn_logits, _, fpn_offsets, _ = self.model_ema.fuse_and_predict_mst(
                        fpn, fpn_masks, text_encoded, text_masks_encoded, text_size
                    )
                else:
                    fpn_logits, _, fpn_offsets, _ = self.model_ema.fuse_and_predict(
                        fpn, fpn_masks, text_encoded, text_masks_encoded, text_size
                    )
                
                fpn_n_points = [m.size(-1) for m in fpn_masks]
                fpn_points = self.pt_gen(fpn_n_points)
                fpn_masks_squeezed = [m.squeeze(1) for m in fpn_masks]
                
                for query_idx in range(len(tokens)):
                    query_logits = tuple(layer[query_idx:query_idx+1] for layer in fpn_logits)
                    query_offsets = tuple(layer[query_idx:query_idx+1] for layer in fpn_offsets)
                    
                    segs, scores = self._collect_segments_val(
                        fpn_points, query_logits, query_offsets, fpn_masks_squeezed,
                        window_ext[query_idx] if window_ext is not None else None
                    )
                    segs = segs + window_offset / self.vid_stride
                    window_segs_list.append(segs.cpu())
                    window_scores_list.append(scores.cpu())
            else:
                # Forward pass one query at a time
                for q_idx, (text_enc, text_mask_enc) in enumerate(zip(text_list, text_mask_list)):
                    if self.early_fusion:
                        w, wm = self.model_ema.vid_proj(window, window_mask)
                        w, wm = self.model_ema.fusion(w, wm, text_enc, text_mask_enc)
                    else:
                        w, wm = window, window_mask

                    fpn, fpn_masks, _, _ = self.model_ema.encode_video(w, wm)

                    if self.use_mst:
                        fpn_logits, _, fpn_offsets, _ = self.model_ema.fuse_and_predict_mst(
                            fpn, fpn_masks, text_enc, text_mask_enc
                        )
                    else:
                        fpn_logits, _, fpn_offsets, _ = self.model_ema.fuse_and_predict(
                            fpn, fpn_masks, text_enc, text_mask_enc
                        )

                    fpn_n_points = [m.size(-1) for m in fpn_masks]
                    fpn_points = self.pt_gen(fpn_n_points)
                    fpn_masks_squeezed = [m.squeeze(1) for m in fpn_masks]

                    # fuse_and_predict returns results for a single query (batch=1)
                    query_logits = tuple(layer[0:1] for layer in fpn_logits)
                    query_offsets = tuple(layer[0:1] for layer in fpn_offsets)

                    segs, scores = self._collect_segments_val(
                        fpn_points, query_logits, query_offsets, fpn_masks_squeezed,
                        window_ext[q_idx] if window_ext is not None else None
                    )
                    segs = segs + window_offset / self.vid_stride
                    window_segs_list.append(segs.cpu())
                    window_scores_list.append(scores.cpu())
            
            segs_list.append(window_segs_list)
            scores_list.append(window_scores_list)
        
        # Combine results from all windows
        combined_segs = [torch.cat([w[i] for w in segs_list]) for i in range(len(tokens))]
        combined_scores = [torch.cat([w[i] for w in scores_list]) for i in range(len(tokens))]
        
        results = []
        for segs, scores in zip(combined_segs, combined_scores):
            # Top-k selection
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]
            
            # NMS
            segs, scores = self.batched_nms_val(segs[idx].float().cpu(), scores[idx].float().cpu())
            
            # Convert to timestamps
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']
                
                segs = segs * self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)
            
            results.append({'segments': segs, 'scores': scores})
        
        return results
    
    def _collect_segments_val(self, fpn_points, fpn_logits, fpn_offsets, fpn_masks, ext_scores):
        """Collect and NMS segments from FPN outputs for validation."""
        points = torch.cat(fpn_points)
        logits = torch.cat(fpn_logits, dim=1)[0]
        offsets = torch.cat(fpn_offsets, dim=1)[0]
        masks = torch.cat(fpn_masks, dim=1)[0]
        
        # Filter by mask and confidence
        scores = logits.sigmoid()
        valid = masks & (scores > self.pre_nms_thresh)
        
        points = points[valid]
        scores = scores[valid]
        offsets = offsets[valid]
        
        # Top-k selection
        if len(scores) > self.pre_nms_topk:
            topk_idx = scores.topk(self.pre_nms_topk).indices
            points = points[topk_idx]
            scores = scores[topk_idx]
            offsets = offsets[topk_idx]
        
        # Convert offsets to segments
        segs = torch.stack([
            points[:, 0] - offsets[:, 0] * points[:, 3],
            points[:, 0] + offsets[:, 1] * points[:, 3]
        ], dim=-1)
        
        # Filter short segments
        seg_lens = segs[:, 1] - segs[:, 0]
        valid = seg_lens > self.seg_len_thresh
        segs = segs[valid]
        scores = scores[valid]
        
        return segs, scores
    
    def _save_best_model(self):
        """Save best model checkpoint."""
        if get_rank() != 0:
            return
        
        model_dir = os.path.join(self.opt['_root'], 'models')
        model_ckpt = {
            'model': self._unwrap(self.model).state_dict(),
            'model_ema': self.model_ema.state_dict(),
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'metric_name': self.metric_name,
        }
        torch.save(model_ckpt, os.path.join(model_dir, 'best.pth'))
        self.logger.write(f"  Saved best model at epoch {self.epoch}")
    
    def _save_validation_history(self):
        """Save validation history to JSON file with all metrics."""
        if get_rank() != 0:
            return
        
        history_path = os.path.join(self.opt['_root'], 'validation_history.json')
        
        history_converted = {}
        for epoch, metrics in self.validation_history.items():
            if isinstance(metrics, dict):
                history_converted[str(epoch)] = {k: float(v) for k, v in metrics.items()}
            else:
                history_converted[str(epoch)] = {self.metric_name: float(metrics)}
        
        history_data = {
            'tracked_metric': self.metric_name,
            'best_metric': float(self.best_metric),
            'best_epoch': self.best_epoch,
            'primary_dataset': self.primary_dataset if hasattr(self, 'primary_dataset') else 'single',
            'history': history_converted
        }
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)


class EvaluatorOriginal:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        # self.model = PtTransformer(opt['model']).cuda()
        self.model = make_models_net(opt).cuda()
        self.load_model()
        self.model.eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"eval_{opt['_ckpt']}.txt"))
        
        # initialize prediction storage
        self.predictions = {}

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
        self.iou_sum = 0.0  # accumulate top-1 IoU for mIoU

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']
        self.max_text_len = opt['eval'].get('max_text_len', 24)
        self.batchify_text_queries = opt['eval'].get('batchify_text_queries', True)
        self.text_batch_size = opt['eval'].get('text_batch_size', 0)
        if self.batchify_text_queries:
            print("Batchify text queries for evaluation")
        else:
            print("Single text query processing for evaluation")

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        print0("Evaluation started.")
        start_time = time.time()
        for data_list in self.dataloader:
            data = data_list[0]
            results = self.predict(data)
            targets = data['segment']
            vid_id = data['vid_id']
            assert len(results) == len(targets)

            # Store predictions for this video
            if vid_id not in self.predictions:
                self.predictions[vid_id] = {
                    'queries': [],
                    'recall_at_iou': {}
                }

            video_iou_counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
            
            for query_idx, (result, target) in enumerate(zip(results, targets)):
                segs, scores = result['segments'], result['scores']
                idx = scores.argsort(descending=True)
                segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)
                
                iou_topk = iou(segs, target)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
                video_iou_counts += (iou_n[:, None] >= self.iou_threshs[None])
                self.iou_sum += iou_topk[0].item()  # mIoU: top-1 IoU
                
                # Store query predictions (top-5 predictions)
                top5_segs = segs[:5] if len(segs) >= 5 else segs
                top5_scores = scores[:5] if len(scores) >= 5 else scores
                
                query_data = {
                    'query_id': query_idx,
                    'ground_truth': target[0].cpu().numpy().tolist(),
                    'predictions': []
                }
                
                for seg, score in zip(top5_segs, top5_scores):
                    query_data['predictions'].append({
                        'segment': seg.cpu().numpy().tolist(),
                        'score': score.item()
                    })
                
                self.predictions[vid_id]['queries'].append(query_data)
            
            # Calculate recall at IoU for this video
            video_metrics = video_iou_counts / len(targets)
            for i, rank in enumerate(self.ranks):
                for j, thresh in enumerate(self.iou_threshs):
                    key = f"Rank@{rank}_IoU@{thresh:.1f}"
                    self.predictions[vid_id]['recall_at_iou'][key] = video_metrics[i, j].item()
            
            self.text_cnt += len(targets)
            self.itr += 1

            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()
        end_time = time.time()
        self.log(is_last=True)
        completion_msg = f"Evaluation completed in {time_str(time.time() - start_time)}."
        print0(completion_msg)
        self.logger.write(completion_msg)
        
        # Save predictions to JSON file
        self.save_predictions()

    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text'] # all text queries for the single video
        if not isinstance(tokens, tuple):
            tokens = (tokens, )
        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)
        with torch.no_grad():
            if self.batchify_text_queries:
                text, text_masks, text_size = self._batchify_text2(
                    text_list=[tokens]
                )
                text = text.cuda(non_blocking=True) # (bs, num_queries, c_t, t)
                text_masks = text_masks.cuda(non_blocking=True) # (bs, num_queries, t)
                text_size = text_size.cuda(non_blocking=True)

                # batched_text_encoded: (num_queries, c_t, t), batched_text_mask_encoded: (num_queries, 1,t)
                text, text_masks = self.model.encode_text2(text, text_masks, text_size)
            else:
                text_list, text_mask_list = tuple(), tuple()
                for text in tokens:
                    text = text[None]
                    text_mask = text.new_full(
                        (1, 1, text.size(-1)), 1, dtype=torch.bool
                    )
                    text = text.cuda(non_blocking=True)
                    text_mask = text_mask.cuda(non_blocking=True)

                    text, text_mask = self.model.encode_text(text, text_mask)
                    text_list += (text, )
                    text_mask_list += (text_mask, )
        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        # Calculate adaptive input_vid_len based on actual window size
        # This ensures we use the minimum padding needed for the FPN constraints
        stride = self.min_chunk_size * self.vid_stride
        input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)
            
            with torch.no_grad():
                fpn, fpn_masks = self.model.encode_video(window, window_mask)
                fpn_logits_list, fpn_offsets_list = tuple(), tuple()
                if self.batchify_text_queries:
                    fpn_logits, fpn_offsets, _ = self.model.fuse_and_predict(fpn, fpn_masks, text, text_masks, text_size)
                    for query_idx in range(len(tokens)):
                        # Extract this query's results from each layer
                        query_logits = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_logits)
                        query_offsets = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_offsets)
                        fpn_logits_list += (query_logits,)
                        fpn_offsets_list += (query_offsets,)
                else:
                    for text, text_mask in zip(text_list, text_mask_list):
                        fpn_logits, fpn_offsets, _ = \
                            self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                        fpn_logits_list += (fpn_logits, )
                        fpn_offsets_list += (fpn_offsets, )

            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks, 
                    window_ext[idx] if window_ext is not None else None
                )
                window_segs += window_offset / self.vid_stride
                window_segs_list += (window_segs.cpu(), )
                window_scores_list += (window_scores.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]

        results = tuple()
        for segs, scores in zip(segs_list, scores_list):
            # only keep top-k scoring boxes
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            # NMS
            segs, scores = self.batched_nms(segs[idx], scores[idx])

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results += ({'segments': segs, 'scores': scores}, )

        return results

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens
        return text, text_masks
    
    def _batchify_text2(self, text_list):
        bs = len(text_list)

        # batch text
        if isinstance(text_list[0], tuple):
            # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, )
                b_text_masks += (b_tm, )
                n += (len(t), )
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)

        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        return text, text_masks, text_size
    
    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        # loop over all FPN levels
        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                # external scores has the same length as the video features
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            # clean up predictions before NMS for efficiency
            ## (1) filter points by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)

        ## (2) only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        ## (3) assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        ## (4) filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]

        return segs, scores

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
            avg_across_iou = metrics[i, :].mean()
            log_str += f"\nRank@{rank}, Avg IoU: {(avg_across_iou * 100):.2f}"
        miou = self.iou_sum / self.text_cnt if self.text_cnt > 0 else 0.0
        log_str += f"\n-----\nmIoU: {(miou * 100):.2f}"
        self.logger.write(log_str)
    
    def save_predictions(self):
        """Save predictions to JSON file"""
        predictions_file = os.path.join(self.opt['_root'], f"predictions_{self.opt['_ckpt']}.json")
        
        # Add overall metrics to the predictions
        overall_metrics = self.counts / self.text_cnt
        summary = {
            'overall_recall_at_iou': {},
            'total_queries': int(self.text_cnt),
            'total_videos': len(self.predictions)
        }
        
        for i, rank in enumerate(self.ranks):
            for j, thresh in enumerate(self.iou_threshs):
                key = f"Rank@{rank}_IoU@{thresh:.1f}"
                summary['overall_recall_at_iou'][key] = overall_metrics[i, j].item()
        summary['mIoU'] = self.iou_sum / self.text_cnt if self.text_cnt > 0 else 0.0
        
        output_data = {
            'summary': summary,
            'videos': self.predictions
        }
        
        with open(predictions_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print0(f"Predictions saved to {predictions_file}")
        self.logger.write(f"Predictions saved to {predictions_file}")

    
class Evaluator:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))

        # prepare dataset
        dataset = make_dataset(opt['eval']['data'], is_training=False)
        self.dataloader, _ = make_dataloader(
            dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
        )
        self.num_itrs = len(self.dataloader)
        self.itr = self.text_cnt = 0

        # load model
        self.model = PtTransformer(opt['model']).cuda()
        self.load_model()
        self.model.eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()

        # build logging utilities
        self.log_interval = self.num_itrs // 10
        self.logger = Logger(os.path.join(opt['_root'], f"eval_{opt['_ckpt']}.txt"))

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
        self.iou_sum = 0.0  # accumulate top-1 IoU for mIoU

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']

    def load_model(self):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{self.opt['_ckpt']}.pth"
        )
        ckpt = torch.load(filename, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['model_ema'])
        print0(f"Loaded checkpoint [epoch {self.opt['_ckpt']}]...")

    @torch.no_grad()
    def run(self):
        print0("Evaluation started.")
        start_time = time.time()
        for data_list in self.dataloader:
            results = self.predict(data_list[0])
            targets = data_list[0]['segment']
            assert len(results) == len(targets)

            for result, target in zip(results, targets):
                segs, scores = result['segments'], result['scores']
                idx = scores.argsort(descending=True)
                segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                target = torch.as_tensor(target, dtype=torch.float)
                target = target.expand(len(segs), -1)
                
                iou_topk = iou(segs, target)
                iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                self.counts += (iou_n[:, None] >= self.iou_threshs[None])
                self.iou_sum += iou_topk[0].item()  # mIoU: top-1 IoU
            self.text_cnt += len(targets)
            self.itr += 1

            if self.itr == 1 or self.itr % self.log_interval == 0:
                self.log()
        
        self.log(is_last=True)
        completion_msg = f"Evaluation completed in {time_str(time.time() - start_time)}."
        print0(completion_msg)
        self.logger.write(completion_msg)

    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )

        text_list, text_mask_list = tuple(), tuple()
        with autocast():  
            for text in tokens:
                text = text[None]
                text_mask = text.new_full(
                    (1, 1, text.size(-1)), 1, dtype=torch.bool
                )
                text = text.cuda(non_blocking=True)
                text_mask = text_mask.cuda(non_blocking=True)

                text, text_mask = self.model.encode_text(text, text_mask)
                text_list += (text, )
                text_mask_list += (text_mask, )

        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)

        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        # window_size = vid_len // 30
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            # pad video features to the next divisible size
            ## NOTE: this ensures the sequence can be perfectly chunked
            ## for efficient local attention
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        with autocast():  
            for window, window_offset, window_ext in \
                zip(windows, window_offsets, window_ext_scores):
                window = F.pad(window, (0, input_vid_len - window_size))[None]
                window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
                window = window.cuda(non_blocking=True)
                window_mask = window_mask.cuda(non_blocking=True)
                if window_ext is not None:
                    window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                    window_ext = window_ext.cuda(non_blocking=True)
                
                fpn, fpn_masks = self.model.encode_video(window, window_mask)
                fpn_n_points = [m.size(-1) for m in fpn_masks]
                fpn_points = self.pt_gen(fpn_n_points)

                fpn_logits_list, fpn_offsets_list = tuple(), tuple()
                for text, text_mask in zip(text_list, text_mask_list):
                    fpn_logits, fpn_offsets, _ = \
                        self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                    fpn_logits_list += (fpn_logits, )
                    fpn_offsets_list += (fpn_offsets, )
                fpn_masks = [m.squeeze(1) for m in fpn_masks]

                # collect segments and their scores
                window_segs_list, window_scores_list = tuple(), tuple()
                for idx, (fpn_logits, fpn_offsets) in \
                    enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                    window_segs, window_scores = self._collect_segments(
                        fpn_points, fpn_logits, fpn_offsets, fpn_masks, 
                        window_ext[idx] if window_ext is not None else None
                    )
                    window_segs += window_offset / self.vid_stride
                    window_segs_list += (window_segs.cpu(), )
                    window_scores_list += (window_scores.cpu(), )

                segs_list += (window_segs_list, )
                scores_list += (window_scores_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]

        results = tuple()
        for segs, scores in zip(segs_list, scores_list):
            # only keep top-k scoring boxes
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            # NMS
            segs, scores = self.batched_nms(segs[idx].float(), scores[idx].float())

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results += ({'segments': segs, 'scores': scores}, )

        return results

    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        # loop over all FPN levels
        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                # external scores has the same length as the video features
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            # clean up predictions before NMS for efficiency
            ## (1) filter points by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )

        points = torch.cat(points_list).float()
        scores = torch.cat(scores_list).float()
        offsets = torch.cat(offsets_list).float()

        ## (2) only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        ## (3) assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        ## (4) filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]

        return segs.float(), scores.float()

    def log(self, is_last=False):
        metrics = self.counts / self.text_cnt
        miou = self.iou_sum / self.text_cnt if self.text_cnt > 0 else 0.0
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}"
                )
            avg_across_iou = metrics[i, :].mean()
            log_str += f"\nRank@{rank}, Avg IoU: {(avg_across_iou * 100):.2f}"
        log_str += f"\n-----\nmIoU: {(miou * 100):.2f}"
        self.logger.write(log_str)


class EvaluatorAuxiliary(EvaluatorOriginal):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.early_fusion = opt['model'].get('early_fusion', True)
        self.use_mst = opt['model'].get('use_mst', False)
        if self.early_fusion:
            self.logger.write("Early fusion enabled")
        
    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )
        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)
        with torch.no_grad():
            if self.batchify_text_queries:
                text, text_mask, text_size = self._batchify_text2(
                    text_list=[tokens]
                )
                text = text.cuda(non_blocking=True) # (bs, num_queries, c_t, t)
                text_mask = text_mask.cuda(non_blocking=True) # (bs, num_queries, t)
                text_size = text_size.cuda(non_blocking=True)

                # batched_text_encoded: (num_queries, c_t, t), batched_text_mask_encoded: (num_queries, 1,t)
                text, text_mask = self.model.encode_text2(text, text_mask, text_size)
            else:
                text_list, text_mask_list = tuple(), tuple()
                for text in tokens:
                    text = text[None]
                    text_mask = text.new_full(
                        (1, 1, text.size(-1)), 1, dtype=torch.bool
                    )
                    text = text.cuda(non_blocking=True)
                    text_mask = text_mask.cuda(non_blocking=True)

                    text, text_mask = self.model.encode_text(text, text_mask)
                    text_list += (text, )
                    text_mask_list += (text_mask, )


        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        # Calculate adaptive input_vid_len based on actual window size
        # This ensures we use the minimum padding needed for the FPN constraints
        stride = self.min_chunk_size * self.vid_stride
        input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        for window, window_offset, window_ext in \
            zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)
            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)



            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            with torch.no_grad():
                if self.batchify_text_queries:
                    if self.early_fusion:
                        window, window_mask = self.model.vid_proj(window, window_mask)
                        window, window_mask = self.model.fusion(window, window_mask, text, text_mask, text_size)
                    fpn, fpn_masks, _, _ = self.model.encode_video(window, window_mask)
                    
                    if self.use_mst:
                        fpn_logits, fpn_logits2, fpn_offsets, _ = \
                            self.model.fuse_and_predict_mst(fpn, fpn_masks, text, text_mask, text_size)
                    else:
                        fpn_logits, _, fpn_offsets, _ = \
                            self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask, text_size)

                    for query_idx in range(len(tokens)):
                        # Extract this query's results from each layer
                        query_logits = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_logits)
                        # query_logits = tuple((fl1[query_idx:query_idx+1] + fl2[query_idx:query_idx+1]) / 2 for fl1, fl2 in zip(fpn_logits, fpn_logits2))
                        # query_logits = tuple(torch.maximum(fl1[query_idx:query_idx+1], fl2[query_idx:query_idx+1]) for fl1, fl2 in zip(fpn_logits, fpn_logits2))
                        query_offsets = tuple(layer_tensor[query_idx:query_idx+1] for layer_tensor in fpn_offsets)
                        fpn_logits_list += (query_logits,)
                        fpn_offsets_list += (query_offsets,)
                else:
                    window_orig = window.clone()
                    window_mask_orig = window_mask.clone()
                    for text, text_mask in zip(text_list, text_mask_list):
                        # window: (1, dim, T)
                        # window_mask: (1, 1, T)
                        if self.early_fusion:
                            window, window_mask = self.model.vid_proj(window_orig, window_mask_orig)
                            window, window_mask = self.model.fusion(window, window_mask, text, text_mask)
                        fpn, fpn_masks, _, _ = self.model.encode_video(window, window_mask)
                        if self.use_mst:
                            fpn_logits, fpn_logits2, fpn_offsets, _ = \
                                self.model.fuse_and_predict_mst(fpn, fpn_masks, text, text_mask)
                        else:
                            fpn_logits, _, fpn_offsets, _ = \
                                self.model.fuse_and_predict(fpn, fpn_masks, text, text_mask)
                        fpn_logits_list += (fpn_logits, )
                        fpn_offsets_list += (fpn_offsets, )
            # fpn, fpn_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in \
                enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks, 
                    window_ext[idx] if window_ext is not None else None
                )
                window_segs += window_offset / self.vid_stride
                window_segs_list += (window_segs.cpu(), )
                window_scores_list += (window_scores.cpu(), )

            segs_list += (window_segs_list, )
            scores_list += (window_scores_list, )

        segs_list = [torch.cat(x) for x in zip(*segs_list)]     # [bs x (n, 2)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)] # [bs x (n,)]

        results = tuple()
        for segs, scores in zip(segs_list, scores_list):
            # only keep top-k scoring boxes
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]

            # NMS
            segs, scores = self.batched_nms(segs[idx], scores[idx])

            # convert segments to timestamps in seconds
            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results += ({'segments': segs, 'scores': scores}, )

        return results

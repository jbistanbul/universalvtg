import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, List, Callable
from libs.dist_utils import print0
from .contrastive_losses import contrastive_subsample_negative_mp


def build_single_level_loss(loss_type: str) -> Callable:
    "Returns a function that computes the contrastive loss for a single pyramid level."
    if loss_type == 'contr_mp':
        return lambda *args, **kw: contrastive_subsample_negative_mp(*args, **kw)
    raise ValueError(f"Unknown loss_type: {loss_type}")


class MultiScaleMaskedGTPointContrastive(nn.Module):

    def __init__(self, opt: dict, vid_embd_dim: int):
        super().__init__()
        contrastive_type = opt.get('contr_type', 'point_gt_contr')
        span_contr_gt = opt.get('span_contr_gt', False)
        self.cross_layer = False
        self.within_layer = False
        self.temp = opt.get('temperature', 0.07)
        self.neg_ratio = opt.get('neg_ratio', 1.0)
             
        if contrastive_type == 'point_gt_contr':
            self.contrastive_loss = self._compute_infonce_loss
            self.within_layer = True
        elif contrastive_type == 'point_gt_contr_pooled':
            self.contrastive_loss = self._compute_infonce_loss_pooled
            self.within_layer = True
        elif contrastive_type == 'point_gt_contr_pooled_cross_layer':
            self.cross_layer = True
            self.crosslayer_contrastive_loss = self._compute_infonce_loss_pooled_cross_layer
        elif contrastive_type == "within_and_cross_layer_pooled":
            self.cross_layer = True
            self.crosslayer_contrastive_loss = self._compute_infonce_loss_pooled_cross_layer
            self.within_layer = True
            self.contrastive_loss = self._compute_infonce_loss_pooled
        else:
            raise ValueError(f'Unknown contrastive type: {contrastive_type}')
        
        # Projector configuration
        self.use_projector = opt.get('use_projector', True)
        if self.use_projector:
            proj_outdim = opt.get('proj_outdim', 256)
            proj_expand = opt.get('proj_expand', 1.0)
            proj_num_layers = opt.get('proj_num_layers', 2)
            self.projector = LNProjector(in_dim=vid_embd_dim, out_dim=proj_outdim, expand=proj_expand, num_layers=proj_num_layers)
        else:
            self.projector = None
            print0('No projector used')
            
        print0(f'Using GT Contrastive Loss of type: {contrastive_type}, temp: {self.temp}, neg_ratio: {self.neg_ratio}, span_contr_gt: {span_contr_gt}, weight: {opt.get("weight", 1.0)}')

    def forward(
        self,
        fpn_fused:      Tuple[Tensor],
        fpn_fused_masks: Tuple[Tensor],
        gt_labels:        Tuple[Tensor],
        gt_labels_span_list:   Tuple[Tensor],
    ) -> Tensor:
        cross_layer_loss = torch.tensor(0.0).cuda()
        within_layer_loss = torch.tensor(0.0).cuda()
        if self.cross_layer:
            cross_layer_loss = self.crosslayer_contrastive_loss(fpn_fused, fpn_fused_masks, gt_labels, gt_labels_span_list)
        if self.within_layer:
            for i, (fpn, fpn_mask, gt_label, gt_labels_span) in enumerate(zip(
                fpn_fused, fpn_fused_masks, gt_labels, gt_labels_span_list
            )):
                loss = self.contrastive_loss(fpn, fpn_mask, gt_label, gt_labels_span)
                within_layer_loss = within_layer_loss + loss
        return cross_layer_loss + within_layer_loss
    
    def _compute_infonce_loss_pooled_cross_layer(self,
                            fpn_list: Tuple[Tensor],
                            fpn_mask_list: Tuple[Tensor],  
                            gt_label_list: Tuple[Tensor],
                            gt_label_span_list: Tuple[Tensor],
                            ) -> Tensor:
        """
        InfoNCE on pooled-GT anchors collected across all FPN layers.
        """
        if len(fpn_list) == 0:
            return torch.tensor(0.0, device='cuda')
        
        device = fpn_list[0].device
        B = fpn_list[0].shape[0]
        
        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        
        for b in range(B):
            batch_anchors = []
            batch_negatives = []
            
            for layer_idx, (fpn, fpn_mask, gt_label) in enumerate(zip(fpn_list, fpn_mask_list, gt_label_list)):
                valid_idx = fpn_mask[b].nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue

                if self.use_projector and self.projector is not None:
                    proj = self.projector(fpn[b, :, valid_idx].T).T
                else:
                    proj = fpn[b, :, valid_idx]
                proj = F.normalize(proj, p=2, dim=0)

                labels = gt_label[b, valid_idx]
                pos_idx = labels.nonzero(as_tuple=True)[0]
                neg_idx = torch.logical_and((~labels), (~gt_label_span_list[layer_idx][b, valid_idx])).nonzero(as_tuple=True)[0]
                
                if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                    continue

                k_neg = int(pos_idx.numel() * self.neg_ratio)
                if neg_idx.numel() > k_neg:
                    sel = torch.randperm(neg_idx.numel(), device=device)[:k_neg]
                    neg_idx = neg_idx[sel]
                if neg_idx.numel() == 0:
                    continue

                anchor = proj[:, pos_idx].mean(dim=1, keepdim=True)
                anchor = F.normalize(anchor, p=2, dim=0)
                batch_anchors.append(anchor.squeeze(1))

                negatives = proj[:, neg_idx]
                batch_negatives.append(negatives)

            if len(batch_anchors) > 1 and len(batch_negatives) > 0:
                anchors_b = torch.stack(batch_anchors, dim=1)
                negatives_b = torch.cat(batch_negatives, dim=1)
                
                num_anchors_b = anchors_b.shape[1]
                
                sim_pos = (anchors_b.T @ anchors_b) / self.temp
                sim_neg = (anchors_b.T @ negatives_b) / self.temp
                
                diagonal_mask = torch.eye(num_anchors_b, device=device, dtype=torch.bool)
                sim_pos_masked = sim_pos.masked_fill(diagonal_mask, -float('inf'))

                if num_anchors_b > 1:
                    log_pos = torch.logsumexp(sim_pos_masked, dim=1)
                    log_all = torch.logsumexp(torch.cat([sim_pos_masked, sim_neg], dim=1), dim=1)
                    batch_loss = -(log_pos - log_all).mean()
                    total_loss += batch_loss
                    valid_batches += 1
            
        if valid_batches == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / valid_batches
        
    def _compute_infonce_loss_pooled(self,
                            fpn: Tensor,
                            fpn_mask: Tensor,
                            gt_label: Tensor,
                            gt_labels_span: Tensor) -> Tensor:
        """
        InfoNCE on pooled-GT anchor:
            • positives : anchor ↔ GT frames
            • negatives : anchor ↔ non-GT frames
        """
        B, D, T = fpn.shape
        device  = fpn.device

        total_loss = torch.zeros(1, device=device)
        total_anchors = 0

        for b in range(B):
            valid_idx = fpn_mask[b].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue

            if self.use_projector and self.projector is not None:
                proj = self.projector(fpn[b, :, valid_idx].T).T
            else:
                proj = fpn[b, :, valid_idx]
            proj = F.normalize(proj, p=2, dim=0)

            labels  = gt_label[b, valid_idx]
            pos_idx = labels.nonzero(as_tuple=True)[0]
            neg_idx = torch.logical_and((~labels), (~gt_labels_span[b, valid_idx])).nonzero(as_tuple=True)[0]
                
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            k_neg = int(pos_idx.numel() * self.neg_ratio)
            if neg_idx.numel() > k_neg:
                sel = torch.randperm(neg_idx.numel(), device=device)[:k_neg]
                neg_idx = neg_idx[sel]
            if neg_idx.numel() == 0:
                continue

            anchor = proj[:, pos_idx].mean(dim=1, keepdim=True)
            anchor = F.normalize(anchor, p=2, dim=0)

            sim_pos = (anchor.T @ proj[:, pos_idx]) / self.temp
            sim_neg = (anchor.T @ proj[:, neg_idx]) / self.temp

            log_pos = torch.logsumexp(sim_pos, dim=1)
            log_all = torch.logsumexp(torch.cat([sim_pos, sim_neg], dim=1), dim=1)
            loss = -(log_pos - log_all)

            total_loss += loss
            total_anchors += 1

        if total_anchors == 0:
            return torch.tensor(0.0, device=device)

        return (total_loss / total_anchors).squeeze()

    def _compute_infonce_loss(self, fpn: Tensor, fpn_mask: Tensor, gt_label: Tensor, gt_labels_span: Tensor) -> Tensor:
            """
            Multi-positive InfoNCE for one FPN level, with log-sum-exp for stability.
            """
            B, D, T = fpn.shape
            device  = fpn.device

            total_loss = torch.zeros(1, device=device)
            total_pos  = 0

            for b in range(B):
                valid_idx = fpn_mask[b].nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue

                if self.use_projector and self.projector is not None:
                    proj = self.projector(fpn[b, :, valid_idx].T)
                    proj = proj.T
                else:
                    proj = fpn[b, :, valid_idx]
                    proj = F.normalize(proj, p=2, dim=0)

                labels   = gt_label[b, valid_idx]
                pos_idx  = labels.nonzero(as_tuple=True)[0]
                neg_idx  = (~labels).nonzero(as_tuple=True)[0]
                if pos_idx.numel() < 2 or neg_idx.numel() == 0:
                    continue

                k_neg = int(pos_idx.numel() * self.neg_ratio)
                if neg_idx.numel() > k_neg:
                    perm    = torch.randperm(neg_idx.numel(), device=device)[:k_neg]
                    neg_idx = neg_idx[perm]

                if neg_idx.numel() == 0:
                    continue

                P, N   = pos_idx.numel(), neg_idx.numel()
                zp, zn = proj[:, pos_idx], proj[:, neg_idx]

                sim_pp = (zp.T @ zp) / self.temp
                sim_pn = (zp.T @ zn) / self.temp

                self_mask = torch.eye(P, device=device, dtype=torch.bool)

                sim_pp_masked = sim_pp.masked_fill(self_mask, -float("inf"))
                
                log_pos = torch.logsumexp(sim_pp_masked, dim=1)

                all_sim = torch.cat([sim_pp_masked, sim_pn], dim=1)
                log_all = torch.logsumexp(all_sim, dim=1)

                loss_vec = -(log_pos - log_all)

                total_loss = total_loss + loss_vec.sum()
                total_pos += P

            if total_pos == 0:
                return torch.tensor(0.0, device=device)

            return (total_loss / total_pos).squeeze()


class MultiScaleMaskedContrastive(nn.Module):

    def __init__(self, opt: dict, vid_embd_dim: int):
        super().__init__()
        contrastive_type = opt.get('contr_type', 'contr_mp')
        self.loss_fn   = build_single_level_loss(contrastive_type)
        self.temp = opt.get('temperature', 0.07)
        self.neg_ratio = opt.get('neg_ratio', 0.20)
        self.gap_ratio = opt.get('gap_ratio', 0.30)
        self.radius = opt.get('radius', 0)
        self.hard_neg = opt.get('hard_neg', False)
        self.cross_video_neg = opt.get('cross_video_neg', False)
        proj_outdim = opt.get('proj_outdim', 256)
        proj_expand = opt.get('proj_expand', 1.0)
        proj_num_layers = opt.get('proj_num_layers', 2)
        self.projector = LNProjector(in_dim=vid_embd_dim, out_dim=proj_outdim, expand=proj_expand, num_layers=proj_num_layers)
        print0(f'Using Contrastive Loss of type: {contrastive_type}, temp: {self.temp}, neg_ratio: {self.neg_ratio}, gap_ratio: {self.gap_ratio}, radius: {self.radius}, hard_neg: {self.hard_neg}, cross_video_neg: {self.cross_video_neg}, weight: {opt.get("weight", 1.0)}')

    def forward(
        self,
        sequence_fpn:      Tuple[Tensor],
        sequence_fpn_mask: Tuple[Tensor],
        anchor_fpn:        Tuple[Tensor],
        anchor_fpn_mask:   Tuple[Tensor],
    ) -> Tensor:
        total = torch.tensor(0.0).cuda()
        num_layers = len(sequence_fpn)
        weights = [1 for i in range(num_layers)]
        for i, (seq, seq_m, anc, anc_m) in enumerate(zip(
            sequence_fpn, sequence_fpn_mask,
            anchor_fpn,   anchor_fpn_mask
        )):
            loss = self.loss_fn(
                anchors=anc,
                seq_tokens=seq,
                anchor_mask=anc_m,
                seq_mask=seq_m,
                projector=self.projector,
                temperature=self.temp,
                neg_ratio=self.neg_ratio,
                gap_ratio=self.gap_ratio,
                radius=self.radius,
                hard_neg=self.hard_neg,
                cross_video_neg=self.cross_video_neg
            )
            total = total + weights[i] * loss
        return total


class LNProjector(nn.Module):
    """
    Configurable MLP projector with 1, 2, or 3 layers.
    """
    def __init__(self, in_dim: int, out_dim: int = 256, expand: float = 2.0, num_layers: int = 2):
        super().__init__()
        
        if num_layers not in [1, 2, 3]:
            raise ValueError(f"num_layers must be 1, 2, or 3, got {num_layers}")
        
        self.num_layers = num_layers
        hid_dim = int(expand * in_dim)

        if num_layers == 1:
            self.fc = nn.Linear(in_dim, out_dim)
            
        elif num_layers == 2:
            self.fc1 = nn.Linear(in_dim, hid_dim, bias=False)
            self.ln1 = nn.LayerNorm(hid_dim)
            self.act1 = nn.GELU()
            self.fc2 = nn.Linear(hid_dim, out_dim)
            
            nn.init.ones_(self.ln1.weight)
            nn.init.zeros_(self.ln1.bias)
            
        elif num_layers == 3:
            self.fc1 = nn.Linear(in_dim, hid_dim, bias=False)
            self.ln1 = nn.LayerNorm(hid_dim)
            self.act1 = nn.GELU()
            
            self.fc2 = nn.Linear(hid_dim, hid_dim, bias=False)
            self.ln2 = nn.LayerNorm(hid_dim)
            self.act2 = nn.GELU()
            
            self.fc3 = nn.Linear(hid_dim, out_dim)
            
            nn.init.ones_(self.ln1.weight)
            nn.init.zeros_(self.ln1.bias)
            nn.init.ones_(self.ln2.weight)
            nn.init.zeros_(self.ln2.bias)

    def forward(self, x):
        if self.num_layers == 1:
            z = self.fc(x)
        elif self.num_layers == 2:
            z = self.fc2(self.act1(self.ln1(self.fc1(x))))
        elif self.num_layers == 3:
            h1 = self.act1(self.ln1(self.fc1(x)))
            h2 = self.act2(self.ln2(self.fc2(h1)))
            z = self.fc3(h2)
        
        return F.normalize(z, p=2, dim=1)
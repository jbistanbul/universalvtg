import torch
import torch.nn.functional as F


def contrastive_subsample_negative_mp(
    anchors:     torch.Tensor,                # (B, D, L2)
    seq_tokens:  torch.Tensor,                # (B, D, L)  ,  L = 2*L2
    anchor_mask: torch.Tensor,                # (B, 1, L2)
    seq_mask:    torch.Tensor,                # (B, 1, L)
    projector:   torch.nn.Module,
    radius:      int   = 0,
    temperature: float = 0.07,
    neg_ratio:   float = 0.20,                # M = neg_ratio·(Nv‑1)
    gap_ratio:   float = 0.30,
    hard_neg:    bool  = False,                # True → top‑M ; False → random‑M
    cross_video_neg: bool = False,            # include other‑video anchors?
) -> torch.Tensor:
    """
    Multi‑positive InfoNCE with either
      • hard‑negative Top‑M   (hard_neg=True),  or
      • random‑negative M     (hard_neg=False).

    Negatives can be restricted to the same video by setting
    `cross_video_neg=False`.
    """
    assert radius >= 0
    B, D, L2 = anchors.shape
    L  = seq_tokens.shape[2]                        # = 2*L2
    dev = anchors.device

    # -------------------------------------------------- 1. Flatten
    N      = B * L2
    a_flat = anchors.permute(0,2,1).reshape(N, D)
    s_flat = seq_tokens.permute(0,2,1).reshape(B*L, D)
    a_mask = anchor_mask.squeeze(1).reshape(N)
    s_mask = seq_mask.squeeze(1).reshape(B*L)

    vid_id = torch.arange(N, device=dev) // L2
    t_id   = torch.arange(N, device=dev) %  L2

    # -------------------------------------------------- 2. Project sequence features once
    z_s_flat = F.normalize(projector(s_flat), dim=-1)  # (B*L, D)

    # -------------------------------------------------- 3. Gather multi‑positives
    pos_feat, pos_mask = [], []
    for k in range(-radius, radius+1):
        new_t   = t_id + k
        in_rng  = (0 <= new_t) & (new_t < L2)

        even = (vid_id * L + 2*new_t).clamp(0, B*L-1)
        odd  = (even + 1).clamp(0, B*L - 1)   

        m_even = in_rng & s_mask[even]
        m_odd  = in_rng & s_mask[odd]

        z_even = z_s_flat[even]
        z_odd  = z_s_flat[odd]

        pos_feat.extend([z_even, z_odd])
        pos_mask.extend([m_even, m_odd])

    z_pos  = torch.stack(pos_feat, dim=1)      # (N, K, D)
    p_mask = torch.stack(pos_mask, dim=1)      # (N, K)

    # -------------------------------------------------- 4. Filter valid anchors
    valid = a_mask & (p_mask.sum(1) > 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=dev)

    idxs   = valid.nonzero(as_tuple=False).view(-1)
    z_a    = F.normalize(projector(a_flat[idxs]), dim=-1)
    z_pos  = z_pos[idxs]
    p_mask = p_mask[idxs]

    Nv   = z_a.size(0)
    vid  = vid_id[idxs]
    t    = t_id[idxs]

    # -------------------------------------------------- 5. Numerator
    sim_pos = (z_a.unsqueeze(1) * z_pos).sum(-1) / temperature
    exp_pos = torch.exp(sim_pos) * p_mask.float()
    exp_pos = exp_pos.sum(1)                    # (Nv,)

    # -------------------------------------------------- 6. Candidate negative mask
    sim_all = torch.matmul(z_a, z_a.T) / temperature

    eye   = torch.eye(Nv, device=dev, dtype=torch.bool)
    vid_eq= vid.unsqueeze(0).eq(vid.unsqueeze(1))

    gap   = max(1, int(gap_ratio * L2))
    near  = vid_eq & (torch.abs(t.unsqueeze(0)-t.unsqueeze(1)) <= gap)

    if cross_video_neg:
        cand = ~(eye | near)                   # far same‑video  + other‑video
    else:
        cand = vid_eq & ~(eye | near)          # far same‑video only

    exp_sim = torch.exp(sim_all) * cand.float()      # (Nv, Nv)

    # -------------------------------------------------- 6. Top‑M  vs. Random‑M  (per‑anchor, proportional to #positives)

    valid_cnt = cand.sum(1)                 # (Nv,)  – how many candidate negs each anchor could use
    pos_cnt   = p_mask.sum(1)               # (Nv,)  – how many *positives* each anchor really has

    # new rule: Mi = neg_ratio · (#positives of that anchor)
    Mi = (neg_ratio * pos_cnt.float()).ceil().long().clamp(min=1)   # (Nv,)
    Mi = torch.minimum(Mi, valid_cnt)       # cannot request more negatives than exist
    max_M = int(Mi.max())                   # still one scalar => can reuse torch.topk

    if hard_neg:         # ----- hardest‑Mi negatives per row --------------------
        masked_sim = exp_sim.masked_fill(~cand, -1)                 # make invalid < any exp()
        _, top_idx = torch.topk(masked_sim, k=max_M, dim=1, largest=True)

    else:                # ----- random‑Mi negatives per row ---------------------
        rnd       = torch.rand_like(exp_sim) * cand.float() - (~cand).float()
        _, top_idx = torch.topk(rnd, k=max_M, dim=1, largest=True)

    # build a binary selection mask that keeps exactly Mi entries from each row
    row_ids   = torch.arange(Nv, device=dev).unsqueeze(1).expand_as(top_idx)   # (Nv,max_M)
    keep_mask = (torch.arange(max_M, device=dev).unsqueeze(0) < Mi.unsqueeze(1))
    sel_msk   = torch.zeros_like(exp_sim, dtype=torch.bool)
    sel_msk[row_ids, top_idx] = keep_mask

    # row‑wise denominator contribution
    row_sum = (exp_sim * sel_msk.float()).sum(1)                   # (Nv,)
    # -------------------------------------------------- 7. InfoNCE loss
    eps   = 1e-9
    loss = -torch.log(exp_pos / (exp_pos + row_sum + eps))
    return loss.mean()
# file: reference.py

import torch


ATOL = 2.0e-2
RTOL = 2.0e-2


def paged_decode_attention_ref(q, k_cache, v_cache, page_table, cache_lens, scale, window, page_size=None):
    if q.dim() != 3:
        raise ValueError("q must have shape [B, Hq, D]")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise ValueError("k_cache and v_cache must have shape [num_pages, Hkv, page_size, D]")

    B, Hq, D = q.shape
    num_pages, Hkv, P, Dk = k_cache.shape
    num_pages_v, Hkv_v, Pv, Dv = v_cache.shape

    if page_size is None:
        page_size = P

    if P != page_size or Pv != page_size:
        raise ValueError("page_size mismatch")
    if num_pages != num_pages_v or Hkv != Hkv_v or D != Dk or D != Dv:
        raise ValueError("cache dimensions mismatch")
    if Hq % Hkv != 0:
        raise ValueError("Hq must be divisible by Hkv")
    if page_table.shape[0] != B:
        raise ValueError("page_table batch dimension mismatch")

    qf = q.float()
    kf = k_cache.float()
    vf = v_cache.float()
    pt = page_table.to(torch.long)
    lens = cache_lens.to(torch.long)

    out = torch.zeros((B, Hq, D), device=q.device, dtype=torch.float32)
    group = Hq // Hkv

    for b in range(B):
        L = int(lens[b].item())
        if L <= 0:
            continue

        max_tokens = pt.shape[1] * int(page_size)
        if L > max_tokens:
            L = max_tokens

        start = max(0, L - int(window)) if int(window) > 0 else 0
        if start >= L:
            continue

        tok = torch.arange(start, L, device=q.device, dtype=torch.long)
        page_ids = tok // int(page_size)
        page_offs = tok - page_ids * int(page_size)
        phys = pt[b, page_ids]

        valid = phys >= 0
        if not bool(valid.any().item()):
            continue

        phys = phys[valid]
        page_offs = page_offs[valid]

        # Gather as [tokens, Hkv, D].
        k_tokens = kf[phys, :, page_offs, :]
        v_tokens = vf[phys, :, page_offs, :]

        for hq in range(Hq):
            hkv = hq // group

            k_block = k_tokens[:, hkv, :]
            v_block = v_tokens[:, hkv, :]

            scores = (k_block * qf[b, hq][None, :]).sum(dim=-1) * float(scale)
            probs = torch.softmax(scores, dim=0)
            out[b, hq] = (probs[:, None] * v_block).sum(dim=0)

    return out

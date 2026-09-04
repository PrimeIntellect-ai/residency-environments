# file: reference.py

import torch


def flash_decode_ref(q, k, v, cache_lens, scale, window):
    if q.dim() != 3:
        raise ValueError("q must have shape [B, Hq, D]")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("k and v must have shape [B, Hkv, S, D]")

    B, Hq, D = q.shape
    Bk, Hkv, S, Dk = k.shape
    Bv, Hkv_v, Sv, Dv = v.shape

    if B != Bk or B != Bv:
        raise ValueError("q, k, v batch dimensions must match")
    if Hkv != Hkv_v:
        raise ValueError("k and v Hkv dimensions must match")
    if S != Sv:
        raise ValueError("k and v sequence dimensions must match")
    if D != Dk or D != Dv:
        raise ValueError("q, k, v head dimensions must match")
    if Hq % Hkv != 0:
        raise ValueError("Hq must be divisible by Hkv")

    out = torch.zeros((B, Hq, D), device=q.device, dtype=torch.float32)

    qf = q.float()
    kf = k.float()
    vf = v.float()
    lens = cache_lens.to(torch.int64)

    group = Hq // Hkv

    for b in range(B):
        L = int(lens[b].item())
        if L <= 0:
            continue
        if L > S:
            L = S

        if window is not None and int(window) > 0:
            start = max(0, L - int(window))
        else:
            start = 0

        if start >= L:
            continue

        for hq in range(Hq):
            hkv = hq // group

            q_vec = qf[b, hq]
            k_block = kf[b, hkv, start:L]
            v_block = vf[b, hkv, start:L]

            scores = (k_block * q_vec[None, :]).sum(dim=-1) * float(scale)
            probs = torch.softmax(scores, dim=0)
            out[b, hq] = (probs[:, None] * v_block).sum(dim=0)

    return out

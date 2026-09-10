# file: solution.py

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    lens_ptr,
    out_ptr,
    part_ptr,
    split_len,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    WINDOW: tl.constexpr,
    NSPLIT: tl.constexpr,
    FINAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    b = pid // HQ
    hq = pid - b * HQ

    group = HQ // HKV
    hkv = hq // group

    offs_d = tl.arange(0, D)

    q_vec = tl.load(q_ptr + (b * HQ + hq) * D + offs_d).to(tl.float32)

    kv_len = tl.load(lens_ptr + b)
    kv_len = tl.minimum(tl.maximum(kv_len, 0), S)

    if WINDOW > 0:
        lo = tl.maximum(0, kv_len - WINDOW)
    else:
        lo = 0

    lo = tl.maximum(lo, sid * split_len)
    hi = tl.minimum(kv_len, (sid + 1) * split_len)
    if NSPLIT == 1:
        hi = kv_len

    neg = -1.0e30
    m_i = neg
    l_i = 0.0
    acc = tl.zeros((D,), tl.float32)

    kv_base = (b * HKV + hkv) * S * D

    for pos in range(lo, hi, BLOCK_N):
        offs_n = pos + tl.arange(0, BLOCK_N)
        valid_n = offs_n < hi

        k_block = tl.load(
            k_ptr + kv_base + offs_n[:, None] * D + offs_d[None, :],
            mask=valid_n[:, None],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(k_block * q_vec[None, :], axis=1) * SCALE
        scores = tl.where(valid_n, scores, neg)

        block_m = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, block_m)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(valid_n, p, 0.0)

        v_block = tl.load(
            v_ptr + kv_base + offs_n[:, None] * D + offs_d[None, :],
            mask=valid_n[:, None],
            other=0.0,
        ).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v_block, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    if FINAL:
        denom = tl.where(l_i > 0.0, l_i, 1.0)
        tl.store(out_ptr + (b * HQ + hq) * D + offs_d, acc / denom)
    else:
        base = (pid * NSPLIT + sid) * (D + 2)
        tl.store(part_ptr + base + offs_d, acc)
        tl.store(part_ptr + base + D, m_i)
        tl.store(part_ptr + base + D + 1, l_i)


@triton.jit
def _flash_decode_combine_kernel(
    part_ptr,
    out_ptr,
    D: tl.constexpr,
    NSPLIT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_s = tl.arange(0, NSPLIT)
    offs_d = tl.arange(0, D)

    base = pid * NSPLIT * (D + 2)

    m_s = tl.load(part_ptr + base + offs_s * (D + 2) + D)
    l_s = tl.load(part_ptr + base + offs_s * (D + 2) + D + 1)

    m_star = tl.max(m_s, axis=0)
    w = tl.exp(m_s - m_star)

    l_tot = tl.sum(w * l_s, axis=0)
    denom = tl.where(l_tot > 0.0, l_tot, 1.0)

    acc = tl.load(
        part_ptr + base + offs_s[:, None] * (D + 2) + offs_d[None, :]
    )
    out = tl.sum(w[:, None] * acc, axis=0) / denom

    tl.store(out_ptr + pid * D + offs_d, out)


def flash_decode(q, k, v, cache_lens, scale, window):
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not cache_lens.is_cuda:
        raise ValueError("q, k, v, and cache_lens must be CUDA tensors")

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
        raise ValueError("Hq must be divisible by Hkv for GQA")
    if D not in (64, 128):
        raise ValueError("D must be 64 or 128")
    if cache_lens.numel() != B:
        raise ValueError("cache_lens must have shape [B]")

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    lens_c = cache_lens.contiguous()
    if lens_c.dtype != torch.int32:
        lens_c = lens_c.to(torch.int32)

    out = torch.empty((B, Hq, D), device=q.device, dtype=torch.float32)

    w = int(window)
    nprog = B * Hq

    nsplit = 1
    if w <= 0 and S >= 1024 and nprog < 128:
        cand = min(8, S // 512, max(1, 1024 // nprog))
        nsplit = 1 << (cand.bit_length() - 1)

    if nsplit <= 1:
        _flash_decode_kernel[(nprog, 1)](
            q_c,
            k_c,
            v_c,
            lens_c,
            out,
            out,
            S,
            Hq,
            Hkv,
            S,
            D,
            float(scale),
            w,
            NSPLIT=1,
            FINAL=True,
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
        )
    else:
        split_len = (S + nsplit - 1) // nsplit
        part = torch.empty(nprog * nsplit * (D + 2), device=q.device, dtype=torch.float32)
        _flash_decode_kernel[(nprog, nsplit)](
            q_c,
            k_c,
            v_c,
            lens_c,
            out,
            part,
            split_len,
            Hq,
            Hkv,
            S,
            D,
            float(scale),
            w,
            NSPLIT=nsplit,
            FINAL=False,
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
        )
        _flash_decode_combine_kernel[(nprog,)](
            part,
            out,
            D,
            NSPLIT=nsplit,
            num_warps=1,
        )

    return out

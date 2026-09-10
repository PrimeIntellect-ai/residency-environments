# file: solution.py

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_hidden_kernel(
    x_ptr,
    sort_idx_ptr,
    row_prefix_ptr,
    w1_ptr,
    hidden_ptr,
    D: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    KROUTE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    blk = tl.program_id(0)
    pid_n = tl.program_id(1)

    # walk experts, deriving the padded-tile prefix from row_prefix on the fly
    total = tl.full((), 0, tl.int32)
    e = tl.full((), 0, tl.int32)
    seg_start = tl.full((), 0, tl.int32)
    seg_end = tl.full((), 0, tl.int32)
    blk0 = tl.full((), 0, tl.int32)
    for j in tl.static_range(E):
        s = tl.load(row_prefix_ptr + j).to(tl.int32)
        t = tl.load(row_prefix_ptr + j + 1).to(tl.int32)
        bj = (t - s + (TILE_M - 1)) // TILE_M
        hit = (blk >= total) & (blk < total + bj)
        e = tl.where(hit, j, e)
        seg_start = tl.where(hit, s, seg_start)
        seg_end = tl.where(hit, t, seg_end)
        blk0 = tl.where(hit, total, blk0)
        total += bj

    if blk < total:
        row0 = seg_start + (blk - blk0) * TILE_M
        offs_m = row0 + tl.arange(0, TILE_M)
        m_valid = offs_m < seg_end
        offs_m_safe = tl.where(m_valid, offs_m, seg_start)

        srt = tl.load(sort_idx_ptr + offs_m_safe).to(tl.int32)
        n = srt // KROUTE

        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
        n_valid = offs_n < H

        acc = tl.zeros((TILE_M, TILE_N), tl.float32)
        offs_k = tl.arange(0, BLOCK_K)

        for k0 in range(0, D, BLOCK_K):
            k_idxs = k0 + offs_k
            a_tile = tl.load(
                x_ptr + n[:, None] * D + k_idxs[None, :],
                mask=m_valid[:, None] & (k_idxs[None, :] < D),
                other=0.0,
            )
            b_tile = tl.load(
                w1_ptr + (e.to(tl.int64) * D + k_idxs[:, None]) * H + offs_n[None, :],
                mask=(k_idxs[:, None] < D) & n_valid[None, :],
                other=0.0,
            )
            acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

        act = acc * tl.sigmoid(acc)

        tl.store(
            hidden_ptr + srt[:, None].to(tl.int64) * H + offs_n[None, :],
            act.to(tl.float16),
            mask=m_valid[:, None] & n_valid[None, :],
        )


@triton.jit
def _moe_output_kernel(
    hidden_ptr,
    sort_idx_ptr,
    row_prefix_ptr,
    gates_ptr,
    w2_ptr,
    out_ptr,
    D: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    blk = tl.program_id(0)
    pid_n = tl.program_id(1)

    total = tl.full((), 0, tl.int32)
    e = tl.full((), 0, tl.int32)
    seg_start = tl.full((), 0, tl.int32)
    seg_end = tl.full((), 0, tl.int32)
    blk0 = tl.full((), 0, tl.int32)
    for j in tl.static_range(E):
        s = tl.load(row_prefix_ptr + j).to(tl.int32)
        t = tl.load(row_prefix_ptr + j + 1).to(tl.int32)
        bj = (t - s + (TILE_M - 1)) // TILE_M
        hit = (blk >= total) & (blk < total + bj)
        e = tl.where(hit, j, e)
        seg_start = tl.where(hit, s, seg_start)
        seg_end = tl.where(hit, t, seg_end)
        blk0 = tl.where(hit, total, blk0)
        total += bj

    if blk < total:
        row0 = seg_start + (blk - blk0) * TILE_M
        offs_m = row0 + tl.arange(0, TILE_M)
        m_valid = offs_m < seg_end
        offs_m_safe = tl.where(m_valid, offs_m, seg_start)

        srt = tl.load(sort_idx_ptr + offs_m_safe).to(tl.int64)

        gate = tl.load(gates_ptr + srt, mask=m_valid, other=0.0).to(tl.float32)

        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
        n_valid = offs_n < D

        acc = tl.zeros((TILE_M, TILE_N), tl.float32)
        offs_k = tl.arange(0, BLOCK_K)

        for k0 in range(0, H, BLOCK_K):
            k_idxs = k0 + offs_k
            a_tile = tl.load(
                hidden_ptr + srt[:, None] * H + k_idxs[None, :],
                mask=m_valid[:, None] & (k_idxs[None, :] < H),
                other=0.0,
            )
            b_tile = tl.load(
                w2_ptr + (e.to(tl.int64) * H + k_idxs[:, None]) * D + offs_n[None, :],
                mask=(k_idxs[:, None] < H) & n_valid[None, :],
                other=0.0,
            )
            acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

        tl.store(
            out_ptr + srt[:, None] * D + offs_n[None, :],
            acc * gate[:, None],
            mask=m_valid[:, None] & n_valid[None, :],
        )


_BOUNDS_CACHE = {}


def _bounds(E, device):
    key = (E, device)
    b = _BOUNDS_CACHE.get(key)
    if b is None:
        b = torch.arange(E + 1, device=device, dtype=torch.int32)
        _BOUNDS_CACHE[key] = b
    return b


def fused_moe_dispatch(x, expert_ids, gates, w1, w2):
    if not x.is_cuda or not expert_ids.is_cuda or not gates.is_cuda:
        raise ValueError("x, expert_ids, and gates must be CUDA tensors")
    if not w1.is_cuda or not w2.is_cuda:
        raise ValueError("w1 and w2 must be CUDA tensors")

    if x.dim() != 2:
        raise ValueError("x must have shape [N, D]")
    if expert_ids.dim() != 2 or gates.dim() != 2:
        raise ValueError("expert_ids and gates must have shape [N, K]")
    if w1.dim() != 3 or w2.dim() != 3:
        raise ValueError("w1 must be [E,D,H], w2 must be [E,H,D]")

    N, D = x.shape
    N_ids, Kroute = expert_ids.shape
    N_gates, K_gates = gates.shape
    E, D_w1, H = w1.shape
    E2, H2, D2 = w2.shape

    if N_ids != N or N_gates != N or K_gates != Kroute:
        raise ValueError("routing tensors must match x batch and route count")
    if E2 != E or D_w1 != D or H2 != H or D2 != D:
        raise ValueError("w1/w2 dimensions must match x")
    if D not in (256, 512, 1024, 2048):
        raise ValueError("D must be one of 256, 512, 1024, 2048")
    if H not in (256, 512, 1024, 2048):
        raise ValueError("H must be one of 256, 512, 1024, 2048")
    if E not in (8, 32):
        raise ValueError("E must be 8 or 32")
    if Kroute not in (1, 2):
        raise ValueError("K must be 1 or 2")

    device = x.device

    x_c = x.contiguous()
    expert_ids_c = expert_ids.to(torch.int32).contiguous()
    gates_c = gates.to(torch.float32).contiguous()
    w1_c = w1.contiguous()
    w2_c = w2.contiguous()

    NK = N * Kroute
    TILE_M = 16

    # Route bookkeeping fully on device (no host sync). Invalid low ids sort to the
    # front (before expert 0), invalid high ids clamp to bucket E at the back; both
    # fall outside [row_prefix[e], row_prefix[e+1]) for e < E and are never computed.
    bucket = expert_ids_c.view(-1).clamp(-1, E)
    sorted_b, sort_idx = torch.sort(bucket)
    row_prefix = torch.searchsorted(sorted_b, _bounds(E, device))
    sort_idx = sort_idx.to(torch.int32)

    max_blocks = (NK + TILE_M - 1) // TILE_M + E

    hidden = torch.empty((NK, H), device=device, dtype=torch.float16)

    if Kroute == 1:
        out_routes = torch.zeros((N, D), device=device, dtype=torch.float32)
    else:
        out_routes = torch.zeros((NK, D), device=device, dtype=torch.float32)

    _moe_hidden_kernel[(max_blocks, triton.cdiv(H, 128))](
        x_c,
        sort_idx,
        row_prefix,
        w1_c,
        hidden,
        D,
        H,
        E,
        Kroute,
        TILE_M=TILE_M,
        TILE_N=128,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )

    _moe_output_kernel[(max_blocks, triton.cdiv(D, 128))](
        hidden,
        sort_idx,
        row_prefix,
        gates_c,
        w2_c,
        out_routes,
        D,
        H,
        E,
        TILE_M=TILE_M,
        TILE_N=128,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )

    if Kroute == 1:
        return out_routes

    return out_routes.view(N, Kroute, D).sum(dim=1)

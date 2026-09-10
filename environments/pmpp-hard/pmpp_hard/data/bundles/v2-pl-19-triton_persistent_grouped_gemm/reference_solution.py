# PMPP_CANARY_19_71767d3aed -- held-out canary; MUST NOT appear in any submission
# file: solution.py

import torch
import triton
import triton.language as tl


@triton.jit
def _persistent_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    group_m_ptr,
    group_n_ptr,
    group_k_ptr,
    a_offsets_ptr,
    b_offsets_ptr,
    bias_offsets_ptr,
    c_offsets_ptr,
    tile_prefix_ptr,
    group_count,
    total_tiles,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACT: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    num_pids = tl.num_programs(0).to(tl.int64)

    offs_m_base = tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n_base = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k_base = tl.arange(0, BLOCK_K).to(tl.int64)

    tile_id = pid

    while tile_id < total_tiles:
        # binary search: greatest g with tile_prefix[g] <= tile_id
        lo = tl.full((), 0, tl.int64)
        hi = tl.full((), 0, tl.int64) + group_count
        while hi - lo > 1:
            mid = (lo + hi) // 2
            p = tl.load(tile_prefix_ptr + mid)
            take = tile_id >= p
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid)

        group = lo
        local_tile = tile_id - tl.load(tile_prefix_ptr + group)

        m = tl.load(group_m_ptr + group).to(tl.int64)
        n = tl.load(group_n_ptr + group).to(tl.int64)
        k = tl.load(group_k_ptr + group).to(tl.int64)

        tiles_n = tl.cdiv(n, tl.full((), BLOCK_N, tl.int64))
        tile_m = local_tile // tiles_n
        tile_n = local_tile - tile_m * tiles_n

        a_base = tl.load(a_offsets_ptr + group).to(tl.int64)
        b_base = tl.load(b_offsets_ptr + group).to(tl.int64)
        bias_base = tl.load(bias_offsets_ptr + group).to(tl.int64)
        c_base = tl.load(c_offsets_ptr + group).to(tl.int64)

        offs_m = tile_m * BLOCK_M + offs_m_base
        offs_n = tile_n * BLOCK_N + offs_n_base

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        k0 = tl.full((), 0, tl.int64)
        while k0 < k:
            offs_k = k0 + offs_k_base

            a_tile = tl.load(
                a_ptr + a_base + offs_m[:, None] * k + offs_k[None, :],
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0,
            )
            b_tile = tl.load(
                b_ptr + b_base + offs_k[:, None] * n + offs_n[None, :],
                mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                other=0.0,
            )

            acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)
            k0 += BLOCK_K

        bias = tl.load(
            bias_ptr + bias_base + offs_n,
            mask=offs_n < n,
            other=0.0,
        ).to(tl.float32)

        x = acc + bias[None, :]

        if ACT == 0:
            x = tl.maximum(x, 0.0)
        else:
            # GELU tanh approximation using sigmoid identity:
            # gelu(x)=0.5*x*(1+tanh(z)), tanh(z)=2*sigmoid(2z)-1
            # => gelu(x)=x*sigmoid(2z)
            x3 = x * x * x
            z = 0.7978845608028654 * (x + 0.044715 * x3)
            x = x * tl.sigmoid(2.0 * z)

        tl.store(
            c_ptr + c_base + offs_m[:, None] * n + offs_n[None, :],
            x,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )

        tile_id += num_pids


def _activation_to_id(activation):
    if isinstance(activation, str):
        act = activation.lower()
        if act == "relu":
            return 0
        if act in ("gelu", "gelu_tanh"):
            return 1
        raise ValueError(f"unknown activation: {activation}")
    return int(activation)


_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_K = 32


def grouped_gemm_bias_act(
    a,
    b,
    bias,
    group_m,
    group_n,
    group_k,
    a_offsets,
    b_offsets,
    bias_offsets,
    c_offsets,
    activation="relu",
):
    if not a.is_cuda or not b.is_cuda or not bias.is_cuda:
        raise ValueError("a, b, and bias must be CUDA tensors")
    if not group_m.is_cuda or not group_n.is_cuda or not group_k.is_cuda:
        raise ValueError("group metadata tensors must be CUDA tensors")
    if not a_offsets.is_cuda or not b_offsets.is_cuda or not bias_offsets.is_cuda or not c_offsets.is_cuda:
        raise ValueError("offset tensors must be CUDA tensors")

    group_count = int(group_m.numel())
    if group_count <= 0:
        raise ValueError("group_count must be positive")

    device = a.device

    # One host round-trip for all sizing metadata (m, n per group + last c offset).
    meta_host = torch.cat(
        (
            group_m.to(torch.int64),
            group_n.to(torch.int64),
            c_offsets[group_count - 1 : group_count],
        )
    ).cpu()
    meta = meta_host.tolist()
    m_host = meta[:group_count]
    n_host = meta[group_count : 2 * group_count]
    c_last = meta[2 * group_count]

    prefix = [0] * (group_count + 1)
    run = 0
    for g in range(group_count):
        run += ((m_host[g] + _BLOCK_M - 1) // _BLOCK_M) * (
            (n_host[g] + _BLOCK_N - 1) // _BLOCK_N
        )
        prefix[g + 1] = run
    total_tiles = run

    total_c = c_last + m_host[-1] * n_host[-1]

    c = torch.empty((total_c,), device=device, dtype=torch.float32)

    if total_tiles == 0 or total_c == 0:
        return c

    tile_prefix = torch.tensor(prefix, device=device, dtype=torch.int64)

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    num_programs = min(max(1, total_tiles), max(1, sm_count * 2))

    _persistent_grouped_gemm_kernel[(num_programs,)](
        a,
        b,
        bias,
        c,
        group_m,
        group_n,
        group_k,
        a_offsets,
        b_offsets,
        bias_offsets,
        c_offsets,
        tile_prefix,
        group_count,
        total_tiles,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        ACT=_activation_to_id(activation),
        num_warps=8,
        num_stages=3,
    )

    return c

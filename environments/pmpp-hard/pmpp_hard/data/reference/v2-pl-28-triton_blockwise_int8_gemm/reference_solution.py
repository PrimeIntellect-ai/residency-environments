# PMPP_CANARY_28_405856f37f -- held-out canary; MUST NOT appear in any submission
# file: solution.py

import torch
import triton
import triton.language as tl


@triton.jit
def _blockwise_int8_gemm_kernel(
    a_ptr,
    b_ptr,
    a_scales_ptr,
    b_scales_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
    SCALE_BLOCK_M: tl.constexpr,
    SCALE_BLOCK_N: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, SCALE_BLOCK_K)

    row_scale_block = (pid_m * TILE_M) // SCALE_BLOCK_M
    col_scale_blocks = offs_n // SCALE_BLOCK_N

    acc = tl.zeros((TILE_M, TILE_N), tl.float32)

    for k0 in range(0, K, SCALE_BLOCK_K):
        k_idx = k0 + offs_k
        k_scale_block = k0 // SCALE_BLOCK_K

        a_tile = tl.load(
            a_ptr + offs_m[:, None] * K + k_idx[None, :],
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < K),
            other=0,
        ).to(tl.int8)

        b_tile = tl.load(
            b_ptr + k_idx[:, None] * N + offs_n[None, :],
            mask=(k_idx[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int8)

        dot_i32 = tl.dot(a_tile, b_tile, out_dtype=tl.int32)

        a_scale = tl.load(
            a_scales_ptr + row_scale_block * NUM_K_BLOCKS + k_scale_block
        ).to(tl.float32)

        b_scale = tl.load(
            b_scales_ptr + k_scale_block * NUM_N_BLOCKS + col_scale_blocks,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float32)

        acc += dot_i32.to(tl.float32) * a_scale * b_scale[None, :]

    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def blockwise_int8_gemm(a, b, a_scales, b_scales, block_m=64, block_n=64, block_k=64):
    if not a.is_cuda or not b.is_cuda or not a_scales.is_cuda or not b_scales.is_cuda:
        raise ValueError("all tensors must be CUDA tensors")

    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise ValueError("a and b must be torch.int8")

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("a and b must be rank-2")

    M, K = a.shape
    Kb, N = b.shape

    if Kb != K:
        raise ValueError("a.shape[1] must equal b.shape[0]")

    block_m = int(block_m)
    block_n = int(block_n)
    block_k = int(block_k)

    if block_m not in (32, 64, 128):
        raise ValueError("block_m must be one of 32, 64, 128")
    if block_n not in (32, 64, 128):
        raise ValueError("block_n must be one of 32, 64, 128")
    if block_k not in (32, 64, 128):
        raise ValueError("block_k must be one of 32, 64, 128")

    num_m_blocks = triton.cdiv(M, block_m)
    num_n_blocks = triton.cdiv(N, block_n)
    num_k_blocks = triton.cdiv(K, block_k)

    if tuple(a_scales.shape) != (num_m_blocks, num_k_blocks):
        raise ValueError("a_scales must have shape [ceil(M/block_m), ceil(K/block_k)]")
    if tuple(b_scales.shape) != (num_k_blocks, num_n_blocks):
        raise ValueError("b_scales must have shape [ceil(K/block_k), ceil(N/block_n)]")

    a_c = a.contiguous()
    b_c = b.contiguous()
    a_scales_c = a_scales.to(torch.float32).contiguous()
    b_scales_c = b_scales.to(torch.float32).contiguous()

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    tile_m = 32
    tile_n = 64

    grid = (triton.cdiv(M, tile_m), triton.cdiv(N, tile_n))

    _blockwise_int8_gemm_kernel[grid](
        a_c,
        b_c,
        a_scales_c,
        b_scales_c,
        c,
        M,
        N,
        K,
        num_k_blocks,
        num_n_blocks,
        block_m,
        block_n,
        block_k,
        TILE_M=tile_m,
        TILE_N=tile_n,
        num_warps=4,
        num_stages=4,
    )

    return c

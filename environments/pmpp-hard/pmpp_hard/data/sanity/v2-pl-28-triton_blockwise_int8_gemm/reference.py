# file: reference.py

import torch


ATOL = 1.5e-1
RTOL = 6.0e-2


def blockwise_int8_gemm_ref(a, b, a_scales, b_scales, block_m=64, block_n=64, block_k=64):
    torch.backends.cuda.matmul.allow_tf32 = False

    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise ValueError("a and b must be torch.int8")

    M, K = a.shape
    Kb, N = b.shape

    if Kb != K:
        raise ValueError("a.shape[1] must equal b.shape[0]")

    block_m = int(block_m)
    block_n = int(block_n)
    block_k = int(block_k)

    num_m_blocks = (M + block_m - 1) // block_m
    num_n_blocks = (N + block_n - 1) // block_n
    num_k_blocks = (K + block_k - 1) // block_k

    if tuple(a_scales.shape) != (num_m_blocks, num_k_blocks):
        raise ValueError("a_scales shape mismatch")
    if tuple(b_scales.shape) != (num_k_blocks, num_n_blocks):
        raise ValueError("b_scales shape mismatch")

    rows = torch.arange(M, device=a.device, dtype=torch.long) // block_m
    ks = torch.arange(K, device=a.device, dtype=torch.long) // block_k
    cols = torch.arange(N, device=a.device, dtype=torch.long) // block_n

    a_scale_full = a_scales.float()[rows[:, None], ks[None, :]]
    b_scale_full = b_scales.float()[ks[:, None], cols[None, :]]

    a_deq = a.float() * a_scale_full
    b_deq = b.float() * b_scale_full

    return a_deq @ b_deq

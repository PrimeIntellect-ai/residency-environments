# file: solution.py

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_rows_kernel(
    x_ptr,
    weight_ptr,
    xn_ptr,
    M,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_K)

    ss = tl.zeros((BLOCK_K,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        xv = tl.load(
            x_ptr + row * K + k_idxs,
            mask=k_idxs < K,
            other=0.0,
        ).to(tl.float32)
        ss += xv * xv

    inv_rms = tl.rsqrt(tl.sum(ss, axis=0) * (1.0 / K) + EPS)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        xv = tl.load(
            x_ptr + row * K + k_idxs,
            mask=k_idxs < K,
            other=0.0,
        ).to(tl.float32)
        wv = tl.load(
            weight_ptr + k_idxs,
            mask=k_idxs < K,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            xn_ptr + row * K + k_idxs,
            (xv * wv * inv_rms).to(tl.float16),
            mask=k_idxs < K,
        )


@triton.jit
def _matmul_bias_act_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    y_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        a_tile = tl.load(
            a_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        b_tile = tl.load(
            b_ptr + k_idxs[:, None] * N + offs_n[None, :],
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    if ACT == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:
        # GELU tanh approximation without tl.tanh:
        # gelu(x)=0.5*x*(1+tanh(z)), z=sqrt(2/pi)*(x+0.044715*x^3)
        # tanh(z)=2*sigmoid(2z)-1 => gelu(x)=x*sigmoid(2z)
        x3 = acc * acc * acc
        z = 0.7978845608028654 * (acc + 0.044715 * x3)
        acc = acc * tl.sigmoid(2.0 * z)

    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _activation_to_id(activation):
    if activation is None:
        return 0

    if isinstance(activation, str):
        act = activation.lower()
        if act in ("none", "identity", ""):
            return 0
        if act == "relu":
            return 1
        if act in ("gelu", "gelu_tanh"):
            return 2
        raise ValueError(f"unknown activation: {activation}")

    return int(activation)


def fused_rmsnorm_matmul(x, weight, w, bias=None, eps=1.0e-5, activation="none"):
    if not x.is_cuda or not weight.is_cuda or not w.is_cuda:
        raise ValueError("x, weight, and w must be CUDA tensors")
    if bias is not None and not bias.is_cuda:
        raise ValueError("bias must be a CUDA tensor")

    if x.dim() != 2:
        raise ValueError("x must have shape [M, K]")
    if w.dim() != 2:
        raise ValueError("w must have shape [K, N]")

    M, K = x.shape
    Kw, N = w.shape

    if Kw != K:
        raise ValueError("w.shape[0] must equal x.shape[1]")
    if weight.numel() != K:
        raise ValueError("weight must have shape [K]")
    if bias is not None and bias.numel() != N:
        raise ValueError("bias must have shape [N]")

    x_c = x.contiguous()
    weight_c = weight.contiguous()
    w_c = w.contiguous()

    if bias is None:
        bias_c = weight_c
        has_bias = False
    else:
        bias_c = bias.contiguous()
        has_bias = True

    xn = torch.empty((M, K), device=x.device, dtype=torch.float16)
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    _rmsnorm_rows_kernel[(M,)](
        x_c,
        weight_c,
        xn,
        M,
        K,
        float(eps),
        BLOCK_K=1024,
        num_warps=8,
    )

    if M > 512:
        block_m, block_n, block_k, warps, stages = 128, 128, 64, 8, 3
    else:
        block_m, block_n, block_k, warps, stages = 64, 128, 64, 8, 4

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)

    _matmul_bias_act_kernel[grid](
        xn,
        w_c,
        bias_c,
        y,
        M,
        K,
        N,
        HAS_BIAS=has_bias,
        ACT=_activation_to_id(activation),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=warps,
        num_stages=stages,
    )

    return y

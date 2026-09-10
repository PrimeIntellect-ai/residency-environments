# file: solution.py

import torch
import triton
import triton.language as tl


@triton.jit
def _round_nearest_even_i32(x):
    ax = tl.abs(x)
    fl = tl.floor(ax)
    frac = ax - fl
    fi = fl.to(tl.int32)

    odd = (fi & 1) != 0
    up = (frac > 0.5) | ((frac == 0.5) & odd)

    ri = fi + up.to(tl.int32)
    return tl.where(x < 0.0, -ri, ri)


@triton.jit
def _fused_layernorm_quant_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    q_ptr,
    scale_ptr,
    dequant_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    x = tl.load(
        x_ptr + row * D + offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    x_masked = tl.where(mask, x, 0.0)
    mean = tl.sum(x_masked, axis=0) * (1.0 / D)

    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) * (1.0 / D)
    inv = tl.rsqrt(var + EPS)

    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    y = centered * inv * weight + bias
    y = tl.where(mask, y, 0.0)

    amax = tl.max(tl.abs(y), axis=0)
    row_scale = tl.where(amax > 0.0, amax / 127.0, 1.0)

    q_float = y / row_scale
    q_i32 = _round_nearest_even_i32(q_float)
    q_i32 = tl.minimum(tl.maximum(q_i32, -127), 127)

    dq = q_i32.to(tl.float32) * row_scale

    tl.store(scale_ptr + row, row_scale)
    tl.store(q_ptr + row * D + offs, q_i32.to(tl.int8), mask=mask)
    tl.store(dequant_ptr + row * D + offs, dq, mask=mask)


def fused_layernorm_quant(x, weight, bias, eps=1.0e-5):
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise ValueError("x, weight, and bias must be CUDA tensors")

    if x.dim() != 2:
        raise ValueError("x must have shape [N, D]")

    N, D = x.shape

    if weight.numel() != D or bias.numel() != D:
        raise ValueError("weight and bias must have shape [D]")
    if D not in (256, 1024, 4096):
        raise ValueError("D must be one of 256, 1024, 4096")

    x_c = x.contiguous()
    weight_c = weight.to(torch.float32).contiguous()
    bias_c = bias.to(torch.float32).contiguous()

    q = torch.empty((N, D), device=x.device, dtype=torch.int8)
    scale = torch.empty((N,), device=x.device, dtype=torch.float32)
    dequant = torch.empty((N, D), device=x.device, dtype=torch.float32)

    block_d = triton.next_power_of_2(D)
    num_warps = 8 if D >= 1024 else 4

    _fused_layernorm_quant_kernel[(N,)](
        x_c,
        weight_c,
        bias_c,
        q,
        scale,
        dequant,
        N,
        D,
        float(eps),
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=4,
    )

    return q, scale, dequant

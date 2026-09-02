# file: solution.py

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_state_kernel(
    x_ptr,
    decay_ptr,
    gate_ptr,
    lengths_ptr,
    chunk_final_ptr,
    chunk_prod_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = tl.program_id(1)

    chunk = pid % NUM_CHUNKS
    tmp = pid // NUM_CHUNKS
    h = tmp % H
    b = tmp // H

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    seq_len = tl.load(lengths_ptr + b)
    t0 = chunk * CHUNK_SIZE

    state = tl.zeros((BLOCK_D,), tl.float32)
    p = tl.full((), 1.0, tl.float32)

    for i in range(0, CHUNK_SIZE):
        t = t0 + i
        valid_t = (t < T) & (t < seq_len)

        x_vals = tl.load(
            x_ptr + (((b * T + t) * H + h) * D + offs_d),
            mask=valid_t & d_mask,
            other=0.0,
        ).to(tl.float32)
        decay = tl.load(
            decay_ptr + ((b * T + t) * H + h),
            mask=valid_t,
            other=1.0,
        ).to(tl.float32)
        gate = tl.load(
            gate_ptr + ((b * T + t) * H + h),
            mask=valid_t,
            other=0.0,
        ).to(tl.float32)

        state = tl.where(valid_t, decay * state + gate * x_vals, state)
        p = tl.where(valid_t, p * decay, p)

    tl.store(
        chunk_final_ptr + ((b * H + h) * NUM_CHUNKS + chunk) * D + offs_d,
        state,
        mask=d_mask,
    )
    tl.store(
        chunk_prod_ptr + (b * H + h) * NUM_CHUNKS + chunk,
        p,
        mask=pid_d == 0,
    )


@triton.jit
def _carry_scan_kernel(
    chunk_final_ptr,
    chunk_prod_ptr,
    carry_in_ptr,
    final_state_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    carry = tl.zeros((BLOCK_D,), tl.float32)

    for c in range(0, NUM_CHUNKS):
        tl.store(
            carry_in_ptr + (pid_bh * NUM_CHUNKS + c) * D + offs_d,
            carry,
            mask=d_mask,
        )
        local_final = tl.load(
            chunk_final_ptr + (pid_bh * NUM_CHUNKS + c) * D + offs_d,
            mask=d_mask,
            other=0.0,
        )
        p = tl.load(chunk_prod_ptr + pid_bh * NUM_CHUNKS + c)
        carry = local_final + p * carry

    tl.store(
        final_state_ptr + pid_bh * D + offs_d,
        carry,
        mask=d_mask,
    )


@triton.jit
def _chunk_output_kernel(
    x_ptr,
    decay_ptr,
    gate_ptr,
    out_scale_ptr,
    skip_ptr,
    lengths_ptr,
    carry_in_ptr,
    y_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = tl.program_id(1)

    chunk = pid % NUM_CHUNKS
    tmp = pid // NUM_CHUNKS
    h = tmp % H
    b = tmp // H

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    seq_len = tl.load(lengths_ptr + b)
    t0 = chunk * CHUNK_SIZE

    out_scale = tl.load(out_scale_ptr + h * D + offs_d, mask=d_mask, other=0.0)
    skip = tl.load(skip_ptr + h * D + offs_d, mask=d_mask, other=0.0)

    state = tl.load(
        carry_in_ptr + ((b * H + h) * NUM_CHUNKS + chunk) * D + offs_d,
        mask=d_mask,
        other=0.0,
    )

    for i in range(0, CHUNK_SIZE):
        t = t0 + i
        valid_t = (t < T) & (t < seq_len)

        x_vals = tl.load(
            x_ptr + (((b * T + t) * H + h) * D + offs_d),
            mask=valid_t & d_mask,
            other=0.0,
        ).to(tl.float32)
        decay = tl.load(
            decay_ptr + ((b * T + t) * H + h),
            mask=valid_t,
            other=1.0,
        ).to(tl.float32)
        gate = tl.load(
            gate_ptr + ((b * T + t) * H + h),
            mask=valid_t,
            other=0.0,
        ).to(tl.float32)

        state = tl.where(valid_t, decay * state + gate * x_vals, state)

        y_vals = tl.where(valid_t, state * out_scale + skip * x_vals, 0.0)

        tl.store(
            y_ptr + (((b * T + t) * H + h) * D + offs_d),
            y_vals,
            mask=(t < T) & d_mask,
        )


def chunked_state_scan(x, decay, gate, out_scale, skip, lengths=None, chunk_size=64):
    if not x.is_cuda or not decay.is_cuda or not gate.is_cuda:
        raise ValueError("x, decay, and gate must be CUDA tensors")
    if not out_scale.is_cuda or not skip.is_cuda:
        raise ValueError("out_scale and skip must be CUDA tensors")
    if lengths is not None and not lengths.is_cuda:
        raise ValueError("lengths must be a CUDA tensor")

    if x.dim() != 4:
        raise ValueError("x must have shape [B, T, H, D]")
    if decay.dim() != 3 or gate.dim() != 3:
        raise ValueError("decay and gate must have shape [B, T, H]")

    B, T, H, D = x.shape

    if decay.shape != (B, T, H):
        raise ValueError("decay shape must match [B, T, H]")
    if gate.shape != (B, T, H):
        raise ValueError("gate shape must match [B, T, H]")
    if out_scale.shape != (H, D):
        raise ValueError("out_scale must have shape [H, D]")
    if skip.shape != (H, D):
        raise ValueError("skip must have shape [H, D]")
    if D not in (32, 64, 128):
        raise ValueError("D must be 32, 64, or 128")
    if chunk_size not in (16, 32, 64, 128):
        raise ValueError("chunk_size must be 16, 32, 64, or 128")

    x_c = x.contiguous()
    decay_c = decay.contiguous()
    gate_c = gate.contiguous()
    out_scale_c = out_scale.contiguous()
    skip_c = skip.contiguous()

    if lengths is None:
        lengths_c = torch.full((B,), T, device=x.device, dtype=torch.int32)
    else:
        lengths_c = lengths.to(torch.int32).contiguous()

    num_chunks = triton.cdiv(T, chunk_size)
    block_d = D

    y = torch.empty((B, T, H, D), device=x.device, dtype=torch.float32)
    chunk_final = torch.empty((B, H, num_chunks, D), device=x.device, dtype=torch.float32)
    chunk_prod = torch.empty((B, H, num_chunks), device=x.device, dtype=torch.float32)
    carry_in = torch.empty((B, H, num_chunks, D), device=x.device, dtype=torch.float32)
    final_state = torch.empty((B, H, D), device=x.device, dtype=torch.float32)

    grid_chunks = (B * H * num_chunks, triton.cdiv(D, block_d))

    _chunk_state_kernel[grid_chunks](
        x_c,
        decay_c,
        gate_c,
        lengths_c,
        chunk_final,
        chunk_prod,
        T,
        H,
        D,
        CHUNK_SIZE=int(chunk_size),
        NUM_CHUNKS=int(num_chunks),
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )

    _carry_scan_kernel[(B * H, triton.cdiv(D, block_d))](
        chunk_final,
        chunk_prod,
        carry_in,
        final_state,
        H,
        D,
        NUM_CHUNKS=int(num_chunks),
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    _chunk_output_kernel[grid_chunks](
        x_c,
        decay_c,
        gate_c,
        out_scale_c,
        skip_c,
        lengths_c,
        carry_in,
        y,
        T,
        H,
        D,
        CHUNK_SIZE=int(chunk_size),
        NUM_CHUNKS=int(num_chunks),
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )

    return y, final_state

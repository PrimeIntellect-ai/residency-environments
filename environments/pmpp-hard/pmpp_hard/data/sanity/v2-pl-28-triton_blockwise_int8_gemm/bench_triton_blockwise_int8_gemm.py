# file: bench_triton_blockwise_int8_gemm.py

import sys

import torch

import solution


SEED = 2026072802


# Anti-hack digest (REWARD_HACK_AUDIT_20260704): the paired perf scorer requires the
# student bench and the reference bench to print the SAME out_fnv before accepting the
# student's timing. The graded output is tolerance-graded fp32 (not bitwise stable across
# implementations), so the digest folds exact-graded facts only: output shape/dtype and a
# per-case ok-flag from an untimed torch oracle check at the grading tolerance.
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_MASK = (1 << 64) - 1

ATOL = 1.5e-1
RTOL = 6.0e-2


def _fnv_update(h, data):
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & FNV_MASK
    return h


def _oracle(a, b, a_scales, b_scales, block_m, block_n, block_k):
    torch.backends.cuda.matmul.allow_tf32 = False
    M, K = a.shape
    N = b.shape[1]
    rows = torch.arange(M, device=a.device, dtype=torch.long) // block_m
    ks = torch.arange(K, device=a.device, dtype=torch.long) // block_k
    cols = torch.arange(N, device=a.device, dtype=torch.long) // block_n
    a_deq = a.float() * a_scales.float()[rows[:, None], ks[None, :]]
    b_deq = b.float() * b_scales.float()[ks[:, None], cols[None, :]]
    return a_deq @ b_deq




def _rand_int8(shape, device, generator, dist):
    if dist == "sparse":
        x16 = torch.randint(
            -127,
            128,
            shape,
            device=device,
            dtype=torch.int16,
            generator=generator,
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.70
        x16 = torch.where(keep, x16, torch.zeros_like(x16))
    elif dist == "ties":
        vals = torch.tensor(
            [-127, -64, -16, -1, 0, 1, 16, 64, 127],
            device=device,
            dtype=torch.int16,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x16 = vals[ids]
    else:
        x16 = torch.randint(
            -127,
            128,
            shape,
            device=device,
            dtype=torch.int16,
            generator=generator,
        )

    return x16.to(torch.int8).contiguous()


def _make_scales(shape, device, generator, dist):
    if dist == "outliers":
        s = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            0.0010, 0.0200, generator=generator
        )
        flat = s.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            flat[(idx % 19) == 0] = 0.1500
            flat[(idx % 23) == 0] = 0.0005
    elif dist == "ties":
        vals = torch.tensor(
            [0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        s = vals[ids]
    else:
        s = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            0.0025, 0.0500, generator=generator
        )

    return s.contiguous()


def _make_case(name, M, N, K, block_m, block_n, block_k, code_dist, scale_dist, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed((SEED + seed_offset) + 880123)

    a = _rand_int8((M, K), device, g, code_dist)
    b = _rand_int8((K, N), device, g, code_dist)

    num_m_blocks = (M + block_m - 1) // block_m
    num_n_blocks = (N + block_n - 1) // block_n
    num_k_blocks = (K + block_k - 1) // block_k

    a_scales = _make_scales((num_m_blocks, num_k_blocks), device, g, scale_dist)
    b_scales = _make_scales((num_k_blocks, num_n_blocks), device, g, scale_dist)

    return name, a, b, a_scales, b_scales, block_m, block_n, block_k, code_dist, scale_dist


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 50
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    cases = [
        _make_case(
            "square_1024_b64",
            1024, 1024, 1024,
            64, 64, 64,
            "uniform",
            "uniform",
            1,
        ),
        _make_case(
            "largeK_512x1024x4096",
            512, 1024, 4096,
            64, 64, 128,
            "sparse",
            "outliers",
            2,
        ),
        _make_case(
            "largeM_4096x1024x1024",
            4096, 1024, 1024,
            128, 64, 64,
            "ties",
            "ties",
            3,
        ),
        _make_case(
            "wideN_1024x4096x2048",
            1024, 4096, 2048,
            64, 128, 64,
            "uniform",
            "outliers",
            4,
        ),
    ]

    for _ in range(8):
        for _, a, b, a_scales, b_scales, bm, bn, bk, _, _ in cases:
            _ = solution.blockwise_int8_gemm(
                a,
                b,
                a_scales,
                b_scales,
                block_m=bm,
                block_n=bn,
                block_k=bk,
            )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        for _, a, b, a_scales, b_scales, bm, bn, bk, _, _ in cases:
            _ = solution.blockwise_int8_gemm(
                a,
                b,
                a_scales,
                b_scales,
                block_m=bm,
                block_n=bn,
                block_k=bk,
            )
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    import struct

    h = FNV_BASIS
    for name, a, b, a_scales, b_scales, bm, bn, bk, _, _ in cases:
        got = solution.blockwise_int8_gemm(a, b, a_scales, b_scales, block_m=bm, block_n=bn, block_k=bk)
        torch.cuda.synchronize()
        exp = _oracle(a, b, a_scales, b_scales, bm, bn, bk)
        ok = (
            got.shape == exp.shape
            and got.dtype == torch.float32
            and torch.allclose(got, exp, atol=ATOL, rtol=RTOL)
        )
        h = _fnv_update(h, struct.pack("<qqB", got.shape[0], got.shape[1], 1 if ok else 0))
        h = _fnv_update(h, str(got.dtype).encode())
        h = _fnv_update(h, name.encode())

    for name, a, b, _, _, bm, bn, bk, code_dist, scale_dist in cases:
        M, K = a.shape
        N = b.shape[1]
        print(
            f"bench_case {name:24s} M={M} N={N} K={K} "
            f"blocks=({bm},{bn},{bk}) codes={code_dist} scales={scale_dist}"
        )

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

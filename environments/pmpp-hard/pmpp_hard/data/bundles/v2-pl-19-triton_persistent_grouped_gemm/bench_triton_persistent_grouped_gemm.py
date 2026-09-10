# file: bench_triton_persistent_grouped_gemm.py

import os
import sys

import torch

import solution


# Input data values derive from PMPP_BENCH_SEED so correct outputs are not
# precomputable offline; shapes stay fixed so timing remains comparable.
SEED = (2026071902 ^ (int(os.environ.get("PMPP_BENCH_SEED", "0")) * 0x9E3779B97F4A7C15)) & (
    (1 << 63) - 1
)


# Anti-hack digest (REWARD_HACK_AUDIT_20260704): the paired perf scorer requires the
# student bench and the reference bench to print the SAME out_fnv before accepting the
# student's timing. The graded output is tolerance-graded fp32 (not bitwise stable across
# implementations), so the digest folds exact-graded facts only: output shape/dtype/numel
# and per-call ok-flags from untimed torch oracle checks at the grading tolerance.
#
# C3 anti-cache design (patterns I1 + P1):
#   - Every timed call k receives a FRESH pregenerated input variant derived from
#     (PMPP_BENCH_SEED, case, k); warmup uses two dedicated warmup-only variants, so no
#     timed variant is ever visible before the timer starts.
#   - The timed loop keeps a clone of every call's output; after the timer each timed
#     call is checked against ITS OWN variant's oracle at the grading tolerance and the
#     per-call ok-flag is folded into out_fnv. A cached/no-op/stale timed call fails its
#     variant's check and breaks the digest no matter how normal its timing looks.
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_MASK = (1 << 64) - 1

ATOL = 8.0e-2
RTOL = 8.0e-2


def _fnv_update(h, data):
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & FNV_MASK
    return h


def _oracle(a, b, bias, dims, ao_h, bo_h, bio_h, co_h, total_c, activation):
    # Host-side dims/offsets: no per-group .item() syncs (this oracle runs once
    # per timed variant, untimed, after the timer).
    torch.backends.cuda.matmul.allow_tf32 = False
    out = torch.empty((total_c,), device=a.device, dtype=torch.float32)
    af, bf = a.float(), b.float()
    act = (
        activation.lower()
        if isinstance(activation, str)
        else ("relu" if int(activation) == 0 else "gelu")
    )
    for g, (M, N, K) in enumerate(dims):
        if M == 0 or N == 0:
            continue
        a0, b0, bias0, c0 = ao_h[g], bo_h[g], bio_h[g], co_h[g]
        A = af[a0 : a0 + M * K].reshape(M, K)
        B = bf[b0 : b0 + K * N].reshape(K, N)
        C = A @ B + bias[bias0 : bias0 + N].float()[None, :]
        if act == "relu":
            C = torch.relu(C)
        else:
            C = torch.nn.functional.gelu(C, approximate="tanh")
        out[c0 : c0 + M * N] = C.reshape(-1)
    return out




def _float_to_half_tensor(x):
    return x.to(torch.float16).contiguous()


def _rand_values(shape, device, generator, dist):
    if dist == "uniform":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.25, 0.25, generator=generator
        )
    elif dist == "powerlaw":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.20, 0.20, generator=generator
        )
        mask = torch.rand(shape, device=device, generator=generator) > 0.55
        x = torch.where(mask, x, torch.zeros_like(x))
    elif dist == "tiny":
        vals = torch.tensor(
            [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x = vals[ids]
    else:
        vals = torch.tensor(
            [-0.1875, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x = vals[ids]
    return x


def _make_case(name, dims, activation, dist, seed_offset):
    device = "cuda"

    group_m_host = []
    group_n_host = []
    group_k_host = []

    a_offsets_host = []
    b_offsets_host = []
    bias_offsets_host = []
    c_offsets_host = []

    total_a = 0
    total_b = 0
    total_bias = 0
    total_c = 0

    for M, N, K in dims:
        group_m_host.append(M)
        group_n_host.append(N)
        group_k_host.append(K)

        a_offsets_host.append(total_a)
        b_offsets_host.append(total_b)
        bias_offsets_host.append(total_bias)
        c_offsets_host.append(total_c)

        total_a += M * K
        total_b += K * N
        total_bias += N
        total_c += M * N

    return {
        "name": name,
        "dims": dims,
        "activation": activation,
        "dist": dist,
        "seed_offset": seed_offset,
        "totals": (total_a, total_b, total_bias, total_c),
        "ao_h": a_offsets_host,
        "bo_h": b_offsets_host,
        "bio_h": bias_offsets_host,
        "co_h": c_offsets_host,
        "gm": torch.tensor(group_m_host, device=device, dtype=torch.int32),
        "gn": torch.tensor(group_n_host, device=device, dtype=torch.int32),
        "gk": torch.tensor(group_k_host, device=device, dtype=torch.int32),
        "ao": torch.tensor(a_offsets_host, device=device, dtype=torch.int64),
        "bo": torch.tensor(b_offsets_host, device=device, dtype=torch.int64),
        "bio": torch.tensor(bias_offsets_host, device=device, dtype=torch.int64),
        "co": torch.tensor(c_offsets_host, device=device, dtype=torch.int64),
    }


def _make_variant(case, k):
    # Input variant for timed iteration k (k >= 0) or a warmup slot (k < 0):
    # a pure function of (PMPP_BENCH_SEED, case, k).
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(
        (SEED + case["seed_offset"] * 1000003 + (k + 16) * 7919) & ((1 << 63) - 1)
    )
    total_a, total_b, total_bias, _ = case["totals"]
    a = _float_to_half_tensor(_rand_values((total_a,), device, g, case["dist"]))
    b = _float_to_half_tensor(_rand_values((total_b,), device, g, case["dist"]))
    bias = torch.empty((total_bias,), device=device, dtype=torch.float32).uniform_(
        -0.20, 0.20, generator=g
    )
    return (a, b, bias)


def _powerlaw_dims():
    dims = []
    for g in range(32):
        if g == 0:
            M = 1024
        elif g == 1:
            M = 512
        elif g < 4:
            M = 256
        elif g < 12:
            M = 64
        else:
            M = 8

        N = 512 if g % 4 == 0 else 256
        K = 512 if g % 3 == 0 else 256
        dims.append((M, N, K))
    return dims


def _many_tiny_dims():
    dims = []
    for g in range(256):
        if g == 0:
            M = 256
        elif g < 16:
            M = 32
        else:
            M = 1 + (g % 4)
        dims.append((M, 128, 128))
    return dims


def _balanced_dims():
    return [(512, 512, 512), (512, 512, 512), (512, 512, 512), (512, 512, 512)]


def _large_four_dims():
    return [(512, 1024, 1024), (256, 768, 2048), (128, 512, 512), (64, 2048, 256)]


def _run_case(case, variant):
    a, b, bias = variant
    return solution.grouped_gemm_bias_act(
        a,
        b,
        bias,
        case["gm"],
        case["gn"],
        case["gk"],
        case["ao"],
        case["bo"],
        case["bio"],
        case["co"],
        activation=case["activation"],
    )


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 30
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    cases = [
        _make_case("powerlaw_G32_gelu", _powerlaw_dims(), "gelu", "powerlaw", 1),
        _make_case("many_tiny_G256_relu", _many_tiny_dims(), "relu", "tiny", 2),
        _make_case("balanced_G4_relu", _balanced_dims(), "relu", "uniform", 3),
        _make_case("large_G4_gelu", _large_four_dims(), "gelu", "ties", 4),
    ]

    # C3 variant ring: one dedicated input variant per timed iteration (capped
    # for oversized local runs) plus two warmup-only variants.
    n_warm = 2
    v_timed = min(iters, 40)
    variants = [
        [_make_variant(case, k) for k in range(v_timed)] for case in cases
    ]
    warm_variants = [
        [_make_variant(case, -(w + 1)) for w in range(n_warm)] for case in cases
    ]
    torch.cuda.synchronize()

    for w in range(8):
        for ci, case in enumerate(cases):
            _ = _run_case(case, warm_variants[ci][w % n_warm])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    # kept[ci][k] snapshots timed call k's output. Preallocated slab + copy_
    # keeps the timed loop free of allocations (allocator growth forces
    # implicit syncs); the copy also de-aliases solutions that reuse their
    # output buffer across calls.
    kept = [
        torch.empty(
            (iters, case["totals"][3]), device="cuda", dtype=torch.float32
        )
        for case in cases
    ]
    shape_ok = [[False] * iters for _ in cases]

    start.record()
    for it in range(iters):
        for ci, case in enumerate(cases):
            out = _run_case(case, variants[ci][it % v_timed])
            if (
                tuple(out.shape) == (case["totals"][3],)
                and out.dtype == torch.float32
            ):
                shape_ok[ci][it] = True
                kept[ci][it].copy_(out)
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    import struct

    h = FNV_BASIS
    # C3: grade EVERY timed call against its own variant's oracle and fold the
    # per-call ok-flag. A replayed/stale output fails its variant's check.
    for ci, case in enumerate(cases):
        total_c = case["totals"][3]
        oracle_cache = {}
        for k in range(iters):
            v = k % v_timed
            if v not in oracle_cache:
                a, b, bias = variants[ci][v]
                oracle_cache[v] = _oracle(
                    a,
                    b,
                    bias,
                    case["dims"],
                    case["ao_h"],
                    case["bo_h"],
                    case["bio_h"],
                    case["co_h"],
                    total_c,
                    case["activation"],
                )
            ok = shape_ok[ci][k] and torch.allclose(
                kept[ci][k], oracle_cache[v], atol=ATOL, rtol=RTOL
            )
            h = _fnv_update(h, struct.pack("<qiB", total_c, k, 1 if ok else 0))
            h = _fnv_update(h, case["name"].encode())

    # One extra untimed call per case on its last timed variant (legacy check).
    for ci, case in enumerate(cases):
        total_c = case["totals"][3]
        v = (iters - 1) % v_timed
        got = _run_case(case, variants[ci][v])
        torch.cuda.synchronize()
        a, b, bias = variants[ci][v]
        exp = _oracle(
            a,
            b,
            bias,
            case["dims"],
            case["ao_h"],
            case["bo_h"],
            case["bio_h"],
            case["co_h"],
            total_c,
            case["activation"],
        )
        ok = (
            tuple(got.shape) == (total_c,)
            and got.dtype == torch.float32
            and torch.allclose(got, exp, atol=ATOL, rtol=RTOL)
        )
        h = _fnv_update(h, struct.pack("<qB", total_c, 1 if ok else 0))
        h = _fnv_update(h, str(got.dtype).encode())
        h = _fnv_update(h, case["name"].encode())

    for case in cases:
        print(
            f"bench_case {case['name']:24s} total_c={case['totals'][3]} "
            f"act={case['activation']}"
        )

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

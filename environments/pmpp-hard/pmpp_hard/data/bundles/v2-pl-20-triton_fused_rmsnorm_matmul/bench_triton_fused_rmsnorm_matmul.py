# file: bench_triton_fused_rmsnorm_matmul.py

import os
import sys

import torch

import solution


def _splitmix64(x):
    mask = (1 << 64) - 1
    z = (x + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return z ^ (z >> 31)


# Case data derives from PMPP_BENCH_SEED (whitened) so input values vary per rollout;
# the shape family in main() stays fixed. Fallback keeps stand-alone runs deterministic.
SEED = _splitmix64(int(os.environ.get("PMPP_BENCH_SEED", "2026072002"))) & ((1 << 62) - 1)


# Anti-hack digest (REWARD_HACK_AUDIT_20260704): the paired perf scorer requires the
# student bench and the reference bench to print the SAME out_fnv before accepting the
# student's timing. The graded output is tolerance-graded fp32 (not bitwise stable across
# implementations), so the digest folds exact-graded facts only: output shape/dtype and a
# per-case ok-flag from an untimed torch oracle check at the grading tolerance.
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_MASK = (1 << 64) - 1

ATOL = 8.0e-2
RTOL = 8.0e-2


def _fnv_update(h, data):
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & FNV_MASK
    return h


def _oracle(x, weight, w, bias, eps, activation):
    torch.backends.cuda.matmul.allow_tf32 = False
    xf, wf, mat = x.float(), weight.float(), w.float()
    var = (xf * xf).mean(dim=1, keepdim=True)
    inv_rms = torch.rsqrt(var + float(eps))
    normed = (xf * inv_rms * wf[None, :]).to(torch.float16).float()
    out = normed @ mat
    if bias is not None:
        out = out + bias.float()[None, :]
    act = (activation or "none").lower() if isinstance(activation, (str, type(None))) else str(activation)
    if act == "relu":
        out = torch.relu(out)
    elif act in ("gelu", "gelu_tanh"):
        out = torch.nn.functional.gelu(out, approximate="tanh")
    return out




def _rand_half(shape, device, generator, dist):
    if dist == "uniform":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.35, 0.35, generator=generator
        )
    elif dist == "zeroish":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.30, 0.30, generator=generator
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.60
        x = torch.where(keep, x, torch.zeros_like(x))
    elif dist == "ties":
        vals = torch.tensor(
            [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x = vals[ids]
    else:
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.15, 0.15, generator=generator
        )
        flat = x.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            flat[(idx % 251) == 0] = 0.75
            flat[(idx % 313) == 0] = -0.75

    return x.to(torch.float16).contiguous()


# C3: every timed call gets its OWN x variant so a no-op/cached call yields a stale
# output that fails its per-call oracle check below. w_mat / weight / bias are shared
# per case (the matmul weights); only the activation input x varies per call, which is
# enough to make every call's output distinct. Warmup uses a DISJOINT set of x
# variants so a cache built during warmup is stale for every timed call. All variants
# derive from (PMPP_BENCH_SEED, case, iter) so the same seed yields a bit-identical
# variant sequence in the student and reference benches (paired digest compare intact).
def _make_case(name, M, K, N, eps, activation, dist, seed_offset, n_timed, n_warm):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(SEED + seed_offset)

    w_mat = _rand_half((K, N), device, g, dist)
    weight = torch.empty((K,), device=device, dtype=torch.float32).uniform_(
        0.25, 1.75, generator=g
    ).contiguous()
    bias = torch.empty((N,), device=device, dtype=torch.float32).uniform_(
        -0.50, 0.50, generator=g
    ).contiguous()

    x_timed = [_rand_half((M, K), device, g, dist) for _ in range(n_timed)]
    x_warm = [_rand_half((M, K), device, g, dist) for _ in range(n_warm)]

    return (name, x_timed, x_warm, weight, w_mat, bias, float(eps), activation, dist)


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 30
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    n_warm = 8

    torch.backends.cuda.matmul.allow_tf32 = False

    cases = [
        _make_case("square_1024", 1024, 1024, 1024, 1.0e-5, "relu", "uniform", 1, iters, n_warm),
        _make_case("largeK_512x4096x1024", 512, 4096, 1024, 1.0e-5, "none", "zeroish", 2, iters, n_warm),
        _make_case("largeM_4096x1024x1024", 4096, 1024, 1024, 1.0e-5, "gelu", "ties", 3, iters, n_warm),
        _make_case("wideN_1024x2048x4096", 1024, 2048, 4096, 1.0e-4, "relu", "peaked", 4, iters, n_warm),
    ]

    # Keeping every timed call's output alive (needed to bind it in the digest)
    # would otherwise force the CUDA caching allocator to grow its pool mid-timing
    # (a cudaMalloc sync that inflates avg_ms ~2x). Pre-grow the pool with plain
    # dummies of the output shapes BEFORE timing — no solution call, no input
    # touched (so this can never pre-populate a cache-replay cheat) — then free
    # them; the freed blocks stay in the pool for the timed loop to reuse.
    _dummies = [
        [torch.empty((x_timed[0].shape[0], w_mat.shape[1]), device="cuda", dtype=torch.float32)
         for _ in range(iters)]
        for (_, x_timed, _xw, _w, w_mat, _b, _e, _a, _d) in cases
    ]
    del _dummies
    torch.cuda.synchronize()

    # Warmup on the disjoint warmup variants (never folded).
    for w in range(n_warm):
        for _, _xt, x_warm, weight, w_mat, bias, eps, act, _ in cases:
            _ = solution.fused_rmsnorm_matmul(x_warm[w], weight, w_mat, bias=bias, eps=eps, activation=act)
    torch.cuda.synchronize()

    # Timed loop: call k uses x variant k; keep every call's output so the digest
    # can bind it (a no-op/cached call stores a stale output -> its oracle ok-flag
    # flips -> digest mismatch).
    outs = [[None] * iters for _ in cases]

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for k in range(iters):
        for ci, (_, x_timed, _xw, weight, w_mat, bias, eps, act, _) in enumerate(cases):
            outs[ci][k] = solution.fused_rmsnorm_matmul(
                x_timed[k], weight, w_mat, bias=bias, eps=eps, activation=act)
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    import struct

    # Anti-hack digest: fold a per-timed-call oracle ok-flag (tolerance-robust, so a
    # correct-but-bitwise-different Triton kernel still matches the reference bench)
    # for EVERY timed call, computed against that call's own x variant. Output is
    # tolerance-graded fp32, so we bind the graded FACT (shape/dtype/ok) per call,
    # not raw bits. A cached/stale output for call k fails allclose vs oracle(x_k)
    # -> ok=0 -> digest differs from the honest reference.
    h = FNV_BASIS
    for ci, (name, x_timed, _xw, weight, w_mat, bias, eps, act, dist) in enumerate(cases):
        for k in range(iters):
            got = outs[ci][k]
            exp = _oracle(x_timed[k], weight, w_mat, bias, eps, act)
            ok = (
                got.shape == exp.shape
                and got.dtype == torch.float32
                and torch.allclose(got, exp, atol=ATOL, rtol=RTOL)
            )
            h = _fnv_update(h, struct.pack("<qqBq", got.shape[0], got.shape[1], 1 if ok else 0, k))
            h = _fnv_update(h, str(got.dtype).encode())
            h = _fnv_update(h, name.encode())

    for name, x_timed, _xw, _, w_mat, _, eps, act, dist in cases:
        M, K = x_timed[0].shape
        N = w_mat.shape[1]
        print(f"bench_case {name:24s} M={M} K={K} N={N} eps={eps} act={act} dist={dist} variants={iters}")

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

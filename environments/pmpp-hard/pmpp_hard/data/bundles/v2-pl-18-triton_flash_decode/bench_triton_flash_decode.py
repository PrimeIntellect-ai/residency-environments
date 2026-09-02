# file: bench_triton_flash_decode.py

import struct
import sys

import torch

import solution


import os

# Input data derives from PMPP_BENCH_SEED (fresh per rollout from the scorer) so a
# memorized answer for the fixed cases is wrong on a fresh seed: the in-bench oracle
# ok-flag flips to 0 and the folded cache_lens bytes change. Shapes/windows stay fixed.
SEED = int(os.environ.get("PMPP_BENCH_SEED", "2026071802"))

# Anti-hack digest (REWARD_HACK_AUDIT_20260704 + C3_TIMED_FOLD 20260705): the paired
# perf scorer requires the student bench and the reference bench to print the SAME
# out_fnv before accepting the student's timing. The graded output here is tolerance-
# graded fp32 (not bitwise stable across implementations), so the digest cannot fold raw
# output bytes; it folds, FOR EVERY TIMED CALL, an ok-flag from an untimed torch-oracle
# tolerance check plus the exact-graded shape/dtype/cache_lens facts.
#
# C3 (kill compute-once-cache): every timed iteration uses a DISTINCT query tensor q
# (derived from (SEED, case, iter)); k/v/cache_lens/window stay fixed so the timed
# workload shape is identical across iterations. Each timed call's output is retained
# and, after the timed region, checked against the oracle for ITS iteration's q. A
# compute-once-then-replay solution returns a stale output for iterations 2..K → those
# oracle checks fail (ok=0) → the folded digest diverges from the reference → perf FAIL.
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_MASK = (1 << 64) - 1

ATOL = 2.0e-2
RTOL = 2.0e-2


def _fnv_update(h, data):
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & FNV_MASK
    return h


def _oracle(q, k, v, cache_lens, scale, window):
    # Vectorized-over-heads reference (bitwise-equivalent to the per-head reference);
    # fast enough to run once per timed call in the untimed digest phase.
    B, Hq, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    group = Hq // Hkv

    out = torch.zeros((B, Hq, D), device=q.device, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    lens = cache_lens.to(torch.int64)

    for b in range(B):
        L = min(int(lens[b].item()), S)
        if L <= 0:
            continue
        start = max(0, L - int(window)) if int(window) > 0 else 0
        if start >= L:
            continue
        kb = kf[b, :, start:L].repeat_interleave(group, dim=0)   # (Hq, Lp, D)
        vb = vf[b, :, start:L].repeat_interleave(group, dim=0)   # (Hq, Lp, D)
        qb = qf[b]                                               # (Hq, D)
        scores = (kb * qb[:, None, :]).sum(dim=-1) * float(scale)  # (Hq, Lp)
        probs = torch.softmax(scores, dim=1)                    # (Hq, Lp)
        out[b] = (probs[:, :, None] * vb).sum(dim=1)            # (Hq, D)
    return out


def _rand_half(shape, device, generator, lo=-0.25, hi=0.25):
    x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
        lo, hi, generator=generator
    )
    return x.to(torch.float16)


# q/k use a wider magnitude band than v so the attention scores (q·k / sqrt(D)) span a
# few units and the softmax is SHARP — i.e. the decode output actually depends on which
# key each query selects. With the original tiny band the scores were ~0 and the output
# collapsed to the near-uniform average of v (independent of q), which let a stale q be
# replayed within tolerance. Sharp softmax makes each timed iteration's distinct q yield
# a materially different output, so a cache-replay fails the per-call oracle check.
QK_LO, QK_HI = -3.0, 3.0


def _make_case(B, Hq, Hkv, S, D, window, seed_offset):
    # Builds the FIXED per-case tensors (k/v/cache_lens/scale/window). q is supplied
    # separately, one variant per timed iteration.
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(SEED + seed_offset)

    q = _rand_half((B, Hq, D), device, g, QK_LO, QK_HI)  # base q (warmup variant 0)
    k = _rand_half((B, Hkv, S, D), device, g, QK_LO, QK_HI)
    v = _rand_half((B, Hkv, S, D), device, g)

    cache_lens = torch.randint(
        low=max(1, S // 2),
        high=S + 1,
        size=(B,),
        device=device,
        dtype=torch.int32,
        generator=g,
    )

    if B > 0:
        cache_lens[0] = S

    scale = 1.0 / (D ** 0.5)
    return q, k, v, cache_lens, scale, int(window)


def _make_q(B, Hq, D, seed_offset, it):
    device = "cuda"
    g = torch.Generator(device=device)
    # Distinct per (case, iteration) so every timed call sees a fresh query.
    g.manual_seed(SEED + 1_000_003 * (seed_offset + 1) + it)
    return _rand_half((B, Hq, D), device, g, QK_LO, QK_HI)


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 100
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    case_dims = [
        (64, 4, 1, 128, 64, 0, 1),
        (8, 8, 2, 2048, 64, 256, 2),
        (4, 16, 4, 2048, 128, 512, 3),
        (2, 8, 1, 4096, 64, 0, 4),
    ]
    cases = [_make_case(*cd) for cd in case_dims]

    # Per-iteration q variants (index 0 = warmup, 1..iters = timed calls).
    q_variants = []
    for cd in case_dims:
        B, Hq, Hkv, S, D, window, off = cd
        q_variants.append([_make_q(B, Hq, D, off, it) for it in range(iters + 1)])

    # Pre-flatten the timed call arguments so the timed loop carries no per-call Python
    # bookkeeping beyond the kernel launch (these cases run in ~microseconds, so any
    # extra indexing would dominate the measurement). warm_args uses variant 0.
    warm_args = []
    timed_args = []
    for ci in range(len(cases)):
        q0, k, v, cache_lens, scale, window = cases[ci]
        warm_args.append((q_variants[ci][0], k, v, cache_lens, scale, window))
    for it in range(iters):
        for ci in range(len(cases)):
            q0, k, v, cache_lens, scale, window = cases[ci]
            timed_args.append((q_variants[ci][it + 1], k, v, cache_lens, scale, window))

    for _ in range(10):
        for a in warm_args:
            _ = solution.flash_decode(*a)

    # Prime the caching allocator to hold as many concurrent output buffers as the timed
    # loop will retain, then release them back to the pool. This keeps the timed loop
    # free of cudaMalloc (the outputs are only KB, but a fresh alloc costs ~as much as
    # these microsecond kernels), so retaining outputs for the digest does not inflate
    # the measured time.
    prime = [solution.flash_decode(*a) for a in timed_args]
    del prime
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    # Retain every timed call's output so the digest can bind all of them.
    outs = []

    start.record()
    for a in timed_args:
        outs.append(solution.flash_decode(*a))
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    h = FNV_BASIS
    idx = 0
    for it in range(iters):
        for ci in range(len(cases)):
            q0, k, v, cache_lens, scale, window = cases[ci]
            qv = q_variants[ci][it + 1]
            out = outs[idx]
            idx += 1
            exp = _oracle(qv, k, v, cache_lens, scale, window)
            ok = (
                out.shape == exp.shape
                and out.dtype == torch.float32
                and torch.allclose(out, exp, rtol=RTOL, atol=ATOL)
            )
            h = _fnv_update(h, struct.pack("<4qB", *out.shape, out.numel(), 1 if ok else 0))
            h = _fnv_update(h, str(out.dtype).encode())
            h = _fnv_update(h, cache_lens.cpu().numpy().tobytes())

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

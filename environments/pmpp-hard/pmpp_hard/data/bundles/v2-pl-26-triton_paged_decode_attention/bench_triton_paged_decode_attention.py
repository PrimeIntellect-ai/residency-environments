# file: bench_triton_paged_decode_attention.py

import struct
import sys

import torch

import solution


import os

# Input data derives from PMPP_BENCH_SEED (fresh per rollout from the scorer) so a
# memorized answer for the fixed cases is wrong on a fresh seed: the in-bench oracle
# ok-flag flips to 0 and the folded cache_lens bytes change. Shapes/windows stay fixed.
SEED = int(os.environ.get("PMPP_BENCH_SEED", "2026072602"))


# Anti-hack digest (REWARD_HACK_AUDIT_20260704 + C3_TIMED_FOLD 20260705): the paired
# perf scorer requires the student bench and the reference bench to print the SAME
# out_fnv before accepting the student's timing. The graded output is tolerance-graded
# fp32 (not bitwise stable across implementations), so the digest folds, FOR EVERY TIMED
# CALL, an ok-flag from an untimed torch-oracle tolerance check plus the exact-graded
# shape/dtype/cache_lens facts.
#
# C3 (kill compute-once-cache): every timed iteration uses a DISTINCT query tensor q
# (derived from (SEED, case, iter)); the paged k/v caches, page_table, cache_lens and
# window stay fixed so the timed workload shape is identical across iterations. Each
# timed call's output is retained and, after the timed region, checked against the oracle
# for ITS iteration's q. A compute-once-then-replay solution returns a stale output for
# iterations 2..K → those oracle checks fail (ok=0) → the folded digest diverges → FAIL.
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_MASK = (1 << 64) - 1

ATOL = 2.0e-2
RTOL = 2.0e-2

# q/k use a wider magnitude band than v so the attention scores (q·k / sqrt(D)) span a
# few units and the softmax is SHARP — the decode output then depends on which key each
# query selects. With a tiny band the scores are ~0, the softmax collapses to a near-
# uniform average of v (independent of q), and a stale q replays within tolerance; sharp
# softmax makes each timed iteration's distinct q yield a materially different output.
QK_LO, QK_HI = -3.0, 3.0
V_LO, V_HI = -0.25, 0.25


def _fnv_update(h, data):
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & FNV_MASK
    return h


def _oracle(q, k_cache, v_cache, page_table, cache_lens, scale, window, page_size):
    # Vectorized-over-heads reference; fast enough to run once per timed call untimed.
    B, Hq, D = q.shape
    Hkv = k_cache.shape[1]
    qf, kf, vf = q.float(), k_cache.float(), v_cache.float()
    pt = page_table.to(torch.long)
    lens = cache_lens.to(torch.long)
    out = torch.zeros((B, Hq, D), device=q.device, dtype=torch.float32)
    group = Hq // Hkv
    for b in range(B):
        L = int(lens[b].item())
        if L <= 0:
            continue
        max_tokens = pt.shape[1] * int(page_size)
        if L > max_tokens:
            L = max_tokens
        start = max(0, L - int(window)) if int(window) > 0 else 0
        if start >= L:
            continue
        tok = torch.arange(start, L, device=q.device, dtype=torch.long)
        page_ids = tok // int(page_size)
        page_offs = tok - page_ids * int(page_size)
        phys = pt[b, page_ids]
        valid = phys >= 0
        if not bool(valid.any().item()):
            continue
        phys = phys[valid]
        page_offs = page_offs[valid]
        k_tokens = kf[phys, :, page_offs, :]   # (Ntok, Hkv, D)
        v_tokens = vf[phys, :, page_offs, :]   # (Ntok, Hkv, D)
        k_by_hq = k_tokens.repeat_interleave(group, dim=1)   # (Ntok, Hq, D)
        v_by_hq = v_tokens.repeat_interleave(group, dim=1)   # (Ntok, Hq, D)
        qb = qf[b]                                            # (Hq, D)
        scores = (k_by_hq * qb[None, :, :]).sum(dim=-1) * float(scale)  # (Ntok, Hq)
        probs = torch.softmax(scores, dim=0)                 # (Ntok, Hq)
        out[b] = (probs[:, :, None] * v_by_hq).sum(dim=0)    # (Hq, D)
    return out


def _rand_half(shape, device, generator, lo, hi):
    x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
        lo, hi, generator=generator
    )
    return x.to(torch.float16).contiguous()


def _make_case(B, Hq, Hkv, max_len, D, page_size, window, seed_offset):
    # Builds the FIXED per-case tensors (paged caches, page_table, cache_lens, scale,
    # window, page_size). q is supplied separately, one variant per timed iteration.
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(SEED + seed_offset)

    max_pages = (max_len + page_size - 1) // page_size
    num_phys_pages = B * max_pages + 31

    q = _rand_half((B, Hq, D), device, g, QK_LO, QK_HI)  # base q (warmup variant 0)
    k_cache = _rand_half((num_phys_pages, Hkv, page_size, D), device, g, QK_LO, QK_HI)
    v_cache = _rand_half((num_phys_pages, Hkv, page_size, D), device, g, V_LO, V_HI)

    all_pages = torch.randperm(num_phys_pages, device=device, generator=g)
    page_table = torch.full((B, max_pages), -1, device=device, dtype=torch.int32)

    cursor = 0
    for b in range(B):
        for p in range(max_pages):
            page_table[b, p] = int(all_pages[cursor].item())
            cursor += 1

    cache_lens = torch.randint(
        low=max(1, max_len // 2),
        high=max_len + 1,
        size=(B,),
        device=device,
        dtype=torch.int32,
        generator=g,
    )
    cache_lens[0] = max_len

    scale = 1.0 / (D ** 0.5)

    return q, k_cache, v_cache, page_table, cache_lens, scale, int(window), int(page_size)


def _make_q(B, Hq, D, seed_offset, it):
    device = "cuda"
    g = torch.Generator(device=device)
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
        (32, 4, 1, 128, 64, 16, 0, 1),
        (8, 8, 2, 2048, 64, 16, 256, 2),
        (4, 16, 4, 2048, 128, 32, 512, 3),
        (2, 8, 1, 4096, 64, 32, 0, 4),
    ]
    cases = [_make_case(*cd) for cd in case_dims]

    # Per-iteration q variants (index 0 = warmup, 1..iters = timed calls).
    q_variants = []
    for cd in case_dims:
        B, Hq, Hkv, max_len, D, page_size, window, off = cd
        q_variants.append([_make_q(B, Hq, D, off, it) for it in range(iters + 1)])

    def call(ci, qv):
        _q, k_cache, v_cache, page_table, cache_lens, scale, window, page_size = cases[ci]
        return solution.paged_decode_attention(
            qv, k_cache, v_cache, page_table, cache_lens, scale, window, page_size=page_size
        )

    # Pre-flatten timed args so the timed loop carries no extra Python bookkeeping.
    warm_calls = [(ci, q_variants[ci][0]) for ci in range(len(cases))]
    timed_calls = [
        (ci, q_variants[ci][it + 1])
        for it in range(iters)
        for ci in range(len(cases))
    ]

    for _ in range(10):
        for ci, qv in warm_calls:
            _ = call(ci, qv)

    # Prime the caching allocator to hold as many concurrent output buffers as the timed
    # loop retains, then release them, so retaining outputs for the digest does not force
    # cudaMalloc during timing (these cases run in ~microseconds).
    prime = [call(ci, qv) for ci, qv in timed_calls]
    del prime
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    outs = []
    start.record()
    for ci, qv in timed_calls:
        outs.append(call(ci, qv))
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    h = FNV_BASIS
    for idx, (ci, qv) in enumerate(timed_calls):
        _q, k_cache, v_cache, page_table, cache_lens, scale, window, page_size = cases[ci]
        got = outs[idx]
        exp = _oracle(qv, k_cache, v_cache, page_table, cache_lens, scale, window, page_size)
        ok = (
            got.shape == exp.shape
            and got.dtype == torch.float32
            and torch.allclose(got, exp, rtol=RTOL, atol=ATOL)
        )
        h = _fnv_update(h, struct.pack("<4qB", *got.shape, got.numel(), 1 if ok else 0))
        h = _fnv_update(h, str(got.dtype).encode())
        h = _fnv_update(h, cache_lens.cpu().numpy().tobytes())

    for i in range(len(cases)):
        _q, k_cache, v_cache, page_table, cache_lens, scale, window, page_size = cases[i]
        B, Hq, D = cases[i][0].shape
        _, Hkv, _, _ = k_cache.shape
        print(
            f"bench_case {i} B={B} Hq={Hq} Hkv={Hkv} D={D} "
            f"pages_per_seq={page_table.shape[1]} page_size={page_size} "
            f"max_len={int(cache_lens.max().item())} window={window}"
        )

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

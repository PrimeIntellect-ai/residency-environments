# file: bench_triton_fused_moe_dispatch.py

import os
import sys

import torch

import solution


# Input data values derive from PMPP_BENCH_SEED so correct outputs are not
# precomputable offline; shapes stay fixed so timing remains comparable.
SEED = (2026073102 ^ (int(os.environ.get("PMPP_BENCH_SEED", "0")) * 0x9E3779B97F4A7C15)) & (
    (1 << 63) - 1
)


# Anti-hack digest (REWARD_HACK_AUDIT_20260704): the paired perf scorer requires the
# student bench and the reference bench to print the SAME out_fnv before accepting the
# student's timing. The graded output is tolerance-graded fp32 (not bitwise stable across
# implementations), so the digest folds exact-graded facts only: output shape/dtype and
# per-call ok-flags from untimed torch oracle checks at the grading tolerance.
#
# C3 anti-cache design (patterns I1 + P1):
#   - Every timed call k receives a FRESH pregenerated (x, ids, gates) variant derived
#     from (PMPP_BENCH_SEED, case, k); expert weights stay fixed per case (in-family).
#     Warmup uses two dedicated warmup-only variants, so no timed variant is ever
#     visible before the timer starts.
#   - The timed loop snapshots every call's output into a preallocated slab; after the
#     timer each timed call is checked against ITS OWN variant's oracle at the grading
#     tolerance and the per-call ok-flag is folded into out_fnv. A cached/no-op/stale
#     timed call fails its variant's check and breaks the digest regardless of timing.
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_MASK = (1 << 64) - 1

ATOL = 1.2e-1
RTOL = 8.0e-2


def _fnv_update(h, data):
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & FNV_MASK
    return h


def _oracle(x, expert_ids, gates, w1, w2):
    torch.backends.cuda.matmul.allow_tf32 = False
    N, D = x.shape
    E = w1.shape[0]
    Kroute = expert_ids.shape[1]
    out = torch.zeros((N, D), device=x.device, dtype=torch.float32)
    xf, w1f, w2f, gatesf = x.float(), w1.float(), w2.float(), gates.float()
    for r in range(Kroute):
        ids_r = expert_ids[:, r].to(torch.long)
        for e in range(E):
            mask = ids_r == e
            if not bool(mask.any().item()):
                continue
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            hidden = xf[idx] @ w1f[e]
            hidden = hidden * torch.sigmoid(hidden)
            out[idx] += gatesf[idx, r][:, None] * (hidden @ w2f[e])
    return out




def _rand_half(shape, device, generator, dist):
    if dist == "hot":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.08, 0.08, generator=generator
        )
        flat = x.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            flat[(idx % 257) == 0] = 0.25
            flat[(idx % 389) == 0] = -0.25
    elif dist == "zeroish":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.10, 0.10, generator=generator
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.65
        x = torch.where(keep, x, torch.zeros_like(x))
    else:
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.10, 0.10, generator=generator
        )

    return x.to(torch.float16).contiguous()


def _make_routes(N, E, Kroute, device, generator, hot):
    if hot:
        hot_ids = torch.randint(0, min(E, 4), (N, Kroute), device=device, generator=generator)
        cold_ids = torch.randint(0, E, (N, Kroute), device=device, generator=generator)
        choose_hot = torch.rand((N, Kroute), device=device, generator=generator) < 0.85
        ids = torch.where(choose_hot, hot_ids, cold_ids).to(torch.int32)
    else:
        ids = torch.randint(0, E, (N, Kroute), device=device, dtype=torch.int32, generator=generator)

    if Kroute == 1:
        gates = torch.ones((N, 1), device=device, dtype=torch.float32)
    else:
        g0 = torch.empty((N,), device=device, dtype=torch.float32).uniform_(0.15, 0.85, generator=generator)
        gates = torch.stack([g0, 1.0 - g0], dim=1).contiguous()

    return ids.contiguous(), gates.contiguous()


def _make_case(name, N, D, H, E, Kroute, hot, value_dist, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(SEED + seed_offset)

    w1 = _rand_half((E, D, H), device, g, value_dist)
    w2 = _rand_half((E, H, D), device, g, value_dist)

    return {
        "name": name,
        "N": N,
        "D": D,
        "H": H,
        "E": E,
        "Kroute": Kroute,
        "hot": hot,
        "dist": value_dist,
        "seed_offset": seed_offset,
        "w1": w1,
        "w2": w2,
    }


def _make_variant(case, k):
    # (x, ids, gates) variant for timed iteration k (k >= 0) or a warmup slot
    # (k < 0): a pure function of (PMPP_BENCH_SEED, case, k). Weights stay
    # fixed per case so timing stays in-family.
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(
        (SEED + case["seed_offset"] * 1000003 + (k + 16) * 7919) & ((1 << 63) - 1)
    )
    x = _rand_half((case["N"], case["D"]), device, g, case["dist"])
    ids, gates = _make_routes(
        case["N"], case["E"], case["Kroute"], device, g, case["hot"]
    )
    return (x, ids, gates)


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 30
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    cases = [
        _make_case("N512_D256_H512_E8_K2_hot", 512, 256, 512, 8, 2, True, "hot", 1),
        _make_case("N1024_D512_H256_E32_K1_hot", 1024, 512, 256, 32, 1, True, "uniform", 2),
        _make_case("N256_D1024_H512_E32_K2", 256, 1024, 512, 32, 2, False, "zeroish", 3),
        _make_case("N256_D512_H2048_E8_K2", 256, 512, 2048, 8, 2, True, "zeroish", 4),
    ]

    # C3 variant ring: one dedicated (x, ids, gates) variant per timed
    # iteration (capped for oversized local runs) plus two warmup-only ones.
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
            x, ids, gates = warm_variants[ci][w % n_warm]
            _ = solution.fused_moe_dispatch(x, ids, gates, case["w1"], case["w2"])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    # kept[ci][k] snapshots timed call k's output. Preallocated slab + copy_
    # keeps the timed loop free of allocations; the copy also de-aliases
    # solutions that reuse their output buffer across calls.
    kept = [
        torch.empty(
            (iters, case["N"], case["D"]), device="cuda", dtype=torch.float32
        )
        for case in cases
    ]
    shape_ok = [[False] * iters for _ in cases]

    start.record()
    for it in range(iters):
        for ci, case in enumerate(cases):
            x, ids, gates = variants[ci][it % v_timed]
            out = solution.fused_moe_dispatch(x, ids, gates, case["w1"], case["w2"])
            if (
                tuple(out.shape) == (case["N"], case["D"])
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
        oracle_cache = {}
        for k in range(iters):
            v = k % v_timed
            if v not in oracle_cache:
                x, ids, gates = variants[ci][v]
                oracle_cache[v] = _oracle(x, ids, gates, case["w1"], case["w2"])
            ok = shape_ok[ci][k] and torch.allclose(
                kept[ci][k], oracle_cache[v], atol=ATOL, rtol=RTOL
            )
            h = _fnv_update(
                h,
                struct.pack(
                    "<qqiB", case["N"], case["D"], k, 1 if ok else 0
                ),
            )
            h = _fnv_update(h, case["name"].encode())

    # One extra untimed call per case on its last timed variant (legacy check).
    for ci, case in enumerate(cases):
        v = (iters - 1) % v_timed
        x, ids, gates = variants[ci][v]
        got = solution.fused_moe_dispatch(x, ids, gates, case["w1"], case["w2"])
        torch.cuda.synchronize()
        exp = _oracle(x, ids, gates, case["w1"], case["w2"])
        ok = (
            got.shape == exp.shape
            and got.dtype == torch.float32
            and torch.allclose(got, exp, atol=ATOL, rtol=RTOL)
        )
        h = _fnv_update(h, struct.pack("<qqB", got.shape[0], got.shape[1], 1 if ok else 0))
        h = _fnv_update(h, str(got.dtype).encode())
        h = _fnv_update(h, case["name"].encode())

    for case in cases:
        print(
            f"bench_case {case['name']:28s} N={case['N']} D={case['D']} "
            f"H={case['H']} E={case['E']} K={case['Kroute']} "
            f"hot={int(case['hot'])} dist={case['dist']}"
        )

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

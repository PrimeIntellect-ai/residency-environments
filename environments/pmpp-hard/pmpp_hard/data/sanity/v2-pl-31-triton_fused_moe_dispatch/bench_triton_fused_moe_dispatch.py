# file: bench_triton_fused_moe_dispatch.py

import sys

import torch

import solution


SEED = 2026073102


# Anti-hack digest (REWARD_HACK_AUDIT_20260704): the paired perf scorer requires the
# student bench and the reference bench to print the SAME out_fnv before accepting the
# student's timing. The graded output is tolerance-graded fp32 (not bitwise stable across
# implementations), so the digest folds exact-graded facts only: output shape/dtype and a
# per-case ok-flag from an untimed torch oracle check at the grading tolerance.
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
    g.manual_seed((SEED + seed_offset) + 880123)

    x = _rand_half((N, D), device, g, value_dist)
    w1 = _rand_half((E, D, H), device, g, value_dist)
    w2 = _rand_half((E, H, D), device, g, value_dist)
    ids, gates = _make_routes(N, E, Kroute, device, g, hot)

    return name, x, ids, gates, w1, w2, hot, value_dist


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

    for _ in range(8):
        for _, x, ids, gates, w1, w2, _, _ in cases:
            _ = solution.fused_moe_dispatch(x, ids, gates, w1, w2)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        for _, x, ids, gates, w1, w2, _, _ in cases:
            _ = solution.fused_moe_dispatch(x, ids, gates, w1, w2)
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    import struct

    h = FNV_BASIS
    for name, x, ids, gates, w1, w2, _, _ in cases:
        got = solution.fused_moe_dispatch(x, ids, gates, w1, w2)
        torch.cuda.synchronize()
        exp = _oracle(x, ids, gates, w1, w2)
        ok = (
            got.shape == exp.shape
            and got.dtype == torch.float32
            and torch.allclose(got, exp, atol=ATOL, rtol=RTOL)
        )
        h = _fnv_update(h, struct.pack("<qqB", got.shape[0], got.shape[1], 1 if ok else 0))
        h = _fnv_update(h, str(got.dtype).encode())
        h = _fnv_update(h, name.encode())

    for name, x, ids, _, w1, _, hot, dist in cases:
        N, D = x.shape
        E, _, H = w1.shape
        Kroute = ids.shape[1]
        print(f"bench_case {name:28s} N={N} D={D} H={H} E={E} K={Kroute} hot={int(hot)} dist={dist}")

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

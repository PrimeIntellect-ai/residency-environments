# file: bench_triton_chunked_state_scan.py

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
SEED = _splitmix64(int(os.environ.get("PMPP_BENCH_SEED", "2026072102"))) & ((1 << 62) - 1)


# Anti-hack digest (REWARD_HACK_AUDIT_20260704): the paired perf scorer requires the
# student bench and the reference bench to print the SAME out_fnv before accepting the
# student's timing. Both graded outputs (y, final_state) are tolerance-graded fp32, so the
# digest folds exact-graded facts only: shapes/dtypes and per-case ok-flags from an untimed
# vectorized torch oracle check at the grading tolerance.
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_MASK = (1 << 64) - 1

ATOL = 5.0e-2
RTOL = 5.0e-2


def _fnv_update(h, data):
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & FNV_MASK
    return h


def _oracle(x, decay, gate, out_scale, skip, lengths, chunk=64):
    B, T, H, D = x.shape
    dev = x.device
    C = (T + chunk - 1) // chunk
    Tp = C * chunk

    xf, df, gf = x.float(), decay.float(), gate.float()
    if lengths is None:
        lens = torch.full((B,), T, device=dev, dtype=torch.int64)
    else:
        lens = lengths.to(torch.int64)
    lens = lens.clamp(0, T)

    if Tp > T:
        xf = torch.cat((xf, torch.zeros((B, Tp - T, H, D), device=dev)), dim=1)
        df = torch.cat((df, torch.ones((B, Tp - T, H), device=dev)), dim=1)
        gf = torch.cat((gf, torch.zeros((B, Tp - T, H), device=dev)), dim=1)

    tidx = torch.arange(Tp, device=dev)
    valid = tidx[None, :] < lens[:, None]
    df = torch.where(valid[:, :, None], df, torch.ones_like(df))
    gf = torch.where(valid[:, :, None], gf, torch.zeros_like(gf))

    xr = xf.view(B, C, chunk, H, D)
    dr = df.view(B, C, chunk, H)
    gr = gf.view(B, C, chunk, H)

    state = torch.zeros((B, C, H, D), device=dev)
    slocal = torch.empty((B, C, chunk, H, D), device=dev)
    for i in range(chunk):
        state = dr[:, :, i, :, None] * state + gr[:, :, i, :, None] * xr[:, :, i]
        slocal[:, :, i] = state

    pref = torch.cumprod(dr, dim=2)
    carry = torch.zeros((B, H, D), device=dev)
    carries = torch.empty((B, C, H, D), device=dev)
    for c in range(C):
        carries[:, c] = carry
        carry = state[:, c] + pref[:, c, -1, :, None] * carry
    final_state = carry

    full = slocal + pref[..., None] * carries[:, :, None]
    fs = full.view(B, Tp, H, D)[:, :T]
    y = fs * out_scale.float()[None, None] + skip.float()[None, None] * xf.view(B, Tp, H, D)[:, :T]
    y = torch.where(valid[:, :T, None, None], y, torch.zeros_like(y))
    return y, final_state




def _rand_half(shape, device, generator, dist):
    if dist == "zeroish":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.30, 0.30, generator=generator
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.65
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
            -0.25, 0.25, generator=generator
        )

    return x.to(torch.float16).contiguous()


# C3: every timed call gets its OWN x variant so a no-op/cached call yields a stale
# (y, final_state) that fails its per-call oracle check below. decay/gate/out_scale/
# skip/lengths are shared per case; only the activation input x varies per call,
# which is enough to make every call's outputs distinct. Warmup uses a DISJOINT set
# of x variants so a cache built during warmup is stale for every timed call. All
# variants derive from (PMPP_BENCH_SEED, case, iter) so the same seed yields a
# bit-identical variant sequence in the student and reference benches.
def _make_case(name, B, T, H, D, chunk_size, dist, seed_offset, n_timed, n_warm):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(SEED + seed_offset)

    decay = torch.empty((B, T, H), device=device, dtype=torch.float32).uniform_(
        0.72, 0.995, generator=g
    ).contiguous()
    gate = torch.empty((B, T, H), device=device, dtype=torch.float32).uniform_(
        -0.35, 0.35, generator=g
    ).contiguous()
    out_scale = torch.empty((H, D), device=device, dtype=torch.float32).uniform_(
        -1.25, 1.25, generator=g
    ).contiguous()
    skip = torch.empty((H, D), device=device, dtype=torch.float32).uniform_(
        -0.50, 0.50, generator=g
    ).contiguous()

    lengths = torch.randint(
        low=max(1, T // 2),
        high=T + 1,
        size=(B,),
        device=device,
        dtype=torch.int32,
        generator=g,
    )
    if B > 0:
        lengths[0] = T

    x_timed = [_rand_half((B, T, H, D), device, g, dist) for _ in range(n_timed)]
    x_warm = [_rand_half((B, T, H, D), device, g, dist) for _ in range(n_warm)]

    return (name, x_timed, x_warm, decay, gate, out_scale, skip, lengths, chunk_size, dist)


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 50
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    n_warm = 8

    cases = [
        _make_case("B4_T512_H4_D64", 4, 512, 4, 64, 64, "zeroish", 1, iters, n_warm),
        _make_case("B8_T1024_H2_D128", 8, 1024, 2, 128, 128, "ties", 2, iters, n_warm),
        _make_case("B2_T2048_H8_D64", 2, 2048, 8, 64, 64, "uniform", 3, iters, n_warm),
        _make_case("B4_T1536_H4_D128", 4, 1536, 4, 128, 32, "zeroish", 4, iters, n_warm),
    ]

    # Warmup on the disjoint warmup variants (never folded).
    for w in range(n_warm):
        for _, _xt, x_warm, decay, gate, out_scale, skip, lengths, chunk_size, _ in cases:
            _ = solution.chunked_state_scan(
                x_warm[w], decay, gate, out_scale, skip, lengths=lengths, chunk_size=chunk_size
            )
    torch.cuda.synchronize()

    # Timed loop: call k uses x variant k. To bind every call we KEEP its final_state
    # ([B,H,D], a few KB) — it is the carry after the ENTIRE recurrence over x_k, so
    # producing it correctly REQUIRES running the full scan (y is a byproduct). A
    # no-op/cached call stores a stale final_state whose per-call oracle ok-flag
    # flips. Only final_state is retained so the timed region stays light.
    states = [[None] * iters for _ in cases]

    # Hoist per-case fields into flat lists so the timed loop does minimal Python
    # (this kernel is ~50us, so per-iter interpreter overhead is not negligible).
    fn = solution.chunked_state_scan
    xts = [c[1] for c in cases]
    decs = [c[3] for c in cases]
    gts = [c[4] for c in cases]
    oss = [c[5] for c in cases]
    sks = [c[6] for c in cases]
    lns = [c[7] for c in cases]
    css = [c[8] for c in cases]
    ncase = len(cases)

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for k in range(iters):
        for ci in range(ncase):
            states[ci][k] = fn(xts[ci][k], decs[ci], gts[ci], oss[ci], sks[ci],
                               lengths=lns[ci], chunk_size=css[ci])[1]
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    import struct

    # Anti-hack digest: fold a per-timed-call oracle ok-flag on final_state
    # (tolerance-robust, so a correct-but-bitwise-different Triton kernel still
    # matches the reference bench) for EVERY timed call, computed against that call's
    # own x variant. A cached/stale final_state for call k fails allclose vs
    # oracle(x_k) -> ok=0 -> digest differs from the honest reference.
    h = FNV_BASIS
    for ci, (name, x_timed, _xw, decay, gate, out_scale, skip, lengths, chunk_size, _) in enumerate(cases):
        for k in range(iters):
            got_state = states[ci][k]
            _exp_y, exp_state = _oracle(x_timed[k], decay, gate, out_scale, skip, lengths)
            ok_s = (
                got_state.shape == exp_state.shape
                and got_state.dtype == torch.float32
                and torch.allclose(got_state, exp_state, atol=ATOL, rtol=RTOL)
            )
            h = _fnv_update(h, struct.pack("<3qBq", *exp_state.shape, 1 if ok_s else 0, k))
            h = _fnv_update(h, str(got_state.dtype).encode())
            h = _fnv_update(h, name.encode())

    for name, x_timed, _xw, _, _, _, _, _, chunk_size, dist in cases:
        B, T, H, D = x_timed[0].shape
        print(f"bench_case {name:20s} B={B} T={T} H={H} D={D} chunk={chunk_size} dist={dist} variants={iters}")

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

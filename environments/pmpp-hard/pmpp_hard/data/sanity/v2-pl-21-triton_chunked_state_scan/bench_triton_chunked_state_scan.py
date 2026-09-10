# file: bench_triton_chunked_state_scan.py

import sys

import torch

import solution


SEED = 2026072102


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


def _make_case(name, B, T, H, D, chunk_size, dist, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed((SEED + seed_offset) + 880123)

    x = _rand_half((B, T, H, D), device, g, dist)
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

    return (name, x, decay, gate, out_scale, skip, lengths, chunk_size, dist)


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 50
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    cases = [
        _make_case("B4_T512_H4_D64", 4, 512, 4, 64, 64, "zeroish", 1),
        _make_case("B8_T1024_H2_D128", 8, 1024, 2, 128, 128, "ties", 2),
        _make_case("B2_T2048_H8_D64", 2, 2048, 8, 64, 64, "uniform", 3),
        _make_case("B4_T1536_H4_D128", 4, 1536, 4, 128, 32, "zeroish", 4),
    ]

    for _ in range(8):
        for _, x, decay, gate, out_scale, skip, lengths, chunk_size, _ in cases:
            _ = solution.chunked_state_scan(
                x,
                decay,
                gate,
                out_scale,
                skip,
                lengths=lengths,
                chunk_size=chunk_size,
            )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        for _, x, decay, gate, out_scale, skip, lengths, chunk_size, _ in cases:
            _ = solution.chunked_state_scan(
                x,
                decay,
                gate,
                out_scale,
                skip,
                lengths=lengths,
                chunk_size=chunk_size,
            )
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    import struct

    h = FNV_BASIS
    for name, x, decay, gate, out_scale, skip, lengths, chunk_size, _ in cases:
        got_y, got_state = solution.chunked_state_scan(
            x, decay, gate, out_scale, skip, lengths=lengths, chunk_size=chunk_size
        )
        torch.cuda.synchronize()
        exp_y, exp_state = _oracle(x, decay, gate, out_scale, skip, lengths)
        ok_y = (
            got_y.shape == exp_y.shape
            and got_y.dtype == torch.float32
            and torch.allclose(got_y, exp_y, atol=ATOL, rtol=RTOL)
        )
        ok_s = (
            got_state.shape == exp_state.shape
            and got_state.dtype == torch.float32
            and torch.allclose(got_state, exp_state, atol=ATOL, rtol=RTOL)
        )
        h = _fnv_update(h, struct.pack("<4qBB", *got_y.shape, 1 if ok_y else 0, 1 if ok_s else 0))
        h = _fnv_update(h, str(got_y.dtype).encode() + str(got_state.dtype).encode())
        h = _fnv_update(h, name.encode())

    for name, x, _, _, _, _, _, chunk_size, dist in cases:
        B, T, H, D = x.shape
        print(f"bench_case {name:20s} B={B} T={T} H={H} D={D} chunk={chunk_size} dist={dist}")

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

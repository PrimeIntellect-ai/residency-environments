# file: bench_triton_flash_decode.py

import struct
import sys

import torch

import solution


SEED = 2026071802

# Anti-hack digest (REWARD_HACK_AUDIT_20260704): the paired perf scorer requires the
# student bench and the reference bench to print the SAME out_fnv before accepting the
# student's timing. The graded output here is tolerance-graded fp32 (not bitwise stable
# across implementations), so the digest folds only exact-graded facts: output shape,
# output dtype, the (immutable) cache_lens bytes, and an ok-flag from an untimed torch
# oracle check at the test's tolerance. A shape/no-op/fabricated-output hack cannot match.
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
        for hq in range(Hq):
            hkv = hq // group
            scores = (kf[b, hkv, start:L] * qf[b, hq][None, :]).sum(dim=-1) * float(scale)
            probs = torch.softmax(scores, dim=0)
            out[b, hq] = (probs[:, None] * vf[b, hkv, start:L]).sum(dim=0)
    return out


def _rand_half(shape, device, generator):
    x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
        -0.25, 0.25, generator=generator
    )
    return x.to(torch.float16)


def _make_case(B, Hq, Hkv, S, D, window, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed((SEED + seed_offset) + 880123)

    q = _rand_half((B, Hq, D), device, g)
    k = _rand_half((B, Hkv, S, D), device, g)
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


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 100
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    cases = [
        _make_case(64, 4, 1, 128, 64, 0, 1),
        _make_case(8, 8, 2, 2048, 64, 256, 2),
        _make_case(4, 16, 4, 2048, 128, 512, 3),
        _make_case(2, 8, 1, 4096, 64, 0, 4),
    ]

    for _ in range(10):
        for args in cases:
            _ = solution.flash_decode(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        for args in cases:
            _ = solution.flash_decode(*args)
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    h = FNV_BASIS
    for args in cases:
        out = solution.flash_decode(*args)
        torch.cuda.synchronize()
        exp = _oracle(*args)
        ok = (
            out.shape == exp.shape
            and out.dtype == torch.float32
            and torch.allclose(out, exp, rtol=RTOL, atol=ATOL)
        )
        h = _fnv_update(h, struct.pack("<4qB", *out.shape, out.numel(), 1 if ok else 0))
        h = _fnv_update(h, str(out.dtype).encode())
        h = _fnv_update(h, args[3].cpu().numpy().tobytes())

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

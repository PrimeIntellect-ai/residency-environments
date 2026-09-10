# file: bench_triton_paged_decode_attention.py

import sys

import torch

import solution


SEED = 2026072602


# Anti-hack digest (REWARD_HACK_AUDIT_20260704): the paired perf scorer requires the
# student bench and the reference bench to print the SAME out_fnv before accepting the
# student's timing. The graded output is tolerance-graded fp32 (not bitwise stable across
# implementations), so the digest folds exact-graded facts only: output shape/dtype, the
# (immutable) cache_lens bytes, and a per-case ok-flag from an untimed torch oracle check
# at the grading tolerance.
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_MASK = (1 << 64) - 1

ATOL = 2.0e-2
RTOL = 2.0e-2


def _fnv_update(h, data):
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & FNV_MASK
    return h


def _oracle(q, k_cache, v_cache, page_table, cache_lens, scale, window, page_size):
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
        k_tokens = kf[phys, :, page_offs, :]
        v_tokens = vf[phys, :, page_offs, :]
        for hq in range(Hq):
            hkv = hq // group
            scores = (k_tokens[:, hkv, :] * qf[b, hq][None, :]).sum(dim=-1) * float(scale)
            probs = torch.softmax(scores, dim=0)
            out[b, hq] = (probs[:, None] * v_tokens[:, hkv, :]).sum(dim=0)
    return out




def _rand_half(shape, device, generator, dist):
    if dist == "sparse":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.25, 0.25, generator=generator
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.70
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


def _make_case(B, Hq, Hkv, max_len, D, page_size, window, dist, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed((SEED + seed_offset) + 880123)

    max_pages = (max_len + page_size - 1) // page_size
    num_phys_pages = B * max_pages + 31

    q = _rand_half((B, Hq, D), device, g, dist)
    k_cache = _rand_half((num_phys_pages, Hkv, page_size, D), device, g, dist)
    v_cache = _rand_half((num_phys_pages, Hkv, page_size, D), device, g, dist)

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


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 100
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    cases = [
        _make_case(32, 4, 1, 128, 64, 16, 0, "uniform", 1),
        _make_case(8, 8, 2, 2048, 64, 16, 256, "sparse", 2),
        _make_case(4, 16, 4, 2048, 128, 32, 512, "ties", 3),
        _make_case(2, 8, 1, 4096, 64, 32, 0, "uniform", 4),
    ]

    for _ in range(10):
        for args in cases:
            _ = solution.paged_decode_attention(*args[:-1], page_size=args[-1])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        for args in cases:
            _ = solution.paged_decode_attention(*args[:-1], page_size=args[-1])
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    import struct

    h = FNV_BASIS
    for args in cases:
        got = solution.paged_decode_attention(*args[:-1], page_size=args[-1])
        torch.cuda.synchronize()
        exp = _oracle(*args)
        ok = (
            got.shape == exp.shape
            and got.dtype == torch.float32
            and torch.allclose(got, exp, rtol=RTOL, atol=ATOL)
        )
        h = _fnv_update(h, struct.pack("<4qB", *got.shape, got.numel(), 1 if ok else 0))
        h = _fnv_update(h, str(got.dtype).encode())
        h = _fnv_update(h, args[4].cpu().numpy().tobytes())

    for i, args in enumerate(cases):
        q, k_cache, _, page_table, cache_lens, _, window, page_size = args
        B, Hq, D = q.shape
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

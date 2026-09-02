# file: test_triton_paged_decode_attention.py

import inspect
import sys

import torch

import reference
import solution


SEED = 2026072601


def _assert_solution_uses_triton():
    src = inspect.getsource(solution)

    if "@triton.jit" not in src:
        raise AssertionError("solution.py must contain at least one @triton.jit kernel")

    banned = [
        "scaled_dot_product_attention",
        "torch.matmul",
        "torch.mm",
        "torch.bmm",
        "einsum",
        "flash_attn",
    ]

    for bad in banned:
        if bad in src:
            raise AssertionError(f"solution.py uses banned high-level op/string: {bad}")


def _rand_half(shape, device, generator, dist):
    if dist == "uniform":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.35, 0.35, generator=generator
        )
    elif dist == "peaked":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.12, 0.12, generator=generator
        )
        flat = x.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            flat[(idx % 251) == 0] = 0.75
            flat[(idx % 313) == 0] = -0.75
    elif dist == "sparse":
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
        raise ValueError(f"unknown dist: {dist}")

    return x.to(torch.float16).contiguous()


def _make_case(
    *,
    B,
    Hq,
    Hkv,
    max_len,
    D,
    page_size,
    window,
    dist,
    ragged,
    zero_len,
    seed_offset,
):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(SEED + seed_offset)

    max_pages = (max_len + page_size - 1) // page_size
    capacity = max_pages * page_size
    num_phys_pages = B * max_pages + 17

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

    if ragged:
        low = 0 if zero_len else 1
        cache_lens = torch.randint(
            low=low,
            high=max_len + 1,
            size=(B,),
            device=device,
            dtype=torch.int32,
            generator=g,
        )
        if B > 0:
            cache_lens[0] = 0 if zero_len else max_len
        if B > 1:
            cache_lens[1] = min(max_len, max(1, int(window) if int(window) > 0 else max_len))
        if B > 2:
            cache_lens[2] = max_len
        if B > 3:
            cache_lens[3] = max(0, max_len - 1)
    else:
        cache_lens = torch.full((B,), max_len, device=device, dtype=torch.int32)

    # Add a few unused/hole pages beyond valid length to ensure the implementation
    # uses cache_lens and does not scan every table entry as live.
    if max_pages >= 3 and B >= 2:
        page_table[1, max_pages - 1] = -1

    scale = 1.0 / (D ** 0.5)

    return q, k_cache, v_cache, page_table, cache_lens, scale, int(window), int(page_size), int(capacity)


def _run_case(name, cfg):
    q, k_cache, v_cache, page_table, cache_lens, scale, window, page_size, _ = _make_case(**cfg)

    q0 = q.clone()
    k0 = k_cache.clone()
    v0 = v_cache.clone()
    pt0 = page_table.clone()
    lens0 = cache_lens.clone()

    got = solution.paged_decode_attention(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        scale,
        window,
        page_size=page_size,
    )
    torch.cuda.synchronize()

    exp = reference.paged_decode_attention_ref(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        scale,
        window,
        page_size=page_size,
    )
    torch.cuda.synchronize()

    if got.shape != exp.shape:
        raise AssertionError(f"{name}: shape mismatch got={got.shape}, expected={exp.shape}")
    if got.dtype != torch.float32:
        raise AssertionError(f"{name}: output dtype must be float32, got={got.dtype}")

    if not torch.equal(q, q0):
        raise AssertionError(f"{name}: q input modified")
    if not torch.equal(k_cache, k0):
        raise AssertionError(f"{name}: k_cache input modified")
    if not torch.equal(v_cache, v0):
        raise AssertionError(f"{name}: v_cache input modified")
    if not torch.equal(page_table, pt0):
        raise AssertionError(f"{name}: page_table input modified")
    if not torch.equal(cache_lens, lens0):
        raise AssertionError(f"{name}: cache_lens input modified")

    ok = torch.allclose(got, exp, atol=reference.ATOL, rtol=reference.RTOL)

    if not ok:
        diff = (got - exp).abs()
        denom = exp.abs().clamp_min(1.0e-8)
        rel = diff / denom
        flat_idx = int(diff.argmax().item())
        raise AssertionError(
            f"{name}: output mismatch max_abs={diff.max().item():.6g} "
            f"max_rel={rel.max().item():.6g} flat_idx={flat_idx} "
            f"tolerance atol={reference.ATOL} rtol={reference.RTOL}"
        )


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    _assert_solution_uses_triton()

    cases = [
        (
            "decode_B1_Hq4_Hkv1_L128_D64_P16_full",
            dict(
                B=1,
                Hq=4,
                Hkv=1,
                max_len=128,
                D=64,
                page_size=16,
                window=0,
                dist="uniform",
                ragged=False,
                zero_len=False,
                seed_offset=1,
            ),
        ),
        (
            "gqa_B8_Hq8_Hkv2_L512_D64_P16_W128",
            dict(
                B=8,
                Hq=8,
                Hkv=2,
                max_len=512,
                D=64,
                page_size=16,
                window=128,
                dist="peaked",
                ragged=True,
                zero_len=False,
                seed_offset=2,
            ),
        ),
        (
            "gqa_B4_Hq16_Hkv4_L2048_D128_P32_W256",
            dict(
                B=4,
                Hq=16,
                Hkv=4,
                max_len=2048,
                D=128,
                page_size=32,
                window=256,
                dist="sparse",
                ragged=True,
                zero_len=False,
                seed_offset=3,
            ),
        ),
        (
            "long_B2_Hq8_Hkv1_L4096_D64_P32_full",
            dict(
                B=2,
                Hq=8,
                Hkv=1,
                max_len=4096,
                D=64,
                page_size=32,
                window=0,
                dist="ties",
                ragged=True,
                zero_len=False,
                seed_offset=4,
            ),
        ),
        (
            "nontile_L257_B8_Hq8_Hkv2_D64_P16_W33",
            dict(
                B=8,
                Hq=8,
                Hkv=2,
                max_len=257,
                D=64,
                page_size=16,
                window=33,
                dist="ties",
                ragged=True,
                zero_len=False,
                seed_offset=5,
            ),
        ),
        (
            "batch_decode_B32_Hq4_Hkv1_L128_D64_P8_W1",
            dict(
                B=32,
                Hq=4,
                Hkv=1,
                max_len=128,
                D=64,
                page_size=8,
                window=1,
                dist="uniform",
                ragged=True,
                zero_len=False,
                seed_offset=6,
            ),
        ),
        (
            "zero_len_edge_B8_Hq8_Hkv2_L128_D64",
            dict(
                B=8,
                Hq=8,
                Hkv=2,
                max_len=128,
                D=64,
                page_size=16,
                window=64,
                dist="uniform",
                ragged=True,
                zero_len=True,
                seed_offset=7,
            ),
        ),
    ]

    passed = 0

    for name, cfg in cases:
        try:
            _run_case(name, cfg)
            passed += 1
            print(f"case {name:48s} PASS")
        except Exception as exc:
            print(f"case {name:48s} FAIL  {exc}")

    print(f"passed {passed} / {len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

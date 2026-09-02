# file: test_triton_flash_decode.py

import inspect
import sys

import torch

import reference
import solution


SEED = 2026071801

ATOL = 2.0e-2
RTOL = 2.0e-2


def _assert_solution_uses_triton():
    src = inspect.getsource(solution)

    if "@triton.jit" not in src:
        raise AssertionError("solution.py must define at least one @triton.jit kernel")

    banned = [
        "scaled_dot_product_attention",
        "torch.matmul",
        "torch.mm",
        "torch.bmm",
        "einsum",
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
            -0.15, 0.15, generator=generator
        )
        flat = x.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            mask_pos = (idx % 97) == 0
            mask_neg = (idx % 131) == 0
            flat[mask_pos] = 0.75
            flat[mask_neg] = -0.75
    elif dist == "long_context":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.20, 0.20, generator=generator
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.65
        x = torch.where(keep, x, torch.zeros_like(x))
    elif dist == "ties":
        vals = torch.tensor(
            [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(
            0,
            vals.numel(),
            shape,
            device=device,
            generator=generator,
        )
        x = vals[ids]
    else:
        raise ValueError(f"unknown dist {dist}")

    return x.to(torch.float16)


def _make_case(
    *,
    B,
    Hq,
    Hkv,
    S,
    D,
    window,
    dist,
    allow_zero_len,
    seed_offset,
):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed((SEED + seed_offset) + 880123)

    q = _rand_half((B, Hq, D), device, g, dist)
    k = _rand_half((B, Hkv, S, D), device, g, dist)
    v = _rand_half((B, Hkv, S, D), device, g, dist)

    if allow_zero_len:
        low = 0
    else:
        low = 1

    cache_lens = torch.randint(
        low=low,
        high=S + 1,
        size=(B,),
        device=device,
        dtype=torch.int32,
        generator=g,
    )

    if B > 0:
        cache_lens[0] = 0 if allow_zero_len else S
    if B > 1:
        cache_lens[1] = min(S, max(1, int(window) if int(window) > 0 else S))
    if B > 2:
        cache_lens[2] = S

    scale = 1.0 / (D ** 0.5)

    return q, k, v, cache_lens, scale, int(window)


def _run_case(name, cfg):
    q, k, v, cache_lens, scale, window = _make_case(**cfg)

    q0 = q.clone()
    k0 = k.clone()
    v0 = v.clone()
    lens0 = cache_lens.clone()

    got = solution.flash_decode(q, k, v, cache_lens, scale, window)
    torch.cuda.synchronize()

    exp = reference.flash_decode_ref(q, k, v, cache_lens, scale, window)
    torch.cuda.synchronize()

    if got.shape != exp.shape:
        raise AssertionError(f"{name}: shape mismatch got {got.shape}, expected {exp.shape}")

    if got.dtype != torch.float32:
        raise AssertionError(f"{name}: output dtype must be torch.float32, got {got.dtype}")

    if not torch.equal(q, q0):
        raise AssertionError(f"{name}: q input modified")
    if not torch.equal(k, k0):
        raise AssertionError(f"{name}: k input modified")
    if not torch.equal(v, v0):
        raise AssertionError(f"{name}: v input modified")
    if not torch.equal(cache_lens, lens0):
        raise AssertionError(f"{name}: cache_lens input modified")

    ok = torch.allclose(got, exp, rtol=RTOL, atol=ATOL)

    if not ok:
        diff = (got - exp).abs()
        denom = exp.abs().clamp_min(1.0e-8)
        rel = diff / denom
        max_abs = diff.max().item()
        max_rel = rel.max().item()
        idx = diff.argmax().item()
        raise AssertionError(
            f"{name}: output mismatch max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
            f"flat_idx={idx} tolerance atol={ATOL} rtol={RTOL}"
        )


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    _assert_solution_uses_triton()

    cases = [
        (
            "decode_B1_Hq4_Hkv1_S128_D64_full",
            dict(
                B=1,
                Hq=4,
                Hkv=1,
                S=128,
                D=64,
                window=0,
                dist="uniform",
                allow_zero_len=False,
                seed_offset=1,
            ),
        ),
        (
            "gqa_B8_Hq8_Hkv2_S512_D64_W128",
            dict(
                B=8,
                Hq=8,
                Hkv=2,
                S=512,
                D=64,
                window=128,
                dist="peaked",
                allow_zero_len=False,
                seed_offset=2,
            ),
        ),
        (
            "gqa_B4_Hq16_Hkv4_S2048_D128_W256",
            dict(
                B=4,
                Hq=16,
                Hkv=4,
                S=2048,
                D=128,
                window=256,
                dist="long_context",
                allow_zero_len=False,
                seed_offset=3,
            ),
        ),
        (
            "long_B2_Hq8_Hkv1_S4096_D64_full",
            dict(
                B=2,
                Hq=8,
                Hkv=1,
                S=4096,
                D=64,
                window=0,
                dist="ties",
                allow_zero_len=False,
                seed_offset=4,
            ),
        ),
        (
            "nontile_B8_Hq8_Hkv2_S257_D64_W33",
            dict(
                B=8,
                Hq=8,
                Hkv=2,
                S=257,
                D=64,
                window=33,
                dist="ties",
                allow_zero_len=False,
                seed_offset=5,
            ),
        ),
        (
            "batch64_decode_B64_Hq4_Hkv1_S128_D64_W1",
            dict(
                B=64,
                Hq=4,
                Hkv=1,
                S=128,
                D=64,
                window=1,
                dist="uniform",
                allow_zero_len=False,
                seed_offset=6,
            ),
        ),
        (
            "zero_len_edge_B8_Hq8_Hkv2_S128_D64",
            dict(
                B=8,
                Hq=8,
                Hkv=2,
                S=128,
                D=64,
                window=64,
                dist="uniform",
                allow_zero_len=True,
                seed_offset=7,
            ),
        ),
    ]

    passed = 0

    for name, cfg in cases:
        try:
            _run_case(name, cfg)
            passed += 1
            print(f"case {name:45s} PASS")
        except Exception as exc:
            print(f"case {name:45s} FAIL  {exc}")

    print(f"passed {passed} / {len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

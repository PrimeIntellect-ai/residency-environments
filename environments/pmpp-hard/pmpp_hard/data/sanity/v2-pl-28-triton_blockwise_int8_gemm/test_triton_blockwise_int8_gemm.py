# file: test_triton_blockwise_int8_gemm.py

import inspect
import sys

import torch

import reference
import solution


SEED = 2026072801


def _assert_solution_uses_triton():
    src = inspect.getsource(solution)

    if "@triton.jit" not in src:
        raise AssertionError("solution.py must contain at least one @triton.jit kernel")

    banned = [
        "torch._int_mm",
        "torch.matmul",
        "torch.mm",
        "torch.bmm",
        "einsum",
    ]

    for bad in banned:
        if bad in src:
            raise AssertionError(f"solution.py uses banned high-level op/string: {bad}")


def _rand_int8(shape, device, generator, dist):
    if dist == "sparse":
        x16 = torch.randint(
            -127,
            128,
            shape,
            device=device,
            dtype=torch.int16,
            generator=generator,
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.70
        x16 = torch.where(keep, x16, torch.zeros_like(x16))
    elif dist == "ties":
        vals = torch.tensor(
            [-127, -64, -16, -1, 0, 1, 16, 64, 127],
            device=device,
            dtype=torch.int16,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x16 = vals[ids]
    elif dist == "outlier_codes":
        x16 = torch.randint(
            -8,
            9,
            shape,
            device=device,
            dtype=torch.int16,
            generator=generator,
        )
        flat = x16.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            flat[(idx % 257) == 0] = 127
            flat[(idx % 389) == 0] = -127
    else:
        x16 = torch.randint(
            -127,
            128,
            shape,
            device=device,
            dtype=torch.int16,
            generator=generator,
        )

    return x16.to(torch.int8).contiguous()


def _make_scales(shape, device, generator, dist):
    if dist == "uniform":
        s = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            0.0025, 0.0500, generator=generator
        )
    elif dist == "outliers":
        s = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            0.0010, 0.0200, generator=generator
        )
        flat = s.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            flat[(idx % 19) == 0] = 0.1500
            flat[(idx % 23) == 0] = 0.0005
    elif dist == "ties":
        vals = torch.tensor(
            [0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        s = vals[ids]
    elif dist == "rowcol_skew":
        base = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            0.0020, 0.0200, generator=generator
        )
        if len(shape) == 2:
            r = torch.arange(shape[0], device=device, dtype=torch.float32)[:, None]
            c = torch.arange(shape[1], device=device, dtype=torch.float32)[None, :]
            s = base * (1.0 + (r % 7.0) * 0.25 + (c % 5.0) * 0.10)
        else:
            s = base
    else:
        raise ValueError(f"unknown scale distribution: {dist}")

    return s.contiguous()


def _make_case(
    name,
    M,
    N,
    K,
    block_m,
    block_n,
    block_k,
    code_dist,
    scale_dist,
    seed_offset,
):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed((SEED + seed_offset) + 880123)

    a = _rand_int8((M, K), device, g, code_dist)
    b = _rand_int8((K, N), device, g, code_dist)

    num_m_blocks = (M + block_m - 1) // block_m
    num_n_blocks = (N + block_n - 1) // block_n
    num_k_blocks = (K + block_k - 1) // block_k

    a_scales = _make_scales((num_m_blocks, num_k_blocks), device, g, scale_dist)
    b_scales = _make_scales((num_k_blocks, num_n_blocks), device, g, scale_dist)

    return {
        "name": name,
        "a": a,
        "b": b,
        "a_scales": a_scales,
        "b_scales": b_scales,
        "block_m": int(block_m),
        "block_n": int(block_n),
        "block_k": int(block_k),
        "code_dist": code_dist,
        "scale_dist": scale_dist,
    }


def _run_case(case):
    a = case["a"]
    b = case["b"]
    a_scales = case["a_scales"]
    b_scales = case["b_scales"]

    a0 = a.clone()
    b0 = b.clone()
    as0 = a_scales.clone()
    bs0 = b_scales.clone()

    got = solution.blockwise_int8_gemm(
        a,
        b,
        a_scales,
        b_scales,
        block_m=case["block_m"],
        block_n=case["block_n"],
        block_k=case["block_k"],
    )
    torch.cuda.synchronize()

    exp = reference.blockwise_int8_gemm_ref(
        a,
        b,
        a_scales,
        b_scales,
        block_m=case["block_m"],
        block_n=case["block_n"],
        block_k=case["block_k"],
    )
    torch.cuda.synchronize()

    if got.shape != exp.shape:
        raise AssertionError(f"{case['name']}: shape mismatch got={got.shape}, expected={exp.shape}")
    if got.dtype != torch.float32:
        raise AssertionError(f"{case['name']}: output dtype must be float32, got={got.dtype}")

    if not torch.equal(a, a0):
        raise AssertionError(f"{case['name']}: input a modified")
    if not torch.equal(b, b0):
        raise AssertionError(f"{case['name']}: input b modified")
    if not torch.equal(a_scales, as0):
        raise AssertionError(f"{case['name']}: input a_scales modified")
    if not torch.equal(b_scales, bs0):
        raise AssertionError(f"{case['name']}: input b_scales modified")

    ok = torch.allclose(got, exp, atol=reference.ATOL, rtol=reference.RTOL)

    if not ok:
        diff = (got - exp).abs()
        denom = exp.abs().clamp_min(1.0e-8)
        rel = diff / denom
        flat_idx = int(diff.argmax().item())
        raise AssertionError(
            f"{case['name']}: output mismatch "
            f"max_abs={diff.max().item():.6g} "
            f"max_rel={rel.max().item():.6g} "
            f"flat_idx={flat_idx} "
            f"tolerance atol={reference.ATOL} rtol={reference.RTOL}"
        )


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    torch.backends.cuda.matmul.allow_tf32 = False
    _assert_solution_uses_triton()

    cases = [
        _make_case(
            "small_256_256_256_b32_uniform",
            256, 256, 256,
            32, 32, 32,
            "uniform",
            "uniform",
            1,
        ),
        _make_case(
            "outlier_512_384_640_b64",
            512, 384, 640,
            64, 64, 64,
            "outlier_codes",
            "outliers",
            2,
        ),
        _make_case(
            "ties_1024_square_b128",
            1024, 1024, 1024,
            128, 128, 128,
            "ties",
            "ties",
            3,
        ),
        _make_case(
            "nontile_768_1152_960_m64n128k32",
            768, 1152, 960,
            64, 128, 32,
            "sparse",
            "rowcol_skew",
            4,
        ),
        _make_case(
            "largeM_2048_1024_512",
            2048, 1024, 512,
            128, 64, 64,
            "uniform",
            "uniform",
            5,
        ),
        _make_case(
            "wideN_256_4096_256",
            256, 4096, 256,
            32, 128, 64,
            "outlier_codes",
            "outliers",
            6,
        ),
        _make_case(
            "mixed_1536_640_768_b32n32k128",
            1536, 640, 768,
            32, 32, 128,
            "ties",
            "rowcol_skew",
            7,
        ),
    ]

    passed = 0

    for case in cases:
        try:
            _run_case(case)
            passed += 1
            M, K = case["a"].shape
            N = case["b"].shape[1]
            print(
                f"case {case['name']:38s} PASS  "
                f"M={M} N={N} K={K} "
                f"blocks=({case['block_m']},{case['block_n']},{case['block_k']}) "
                f"codes={case['code_dist']} scales={case['scale_dist']}"
            )
        except Exception as exc:
            print(f"case {case['name']:38s} FAIL  {exc}")

    print(f"passed {passed} / {len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

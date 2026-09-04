# file: test_triton_persistent_grouped_gemm.py

import inspect
import sys

import torch

import reference
import solution


SEED = 2026071901


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
        "torch.nn.functional.gelu",
    ]

    for bad in banned:
        if bad in src:
            raise AssertionError(f"solution.py uses banned high-level op/string: {bad}")


def _float_to_half_tensor(x):
    return x.to(torch.float16).contiguous()


def _rand_values(shape, device, generator, dist):
    if dist == "uniform":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.25, 0.25, generator=generator
        )
    elif dist == "powerlaw":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.20, 0.20, generator=generator
        )
        mask = torch.rand(shape, device=device, generator=generator) > 0.55
        x = torch.where(mask, x, torch.zeros_like(x))
    elif dist == "tiny":
        vals = torch.tensor(
            [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x = vals[ids]
    elif dist == "ties":
        vals = torch.tensor(
            [-0.1875, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x = vals[ids]
    else:
        raise ValueError(f"unknown dist {dist}")

    return x


def _make_case(name, dims, activation, dist, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(SEED + seed_offset)

    group_count = len(dims)

    group_m_host = []
    group_n_host = []
    group_k_host = []

    a_offsets_host = []
    b_offsets_host = []
    bias_offsets_host = []
    c_offsets_host = []

    total_a = 0
    total_b = 0
    total_bias = 0
    total_c = 0

    for M, N, K in dims:
        group_m_host.append(M)
        group_n_host.append(N)
        group_k_host.append(K)

        a_offsets_host.append(total_a)
        b_offsets_host.append(total_b)
        bias_offsets_host.append(total_bias)
        c_offsets_host.append(total_c)

        total_a += M * K
        total_b += K * N
        total_bias += N
        total_c += M * N

    a_f = _rand_values((total_a,), device, g, dist)
    b_f = _rand_values((total_b,), device, g, dist)
    bias = torch.empty((total_bias,), device=device, dtype=torch.float32).uniform_(
        -0.20, 0.20, generator=g
    )

    a = _float_to_half_tensor(a_f)
    b = _float_to_half_tensor(b_f)

    group_m = torch.tensor(group_m_host, device=device, dtype=torch.int32)
    group_n = torch.tensor(group_n_host, device=device, dtype=torch.int32)
    group_k = torch.tensor(group_k_host, device=device, dtype=torch.int32)

    a_offsets = torch.tensor(a_offsets_host, device=device, dtype=torch.int64)
    b_offsets = torch.tensor(b_offsets_host, device=device, dtype=torch.int64)
    bias_offsets = torch.tensor(bias_offsets_host, device=device, dtype=torch.int64)
    c_offsets = torch.tensor(c_offsets_host, device=device, dtype=torch.int64)

    return {
        "name": name,
        "a": a,
        "b": b,
        "bias": bias,
        "group_m": group_m,
        "group_n": group_n,
        "group_k": group_k,
        "a_offsets": a_offsets,
        "b_offsets": b_offsets,
        "bias_offsets": bias_offsets,
        "c_offsets": c_offsets,
        "activation": activation,
        "total_c": total_c,
        "group_count": group_count,
        "dist": dist,
    }


def _powerlaw_dims():
    dims = []
    for g in range(32):
        if g == 0:
            M = 256
        elif g == 1:
            M = 128
        elif g < 4:
            M = 64
        elif g < 12:
            M = 16
        else:
            M = 4

        N = 128 if g % 3 == 0 else 64
        K = 128 if g % 5 == 0 else 64
        dims.append((M, N, K))
    return dims


def _many_tiny_dims():
    dims = []
    for g in range(256):
        if g == 0:
            M = 64
        elif g < 8:
            M = 16
        else:
            M = 1 + (g % 4)
        dims.append((M, 64, 64))
    return dims


def _zero_group_dims():
    return [
        (64, 64, 64),
        (0, 64, 64),
        (16, 128, 96),
        (8, 65, 72),
    ]


def _run_case(case):
    args = (
        case["a"],
        case["b"],
        case["bias"],
        case["group_m"],
        case["group_n"],
        case["group_k"],
        case["a_offsets"],
        case["b_offsets"],
        case["bias_offsets"],
        case["c_offsets"],
    )

    a0 = case["a"].clone()
    b0 = case["b"].clone()
    bias0 = case["bias"].clone()
    gm0 = case["group_m"].clone()
    gn0 = case["group_n"].clone()
    gk0 = case["group_k"].clone()
    ao0 = case["a_offsets"].clone()
    bo0 = case["b_offsets"].clone()
    bio0 = case["bias_offsets"].clone()
    co0 = case["c_offsets"].clone()

    got = solution.grouped_gemm_bias_act(*args, activation=case["activation"])
    torch.cuda.synchronize()

    exp = reference.grouped_gemm_bias_act_ref(*args, activation=case["activation"])
    torch.cuda.synchronize()

    if got.shape != exp.shape:
        raise AssertionError(f"{case['name']}: shape mismatch got={got.shape} expected={exp.shape}")
    if got.dtype != torch.float32:
        raise AssertionError(f"{case['name']}: output dtype must be float32, got={got.dtype}")

    if not torch.equal(case["a"], a0):
        raise AssertionError(f"{case['name']}: input a modified")
    if not torch.equal(case["b"], b0):
        raise AssertionError(f"{case['name']}: input b modified")
    if not torch.equal(case["bias"], bias0):
        raise AssertionError(f"{case['name']}: input bias modified")
    if not torch.equal(case["group_m"], gm0):
        raise AssertionError(f"{case['name']}: input group_m modified")
    if not torch.equal(case["group_n"], gn0):
        raise AssertionError(f"{case['name']}: input group_n modified")
    if not torch.equal(case["group_k"], gk0):
        raise AssertionError(f"{case['name']}: input group_k modified")
    if not torch.equal(case["a_offsets"], ao0):
        raise AssertionError(f"{case['name']}: input a_offsets modified")
    if not torch.equal(case["b_offsets"], bo0):
        raise AssertionError(f"{case['name']}: input b_offsets modified")
    if not torch.equal(case["bias_offsets"], bio0):
        raise AssertionError(f"{case['name']}: input bias_offsets modified")
    if not torch.equal(case["c_offsets"], co0):
        raise AssertionError(f"{case['name']}: input c_offsets modified")

    ok = torch.allclose(got, exp, atol=reference.ATOL, rtol=reference.RTOL)
    if not ok:
        diff = (got - exp).abs()
        denom = exp.abs().clamp_min(1.0e-8)
        rel = diff / denom
        idx = int(diff.argmax().item())
        raise AssertionError(
            f"{case['name']}: mismatch max_abs={diff.max().item():.6g} "
            f"max_rel={rel.max().item():.6g} flat_idx={idx} "
            f"tolerance atol={reference.ATOL} rtol={reference.RTOL}"
        )


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    _assert_solution_uses_triton()

    cases = [
        _make_case(
            "single_group_relu",
            [(64, 64, 64)],
            "relu",
            "uniform",
            1,
        ),
        _make_case(
            "g4_zero_group_nontile_gelu",
            _zero_group_dims(),
            "gelu",
            "uniform",
            2,
        ),
        _make_case(
            "g32_powerlaw_relu",
            _powerlaw_dims(),
            "relu",
            "powerlaw",
            3,
        ),
        _make_case(
            "g256_many_tiny_gelu",
            _many_tiny_dims(),
            "gelu",
            "tiny",
            4,
        ),
        _make_case(
            "g4_large_nk_ties",
            [(8, 512, 64), (4, 1024, 128), (2, 256, 512), (1, 2048, 96)],
            "gelu",
            "ties",
            5,
        ),
        _make_case(
            "g32_mixed_k_not_tile",
            [(16 if g < 4 else 4, 65 if g % 2 else 96, 72 if g % 3 else 80) for g in range(32)],
            "relu",
            "ties",
            6,
        ),
    ]

    passed = 0

    for case in cases:
        try:
            _run_case(case)
            passed += 1
            print(
                f"case {case['name']:32s} PASS  "
                f"G={case['group_count']} act={case['activation']} total_c={case['total_c']}"
            )
        except Exception as exc:
            print(f"case {case['name']:32s} FAIL  {exc}")

    print(f"passed {passed} / {len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

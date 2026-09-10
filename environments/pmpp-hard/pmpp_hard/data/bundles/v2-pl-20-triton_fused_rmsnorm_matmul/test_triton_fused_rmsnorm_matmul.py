# file: test_triton_fused_rmsnorm_matmul.py

import inspect
import sys

import torch

import reference
import solution


SEED = 2026072001


def _assert_solution_uses_triton():
    src = inspect.getsource(solution)

    if "@triton.jit" not in src:
        raise AssertionError("solution.py must contain at least one @triton.jit kernel")

    banned = [
        "torch.nn.functional",
        "torch.matmul",
        "torch.mm",
        "torch.bmm",
        "einsum",
        "rms_norm",
        "layer_norm",
        "linear(",
    ]

    for bad in banned:
        if bad in src:
            raise AssertionError(f"solution.py uses banned high-level op/string: {bad}")


def _rand_half(shape, device, generator, dist):
    if dist == "uniform":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.35, 0.35, generator=generator
        )
    elif dist == "zeroish":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.30, 0.30, generator=generator
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.60
        x = torch.where(keep, x, torch.zeros_like(x))
    elif dist == "ties":
        vals = torch.tensor(
            [-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x = vals[ids]
    elif dist == "peaked":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.15, 0.15, generator=generator
        )
        flat = x.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            flat[(idx % 251) == 0] = 0.75
            flat[(idx % 313) == 0] = -0.75
    else:
        raise ValueError(f"unknown distribution: {dist}")

    return x.to(torch.float16).contiguous()


def _rand_weight(K, device, generator, dist):
    if dist == "ties":
        vals = torch.tensor(
            [-1.25, -0.75, -0.25, 0.0, 0.25, 0.75, 1.25],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), (K,), device=device, generator=generator)
        return vals[ids].contiguous()

    w = torch.empty((K,), device=device, dtype=torch.float32).uniform_(
        0.25, 1.75, generator=generator
    )

    if dist == "zeroish":
        idx = torch.arange(K, device=device)
        w = torch.where((idx % 17) == 0, torch.zeros_like(w), w)

    if dist == "peaked":
        idx = torch.arange(K, device=device)
        w = torch.where((idx % 19) == 0, -w, w)

    return w.contiguous()


def _rand_bias(N, device, generator, dist):
    if dist == "ties":
        vals = torch.tensor(
            [-0.25, -0.125, 0.0, 0.125, 0.25],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), (N,), device=device, generator=generator)
        return vals[ids].contiguous()

    return torch.empty((N,), device=device, dtype=torch.float32).uniform_(
        -0.50, 0.50, generator=generator
    ).contiguous()


def _make_case(name, M, K, N, eps, activation, dist, seed_offset, with_bias=True):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(SEED + seed_offset)

    x = _rand_half((M, K), device, g, dist)
    weight = _rand_weight(K, device, g, dist)
    w = _rand_half((K, N), device, g, dist)
    bias = _rand_bias(N, device, g, dist) if with_bias else None

    return {
        "name": name,
        "x": x,
        "weight": weight,
        "w": w,
        "bias": bias,
        "eps": float(eps),
        "activation": activation,
        "dist": dist,
    }


def _run_case(case):
    x = case["x"]
    weight = case["weight"]
    w = case["w"]
    bias = case["bias"]

    x0 = x.clone()
    weight0 = weight.clone()
    w0 = w.clone()
    bias0 = bias.clone() if bias is not None else None

    got = solution.fused_rmsnorm_matmul(
        x,
        weight,
        w,
        bias=bias,
        eps=case["eps"],
        activation=case["activation"],
    )
    torch.cuda.synchronize()

    exp = reference.fused_rmsnorm_matmul_ref(
        x,
        weight,
        w,
        bias=bias,
        eps=case["eps"],
        activation=case["activation"],
    )
    torch.cuda.synchronize()

    if got.shape != exp.shape:
        raise AssertionError(f"{case['name']}: shape mismatch got={got.shape}, expected={exp.shape}")
    if got.dtype != torch.float32:
        raise AssertionError(f"{case['name']}: output dtype must be float32, got={got.dtype}")

    if not torch.equal(x, x0):
        raise AssertionError(f"{case['name']}: input x modified")
    if not torch.equal(weight, weight0):
        raise AssertionError(f"{case['name']}: input weight modified")
    if not torch.equal(w, w0):
        raise AssertionError(f"{case['name']}: input w modified")
    if bias is not None and not torch.equal(bias, bias0):
        raise AssertionError(f"{case['name']}: input bias modified")

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
            "small_256x256x256_none",
            256, 256, 256,
            1.0e-5,
            "none",
            "uniform",
            1,
        ),
        _make_case(
            "relu_512x1024x512_zeroish",
            512, 1024, 512,
            1.0e-6,
            "relu",
            "zeroish",
            2,
        ),
        _make_case(
            "gelu_1024x512x1024_ties",
            1024, 512, 1024,
            1.0e-5,
            "gelu",
            "ties",
            3,
        ),
        _make_case(
            "largeK_256x4096x256_peaked",
            256, 4096, 256,
            1.0e-4,
            "none",
            "peaked",
            4,
        ),
        _make_case(
            "largeM_2048x256x512_relu",
            2048, 256, 512,
            1.0e-5,
            "relu",
            "uniform",
            5,
        ),
        _make_case(
            "no_bias_512x768x384_gelu",
            512, 768, 384,
            1.0e-3,
            "gelu_tanh",
            "zeroish",
            6,
            with_bias=False,
        ),
    ]

    passed = 0

    for case in cases:
        try:
            _run_case(case)
            passed += 1
            M, K = case["x"].shape
            N = case["w"].shape[1]
            print(
                f"case {case['name']:34s} PASS  "
                f"M={M} K={K} N={N} act={case['activation']} dist={case['dist']}"
            )
        except Exception as exc:
            print(f"case {case['name']:34s} FAIL  {exc}")

    print(f"passed {passed} / {len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

# file: test_triton_fused_layernorm_quant.py

import inspect
import sys

import torch

import reference
import solution


SEED = 2026080101


def _assert_solution_uses_triton():
    src = inspect.getsource(solution)

    if "@triton.jit" not in src:
        raise AssertionError("solution.py must contain at least one @triton.jit kernel")

    banned = [
        "torch.nn.functional.layer_norm",
        "layer_norm(",
        "torch.var_mean",
        "torch.std_mean",
        "torch.quantize",
    ]

    for bad in banned:
        if bad in src:
            raise AssertionError(f"solution.py uses banned high-level op/string: {bad}")


def _rand_x(shape, device, generator, dist):
    N, D = shape

    if dist == "uniform":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -2.0, 2.0, generator=generator
        )
    elif dist == "outliers":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -1.5, 1.5, generator=generator
        )
        rows = torch.arange(N, device=device)
        cols = (rows * 17) % D
        x[rows[::97], cols[::97]] = 512.0
        x[rows[::131], cols[::131]] = -512.0
    elif dist == "all_equal":
        base = torch.empty((N, 1), device=device, dtype=torch.float32).uniform_(
            -4.0, 4.0, generator=generator
        )
        x = base.expand(N, D).clone()
    elif dist == "near_zero_var":
        base = torch.empty((N, 1), device=device, dtype=torch.float32).uniform_(
            -0.5, 0.5, generator=generator
        )
        noise = torch.zeros((N, D), device=device, dtype=torch.float32)
        noise[:, ::64] = 1.0 / 1024.0
        noise[:, ::97] = -1.0 / 1024.0
        x = base + noise
    elif dist == "ties":
        vals = torch.tensor(
            [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x = vals[ids]
    elif dist == "zeroish":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -1.0, 1.0, generator=generator
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.65
        x = torch.where(keep, x, torch.zeros_like(x))
    else:
        raise ValueError(f"unknown x distribution {dist}")

    return x.contiguous()


def _rand_weight_bias(D, device, generator, dist):
    if dist == "all_equal":
        weight_vals = torch.tensor([0.5, 1.0, 1.5, 2.0], device=device, dtype=torch.float32)
        bias_vals = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=device, dtype=torch.float32)
        idx_w = torch.arange(D, device=device) % weight_vals.numel()
        idx_b = torch.arange(D, device=device) % bias_vals.numel()
        weight = weight_vals[idx_w].contiguous()
        bias = bias_vals[idx_b].contiguous()
    elif dist == "outliers":
        weight = torch.empty((D,), device=device, dtype=torch.float32).uniform_(
            -2.0, 2.0, generator=generator
        )
        bias = torch.empty((D,), device=device, dtype=torch.float32).uniform_(
            -0.75, 0.75, generator=generator
        )
        weight[::257] = -3.0
        weight[::149] = 3.0
        bias[::509] = 8.0
        bias[::331] = -8.0
    elif dist == "zeroish":
        weight = torch.empty((D,), device=device, dtype=torch.float32).uniform_(
            -1.0, 1.0, generator=generator
        )
        bias = torch.empty((D,), device=device, dtype=torch.float32).uniform_(
            -0.03125, 0.03125, generator=generator
        )
        weight[::17] = 0.0
    elif dist == "ties":
        w_vals = torch.tensor(
            [-1.5, -1.0, -0.5, 0.25, 0.5, 1.0, 1.5, 2.0],
            device=device,
            dtype=torch.float32,
        )
        b_vals = torch.tensor(
            [-0.75, -0.25, -0.125, 0.0, 0.125, 0.25, 0.75],
            device=device,
            dtype=torch.float32,
        )
        weight = w_vals[(torch.arange(D, device=device) * 13 + 5) % w_vals.numel()].contiguous()
        bias = b_vals[(torch.arange(D, device=device) * 7 + 3) % b_vals.numel()].contiguous()
    else:
        weight = torch.empty((D,), device=device, dtype=torch.float32).uniform_(
            -1.5, 1.5, generator=generator
        )
        bias = torch.empty((D,), device=device, dtype=torch.float32).uniform_(
            -0.5, 0.5, generator=generator
        )

    return weight.contiguous(), bias.contiguous()


def _make_case(name, N, D, eps, dist, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(SEED + seed_offset)

    x = _rand_x((N, D), device, g, dist)
    weight, bias = _rand_weight_bias(D, device, g, dist)

    return {
        "name": name,
        "x": x,
        "weight": weight,
        "bias": bias,
        "eps": float(eps),
        "dist": dist,
    }


def _check_q_codes(name, got_q, exp_q):
    diff = (got_q.to(torch.int16) - exp_q.to(torch.int16)).abs()

    # CUDA/Triton and torch can disagree by one code at fp32 reduction or
    # round-half-even boundaries. This task grades exact quantization structure
    # but accepts +/-1 code drift, while scale and dequant still have tolerances.
    if not bool((diff <= 1).all().item()):
        bad = (diff > 1).nonzero(as_tuple=False)[0]
        r = int(bad[0].item())
        d = int(bad[1].item())
        raise AssertionError(
            f"{name}: q_int8 mismatch row={r}, d={d}, "
            f"got={int(got_q[r, d].item())}, expected={int(exp_q[r, d].item())}, "
            f"diff={int(diff[r, d].item())}"
        )


def _run_case(case):
    x = case["x"]
    weight = case["weight"]
    bias = case["bias"]

    x0 = x.clone()
    weight0 = weight.clone()
    bias0 = bias.clone()

    got_q, got_scale, got_dequant = solution.fused_layernorm_quant(
        x,
        weight,
        bias,
        eps=case["eps"],
    )
    torch.cuda.synchronize()

    exp_q, exp_scale, exp_dequant, _ = reference.fused_layernorm_quant_ref_with_pre(
        x,
        weight,
        bias,
        eps=case["eps"],
    )
    torch.cuda.synchronize()

    if got_q.shape != exp_q.shape:
        raise AssertionError(f"{case['name']}: q shape mismatch")
    if got_scale.shape != exp_scale.shape:
        raise AssertionError(f"{case['name']}: scale shape mismatch")
    if got_dequant.shape != exp_dequant.shape:
        raise AssertionError(f"{case['name']}: dequant shape mismatch")

    if got_q.dtype != torch.int8:
        raise AssertionError(f"{case['name']}: q dtype must be int8")
    if got_scale.dtype != torch.float32:
        raise AssertionError(f"{case['name']}: scale dtype must be float32")
    if got_dequant.dtype != torch.float32:
        raise AssertionError(f"{case['name']}: dequant dtype must be float32")

    if not torch.equal(x, x0):
        raise AssertionError(f"{case['name']}: input x modified")
    if not torch.equal(weight, weight0):
        raise AssertionError(f"{case['name']}: input weight modified")
    if not torch.equal(bias, bias0):
        raise AssertionError(f"{case['name']}: input bias modified")

    if not torch.allclose(
        got_scale,
        exp_scale,
        atol=reference.SCALE_ATOL,
        rtol=reference.SCALE_RTOL,
    ):
        diff = (got_scale - exp_scale).abs()
        idx = int(diff.argmax().item())
        raise AssertionError(
            f"{case['name']}: scale mismatch row={idx}, "
            f"got={got_scale[idx].item():.8g}, expected={exp_scale[idx].item():.8g}, "
            f"diff={diff[idx].item():.8g}"
        )

    _check_q_codes(case["name"], got_q, exp_q)

    reconstructed = got_q.float() * got_scale[:, None]
    if not torch.allclose(got_dequant, reconstructed, atol=1.0e-5, rtol=1.0e-5):
        diff = (got_dequant - reconstructed).abs()
        idx = int(diff.argmax().item())
        raise AssertionError(
            f"{case['name']}: dequant is not internally q*scale, "
            f"max_diff={diff.max().item():.8g}, flat_idx={idx}"
        )

    deq_diff = (got_dequant - exp_dequant).abs()
    allowed = (
        reference.DEQUANT_ATOL
        + reference.DEQUANT_RTOL * exp_dequant.abs()
        + got_scale[:, None].abs() * 1.05
    )

    if not bool((deq_diff <= allowed).all().item()):
        bad = (deq_diff > allowed).nonzero(as_tuple=False)[0]
        r = int(bad[0].item())
        d = int(bad[1].item())
        raise AssertionError(
            f"{case['name']}: dequant mismatch row={r}, d={d}, "
            f"got={got_dequant[r, d].item():.8g}, expected={exp_dequant[r, d].item():.8g}, "
            f"diff={deq_diff[r, d].item():.8g}, allowed={allowed[r, d].item():.8g}"
        )


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    _assert_solution_uses_triton()

    cases = [
        _make_case("uniform_N1024_D256", 1024, 256, 1.0e-5, "uniform", 1),
        _make_case("outliers_N4096_D1024", 4096, 1024, 1.0e-5, "outliers", 2),
        _make_case("all_equal_N2048_D4096", 2048, 4096, 1.0e-5, "all_equal", 3),
        _make_case("near_zero_var_N8192_D256", 8192, 256, 1.0e-7, "near_zero_var", 4),
        _make_case("ties_N4096_D1024", 4096, 1024, 1.0e-5, "ties", 5),
        _make_case("large_N131072_D256_outliers", 131072, 256, 1.0e-5, "outliers", 6),
        _make_case("zeroish_N1024_D4096", 1024, 4096, 1.0e-5, "zeroish", 7),
    ]

    passed = 0

    for case in cases:
        try:
            _run_case(case)
            passed += 1
            N, D = case["x"].shape
            print(
                f"case {case['name']:34s} PASS  "
                f"N={N} D={D} eps={case['eps']} dist={case['dist']}"
            )
        except Exception as exc:
            print(f"case {case['name']:34s} FAIL  {exc}")

    print(f"passed {passed} / {len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

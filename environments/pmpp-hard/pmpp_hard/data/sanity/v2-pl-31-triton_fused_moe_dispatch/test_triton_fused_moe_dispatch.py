# file: test_triton_fused_moe_dispatch.py

import inspect
import sys

import torch

import reference
import solution


SEED = 2026073101


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
        "scatter",
        "index_add",
        "index_put",
        "bincount",
        "segment",
        "grouped",
    ]

    for bad in banned:
        if bad in src:
            raise AssertionError(f"solution.py uses banned high-level op/string: {bad}")


def _rand_half(shape, device, generator, dist):
    if dist == "uniform":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.10, 0.10, generator=generator
        )
    elif dist == "hot":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.08, 0.08, generator=generator
        )
        flat = x.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            flat[(idx % 257) == 0] = 0.25
            flat[(idx % 389) == 0] = -0.25
    elif dist == "ties":
        vals = torch.tensor(
            [-0.125, -0.0625, -0.03125, 0.0, 0.03125, 0.0625, 0.125],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x = vals[ids]
    elif dist == "zeroish":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.10, 0.10, generator=generator
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.65
        x = torch.where(keep, x, torch.zeros_like(x))
    else:
        raise ValueError(f"unknown dist {dist}")

    return x.to(torch.float16).contiguous()


def _make_routes(N, E, Kroute, device, generator, dist):
    if dist == "hot":
        ids = torch.empty((N, Kroute), device=device, dtype=torch.int32)

        hot = torch.randint(0, min(E, 4), (N, Kroute), device=device, generator=generator)
        cold = torch.randint(0, E, (N, Kroute), device=device, generator=generator)
        choose_hot = torch.rand((N, Kroute), device=device, generator=generator) < 0.85
        ids[:] = torch.where(choose_hot, hot, cold).to(torch.int32)
    elif dist == "single_expert":
        ids = torch.zeros((N, Kroute), device=device, dtype=torch.int32)
        if Kroute == 2 and E > 1:
            ids[:, 1] = 1
    else:
        ids = torch.randint(0, E, (N, Kroute), device=device, dtype=torch.int32, generator=generator)

    if Kroute == 1:
        gates = torch.ones((N, 1), device=device, dtype=torch.float32)
    else:
        g0 = torch.empty((N,), device=device, dtype=torch.float32).uniform_(
            0.15, 0.85, generator=generator
        )
        gates = torch.stack([g0, 1.0 - g0], dim=1).contiguous()

    return ids.contiguous(), gates.contiguous()


def _make_case(name, N, D, H, E, Kroute, route_dist, value_dist, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed((SEED + seed_offset) + 880123)

    x = _rand_half((N, D), device, g, value_dist)
    w1 = _rand_half((E, D, H), device, g, value_dist)
    w2 = _rand_half((E, H, D), device, g, value_dist)
    expert_ids, gates = _make_routes(N, E, Kroute, device, g, route_dist)

    return {
        "name": name,
        "x": x,
        "expert_ids": expert_ids,
        "gates": gates,
        "w1": w1,
        "w2": w2,
        "route_dist": route_dist,
        "value_dist": value_dist,
    }


def _run_case(case):
    x = case["x"]
    expert_ids = case["expert_ids"]
    gates = case["gates"]
    w1 = case["w1"]
    w2 = case["w2"]

    x0 = x.clone()
    ids0 = expert_ids.clone()
    gates0 = gates.clone()
    w10 = w1.clone()
    w20 = w2.clone()

    got = solution.fused_moe_dispatch(x, expert_ids, gates, w1, w2)
    torch.cuda.synchronize()

    exp = reference.fused_moe_dispatch_ref(x, expert_ids, gates, w1, w2)
    torch.cuda.synchronize()

    if got.shape != exp.shape:
        raise AssertionError(f"{case['name']}: shape mismatch got={got.shape}, expected={exp.shape}")
    if got.dtype != torch.float32:
        raise AssertionError(f"{case['name']}: output dtype must be float32, got={got.dtype}")

    if not torch.equal(x, x0):
        raise AssertionError(f"{case['name']}: input x modified")
    if not torch.equal(expert_ids, ids0):
        raise AssertionError(f"{case['name']}: input expert_ids modified")
    if not torch.equal(gates, gates0):
        raise AssertionError(f"{case['name']}: input gates modified")
    if not torch.equal(w1, w10):
        raise AssertionError(f"{case['name']}: input w1 modified")
    if not torch.equal(w2, w20):
        raise AssertionError(f"{case['name']}: input w2 modified")

    ok = torch.allclose(got, exp, atol=reference.ATOL, rtol=reference.RTOL)

    if not ok:
        diff = (got - exp).abs()
        denom = exp.abs().clamp_min(1.0e-8)
        rel = diff / denom
        idx = int(diff.argmax().item())
        raise AssertionError(
            f"{case['name']}: output mismatch "
            f"max_abs={diff.max().item():.6g} "
            f"max_rel={rel.max().item():.6g} "
            f"flat_idx={idx} "
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
            "uniform_N256_D256_H256_E8_K1",
            256, 256, 256, 8, 1,
            "uniform",
            "uniform",
            1,
        ),
        _make_case(
            "hot_N512_D256_H512_E8_K2",
            512, 256, 512, 8, 2,
            "hot",
            "hot",
            2,
        ),
        _make_case(
            "uniform_N256_D1024_H512_E32_K2",
            256, 1024, 512, 32, 2,
            "uniform",
            "zeroish",
            3,
        ),
        _make_case(
            "hot_N1024_D512_H256_E32_K1",
            1024, 512, 256, 32, 1,
            "hot",
            "uniform",
            4,
        ),
        _make_case(
            "ties_N128_D2048_H512_E8_K2",
            128, 2048, 512, 8, 2,
            "single_expert",
            "ties",
            5,
        ),
        _make_case(
            "hot_N256_D512_H2048_E8_K2",
            256, 512, 2048, 8, 2,
            "hot",
            "zeroish",
            6,
        ),
    ]

    passed = 0

    for case in cases:
        try:
            _run_case(case)
            passed += 1
            N, D = case["x"].shape
            E, _, H = case["w1"].shape
            Kroute = case["expert_ids"].shape[1]
            print(
                f"case {case['name']:38s} PASS  "
                f"N={N} D={D} H={H} E={E} K={Kroute} "
                f"route={case['route_dist']} values={case['value_dist']}"
            )
        except Exception as exc:
            print(f"case {case['name']:38s} FAIL  {exc}")

    print(f"passed {passed} / {len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

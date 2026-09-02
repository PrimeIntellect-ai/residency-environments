# file: test_triton_chunked_state_scan.py

import inspect
import sys

import torch

import reference
import solution


SEED = 2026072101


def _assert_solution_uses_triton():
    src = inspect.getsource(solution)

    if "@triton.jit" not in src:
        raise AssertionError("solution.py must contain at least one @triton.jit kernel")

    banned = [
        "torch.cumsum",
        "torch.cumprod",
        "torch.scan",
        "associative_scan",
        "torch.nn.functional",
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
            -0.25, 0.25, generator=generator
        )
    elif dist == "zeroish":
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
    elif dist == "spiky":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.08, 0.08, generator=generator
        )
        flat = x.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            flat[(idx % 257) == 0] = 0.50
            flat[(idx % 389) == 0] = -0.50
    else:
        raise ValueError(f"unknown distribution: {dist}")

    return x.to(torch.float16).contiguous()


def _make_case(name, B, T, H, D, chunk_size, dist, ragged, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed((SEED + seed_offset) + 880123)

    x = _rand_half((B, T, H, D), device, g, dist)

    decay = torch.empty((B, T, H), device=device, dtype=torch.float32).uniform_(
        0.72, 0.995, generator=g
    )
    gate = torch.empty((B, T, H), device=device, dtype=torch.float32).uniform_(
        -0.35, 0.35, generator=g
    )

    if dist == "zeroish":
        gate_mask = torch.rand((B, T, H), device=device, generator=g) > 0.50
        gate = torch.where(gate_mask, gate, torch.zeros_like(gate))

    if dist == "ties":
        decay_vals = torch.tensor([0.75, 0.875, 0.9375, 0.96875], device=device)
        gate_vals = torch.tensor([-0.25, -0.125, 0.0, 0.125, 0.25], device=device)
        decay_ids = torch.randint(0, decay_vals.numel(), (B, T, H), device=device, generator=g)
        gate_ids = torch.randint(0, gate_vals.numel(), (B, T, H), device=device, generator=g)
        decay = decay_vals[decay_ids].contiguous()
        gate = gate_vals[gate_ids].contiguous()

    out_scale = torch.empty((H, D), device=device, dtype=torch.float32).uniform_(
        -1.25, 1.25, generator=g
    ).contiguous()
    skip = torch.empty((H, D), device=device, dtype=torch.float32).uniform_(
        -0.50, 0.50, generator=g
    ).contiguous()

    if ragged:
        lengths = torch.randint(
            low=0,
            high=T + 1,
            size=(B,),
            device=device,
            dtype=torch.int32,
            generator=g,
        )
        if B > 0:
            lengths[0] = 0
        if B > 1:
            lengths[1] = 1
        if B > 2:
            lengths[2] = T
        if B > 3:
            lengths[3] = max(0, T - 1)
    else:
        lengths = torch.full((B,), T, device=device, dtype=torch.int32)

    return {
        "name": name,
        "x": x,
        "decay": decay.contiguous(),
        "gate": gate.contiguous(),
        "out_scale": out_scale,
        "skip": skip,
        "lengths": lengths.contiguous(),
        "chunk_size": int(chunk_size),
        "dist": dist,
    }


def _run_case(case):
    x0 = case["x"].clone()
    decay0 = case["decay"].clone()
    gate0 = case["gate"].clone()
    out_scale0 = case["out_scale"].clone()
    skip0 = case["skip"].clone()
    lengths0 = case["lengths"].clone()

    got_y, got_state = solution.chunked_state_scan(
        case["x"],
        case["decay"],
        case["gate"],
        case["out_scale"],
        case["skip"],
        lengths=case["lengths"],
        chunk_size=case["chunk_size"],
    )
    torch.cuda.synchronize()

    exp_y, exp_state = reference.chunked_state_scan_ref(
        case["x"],
        case["decay"],
        case["gate"],
        case["out_scale"],
        case["skip"],
        lengths=case["lengths"],
        chunk_size=case["chunk_size"],
    )
    torch.cuda.synchronize()

    if got_y.shape != exp_y.shape:
        raise AssertionError(f"{case['name']}: y shape mismatch got={got_y.shape}, expected={exp_y.shape}")
    if got_state.shape != exp_state.shape:
        raise AssertionError(f"{case['name']}: final_state shape mismatch got={got_state.shape}, expected={exp_state.shape}")
    if got_y.dtype != torch.float32:
        raise AssertionError(f"{case['name']}: y dtype must be float32")
    if got_state.dtype != torch.float32:
        raise AssertionError(f"{case['name']}: final_state dtype must be float32")

    if not torch.equal(case["x"], x0):
        raise AssertionError(f"{case['name']}: input x modified")
    if not torch.equal(case["decay"], decay0):
        raise AssertionError(f"{case['name']}: input decay modified")
    if not torch.equal(case["gate"], gate0):
        raise AssertionError(f"{case['name']}: input gate modified")
    if not torch.equal(case["out_scale"], out_scale0):
        raise AssertionError(f"{case['name']}: input out_scale modified")
    if not torch.equal(case["skip"], skip0):
        raise AssertionError(f"{case['name']}: input skip modified")
    if not torch.equal(case["lengths"], lengths0):
        raise AssertionError(f"{case['name']}: input lengths modified")

    y_ok = torch.allclose(got_y, exp_y, atol=reference.ATOL, rtol=reference.RTOL)
    state_ok = torch.allclose(got_state, exp_state, atol=reference.ATOL, rtol=reference.RTOL)

    if not y_ok:
        diff = (got_y - exp_y).abs()
        denom = exp_y.abs().clamp_min(1.0e-8)
        rel = diff / denom
        idx = int(diff.argmax().item())
        raise AssertionError(
            f"{case['name']}: y mismatch max_abs={diff.max().item():.6g} "
            f"max_rel={rel.max().item():.6g} flat_idx={idx} "
            f"tolerance atol={reference.ATOL} rtol={reference.RTOL}"
        )

    if not state_ok:
        diff = (got_state - exp_state).abs()
        denom = exp_state.abs().clamp_min(1.0e-8)
        rel = diff / denom
        idx = int(diff.argmax().item())
        raise AssertionError(
            f"{case['name']}: final_state mismatch max_abs={diff.max().item():.6g} "
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
            "small_B1_T128_H1_D32_chunk16",
            B=1,
            T=128,
            H=1,
            D=32,
            chunk_size=16,
            dist="uniform",
            ragged=False,
            seed_offset=1,
        ),
        _make_case(
            "ragged_B4_T512_H4_D64_chunk64",
            B=4,
            T=512,
            H=4,
            D=64,
            chunk_size=64,
            dist="zeroish",
            ragged=True,
            seed_offset=2,
        ),
        _make_case(
            "ties_B8_T1024_H2_D128_chunk128",
            B=8,
            T=1024,
            H=2,
            D=128,
            chunk_size=128,
            dist="ties",
            ragged=True,
            seed_offset=3,
        ),
        _make_case(
            "long_B2_T2048_H8_D64_chunk64",
            B=2,
            T=2048,
            H=8,
            D=64,
            chunk_size=64,
            dist="spiky",
            ragged=True,
            seed_offset=4,
        ),
        _make_case(
            "boundary_B3_T257_H3_D32_chunk32",
            B=3,
            T=257,
            H=3,
            D=32,
            chunk_size=32,
            dist="uniform",
            ragged=True,
            seed_offset=5,
        ),
        _make_case(
            "chunk16_B4_T768_H4_D128",
            B=4,
            T=768,
            H=4,
            D=128,
            chunk_size=16,
            dist="zeroish",
            ragged=False,
            seed_offset=6,
        ),
    ]

    passed = 0

    for case in cases:
        try:
            _run_case(case)
            passed += 1
            B, T, H, D = case["x"].shape
            print(
                f"case {case['name']:38s} PASS  "
                f"B={B} T={T} H={H} D={D} chunk={case['chunk_size']} dist={case['dist']}"
            )
        except Exception as exc:
            print(f"case {case['name']:38s} FAIL  {exc}")

    print(f"passed {passed} / {len(cases)}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

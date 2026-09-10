# file: bench_triton_fused_layernorm_quant.py

import sys

import torch

import solution


SEED = 2026080102


def _rand_x(shape, device, generator, dist):
    N, D = shape

    if dist == "outliers":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -1.5, 1.5, generator=generator
        )
        rows = torch.arange(N, device=device)
        cols = (rows * 17) % D
        x[rows[::97], cols[::97]] = 512.0
        x[rows[::131], cols[::131]] = -512.0
    elif dist == "zeroish":
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -1.0, 1.0, generator=generator
        )
        keep = torch.rand(shape, device=device, generator=generator) > 0.65
        x = torch.where(keep, x, torch.zeros_like(x))
    elif dist == "ties":
        vals = torch.tensor(
            [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x = vals[ids]
    else:
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -2.0, 2.0, generator=generator
        )

    return x.contiguous()


def _rand_weight_bias(D, device, generator, dist):
    if dist == "outliers":
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
    g.manual_seed((SEED + seed_offset) + 880123)

    x = _rand_x((N, D), device, g, dist)
    weight, bias = _rand_weight_bias(D, device, g, dist)

    return name, x, weight, bias, float(eps), dist


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 50
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    cases = [
        _make_case("N65536_D256_uniform", 65536, 256, 1.0e-5, "uniform", 1),
        _make_case("N16384_D1024_outliers", 16384, 1024, 1.0e-5, "outliers", 2),
        _make_case("N4096_D4096_zeroish", 4096, 4096, 1.0e-5, "zeroish", 3),
        _make_case("N131072_D256_ties", 131072, 256, 1.0e-5, "ties", 4),
    ]

    for _ in range(8):
        for _, x, weight, bias, eps, _ in cases:
            _ = solution.fused_layernorm_quant(x, weight, bias, eps=eps)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        for _, x, weight, bias, eps, _ in cases:
            _ = solution.fused_layernorm_quant(x, weight, bias, eps=eps)
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    for name, x, _, _, eps, dist in cases:
        N, D = x.shape
        print(f"bench_case {name:26s} N={N} D={D} eps={eps} dist={dist}")

    print(f"avg_ms={avg_ms:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

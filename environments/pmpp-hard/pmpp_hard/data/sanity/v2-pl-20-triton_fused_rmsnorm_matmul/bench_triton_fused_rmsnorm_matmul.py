# file: bench_triton_fused_rmsnorm_matmul.py

import sys

import torch

import solution


SEED = 2026072002


# Anti-hack digest (REWARD_HACK_AUDIT_20260704): the paired perf scorer requires the
# student bench and the reference bench to print the SAME out_fnv before accepting the
# student's timing. The graded output is tolerance-graded fp32 (not bitwise stable across
# implementations), so the digest folds exact-graded facts only: output shape/dtype and a
# per-case ok-flag from an untimed torch oracle check at the grading tolerance.
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_MASK = (1 << 64) - 1

ATOL = 8.0e-2
RTOL = 8.0e-2


def _fnv_update(h, data):
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & FNV_MASK
    return h


def _oracle(x, weight, w, bias, eps, activation):
    torch.backends.cuda.matmul.allow_tf32 = False
    xf, wf, mat = x.float(), weight.float(), w.float()
    var = (xf * xf).mean(dim=1, keepdim=True)
    inv_rms = torch.rsqrt(var + float(eps))
    normed = (xf * inv_rms * wf[None, :]).to(torch.float16).float()
    out = normed @ mat
    if bias is not None:
        out = out + bias.float()[None, :]
    act = (activation or "none").lower() if isinstance(activation, (str, type(None))) else str(activation)
    if act == "relu":
        out = torch.relu(out)
    elif act in ("gelu", "gelu_tanh"):
        out = torch.nn.functional.gelu(out, approximate="tanh")
    return out




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
    else:
        x = torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            -0.15, 0.15, generator=generator
        )
        flat = x.reshape(-1)
        if flat.numel() > 0:
            idx = torch.arange(flat.numel(), device=device)
            flat[(idx % 251) == 0] = 0.75
            flat[(idx % 313) == 0] = -0.75

    return x.to(torch.float16).contiguous()


def _make_case(name, M, K, N, eps, activation, dist, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed((SEED + seed_offset) + 880123)

    x = _rand_half((M, K), device, g, dist)
    w_mat = _rand_half((K, N), device, g, dist)

    weight = torch.empty((K,), device=device, dtype=torch.float32).uniform_(
        0.25, 1.75, generator=g
    ).contiguous()

    bias = torch.empty((N,), device=device, dtype=torch.float32).uniform_(
        -0.50, 0.50, generator=g
    ).contiguous()

    return (name, x, weight, w_mat, bias, float(eps), activation, dist)


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 30
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    torch.backends.cuda.matmul.allow_tf32 = False

    cases = [
        _make_case("square_1024", 1024, 1024, 1024, 1.0e-5, "relu", "uniform", 1),
        _make_case("largeK_512x4096x1024", 512, 4096, 1024, 1.0e-5, "none", "zeroish", 2),
        _make_case("largeM_4096x1024x1024", 4096, 1024, 1024, 1.0e-5, "gelu", "ties", 3),
        _make_case("wideN_1024x2048x4096", 1024, 2048, 4096, 1.0e-4, "relu", "peaked", 4),
    ]

    for _ in range(8):
        for _, x, weight, w_mat, bias, eps, act, _ in cases:
            _ = solution.fused_rmsnorm_matmul(x, weight, w_mat, bias=bias, eps=eps, activation=act)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        for _, x, weight, w_mat, bias, eps, act, _ in cases:
            _ = solution.fused_rmsnorm_matmul(x, weight, w_mat, bias=bias, eps=eps, activation=act)
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    import struct

    h = FNV_BASIS
    for name, x, weight, w_mat, bias, eps, act, dist in cases:
        got = solution.fused_rmsnorm_matmul(x, weight, w_mat, bias=bias, eps=eps, activation=act)
        torch.cuda.synchronize()
        exp = _oracle(x, weight, w_mat, bias, eps, act)
        ok = (
            got.shape == exp.shape
            and got.dtype == torch.float32
            and torch.allclose(got, exp, atol=ATOL, rtol=RTOL)
        )
        h = _fnv_update(h, struct.pack("<qqB", got.shape[0], got.shape[1], 1 if ok else 0))
        h = _fnv_update(h, str(got.dtype).encode())
        h = _fnv_update(h, name.encode())

    for name, x, _, w_mat, _, eps, act, dist in cases:
        M, K = x.shape
        N = w_mat.shape[1]
        print(f"bench_case {name:24s} M={M} K={K} N={N} eps={eps} act={act} dist={dist}")

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

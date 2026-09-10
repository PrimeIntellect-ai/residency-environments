# file: bench_triton_persistent_grouped_gemm.py

import sys

import torch

import solution


SEED = 2026071902


# Anti-hack digest (REWARD_HACK_AUDIT_20260704): the paired perf scorer requires the
# student bench and the reference bench to print the SAME out_fnv before accepting the
# student's timing. The graded output is tolerance-graded fp32 (not bitwise stable across
# implementations), so the digest folds exact-graded facts only: output shape/dtype/numel
# and a per-case ok-flag from an untimed torch oracle check at the grading tolerance.
FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
FNV_MASK = (1 << 64) - 1

ATOL = 8.0e-2
RTOL = 8.0e-2


def _fnv_update(h, data):
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & FNV_MASK
    return h


def _oracle(a, b, bias, gm, gn, gk, ao, bo, bio, co, activation):
    torch.backends.cuda.matmul.allow_tf32 = False
    G = int(gm.numel())
    last = G - 1
    total_c = int((co[last] + gm[last].to(torch.int64) * gn[last].to(torch.int64)).item())
    out = torch.empty((total_c,), device=a.device, dtype=torch.float32)
    af, bf = a.float(), b.float()
    for g in range(G):
        M, N, K = int(gm[g].item()), int(gn[g].item()), int(gk[g].item())
        if M == 0 or N == 0:
            continue
        a0, b0, bias0, c0 = int(ao[g].item()), int(bo[g].item()), int(bio[g].item()), int(co[g].item())
        A = af[a0 : a0 + M * K].reshape(M, K)
        B = bf[b0 : b0 + K * N].reshape(K, N)
        C = A @ B + bias[bias0 : bias0 + N].float()[None, :]
        if (activation.lower() if isinstance(activation, str) else ("relu" if int(activation) == 0 else "gelu")) == "relu":
            C = torch.relu(C)
        else:
            C = torch.nn.functional.gelu(C, approximate="tanh")
        out[c0 : c0 + M * N] = C.reshape(-1)
    return out




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
    else:
        vals = torch.tensor(
            [-0.1875, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.1875],
            device=device,
            dtype=torch.float32,
        )
        ids = torch.randint(0, vals.numel(), shape, device=device, generator=generator)
        x = vals[ids]
    return x


def _make_case(name, dims, activation, dist, seed_offset):
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed((SEED + seed_offset) + 880123)

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

    a = _float_to_half_tensor(_rand_values((total_a,), device, g, dist))
    b = _float_to_half_tensor(_rand_values((total_b,), device, g, dist))
    bias = torch.empty((total_bias,), device=device, dtype=torch.float32).uniform_(
        -0.20, 0.20, generator=g
    )

    return (
        name,
        a,
        b,
        bias,
        torch.tensor(group_m_host, device=device, dtype=torch.int32),
        torch.tensor(group_n_host, device=device, dtype=torch.int32),
        torch.tensor(group_k_host, device=device, dtype=torch.int32),
        torch.tensor(a_offsets_host, device=device, dtype=torch.int64),
        torch.tensor(b_offsets_host, device=device, dtype=torch.int64),
        torch.tensor(bias_offsets_host, device=device, dtype=torch.int64),
        torch.tensor(c_offsets_host, device=device, dtype=torch.int64),
        activation,
        total_c,
    )


def _powerlaw_dims():
    dims = []
    for g in range(32):
        if g == 0:
            M = 1024
        elif g == 1:
            M = 512
        elif g < 4:
            M = 256
        elif g < 12:
            M = 64
        else:
            M = 8

        N = 512 if g % 4 == 0 else 256
        K = 512 if g % 3 == 0 else 256
        dims.append((M, N, K))
    return dims


def _many_tiny_dims():
    dims = []
    for g in range(256):
        if g == 0:
            M = 256
        elif g < 16:
            M = 32
        else:
            M = 1 + (g % 4)
        dims.append((M, 128, 128))
    return dims


def _balanced_dims():
    return [(512, 512, 512), (512, 512, 512), (512, 512, 512), (512, 512, 512)]


def _large_four_dims():
    return [(512, 1024, 1024), (256, 768, 2048), (128, 512, 512), (64, 2048, 256)]


def main():
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 1

    iters = 30
    if len(sys.argv) >= 2:
        iters = max(1, int(sys.argv[1]))

    cases = [
        _make_case("powerlaw_G32_gelu", _powerlaw_dims(), "gelu", "powerlaw", 1),
        _make_case("many_tiny_G256_relu", _many_tiny_dims(), "relu", "tiny", 2),
        _make_case("balanced_G4_relu", _balanced_dims(), "relu", "uniform", 3),
        _make_case("large_G4_gelu", _large_four_dims(), "gelu", "ties", 4),
    ]

    for _ in range(8):
        for case in cases:
            _, a, b, bias, gm, gn, gk, ao, bo, bio, co, act, _ = case
            _ = solution.grouped_gemm_bias_act(a, b, bias, gm, gn, gk, ao, bo, bio, co, activation=act)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        for case in cases:
            _, a, b, bias, gm, gn, gk, ao, bo, bio, co, act, _ = case
            _ = solution.grouped_gemm_bias_act(a, b, bias, gm, gn, gk, ao, bo, bio, co, activation=act)
    stop.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    avg_ms = elapsed_ms / float(iters * len(cases))

    import struct

    h = FNV_BASIS
    for case in cases:
        name, a, b, bias, gm, gn, gk, ao, bo, bio, co, act, total_c = case
        got = solution.grouped_gemm_bias_act(a, b, bias, gm, gn, gk, ao, bo, bio, co, activation=act)
        torch.cuda.synchronize()
        exp = _oracle(a, b, bias, gm, gn, gk, ao, bo, bio, co, act)
        ok = (
            tuple(got.shape) == (total_c,)
            and got.dtype == torch.float32
            and torch.allclose(got, exp, atol=ATOL, rtol=RTOL)
        )
        h = _fnv_update(h, struct.pack("<qB", total_c, 1 if ok else 0))
        h = _fnv_update(h, str(got.dtype).encode())
        h = _fnv_update(h, name.encode())

    for case in cases:
        print(f"bench_case {case[0]:24s} total_c={case[-1]} act={case[-2]}")

    print(f"avg_ms={avg_ms:.6f}")
    print(f"out_fnv={h:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# PMPP_CANARY_19_72191b9310 -- held-out canary; MUST NOT appear in any submission
# file: reference.py

import torch


ATOL = 8.0e-2
RTOL = 8.0e-2


def _apply_activation(x, activation):
    if isinstance(activation, str):
        act = activation.lower()
    else:
        act = "relu" if int(activation) == 0 else "gelu_tanh"

    if act == "relu":
        return torch.relu(x)

    if act in ("gelu", "gelu_tanh"):
        return torch.nn.functional.gelu(x, approximate="tanh")

    raise ValueError(f"unknown activation: {activation}")


def grouped_gemm_bias_act_ref(
    a,
    b,
    bias,
    group_m,
    group_n,
    group_k,
    a_offsets,
    b_offsets,
    bias_offsets,
    c_offsets,
    activation="relu",
):
    torch.backends.cuda.matmul.allow_tf32 = False

    group_count = int(group_m.numel())

    last = group_count - 1
    total_c = int((c_offsets[last] + group_m[last].to(torch.int64) * group_n[last].to(torch.int64)).item())

    out = torch.empty((total_c,), device=a.device, dtype=torch.float32)

    af = a.float()
    bf = b.float()

    for g in range(group_count):
        M = int(group_m[g].item())
        N = int(group_n[g].item())
        K = int(group_k[g].item())

        c0 = int(c_offsets[g].item())

        if M == 0 or N == 0:
            continue

        a0 = int(a_offsets[g].item())
        b0 = int(b_offsets[g].item())
        bias0 = int(bias_offsets[g].item())

        A = af[a0 : a0 + M * K].reshape(M, K)
        B = bf[b0 : b0 + K * N].reshape(K, N)
        Bias = bias[bias0 : bias0 + N].float()

        C = A @ B
        C = C + Bias[None, :]
        C = _apply_activation(C, activation)

        out[c0 : c0 + M * N] = C.reshape(-1)

    return out

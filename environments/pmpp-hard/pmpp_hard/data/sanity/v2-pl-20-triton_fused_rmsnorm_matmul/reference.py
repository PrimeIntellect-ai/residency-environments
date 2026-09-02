# file: reference.py

import torch


ATOL = 8.0e-2
RTOL = 8.0e-2


def _apply_activation(x, activation):
    if activation is None:
        return x

    if isinstance(activation, str):
        act = activation.lower()
    else:
        act = {0: "none", 1: "relu", 2: "gelu"}[int(activation)]

    if act in ("none", "identity", ""):
        return x
    if act == "relu":
        return torch.relu(x)
    if act in ("gelu", "gelu_tanh"):
        return torch.nn.functional.gelu(x, approximate="tanh")

    raise ValueError(f"unknown activation: {activation}")


def fused_rmsnorm_matmul_ref(x, weight, w, bias=None, eps=1.0e-5, activation="none"):
    torch.backends.cuda.matmul.allow_tf32 = False

    xf = x.float()
    wf = weight.float()
    mat = w.float()

    var = (xf * xf).mean(dim=1, keepdim=True)
    inv_rms = torch.rsqrt(var + float(eps))

    # The Triton solution uses tensor-core dot on a normalized fp16 tile.
    # The oracle mirrors that contract: rmsnorm arithmetic is fp32, the
    # normalized activation is rounded to fp16, and the matmul accumulates fp32.
    normed = (xf * inv_rms * wf[None, :]).to(torch.float16).float()

    out = normed @ mat

    if bias is not None:
        out = out + bias.float()[None, :]

    out = _apply_activation(out, activation)
    return out

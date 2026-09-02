# PMPP_CANARY_33_4a82784f72 -- held-out canary; MUST NOT appear in any submission
# file: reference.py

import torch


SCALE_ATOL = 2.0e-3
SCALE_RTOL = 2.0e-3
DEQUANT_ATOL = 5.0e-3
DEQUANT_RTOL = 2.0e-3

# Codes are expected to be exact except for numerically tie-sensitive cases.
# The test accepts +/-1 only when the reference pre-round value is close to an
# x.5 boundary or near int8 saturation.
Q_TIE_EPS = 3.5e-2


def _round_nearest_even_torch(x):
    return torch.round(x)


def fused_layernorm_quant_ref_with_pre(x, weight, bias, eps=1.0e-5):
    xf = x.float()
    wf = weight.float()
    bf = bias.float()

    mean = xf.mean(dim=1, keepdim=True)
    centered = xf - mean
    var = (centered * centered).mean(dim=1, keepdim=True)
    inv = torch.rsqrt(var + float(eps))

    y = centered * inv * wf[None, :] + bf[None, :]

    amax = y.abs().amax(dim=1)
    scale = torch.where(amax > 0.0, amax / 127.0, torch.ones_like(amax))

    q_pre = y / scale[:, None]
    q_i32 = _round_nearest_even_torch(q_pre).clamp(-127, 127).to(torch.int32)
    q = q_i32.to(torch.int8)
    dequant = q_i32.float() * scale[:, None]

    return q, scale, dequant, q_pre


def fused_layernorm_quant_ref(x, weight, bias, eps=1.0e-5):
    q, scale, dequant, _ = fused_layernorm_quant_ref_with_pre(x, weight, bias, eps)
    return q, scale, dequant

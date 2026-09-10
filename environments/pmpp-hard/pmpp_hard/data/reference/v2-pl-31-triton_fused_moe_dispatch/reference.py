# PMPP_CANARY_31_2b4dc802a9 -- held-out canary; MUST NOT appear in any submission
# file: reference.py

import torch


ATOL = 1.2e-1
RTOL = 8.0e-2


def fused_moe_dispatch_ref(x, expert_ids, gates, w1, w2):
    torch.backends.cuda.matmul.allow_tf32 = False

    if x.dim() != 2:
        raise ValueError("x must have shape [N, D]")

    N, D = x.shape
    E, D1, H = w1.shape
    E2, H2, D2 = w2.shape

    if D1 != D or E2 != E or H2 != H or D2 != D:
        raise ValueError("weight shape mismatch")

    Kroute = expert_ids.shape[1]

    out = torch.zeros((N, D), device=x.device, dtype=torch.float32)

    xf = x.float()
    w1f = w1.float()
    w2f = w2.float()
    gatesf = gates.float()

    for r in range(Kroute):
        ids_r = expert_ids[:, r].to(torch.long)

        for e in range(E):
            mask = ids_r == e
            if not bool(mask.any().item()):
                continue

            idx = torch.nonzero(mask, as_tuple=False).flatten()
            x_e = xf[idx]
            hidden = x_e @ w1f[e]
            hidden = hidden * torch.sigmoid(hidden)
            y_e = hidden @ w2f[e]
            out[idx] += gatesf[idx, r][:, None] * y_e

    return out

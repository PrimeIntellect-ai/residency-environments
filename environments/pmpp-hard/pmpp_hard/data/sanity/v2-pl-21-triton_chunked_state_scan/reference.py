# file: reference.py

import torch


ATOL = 5.0e-2
RTOL = 5.0e-2


def chunked_state_scan_ref(x, decay, gate, out_scale, skip, lengths=None, chunk_size=64):
    del chunk_size

    B, T, H, D = x.shape

    xf = x.float()
    decayf = decay.float()
    gatef = gate.float()
    out_scalef = out_scale.float()
    skipf = skip.float()

    if lengths is None:
        lens = torch.full((B,), T, device=x.device, dtype=torch.int64)
    else:
        lens = lengths.to(torch.int64)

    y = torch.zeros((B, T, H, D), device=x.device, dtype=torch.float32)
    final_state = torch.zeros((B, H, D), device=x.device, dtype=torch.float32)

    for b in range(B):
        L = int(lens[b].item())
        if L < 0:
            L = 0
        if L > T:
            L = T

        for h in range(H):
            state = torch.zeros((D,), device=x.device, dtype=torch.float32)

            for t in range(L):
                state = decayf[b, t, h] * state + gatef[b, t, h] * xf[b, t, h]
                y[b, t, h] = state * out_scalef[h] + skipf[h] * xf[b, t, h]

            final_state[b, h] = state

    return y, final_state

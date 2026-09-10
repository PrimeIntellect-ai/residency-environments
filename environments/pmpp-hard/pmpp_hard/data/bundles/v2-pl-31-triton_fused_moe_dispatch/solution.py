import torch, triton, triton.language as tl
@triton.jit
def _noop(): pass
def fused_moe_dispatch(*a, **k):
    raise NotImplementedError('implement fused_moe_dispatch with triton kernels')

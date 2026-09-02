import torch, triton, triton.language as tl
@triton.jit
def _noop(): pass
def fused_layernorm_quant(*a, **k):
    raise NotImplementedError('implement fused_layernorm_quant with triton kernels')

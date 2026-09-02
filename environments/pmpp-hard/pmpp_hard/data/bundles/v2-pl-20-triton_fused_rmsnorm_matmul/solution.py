import torch, triton, triton.language as tl
@triton.jit
def _noop(): pass
def fused_rmsnorm_matmul(*a, **k):
    raise NotImplementedError('implement fused_rmsnorm_matmul with triton kernels')

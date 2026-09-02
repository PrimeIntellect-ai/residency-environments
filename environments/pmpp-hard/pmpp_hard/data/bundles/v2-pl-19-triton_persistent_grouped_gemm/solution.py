import torch, triton, triton.language as tl
@triton.jit
def _noop(): pass
def grouped_gemm(*a, **k):
    raise NotImplementedError('implement grouped_gemm with triton kernels')

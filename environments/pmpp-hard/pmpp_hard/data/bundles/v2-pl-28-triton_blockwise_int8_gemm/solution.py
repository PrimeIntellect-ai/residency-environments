import torch, triton, triton.language as tl
@triton.jit
def _noop(): pass
def blockwise_int8_gemm(*a, **k):
    raise NotImplementedError('implement blockwise_int8_gemm with triton kernels')

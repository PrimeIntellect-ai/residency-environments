import torch, triton, triton.language as tl
@triton.jit
def _noop(): pass
def chunked_state_scan(*a, **k):
    raise NotImplementedError('implement chunked_state_scan with triton kernels')

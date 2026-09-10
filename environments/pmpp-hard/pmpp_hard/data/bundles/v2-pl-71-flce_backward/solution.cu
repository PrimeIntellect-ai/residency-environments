// Empty starter stub (no-op). Implement the solution_* ABI from flce_backward_common.h.
#include "flce_backward_common.h"
#include <cuda_runtime.h>

extern "C" size_t solution_workspace_bytes(const FLCBSpec* spec) { return 0; }

extern "C" cudaError_t solution_init(const FLCBSpec* spec, void** state, cudaStream_t stream) {
    if (state) *state = nullptr;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(void* state, const FLCBRunSpec* run, const void* inputs,
                                    void* outputs, void* workspace, size_t workspace_bytes,
                                    cudaStream_t stream) {
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) { return cudaSuccess; }

extern "C" void solution_destroy(void* state) {}

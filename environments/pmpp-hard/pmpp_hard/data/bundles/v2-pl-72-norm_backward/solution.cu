// solution.cu — empty no-op ABI stub (student fills this in).
// It returns success but writes nothing. It must never contain reference code.
#include "norm_backward_common.h"
#include <cstdlib>

extern "C" size_t solution_workspace_bytes(const Spec*) { return 0; }

extern "C" cudaError_t solution_init(const Spec*, void** state, cudaStream_t) {
    if (state) *state = malloc(1);
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void*, const RunSpec*, const void*, void*, void*, size_t, cudaStream_t) {
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void*, cudaStream_t) { return cudaSuccess; }

extern "C" void solution_destroy(void* state) { free(state); }

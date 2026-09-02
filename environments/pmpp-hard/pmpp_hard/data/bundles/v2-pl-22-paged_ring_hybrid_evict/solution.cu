// file: solution.cu
// Empty starter stub (no-op). Implement the solution_* ABI below.
#include "paged_ring_hybrid_evict_common.h"

extern "C" size_t solution_workspace_bytes(const PrheProblemSpec* spec) { return 0; }

extern "C" cudaError_t solution_init(
    const PrheProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) { if (state_out) *state_out = nullptr; return cudaSuccess; }

extern "C" cudaError_t solution_run(
    void* state,
    const PrheRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) { return cudaSuccess; }

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) { return cudaSuccess; }

extern "C" void solution_destroy(void* state) { }


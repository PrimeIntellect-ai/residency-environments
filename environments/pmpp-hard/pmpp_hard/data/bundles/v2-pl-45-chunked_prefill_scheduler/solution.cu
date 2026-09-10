#include "chunked_prefill_scheduler_common.h"

extern "C" size_t solution_workspace_bytes(const CpsProblemSpec* spec) { return 0; }
extern "C" cudaError_t solution_init(
    const CpsProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) { if (state_out) *state_out = nullptr; return cudaSuccess; }
extern "C" cudaError_t solution_run(
    void* state,
    const CpsOp* op,
    const void* inputs,        // unused; kept for ABI symmetry (pass nullptr)
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) { return cudaSuccess; }
extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) { return cudaSuccess; }
extern "C" void solution_destroy(void* state) { }

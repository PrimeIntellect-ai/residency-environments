// file: solution.cu
// Empty starter stub (no-op). Implement the solution_* ABI below.
// The ABI structs and constants live in dcrs_allreduce_common.h.
#include "dcrs_allreduce_common.h"

extern "C" size_t solution_workspace_bytes(const DcrSpec* spec) {
    (void)spec;
    return 0;
}

extern "C" cudaError_t solution_init(
    const DcrSpec* spec,
    void** state,
    cudaStream_t stream) {
    (void)spec;
    (void)stream;
    if (state) *state = nullptr;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const DcrRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)state;
    (void)run;
    (void)inputs;
    (void)outputs;
    (void)workspace;
    (void)workspace_bytes;
    (void)stream;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    (void)state;
    (void)stream;
    return cudaSuccess;
}

extern "C" void solution_destroy(void* state) {
    (void)state;
}

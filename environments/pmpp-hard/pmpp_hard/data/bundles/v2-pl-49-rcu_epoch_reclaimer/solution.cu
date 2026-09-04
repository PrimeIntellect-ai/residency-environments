#include "rcu_epoch_reclaimer_common.h"

extern "C" size_t solution_workspace_bytes(const RcuProblemSpec* spec) { return 0; }
extern "C" cudaError_t solution_init(
    const RcuProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) { return cudaSuccess; }
extern "C" cudaError_t solution_run(
    void* state,
    const RcuRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) { return cudaSuccess; }
extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) { return cudaSuccess; }
extern "C" void solution_destroy(void* state) { }

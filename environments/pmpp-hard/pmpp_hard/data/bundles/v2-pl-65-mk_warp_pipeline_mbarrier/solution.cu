#include "mk_warp_pipeline_mbarrier_common.h"

extern "C" size_t solution_workspace_bytes(const MkwpProblemSpec* spec) { return 0; }
extern "C" cudaError_t solution_init(
    const MkwpProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) { return cudaSuccess; }
extern "C" cudaError_t solution_run(
    void* state,
    const MkwpRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) { return cudaSuccess; }
extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) { return cudaSuccess; }
extern "C" void solution_destroy(void* state) { }

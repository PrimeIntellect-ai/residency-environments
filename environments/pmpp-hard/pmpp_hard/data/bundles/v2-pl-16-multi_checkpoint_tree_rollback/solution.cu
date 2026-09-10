#include "multi_checkpoint_tree_rollback_common.h"
extern "C" size_t solution_workspace_bytes(const MctrProblemSpec*){return 0;}
extern "C" cudaError_t solution_init(const MctrProblemSpec*, void** s, cudaStream_t){if(s)*s=nullptr;return cudaSuccess;}
extern "C" cudaError_t solution_run(void*, const MctrRunSpec*, const void*, void*, void*, size_t, cudaStream_t){return cudaSuccess;}
extern "C" cudaError_t solution_reset(void*, cudaStream_t){return cudaSuccess;}
extern "C" void solution_destroy(void*){}

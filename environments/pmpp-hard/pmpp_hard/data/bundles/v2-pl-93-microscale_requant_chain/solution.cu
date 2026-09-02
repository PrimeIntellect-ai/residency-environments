#include "microscale_requant_chain_common.h"
extern "C" size_t solution_workspace_bytes(const MrqProblemSpec*){return 0;}
extern "C" cudaError_t solution_init(const MrqProblemSpec*, void** s, cudaStream_t){if(s)*s=nullptr;return cudaSuccess;}
extern "C" cudaError_t solution_run(void*, const MrqRunSpec*, const void*, void*, void*, size_t, cudaStream_t){return cudaSuccess;}
extern "C" cudaError_t solution_reset(void*, cudaStream_t){return cudaSuccess;}
extern "C" void solution_destroy(void*){}

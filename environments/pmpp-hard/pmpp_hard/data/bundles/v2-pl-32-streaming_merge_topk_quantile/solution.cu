#include "streaming_merge_topk_quantile_common.h"
extern "C" size_t solution_workspace_bytes(const SmtqProblemSpec*){return 0;}
extern "C" cudaError_t solution_init(const SmtqProblemSpec*, void** s, cudaStream_t){if(s)*s=nullptr;return cudaSuccess;}
extern "C" cudaError_t solution_run(void*, const SmtqRunSpec*, const void*, void*, void*, size_t, cudaStream_t){return cudaSuccess;}
extern "C" cudaError_t solution_reset(void*, cudaStream_t){return cudaSuccess;}
extern "C" void solution_destroy(void*){}

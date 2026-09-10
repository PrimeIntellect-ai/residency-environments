// file: paged_ring_hybrid_evict_common.h

#ifndef PAGED_RING_HYBRID_EVICT_COMMON_H_
#define PAGED_RING_HYBRID_EVICT_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define PRHE_ABI_VERSION 1

#define PRHE_MIN_B 1
#define PRHE_MAX_B 64
#define PRHE_MIN_MAX_LEN 16
#define PRHE_MAX_MAX_LEN 65536
#define PRHE_MIN_PAGE_SIZE 4
#define PRHE_MAX_PAGE_SIZE 128
#define PRHE_MIN_WINDOW 1
#define PRHE_MAX_WINDOW 4096
#define PRHE_MIN_MAX_PAGES 1
#define PRHE_MAX_MAX_PAGES 65536
#define PRHE_MAX_NEW_TOKENS 4
#define PRHE_MAX_STEPS 64

/*
CONTRACT: paged_ring_hybrid_evict

Stateful exact-integer hybrid KV/page manager.

Persistent state:
  - A global physical page pool with max_pages pages.
  - Each page has page_size int32 token slots.
  - Each page has owner sequence, logical page id, and last_used step.
  - Per-sequence page table maps logical_page -> physical_page.
  - Per-sequence logical length, ring_head, and ring_count.
  - The sliding-window ring is logically the last min(length, window_size)
    appended tokens. The implementation may represent it implicitly.

Inputs per step:
  active_count rows, each with:
    active_seq[row]
    append_count[row], clamped to [0, max_new_tokens]
    token_values[row, max_new_tokens]

Append:
  For each active row in row order, append up to append_count values.
  If length[seq] == max_len, extra tokens are discarded.
  If the target logical page is not resident, allocate a free physical page.
  If no page is free, evict the least-recently-used non-pinned page:
    - page A is older than page B if last_used[A] < last_used[B]
    - ties choose smaller physical page id
  A page is pinned iff it intersects the current live window of its owner at
  the moment allocation is attempted.
  If no free or evictable page exists, the append token is dropped and length
  is unchanged.

Live window:
  For sequence s with length L:
    start = max(0, L - window_size)
    live logical token positions are [start, L).
  live_count is the number of resident live tokens in that range.
  live_sum is int64 wrapping sum of resident live token values.
  live_hash is FNV-1a over:
    live_count int32,
    then for each resident live token in increasing logical-position order:
      logical_position int32,
      value int32

After each step, every resident page touched by live-window reduction has
last_used = current step id.

Outputs after every step:
  live_count[B]
  live_sum[B]
  live_hash[B]
  page_table_checksum[1]
  evicted_count[1]
  free_pages[1]

page_table_checksum is FNV-1a over:
  B, max_len, page_size, window_size, max_pages, current step counter,
  then for every sequence:
    length, ring_head, ring_count, full page_table entries,
  then for every physical page:
    page_owner, page_logical, page_last_used.

Rules:
  - solution_init may allocate persistent state.
  - solution_run may not call cudaMalloc/cudaFree.
  - solution_run may launch kernels and use the supplied workspace.
  - Exact-int only; no floating point.
*/

struct alignas(8) PrheProblemSpec {
    int32_t abi_version;
    int32_t B;
    int32_t max_len;
    int32_t page_size;
    int32_t window_size;
    int32_t max_pages;
    int32_t max_active;
    int32_t max_new_tokens;
    int32_t max_steps;
    int32_t flags;
    int32_t reserved[6];
};

struct alignas(8) PrheRunSpec {
    int32_t abi_version;
    int32_t active_count;
    int32_t step_id;
    int32_t reserved[13];
};

struct alignas(8) PrheInputs {
    const int32_t* active_seq;
    const int32_t* append_count;
    const int32_t* token_values;
};

struct alignas(8) PrheOutputs {
    int32_t* live_count;
    int64_t* live_sum;
    uint64_t* live_hash;
    uint64_t* page_table_checksum;
    int32_t* evicted_count;
    int32_t* free_pages;
};

static_assert(sizeof(PrheProblemSpec) == 64, "PrheProblemSpec layout drift");
static_assert(sizeof(PrheRunSpec) == 64, "PrheRunSpec layout drift");
static_assert(sizeof(PrheInputs) == 24, "PrheInputs layout drift");
static_assert(sizeof(PrheOutputs) == 48, "PrheOutputs layout drift");

static inline size_t prhe_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int prhe_ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

static inline int prhe_max_logical_pages_host(int max_len, int page_size) {
    return (max_len + page_size - 1) / page_size;
}

static inline int prhe_validate_problem_spec(const PrheProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != PRHE_ABI_VERSION) return 0;
    if (spec->B < PRHE_MIN_B || spec->B > PRHE_MAX_B) return 0;
    if (spec->max_len < PRHE_MIN_MAX_LEN || spec->max_len > PRHE_MAX_MAX_LEN) return 0;
    if (spec->page_size < PRHE_MIN_PAGE_SIZE || spec->page_size > PRHE_MAX_PAGE_SIZE) return 0;
    if (spec->window_size < PRHE_MIN_WINDOW || spec->window_size > PRHE_MAX_WINDOW) return 0;
    if (spec->max_pages < PRHE_MIN_MAX_PAGES || spec->max_pages > PRHE_MAX_MAX_PAGES) return 0;
    if (spec->max_active < 0 || spec->max_active > spec->B) return 0;
    if (spec->max_new_tokens < 1 || spec->max_new_tokens > PRHE_MAX_NEW_TOKENS) return 0;
    if (spec->max_steps < 1 || spec->max_steps > PRHE_MAX_STEPS) return 0;
    return 1;
}

static inline int prhe_validate_run_spec(const PrheRunSpec* run, const PrheProblemSpec* spec) {
    if (!run || !spec) return 0;
    if (run->abi_version != PRHE_ABI_VERSION) return 0;
    if (run->active_count < 0 || run->active_count > spec->max_active) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const PrheProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const PrheProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const PrheRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // PAGED_RING_HYBRID_EVICT_COMMON_H_

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

// === DETERMINISM & EXACT OUTPUT SERIALIZATION (normative) ===
//
// This section is SELF-CONTAINED. Every grader-enforced number/hash below is
// fully specified here; nothing is deferred to any other file.
//
// ---- derived constant ----
//   max_logical_pages = ceil(max_len / page_size)
//                     = (max_len + page_size - 1) / page_size.
//   This is the page_table column count and the upper bound on logical page id.
//
// ---- step counter ----
//   A persistent int32 step_counter starts at 0 (reset by solution_reset).
//   At the START of every solution_run (every step, even active_count==0), it
//   is incremented by one; that post-increment value `step` is the value used
//   for all page last_used writes this step AND folded into
//   page_table_checksum. (Plain int32; wraps mod 2^32.)
//
// ---- FNV-1a-64 primitive ----
//   basis = 1469598103934665603 (0x14650FB0739D0383)
//   prime = 1099511628211       (0x00000100000001B3)
//   fold one byte b:  h = (h XOR b) * prime  (mod 2^64).
//   Multi-byte fields are folded byte-by-byte in LITTLE-ENDIAN order (the raw
//   in-memory bytes, index 0..size-1). Every int32 is folded as its 4 raw
//   bytes. All hashes start from h = basis.
//
// ---- append order & eviction (per step) ----
//   Rows are processed in row order r=0..active_count-1. append_count[r] is
//   clamped to [0, max_new_tokens] (and to PRHE_MAX_NEW_TOKENS=4). For each
//   row, values token_values[r][0..cnt) are appended one at a time via the
//   append-one procedure below, in index order.
//
//   append-one(seq, value):
//     - if seq not in [0,B): ignore.
//     - if length[seq] >= max_len: ignore (token discarded, length unchanged).
//     - logical_page = length[seq] / page_size;
//       page_offset = length[seq] - logical_page*page_size;
//       table_idx   = seq*max_logical_pages + logical_page.
//     - phys = page_table[table_idx]. If phys < 0, allocate:
//         * Scan physical pages p=0..max_pages-1; return the FIRST p with
//           page_owner[p] < 0 (free). (lowest free physical id)
//         * If none free, choose an eviction VICTIM among pages whose owner is
//           in [0,B) AND that do NOT intersect their owner's current live
//           window (see below). Among those candidates pick min last_used;
//           tie -> min physical page id. If a victim is found: if its owner's
//           page_table entry still points at it, clear that entry to -1; set
//           the victim page owner=-1, logical=-1, last_used=step; increment the
//           per-step evicted counter; the victim physical id is the allocation.
//         * If NO free page and NO evictable victim: the append token is
//           dropped, length unchanged, return.
//       After obtaining phys for a fresh page: page_table[table_idx]=phys,
//       page_owner[phys]=seq, page_logical[phys]=logical_page,
//       page_last_used[phys]=step.
//     - Write page_data[phys*page_size+page_offset]=value;
//       page_last_used[phys]=step; length[seq]+=1;
//       ring_head[seq]=length[seq] % window_size;
//       ring_count[seq]=min(length[seq], window_size).
//
//   live-window intersection test for page (owner=seq, logical_page=lp), used
//   to decide pinning during eviction-candidate selection:
//     len=length[seq]; if len<=0 -> NOT intersecting.
//     start = max(0, len - window_size); end = len - 1.
//     page_start = lp*page_size; page_end = page_start + page_size - 1.
//     intersects iff (page_start <= end) AND (page_end >= start).
//   A page is "pinned" (ineligible as victim) iff it intersects, evaluated at
//   the moment of the allocation attempt.
//
// ---- live_count / live_sum / live_hash[seq] (per step, ALL B sequences) ----
//   Computed AFTER all appends, for seq=0..B-1. len=length[seq];
//   start = max(0, len - window_size). Iterate logical positions pos in
//   [start, len) ascending:
//     lp = pos/page_size; off = pos - lp*page_size;
//     phys = page_table[seq*max_logical_pages + lp];
//     a position is RESIDENT iff phys >= 0 (its logical page is mapped).
//   live_count = number of resident positions in [start,len).
//   live_sum: uint64 accumulator sum_bits=0; for each resident pos add
//     (uint64_t)(int64_t)page_data[phys*page_size+off]; live_sum=(int64_t)sum_bits.
//   live_hash: h=basis; fold live_count as int32; then for each resident pos in
//     ascending pos order fold pos as int32 THEN value as int32. (Non-resident
//     positions are skipped entirely — not folded.)
//   SIDE EFFECT: while folding live_hash, every resident page touched sets
//     page_last_used[phys]=step. (This happens during the live_hash pass over
//     [start,len), so the checksum below sees these updated last_used values.)
//
// ---- page_table_checksum[1] (single global hash, computed AFTER live pass) ----
//   h = basis. Fold IN THIS EXACT ORDER, each scalar as int32:
//     B, max_len, page_size, window_size, max_pages, step.
//   Then for seq = 0..B-1 (ascending):
//     fold length[seq], ring_head[seq], ring_count[seq] (each int32);
//     then for lp = 0..max_logical_pages-1 (ascending):
//       fold page_table[seq*max_logical_pages + lp] as int32 (-1 if absent).
//   Then for p = 0..max_pages-1 (ascending physical page id):
//     fold page_owner[p], page_logical[p], page_last_used[p] (each int32).
//   page_table_checksum[0] = h.
//
// ---- evicted_count[1] / free_pages[1] ----
//   evicted_count = number of evictions performed during THIS step only
//     (reset to 0 at step start; counts victim evictions from append allocs).
//   free_pages = number of physical pages p in [0,max_pages) with
//     page_owner[p] < 0, counted after the step.
//
// ---- initial / reset state ----
//   page_owner[*] = -1, page_logical[*] = -1, page_last_used[*] = 0,
//   page_data[*] = 0, page_table[*] = -1, length[*]=ring_head[*]=ring_count[*]=0,
//   step_counter = 0.
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

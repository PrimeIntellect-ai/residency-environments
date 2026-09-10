// file: graph_frontier_bfs_state_common.h

#ifndef GRAPH_FRONTIER_BFS_STATE_COMMON_H_
#define GRAPH_FRONTIER_BFS_STATE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define GFBF_ABI_VERSION 1

#define GFBF_MIN_NODES 1
#define GFBF_MAX_NODES 131072
#define GFBF_MIN_EDGES 0
#define GFBF_MAX_EDGES 4194304
#define GFBF_MAX_UPDATES 4096

#define GFBF_FLAG_RELOAD_GRAPH 1

/*
CONTRACT: graph_frontier_bfs_state

Stateful exact-int BFS over a fixed CSR topology with persistent edge weights.

Graph:
  row_offsets[num_nodes + 1] int32
  col_indices[num_edges] int32
  initial_edge_weight[num_edges] int32

An edge is traversable iff persistent edge_weight[e] > 0.

Persistent state:
  edge_weight[num_edges]
  visited[num_nodes]      int32 0/1
  distance[num_nodes]     int32, -1 if unvisited
  current_frontier[]      sorted unique node ids at current_level
  current_level           int32, starts at 0 after source reset
  source                  int32
  graph_initialized       int32
  bfs_initialized         int32

Step order:
  1. If graph is uninitialized, or run.flags has GFBF_FLAG_RELOAD_GRAPH,
     copy initial_edge_weight into persistent edge_weight and clear BFS state.
  2. Apply edge updates:
       update_edge_index[i] in [0,num_edges) gets update_edge_weight[i].
     Invalid update indices are ignored.
  3. If reset_source >= 0, clear BFS and set that source.
     Else if BFS is uninitialized, clear BFS and set run.source.
  4. If any edge update was applied without reset, rederive the BFS state
     from the current source up to the current_level using updated weights.
  5. Expand current_frontier by one BFS level, build sorted-unique
     next_frontier, mark newly visited nodes, and set it as current_frontier.

Expansion:
  For every node u in current_frontier, scan CSR edges in row order.
  For every traversable edge u -> v where v is a valid node and not already
  visited, v becomes a candidate for the next frontier.
  next_frontier is emitted in increasing node id order, therefore sorted unique.

Outputs after the expansion:
  frontier_size[0]          size of next_frontier
  newly_visited_count[0]    same as frontier_size
  next_frontier[0:size]     sorted unique node ids
  FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
    the agent MUST use this exact basis or every checksum fails):
      offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
        the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
      prime        = 1099511628211  (0x100000001B3).
      fold: start h = offset basis; absorb each field's raw bytes little-endian at
        its full declared width (u8=1, u32=4, u64=8, i64=8 two's-complement); per
        byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator / final mix.

  next_frontier_checksum[0] FNV-1a over size then next_frontier entries
  total_visited[0]          count of visited nodes after expansion
  visited_checksum[0]       FNV-1a over num_nodes then every
                             (visited[node], distance[node])
  level_histogram_checksum[0]
                            FNV-1a over current_level, then for level
                            0..current_level the number of nodes at that level

Reset semantics:
  solution_reset clears both graph and BFS initialization flags. The next
  solution_run reloads initial_edge_weight and initializes from run.source or
  run.reset_source.

Rules:
  - solution_init may allocate persistent state.
  - solution_run may not call cudaMalloc/cudaFree.
  - solution_run may launch kernels and use provided workspace.
  - All graded outputs are exact.
*/

struct alignas(8) GfbfProblemSpec {
    int32_t abi_version;
    int32_t max_nodes;
    int32_t max_edges;
    int32_t max_updates;
    int32_t flags;
    int32_t reserved[11];
};

struct alignas(8) GfbfRunSpec {
    int32_t abi_version;
    int32_t num_nodes;
    int32_t num_edges;
    int32_t source;
    int32_t reset_source;
    int32_t update_count;
    int32_t step_id;
    int32_t flags;
    int32_t reserved[8];
};

struct alignas(8) GfbfInputs {
    const int32_t* row_offsets;
    const int32_t* col_indices;
    const int32_t* initial_edge_weight;
    const int32_t* update_edge_index;
    const int32_t* update_edge_weight;
};

struct alignas(8) GfbfOutputs {
    int32_t* frontier_size;
    int32_t* newly_visited_count;
    uint64_t* visited_checksum;
    uint64_t* level_histogram_checksum;
    int32_t* next_frontier;
    uint64_t* next_frontier_checksum;
    int32_t* total_visited;
};

static_assert(sizeof(GfbfProblemSpec) == 64, "GfbfProblemSpec layout drift");
static_assert(sizeof(GfbfRunSpec) == 64, "GfbfRunSpec layout drift");
static_assert(sizeof(GfbfInputs) == 40, "GfbfInputs layout drift");
static_assert(sizeof(GfbfOutputs) == 56, "GfbfOutputs layout drift");

static inline size_t gfbf_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int gfbf_validate_problem_spec(const GfbfProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != GFBF_ABI_VERSION) return 0;
    if (spec->max_nodes < GFBF_MIN_NODES || spec->max_nodes > GFBF_MAX_NODES) return 0;
    if (spec->max_edges < GFBF_MIN_EDGES || spec->max_edges > GFBF_MAX_EDGES) return 0;
    if (spec->max_updates < 0 || spec->max_updates > GFBF_MAX_UPDATES) return 0;
    return 1;
}

static inline int gfbf_validate_run_spec(const GfbfRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != GFBF_ABI_VERSION) return 0;
    if (run->num_nodes < GFBF_MIN_NODES || run->num_nodes > GFBF_MAX_NODES) return 0;
    if (run->num_edges < GFBF_MIN_EDGES || run->num_edges > GFBF_MAX_EDGES) return 0;
    if (run->update_count < 0 || run->update_count > GFBF_MAX_UPDATES) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const GfbfProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const GfbfProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const GfbfRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // GRAPH_FRONTIER_BFS_STATE_COMMON_H_

// PMPP_CANARY_30_42c5f968be -- held-out canary; MUST NOT appear in any submission
// file: graph_frontier_bfs_state_reference.cu
//
// Fast bit-exact GPU reference. The original oracle ran the entire multi-step
// BFS + all checksums in a single <<<1,1>>> thread. This version preserves the
// exact algorithm (identical accumulation order, rounding, node-id ordering,
// FNV byte stream) but executes every O(N)/O(N*L) inner loop in parallel:
//   - graph reload / edge scan / next-frontier build via parallel stream
//     compaction (ascending node-id order == the oracle's emission order),
//   - level histogram via a single parallel pass + atomics,
//   - total_visited via parallel reduction,
//   - visited_checksum via the FNV-1a affine-segment trick (per fixed byte
//     chunk, FNV(chunk,h) = A*(h & ~0xFF) + T[h & 0xFF], A = prime^len).
// The step-to-step chain stays serial (inherent BFS dependency); that floor is
// just per-launch overhead, the same floor a real student solution hits.

#include "graph_frontier_bfs_state_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GFBF_FNV_OFFSET 1469598103934665603ULL
#define GFBF_FNV_PRIME  1099511628211ULL

// Elementwise launch geometry.
#define GFBF_BLK 256
// Values per FNV segment (each int32 field contributes 4 little-endian bytes).
#define GFBF_FNV_VPS 64

struct GfbfReferenceState {
    GfbfProblemSpec spec;

    int32_t* edge_weight;
    int32_t* visited;
    int32_t* distance;
    int32_t* frontier;
    int32_t* candidate;

    int32_t* frontier_size_state;
    int32_t* current_level;
    int32_t* source;
    int32_t* graph_initialized;
    int32_t* bfs_initialized;

    // scratch / orchestration
    int32_t* decision;      // device: [reload, do_reset, needs_rederive, target]
    int32_t* h_decision;    // pinned host mirror
    int32_t* counts;        // level histogram bins
    uint64_t* seg_A;        // per-FNV-segment A = prime^bytes
    uint64_t* seg_T;        // per-FNV-segment T[256]
    int      seg_cap;       // number of segments allocated
};

// ---------------------------------------------------------------------------
// FNV helpers (device, single-thread serial folds for the small streams)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t gfbf_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= GFBF_FNV_PRIME;
    return h;
}

__device__ __forceinline__ uint64_t gfbf_fnv_i32(uint64_t h, int32_t v) {
    const uint32_t u = static_cast<uint32_t>(v);
    h = gfbf_fnv_byte(h, static_cast<uint8_t>(u & 0xFF));
    h = gfbf_fnv_byte(h, static_cast<uint8_t>((u >> 8) & 0xFF));
    h = gfbf_fnv_byte(h, static_cast<uint8_t>((u >> 16) & 0xFF));
    h = gfbf_fnv_byte(h, static_cast<uint8_t>((u >> 24) & 0xFF));
    return h;
}

// ---------------------------------------------------------------------------
// Per-step scalar control: decide reload / reset / rederive, mutate the cheap
// scalar state (graph/bfs flags, source). No edge or O(N) work here.
// ---------------------------------------------------------------------------

__global__ void gfbf_control_kernel(
    int N,
    int E,
    int max_updates,
    int run_source,
    int reset_source,
    int update_count,
    int flags,
    const int32_t* __restrict__ update_edge_index,
    int32_t* __restrict__ edge_weight_unused,
    int32_t* __restrict__ current_level,
    int32_t* __restrict__ source,
    int32_t* __restrict__ graph_initialized,
    int32_t* __restrict__ bfs_initialized,
    int32_t* __restrict__ decision) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    (void)edge_weight_unused;

    const int reload = (graph_initialized[0] == 0) || ((flags & GFBF_FLAG_RELOAD_GRAPH) != 0);
    if (reload) {
        graph_initialized[0] = 1;
        bfs_initialized[0] = 0;
    }

    int clamped = update_count;
    if (clamped > max_updates) clamped = max_updates;

    int applied = 0;
    for (int i = 0; i < clamped; ++i) {
        const int e = update_edge_index[i];
        if (e >= 0 && e < E) applied += 1;
    }

    const int do_reset = (bfs_initialized[0] == 0) || (reset_source >= 0);
    const int needs_rederive = (!do_reset && applied > 0) ? 1 : 0;
    const int target = current_level[0];

    if (do_reset) {
        const int src = reset_source >= 0 ? reset_source : run_source;
        source[0] = src;
        bfs_initialized[0] = 1;
    }

    decision[0] = reload;
    decision[1] = do_reset;
    decision[2] = needs_rederive;
    decision[3] = target;
}

// Parallel graph reload: edge_weight[e] = initial_edge_weight[e].
__global__ void gfbf_reload_kernel(
    int E,
    const int32_t* __restrict__ initial_edge_weight,
    int32_t* __restrict__ edge_weight) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < E) edge_weight[e] = initial_edge_weight[e];
}

// Serial edge updates (single thread => last-write-wins matches the oracle for
// duplicate target indices). update_count is small (<= GFBF_MAX_UPDATES).
__global__ void gfbf_apply_updates_kernel(
    int E,
    int max_updates,
    int update_count,
    const int32_t* __restrict__ update_edge_index,
    const int32_t* __restrict__ update_edge_weight,
    int32_t* __restrict__ edge_weight) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int clamped = update_count;
    if (clamped > max_updates) clamped = max_updates;
    for (int i = 0; i < clamped; ++i) {
        const int e = update_edge_index[i];
        if (e >= 0 && e < E) edge_weight[e] = update_edge_weight[i];
    }
}

// Clear BFS state: visited=0, distance=-1; scalar level/frontier_size=0.
__global__ void gfbf_clear_kernel(
    int N,
    int32_t* __restrict__ visited,
    int32_t* __restrict__ distance,
    int32_t* __restrict__ frontier_size_state,
    int32_t* __restrict__ current_level) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        visited[i] = 0;
        distance[i] = -1;
    }
    if (i == 0) {
        frontier_size_state[0] = 0;
        current_level[0] = 0;
    }
}

// Initialize source (reads source[0], set by control on reset; persisted for
// rederive). Matches gfbf_initialize_source_device.
__global__ void gfbf_init_source_kernel(
    int N,
    int32_t* __restrict__ visited,
    int32_t* __restrict__ distance,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size_state,
    const int32_t* __restrict__ source) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int src = source[0];
    if (src >= 0 && src < N) {
        visited[src] = 1;
        distance[src] = 0;
        frontier[0] = src;
        frontier_size_state[0] = 1;
    } else {
        frontier_size_state[0] = 0;
    }
}

// ---------------------------------------------------------------------------
// One BFS level expansion (mirrors gfbf_expand_once_device exactly).
//   mark:    candidate[v]=1 for every traversable edge u->v with visited[v]==0,
//            u drawn from frontier[0..fsz).
//   compact: emit next frontier in ascending node id v (== oracle order) via a
//            single-block prefix scan; set visited/distance; update scalars.
// candidate is zeroed with cudaMemsetAsync before mark.
// ---------------------------------------------------------------------------

__global__ void gfbf_mark_kernel(
    int N,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const int32_t* __restrict__ edge_weight,
    const int32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier,
    const int32_t* __restrict__ frontier_size_state,
    int32_t* __restrict__ candidate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int fsz = frontier_size_state[0];
    if (i >= fsz) return;

    const int u = frontier[i];
    if (u < 0 || u >= N) return;

    const int row_begin = row_offsets[u];
    const int row_end = row_offsets[u + 1];
    for (int e = row_begin; e < row_end; ++e) {
        if (edge_weight[e] <= 0) continue;
        const int v = col_indices[e];
        if (v >= 0 && v < N && visited[v] == 0) {
            candidate[v] = 1;  // benign race: all writers store 1
        }
    }
}

// Single-block ascending-order stream compaction of candidate&!visited nodes.
__global__ void gfbf_compact_kernel(
    int N,
    int is_final,
    int32_t* __restrict__ visited,
    int32_t* __restrict__ distance,
    int32_t* __restrict__ candidate,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size_state,
    int32_t* __restrict__ current_level,
    int32_t* __restrict__ out_next_frontier,
    int32_t* __restrict__ out_frontier_size,
    int32_t* __restrict__ out_newly_visited_count) {
    extern __shared__ int s_scan[];  // blockDim.x entries
    const int t = threadIdx.x;
    const int nthreads = blockDim.x;
    const int chunk = (N + nthreads - 1) / nthreads;
    const int lo = t * chunk;
    int hi = lo + chunk;
    if (hi > N) hi = N;

    const int next_level = current_level[0] + 1;

    // Phase 1: local count of masked nodes.
    int lc = 0;
    for (int v = lo; v < hi; ++v) {
        if (candidate[v] != 0 && visited[v] == 0) ++lc;
    }
    s_scan[t] = lc;
    __syncthreads();

    // Blelloch-style exclusive scan over blockDim.x (power-of-two blockDim).
    for (int off = 1; off < nthreads; off <<= 1) {
        int val = 0;
        if (t >= off) val = s_scan[t - off];
        __syncthreads();
        if (t >= off) s_scan[t] += val;
        __syncthreads();
    }
    const int base = s_scan[t] - lc;               // exclusive prefix
    const int total = s_scan[nthreads - 1];        // inclusive last == count

    // Phase 2: scatter ascending v; mark visited/distance.
    int pos = base;
    for (int v = lo; v < hi; ++v) {
        if (candidate[v] != 0 && visited[v] == 0) {
            frontier[pos++] = v;
            visited[v] = 1;
            distance[v] = next_level;
        }
    }
    __syncthreads();

    if (t == 0) {
        frontier_size_state[0] = total;
        current_level[0] = next_level;
        if (is_final) {
            out_frontier_size[0] = total;
            out_newly_visited_count[0] = total;
        }
    }

    if (is_final) {
        for (int i = t; i < total; i += nthreads) {
            out_next_frontier[i] = frontier[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Outputs: total_visited, level histogram, visited_checksum, frontier_checksum.
// ---------------------------------------------------------------------------

__global__ void gfbf_total_visited_kernel(
    int N,
    const int32_t* __restrict__ visited,
    int32_t* __restrict__ out_total_visited) {
    __shared__ int s_red[GFBF_BLK];
    const int t = threadIdx.x;
    int acc = 0;
    for (int i = blockIdx.x * blockDim.x + t; i < N; i += blockDim.x * gridDim.x) {
        if (visited[i] != 0) ++acc;
    }
    s_red[t] = acc;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (t < off) s_red[t] += s_red[t + off];
        __syncthreads();
    }
    if (t == 0) atomicAdd(out_total_visited, s_red[0]);
}

// Level histogram: counts[d] = #{ i : distance[i] == d }, 0 <= d <= current_level.
__global__ void gfbf_histogram_kernel(
    int N,
    int current_level,
    const int32_t* __restrict__ distance,
    int32_t* __restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int d = distance[i];
    if (d >= 0 && d <= current_level) atomicAdd(&counts[d], 1);
}

// Serial FNV over current_level then counts[0..current_level] (small stream).
__global__ void gfbf_level_hist_fnv_kernel(
    int current_level,
    const int32_t* __restrict__ counts,
    uint64_t* __restrict__ out_level_histogram_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t h = GFBF_FNV_OFFSET;
    h = gfbf_fnv_i32(h, current_level);
    for (int level = 0; level <= current_level; ++level) {
        h = gfbf_fnv_i32(h, counts[level]);
    }
    out_level_histogram_checksum[0] = h;
}

// Serial FNV over next_count then frontier[0..next_count) (small stream).
__global__ void gfbf_frontier_fnv_kernel(
    const int32_t* __restrict__ frontier,
    const int32_t* __restrict__ frontier_size_state,
    uint64_t* __restrict__ out_next_frontier_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int count = frontier_size_state[0];
    uint64_t h = GFBF_FNV_OFFSET;
    h = gfbf_fnv_i32(h, count);
    for (int i = 0; i < count; ++i) {
        h = gfbf_fnv_i32(h, frontier[i]);
    }
    out_next_frontier_checksum[0] = h;
}

// visited_checksum stream: value[0]=N, value[2k+1]=visited[k], value[2k+2]=distance[k].
__device__ __forceinline__ int32_t gfbf_visited_stream_value(
    int j, int N, const int32_t* visited, const int32_t* distance) {
    if (j == 0) return N;
    const int j2 = j - 1;
    const int k = j2 >> 1;
    return (j2 & 1) == 0 ? visited[k] : distance[k];
}

// Per-segment FNV: block s computes A_s and T_s[t] = FNV(segment, h=t).
__global__ void gfbf_fnv_segment_kernel(
    int M,                 // total int32 values in the stream (2N+1)
    int N,
    const int32_t* __restrict__ visited,
    const int32_t* __restrict__ distance,
    uint64_t* __restrict__ seg_A,
    uint64_t* __restrict__ seg_T) {
    const int s = blockIdx.x;
    const int t = threadIdx.x;  // 0..255, the incoming low-byte value
    const int lo = s * GFBF_FNV_VPS;
    if (lo >= M) return;
    int hi = lo + GFBF_FNV_VPS;
    if (hi > M) hi = M;

    uint64_t h = static_cast<uint64_t>(t);
    for (int j = lo; j < hi; ++j) {
        const int32_t val = gfbf_visited_stream_value(j, N, visited, distance);
        const uint32_t u = static_cast<uint32_t>(val);
        h = gfbf_fnv_byte(h, static_cast<uint8_t>(u & 0xFF));
        h = gfbf_fnv_byte(h, static_cast<uint8_t>((u >> 8) & 0xFF));
        h = gfbf_fnv_byte(h, static_cast<uint8_t>((u >> 16) & 0xFF));
        h = gfbf_fnv_byte(h, static_cast<uint8_t>((u >> 24) & 0xFF));
    }
    seg_T[static_cast<size_t>(s) * 256 + t] = h;

    if (t == 0) {
        const int nbytes = 4 * (hi - lo);
        uint64_t A = 1;
        for (int k = 0; k < nbytes; ++k) A *= GFBF_FNV_PRIME;
        seg_A[s] = A;
    }
}

// Serial combine of P segments: h = A_s*(h & ~0xFF) + T_s[h & 0xFF].
__global__ void gfbf_fnv_combine_kernel(
    int P,
    const uint64_t* __restrict__ seg_A,
    const uint64_t* __restrict__ seg_T,
    uint64_t* __restrict__ out_visited_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t h = GFBF_FNV_OFFSET;
    for (int s = 0; s < P; ++s) {
        const uint64_t low = h & 0xFFULL;
        h = seg_A[s] * (h & ~0xFFULL) + seg_T[static_cast<size_t>(s) * 256 + low];
    }
    out_visited_checksum[0] = h;
}

// ---------------------------------------------------------------------------
// Host orchestration.
// ---------------------------------------------------------------------------

static void gfbf_expand_phase(
    GfbfReferenceState* st,
    const GfbfInputs* in,
    int N,
    int is_final,
    int32_t* out_next_frontier,
    int32_t* out_frontier_size,
    int32_t* out_newly_visited_count,
    cudaStream_t stream) {
    cudaMemsetAsync(st->candidate, 0, sizeof(int32_t) * (size_t)N, stream);

    const int mark_blocks = (N + GFBF_BLK - 1) / GFBF_BLK;
    gfbf_mark_kernel<<<mark_blocks, GFBF_BLK, 0, stream>>>(
        N, in->row_offsets, in->col_indices, st->edge_weight,
        st->visited, st->frontier, st->frontier_size_state, st->candidate);

    const int cblk = 1024;
    gfbf_compact_kernel<<<1, cblk, sizeof(int) * cblk, stream>>>(
        N, is_final, st->visited, st->distance, st->candidate, st->frontier,
        st->frontier_size_state, st->current_level,
        out_next_frontier, out_frontier_size, out_newly_visited_count);
}

static cudaError_t gfbf_reference_reset_state(GfbfReferenceState* st, cudaStream_t stream);

extern "C" size_t solution_workspace_bytes(const GfbfProblemSpec* spec) {
    if (!gfbf_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const GfbfProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!gfbf_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    GfbfReferenceState* st =
        static_cast<GfbfReferenceState*>(malloc(sizeof(GfbfReferenceState)));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(GfbfReferenceState));
    memcpy(&st->spec, spec, sizeof(GfbfProblemSpec));

    const size_t Nn = (size_t)spec->max_nodes;
    const size_t Ee = (size_t)spec->max_edges;
    const int M_max = 2 * spec->max_nodes + 1;
    const int seg_cap = (M_max + GFBF_FNV_VPS - 1) / GFBF_FNV_VPS;
    st->seg_cap = seg_cap;
    // level histogram bins; current_level can exceed max_nodes over long runs.
    const size_t counts_cap = (Nn > (1u << 20)) ? Nn + 2 : (size_t)(1u << 20);

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&st->edge_weight, sizeof(int32_t) * (Ee ? Ee : 1)); if (err) goto fail;
    err = cudaMalloc((void**)&st->visited, sizeof(int32_t) * Nn); if (err) goto fail;
    err = cudaMalloc((void**)&st->distance, sizeof(int32_t) * Nn); if (err) goto fail;
    err = cudaMalloc((void**)&st->frontier, sizeof(int32_t) * Nn); if (err) goto fail;
    err = cudaMalloc((void**)&st->candidate, sizeof(int32_t) * Nn); if (err) goto fail;
    err = cudaMalloc((void**)&st->frontier_size_state, sizeof(int32_t)); if (err) goto fail;
    err = cudaMalloc((void**)&st->current_level, sizeof(int32_t)); if (err) goto fail;
    err = cudaMalloc((void**)&st->source, sizeof(int32_t)); if (err) goto fail;
    err = cudaMalloc((void**)&st->graph_initialized, sizeof(int32_t)); if (err) goto fail;
    err = cudaMalloc((void**)&st->bfs_initialized, sizeof(int32_t)); if (err) goto fail;
    err = cudaMalloc((void**)&st->decision, sizeof(int32_t) * 4); if (err) goto fail;
    err = cudaMalloc((void**)&st->counts, sizeof(int32_t) * counts_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->seg_A, sizeof(uint64_t) * (size_t)seg_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->seg_T, sizeof(uint64_t) * (size_t)seg_cap * 256); if (err) goto fail;
    err = cudaMallocHost((void**)&st->h_decision, sizeof(int32_t) * 4); if (err) goto fail;

    err = gfbf_reference_reset_state(st, stream); if (err) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->edge_weight) cudaFree(st->edge_weight);
    if (st->visited) cudaFree(st->visited);
    if (st->distance) cudaFree(st->distance);
    if (st->frontier) cudaFree(st->frontier);
    if (st->candidate) cudaFree(st->candidate);
    if (st->frontier_size_state) cudaFree(st->frontier_size_state);
    if (st->current_level) cudaFree(st->current_level);
    if (st->source) cudaFree(st->source);
    if (st->graph_initialized) cudaFree(st->graph_initialized);
    if (st->bfs_initialized) cudaFree(st->bfs_initialized);
    if (st->decision) cudaFree(st->decision);
    if (st->counts) cudaFree(st->counts);
    if (st->seg_A) cudaFree(st->seg_A);
    if (st->seg_T) cudaFree(st->seg_T);
    if (st->h_decision) cudaFreeHost(st->h_decision);
    free(st);
    return err;
}

static cudaError_t gfbf_reference_reset_state(GfbfReferenceState* st, cudaStream_t stream) {
    cudaError_t err;
    err = cudaMemsetAsync(st->edge_weight, 0, sizeof(int32_t) * (size_t)st->spec.max_edges, stream); if (err) return err;
    err = cudaMemsetAsync(st->visited, 0, sizeof(int32_t) * (size_t)st->spec.max_nodes, stream); if (err) return err;
    err = cudaMemsetAsync(st->distance, 0xff, sizeof(int32_t) * (size_t)st->spec.max_nodes, stream); if (err) return err;
    err = cudaMemsetAsync(st->frontier, 0, sizeof(int32_t) * (size_t)st->spec.max_nodes, stream); if (err) return err;
    err = cudaMemsetAsync(st->candidate, 0, sizeof(int32_t) * (size_t)st->spec.max_nodes, stream); if (err) return err;
    err = cudaMemsetAsync(st->frontier_size_state, 0, sizeof(int32_t), stream); if (err) return err;
    err = cudaMemsetAsync(st->current_level, 0, sizeof(int32_t), stream); if (err) return err;
    err = cudaMemsetAsync(st->source, 0xff, sizeof(int32_t), stream); if (err) return err;
    err = cudaMemsetAsync(st->graph_initialized, 0, sizeof(int32_t), stream); if (err) return err;
    err = cudaMemsetAsync(st->bfs_initialized, 0, sizeof(int32_t), stream); if (err) return err;
    return cudaSuccess;
}

extern "C" cudaError_t solution_run(
    void* state,
    const GfbfRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;

    if (!state || !gfbf_validate_run_spec(run) || !inputs_void || !outputs_void) {
        return cudaErrorInvalidValue;
    }
    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    GfbfReferenceState* st = static_cast<GfbfReferenceState*>(state);
    const GfbfInputs* in = static_cast<const GfbfInputs*>(inputs_void);
    GfbfOutputs* out = static_cast<GfbfOutputs*>(outputs_void);

    const int N = run->num_nodes;
    const int E = run->num_edges;

    if (N > st->spec.max_nodes || E > st->spec.max_edges ||
        run->update_count > st->spec.max_updates) {
        return cudaErrorInvalidValue;
    }
    if (!in->row_offsets || !in->col_indices || !in->initial_edge_weight) {
        return cudaErrorInvalidValue;
    }
    if (run->update_count > 0 && (!in->update_edge_index || !in->update_edge_weight)) {
        return cudaErrorInvalidValue;
    }
    if (!out->frontier_size || !out->newly_visited_count ||
        !out->visited_checksum || !out->level_histogram_checksum ||
        !out->next_frontier || !out->next_frontier_checksum || !out->total_visited) {
        return cudaErrorInvalidValue;
    }

    // 1) scalar control decisions.
    gfbf_control_kernel<<<1, 1, 0, stream>>>(
        N, E, st->spec.max_updates, run->source, run->reset_source,
        run->update_count, run->flags, in->update_edge_index, st->edge_weight,
        st->current_level, st->source, st->graph_initialized,
        st->bfs_initialized, st->decision);

    cudaError_t err = cudaMemcpyAsync(st->h_decision, st->decision,
        sizeof(int32_t) * 4, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return err;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return err;

    const int reload = st->h_decision[0];
    const int do_reset = st->h_decision[1];
    const int needs_rederive = st->h_decision[2];
    const int target = st->h_decision[3];

    // 2) reload (parallel) then serial updates.
    if (reload) {
        const int eblocks = (E + GFBF_BLK - 1) / GFBF_BLK;
        if (eblocks > 0) {
            gfbf_reload_kernel<<<eblocks, GFBF_BLK, 0, stream>>>(
                E, in->initial_edge_weight, st->edge_weight);
        }
    }
    gfbf_apply_updates_kernel<<<1, 1, 0, stream>>>(
        E, st->spec.max_updates, run->update_count,
        in->update_edge_index, in->update_edge_weight, st->edge_weight);

    // 3) reset / rederive.
    const int nblocks = (N + GFBF_BLK - 1) / GFBF_BLK;
    if (do_reset) {
        gfbf_clear_kernel<<<nblocks, GFBF_BLK, 0, stream>>>(
            N, st->visited, st->distance, st->frontier_size_state, st->current_level);
        gfbf_init_source_kernel<<<1, 1, 0, stream>>>(
            N, st->visited, st->distance, st->frontier, st->frontier_size_state, st->source);
    } else if (needs_rederive) {
        gfbf_clear_kernel<<<nblocks, GFBF_BLK, 0, stream>>>(
            N, st->visited, st->distance, st->frontier_size_state, st->current_level);
        gfbf_init_source_kernel<<<1, 1, 0, stream>>>(
            N, st->visited, st->distance, st->frontier, st->frontier_size_state, st->source);
        for (int level = 0; level < target; ++level) {
            gfbf_expand_phase(st, in, N, /*is_final=*/0, nullptr, nullptr, nullptr, stream);
        }
    }

    // 4) final expansion (writes frontier outputs).
    gfbf_expand_phase(st, in, N, /*is_final=*/1,
        out->next_frontier, out->frontier_size, out->newly_visited_count, stream);

    // final current_level after this step's expansion.
    const int final_level = do_reset ? 1 : (target + 1);

    // 5a) total_visited (parallel reduction).
    cudaMemsetAsync(out->total_visited, 0, sizeof(int32_t), stream);
    int tv_blocks = (N + GFBF_BLK - 1) / GFBF_BLK;
    if (tv_blocks > 256) tv_blocks = 256;
    gfbf_total_visited_kernel<<<tv_blocks, GFBF_BLK, 0, stream>>>(
        N, st->visited, out->total_visited);

    // 5b) next_frontier checksum (small serial stream).
    gfbf_frontier_fnv_kernel<<<1, 1, 0, stream>>>(
        st->frontier, st->frontier_size_state, out->next_frontier_checksum);

    // 5c) level histogram checksum (parallel bin + small serial fold).
    cudaMemsetAsync(st->counts, 0, sizeof(int32_t) * (size_t)(final_level + 1), stream);
    gfbf_histogram_kernel<<<nblocks, GFBF_BLK, 0, stream>>>(
        N, final_level, st->distance, st->counts);
    gfbf_level_hist_fnv_kernel<<<1, 1, 0, stream>>>(
        final_level, st->counts, out->level_histogram_checksum);

    // 5d) visited checksum (parallel FNV affine segments + serial combine).
    const int M = 2 * N + 1;
    const int P = (M + GFBF_FNV_VPS - 1) / GFBF_FNV_VPS;
    gfbf_fnv_segment_kernel<<<P, 256, 0, stream>>>(
        M, N, st->visited, st->distance, st->seg_A, st->seg_T);
    gfbf_fnv_combine_kernel<<<1, 1, 0, stream>>>(
        P, st->seg_A, st->seg_T, out->visited_checksum);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return gfbf_reference_reset_state(static_cast<GfbfReferenceState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    GfbfReferenceState* st = static_cast<GfbfReferenceState*>(state);
    if (st->edge_weight) cudaFree(st->edge_weight);
    if (st->visited) cudaFree(st->visited);
    if (st->distance) cudaFree(st->distance);
    if (st->frontier) cudaFree(st->frontier);
    if (st->candidate) cudaFree(st->candidate);
    if (st->frontier_size_state) cudaFree(st->frontier_size_state);
    if (st->current_level) cudaFree(st->current_level);
    if (st->source) cudaFree(st->source);
    if (st->graph_initialized) cudaFree(st->graph_initialized);
    if (st->bfs_initialized) cudaFree(st->bfs_initialized);
    if (st->decision) cudaFree(st->decision);
    if (st->counts) cudaFree(st->counts);
    if (st->seg_A) cudaFree(st->seg_A);
    if (st->seg_T) cudaFree(st->seg_T);
    if (st->h_decision) cudaFreeHost(st->h_decision);
    free(st);
}

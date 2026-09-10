// PMPP_CANARY_36_f7cdb512eb -- held-out canary; MUST NOT appear in any submission
// file: radix_prefix_cache_tree_reference.cu
//
// Reference GPU implementation. Persistent radix tree stored as flat SoA
// device arrays. A single-threaded kernel replays the whole op batch. Children
// are tracked with an intrusive singly-linked list per node (head/next arrays),
// which is a structurally different representation from the CPU oracle's
// std::map; segments live in a shared token pool with (offset,len) views so a
// split only re-points the suffix view without copying.

#include "radix_prefix_cache_tree_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct RpctRefState {
    RpctProblemSpec spec;

    // node SoA, index by node id; index 0 == root.
    int32_t* parent;        // [node_cap]
    int32_t* ref_count;     // [node_cap]
    int64_t* lru;           // [node_cap]
    int32_t* seg_off;       // [node_cap] offset into seg_pool
    int32_t* seg_len;       // [node_cap]
    int32_t* alive;         // [node_cap] 0/1
    int32_t* child_head;    // [node_cap] first child slot or -1

    // child linked-list slots (one slot per non-root node; slot id == node id).
    int32_t* child_next;    // [node_cap] next sibling slot or -1

    // request -> terminal node id; -1 means not live.
    int32_t* req_terminal;  // [max_requests]

    // segment token pool (append-only within a step run; reset clears).
    int32_t* seg_pool;      // [seg_pool_cap]

    // scalars (single-element device arrays).
    int32_t* clock_lo;      // low/high halves of int64 clock kept as int64 below
    int64_t* clock64;       // [1]
    int32_t* next_node_id;  // [1]
    int32_t* seg_pool_used; // [1]

    int32_t node_cap;
    int32_t seg_pool_cap;
};

__device__ __forceinline__ uint64_t rpct_ref_fnv_byte(uint64_t h, uint8_t b) {
    h ^= (uint64_t)b;
    h *= 1099511628211ULL;
    return h;
}

__device__ void rpct_ref_fnv_bytes(uint64_t* h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) v = rpct_ref_fnv_byte(v, b[i]);
    *h = v;
}

// Walk the child list of `node` to find the child whose first segment token
// equals `first`. Returns child id or -1.
__device__ int32_t rpct_ref_find_child(
    const int32_t* __restrict__ child_head,
    const int32_t* __restrict__ child_next,
    const int32_t* __restrict__ seg_off,
    const int32_t* __restrict__ seg_pool,
    int32_t node,
    int32_t first) {
    int32_t c = child_head[node];
    while (c != -1) {
        if (seg_pool[seg_off[c]] == first) return c;
        c = child_next[c];
    }
    return -1;
}

__device__ void rpct_ref_add_child(
    int32_t* __restrict__ child_head,
    int32_t* __restrict__ child_next,
    int32_t parent,
    int32_t child) {
    child_next[child] = child_head[parent];
    child_head[parent] = child;
}

__device__ void rpct_ref_remove_child(
    int32_t* __restrict__ child_head,
    int32_t* __restrict__ child_next,
    int32_t parent,
    int32_t child) {
    int32_t c = child_head[parent];
    int32_t prev = -1;
    while (c != -1) {
        if (c == child) {
            if (prev == -1) child_head[parent] = child_next[c];
            else child_next[prev] = child_next[c];
            child_next[c] = -1;
            return;
        }
        prev = c;
        c = child_next[c];
    }
}

__global__ void rpct_reference_step_kernel(
    int op_count,
    int node_cap,
    int seg_pool_cap,
    int max_requests,
    const int32_t* __restrict__ op_kind,
    const int32_t* __restrict__ op_arg_a,
    const int32_t* __restrict__ op_token_offset,
    const int32_t* __restrict__ op_token_len,
    const int32_t* __restrict__ op_tokens,
    int32_t* __restrict__ parent,
    int32_t* __restrict__ ref_count,
    int64_t* __restrict__ lru,
    int32_t* __restrict__ seg_off,
    int32_t* __restrict__ seg_len,
    int32_t* __restrict__ alive,
    int32_t* __restrict__ child_head,
    int32_t* __restrict__ child_next,
    int32_t* __restrict__ req_terminal,
    int32_t* __restrict__ seg_pool,
    int64_t* __restrict__ clock64,
    int32_t* __restrict__ next_node_id,
    int32_t* __restrict__ seg_pool_used,
    int32_t* __restrict__ out_matched_prefix_len,
    int32_t* __restrict__ out_num_nodes,
    int32_t* __restrict__ out_num_tokens_cached,
    int32_t* __restrict__ out_num_evicted_nodes,
    int32_t* __restrict__ out_evicted_tokens,
    uint64_t* __restrict__ out_tree_checksum,
    uint64_t* __restrict__ out_state_checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int64_t clk = clock64[0];
    int32_t nid = next_node_id[0];
    int32_t pool_used = seg_pool_used[0];

    for (int i = 0; i < op_count; ++i) {
        const int32_t kind = op_kind[i];
        int32_t matched = -1;
        int32_t evict_nodes = 0;
        int32_t evict_tokens = 0;

        if (kind == RPCT_OP_INSERT) {
            clk += 1;
            const int64_t now = clk;
            const int32_t req = op_arg_a[i];
            const int32_t off = op_token_offset[i];
            const int32_t len = op_token_len[i];

            int32_t cur = 0;
            int32_t pos = 0;

            while (pos < len) {
                const int32_t first = op_tokens[off + pos];
                const int32_t child = rpct_ref_find_child(
                    child_head, child_next, seg_off, seg_pool, cur, first);
                if (child == -1) break;

                const int32_t cseg_off = seg_off[child];
                const int32_t cseg_len = seg_len[child];
                int32_t k = 0;
                while (k < cseg_len && pos + k < len &&
                       seg_pool[cseg_off + k] == op_tokens[off + pos + k]) {
                    ++k;
                }

                if (k == cseg_len) {
                    cur = child;
                    pos += cseg_len;
                    continue;
                }

                // split child at k. allocate new internal node P.
                const int32_t P = nid++;
                parent[P] = cur;
                ref_count[P] = ref_count[child];
                lru[P] = lru[child];
                alive[P] = 1;
                child_head[P] = -1;
                child_next[P] = -1;
                // P segment = child.segment[0..k): reuse the same pool prefix.
                seg_off[P] = cseg_off;
                seg_len[P] = k;
                // child keeps suffix [k..)
                seg_off[child] = cseg_off + k;
                seg_len[child] = cseg_len - k;
                // rewire: remove child from cur, add P to cur, add child to P.
                rpct_ref_remove_child(child_head, child_next, cur, child);
                rpct_ref_add_child(child_head, child_next, cur, P);
                parent[child] = P;
                rpct_ref_add_child(child_head, child_next, P, child);

                cur = P;
                pos += k;
                break;
            }

            matched = pos;

            if (pos < len) {
                const int32_t L = nid++;
                const int32_t rem = len - pos;
                const int32_t base = pool_used;
                for (int32_t t = 0; t < rem; ++t) {
                    seg_pool[base + t] = op_tokens[off + pos + t];
                }
                pool_used += rem;
                parent[L] = cur;
                ref_count[L] = 0;
                lru[L] = now;
                alive[L] = 1;
                child_head[L] = -1;
                child_next[L] = -1;
                seg_off[L] = base;
                seg_len[L] = rem;
                rpct_ref_add_child(child_head, child_next, cur, L);
                cur = L;
            }

            int32_t walk = cur;
            while (walk != 0) {
                ref_count[walk] += 1;
                lru[walk] = now;
                walk = parent[walk];
            }

            if (req >= 0 && req < max_requests) req_terminal[req] = cur;

        } else if (kind == RPCT_OP_RELEASE) {
            clk += 1;
            const int32_t req = op_arg_a[i];
            if (req >= 0 && req < max_requests && req_terminal[req] != -1) {
                int32_t walk = req_terminal[req];
                while (walk != 0) {
                    if (ref_count[walk] > 0) ref_count[walk] -= 1;
                    walk = parent[walk];
                }
                req_terminal[req] = -1;
            }

        } else {  // EVICT
            clk += 1;
            const int32_t target = op_arg_a[i];
            int32_t en = 0;
            int32_t et = 0;
            if (target > 0) {
                while (et < target) {
                    int32_t best = -1;
                    int64_t best_ts = 0;
                    for (int32_t n = 1; n < nid; ++n) {
                        if (!alive[n]) continue;
                        if (ref_count[n] != 0) continue;
                        if (child_head[n] != -1) continue;  // must be leaf
                        const int64_t ts = lru[n];
                        if (best == -1 || ts < best_ts ||
                            (ts == best_ts && n < best)) {
                            best = n;
                            best_ts = ts;
                        }
                    }
                    if (best == -1) break;
                    const int32_t par = parent[best];
                    et += seg_len[best];
                    en += 1;
                    rpct_ref_remove_child(child_head, child_next, par, best);
                    alive[best] = 0;
                }
            }
            evict_nodes = en;
            evict_tokens = et;
        }

        // recompute counts + checksums over alive non-root nodes by ascending id
        int32_t nn = 0;
        int32_t nt = 0;
        for (int32_t n = 1; n < nid; ++n) {
            if (alive[n]) {
                ++nn;
                nt += seg_len[n];
            }
        }

        uint64_t th = 1469598103934665603ULL;
        rpct_ref_fnv_bytes(&th, &nn, sizeof(int32_t));
        for (int32_t n = 1; n < nid; ++n) {
            if (!alive[n]) continue;
            int32_t id = n;
            int32_t pid = parent[n];
            int32_t rc = ref_count[n];
            int32_t sl = seg_len[n];
            rpct_ref_fnv_bytes(&th, &id, sizeof(int32_t));
            rpct_ref_fnv_bytes(&th, &pid, sizeof(int32_t));
            rpct_ref_fnv_bytes(&th, &rc, sizeof(int32_t));
            rpct_ref_fnv_bytes(&th, &sl, sizeof(int32_t));
            const int32_t so = seg_off[n];
            for (int32_t t = 0; t < sl; ++t) {
                int32_t tok = seg_pool[so + t];
                rpct_ref_fnv_bytes(&th, &tok, sizeof(int32_t));
            }
        }

        uint64_t sh = 1469598103934665603ULL;
        rpct_ref_fnv_bytes(&sh, &clk, sizeof(int64_t));
        rpct_ref_fnv_bytes(&sh, &nn, sizeof(int32_t));
        rpct_ref_fnv_bytes(&sh, &nt, sizeof(int32_t));
        rpct_ref_fnv_bytes(&sh, &nid, sizeof(int32_t));
        rpct_ref_fnv_bytes(&sh, &th, sizeof(uint64_t));

        out_matched_prefix_len[i] = matched;
        out_num_nodes[i] = nn;
        out_num_tokens_cached[i] = nt;
        out_num_evicted_nodes[i] = evict_nodes;
        out_evicted_tokens[i] = evict_tokens;
        out_tree_checksum[i] = th;
        out_state_checksum[i] = sh;
    }

    clock64[0] = clk;
    next_node_id[0] = nid;
    seg_pool_used[0] = pool_used;
}

static cudaError_t rpct_ref_reset_state(RpctRefState* st, cudaStream_t stream) {
    cudaError_t err;
    err = cudaMemsetAsync(st->parent, 0, sizeof(int32_t) * (size_t)st->node_cap, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->ref_count, 0, sizeof(int32_t) * (size_t)st->node_cap, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->lru, 0, sizeof(int64_t) * (size_t)st->node_cap, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->seg_off, 0, sizeof(int32_t) * (size_t)st->node_cap, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->seg_len, 0, sizeof(int32_t) * (size_t)st->node_cap, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->alive, 0, sizeof(int32_t) * (size_t)st->node_cap, stream);
    if (err != cudaSuccess) return err;
    // child_head / child_next = -1
    err = cudaMemsetAsync(st->child_head, 0xff, sizeof(int32_t) * (size_t)st->node_cap, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemsetAsync(st->child_next, 0xff, sizeof(int32_t) * (size_t)st->node_cap, stream);
    if (err != cudaSuccess) return err;
    // req_terminal = -1
    err = cudaMemsetAsync(st->req_terminal, 0xff, sizeof(int32_t) * (size_t)st->spec.max_requests, stream);
    if (err != cudaSuccess) return err;

    // root: alive[0]=1, parent[0]=-1, rest 0. clock=0, next_node_id=1, pool_used=0.
    int32_t one = 1;
    int32_t neg = -1;
    int32_t zero = 0;
    int64_t zero64 = 0;
    err = cudaMemcpyAsync(&st->alive[0], &one, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(&st->parent[0], &neg, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(st->clock64, &zero64, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(st->next_node_id, &one, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(st->seg_pool_used, &zero, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

extern "C" size_t solution_workspace_bytes(const RpctProblemSpec* spec) {
    if (!rpct_validate_problem_spec(spec)) return 0;
    return 128;
}

extern "C" cudaError_t solution_init(
    const RpctProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!rpct_validate_problem_spec(spec) || !state_out) return cudaErrorInvalidValue;

    RpctRefState* st = (RpctRefState*)malloc(sizeof(RpctRefState));
    if (!st) return cudaErrorMemoryAllocation;
    memset(st, 0, sizeof(RpctRefState));
    memcpy(&st->spec, spec, sizeof(RpctProblemSpec));

    st->node_cap = spec->max_nodes;
    // pool must hold all original tokens plus split prefixes never grow it
    // (splits reuse existing pool ranges). Leaves append remainders; total new
    // tokens across a fresh run bounded by max_tokens. Give generous headroom.
    st->seg_pool_cap = spec->max_tokens > 0 ? spec->max_tokens : 1;

    cudaError_t err;
    err = cudaMalloc((void**)&st->parent, sizeof(int32_t) * (size_t)st->node_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->ref_count, sizeof(int32_t) * (size_t)st->node_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->lru, sizeof(int64_t) * (size_t)st->node_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->seg_off, sizeof(int32_t) * (size_t)st->node_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->seg_len, sizeof(int32_t) * (size_t)st->node_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->alive, sizeof(int32_t) * (size_t)st->node_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->child_head, sizeof(int32_t) * (size_t)st->node_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->child_next, sizeof(int32_t) * (size_t)st->node_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->req_terminal, sizeof(int32_t) * (size_t)st->spec.max_requests); if (err) goto fail;
    err = cudaMalloc((void**)&st->seg_pool, sizeof(int32_t) * (size_t)st->seg_pool_cap); if (err) goto fail;
    err = cudaMalloc((void**)&st->clock64, sizeof(int64_t)); if (err) goto fail;
    err = cudaMalloc((void**)&st->next_node_id, sizeof(int32_t)); if (err) goto fail;
    err = cudaMalloc((void**)&st->seg_pool_used, sizeof(int32_t)); if (err) goto fail;

    err = rpct_ref_reset_state(st, stream);
    if (err) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->parent) cudaFree(st->parent);
    if (st->ref_count) cudaFree(st->ref_count);
    if (st->lru) cudaFree(st->lru);
    if (st->seg_off) cudaFree(st->seg_off);
    if (st->seg_len) cudaFree(st->seg_len);
    if (st->alive) cudaFree(st->alive);
    if (st->child_head) cudaFree(st->child_head);
    if (st->child_next) cudaFree(st->child_next);
    if (st->req_terminal) cudaFree(st->req_terminal);
    if (st->seg_pool) cudaFree(st->seg_pool);
    if (st->clock64) cudaFree(st->clock64);
    if (st->next_node_id) cudaFree(st->next_node_id);
    if (st->seg_pool_used) cudaFree(st->seg_pool_used);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const RpctRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    (void)workspace;
    if (!state || !rpct_validate_run_spec(run) || !inputs_void || !outputs_void)
        return cudaErrorInvalidValue;
    if (workspace_bytes < 128) return cudaErrorInvalidValue;

    RpctRefState* st = (RpctRefState*)state;
    const RpctInputs* in = (const RpctInputs*)inputs_void;
    RpctOutputs* out = (RpctOutputs*)outputs_void;

    if (run->op_count > st->spec.max_ops) return cudaErrorInvalidValue;
    if (run->total_op_tokens > st->spec.max_op_tokens) return cudaErrorInvalidValue;

    if (run->op_count > 0) {
        if (!in->op_kind || !in->op_arg_a || !in->op_token_offset || !in->op_token_len)
            return cudaErrorInvalidValue;
        if (!out->matched_prefix_len || !out->num_nodes || !out->num_tokens_cached ||
            !out->num_evicted_nodes || !out->evicted_tokens || !out->tree_checksum ||
            !out->state_checksum)
            return cudaErrorInvalidValue;
    }

    if (run->op_count == 0) return cudaSuccess;

    rpct_reference_step_kernel<<<1, 1, 0, stream>>>(
        run->op_count,
        st->node_cap,
        st->seg_pool_cap,
        st->spec.max_requests,
        in->op_kind,
        in->op_arg_a,
        in->op_token_offset,
        in->op_token_len,
        in->op_tokens,
        st->parent,
        st->ref_count,
        st->lru,
        st->seg_off,
        st->seg_len,
        st->alive,
        st->child_head,
        st->child_next,
        st->req_terminal,
        st->seg_pool,
        st->clock64,
        st->next_node_id,
        st->seg_pool_used,
        out->matched_prefix_len,
        out->num_nodes,
        out->num_tokens_cached,
        out->num_evicted_nodes,
        out->evicted_tokens,
        out->tree_checksum,
        out->state_checksum);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(void* state, cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return rpct_ref_reset_state((RpctRefState*)state, stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;
    RpctRefState* st = (RpctRefState*)state;
    if (st->parent) cudaFree(st->parent);
    if (st->ref_count) cudaFree(st->ref_count);
    if (st->lru) cudaFree(st->lru);
    if (st->seg_off) cudaFree(st->seg_off);
    if (st->seg_len) cudaFree(st->seg_len);
    if (st->alive) cudaFree(st->alive);
    if (st->child_head) cudaFree(st->child_head);
    if (st->child_next) cudaFree(st->child_next);
    if (st->req_terminal) cudaFree(st->req_terminal);
    if (st->seg_pool) cudaFree(st->seg_pool);
    if (st->clock64) cudaFree(st->clock64);
    if (st->next_node_id) cudaFree(st->next_node_id);
    if (st->seg_pool_used) cudaFree(st->seg_pool_used);
    free(st);
}

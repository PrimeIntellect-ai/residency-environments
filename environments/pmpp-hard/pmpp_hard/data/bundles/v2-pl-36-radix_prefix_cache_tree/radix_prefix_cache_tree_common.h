// file: radix_prefix_cache_tree_common.h

#ifndef RADIX_PREFIX_CACHE_TREE_COMMON_H_
#define RADIX_PREFIX_CACHE_TREE_COMMON_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define RPCT_ABI_VERSION 1

// Capacity limits (problem-spec bounds).
#define RPCT_MIN_NODES 2
#define RPCT_MAX_NODES 131072
#define RPCT_MAX_TOKENS_TOTAL 8388608
#define RPCT_MAX_OPS 4096
#define RPCT_MAX_REQUESTS 65536
#define RPCT_MAX_OP_TOKENS 65536

// Op kinds.
#define RPCT_OP_INSERT 0
#define RPCT_OP_RELEASE 1
#define RPCT_OP_EVICT 2

/*
CONTRACT: radix_prefix_cache_tree

A stateful radix (prefix) tree over sequences of int32 tokens, modeled on
SGLang RadixAttention prefix caching: shared-prefix token blocks are reused
across requests via a reference-counted radix tree, with LRU eviction of
unreferenced leaf nodes.

============================ TREE MODEL ============================

The tree always contains a ROOT node with id 0. The root carries an EMPTY
token segment, is never counted, never ref-counted, and is never evicted.

Every non-root node N has:
  - id            : monotonically assigned int32 (root is 0; first allocated
                    node is 1, then 2, ...). Ids are NEVER reused, even after
                    eviction (the monotonic counter only increases).
  - parent_id     : id of N's parent.
  - ref_count     : int32 >= 0. Number of live requests whose token path
                    passes through N.
  - lru_timestamp : int64 logical clock value of the most recent op that
                    "touched" N (see clock rules).
  - segment       : the non-empty list of tokens labeling the edge from
                    parent(N) to N. segment_len = number of tokens.

Radix invariant: among the children of any node, no two child segments share
the same first token. A node is a LEAF iff it has zero children.

num_tokens_cached = sum of segment_len over all non-root nodes currently in
the tree.

num_nodes = number of non-root nodes currently in the tree (root excluded).

========================= LOGICAL CLOCK =========================

A single int64 logical clock `clock` persists in state. It starts at 0 after
init/reset. At the START of processing EACH op in a step (INSERT, RELEASE, or
EVICT), clock is incremented by 1 BEFORE that op does anything. The resulting
value is "the current clock" used to stamp lru_timestamp for nodes touched by
that op. (RELEASE and EVICT also consume a tick even though RELEASE does not
re-stamp matched nodes and EVICT stamps nothing.)

========================= OPS, IN ORDER =========================

A step supplies an ordered list of ops (op_count of them). They are applied
strictly in array order. Op i has:
  op_kind[i], op_arg_a[i], op_arg_b[i], op_token_offset[i], op_token_len[i]
where op_token_* index into the flat op_tokens[] array (only used by INSERT).

--- INSERT(request_id = op_arg_a, tokens = op_tokens[offset .. offset+len)) ---

  Pre: request_id must not currently be live (have an outstanding terminal
  node). Tokens length len >= 1.

  1. Longest-prefix match. Starting at root, repeatedly: among the current
     node's children find the child whose segment[0] == tokens[pos] (the next
     unmatched token). If none, stop. Otherwise compare tokens[pos..] against
     that child's segment token by token:
       - If the entire child segment matches a prefix of the remaining tokens,
         descend into the child (pos += segment_len) and continue matching.
       - If they diverge mid-segment after k>=1 matched tokens (k < segment_len),
         SPLIT the child: create a NEW node P (fresh id) inserted BETWEEN the
         child's parent and the child. P takes segment = child.segment[0..k);
         the child keeps segment = child.segment[k..). P.parent = child.parent;
         child.parent = P. P.ref_count = child.ref_count (inherited).
         P.lru_timestamp = child.lru_timestamp (inherited; restamped below if
         touched). Now pos += k and matching stops at P (no further child can
         extend the match because tokens diverged inside this edge).

     matched_prefix_len for this INSERT = pos at the moment matching stops,
     i.e. the number of tokens matched against PRE-EXISTING tree content
     (counting tokens consumed by a split, since those existed before).

  2. Append remainder. If pos < len, create exactly ONE new LEAF node L (fresh
     id) as a child of the node where matching stopped, with
     segment = tokens[pos..len). (RadixAttention stores the whole remainder on
     a single edge.) The request's terminal node is L. If pos == len (the full
     sequence already existed), the terminal node is the node where matching
     stopped (which is a node boundary by construction).

  3. Ref-count + touch. Increment ref_count by 1 on EVERY non-root node on the
     path from root to the terminal node (i.e. the chain of ancestors of the
     terminal node, plus the terminal node itself, excluding root). Set
     lru_timestamp = current clock on each of those same nodes. Newly created
     nodes (the split node P inherits an existing ref_count BEFORE this add;
     the leaf L starts at ref_count 0 BEFORE this add) participate in this
     single +1 just like existing path nodes.

     Record terminal node id for request_id (overwriting any stale mapping).

  Output for this op: matched_prefix_len (as defined in step 1). For non-INSERT
  ops matched_prefix_len is reported as -1.

--- RELEASE(request_id = op_arg_a) ---

  If request_id is not currently live, this op is a no-op (still consumes a
  clock tick). Otherwise, walk from the request's recorded terminal node up
  through parent pointers to (but excluding) root, decrementing ref_count by 1
  on each visited node. ref_count never goes below 0 (it cannot, by
  construction). RELEASE does NOT modify lru_timestamp and does NOT delete any
  node (eviction is separate). The request becomes not-live.

  Output: matched_prefix_len = -1.

--- EVICT(num_tokens_target = op_arg_a) ---

  Reclaim tokens by deleting unreferenced LEAF nodes in LRU order until at
  least num_tokens_target tokens have been reclaimed, or no evictable node
  remains.

  An evictable node is a non-root node with ref_count == 0 AND zero children
  (a leaf). Repeatedly:
    - Among all currently-evictable nodes, pick the one with the SMALLEST
      lru_timestamp; ties broken by SMALLEST node id.
    - If none exists, stop.
    - Delete it: remove it from its parent's children, add its segment_len to
      evicted_tokens, increment num_evicted_nodes. Deleting it may turn its
      parent into a new leaf, which becomes evictable in a later iteration of
      THIS evict op if the parent also has ref_count == 0.
    - Stop once evicted_tokens >= num_tokens_target.
  num_tokens_target <= 0 evicts nothing.

  EVICT does not change lru_timestamp of surviving nodes.

  Output: matched_prefix_len = -1.

========================= PER-OP OUTPUTS =========================

After EACH op (in op order) the following are written to the op's output slot:
  matched_prefix_len[i]  int32  (INSERT: as defined; else -1)
  num_nodes[i]           int32  non-root node count AFTER the op
  num_tokens_cached[i]   int32  sum of segment_len over non-root nodes AFTER op
  num_evicted_nodes[i]   int32  nodes deleted by THIS op (EVICT only; else 0)
  evicted_tokens[i]      int32  tokens reclaimed by THIS op (EVICT only; else 0)
  FNV-1a-64 PRIMITIVE (NORMATIVE -- this project uses a NON-CANONICAL basis;
    the agent MUST use this exact basis or every checksum fails):
      offset basis = 1469598103934665603  (hex 0x14650FB0739D0383) -- this is NOT
        the canonical FNV-1a-64 basis 14695981039346656037 (0xCBF29CE484222325).
      prime        = 1099511628211  (0x100000001B3).
      fold: start h = offset basis; absorb each field's raw bytes little-endian at
        its full declared width (u8=1, u32=4, u64=8, i64=8 two's-complement); per
        byte b: h = (h XOR b) * prime (mod 2^64). No length prefix / terminator / final mix.

  tree_checksum[i]       uint64 FNV-1a-64 over the canonical tree (see below)
  state_checksum[i]      uint64 FNV-1a-64 over (clock, num_nodes,
                                num_tokens_cached, next_node_id) then
                                tree_checksum

tree_checksum canonical traversal: iterate non-root nodes in ASCENDING current
node id. For each such node hash, in order, as 4-byte little-endian int32s
unless noted:
    id, parent_id, ref_count, segment_len, then each segment token (int32).
ref_count and lru_timestamp interplay: lru_timestamp is NOT part of the
checksum (it is internal ordering state), but ref_count IS. The hash begins
from the FNV-1a-64 offset basis and first absorbs the int32 count of non-root
nodes, then the per-node records described above.

state_checksum: FNV offset basis, absorb int64 clock, int32 num_nodes, int32
num_tokens_cached, int32 next_node_id (the value the monotonic counter will
assign NEXT), then absorb the 8 bytes of tree_checksum.

============================ RULES ============================
  - solution_init may allocate persistent state.
  - solution_run may NOT call cudaMalloc/cudaFree; it uses provided workspace
    and persistent state only.
  - solution_run may launch kernels.
  - solution_reset clears the entire tree back to just the root, clock=0,
    next_node_id=1, all requests not-live.
  - All graded outputs are EXACT integers. No floats, no tolerance.
*/

struct alignas(8) RpctProblemSpec {
    int32_t abi_version;
    int32_t max_nodes;       // capacity for non-root nodes + root
    int32_t max_tokens;      // capacity for total cached tokens
    int32_t max_ops;         // max ops per step
    int32_t max_requests;    // request_id in [0, max_requests)
    int32_t max_op_tokens;   // capacity of flat op_tokens per step
    int32_t reserved[10];
};

struct alignas(8) RpctRunSpec {
    int32_t abi_version;
    int32_t op_count;
    int32_t total_op_tokens; // length of op_tokens used this step
    int32_t step_id;
    int32_t flags;
    int32_t reserved[11];
};

struct alignas(8) RpctInputs {
    const int32_t* op_kind;          // [op_count]
    const int32_t* op_arg_a;         // [op_count]
    const int32_t* op_arg_b;         // [op_count] (reserved)
    const int32_t* op_token_offset;  // [op_count]
    const int32_t* op_token_len;     // [op_count]
    const int32_t* op_tokens;        // [total_op_tokens]
};

struct alignas(8) RpctOutputs {
    int32_t* matched_prefix_len;  // [op_count]
    int32_t* num_nodes;           // [op_count]
    int32_t* num_tokens_cached;   // [op_count]
    int32_t* num_evicted_nodes;   // [op_count]
    int32_t* evicted_tokens;      // [op_count]
    uint64_t* tree_checksum;      // [op_count]
    uint64_t* state_checksum;     // [op_count]
};

static_assert(sizeof(RpctProblemSpec) == 64, "RpctProblemSpec layout drift");
static_assert(sizeof(RpctRunSpec) == 64, "RpctRunSpec layout drift");
static_assert(sizeof(RpctInputs) == 48, "RpctInputs layout drift");
static_assert(sizeof(RpctOutputs) == 56, "RpctOutputs layout drift");

static inline size_t rpct_align_up_size(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

static inline int rpct_validate_problem_spec(const RpctProblemSpec* spec) {
    if (!spec) return 0;
    if (spec->abi_version != RPCT_ABI_VERSION) return 0;
    if (spec->max_nodes < RPCT_MIN_NODES || spec->max_nodes > RPCT_MAX_NODES) return 0;
    if (spec->max_tokens < 0 || spec->max_tokens > RPCT_MAX_TOKENS_TOTAL) return 0;
    if (spec->max_ops < 0 || spec->max_ops > RPCT_MAX_OPS) return 0;
    if (spec->max_requests < 1 || spec->max_requests > RPCT_MAX_REQUESTS) return 0;
    if (spec->max_op_tokens < 0 || spec->max_op_tokens > RPCT_MAX_OP_TOKENS) return 0;
    return 1;
}

static inline int rpct_validate_run_spec(const RpctRunSpec* run) {
    if (!run) return 0;
    if (run->abi_version != RPCT_ABI_VERSION) return 0;
    if (run->op_count < 0 || run->op_count > RPCT_MAX_OPS) return 0;
    if (run->total_op_tokens < 0 || run->total_op_tokens > RPCT_MAX_OP_TOKENS) return 0;
    return 1;
}

extern "C" size_t solution_workspace_bytes(const RpctProblemSpec* spec);

extern "C" cudaError_t solution_init(
    const RpctProblemSpec* spec,
    void** state_out,
    cudaStream_t stream);

extern "C" cudaError_t solution_run(
    void* state,
    const RpctRunSpec* run,
    const void* inputs,
    void* outputs,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream);

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream);

extern "C" void solution_destroy(void* state);

#endif  // RADIX_PREFIX_CACHE_TREE_COMMON_H_

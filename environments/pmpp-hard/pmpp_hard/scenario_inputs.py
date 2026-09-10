"""Seed payloads and relabel hand-built cases without changing their schedules."""

from pmpp_hard.errors import PoolIntegrityError

# Exact constructor anchors keep transformations separate from oracle arithmetic.
REPLACEMENTS = {
    "v2-pl-36-radix_prefix_cache_tree": (
        (
            b"o.tokens = std::move(toks);",
            b"for (auto& token : toks) token = pmpp_input_label(token); o.tokens = std::move(toks);",
        ),
    ),
    "v2-pl-48-raft_log_snapshot": (
        (
            b"o.value = payload;",
            b"o.value = pmpp_input_payload(payload);",
        ),
    ),
    "v2-pl-49-rcu_epoch_reclaimer": (
        (
            b"return mk(RCU_OP_ALLOC, req, 0, 0, val);",
            b"return mk(RCU_OP_ALLOC, req, 0, 0, pmpp_input_payload(val));",
        ),
    ),
    "v2-pl-50-beps_tree_buffer": (
        (
            b"o.value = val;",
            b"o.value = pmpp_input_payload(val);",
        ),
        (
            b"o.value = delta;",
            b"o.value = pmpp_input_payload(delta);",
        ),
    ),
    "v2-pl-52-wormhole_mesh_credits": (
        (
            b"r.a0 = a0;",
            b"r.a0 = (op == WMC_OP_INJECT || op == WMC_OP_DROP) ? pmpp_input_label(a0) : a0;",
        ),
    ),
    "v2-pl-53-incremental_gc_weakref": (
        (
            b"r.size_arg=size;",
            b"r.size_arg = opcode == IGCW_OP_ALLOC && size > 0 ? static_cast<int64_t>(1 + (pmpp_input_value(size) & 0x7fffffffULL)) : size;",
        ),
    ),
    "v2-pl-61-mk_chunked_dep_counters": (
        (
            b"op.arg_e = e;",
            b"op.arg_e = (kind == MK_OP_PRODUCE || kind == MK_OP_ARM_WAIT) ? static_cast<int>(pmpp_input_value(e) & 0x7fffffffULL) : e;",
        ),
    ),
    "v2-pl-64-mk_fleet_two_level": (
        (
            b"r.payload_seed = seed;",
            b"r.payload_seed = static_cast<int>(pmpp_input_value(seed) & 0x7fffffffULL);",
        ),
    ),
    "v2-pl-67-mk_pipeline_scoreboard": (
        (
            b"out_counter, seed, reads, writes);",
            b"out_counter, pmpp_input_value(seed), reads, writes);",
        ),
    ),
    "v2-pl-68-mk_multigpu_allreduce": (
        (
            b"r.a3 = a3;",
            b"r.a3 = op == MGA_OP_BEGIN_ALLREDUCE ? static_cast<int>(pmpp_input_value(a3) & 0x7fffffffULL) : a3;",
        ),
    ),
}

_GRAPH_TASK = "v2-pl-30-graph_frontier_bfs_state"
TASKS = REPLACEMENTS.keys() | {_GRAPH_TASK}

_HELPERS = """
static uint64_t pmpp_input_value(uint64_t value) {
    uint64_t z = value ^ PMPP_INPUT_PARAMETER;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
// Preserve zero and sign cases; bound magnitudes to keep scenario sums in range.
static int64_t pmpp_input_payload(int64_t value) {
    if (value == 0) return 0;
    const uint64_t magnitude = value < 0 ? 0ULL - uint64_t(value) : uint64_t(value);
    const int64_t fresh = 1 + (pmpp_input_value(magnitude) & 0x7fffffffULL);
    return value < 0 ? -fresh : fresh;
}
// A bijection on nonnegative int32 labels; negative invalid IDs stay invalid.
static int32_t pmpp_input_label(int32_t value) {
    if (value < 0) return value;
    const uint64_t multiplier = pmpp_input_value(0) | 1ULL;
    return static_cast<int32_t>((multiplier * static_cast<uint32_t>(value)
                                + pmpp_input_value(1)) & 0x7fffffffULL);
}

"""

_GRAPH_RELABEL = b"""std::vector<Scenario> scenarios = build_scenarios();
        for (auto& sc : scenarios) {
            const GraphHost original = sc.graph;
            const int n = original.N;
            std::vector<int> label(n), inverse(n);
            for (int i = 0; i < n; ++i) label[i] = i;
            for (int i = n - 1; i > 0; --i) {
                const int j = pmpp_input_value((uint64_t(n) << 32) | uint64_t(i)) % (i + 1);
                std::swap(label[i], label[j]);
            }
            for (int i = 0; i < n; ++i) inverse[label[i]] = i;
            std::vector<int> edge_index(original.col_indices.size());
            sc.graph.col_indices.clear();
            sc.graph.weights.clear();
            for (int u = 0; u < n; ++u) {
                sc.graph.row_offsets[u] = sc.graph.col_indices.size();
                const int old_u = inverse[u];
                std::vector<std::pair<int, int>> row;
                for (int e = original.row_offsets[old_u]; e < original.row_offsets[old_u + 1]; ++e)
                    row.push_back({label[original.col_indices[e]], e});
                std::sort(row.begin(), row.end());
                for (auto entry : row) {
                    edge_index[entry.second] = sc.graph.col_indices.size();
                    sc.graph.col_indices.push_back(entry.first);
                    sc.graph.weights.push_back(original.weights[entry.second]);
                }
            }
            sc.graph.row_offsets[n] = sc.graph.col_indices.size();
            for (auto& step : sc.steps) {
                if (step.run.source >= 0 && step.run.source < n)
                    step.run.source = label[step.run.source];
                if (step.run.reset_source >= 0 && step.run.reset_source < n)
                    step.run.reset_source = label[step.run.reset_source];
                for (auto& e : step.update_edge_index)
                    if (e >= 0 && size_t(e) < edge_index.size()) e = edge_index[e];
            }
        }"""


def replace_once(
    task_id: str, source: bytes, anchor: bytes, replacement: bytes
) -> bytes:
    if source.count(anchor) != 1:
        raise PoolIntegrityError(
            f"{task_id}: unexpected scenario input anchor {anchor!r}"
        )
    return source.replace(anchor, replacement)


def render(task_id: str, source: bytes, parameter: int) -> bytes:
    helpers = _HELPERS.replace(
        "PMPP_INPUT_PARAMETER", f"0x{parameter:016x}ULL"
    ).encode()
    source = replace_once(
        task_id, source, b"struct Scenario {", helpers + b"struct Scenario {"
    )
    if task_id == _GRAPH_TASK:
        return replace_once(
            task_id,
            source,
            b"std::vector<Scenario> scenarios = build_scenarios();",
            _GRAPH_RELABEL,
        )
    for anchor, replacement in REPLACEMENTS[task_id]:
        source = replace_once(task_id, source, anchor, replacement)
    return source

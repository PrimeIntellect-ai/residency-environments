// file: test_graph_frontier_bfs_state.cu

#include "graph_frontier_bfs_state_common.h"
#include "graph_frontier_bfs_state_oracle.hpp"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint64_t g_state = 0x64d7b0192aefc538ULL;
static constexpr uint8_t kGuardByte = 0xA5;
static constexpr size_t kGuardBytes = 256;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _err = (expr);                                               \
        if (_err != cudaSuccess) {                                               \
            std::ostringstream _oss;                                             \
            _oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " : "     \
                 << cudaGetErrorString(_err);                                   \
            throw std::runtime_error(_oss.str());                               \
        }                                                                       \
    } while (0)

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }

    void allocate(size_t n) {
        count = n;
        if (n > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
    }

    void upload(const std::vector<T>& host) {
        if (host.size() != count) throw std::runtime_error("upload size mismatch");
        if (count > 0) CUDA_CHECK(cudaMemcpy(ptr, host.data(), sizeof(T) * count, cudaMemcpyHostToDevice));
    }

    std::vector<T> download() const {
        std::vector<T> host(count);
        if (count > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost));
        return host;
    }
};

template <typename T>
struct GuardedDeviceBuffer {
    uint8_t* raw = nullptr;
    T* ptr = nullptr;
    size_t count = 0;
    size_t data_bytes = 0;

    GuardedDeviceBuffer() = default;
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

    ~GuardedDeviceBuffer() {
        if (raw) cudaFree(raw);
    }

    void allocate(size_t n) {
        count = n;
        data_bytes = sizeof(T) * count;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raw), kGuardBytes + data_bytes + kGuardBytes));
        CUDA_CHECK(cudaMemset(raw, kGuardByte, kGuardBytes + data_bytes + kGuardBytes));
        ptr = reinterpret_cast<T*>(raw + kGuardBytes);
    }

    std::vector<T> download_data() const {
        std::vector<T> host(count);
        if (count > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr, data_bytes, cudaMemcpyDeviceToHost));
        return host;
    }

    bool check_guards(const char* name, std::string* error) const {
        std::vector<uint8_t> left(kGuardBytes), right(kGuardBytes);
        CUDA_CHECK(cudaMemcpy(left.data(), raw, kGuardBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(right.data(), raw + kGuardBytes + data_bytes, kGuardBytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < kGuardBytes; ++i) {
            if (left[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "left guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }
            if (right[i] != kGuardByte) {
                if (error) {
                    std::ostringstream oss;
                    oss << "right guard corrupted for " << name << " at byte " << i;
                    *error = oss.str();
                }
                return false;
            }
        }
        return true;
    }
};

struct GraphHost {
    std::string name;
    int N = 0;
    std::vector<int32_t> row_offsets;
    std::vector<int32_t> col_indices;
    std::vector<int32_t> weights;
};

struct StepHost {
    GfbfRunSpec run;
    std::vector<int32_t> update_edge_index;
    std::vector<int32_t> update_edge_weight;
};

struct StepResult {
    int32_t frontier_size = 0;
    int32_t newly_visited_count = 0;
    uint64_t visited_checksum = 0;
    uint64_t level_histogram_checksum = 0;
    std::vector<int32_t> next_frontier;
    uint64_t next_frontier_checksum = 0;
    int32_t total_visited = 0;
};

struct Scenario {
    std::string name;
    GraphHost graph;
    std::vector<StepHost> steps;
    bool make_permuted_graph = false;
};

static void add_edge(GraphHost& g, int u, int v, int w) {
    (void)w;
    if (u < 0 || u >= g.N || v < 0 || v >= g.N) return;
    g.col_indices.push_back(v);
    g.weights.push_back(w);
}

static GraphHost finalize_from_adj(const std::string& name, const std::vector<std::vector<std::pair<int, int>>>& adj) {
    GraphHost g;
    g.name = name;
    g.N = static_cast<int>(adj.size());
    g.row_offsets.assign(static_cast<size_t>(g.N) + 1, 0);

    for (int u = 0; u < g.N; ++u) {
        g.row_offsets[static_cast<size_t>(u)] = static_cast<int32_t>(g.col_indices.size());
        std::vector<std::pair<int, int>> row = adj[static_cast<size_t>(u)];
        std::sort(row.begin(), row.end());
        for (auto p : row) {
            g.col_indices.push_back(p.first);
            g.weights.push_back(p.second);
        }
    }
    g.row_offsets[static_cast<size_t>(g.N)] = static_cast<int32_t>(g.col_indices.size());
    return g;
}

static GraphHost make_uniform_graph(int N, int degree) {
    std::vector<std::vector<std::pair<int, int>>> adj(static_cast<size_t>(N));
    for (int u = 0; u < N; ++u) {
        for (int j = 1; j <= degree; ++j) {
            int v = (u + j * 17 + (u % 11)) % N;
            adj[static_cast<size_t>(u)].push_back({v, 1});
        }
    }
    return finalize_from_adj("uniform", adj);
}

static GraphHost make_powerlaw_graph(int N) {
    std::vector<std::vector<std::pair<int, int>>> adj(static_cast<size_t>(N));
    for (int v = 1; v < std::min(N, 512); ++v) {
        adj[0].push_back({v, 1});
    }
    for (int u = 1; u < N; ++u) {
        adj[static_cast<size_t>(u)].push_back({(u + 1) % N, 1});
        adj[static_cast<size_t>(u)].push_back({(u * 7 + 13) % N, 1});
        if ((u % 16) == 0) adj[static_cast<size_t>(u)].push_back({0, 1});
    }
    return finalize_from_adj("powerlaw", adj);
}

static GraphHost make_grid_graph(int side) {
    const int N = side * side;
    std::vector<std::vector<std::pair<int, int>>> adj(static_cast<size_t>(N));

    for (int r = 0; r < side; ++r) {
        for (int c = 0; c < side; ++c) {
            const int u = r * side + c;
            if (r > 0) adj[static_cast<size_t>(u)].push_back({(r - 1) * side + c, 1});
            if (r + 1 < side) adj[static_cast<size_t>(u)].push_back({(r + 1) * side + c, 1});
            if (c > 0) adj[static_cast<size_t>(u)].push_back({r * side + c - 1, 1});
            if (c + 1 < side) adj[static_cast<size_t>(u)].push_back({r * side + c + 1, 1});
        }
    }

    return finalize_from_adj("grid", adj);
}

static GraphHost reverse_each_row(const GraphHost& src) {
    GraphHost g = src;
    g.name = src.name + "_reversed_rows";
    g.col_indices.clear();
    g.weights.clear();
    g.row_offsets.assign(static_cast<size_t>(src.N) + 1, 0);

    for (int u = 0; u < src.N; ++u) {
        g.row_offsets[static_cast<size_t>(u)] = static_cast<int32_t>(g.col_indices.size());
        const int begin = src.row_offsets[static_cast<size_t>(u)];
        const int end = src.row_offsets[static_cast<size_t>(u + 1)];
        for (int e = end - 1; e >= begin; --e) {
            g.col_indices.push_back(src.col_indices[static_cast<size_t>(e)]);
            g.weights.push_back(src.weights[static_cast<size_t>(e)]);
        }
    }
    g.row_offsets[static_cast<size_t>(src.N)] = static_cast<int32_t>(g.col_indices.size());
    return g;
}

static int find_edge(const GraphHost& g, int u, int v) {
    if (u < 0 || u >= g.N) return -1;
    for (int e = g.row_offsets[static_cast<size_t>(u)]; e < g.row_offsets[static_cast<size_t>(u + 1)]; ++e) {
        if (g.col_indices[static_cast<size_t>(e)] == v) return e;
    }
    return -1;
}

static StepHost make_step(
    const GraphHost& graph,
    int step_id,
    int source,
    int reset_source,
    int flags,
    const std::vector<std::pair<int, int>>& updates) {
    StepHost step;
    step.run = {};
    step.run.abi_version = GFBF_ABI_VERSION;
    step.run.num_nodes = graph.N;
    step.run.num_edges = static_cast<int32_t>(graph.col_indices.size());
    step.run.source = source;
    step.run.reset_source = reset_source;
    step.run.update_count = static_cast<int32_t>(updates.size());
    step.run.step_id = step_id;
    step.run.flags = flags;

    if (!gfbf_validate_run_spec(&step.run)) throw std::runtime_error("invalid GfbfRunSpec");

    const size_t rows = std::max<size_t>(1, updates.size());
    step.update_edge_index.assign(rows, -1);
    step.update_edge_weight.assign(rows, 0);

    for (size_t i = 0; i < updates.size(); ++i) {
        step.update_edge_index[i] = updates[i].first;
        step.update_edge_weight[i] = updates[i].second;
    }

    return step;
}

static Scenario make_uniform_scenario() {
    Scenario sc;
    sc.name = "uniform_update_reset";
    sc.graph = make_uniform_graph(768, 4);

    int disable = find_edge(sc.graph, 0, (0 + 17 + (0 % 11)) % sc.graph.N);
    int enable = find_edge(sc.graph, 13, (13 + 17 + (13 % 11)) % sc.graph.N);

    for (int s = 0; s < 20; ++s) {
        std::vector<std::pair<int, int>> updates;
        int reset = -1;
        int flags = 0;

        if (s == 0) flags = GFBF_FLAG_RELOAD_GRAPH;
        if (s == 4 && disable >= 0) updates.push_back({disable, 0});
        if (s == 7 && enable >= 0) updates.push_back({enable, 0});
        if (s == 10) reset = 5;
        if (s == 14 && disable >= 0) updates.push_back({disable, 1});

        sc.steps.push_back(make_step(sc.graph, s, 0, reset, flags, updates));
    }

    return sc;
}

static Scenario make_powerlaw_scenario() {
    Scenario sc;
    sc.name = "powerlaw_hub_updates";
    sc.graph = make_powerlaw_graph(1024);

    std::vector<int> hub_edges;
    for (int e = sc.graph.row_offsets[0]; e < sc.graph.row_offsets[1] && (int)hub_edges.size() < 8; ++e) {
        hub_edges.push_back(e);
    }

    for (int s = 0; s < 18; ++s) {
        std::vector<std::pair<int, int>> updates;
        int reset = -1;
        int flags = s == 0 ? GFBF_FLAG_RELOAD_GRAPH : 0;

        if (s == 3) {
            for (int e : hub_edges) updates.push_back({e, 0});
        }
        if (s == 8) reset = 32;
        if (s == 11) {
            for (int e : hub_edges) updates.push_back({e, 1});
        }

        sc.steps.push_back(make_step(sc.graph, s, 0, reset, flags, updates));
    }

    return sc;
}

static Scenario make_grid_permute_scenario() {
    Scenario sc;
    sc.name = "grid_permuted_replay";
    sc.graph = make_grid_graph(48);
    sc.make_permuted_graph = true;

    for (int s = 0; s < 24; ++s) {
        int reset = -1;
        int flags = s == 0 ? GFBF_FLAG_RELOAD_GRAPH : 0;
        if (s == 12) reset = 47 * 48 + 47;
        sc.steps.push_back(make_step(sc.graph, s, 0, reset, flags, {}));
    }

    return sc;
}

static std::vector<Scenario> build_scenarios() {
    std::vector<Scenario> scenarios;
    scenarios.push_back(make_uniform_scenario());
    scenarios.push_back(make_powerlaw_scenario());
    scenarios.push_back(make_grid_permute_scenario());
    return scenarios;
}

static bool check_input_unchanged(
    const GraphHost& graph,
    const StepHost& step,
    const DeviceBuffer<int32_t>& d_row_offsets,
    const DeviceBuffer<int32_t>& d_col_indices,
    const DeviceBuffer<int32_t>& d_initial_weight,
    const DeviceBuffer<int32_t>& d_update_edge_index,
    const DeviceBuffer<int32_t>& d_update_edge_weight,
    std::string* error) {
    if (d_row_offsets.download() != graph.row_offsets) {
        if (error) *error = "input row_offsets modified";
        return false;
    }
    if (d_col_indices.download() != graph.col_indices) {
        if (error) *error = "input col_indices modified";
        return false;
    }
    if (d_initial_weight.download() != graph.weights) {
        if (error) *error = "input initial_edge_weight modified";
        return false;
    }
    if (d_update_edge_index.download() != step.update_edge_index) {
        if (error) *error = "input update_edge_index modified";
        return false;
    }
    if (d_update_edge_weight.download() != step.update_edge_weight) {
        if (error) *error = "input update_edge_weight modified";
        return false;
    }
    return true;
}

static bool run_scenario_once(
    const Scenario& sc,
    bool verbose,
    std::vector<StepResult>* results,
    int* passed_steps,
    int* total_steps,
    std::string* first_error) {
    GfbfProblemSpec spec = {};
    spec.abi_version = GFBF_ABI_VERSION;
    spec.max_nodes = sc.graph.N;
    spec.max_edges = static_cast<int32_t>(sc.graph.col_indices.size());
    spec.max_updates = 64;

    size_t workspace_bytes = solution_workspace_bytes(&spec); if (workspace_bytes == 0) workspace_bytes = 1; /*PMPP_WS0_FIX*/
    if (workspace_bytes == 0) {
        if (first_error) *first_error = "solution_workspace_bytes returned 0";
        return false;
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    void* state = nullptr;
    CUDA_CHECK(solution_init(&spec, &state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer<uint8_t> workspace;
    workspace.allocate(workspace_bytes);

    DeviceBuffer<int32_t> d_row_offsets;
    DeviceBuffer<int32_t> d_col_indices;
    DeviceBuffer<int32_t> d_initial_weight;

    d_row_offsets.allocate(sc.graph.row_offsets.size());
    d_col_indices.allocate(sc.graph.col_indices.size());
    d_initial_weight.allocate(sc.graph.weights.size());

    d_row_offsets.upload(sc.graph.row_offsets);
    d_col_indices.upload(sc.graph.col_indices);
    d_initial_weight.upload(sc.graph.weights);

    GfbfOracleState oracle;
    oracle.init(spec);

    CUDA_CHECK(solution_reset(state, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    oracle.reset();

    if (results) {
        results->clear();
        results->reserve(sc.steps.size());
    }

    bool all_ok = true;

    for (size_t i = 0; i < sc.steps.size(); ++i) {
        const StepHost& step = sc.steps[i];

        DeviceBuffer<int32_t> d_update_idx;
        DeviceBuffer<int32_t> d_update_w;
        d_update_idx.allocate(step.update_edge_index.size());
        d_update_w.allocate(step.update_edge_weight.size());
        d_update_idx.upload(step.update_edge_index);
        d_update_w.upload(step.update_edge_weight);

        GuardedDeviceBuffer<int32_t> d_frontier_size;
        GuardedDeviceBuffer<int32_t> d_newly_visited_count;
        GuardedDeviceBuffer<uint64_t> d_visited_checksum;
        GuardedDeviceBuffer<uint64_t> d_level_histogram_checksum;
        GuardedDeviceBuffer<int32_t> d_next_frontier;
        GuardedDeviceBuffer<uint64_t> d_next_frontier_checksum;
        GuardedDeviceBuffer<int32_t> d_total_visited;

        d_frontier_size.allocate(1);
        d_newly_visited_count.allocate(1);
        d_visited_checksum.allocate(1);
        d_level_histogram_checksum.allocate(1);
        d_next_frontier.allocate(static_cast<size_t>(sc.graph.N));
        d_next_frontier_checksum.allocate(1);
        d_total_visited.allocate(1);

        GfbfInputs inputs = {};
        inputs.row_offsets = d_row_offsets.ptr;
        inputs.col_indices = d_col_indices.ptr;
        inputs.initial_edge_weight = d_initial_weight.ptr;
        inputs.update_edge_index = d_update_idx.ptr;
        inputs.update_edge_weight = d_update_w.ptr;

        GfbfOutputs outputs = {};
        outputs.frontier_size = d_frontier_size.ptr;
        outputs.newly_visited_count = d_newly_visited_count.ptr;
        outputs.visited_checksum = d_visited_checksum.ptr;
        outputs.level_histogram_checksum = d_level_histogram_checksum.ptr;
        outputs.next_frontier = d_next_frontier.ptr;
        outputs.next_frontier_checksum = d_next_frontier_checksum.ptr;
        outputs.total_visited = d_total_visited.ptr;

        std::string error;

        CUDA_CHECK(solution_run(
            state,
            &step.run,
            &inputs,
            &outputs,
            workspace.ptr,
            workspace_bytes,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        bool ok = true;
        ok = ok && check_input_unchanged(sc.graph, step, d_row_offsets, d_col_indices, d_initial_weight, d_update_idx, d_update_w, &error);
        ok = ok && d_frontier_size.check_guards("frontier_size", &error);
        ok = ok && d_newly_visited_count.check_guards("newly_visited_count", &error);
        ok = ok && d_visited_checksum.check_guards("visited_checksum", &error);
        ok = ok && d_level_histogram_checksum.check_guards("level_histogram_checksum", &error);
        ok = ok && d_next_frontier.check_guards("next_frontier", &error);
        ok = ok && d_next_frontier_checksum.check_guards("next_frontier_checksum", &error);
        ok = ok && d_total_visited.check_guards("total_visited", &error);

        const std::vector<int32_t> h_frontier_size = d_frontier_size.download_data();
        const std::vector<int32_t> h_new_count = d_newly_visited_count.download_data();
        const std::vector<uint64_t> h_visited_hash = d_visited_checksum.download_data();
        const std::vector<uint64_t> h_level_hash = d_level_histogram_checksum.download_data();
        const std::vector<int32_t> h_frontier = d_next_frontier.download_data();
        const std::vector<uint64_t> h_frontier_hash = d_next_frontier_checksum.download_data();
        const std::vector<int32_t> h_total = d_total_visited.download_data();

        GfbfHostInputsView host_inputs = {};
        host_inputs.row_offsets = sc.graph.row_offsets.data();
        host_inputs.col_indices = sc.graph.col_indices.data();
        host_inputs.initial_edge_weight = sc.graph.weights.data();
        host_inputs.update_edge_index = step.update_edge_index.data();
        host_inputs.update_edge_weight = step.update_edge_weight.data();

        GfbfExpected expected;
        oracle.step_once(step.run, host_inputs, &expected);

        GfbfHostOutputsView got = {};
        got.frontier_size = h_frontier_size.data();
        got.newly_visited_count = h_new_count.data();
        got.visited_checksum = h_visited_hash.data();
        got.level_histogram_checksum = h_level_hash.data();
        got.next_frontier = h_frontier.data();
        got.next_frontier_checksum = h_frontier_hash.data();
        got.total_visited = h_total.data();

        ok = ok && gfbf_check_all_outputs(expected, got, &error);

        StepResult result;
        result.frontier_size = h_frontier_size[0];
        result.newly_visited_count = h_new_count[0];
        result.visited_checksum = h_visited_hash[0];
        result.level_histogram_checksum = h_level_hash[0];
        result.next_frontier.assign(h_frontier.begin(), h_frontier.begin() + result.frontier_size);
        result.next_frontier_checksum = h_frontier_hash[0];
        result.total_visited = h_total[0];

        ++(*total_steps);
        if (ok) {
            ++(*passed_steps);
        } else {
            all_ok = false;
            if (first_error && first_error->empty()) {
                std::ostringstream oss;
                oss << sc.name << " step " << i << ": " << error;
                *first_error = oss.str();
            }
        }

        if (results) results->push_back(result);

        if (verbose) {
            std::printf(
                "scenario %-28s step %02zu/%02zu frontier=%d updates=%d %s%s%s\n",
                sc.name.c_str(),
                i,
                sc.steps.size(),
                result.frontier_size,
                step.run.update_count,
                ok ? "PASS" : "FAIL",
                ok ? "" : "  ",
                ok ? "" : error.c_str());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));

    return all_ok;
}

static bool compare_results(
    const std::vector<StepResult>& a,
    const std::vector<StepResult>& b,
    std::string* error) {
    if (a.size() != b.size()) {
        if (error) *error = "step count mismatch";
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].frontier_size != b[i].frontier_size ||
            a[i].newly_visited_count != b[i].newly_visited_count ||
            a[i].visited_checksum != b[i].visited_checksum ||
            a[i].level_histogram_checksum != b[i].level_histogram_checksum ||
            a[i].next_frontier_checksum != b[i].next_frontier_checksum ||
            a[i].total_visited != b[i].total_visited ||
            a[i].next_frontier != b[i].next_frontier) {
            if (error) {
                std::ostringstream oss;
                oss << "replay mismatch at step " << i;
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        std::vector<Scenario> scenarios = build_scenarios();
        int passed = 0;
        int total = 0;
        bool all_ok = true;

        for (const Scenario& sc : scenarios) {
            std::vector<StepResult> base_results;
            std::vector<StepResult> replay_results;

            std::string error;

            const bool ok_base = run_scenario_once(sc, true, &base_results, &passed, &total, &error);
            const bool ok_replay = run_scenario_once(sc, false, &replay_results, &passed, &total, &error);

            if (ok_base && ok_replay) {
                std::string compare_error;
                if (compare_results(base_results, replay_results, &compare_error)) {
                    std::printf("scenario %-28s exact replay PASS\n", sc.name.c_str());
                } else {
                    all_ok = false;
                    std::printf("scenario %-28s exact replay FAIL  %s\n", sc.name.c_str(), compare_error.c_str());
                }
            } else {
                all_ok = false;
                std::printf("scenario %-28s FAIL  %s\n", sc.name.c_str(), error.c_str());
            }

            if (sc.make_permuted_graph) {
                Scenario perm = sc;
                perm.name = sc.name + "_row_permuted";
                perm.graph = reverse_each_row(sc.graph);

                std::vector<StepResult> perm_results;
                std::string perm_error;
                const bool ok_perm = run_scenario_once(perm, false, &perm_results, &passed, &total, &perm_error);

                if (ok_perm) {
                    std::string compare_error;
                    if (compare_results(base_results, perm_results, &compare_error)) {
                        std::printf("scenario %-28s permuted-row replay PASS\n", sc.name.c_str());
                    } else {
                        all_ok = false;
                        std::printf("scenario %-28s permuted-row replay FAIL  %s\n", sc.name.c_str(), compare_error.c_str());
                    }
                } else {
                    all_ok = false;
                    std::printf("scenario %-28s permuted-row replay FAIL  %s\n", sc.name.c_str(), perm_error.c_str());
                }
            }
        }

        std::printf("passed %d / %d\n", passed, total);
        return all_ok && passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}

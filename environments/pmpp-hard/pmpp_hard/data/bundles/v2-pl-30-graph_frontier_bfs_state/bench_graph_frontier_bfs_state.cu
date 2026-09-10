// file: bench_graph_frontier_bfs_state.cu

#include "graph_frontier_bfs_state_common.h"
#include "pmpp_bench_digest.cuh"

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
};

struct GraphHost {
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

struct DeviceStep {
    StepHost host;

    DeviceBuffer<int32_t> update_idx;
    DeviceBuffer<int32_t> update_weight;

    DeviceBuffer<int32_t> frontier_size;
    DeviceBuffer<int32_t> newly_visited_count;
    DeviceBuffer<uint64_t> visited_checksum;
    DeviceBuffer<uint64_t> level_histogram_checksum;
    DeviceBuffer<int32_t> next_frontier;
    DeviceBuffer<uint64_t> next_frontier_checksum;
    DeviceBuffer<int32_t> total_visited;

    GfbfInputs inputs;
    GfbfOutputs outputs;
};

static GraphHost make_powerlaw_graph(int N, uint64_t seed) {
    std::vector<std::vector<std::pair<int, int>>> adj(static_cast<size_t>(N));

    // The two pseudo-random neighbor maps derive from PMPP_BENCH_SEED so BFS
    // results are not precomputable offline; N, per-node degree, the hub node
    // and the ring edge stay fixed so the timed workload shape is comparable.
    auto mix = [](uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    const int64_t a1 = 3 + 2 * (int64_t)(mix(seed ^ 0xa1a1a1a1ULL) % 997);
    const int64_t b1 = (int64_t)(mix(seed ^ 0xb1b1b1b1ULL) % (uint64_t)N);
    const int64_t a2 = 3 + 2 * (int64_t)(mix(seed ^ 0xa2a2a2a2ULL) % 997);
    const int64_t b2 = (int64_t)(mix(seed ^ 0xb2b2b2b2ULL) % (uint64_t)N);

    for (int v = 1; v < std::min(N, 1024); ++v) adj[0].push_back({v, 1});
    for (int u = 1; u < N; ++u) {
        adj[static_cast<size_t>(u)].push_back({(u + 1) % N, 1});
        adj[static_cast<size_t>(u)].push_back({(int)(((int64_t)u * a1 + b1) % N), 1});
        adj[static_cast<size_t>(u)].push_back({(int)(((int64_t)u * a2 + b2) % N), 1});
        if ((u % 16) == 0) adj[static_cast<size_t>(u)].push_back({0, 1});
    }

    GraphHost g;
    g.N = N;
    g.row_offsets.assign(static_cast<size_t>(N) + 1, 0);

    for (int u = 0; u < N; ++u) {
        g.row_offsets[static_cast<size_t>(u)] = static_cast<int32_t>(g.col_indices.size());
        std::sort(adj[static_cast<size_t>(u)].begin(), adj[static_cast<size_t>(u)].end());
        for (auto p : adj[static_cast<size_t>(u)]) {
            g.col_indices.push_back(p.first);
            g.weights.push_back(p.second);
        }
    }

    g.row_offsets[static_cast<size_t>(N)] = static_cast<int32_t>(g.col_indices.size());
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

    const size_t rows = std::max<size_t>(1, updates.size());
    step.update_edge_index.assign(rows, -1);
    step.update_edge_weight.assign(rows, 0);

    for (size_t i = 0; i < updates.size(); ++i) {
        step.update_edge_index[i] = updates[i].first;
        step.update_edge_weight[i] = updates[i].second;
    }

    return step;
}

static std::vector<StepHost> make_steps(const GraphHost& graph) {
    std::vector<StepHost> steps;
    std::vector<int> hub_edges;

    for (int e = graph.row_offsets[0]; e < graph.row_offsets[1] && static_cast<int>(hub_edges.size()) < 16; ++e) {
        hub_edges.push_back(e);
    }

    for (int s = 0; s < 32; ++s) {
        std::vector<std::pair<int, int>> updates;
        int reset = -1;
        int flags = s == 0 ? GFBF_FLAG_RELOAD_GRAPH : 0;

        if (s == 5) {
            for (int e : hub_edges) updates.push_back({e, 0});
        }
        if (s == 12) reset = 64;
        if (s == 18) {
            for (int e : hub_edges) updates.push_back({e, 1});
        }
        if (s == 24) reset = 0;

        steps.push_back(make_step(graph, s, 0, reset, flags, updates));
    }

    return steps;
}

static DeviceStep* make_device_step(const StepHost& h, int max_nodes) {
    DeviceStep* ds = new DeviceStep();
    ds->host = h;

    ds->update_idx.allocate(h.update_edge_index.size());
    ds->update_weight.allocate(h.update_edge_weight.size());
    ds->update_idx.upload(h.update_edge_index);
    ds->update_weight.upload(h.update_edge_weight);

    ds->frontier_size.allocate(1);
    ds->newly_visited_count.allocate(1);
    ds->visited_checksum.allocate(1);
    ds->level_histogram_checksum.allocate(1);
    ds->next_frontier.allocate(static_cast<size_t>(max_nodes));
    ds->next_frontier_checksum.allocate(1);
    ds->total_visited.allocate(1);

    ds->outputs = {};
    ds->outputs.frontier_size = ds->frontier_size.ptr;
    ds->outputs.newly_visited_count = ds->newly_visited_count.ptr;
    ds->outputs.visited_checksum = ds->visited_checksum.ptr;
    ds->outputs.level_histogram_checksum = ds->level_histogram_checksum.ptr;
    ds->outputs.next_frontier = ds->next_frontier.ptr;
    ds->outputs.next_frontier_checksum = ds->next_frontier_checksum.ptr;
    ds->outputs.total_visited = ds->total_visited.ptr;

    return ds;
}

static void run_sequence(
    void* state,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    const GraphHost& graph,
    const DeviceBuffer<int32_t>& row_offsets,
    const DeviceBuffer<int32_t>& col_indices,
    const DeviceBuffer<int32_t>& weights,
    const std::vector<DeviceStep*>& steps) {
    for (DeviceStep* ds : steps) {
        ds->inputs = {};
        ds->inputs.row_offsets = row_offsets.ptr;
        ds->inputs.col_indices = col_indices.ptr;
        ds->inputs.initial_edge_weight = weights.ptr;
        ds->inputs.update_edge_index = ds->update_idx.ptr;
        ds->inputs.update_edge_weight = ds->update_weight.ptr;

        CUDA_CHECK(solution_run(
            state,
            &ds->host.run,
            &ds->inputs,
            &ds->outputs,
            workspace,
            workspace_bytes,
            stream));
    }
}

int main(int argc, char** argv) {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        int iters = 20;
        if (argc >= 2) iters = std::max(1, std::atoi(argv[1]));

        // One INDEPENDENT graph variant per timed iteration (plus variant 0 for
        // warmup). make_powerlaw_graph keeps N, per-node degree, hub node and ring
        // edge FIXED across seeds (identical edge count → comparable timed workload),
        // but the random neighbor targets derive from the per-variant seed, so BFS
        // results differ each iteration. Each iteration's graded outputs are folded
        // into out_fnv, so a compute-once-then-replay solution serves stale outputs for
        // iterations 2..K → the folded digest diverges → perf FAIL. Per-variant seeds
        // come from a SplitMix64 stream over PMPP_BENCH_SEED.
        struct GraphVariant {
            GraphHost graph;
            DeviceBuffer<int32_t> row_offsets;
            DeviceBuffer<int32_t> col_indices;
            DeviceBuffer<int32_t> weights;
            std::vector<DeviceStep*> steps;
        };

        uint64_t vseed_state = pmpp::bench_seed(0xc3a5c85c97cb3127ULL) ^ 0x700000000ULL;
        auto next_seed = [&]() -> uint64_t {
            vseed_state += 0x9e3779b97f4a7c15ULL;
            uint64_t z = vseed_state;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31);
        };

        auto make_variant = [&](uint64_t seed) -> GraphVariant* {
            GraphVariant* gv = new GraphVariant();
            gv->graph = make_powerlaw_graph(8192, seed);
            gv->row_offsets.allocate(gv->graph.row_offsets.size());
            gv->col_indices.allocate(gv->graph.col_indices.size());
            gv->weights.allocate(gv->graph.weights.size());
            gv->row_offsets.upload(gv->graph.row_offsets);
            gv->col_indices.upload(gv->graph.col_indices);
            gv->weights.upload(gv->graph.weights);
            std::vector<StepHost> host_steps = make_steps(gv->graph);
            gv->steps.reserve(host_steps.size());
            for (const StepHost& h : host_steps) gv->steps.push_back(make_device_step(h, gv->graph.N));
            return gv;
        };

        std::vector<GraphVariant*> variants((size_t)iters + 1, nullptr);
        for (int v = 0; v <= iters; ++v) variants[(size_t)v] = make_variant(next_seed());

        GfbfProblemSpec spec = {};
        spec.abi_version = GFBF_ABI_VERSION;
        spec.max_nodes = variants[0]->graph.N;
        spec.max_edges = static_cast<int32_t>(variants[0]->graph.col_indices.size());
        spec.max_updates = 64;

        const size_t workspace_bytes = solution_workspace_bytes(&spec);
        if (workspace_bytes == 0) throw std::runtime_error("solution_workspace_bytes returned 0");

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* state = nullptr;
        CUDA_CHECK(solution_init(&spec, &state, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        DeviceBuffer<uint8_t> workspace;
        workspace.allocate(workspace_bytes);

        std::printf(
            "bench_graph nodes=%d edges=%zu steps=%zu\n",
            variants[0]->graph.N,
            variants[0]->graph.col_indices.size(),
            variants[0]->steps.size());

        auto run_variant = [&](GraphVariant* gv) {
            run_sequence(state, workspace.ptr, workspace_bytes, stream, gv->graph,
                         gv->row_offsets, gv->col_indices, gv->weights, gv->steps);
        };

        for (int warm = 0; warm < 3; ++warm) {
            CUDA_CHECK(solution_reset(state, stream));
            run_variant(variants[0]);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        cudaEvent_t start = nullptr, stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        double total_ms = 0.0;

        // Fold each timed iteration's graded outputs (outside the timed region) so a
        // stale/replayed output on any iteration breaks the digest. next_frontier array
        // content is covered by the exact-graded next_frontier_checksum, so the scalar
        // outputs suffice.
        pmpp::OutFnv dg;

        for (int iter = 0; iter < iters; ++iter) {
            GraphVariant* gv = variants[(size_t)iter + 1];

            CUDA_CHECK(solution_reset(state, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaEventRecord(start, stream));
            run_variant(gv);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += static_cast<double>(ms);

            for (DeviceStep* ds : gv->steps) {
                dg.dev(ds->frontier_size.ptr, sizeof(int32_t));
                dg.dev(ds->newly_visited_count.ptr, sizeof(int32_t));
                dg.dev(ds->visited_checksum.ptr, sizeof(uint64_t));
                dg.dev(ds->level_histogram_checksum.ptr, sizeof(uint64_t));
                dg.dev(ds->next_frontier_checksum.ptr, sizeof(uint64_t));
                dg.dev(ds->total_visited.ptr, sizeof(int32_t));
            }
        }

        std::printf("avg_ms=%.6f\n", total_ms / static_cast<double>(iters));
        dg.print();

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        for (GraphVariant* gv : variants) {
            for (DeviceStep* ds : gv->steps) delete ds;
            delete gv;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        solution_destroy(state);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fatal: %s\n", ex.what());
        return 1;
    }
}

// file: graph_frontier_bfs_state_oracle.hpp

#ifndef GRAPH_FRONTIER_BFS_STATE_ORACLE_HPP_
#define GRAPH_FRONTIER_BFS_STATE_ORACLE_HPP_

#include "graph_frontier_bfs_state_common.h"

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

struct GfbfHostInputsView {
    const int32_t* row_offsets;
    const int32_t* col_indices;
    const int32_t* initial_edge_weight;
    const int32_t* update_edge_index;
    const int32_t* update_edge_weight;
};

struct GfbfHostOutputsView {
    const int32_t* frontier_size;
    const int32_t* newly_visited_count;
    const uint64_t* visited_checksum;
    const uint64_t* level_histogram_checksum;
    const int32_t* next_frontier;
    const uint64_t* next_frontier_checksum;
    const int32_t* total_visited;
};

struct GfbfExpected {
    int32_t frontier_size = 0;
    int32_t newly_visited_count = 0;
    uint64_t visited_checksum = 0;
    uint64_t level_histogram_checksum = 0;
    std::vector<int32_t> next_frontier;
    uint64_t next_frontier_checksum = 0;
    int32_t total_visited = 0;
};

static inline uint64_t gfbf_oracle_fnv_byte(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

static inline void gfbf_oracle_fnv_bytes(uint64_t* h, const void* ptr, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr);
    uint64_t v = *h;

    for (size_t i = 0; i < n; ++i) {
        v = gfbf_oracle_fnv_byte(v, p[i]);
    }

    *h = v;
}

struct GfbfOracleState {
    GfbfProblemSpec spec{};
    int num_nodes = 0;
    int num_edges = 0;

    bool graph_initialized = false;
    bool bfs_initialized = false;

    int32_t source = -1;
    int32_t current_level = 0;

    std::vector<int32_t> edge_weight;
    std::vector<int32_t> visited;
    std::vector<int32_t> distance;
    std::vector<int32_t> frontier;

    void init(const GfbfProblemSpec& s) {
        spec = s;
        edge_weight.assign((size_t)spec.max_edges, 0);
        visited.assign((size_t)spec.max_nodes, 0);
        distance.assign((size_t)spec.max_nodes, -1);
        frontier.clear();
        graph_initialized = false;
        bfs_initialized = false;
        source = -1;
        current_level = 0;
        num_nodes = 0;
        num_edges = 0;
    }

    void reset() {
        std::fill(edge_weight.begin(), edge_weight.end(), 0);
        std::fill(visited.begin(), visited.end(), 0);
        std::fill(distance.begin(), distance.end(), -1);
        frontier.clear();
        graph_initialized = false;
        bfs_initialized = false;
        source = -1;
        current_level = 0;
        num_nodes = 0;
        num_edges = 0;
    }

    void clear_bfs(int N) {
        std::fill(visited.begin(), visited.begin() + N, 0);
        std::fill(distance.begin(), distance.begin() + N, -1);
        frontier.clear();
        current_level = 0;
    }

    void initialize_source(int N, int src) {
        if (src >= 0 && src < N) {
            visited[(size_t)src] = 1;
            distance[(size_t)src] = 0;
            frontier.assign(1, src);
        } else {
            frontier.clear();
        }
    }

    std::vector<int32_t> expand_once(
        int N,
        const int32_t* row_offsets,
        const int32_t* col_indices,
        int next_level) {
        std::vector<int32_t> candidate((size_t)N, 0);

        for (int u : frontier) {
            if (u < 0 || u >= N) continue;

            const int begin = row_offsets[u];
            const int end = row_offsets[u + 1];

            for (int e = begin; e < end; ++e) {
                if (edge_weight[(size_t)e] <= 0) continue;

                const int v = col_indices[e];
                if (v >= 0 && v < N && visited[(size_t)v] == 0) {
                    candidate[(size_t)v] = 1;
                }
            }
        }

        std::vector<int32_t> next;

        for (int v = 0; v < N; ++v) {
            if (candidate[(size_t)v] != 0 && visited[(size_t)v] == 0) {
                visited[(size_t)v] = 1;
                distance[(size_t)v] = next_level;
                next.push_back(v);
            }
        }

        frontier = next;
        return next;
    }

    void rederive_to_level(
        int N,
        int target_level,
        const int32_t* row_offsets,
        const int32_t* col_indices) {
        clear_bfs(N);
        initialize_source(N, source);

        for (int level = 0; level < target_level; ++level) {
            expand_once(N, row_offsets, col_indices, level + 1);
        }

        current_level = target_level;
    }

    uint64_t frontier_checksum() const {
        uint64_t h = 1469598103934665603ULL;
        int32_t count = static_cast<int32_t>(frontier.size());
        gfbf_oracle_fnv_bytes(&h, &count, sizeof(int32_t));

        for (int32_t v : frontier) {
            gfbf_oracle_fnv_bytes(&h, &v, sizeof(int32_t));
        }

        return h;
    }

    uint64_t visited_hash(int N) const {
        uint64_t h = 1469598103934665603ULL;
        gfbf_oracle_fnv_bytes(&h, &N, sizeof(int32_t));

        for (int i = 0; i < N; ++i) {
            gfbf_oracle_fnv_bytes(&h, &visited[(size_t)i], sizeof(int32_t));
            gfbf_oracle_fnv_bytes(&h, &distance[(size_t)i], sizeof(int32_t));
        }

        return h;
    }

    uint64_t level_hist_hash(int N) const {
        uint64_t h = 1469598103934665603ULL;
        gfbf_oracle_fnv_bytes(&h, &current_level, sizeof(int32_t));

        for (int level = 0; level <= current_level; ++level) {
            int32_t count = 0;
            for (int i = 0; i < N; ++i) {
                if (distance[(size_t)i] == level) ++count;
            }
            gfbf_oracle_fnv_bytes(&h, &count, sizeof(int32_t));
        }

        return h;
    }

    int count_visited(int N) const {
        int count = 0;
        for (int i = 0; i < N; ++i) {
            if (visited[(size_t)i] != 0) ++count;
        }
        return count;
    }

    void step_once(
        const GfbfRunSpec& run,
        const GfbfHostInputsView& in,
        GfbfExpected* expected) {
        const int N = run.num_nodes;
        const int E = run.num_edges;

        const bool reload =
            !graph_initialized || ((run.flags & GFBF_FLAG_RELOAD_GRAPH) != 0);

        if (reload) {
            for (int e = 0; e < E; ++e) {
                edge_weight[(size_t)e] = in.initial_edge_weight[e];
            }

            graph_initialized = true;
            bfs_initialized = false;
            num_nodes = N;
            num_edges = E;
        }

        int applied_updates = 0;

        for (int i = 0; i < run.update_count; ++i) {
            const int e = in.update_edge_index[i];

            if (e >= 0 && e < E) {
                edge_weight[(size_t)e] = in.update_edge_weight[i];
                ++applied_updates;
            }
        }

        const bool do_reset = !bfs_initialized || run.reset_source >= 0;

        if (do_reset) {
            source = run.reset_source >= 0 ? run.reset_source : run.source;
            clear_bfs(N);
            initialize_source(N, source);
            bfs_initialized = true;
        } else if (applied_updates > 0) {
            rederive_to_level(N, current_level, in.row_offsets, in.col_indices);
        }

        const int next_level = current_level + 1;
        expand_once(N, in.row_offsets, in.col_indices, next_level);
        current_level = next_level;

        expected->frontier_size = static_cast<int32_t>(frontier.size());
        expected->newly_visited_count = expected->frontier_size;
        expected->next_frontier = frontier;
        expected->next_frontier_checksum = frontier_checksum();
        expected->visited_checksum = visited_hash(N);
        expected->level_histogram_checksum = level_hist_hash(N);
        expected->total_visited = count_visited(N);
    }
};

static inline bool gfbf_check_all_outputs(
    const GfbfExpected& expected,
    const GfbfHostOutputsView& got,
    std::string* error) {
    if (got.frontier_size[0] != expected.frontier_size) {
        if (error) {
            std::ostringstream oss;
            oss << "frontier_size mismatch: got " << got.frontier_size[0]
                << ", expected " << expected.frontier_size;
            *error = oss.str();
        }
        return false;
    }

    if (got.newly_visited_count[0] != expected.newly_visited_count) {
        if (error) {
            std::ostringstream oss;
            oss << "newly_visited_count mismatch: got " << got.newly_visited_count[0]
                << ", expected " << expected.newly_visited_count;
            *error = oss.str();
        }
        return false;
    }

    if (got.total_visited[0] != expected.total_visited) {
        if (error) {
            std::ostringstream oss;
            oss << "total_visited mismatch: got " << got.total_visited[0]
                << ", expected " << expected.total_visited;
            *error = oss.str();
        }
        return false;
    }

    if (got.visited_checksum[0] != expected.visited_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "visited_checksum mismatch: got 0x"
                << std::hex << got.visited_checksum[0]
                << ", expected 0x" << expected.visited_checksum;
            *error = oss.str();
        }
        return false;
    }

    if (got.level_histogram_checksum[0] != expected.level_histogram_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "level_histogram_checksum mismatch: got 0x"
                << std::hex << got.level_histogram_checksum[0]
                << ", expected 0x" << expected.level_histogram_checksum;
            *error = oss.str();
        }
        return false;
    }

    if (got.next_frontier_checksum[0] != expected.next_frontier_checksum) {
        if (error) {
            std::ostringstream oss;
            oss << "next_frontier_checksum mismatch: got 0x"
                << std::hex << got.next_frontier_checksum[0]
                << ", expected 0x" << expected.next_frontier_checksum;
            *error = oss.str();
        }
        return false;
    }

    for (int i = 0; i < expected.frontier_size; ++i) {
        if (got.next_frontier[i] != expected.next_frontier[(size_t)i]) {
            if (error) {
                std::ostringstream oss;
                oss << "next_frontier mismatch at i=" << i
                    << ": got " << got.next_frontier[i]
                    << ", expected " << expected.next_frontier[(size_t)i];
                *error = oss.str();
            }
            return false;
        }
    }

    return true;
}

/*
GRADER MODEL

Use:
  oracle.init(problem_spec)
  solution_reset + oracle.reset()
  for each step:
    solution_run(...)
    oracle.step_once(...)
    gfbf_check_all_outputs(...)

Required harness coverage:
  - uniform-degree graphs
  - power-law/hub graphs
  - grid-like graphs
  - edge_weight updates disabling/enabling edges
  - reset_source mid-run
  - empty frontier behavior
  - replay after reset
*/

#endif  // GRAPH_FRONTIER_BFS_STATE_ORACLE_HPP_

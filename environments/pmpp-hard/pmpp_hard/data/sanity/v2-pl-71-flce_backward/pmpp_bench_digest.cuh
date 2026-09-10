// pmpp_bench_digest.cuh — shared out_fnv helper for anti-hack perf verification.
// (REWARD_HACK_AUDIT_20260704) The paired perf scorer requires the student bench and the
// reference bench to print the SAME out_fnv over the GRADED outputs on the same PMPP_BENCH_SEED
// before it accepts the student's timing. A shape/threshold/proc no-op that fabricates or skips
// outputs cannot match, so it perf-FAILS. Fold EXACTLY the fields the test grades (not scratch/
// padding, which may legitimately differ) — mirror the task's oracle output set.
//
// Usage in bench_<slug>.cu, AFTER the timed loop + a final untimed solution_run + cudaDeviceSynchronize:
//     #include "pmpp_bench_digest.cuh"
//     pmpp::OutFnv dg;
//     dg.dev(d_dx,   (size_t)rows * H * sizeof(uint16_t));   // each graded output buffer
//     dg.dev(d_loss, (size_t)rows * sizeof(float));
//     ... (all graded fields, in a fixed order)
//     dg.print();                                            // prints  out_fnv=<16 hex>
// If a shape is randomized via PMPP_BENCH_SEED, run one untimed solution_run on the SAME seed
// before folding so student and reference digest identical inputs.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

namespace pmpp {

// FNV-1a-64 with this project's NON-CANONICAL basis (must match every contract's grader).
struct OutFnv {
    uint64_t h = 1469598103934665603ULL;              // 0x14650FB0739D0383 (project basis)
    static constexpr uint64_t PRIME = 1099511628211ULL;

    void bytes(const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= PRIME; }
    }
    // Fold a device buffer by copying it to the host first. No-op on null/zero (a skipped
    // output folds nothing → different digest than the reference that wrote it).
    void dev(const void* dptr, size_t nbytes) {
        if (!dptr || !nbytes) return;
        std::vector<uint8_t> host(nbytes);
        if (cudaMemcpy(host.data(), dptr, nbytes, cudaMemcpyDeviceToHost) != cudaSuccess) return;
        bytes(host.data(), nbytes);
    }
    void print() const { std::printf("out_fnv=%016llx\n", (unsigned long long)h); }
};

// PMPP_BENCH_SEED (fresh per rollout from the scorer). Fallback keeps stand-alone runs deterministic.
inline uint64_t bench_seed(uint64_t fallback) {
    const char* s = std::getenv("PMPP_BENCH_SEED");
    return s ? std::strtoull(s, nullptr, 10) : fallback;
}

}  // namespace pmpp

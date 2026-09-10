// file: bench_mk_event_tensor_runtime.cu
//
// Throughput micro-benchmark for the MK4 Event-Tensor Pop/Push Runtime. Builds a
// large synthetic op stream (defines, ready/blocked registers, signals that zero
// cells and push successors, pops/starts/completes) and times repeated
// solution_run calls. Correctness is covered by the test harness; this only
// measures steady-state op throughput.

#include "mk_event_tensor_runtime_common.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %d: %s\n", #expr, __LINE__, cudaGetErrorString(e)); \
    return 1; } } while (0)

template <typename T>
static T* dev_alloc_upload(const std::vector<T>& h) {
    T* p = nullptr;
    size_t n = h.empty() ? 1 : h.size();
    cudaMalloc(&p, sizeof(T) * n);
    if (!h.empty()) cudaMemcpy(p, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice);
    return p;
}

int main() {
    MkProblemSpec spec = {};
    spec.abi_version = MK_ABI_VERSION;
    spec.tensor_count = 64;
    spec.max_cells_per_tensor = 256;
    spec.max_tasks = 1024;
    spec.max_task_deps = 8;
    spec.max_task_outputs = 8;
    spec.scheduler_count = 16;
    spec.max_ready_per_scheduler = 1024;
    spec.max_popped_per_scheduler = 1024;
    spec.max_batch = 8192;
    spec.max_steps = 64;
    if (!mk_validate_problem_spec(&spec)) { std::fprintf(stderr, "bad spec\n"); return 1; }

    // Build a batch: define a tensor, then a wave of producers (ready) each with
    // one output cell, and consumers (blocked) on those cells, then signals.
    const int B = 2048;
    std::vector<int32_t> op_kind(B, 0);
    std::vector<uint32_t> tensor(B, 0), ci(B, 0), cj(B, 0), rank(B, 0), dim0(B, 0), dim1(B, 0), sched(B, 0);
    std::vector<uint64_t> task(B, 0), amount(B, 0), mask(B, 0), payload(B, 0);
    std::vector<uint32_t> doff(B, 0), dcnt(B, 0), ooff(B, 0), ocnt(B, 0);
    std::vector<uint32_t> dt, di, dj, ot, oi, oj;
    std::vector<uint64_t> oa;

    auto push_dep = [&](uint32_t r, uint32_t t, uint32_t i, uint32_t j) {
        if (dcnt[r] == 0) doff[r] = (uint32_t)dt.size();
        dt.push_back(t); di.push_back(i); dj.push_back(j); dcnt[r]++;
    };
    auto push_out = [&](uint32_t r, uint32_t t, uint32_t i, uint32_t j, uint64_t a) {
        if (ocnt[r] == 0) ooff[r] = (uint32_t)ot.size();
        ot.push_back(t); oi.push_back(i); oj.push_back(j); oa.push_back(a); ocnt[r]++;
    };

    int r = 0;
    // Define tensor 0 with 256 cells init=1.
    op_kind[r] = MK_OP_DEFINE_TENSOR; tensor[r] = 0; rank[r] = 1; dim0[r] = 256; amount[r] = 1; r++;
    uint64_t tid = 1;
    for (int c = 0; c < 200 && r < B - 1; ++c) {
        // producer: ready, output decrements cell c by 1.
        op_kind[r] = MK_OP_REGISTER_TASK; task[r] = tid++; sched[r] = c % spec.scheduler_count;
        push_out(r, 0, c, 0, 1); r++;
        if (r >= B) break;
        // consumer: blocked on cell c.
        op_kind[r] = MK_OP_REGISTER_TASK; task[r] = tid++; sched[r] = c % spec.scheduler_count;
        push_dep(r, 0, c, 0); r++;
    }
    int batch = r;

    if (dt.empty()) { dt.push_back(0); di.push_back(0); dj.push_back(0); }
    if (ot.empty()) { ot.push_back(0); oi.push_back(0); oj.push_back(0); oa.push_back(0); }

    MkInputs in = {};
    in.op_kind = dev_alloc_upload(op_kind); in.tensor_id = dev_alloc_upload(tensor);
    in.ci = dev_alloc_upload(ci); in.cj = dev_alloc_upload(cj); in.rank = dev_alloc_upload(rank);
    in.dim0 = dev_alloc_upload(dim0); in.dim1 = dev_alloc_upload(dim1);
    in.task_id = dev_alloc_upload(task); in.sched = dev_alloc_upload(sched);
    in.amount = dev_alloc_upload(amount); in.mask = dev_alloc_upload(mask);
    in.payload = dev_alloc_upload(payload); in.dep_off = dev_alloc_upload(doff);
    in.dep_count = dev_alloc_upload(dcnt); in.out_off = dev_alloc_upload(ooff);
    in.out_count = dev_alloc_upload(ocnt); in.dep_tensor = dev_alloc_upload(dt);
    in.dep_i = dev_alloc_upload(di); in.dep_j = dev_alloc_upload(dj);
    in.out_tensor = dev_alloc_upload(ot); in.out_i = dev_alloc_upload(oi);
    in.out_j = dev_alloc_upload(oj); in.out_amount = dev_alloc_upload(oa);

    MkOutputs out = {};
    int32_t* counts[18];
    for (int i = 0; i < 18; ++i) { cudaMalloc(&counts[i], sizeof(int32_t)); }
    int32_t** ocp = (int32_t**)&out;
    for (int i = 0; i < 18; ++i) ocp[i] = counts[i];
    uint64_t* h5[5];
    for (int i = 0; i < 5; ++i) cudaMalloc(&h5[i], sizeof(uint64_t));
    out.mk_event_hash = h5[0]; out.cell_hash = h5[1]; out.task_hash = h5[2];
    out.queue_hash = h5[3]; out.reverse_dep_hash = h5[4];

    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    void* state = nullptr;
    CK(solution_init(&spec, &state, stream));
    CK(cudaStreamSynchronize(stream));

    size_t wb = solution_workspace_bytes(&spec);
    if (wb == 0) wb = 1;
    void* ws = nullptr; CK(cudaMalloc(&ws, wb));

    MkRunSpec run = {}; run.abi_version = MK_ABI_VERSION; run.batch_size = batch; run.step_id = 0;

    // warmup
    CK(solution_run(state, &run, &in, &out, ws, wb, stream));
    CK(cudaStreamSynchronize(stream));

    const int iters = 200;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    CK(solution_reset(state, stream)); CK(cudaStreamSynchronize(stream));
    cudaEventRecord(a, stream);
    for (int it = 0; it < iters; ++it) {
        run.step_id = it;
        solution_run(state, &run, &in, &out, ws, wb, stream);
    }
    cudaEventRecord(b, stream);
    CK(cudaStreamSynchronize(stream));
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    double ops = (double)batch * iters;
    std::printf("mk_event_tensor_runtime bench: batch=%d iters=%d total_ms=%.3f us/step=%.2f Mops/s=%.2f\n",
        batch, iters, ms, (ms * 1000.0) / iters, ops / (ms * 1e3));

    solution_destroy(state);
    cudaStreamDestroy(stream);
    return 0;
}

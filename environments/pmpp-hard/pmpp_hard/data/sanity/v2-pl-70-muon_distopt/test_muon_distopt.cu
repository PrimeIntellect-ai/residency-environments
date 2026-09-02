// harness.cu - Solution-agnostic test driver for muon_distopt_stateful_v2
// Calls the solution ABI, feeds deterministic inputs per §6 case families,
// reads and prints RunOutputHeader checksums. Does NOT embed the reference algorithm.

#include "muon_distopt_common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

// ---- Solution ABI (provided by solution.cu) ----
extern "C" {
    size_t solution_workspace_bytes(const Spec* spec);
    cudaError_t solution_init(const Spec* spec, void** state, cudaStream_t stream);
    cudaError_t solution_run(void* state, const RunSpec* run, const void* inputs,
                             void* outputs, void* workspace, size_t workspace_bytes,
                             cudaStream_t stream);
    cudaError_t solution_reset(void* state, cudaStream_t stream);
    void solution_destroy(void* state);
}

// ---- CUDA error checking ----
#define CUDA_CHK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// ---- Host utilities ----
static uint32_t shard_start_h(uint32_t numel, uint32_t rank, uint32_t R) {
    return (uint32_t)(((uint64_t)numel * (uint64_t)rank) / (uint64_t)R);
}
static uint32_t shard_end_h(uint32_t numel, uint32_t rank, uint32_t R) {
    return (uint32_t)(((uint64_t)numel * (uint64_t)(rank + 1u)) / (uint64_t)R);
}
static uint64_t spec_total_numel(const Spec* spec) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < spec->num_params; ++i)
        total += (uint64_t)spec->params[i].rows * spec->params[i].cols;
    return total;
}

// splitmix64 for deterministic gradient generation
static uint64_t splitmix64_h(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Generate a small-magnitude normal BF16 gradient value (exp 120-135)
static uint16_t gen_grad_bf16(uint64_t seed) {
    uint64_t x = splitmix64_h(seed);
    uint16_t sign = (uint16_t)((x >> 63) & 1ull);
    uint16_t exp  = (uint16_t)(120u + (uint32_t)((x >> 56) & 0xFull));
    uint16_t mant = (uint16_t)((x >> 16) & 0x7Full);
    return (uint16_t)((sign << 15u) | (exp << 7u) | mant);
}

// ---- FNV aggregate over a family's per-run final_hashes ----
static uint64_t fnv_fold_u64(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (uint64_t)((v >> (8 * i)) & 0xFFull);
        h *= MD_FNV_PRIME;
    }
    return h;
}

// ---- Device buffer management ----
struct DeviceBuffers {
    void*  d_workspace;
    void*  d_inputs;
    void*  d_outputs;
    size_t workspace_size;
    size_t input_buf_size;
    size_t output_size;
};

static DeviceBuffers alloc_buffers(const Spec* spec) {
    DeviceBuffers b;
    b.workspace_size = solution_workspace_bytes(spec);
    uint64_t tn = spec_total_numel(spec);
    b.output_size = sizeof(RunOutputHeader) + sizeof(uint16_t) * (size_t)tn;
    // generous input buffer: max ops * sizeof(GradOp) + 64K BF16 grads
    b.input_buf_size = sizeof(GradOp) * (size_t)spec->max_ops_per_run
                     + sizeof(uint16_t) * 65536u;
    if (b.workspace_size == 0) b.workspace_size = 64; // must not be 0 for cudaMalloc
    CUDA_CHK(cudaMalloc(&b.d_workspace, b.workspace_size));
    CUDA_CHK(cudaMalloc(&b.d_inputs, b.input_buf_size));
    CUDA_CHK(cudaMalloc(&b.d_outputs, b.output_size));
    return b;
}
static void free_buffers(DeviceBuffers& b) {
    cudaFree(b.d_workspace);
    cudaFree(b.d_inputs);
    cudaFree(b.d_outputs);
}

// Upload ops+grads to device, call solution_run, copy header back.
// Returns the header; sets *ret_err to the cudaError_t from solution_run.
static RunOutputHeader run_one(
    void* state,
    DeviceBuffers& buf,
    cudaStream_t stream,
    const std::vector<GradOp>& ops,
    const std::vector<uint16_t>& grads,
    uint64_t run_tag,
    size_t ws_override,     // 0 = use buf.workspace_size
    cudaError_t* ret_err)
{
    // build host input buffer
    size_t ops_bytes   = sizeof(GradOp) * ops.size();
    size_t grads_bytes = sizeof(uint16_t) * grads.size();
    size_t total_in    = ops_bytes + grads_bytes;

    std::vector<uint8_t> host_in(total_in, 0);
    if (!ops.empty())   memcpy(host_in.data(), ops.data(), ops_bytes);
    if (!grads.empty()) memcpy(host_in.data() + ops_bytes, grads.data(), grads_bytes);

    CUDA_CHK(cudaMemcpyAsync(buf.d_inputs, host_in.data(), total_in,
                             cudaMemcpyHostToDevice, stream));

    RunSpec rs;
    memset(&rs, 0, sizeof(rs));
    rs.run_tag         = run_tag;
    rs.op_count        = (uint32_t)ops.size();
    rs.grad_value_count = (uint32_t)grads.size();
    rs.flags           = 0;
    rs.reserved        = 0;

    size_t ws = ws_override ? ws_override : buf.workspace_size;
    cudaError_t e = solution_run(state, &rs, buf.d_inputs, buf.d_outputs,
                                 buf.d_workspace, ws, stream);
    CUDA_CHK(cudaStreamSynchronize(stream));
    if (ret_err) *ret_err = e;

    RunOutputHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    CUDA_CHK(cudaMemcpy(&hdr, buf.d_outputs, sizeof(RunOutputHeader),
                        cudaMemcpyDeviceToHost));
    return hdr;
}

// Print one run's checksums
static void print_hdr(const char* fam, uint32_t step, const RunOutputHeader& h) {
    printf("%s step=%u status=%u attempted=%llu successful=%llu reset=%llu "
           "weight=0x%016llx state=0x%016llx timeline=0x%016llx "
           "counter=0x%016llx final=0x%016llx\n",
           fam, step,
           h.status,
           (unsigned long long)h.attempted_runs,
           (unsigned long long)h.successful_runs,
           (unsigned long long)h.reset_count,
           (unsigned long long)h.weight_hash,
           (unsigned long long)h.state_hash,
           (unsigned long long)h.timeline_hash,
           (unsigned long long)h.counter_hash,
           (unsigned long long)h.final_hash);
}

// Helper: build complete-shard ops for all params/ranks (no invalids/duplicates).
// step_seed differentiates gradient values across runs.
static void build_complete_ops(const Spec* spec, uint32_t step_seed,
                               std::vector<GradOp>& ops,
                               std::vector<uint16_t>& grads)
{
    ops.clear(); grads.clear();
    uint32_t grad_offset = 0;
    for (uint32_t pid = 0; pid < spec->num_params; ++pid) {
        uint32_t numel = spec->params[pid].rows * spec->params[pid].cols;
        for (uint32_t r = 0; r < spec->num_ranks; ++r) {
            uint32_t ss = shard_start_h(numel, r, spec->num_ranks);
            uint32_t se = shard_end_h(numel, r, spec->num_ranks);
            uint32_t slen = se - ss;
            if (slen == 0) continue;
            GradOp op = {};
            op.param_id      = pid;
            op.src_rank      = r;
            op.dst_rank      = r;
            op.owner_offset  = 0;
            op.elem_count    = slen;
            op.grad_offset   = grad_offset;
            ops.push_back(op);
            for (uint32_t k = 0; k < slen; ++k) {
                uint64_t seed = ((uint64_t)step_seed << 24)
                              | ((uint64_t)pid << 16)
                              | ((uint64_t)r << 8)
                              | (uint64_t)(k & 0xFFu);
                seed ^= (uint64_t)k >> 8;
                grads.push_back(gen_grad_bf16(seed));
            }
            grad_offset += slen;
        }
    }
}

// ============================================================
// FAMILY A: Simple warmup / state persistence
// 4 params: Muon 8x8, Muon 7x31, Adam 1x64, forced-Adam 16x16
// R=2, 5 runs, complete shards, no invalid ops.
// ============================================================
static uint64_t run_family_a() {
    printf("\n=== FAMILY_A: Simple warmup / state persistence ===\n");

    Spec spec; memset(&spec, 0, sizeof(spec));
    spec.magic           = MD_MAGIC;
    spec.version         = MD_VERSION;
    spec.num_params      = 4;
    spec.num_ranks       = 2;
    spec.max_ops_per_run = 32;
    spec.global_seed     = 0x1234567890A30394ull;

    // p0: Muon 8x8
    spec.params[0].rows = 8; spec.params[0].cols = 8;
    spec.params[0].storage_dtype = MD_DTYPE_BF16;
    spec.params[0].init_seed = 0xAABBCCDD00000000ull;
    // p1: Muon 7x31
    spec.params[1].rows = 7; spec.params[1].cols = 31;
    spec.params[1].storage_dtype = MD_DTYPE_BF16;
    spec.params[1].init_seed = 0xAABBCCDD00000001ull;
    // p2: Adam 1x64
    spec.params[2].rows = 1; spec.params[2].cols = 64;
    spec.params[2].storage_dtype = MD_DTYPE_BF16;
    spec.params[2].init_seed = 0xAABBCCDD00000002ull;
    // p3: forced-Adam 16x16
    spec.params[3].rows = 16; spec.params[3].cols = 16;
    spec.params[3].storage_dtype = MD_DTYPE_BF16;
    spec.params[3].flags = MD_PARAM_FORCE_ADAMW;
    spec.params[3].init_seed = 0xAABBCCDD00000003ull;

    cudaStream_t stream; CUDA_CHK(cudaStreamCreate(&stream));
    DeviceBuffers buf = alloc_buffers(&spec);
    void* state = nullptr;
    CUDA_CHK(solution_init(&spec, &state, stream));
    CUDA_CHK(cudaStreamSynchronize(stream));

    uint64_t agg = MD_FNV_BASIS;
    for (uint32_t step = 0; step < 5; ++step) {
        std::vector<GradOp> ops; std::vector<uint16_t> grads;
        build_complete_ops(&spec, step, ops, grads);
        cudaError_t e;
        RunOutputHeader hdr = run_one(state, buf, stream, ops, grads,
                                      0xA000ull | step, 0, &e);
        print_hdr("FAMILY_A", step, hdr);
        agg = fnv_fold_u64(agg, hdr.final_hash);
    }
    printf("FAMILY_A aggregate=0x%016llx\n", (unsigned long long)agg);

    solution_destroy(state);
    free_buffers(buf);
    CUDA_CHK(cudaStreamDestroy(stream));
    return agg;
}

// ============================================================
// FAMILY B: Transpose and ragged shard stress
// Muon: 127x17, 17x127, 3x251, 251x3; Adam: 1x1, 1x997
// R=3 (fixed choice from "3 or 5"; noted in REPORT.md).
// Ops arrive out-of-order by param (reverse param order in ops array).
// Missing rank-0 contribution for the first two Muon params.
// ============================================================
static uint64_t run_family_b() {
    printf("\n=== FAMILY_B: Transpose and ragged shard stress ===\n");

    Spec spec; memset(&spec, 0, sizeof(spec));
    spec.magic           = MD_MAGIC;
    spec.version         = MD_VERSION;
    spec.num_params      = 6;
    spec.num_ranks       = 3;
    spec.max_ops_per_run = 64;
    spec.global_seed     = 0xFEDCBA98765CFC6Bull;

    // p0: Muon 127x17
    spec.params[0].rows = 127; spec.params[0].cols = 17;
    spec.params[0].storage_dtype = MD_DTYPE_BF16;
    spec.params[0].init_seed = 0xBB00000000000000ull;
    // p1: Muon 17x127
    spec.params[1].rows = 17; spec.params[1].cols = 127;
    spec.params[1].storage_dtype = MD_DTYPE_BF16;
    spec.params[1].init_seed = 0xBB00000000000001ull;
    // p2: Muon 3x251
    spec.params[2].rows = 3; spec.params[2].cols = 251;
    spec.params[2].storage_dtype = MD_DTYPE_BF16;
    spec.params[2].init_seed = 0xBB00000000000002ull;
    // p3: Muon 251x3
    spec.params[3].rows = 251; spec.params[3].cols = 3;
    spec.params[3].storage_dtype = MD_DTYPE_BF16;
    spec.params[3].init_seed = 0xBB00000000000003ull;
    // p4: Adam 1x1
    spec.params[4].rows = 1; spec.params[4].cols = 1;
    spec.params[4].storage_dtype = MD_DTYPE_BF16;
    spec.params[4].init_seed = 0xBB00000000000004ull;
    // p5: Adam 1x997
    spec.params[5].rows = 1; spec.params[5].cols = 997;
    spec.params[5].storage_dtype = MD_DTYPE_BF16;
    spec.params[5].init_seed = 0xBB00000000000005ull;

    cudaStream_t stream; CUDA_CHK(cudaStreamCreate(&stream));
    DeviceBuffers buf = alloc_buffers(&spec);
    void* state = nullptr;
    CUDA_CHK(solution_init(&spec, &state, stream));
    CUDA_CHK(cudaStreamSynchronize(stream));

    uint64_t agg = MD_FNV_BASIS;

    // 3 runs: out-of-order param order, missing rank-0 for p0,p1
    for (uint32_t step = 0; step < 3; ++step) {
        std::vector<GradOp> ops; std::vector<uint16_t> grads;
        uint32_t grad_offset = 0;

        // Generate ops in REVERSE param order to stress op ordering.
        // Missing: rank-0 contributions for p0 and p1.
        for (int pid = (int)spec.num_params - 1; pid >= 0; --pid) {
            uint32_t numel = spec.params[pid].rows * spec.params[pid].cols;
            uint32_t start_rank = ((uint32_t)pid < 2u) ? 1u : 0u; // skip rank 0 for p0,p1
            for (uint32_t r = start_rank; r < spec.num_ranks; ++r) {
                uint32_t ss = shard_start_h(numel, r, spec.num_ranks);
                uint32_t se = shard_end_h(numel, r, spec.num_ranks);
                uint32_t slen = se - ss;
                if (slen == 0) continue;
                GradOp op = {};
                op.param_id     = (uint32_t)pid;
                op.src_rank     = r;
                op.dst_rank     = r;
                op.owner_offset = 0;
                op.elem_count   = slen;
                op.grad_offset  = grad_offset;
                ops.push_back(op);
                for (uint32_t k = 0; k < slen; ++k) {
                    uint64_t seed = 0xB000ull
                        | ((uint64_t)step << 32)
                        | ((uint64_t)pid << 20)
                        | ((uint64_t)r << 16)
                        | (uint64_t)(k & 0xFFFFu);
                    grads.push_back(gen_grad_bf16(seed));
                }
                grad_offset += slen;
            }
        }

        cudaError_t e;
        RunOutputHeader hdr = run_one(state, buf, stream, ops, grads,
                                      0xB000ull | step, 0, &e);
        print_hdr("FAMILY_B", step, hdr);
        agg = fnv_fold_u64(agg, hdr.final_hash);
    }
    printf("FAMILY_B aggregate=0x%016llx\n", (unsigned long long)agg);

    solution_destroy(state);
    free_buffers(buf);
    CUDA_CHK(cudaStreamDestroy(stream));
    return agg;
}

// ============================================================
// FAMILY C: Duplicate and overlap stress
// 4 params mixed Muon/Adam. R=4.
// Includes: exact duplicate GradOps, zero-length valid ops,
// two ops covering the same range but from different src_ranks
// (overlapping but not duplicate by key).
// ============================================================
static uint64_t run_family_c() {
    printf("\n=== FAMILY_C: Duplicate and overlap stress ===\n");

    Spec spec; memset(&spec, 0, sizeof(spec));
    spec.magic           = MD_MAGIC;
    spec.version         = MD_VERSION;
    spec.num_params      = 4;
    spec.num_ranks       = 4;
    spec.max_ops_per_run = 64;
    spec.global_seed     = 0xCAFEBABEDEA57094ull;

    // p0: Muon 8x8
    spec.params[0].rows = 8; spec.params[0].cols = 8;
    spec.params[0].storage_dtype = MD_DTYPE_BF16;
    spec.params[0].init_seed = 0xCC00000000000000ull;
    // p1: Muon 32x32
    spec.params[1].rows = 32; spec.params[1].cols = 32;
    spec.params[1].storage_dtype = MD_DTYPE_BF16;
    spec.params[1].init_seed = 0xCC00000000000001ull;
    // p2: Adam 1x128
    spec.params[2].rows = 1; spec.params[2].cols = 128;
    spec.params[2].storage_dtype = MD_DTYPE_BF16;
    spec.params[2].init_seed = 0xCC00000000000002ull;
    // p3: forced-Adam 8x4
    spec.params[3].rows = 8; spec.params[3].cols = 4;
    spec.params[3].storage_dtype = MD_DTYPE_BF16;
    spec.params[3].flags = MD_PARAM_FORCE_ADAMW;
    spec.params[3].init_seed = 0xCC00000000000003ull;

    cudaStream_t stream; CUDA_CHK(cudaStreamCreate(&stream));
    DeviceBuffers buf = alloc_buffers(&spec);
    void* state = nullptr;
    CUDA_CHK(solution_init(&spec, &state, stream));
    CUDA_CHK(cudaStreamSynchronize(stream));

    uint64_t agg = MD_FNV_BASIS;

    for (uint32_t step = 0; step < 3; ++step) {
        std::vector<GradOp> ops; std::vector<uint16_t> grads;
        uint32_t grad_offset = 0;

        // Normal complete ops for all params
        for (uint32_t pid = 0; pid < spec.num_params; ++pid) {
            uint32_t numel = spec.params[pid].rows * spec.params[pid].cols;
            for (uint32_t r = 0; r < spec.num_ranks; ++r) {
                uint32_t ss = shard_start_h(numel, r, spec.num_ranks);
                uint32_t se = shard_end_h(numel, r, spec.num_ranks);
                uint32_t slen = se - ss;
                if (slen == 0) continue;
                GradOp op = {};
                op.param_id     = pid;
                op.src_rank     = r;
                op.dst_rank     = r;
                op.owner_offset = 0;
                op.elem_count   = slen;
                op.grad_offset  = grad_offset;
                ops.push_back(op);
                for (uint32_t k = 0; k < slen; ++k) {
                    uint64_t seed = 0xC000ull | ((uint64_t)step<<24)
                                  | ((uint64_t)pid<<16) | ((uint64_t)r<<8) | k;
                    grads.push_back(gen_grad_bf16(seed));
                }
                grad_offset += slen;
            }
        }

        // Exact duplicate: repeat the first op (same 6-tuple key)
        if (!ops.empty()) {
            GradOp dup = ops[0]; // identical to ops[0]
            dup.grad_offset = ops[0].grad_offset; // same grad_offset => exact duplicate
            ops.push_back(dup);
            // No new grad values needed (same grad_offset → same data)
        }

        // Zero-length valid op for p0, rank 0, dst_rank 0
        {
            GradOp zop = {};
            zop.param_id     = 0;
            zop.src_rank     = 0;
            zop.dst_rank     = 0;
            zop.owner_offset = 0;
            zop.elem_count   = 0;  // zero-length valid op
            zop.grad_offset  = 0;  // valid offset (0 <= grad_value_count)
            ops.push_back(zop);
        }

        // Two ops covering overlapping range but different src_ranks (not duplicate by key)
        {
            uint32_t numel1 = spec.params[1].rows * spec.params[1].cols;
            uint32_t ss1 = shard_start_h(numel1, 1u, spec.num_ranks);
            uint32_t se1 = shard_end_h(numel1, 1u, spec.num_ranks);
            uint32_t slen1 = se1 - ss1;
            if (slen1 >= 4) {
                // op A: src=0, dst=1, first half
                GradOp opA = {};
                opA.param_id     = 1;
                opA.src_rank     = 0;     // different src from normal
                opA.dst_rank     = 1;
                opA.owner_offset = 0;
                opA.elem_count   = slen1 / 2;
                opA.grad_offset  = grad_offset;
                ops.push_back(opA);
                for (uint32_t k = 0; k < opA.elem_count; ++k) {
                    grads.push_back(gen_grad_bf16(0xC100ull | k | ((uint64_t)step<<16)));
                }
                grad_offset += opA.elem_count;

                // op B: src=0, dst=1, second half (overlaps with opA's range?)
                // No, they're adjacent, not overlapping. That's fine.
                GradOp opB = {};
                opB.param_id     = 1;
                opB.src_rank     = 0;
                opB.dst_rank     = 1;
                opB.owner_offset = 0;
                opB.elem_count   = slen1 / 2;
                opB.grad_offset  = grad_offset;
                ops.push_back(opB);
                for (uint32_t k = 0; k < opB.elem_count; ++k) {
                    grads.push_back(gen_grad_bf16(0xC200ull | k | ((uint64_t)step<<16)));
                }
                grad_offset += opB.elem_count;
                // opA and opB have same (param,src,dst,owner_offset,elem_count) but
                // different grad_offset → not duplicate by the 6-tuple key, but
                // they cover the same destination range (overlapping accumulation).
            }
        }

        cudaError_t e;
        RunOutputHeader hdr = run_one(state, buf, stream, ops, grads,
                                      0xC000ull | step, 0, &e);
        print_hdr("FAMILY_C", step, hdr);
        agg = fnv_fold_u64(agg, hdr.final_hash);
    }
    printf("FAMILY_C aggregate=0x%016llx\n", (unsigned long long)agg);

    solution_destroy(state);
    free_buffers(buf);
    CUDA_CHK(cudaStreamDestroy(stream));
    return agg;
}

// ============================================================
// FAMILY D: Nonfinite canonicalization
// Small shapes. Gradients include NaN, +/-Inf, -0.0, +0.0.
// ============================================================
static uint64_t run_family_d() {
    printf("\n=== FAMILY_D: Nonfinite canonicalization ===\n");

    Spec spec; memset(&spec, 0, sizeof(spec));
    spec.magic           = MD_MAGIC;
    spec.version         = MD_VERSION;
    spec.num_params      = 5;
    spec.num_ranks       = 2;
    spec.max_ops_per_run = 32;
    spec.global_seed     = 0xDEADDEADDEA510D6ull;

    // p0: Muon 4x4
    spec.params[0].rows = 4; spec.params[0].cols = 4;
    spec.params[0].storage_dtype = MD_DTYPE_BF16;
    spec.params[0].init_seed = 0xDD00000000000000ull;
    // p1: Muon 5x9
    spec.params[1].rows = 5; spec.params[1].cols = 9;
    spec.params[1].storage_dtype = MD_DTYPE_BF16;
    spec.params[1].init_seed = 0xDD00000000000001ull;
    // p2: Muon 9x5
    spec.params[2].rows = 9; spec.params[2].cols = 5;
    spec.params[2].storage_dtype = MD_DTYPE_BF16;
    spec.params[2].init_seed = 0xDD00000000000002ull;
    // p3: Adam 1x16
    spec.params[3].rows = 1; spec.params[3].cols = 16;
    spec.params[3].storage_dtype = MD_DTYPE_BF16;
    spec.params[3].init_seed = 0xDD00000000000003ull;
    // p4: forced-Adam 3x3
    spec.params[4].rows = 3; spec.params[4].cols = 3;
    spec.params[4].storage_dtype = MD_DTYPE_BF16;
    spec.params[4].flags = MD_PARAM_FORCE_ADAMW;
    spec.params[4].init_seed = 0xDD00000000000004ull;

    cudaStream_t stream; CUDA_CHK(cudaStreamCreate(&stream));
    DeviceBuffers buf = alloc_buffers(&spec);
    void* state = nullptr;
    CUDA_CHK(solution_init(&spec, &state, stream));
    CUDA_CHK(cudaStreamSynchronize(stream));

    // Special BF16 values
    static const uint16_t BF16_NAN1  = 0x7FC1u; // NaN with nonzero mant
    static const uint16_t BF16_NAN2  = 0x7FFFu; // NaN
    static const uint16_t BF16_NAN3  = 0xFFC1u; // NaN (negative)
    static const uint16_t BF16_PINF  = 0x7F80u; // +Inf
    static const uint16_t BF16_NINF  = 0xFF80u; // -Inf
    static const uint16_t BF16_NEG0  = 0x8000u; // -0.0
    static const uint16_t BF16_POS0  = 0x0000u; // +0.0
    static const uint16_t BF16_ONE   = 0x3F80u; // 1.0

    uint64_t agg = MD_FNV_BASIS;

    for (uint32_t step = 0; step < 2; ++step) {
        std::vector<GradOp> ops; std::vector<uint16_t> grads;
        uint32_t grad_offset = 0;

        // Array of special values to cycle through
        uint16_t specials[] = {BF16_NAN1, BF16_NAN2, BF16_NAN3, BF16_PINF,
                               BF16_NINF, BF16_NEG0, BF16_POS0, BF16_ONE};
        uint32_t nspec = 8;

        for (uint32_t pid = 0; pid < spec.num_params; ++pid) {
            uint32_t numel = spec.params[pid].rows * spec.params[pid].cols;
            for (uint32_t r = 0; r < spec.num_ranks; ++r) {
                uint32_t ss = shard_start_h(numel, r, spec.num_ranks);
                uint32_t se = shard_end_h(numel, r, spec.num_ranks);
                uint32_t slen = se - ss;
                if (slen == 0) continue;
                GradOp op = {};
                op.param_id     = pid;
                op.src_rank     = r;
                op.dst_rank     = r;
                op.owner_offset = 0;
                op.elem_count   = slen;
                op.grad_offset  = grad_offset;
                ops.push_back(op);
                // Mix special values and normal values
                for (uint32_t k = 0; k < slen; ++k) {
                    uint16_t v;
                    uint32_t idx = (pid * spec.num_ranks * 97u + r * 13u + k) % (nspec * 3u);
                    if (idx < nspec) {
                        v = specials[idx]; // special value
                    } else {
                        uint64_t seed = 0xD000ull | ((uint64_t)step<<24)
                                      | ((uint64_t)pid<<16) | ((uint64_t)r<<8) | k;
                        v = gen_grad_bf16(seed);
                    }
                    grads.push_back(v);
                }
                grad_offset += slen;
            }
        }

        cudaError_t e;
        RunOutputHeader hdr = run_one(state, buf, stream, ops, grads,
                                      0xD000ull | step, 0, &e);
        print_hdr("FAMILY_D", step, hdr);
        printf("FAMILY_D step=%u nan_inputs=%llu inf_inputs=%llu\n",
               step,
               (unsigned long long)hdr.last_nan_inputs,
               (unsigned long long)hdr.last_inf_inputs);
        agg = fnv_fold_u64(agg, hdr.final_hash);
    }
    printf("FAMILY_D aggregate=0x%016llx\n", (unsigned long long)agg);

    solution_destroy(state);
    free_buffers(buf);
    CUDA_CHK(cudaStreamDestroy(stream));
    return agg;
}

// ============================================================
// FAMILY E: Invalid op handling
// R=7. Mix of invalid ops (each invalid reason) + valid ops.
// ============================================================
static uint64_t run_family_e() {
    printf("\n=== FAMILY_E: Invalid op handling ===\n");

    Spec spec; memset(&spec, 0, sizeof(spec));
    spec.magic           = MD_MAGIC;
    spec.version         = MD_VERSION;
    spec.num_params      = 3;
    spec.num_ranks       = 7;
    spec.max_ops_per_run = 64;
    spec.global_seed     = 0xE1E2E3E4E5EE2993ull;

    // p0: Muon 8x8
    spec.params[0].rows = 8; spec.params[0].cols = 8;
    spec.params[0].storage_dtype = MD_DTYPE_BF16;
    spec.params[0].init_seed = 0xEE00000000000000ull;
    // p1: Adam 1x32
    spec.params[1].rows = 1; spec.params[1].cols = 32;
    spec.params[1].storage_dtype = MD_DTYPE_BF16;
    spec.params[1].init_seed = 0xEE00000000000001ull;
    // p2: Muon 16x16
    spec.params[2].rows = 16; spec.params[2].cols = 16;
    spec.params[2].storage_dtype = MD_DTYPE_BF16;
    spec.params[2].init_seed = 0xEE00000000000002ull;

    cudaStream_t stream; CUDA_CHK(cudaStreamCreate(&stream));
    DeviceBuffers buf = alloc_buffers(&spec);
    void* state = nullptr;
    CUDA_CHK(solution_init(&spec, &state, stream));
    CUDA_CHK(cudaStreamSynchronize(stream));

    uint64_t agg = MD_FNV_BASIS;

    for (uint32_t step = 0; step < 2; ++step) {
        std::vector<GradOp> ops; std::vector<uint16_t> grads;
        uint32_t grad_offset = 0;

        // ---- One valid op per param (rank 0 → shard 0) ----
        for (uint32_t pid = 0; pid < spec.num_params; ++pid) {
            uint32_t numel = spec.params[pid].rows * spec.params[pid].cols;
            uint32_t ss = shard_start_h(numel, 0u, spec.num_ranks);
            uint32_t se = shard_end_h(numel, 0u, spec.num_ranks);
            uint32_t slen = se - ss;
            if (slen > 0) {
                GradOp op = {};
                op.param_id     = pid;
                op.src_rank     = 0;
                op.dst_rank     = 0;
                op.owner_offset = 0;
                op.elem_count   = slen;
                op.grad_offset  = grad_offset;
                ops.push_back(op);
                for (uint32_t k = 0; k < slen; ++k) {
                    grads.push_back(gen_grad_bf16(0xE100ull | k | ((uint64_t)step<<16)));
                }
                grad_offset += slen;
            }
        }

        // ---- Invalid ops (one per reason code) ----

        // Reason 1: bad param_id
        {
            GradOp op = {};
            op.param_id     = spec.num_params; // invalid
            op.src_rank     = 0;
            op.dst_rank     = 0;
            op.owner_offset = 0;
            op.elem_count   = 1;
            op.grad_offset  = 0;
            ops.push_back(op);
        }

        // Reason 2: bad src_rank
        {
            GradOp op = {};
            op.param_id     = 0;
            op.src_rank     = spec.num_ranks; // invalid
            op.dst_rank     = 0;
            op.owner_offset = 0;
            op.elem_count   = 1;
            op.grad_offset  = 0;
            ops.push_back(op);
        }

        // Reason 3: bad dst_rank
        {
            GradOp op = {};
            op.param_id     = 0;
            op.src_rank     = 0;
            op.dst_rank     = spec.num_ranks; // invalid
            op.owner_offset = 0;
            op.elem_count   = 1;
            op.grad_offset  = 0;
            ops.push_back(op);
        }

        // Reason 4: owner range overflow (elem_count > shard_len - owner_offset)
        {
            uint32_t numel0 = spec.params[0].rows * spec.params[0].cols;
            uint32_t ss0 = shard_start_h(numel0, 0u, spec.num_ranks);
            uint32_t se0 = shard_end_h(numel0, 0u, spec.num_ranks);
            uint32_t slen0 = se0 - ss0;
            GradOp op = {};
            op.param_id     = 0;
            op.src_rank     = 0;
            op.dst_rank     = 0;
            op.owner_offset = 0;
            op.elem_count   = slen0 + 1; // overflow
            op.grad_offset  = 0;
            ops.push_back(op);
        }

        // Reason 5: grad range overflow (grad_offset > grad_value_count).
        // The actual grad_offset is patched AFTER all grad values are appended
        // (see fixup below) so it is unambiguously beyond the final grad buffer.
        // Otherwise the interleaved valid op appended later would grow
        // grad_value_count enough to make this op resolve as VALID.
        size_t reason5_op_index;
        {
            uint32_t numel0 = spec.params[0].rows * spec.params[0].cols;
            uint32_t slen0 = shard_end_h(numel0, 0u, spec.num_ranks)
                           - shard_start_h(numel0, 0u, spec.num_ranks);
            GradOp op = {};
            op.param_id     = 0;
            op.src_rank     = 1;
            op.dst_rank     = 0;
            op.owner_offset = 0;
            op.elem_count   = slen0;
            op.grad_offset  = 0; // placeholder; patched after final grad append
            reason5_op_index = ops.size();
            ops.push_back(op);
        }

        // ---- Interleaved valid op after invalid ops ----
        {
            uint32_t numel2 = spec.params[2].rows * spec.params[2].cols;
            uint32_t ss2 = shard_start_h(numel2, 1u, spec.num_ranks);
            uint32_t se2 = shard_end_h(numel2, 1u, spec.num_ranks);
            uint32_t slen2 = se2 - ss2;
            if (slen2 > 0) {
                GradOp op = {};
                op.param_id     = 2;
                op.src_rank     = 1;
                op.dst_rank     = 1;
                op.owner_offset = 0;
                op.elem_count   = slen2;
                op.grad_offset  = grad_offset;
                ops.push_back(op);
                for (uint32_t k = 0; k < slen2; ++k) {
                    grads.push_back(gen_grad_bf16(0xE200ull | k | ((uint64_t)step<<16)));
                }
                grad_offset += slen2;
            }
        }

        // Fixup for the reason-5 (grad range overflow) op: now that every grad
        // value for this run has been appended, set grad_offset strictly beyond
        // the final grad_value_count so the op is unambiguously invalid.
        ops[reason5_op_index].grad_offset = (uint32_t)grads.size() + 1u;

        cudaError_t e;
        RunOutputHeader hdr = run_one(state, buf, stream, ops, grads,
                                      0xE000ull | step, 0, &e);
        print_hdr("FAMILY_E", step, hdr);
        printf("FAMILY_E step=%u valid=%u invalid=%u\n",
               step, hdr.last_valid_ops, hdr.last_invalid_ops);
        agg = fnv_fold_u64(agg, hdr.final_hash);
    }
    printf("FAMILY_E aggregate=0x%016llx\n", (unsigned long long)agg);

    solution_destroy(state);
    free_buffers(buf);
    CUDA_CHK(cudaStreamDestroy(stream));
    return agg;
}

// ============================================================
// FAMILY F: Reset / OOM
// Run 3 valid steps, reset, re-run first step, then OOM test.
// ============================================================
static uint64_t run_family_f() {
    printf("\n=== FAMILY_F: Reset / OOM ===\n");

    Spec spec; memset(&spec, 0, sizeof(spec));
    spec.magic           = MD_MAGIC;
    spec.version         = MD_VERSION;
    spec.num_params      = 3;
    spec.num_ranks       = 2;
    spec.max_ops_per_run = 32;
    spec.global_seed     = 0xF0F1F2F3F4FD388Cull;

    // p0: Muon 8x8
    spec.params[0].rows = 8; spec.params[0].cols = 8;
    spec.params[0].storage_dtype = MD_DTYPE_BF16;
    spec.params[0].init_seed = 0xFF00000000000000ull;
    // p1: Adam 1x32
    spec.params[1].rows = 1; spec.params[1].cols = 32;
    spec.params[1].storage_dtype = MD_DTYPE_BF16;
    spec.params[1].init_seed = 0xFF00000000000001ull;
    // p2: forced-Adam 4x4
    spec.params[2].rows = 4; spec.params[2].cols = 4;
    spec.params[2].storage_dtype = MD_DTYPE_BF16;
    spec.params[2].flags = MD_PARAM_FORCE_ADAMW;
    spec.params[2].init_seed = 0xFF00000000000002ull;

    cudaStream_t stream; CUDA_CHK(cudaStreamCreate(&stream));
    DeviceBuffers buf = alloc_buffers(&spec);
    void* state = nullptr;
    CUDA_CHK(solution_init(&spec, &state, stream));
    CUDA_CHK(cudaStreamSynchronize(stream));

    uint64_t agg = MD_FNV_BASIS;

    // Store first step's ops/grads for replay after reset
    std::vector<GradOp> step0_ops; std::vector<uint16_t> step0_grads;

    // 3 valid steps
    for (uint32_t step = 0; step < 3; ++step) {
        std::vector<GradOp> ops; std::vector<uint16_t> grads;
        build_complete_ops(&spec, step, ops, grads);
        if (step == 0) { step0_ops = ops; step0_grads = grads; }
        cudaError_t e;
        RunOutputHeader hdr = run_one(state, buf, stream, ops, grads,
                                      0xF000ull | step, 0, &e);
        print_hdr("FAMILY_F", step, hdr);
        agg = fnv_fold_u64(agg, hdr.final_hash);
    }

    // Reset
    CUDA_CHK(solution_reset(state, stream));
    CUDA_CHK(cudaStreamSynchronize(stream));
    printf("FAMILY_F reset called\n");

    // Re-run step 0 (same ops, same grads, same run_tag)
    {
        cudaError_t e;
        RunOutputHeader hdr = run_one(state, buf, stream, step0_ops, step0_grads,
                                      0xF000ull | 0u, 0, &e);
        print_hdr("FAMILY_F post-reset-step0", 0, hdr);
        printf("FAMILY_F post-reset reset_count=%llu\n",
               (unsigned long long)hdr.reset_count);
        agg = fnv_fold_u64(agg, hdr.final_hash);
    }

    // OOM test: workspace_bytes = required - 1
    if (buf.workspace_size > 0) {
        cudaError_t e;
        RunOutputHeader hdr = run_one(state, buf, stream, step0_ops, step0_grads,
                                      0xF0FFull, buf.workspace_size - 1, &e);
        printf("FAMILY_F OOM: solution_run returned %d (expect %d=cudaErrorMemoryAllocation) "
               "status=%u attempted=%llu successful=%llu\n",
               (int)e, (int)cudaErrorMemoryAllocation,
               hdr.status,
               (unsigned long long)hdr.attempted_runs,
               (unsigned long long)hdr.successful_runs);
        agg = fnv_fold_u64(agg, hdr.final_hash);
    }

    printf("FAMILY_F aggregate=0x%016llx\n", (unsigned long long)agg);

    solution_destroy(state);
    free_buffers(buf);
    CUDA_CHK(cudaStreamDestroy(stream));
    return agg;
}

// ============================================================
// Golden per-family aggregates (embedded; see GOLDEN.txt / REPORT.md).
// Each aggregate is the FNV fold (MD_FNV_BASIS / MD_FNV_PRIME) over that
// family's per-run final_hashes, transitively pinning every per-step hash.
// ============================================================
struct GoldenFamily { const char* name; uint64_t aggregate; };
static const GoldenFamily GOLDEN[6] = {
    { "FAMILY_A", 0x9f885f7090a55cd5ull },
    { "FAMILY_B", 0x8b61d56163d05997ull },
    { "FAMILY_C", 0x78fba917c31e2810ull },
    { "FAMILY_D", 0x639b8601ffe9c23bull },
    { "FAMILY_E", 0xf9f11abe0dda1d3full },
    { "FAMILY_F", 0x0769f9a067031d43ull },
};

// ============================================================
// main
// ============================================================
int main() {
    // Print CUDA device info
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    uint64_t got[6];
    got[0] = run_family_a();
    got[1] = run_family_b();
    got[2] = run_family_c();
    got[3] = run_family_d();
    got[4] = run_family_e();
    got[5] = run_family_f();

    // Compare each family's aggregate to embedded golden; count passes.
    int passed = 0;
    printf("\n=== Family checksum verdicts ===\n");
    for (int i = 0; i < 6; ++i) {
        bool ok = (got[i] == GOLDEN[i].aggregate);
        if (ok) ++passed;
        printf("%s aggregate=0x%016llx golden=0x%016llx %s\n",
               GOLDEN[i].name,
               (unsigned long long)got[i],
               (unsigned long long)GOLDEN[i].aggregate,
               ok ? "OK" : "MISMATCH");
    }
    printf("passed %d / 6\n", passed);

    CUDA_CHK(cudaDeviceReset());
    printf("\nDone.\n");
    return 0;
}

// harness.cu — solution-agnostic test driver for stateful_norm_bwd_cachegrad_v2
//
// Reads outputs via the ABI only (InputPtrs / OutputPtrs / solution_* calls).
// Does NOT embed or call any reference algorithm.
// Computes FNV-1a-64 checksums exactly as §2.12 specifies and prints them.

#include "norm_backward_common.h"
#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// FNV-1a-64 helpers (§2.12)
// ---------------------------------------------------------------------------
static inline uint64_t fnv_byte(uint64_t h, uint8_t b) {
    return (h ^ uint64_t(b)) * LNR_FNV_PRIME;
}

static uint64_t fnv_bytes(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) h = fnv_byte(h, p[i]);
    return h;
}

// fold a u8
static inline uint64_t fnv_u8(uint64_t h, uint8_t v)  { return fnv_bytes(h, &v, 1); }
// fold a u32 little-endian
static inline uint64_t fnv_u32(uint64_t h, uint32_t v) { return fnv_bytes(h, &v, 4); }
// fold a u64 little-endian
static inline uint64_t fnv_u64(uint64_t h, uint64_t v) { return fnv_bytes(h, &v, 8); }
// fold a fp32 as raw 4 bytes little-endian
static inline uint64_t fnv_f32(uint64_t h, float v)    { return fnv_bytes(h, &v, 4); }

// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr) do {                                       \
    cudaError_t _e = (expr);                                        \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
            cudaGetErrorString(_e), __FILE__, __LINE__);            \
        exit(1);                                                    \
    }                                                               \
} while(0)

// ---------------------------------------------------------------------------
// Host copies of device outputs (allocated by harness)
// ---------------------------------------------------------------------------
struct HostOutputs {
    std::vector<uint8_t> dx;          // storage_dtype bytes
    std::vector<float>   flush_dgamma;
    std::vector<float>   flush_dbeta;
    std::vector<float>   accum_dgamma;
    std::vector<float>   accum_dbeta;
    std::vector<float>   cache_mean;
    std::vector<float>   cache_rstd;
    std::vector<uint32_t> cache_kind;
    std::vector<uint64_t> cache_gen;
    std::vector<TimelineRecord> timeline;
    CounterSnapshot counters;
};

// ---------------------------------------------------------------------------
// Device output buffer set
// ---------------------------------------------------------------------------
struct DevOutputs {
    void*           dx              = nullptr;
    float*          flush_dgamma    = nullptr;
    float*          flush_dbeta     = nullptr;
    float*          accum_dgamma    = nullptr;
    float*          accum_dbeta     = nullptr;
    float*          cache_mean      = nullptr;
    float*          cache_rstd      = nullptr;
    uint32_t*       cache_kind      = nullptr;
    uint64_t*       cache_gen       = nullptr;
    TimelineRecord* timeline        = nullptr;
    CounterSnapshot* counters       = nullptr;
};

static void alloc_dev_outputs(
    DevOutputs& d,
    const Spec& s,
    uint32_t dx_rows,
    uint32_t flush_records,
    uint32_t op_count)
{
    size_t elem = (s.storage_dtype == LNR_DTYPE_BF16) ? 2 : 4;
    CUDA_CHECK(cudaMalloc(&d.dx, size_t(dx_rows) * s.hidden_size * elem + 1));
    CUDA_CHECK(cudaMalloc(&d.flush_dgamma, size_t(flush_records) * s.hidden_size * sizeof(float) + 4));
    CUDA_CHECK(cudaMalloc(&d.flush_dbeta,  size_t(flush_records) * s.hidden_size * sizeof(float) + 4));
    CUDA_CHECK(cudaMalloc(&d.accum_dgamma, size_t(s.param_count) * s.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.accum_dbeta,  size_t(s.param_count) * s.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.cache_mean, size_t(s.max_cache_rows) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.cache_rstd, size_t(s.max_cache_rows) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.cache_kind, size_t(s.max_cache_rows) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d.cache_gen,  size_t(s.max_cache_rows) * sizeof(uint64_t)));
    if (op_count > 0)
        CUDA_CHECK(cudaMalloc(&d.timeline, size_t(op_count) * sizeof(TimelineRecord)));
    CUDA_CHECK(cudaMalloc(&d.counters, sizeof(CounterSnapshot)));
}

static void free_dev_outputs(DevOutputs& d) {
    cudaFree(d.dx);
    cudaFree(d.flush_dgamma);
    cudaFree(d.flush_dbeta);
    cudaFree(d.accum_dgamma);
    cudaFree(d.accum_dbeta);
    cudaFree(d.cache_mean);
    cudaFree(d.cache_rstd);
    cudaFree(d.cache_kind);
    cudaFree(d.cache_gen);
    cudaFree(d.timeline);
    cudaFree(d.counters);
}

static void copy_dev_to_host(
    HostOutputs& h,
    const DevOutputs& d,
    const Spec& s,
    uint32_t dx_rows,
    uint32_t flush_records,
    uint32_t op_count)
{
    size_t elem = (s.storage_dtype == LNR_DTYPE_BF16) ? 2 : 4;
    size_t dx_bytes = size_t(dx_rows) * s.hidden_size * elem;
    h.dx.resize(dx_bytes);
    if (dx_bytes) CUDA_CHECK(cudaMemcpy(h.dx.data(), d.dx, dx_bytes, cudaMemcpyDeviceToHost));

    size_t fg_elems = size_t(flush_records) * s.hidden_size;
    h.flush_dgamma.resize(fg_elems);
    h.flush_dbeta.resize(fg_elems);
    if (fg_elems) {
        CUDA_CHECK(cudaMemcpy(h.flush_dgamma.data(), d.flush_dgamma, fg_elems * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h.flush_dbeta.data(),  d.flush_dbeta,  fg_elems * sizeof(float), cudaMemcpyDeviceToHost));
    }

    size_t pn = size_t(s.param_count) * s.hidden_size;
    h.accum_dgamma.resize(pn);
    h.accum_dbeta.resize(pn);
    CUDA_CHECK(cudaMemcpy(h.accum_dgamma.data(), d.accum_dgamma, pn * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.accum_dbeta.data(),  d.accum_dbeta,  pn * sizeof(float), cudaMemcpyDeviceToHost));

    h.cache_mean.resize(s.max_cache_rows);
    h.cache_rstd.resize(s.max_cache_rows);
    h.cache_kind.resize(s.max_cache_rows);
    h.cache_gen.resize(s.max_cache_rows);
    CUDA_CHECK(cudaMemcpy(h.cache_mean.data(), d.cache_mean, size_t(s.max_cache_rows) * sizeof(float),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.cache_rstd.data(), d.cache_rstd, size_t(s.max_cache_rows) * sizeof(float),    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.cache_kind.data(), d.cache_kind, size_t(s.max_cache_rows) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.cache_gen.data(),  d.cache_gen,  size_t(s.max_cache_rows) * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    h.timeline.resize(op_count);
    if (op_count)
        CUDA_CHECK(cudaMemcpy(h.timeline.data(), d.timeline, size_t(op_count) * sizeof(TimelineRecord), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(&h.counters, d.counters, sizeof(CounterSnapshot), cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Checksum computation (§2.12) — harness knows only byte layouts from common.h
// ---------------------------------------------------------------------------
struct RunChecksums {
    uint64_t H_dx;
    uint64_t H_flush_dgamma;
    uint64_t H_flush_dbeta;
    uint64_t H_accum_dgamma;
    uint64_t H_accum_dbeta;
    uint64_t H_cache;
    uint64_t H_timeline;
    uint64_t H_counters;
    uint64_t H_all;
};

static RunChecksums compute_checksums(
    const HostOutputs& h,
    const Spec& s,
    uint32_t dx_rows,
    uint32_t flush_records,
    uint32_t op_count)
{
    const uint64_t B = LNR_FNV_BASIS;

    // H_dx: dx_rows x hidden_size, storage_dtype raw bytes
    uint64_t H_dx = B;
    H_dx = fnv_bytes(H_dx, h.dx.data(), h.dx.size());

    // H_flush_dgamma: flush_records x hidden_size fp32
    uint64_t H_fdg = B;
    H_fdg = fnv_bytes(H_fdg, h.flush_dgamma.data(), h.flush_dgamma.size() * sizeof(float));

    // H_flush_dbeta: flush_records x hidden_size fp32
    uint64_t H_fdb = B;
    H_fdb = fnv_bytes(H_fdb, h.flush_dbeta.data(), h.flush_dbeta.size() * sizeof(float));

    // H_accum_dgamma: param_count x hidden_size fp32
    uint64_t H_adg = B;
    H_adg = fnv_bytes(H_adg, h.accum_dgamma.data(), h.accum_dgamma.size() * sizeof(float));

    // H_accum_dbeta: param_count x hidden_size fp32
    uint64_t H_adb = B;
    H_adb = fnv_bytes(H_adb, h.accum_dbeta.data(), h.accum_dbeta.size() * sizeof(float));

    // H_cache: mean fp32 [max_cache_rows], rstd fp32, kind u32, gen u64 in that order
    uint64_t H_cache = B;
    H_cache = fnv_bytes(H_cache, h.cache_mean.data(), h.cache_mean.size() * sizeof(float));
    H_cache = fnv_bytes(H_cache, h.cache_rstd.data(), h.cache_rstd.size() * sizeof(float));
    H_cache = fnv_bytes(H_cache, h.cache_kind.data(), h.cache_kind.size() * sizeof(uint32_t));
    H_cache = fnv_bytes(H_cache, h.cache_gen.data(),  h.cache_gen.size()  * sizeof(uint64_t));

    // H_timeline: for each op fold TimelineRecord fields in common.h order
    // Fields per TimelineRecord (§2 common.h):
    //   run_id u64, global_op_index u64,
    //   op_index_in_run u32, kind u8, status u8, cache_kind_required u8, reserved_a u8,
    //   param_id u32, rows u32, x_row_base u32, dy_row_base u32,
    //   cache_base u32, dx_out_base u32, flush_out_base u32,
    //   partial_base u32, partial_count u32,
    //   cache_generation_first u64, cache_generation_last u64,
    //   counter_snapshot_after_op u64
    uint64_t H_tl = B;
    for (uint32_t i = 0; i < op_count; ++i) {
        const TimelineRecord& r = h.timeline[i];
        H_tl = fnv_u64(H_tl, r.run_id);
        H_tl = fnv_u64(H_tl, r.global_op_index);
        H_tl = fnv_u32(H_tl, r.op_index_in_run);
        H_tl = fnv_u8 (H_tl, r.kind);
        H_tl = fnv_u8 (H_tl, r.status);
        H_tl = fnv_u8 (H_tl, r.cache_kind_required);
        H_tl = fnv_u8 (H_tl, r.reserved_a);
        H_tl = fnv_u32(H_tl, r.param_id);
        H_tl = fnv_u32(H_tl, r.rows);
        H_tl = fnv_u32(H_tl, r.x_row_base);
        H_tl = fnv_u32(H_tl, r.dy_row_base);
        H_tl = fnv_u32(H_tl, r.cache_base);
        H_tl = fnv_u32(H_tl, r.dx_out_base);
        H_tl = fnv_u32(H_tl, r.flush_out_base);
        H_tl = fnv_u32(H_tl, r.partial_base);
        H_tl = fnv_u32(H_tl, r.partial_count);
        H_tl = fnv_u64(H_tl, r.cache_generation_first);
        H_tl = fnv_u64(H_tl, r.cache_generation_last);
        H_tl = fnv_u64(H_tl, r.counter_snapshot_after_op);
    }

    // H_counters: fold CounterSnapshot fields in common.h order
    uint64_t H_cnt = B;
    {
        const CounterSnapshot& c = h.counters;
        H_cnt = fnv_u64(H_cnt, c.run_count);
        H_cnt = fnv_u64(H_cnt, c.op_count);
        H_cnt = fnv_u64(H_cnt, c.save_ln_rows);
        H_cnt = fnv_u64(H_cnt, c.save_rms_rows);
        H_cnt = fnv_u64(H_cnt, c.bwd_ln_rows);
        H_cnt = fnv_u64(H_cnt, c.bwd_rms_rows);
        H_cnt = fnv_u64(H_cnt, c.partial_blocks);
        H_cnt = fnv_u64(H_cnt, c.flush_count);
        H_cnt = fnv_u64(H_cnt, c.invalid_ops);
        H_cnt = fnv_u64(H_cnt, c.cache_miss_ops);
        H_cnt = fnv_u64(H_cnt, c.cache_overwrite_rows);
        H_cnt = fnv_u64(H_cnt, c.cache_generation_counter);
        H_cnt = fnv_u64(H_cnt, c.last_run_id);
    }

    // H_all: start from basis, fold eight u64 hashes
    uint64_t H_all = B;
    H_all = fnv_u64(H_all, H_dx);
    H_all = fnv_u64(H_all, H_fdg);
    H_all = fnv_u64(H_all, H_fdb);
    H_all = fnv_u64(H_all, H_adg);
    H_all = fnv_u64(H_all, H_adb);
    H_all = fnv_u64(H_all, H_cache);
    H_all = fnv_u64(H_all, H_tl);
    H_all = fnv_u64(H_all, H_cnt);

    return RunChecksums{H_dx, H_fdg, H_fdb, H_adg, H_adb, H_cache, H_tl, H_cnt, H_all};
}

// ---------------------------------------------------------------------------
// Embedded golden checksums (from GOLDEN.txt — self-contained, no external file)
// Order per case: H_dx, H_flush_dgamma, H_flush_dbeta, H_accum_dgamma,
//                 H_accum_dbeta, H_cache, H_timeline, H_counters, H_all
// ---------------------------------------------------------------------------
struct GoldenCase {
    const char* label;
    uint64_t h[9];
};

static const GoldenCase g_golden[] = {
    {"A.run0",  {0xf0be6b7acbd0496cull,0xfefbfd440105419full,0xc9e9d1156d0e5ba9ull,0xdce53c1df8560f83ull,0xdce53c1df8560f83ull,0x7deb5d524c1ebe1aull,0x3d2dc569e567d2cfull,0x11b6bc7aa7a5647dull,0x183009462ef33e51ull}},
    {"B.run0",  {0x6a659ade2009f023ull,0x897b8352042abb90ull,0x218035740d3f1b83ull,0x218035740d3f1b83ull,0x218035740d3f1b83ull,0x6672d50b35a7c744ull,0x4f5c114fa1e4428bull,0x2448136e1b34be1eull,0x96e82d8823a66930ull}},
    {"C.run0",  {0xfb2a63fa06080b5cull,0x25371b62bda4889bull,0x62d9a2c1c5c4dd7aull,0xe8c207397dd472b4ull,0xa75008440762b1e3ull,0xe958a9163455eca9ull,0x5694177cd48ae597ull,0x4349a4acadeec1d8ull,0x368d78c8acda4aedull}},
    {"D.run0",  {0x1d97b37c33aa342aull,0xb43a063055adc383ull,0xb43a063055adc383ull,0xc8e764882962c6f7ull,0xb00e708220bcb583ull,0x49660cafe292f4efull,0xfdf2a5e6ff5f6943ull,0xc7978cb5cfc458a3ull,0xc188f8cfa685539eull}},
    {"D.run1",  {0xb4f071f0799a4c5dull,0xb43a063055adc383ull,0xb43a063055adc383ull,0x34a2315c8bd852afull,0xe876d644517d35e1ull,0xc0038bd68ca64c3dull,0xa51b7bfc32df9b98ull,0xeb7740e07841cbeeull,0x4f1035e974cc249bull}},
    {"D.run2",  {0x14650fb0739d0383ull,0xbf8b7d0c02a0d2afull,0xb0f61feca46cb5e1ull,0x4b404aa0a9cf4383ull,0x4b404aa0a9cf4383ull,0xc0038bd68ca64c3dull,0xbf9ea5641987f349ull,0x079e551287b37fc0ull,0xeb9619282865d9e0ull}},
    {"E.run0",  {0x14650fb0739d0383ull,0x14650fb0739d0383ull,0x14650fb0739d0383ull,0xdce53c1df8560f83ull,0xdce53c1df8560f83ull,0xd5561a2b36e2d18eull,0xbf2217b2889118fdull,0xc2f306205a33509dull,0xd1c27cbc459587bbull}},
    {"E.run1",  {0xf43058da88decc0dull,0xdce53c1df8560f83ull,0xdce53c1df8560f83ull,0x7a36bf860ddf6443ull,0x53868b49306a5e85ull,0xd5561a2b36e2d18eull,0x197a383701c959a4ull,0xd903cf02d8d0c45bull,0x8e55c3bf745ae0a2ull}},
    {"F.run0",  {0x39bfe1b50698bdb8ull,0xdce53c1df8560f83ull,0xdce53c1df8560f83ull,0x3255b39eaee18098ull,0xdce53c1df8560f83ull,0xf8db33a90ec865d1ull,0x205bbd7ec35a0103ull,0x17f8e2a4422e714bull,0x9ca69726bbe3b7a6ull}},
    {"G.run0",  {0x55e69ffe9ef9c3b3ull,0xd4dbe31c5ee58a1cull,0xcf82d2b59fbbc4b3ull,0x218035740d3f1b83ull,0x218035740d3f1b83ull,0xbe6c73fcf083d8fdull,0x8fa9d63907c8925bull,0xc862069ce03ed049ull,0xdad395231c8fb5dfull}},
    {"H.N=1",   {0xa31e272015f12c43ull,0x315446a086a23133ull,0x721cdbe3331f499cull,0x315446a086a23133ull,0x315446a086a23133ull,0xf4703ac9db7b75c9ull,0x1011717e0e8b8a4dull,0x323894abcdaefe5bull,0x15b740719ee3a090ull}},
    {"H.N=3",   {0xea1846aa7b4c2441ull,0xebfe7bb0e6e47806ull,0xb206fa0ef70d4156ull,0xc327e8bf29be1593ull,0xc327e8bf29be1593ull,0x5a34c5901b014579ull,0x73779c596ab82cf2ull,0x0cadba69f35ae600ull,0x9c6413c94b0228b5ull}},
    {"H.N=31",  {0x410f219a0dee67a6ull,0x0ab99b15452b31f6ull,0x166484ce7bc7b871ull,0xc02c7995351e3ed3ull,0xc02c7995351e3ed3ull,0x6f6d1f491409dab4ull,0x68ea226e7bd6a15bull,0x8da1d89ed3628d01ull,0xe92866f14c9625f6ull}},
    {"H.N=255", {0x3dd790ce7f127151ull,0x6e35ec173805cd4cull,0x9ab2d19dbf59dc38ull,0xf0860c9f566e48d3ull,0xf0860c9f566e48d3ull,0x3759d77cfc147d64ull,0x68e42965164d2628ull,0x5fed36a55088c5aeull,0x648bea825e7690cbull}},
    {"H.N=257", {0x104a1bb72728db07ull,0xd8bcf186ebf9ee0dull,0xb59ff62e12949db2ull,0x5f9a43bf40416133ull,0x5f9a43bf40416133ull,0x2a6dbc348739b296ull,0x722d273e9c146d41ull,0xb64d7887a1f1d5d7ull,0x011f759b73bbe5e2ull}},
    {"H.N=511", {0xca2690532b3d61c0ull,0x21ee99b4d93908e9ull,0xbbcbb745e38fee10ull,0xdf1593f2780778d3ull,0xdf1593f2780778d3ull,0x08a97f88ee61343aull,0x915c749fcbcde576ull,0x806f0ed676925f8cull,0x30205e92cadc31aeull}},
    {"H.N=769", {0xec8dfd19638af0dfull,0x1faea27aaa1cca87ull,0x67bd2120d74891eaull,0x6bec87c4cc7fc133ull,0x6bec87c4cc7fc133ull,0x5eb0946144e799edull,0xfbe8cbfe49572cdfull,0x01632d0b569a068dull,0xee69a4d452dbfb23ull}},
    {"M.run0",  {0x6a2d784cb369963eull,0xe7b8fe50c6bb8cf2ull,0xe0adca725e9454f6ull,0xdce53c1df8560f83ull,0xdce53c1df8560f83ull,0x939a737482a5e5d9ull,0x94ee238f699e6e3cull,0x244d4c55568101d6ull,0x46a4c45c70873037ull}},
    {"M.run1",  {0x6a2d784cb369963eull,0xe7b8fe50c6bb8cf2ull,0xe0adca725e9454f6ull,0xdce53c1df8560f83ull,0xdce53c1df8560f83ull,0xce63ac54a261e8f6ull,0xbb7e3e23a6589519ull,0x8984fc84434f2409ull,0xa3dd4ed5b9d99b1full}},
    {"M.run2",  {0x6a2d784cb369963eull,0xe7b8fe50c6bb8cf2ull,0xe0adca725e9454f6ull,0xdce53c1df8560f83ull,0xdce53c1df8560f83ull,0x323b180a30a77979ull,0xedf4d05c68bb13faull,0xaa6bc745c38f2a57ull,0x13641e52bfa03c22ull}},
};
static const int g_golden_count = (int)(sizeof(g_golden) / sizeof(g_golden[0]));

static int g_cases_total  = 0;  // M: number of golden cases actually exercised
static int g_cases_passed = 0;  // N: cases whose full 9-hash set matched

// Compare one case's checksums against embedded golden; update counters.
static void check_case(const char* label, const RunChecksums& cs) {
    const uint64_t got[9] = {
        cs.H_dx, cs.H_flush_dgamma, cs.H_flush_dbeta,
        cs.H_accum_dgamma, cs.H_accum_dbeta,
        cs.H_cache, cs.H_timeline, cs.H_counters, cs.H_all};

    const GoldenCase* g = nullptr;
    for (int i = 0; i < g_golden_count; ++i) {
        if (strcmp(g_golden[i].label, label) == 0) { g = &g_golden[i]; break; }
    }
    g_cases_total += 1;
    if (!g) {
        printf("  [%s] NO GOLDEN ENTRY -> FAIL\n", label);
        return;
    }
    bool ok = true;
    for (int i = 0; i < 9; ++i) {
        if (got[i] != g->h[i]) { ok = false; break; }
    }
    if (ok) {
        g_cases_passed += 1;
    } else {
        printf("  [%s] MISMATCH vs golden -> FAIL\n", label);
    }
}

static void print_checksums(const char* label, const RunChecksums& cs) {
    check_case(label, cs);
    printf("  [%s] H_dx            = %016llx\n", label, (unsigned long long)cs.H_dx);
    printf("  [%s] H_flush_dgamma  = %016llx\n", label, (unsigned long long)cs.H_flush_dgamma);
    printf("  [%s] H_flush_dbeta   = %016llx\n", label, (unsigned long long)cs.H_flush_dbeta);
    printf("  [%s] H_accum_dgamma  = %016llx\n", label, (unsigned long long)cs.H_accum_dgamma);
    printf("  [%s] H_accum_dbeta   = %016llx\n", label, (unsigned long long)cs.H_accum_dbeta);
    printf("  [%s] H_cache         = %016llx\n", label, (unsigned long long)cs.H_cache);
    printf("  [%s] H_timeline      = %016llx\n", label, (unsigned long long)cs.H_timeline);
    printf("  [%s] H_counters      = %016llx\n", label, (unsigned long long)cs.H_counters);
    printf("  [%s] H_all           = %016llx\n", label, (unsigned long long)cs.H_all);
}

// ---------------------------------------------------------------------------
// Random float helpers for input generation (no reference math inside)
// ---------------------------------------------------------------------------
static uint64_t g_rng = 0x123456789abcdef0ull ^ 0x000000000008ce7bull;

static uint64_t rng_next() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

// float in [-1, 1]
static float rng_f32() {
    uint32_t u = uint32_t(rng_next() >> 32);
    // map to [0,1) then scale
    float f = float(u >> 8) / float(1u << 24);
    return f * 2.0f - 1.0f;
}

static uint16_t f32_to_bf16_rne_host(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    // check nan
    if (((u & 0x7f800000u) == 0x7f800000u) && (u & 0x007fffffu))
        return 0x7fc0u;
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t bias = 0x7fffu + lsb;
    return uint16_t((u + bias) >> 16);
}

static void fill_storage(std::vector<uint8_t>& buf, uint32_t rows, uint32_t stride,
                         uint32_t dtype, uint64_t seed_offset) {
    // seed specific to this buffer
    uint64_t saved = g_rng;
    g_rng ^= seed_offset;

    size_t elem = (dtype == LNR_DTYPE_BF16) ? 2 : 4;
    buf.resize(size_t(rows) * stride * elem);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < stride; ++c) {
            float fv = rng_f32();
            size_t idx = size_t(r) * stride + c;
            if (dtype == LNR_DTYPE_BF16) {
                uint16_t h = f32_to_bf16_rne_host(fv);
                memcpy(buf.data() + idx * 2, &h, 2);
            } else {
                memcpy(buf.data() + idx * 4, &fv, 4);
            }
        }
    }
    g_rng = saved ^ seed_offset;
}

// ---------------------------------------------------------------------------
// Run-level helpers
// ---------------------------------------------------------------------------
struct RunContext {
    const Spec* spec;
    void* state;
    void* workspace;
    size_t workspace_bytes;
    cudaStream_t stream;
};

// Execute one run and return checksums. Prints per-run detail.
static RunChecksums execute_run(
    const RunContext& ctx,
    uint64_t run_id,
    uint32_t input_rows,
    uint32_t dx_rows,
    uint32_t flush_records,
    const std::vector<uint8_t>& h_x_buf,
    const std::vector<uint8_t>& h_dy_buf,
    const std::vector<uint8_t>& h_gamma_buf,
    uint32_t x_stride,
    uint32_t dy_stride,
    uint32_t gamma_stride,
    const std::vector<OpDesc>& ops)
{
    const Spec& s = *ctx.spec;
    uint32_t op_count = uint32_t(ops.size());

    // Upload inputs
    size_t elem = (s.storage_dtype == LNR_DTYPE_BF16) ? 2 : 4;
    void *d_x, *d_dy, *d_gamma;
    CUDA_CHECK(cudaMalloc(&d_x,     h_x_buf.size()));
    CUDA_CHECK(cudaMalloc(&d_dy,    h_dy_buf.size()));
    CUDA_CHECK(cudaMalloc(&d_gamma, h_gamma_buf.size()));
    CUDA_CHECK(cudaMemcpy(d_x,     h_x_buf.data(),     h_x_buf.size(),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dy,    h_dy_buf.data(),    h_dy_buf.size(),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma_buf.data(), h_gamma_buf.size(), cudaMemcpyHostToDevice));

    // Alloc outputs
    DevOutputs dout;
    alloc_dev_outputs(dout, s, dx_rows, flush_records, op_count);

    // Build RunSpec + InputPtrs + OutputPtrs
    RunSpec run{};
    run.run_id          = run_id;
    run.op_count        = op_count;
    run.input_rows      = input_rows;
    run.dx_rows         = dx_rows;
    run.flush_records   = flush_records;
    run.x_stride_elems  = x_stride;
    run.dy_stride_elems = dy_stride;
    run.gamma_stride_elems = gamma_stride;
    run.reserved0       = 0;
    run.ops             = ops.empty() ? nullptr : ops.data();

    InputPtrs inp{};
    inp.x     = d_x;
    inp.dy    = d_dy;
    inp.gamma = d_gamma;

    OutputPtrs outp{};
    outp.dx                    = dout.dx;
    outp.flush_dgamma          = dout.flush_dgamma;
    outp.flush_dbeta           = dout.flush_dbeta;
    outp.accum_dgamma_snapshot = dout.accum_dgamma;
    outp.accum_dbeta_snapshot  = dout.accum_dbeta;
    outp.cache_mean_snapshot   = dout.cache_mean;
    outp.cache_rstd_snapshot   = dout.cache_rstd;
    outp.cache_kind_snapshot   = dout.cache_kind;
    outp.cache_gen_snapshot    = dout.cache_gen;
    outp.timeline              = dout.timeline;
    outp.counters              = dout.counters;

    cudaError_t e = solution_run(
        ctx.state, &run, &inp, &outp,
        ctx.workspace, ctx.workspace_bytes, ctx.stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "solution_run returned error: %s\n", cudaGetErrorString(e));
        exit(1);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    HostOutputs hout;
    copy_dev_to_host(hout, dout, s, dx_rows, flush_records, op_count);

    RunChecksums cs = compute_checksums(hout, s, dx_rows, flush_records, op_count);

    cudaFree(d_x); cudaFree(d_dy); cudaFree(d_gamma);
    free_dev_outputs(dout);

    return cs;
}

// ---------------------------------------------------------------------------
// Scenario A: Minimal sanity — N=64, bf16, 1 param
// SAVE_LN(rows=4), BWD_LN(rows=4), FLUSH
// ---------------------------------------------------------------------------
static void scenario_A(uint64_t& run_id_ctr) {
    printf("\n=== Scenario A: Minimal sanity (N=64, bf16, 1 param) ===\n");

    Spec s{};
    s.magic   = LNRBWD_MAGIC;
    s.version = LNRBWD_VERSION;
    s.hidden_size              = 64;
    s.param_count              = 1;
    s.max_cache_rows           = 64;
    s.max_input_rows_per_run   = 16;
    s.max_dx_rows_per_run      = 16;
    s.max_backward_rows_per_run= 16;
    s.max_ops_per_run          = 12;
    s.max_flush_records_per_run= 4;
    s.storage_dtype            = LNR_DTYPE_BF16;
    s.eps_ln                   = 1e-5f;
    s.eps_rms                  = 1e-5f;
    s.counter_seed             = 0;

    void* state = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(solution_init(&s, &state, stream));

    size_t wbytes = solution_workspace_bytes(&s);
    void* workspace = nullptr;
    if (wbytes) CUDA_CHECK(cudaMalloc(&workspace, wbytes));

    RunContext ctx{&s, state, workspace, wbytes, stream};

    uint32_t N = 64, rows = 4, stride = 64;

    std::vector<uint8_t> x_buf, dy_buf, gamma_buf;
    fill_storage(x_buf,     rows,  stride, LNR_DTYPE_BF16, 0xA001);
    fill_storage(dy_buf,    rows,  stride, LNR_DTYPE_BF16, 0xA002);
    fill_storage(gamma_buf, 1,     stride, LNR_DTYPE_BF16, 0xA003);

    // Run 0: SAVE_LN + BWD_LN + FLUSH
    std::vector<OpDesc> ops(3);
    // SAVE_LN
    ops[0].kind = LNR_OP_SAVE_LN; ops[0].x_row_base = 0; ops[0].cache_base = 0; ops[0].rows = rows;
    // BWD_LN
    ops[1].kind = LNR_OP_BWD_LN;  ops[1].param_id = 0;
    ops[1].x_row_base = 0; ops[1].dy_row_base = 0; ops[1].cache_base = 0;
    ops[1].rows = rows; ops[1].dx_out_base = 0;
    // FLUSH
    ops[2].kind = LNR_OP_FLUSH; ops[2].param_id = 0; ops[2].flush_out_base = 0;

    RunChecksums cs0 = execute_run(ctx, ++run_id_ctr, rows, rows, 1, x_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
    print_checksums("A.run0", cs0);

    if (workspace) cudaFree(workspace);
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// Scenario B: RMS branch — N=128, bf16
// ---------------------------------------------------------------------------
static void scenario_B(uint64_t& run_id_ctr) {
    printf("\n=== Scenario B: RMS branch (N=128, bf16) ===\n");

    Spec s{};
    s.magic   = LNRBWD_MAGIC;
    s.version = LNRBWD_VERSION;
    s.hidden_size              = 128;
    s.param_count              = 1;
    s.max_cache_rows           = 64;
    s.max_input_rows_per_run   = 16;
    s.max_dx_rows_per_run      = 16;
    s.max_backward_rows_per_run= 16;
    s.max_ops_per_run          = 12;
    s.max_flush_records_per_run= 4;
    s.storage_dtype            = LNR_DTYPE_BF16;
    s.eps_ln                   = 1e-5f;
    s.eps_rms                  = 1e-6f;
    s.counter_seed             = 0;

    void* state = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(solution_init(&s, &state, stream));

    size_t wbytes = solution_workspace_bytes(&s);
    void* workspace = nullptr;
    if (wbytes) CUDA_CHECK(cudaMalloc(&workspace, wbytes));

    RunContext ctx{&s, state, workspace, wbytes, stream};

    uint32_t rows = 8, stride = 128;

    std::vector<uint8_t> x_buf, dy_buf, gamma_buf;
    fill_storage(x_buf,     rows, stride, LNR_DTYPE_BF16, 0xB001);
    fill_storage(dy_buf,    rows, stride, LNR_DTYPE_BF16, 0xB002);
    fill_storage(gamma_buf, 1,    stride, LNR_DTYPE_BF16, 0xB003);

    std::vector<OpDesc> ops(3);
    ops[0].kind = LNR_OP_SAVE_RMS; ops[0].x_row_base = 0; ops[0].cache_base = 0; ops[0].rows = rows;
    ops[1].kind = LNR_OP_BWD_RMS;  ops[1].param_id = 0;
    ops[1].x_row_base = 0; ops[1].dy_row_base = 0; ops[1].cache_base = 0;
    ops[1].rows = rows; ops[1].dx_out_base = 0;
    ops[2].kind = LNR_OP_FLUSH; ops[2].param_id = 0; ops[2].flush_out_base = 0;

    RunChecksums cs = execute_run(ctx, ++run_id_ctr, rows, rows, 1, x_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
    print_checksums("B.run0", cs);

    if (workspace) cudaFree(workspace);
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// Scenario C: Mixed LN/RMS in one run — N=257, bf16, 2 params
// ---------------------------------------------------------------------------
static void scenario_C(uint64_t& run_id_ctr) {
    printf("\n=== Scenario C: Mixed LN+RMS, 2 params (N=257, bf16) ===\n");

    Spec s{};
    s.magic   = LNRBWD_MAGIC;
    s.version = LNRBWD_VERSION;
    s.hidden_size              = 257;
    s.param_count              = 2;
    s.max_cache_rows           = 512;
    s.max_input_rows_per_run   = 96;
    s.max_dx_rows_per_run      = 96;
    s.max_backward_rows_per_run= 96;
    s.max_ops_per_run          = 32;
    s.max_flush_records_per_run= 8;
    s.storage_dtype            = LNR_DTYPE_BF16;
    s.eps_ln                   = 1e-5f;
    s.eps_rms                  = 1e-5f;
    s.counter_seed             = 0;

    void* state = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(solution_init(&s, &state, stream));

    size_t wbytes = solution_workspace_bytes(&s);
    void* workspace = nullptr;
    if (wbytes) CUDA_CHECK(cudaMalloc(&workspace, wbytes));

    RunContext ctx{&s, state, workspace, wbytes, stream};

    // Note: stride=257 (no padding) for this scenario
    uint32_t stride = 257, rows = 32;

    std::vector<uint8_t> x_buf, dy_buf, gamma_buf;
    fill_storage(x_buf,     rows, stride, LNR_DTYPE_BF16, 0xC001);
    fill_storage(dy_buf,    rows, stride, LNR_DTYPE_BF16, 0xC002);
    fill_storage(gamma_buf, 2,    stride, LNR_DTYPE_BF16, 0xC003);

    // SAVE_LN slots 0..15, SAVE_RMS slots 16..31,
    // BWD_LN param0 over rows 0..15, BWD_RMS param1 over rows 16..31,
    // FLUSH param0, FLUSH param0 again (verify zero-after-flush)
    std::vector<OpDesc> ops;
    {
        OpDesc op{};
        op.kind = LNR_OP_SAVE_LN; op.x_row_base = 0; op.cache_base = 0; op.rows = 16;
        ops.push_back(op);
    }
    {
        OpDesc op{};
        op.kind = LNR_OP_SAVE_RMS; op.x_row_base = 16; op.cache_base = 16; op.rows = 16;
        ops.push_back(op);
    }
    {
        OpDesc op{};
        op.kind = LNR_OP_BWD_LN; op.param_id = 0;
        op.x_row_base = 0; op.dy_row_base = 0; op.cache_base = 0;
        op.rows = 16; op.dx_out_base = 0;
        ops.push_back(op);
    }
    {
        OpDesc op{};
        op.kind = LNR_OP_BWD_RMS; op.param_id = 1;
        op.x_row_base = 16; op.dy_row_base = 16; op.cache_base = 16;
        op.rows = 16; op.dx_out_base = 16;
        ops.push_back(op);
    }
    {
        OpDesc op{};
        op.kind = LNR_OP_FLUSH; op.param_id = 0; op.flush_out_base = 0;
        ops.push_back(op);
    }
    {
        OpDesc op{};
        op.kind = LNR_OP_FLUSH; op.param_id = 0; op.flush_out_base = 1;
        ops.push_back(op);
    }

    RunChecksums cs = execute_run(ctx, ++run_id_ctr, rows, rows, 4, x_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
    print_checksums("C.run0", cs);

    if (workspace) cudaFree(workspace);
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// Scenario D: Multi-run persistent accumulation — N=1024, f32
// Run0: saves+bwd (no flush); Run1: more bwd same param; Run2: flush
// ---------------------------------------------------------------------------
static void scenario_D(uint64_t& run_id_ctr) {
    printf("\n=== Scenario D: Multi-run accumulation (N=1024, f32) ===\n");

    Spec s{};
    s.magic   = LNRBWD_MAGIC;
    s.version = LNRBWD_VERSION;
    s.hidden_size              = 1024;
    s.param_count              = 3;
    s.max_cache_rows           = 2048;
    s.max_input_rows_per_run   = 256;
    s.max_dx_rows_per_run      = 256;
    s.max_backward_rows_per_run= 256;
    s.max_ops_per_run          = 64;
    s.max_flush_records_per_run= 8;
    s.storage_dtype            = LNR_DTYPE_F32;
    s.eps_ln                   = 1e-5f;
    s.eps_rms                  = 1e-5f;
    s.counter_seed             = 0;

    void* state = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(solution_init(&s, &state, stream));

    size_t wbytes = solution_workspace_bytes(&s);
    void* workspace = nullptr;
    if (wbytes) CUDA_CHECK(cudaMalloc(&workspace, wbytes));

    RunContext ctx{&s, state, workspace, wbytes, stream};

    uint32_t stride = 1024;

    // Inputs for all runs (use same buffers for simplicity; cache reuse is the test)
    std::vector<uint8_t> x_buf, dy_buf, gamma_buf;
    fill_storage(x_buf,     32, stride, LNR_DTYPE_F32, 0xD001);
    fill_storage(dy_buf,    32, stride, LNR_DTYPE_F32, 0xD002);
    fill_storage(gamma_buf, 3,  stride, LNR_DTYPE_F32, 0xD003);

    // Run 0: SAVE_LN 16 rows into cache[0..15], BWD_LN 16 rows no flush
    {
        std::vector<OpDesc> ops;
        {
            OpDesc op{}; op.kind = LNR_OP_SAVE_LN; op.x_row_base = 0; op.cache_base = 0; op.rows = 16;
            ops.push_back(op);
        }
        {
            OpDesc op{}; op.kind = LNR_OP_BWD_LN; op.param_id = 0;
            op.x_row_base = 0; op.dy_row_base = 0; op.cache_base = 0;
            op.rows = 16; op.dx_out_base = 0;
            ops.push_back(op);
        }
        RunChecksums cs = execute_run(ctx, ++run_id_ctr, 32, 32, 1, x_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
        print_checksums("D.run0", cs);
    }

    // Run 1: SAVE_LN 8 more rows into cache[16..23], BWD_LN 8 rows same param 0
    {
        std::vector<OpDesc> ops;
        {
            OpDesc op{}; op.kind = LNR_OP_SAVE_LN; op.x_row_base = 0; op.cache_base = 16; op.rows = 8;
            ops.push_back(op);
        }
        {
            OpDesc op{}; op.kind = LNR_OP_BWD_LN; op.param_id = 0;
            op.x_row_base = 0; op.dy_row_base = 0; op.cache_base = 16;
            op.rows = 8; op.dx_out_base = 0;
            ops.push_back(op);
        }
        RunChecksums cs = execute_run(ctx, ++run_id_ctr, 32, 32, 1, x_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
        print_checksums("D.run1", cs);
    }

    // Run 2: FLUSH param 0 — accumulated across both prior runs
    {
        std::vector<OpDesc> ops;
        {
            OpDesc op{}; op.kind = LNR_OP_FLUSH; op.param_id = 0; op.flush_out_base = 0;
            ops.push_back(op);
        }
        RunChecksums cs = execute_run(ctx, ++run_id_ctr, 0, 0, 1, x_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
        print_checksums("D.run2", cs);
    }

    if (workspace) cudaFree(workspace);
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// Scenario E: Cache reuse vs recompute trap
// Run0: SAVE_LN from x_A; Run1: BWD_LN with cache still from x_A but new x_B in buffer
// ---------------------------------------------------------------------------
static void scenario_E(uint64_t& run_id_ctr) {
    printf("\n=== Scenario E: Cache reuse trap (N=64, f32) ===\n");

    Spec s{};
    s.magic   = LNRBWD_MAGIC;
    s.version = LNRBWD_VERSION;
    s.hidden_size              = 64;
    s.param_count              = 1;
    s.max_cache_rows           = 64;
    s.max_input_rows_per_run   = 16;
    s.max_dx_rows_per_run      = 16;
    s.max_backward_rows_per_run= 16;
    s.max_ops_per_run          = 12;
    s.max_flush_records_per_run= 4;
    s.storage_dtype            = LNR_DTYPE_F32;
    s.eps_ln                   = 1e-5f;
    s.eps_rms                  = 1e-5f;
    s.counter_seed             = 0;

    void* state = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(solution_init(&s, &state, stream));

    size_t wbytes = solution_workspace_bytes(&s);
    void* workspace = nullptr;
    if (wbytes) CUDA_CHECK(cudaMalloc(&workspace, wbytes));

    RunContext ctx{&s, state, workspace, wbytes, stream};

    uint32_t stride = 64, rows = 4;

    // x_A for SAVE
    std::vector<uint8_t> xA_buf, dy_buf, gamma_buf, xB_buf;
    fill_storage(xA_buf,    rows, stride, LNR_DTYPE_F32, 0xE001);
    fill_storage(dy_buf,    rows, stride, LNR_DTYPE_F32, 0xE002);
    fill_storage(gamma_buf, 1,    stride, LNR_DTYPE_F32, 0xE003);
    fill_storage(xB_buf,    rows, stride, LNR_DTYPE_F32, 0xE004); // different x

    // Run 0: SAVE_LN from x_A
    {
        std::vector<OpDesc> ops;
        OpDesc op{}; op.kind = LNR_OP_SAVE_LN; op.x_row_base = 0; op.cache_base = 0; op.rows = rows;
        ops.push_back(op);
        RunChecksums cs = execute_run(ctx, ++run_id_ctr, rows, 0, 0, xA_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
        print_checksums("E.run0", cs);
    }

    // Run 1: BWD_LN — x buffer is xB but cache holds xA's stats
    {
        std::vector<OpDesc> ops;
        OpDesc op{}; op.kind = LNR_OP_BWD_LN; op.param_id = 0;
        op.x_row_base = 0; op.dy_row_base = 0; op.cache_base = 0;
        op.rows = rows; op.dx_out_base = 0;
        ops.push_back(op);
        RunChecksums cs = execute_run(ctx, ++run_id_ctr, rows, rows, 1, xB_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
        print_checksums("E.run1", cs);
    }

    if (workspace) cudaFree(workspace);
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// Scenario F: Cache overwrite + kind mismatch
// ---------------------------------------------------------------------------
static void scenario_F(uint64_t& run_id_ctr) {
    printf("\n=== Scenario F: Cache overwrite + kind mismatch (N=64, bf16) ===\n");

    Spec s{};
    s.magic   = LNRBWD_MAGIC;
    s.version = LNRBWD_VERSION;
    s.hidden_size              = 64;
    s.param_count              = 1;
    s.max_cache_rows           = 64;
    s.max_input_rows_per_run   = 32;
    s.max_dx_rows_per_run      = 32;
    s.max_backward_rows_per_run= 32;
    s.max_ops_per_run          = 16;
    s.max_flush_records_per_run= 4;
    s.storage_dtype            = LNR_DTYPE_BF16;
    s.eps_ln                   = 1e-5f;
    s.eps_rms                  = 1e-5f;
    s.counter_seed             = 0;

    void* state = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(solution_init(&s, &state, stream));

    size_t wbytes = solution_workspace_bytes(&s);
    void* workspace = nullptr;
    if (wbytes) CUDA_CHECK(cudaMalloc(&workspace, wbytes));

    RunContext ctx{&s, state, workspace, wbytes, stream};

    uint32_t stride = 64, rows = 26;

    std::vector<uint8_t> x_buf, dy_buf, gamma_buf;
    fill_storage(x_buf,     rows, stride, LNR_DTYPE_BF16, 0xF001);
    fill_storage(dy_buf,    rows, stride, LNR_DTYPE_BF16, 0xF002);
    fill_storage(gamma_buf, 1,    stride, LNR_DTYPE_BF16, 0xF003);

    // SAVE_LN slots 0..20 (21 rows)
    // SAVE_RMS slots 15..25 (11 rows) — overwrites 15..20 with RMS
    // BWD_LN over slots 0..20 — must cache-miss on 15..20 (RMS)
    // BWD_RMS over slots 15..25 — must hit (all RMS)
    std::vector<OpDesc> ops;
    { OpDesc op{}; op.kind=LNR_OP_SAVE_LN;  op.x_row_base=0;  op.cache_base=0;  op.rows=21; ops.push_back(op); }
    { OpDesc op{}; op.kind=LNR_OP_SAVE_RMS; op.x_row_base=0;  op.cache_base=15; op.rows=11; ops.push_back(op); }
    // BWD_LN over 0..20 — cache miss (slots 15..20 are RMS)
    { OpDesc op{}; op.kind=LNR_OP_BWD_LN; op.param_id=0;
      op.x_row_base=0; op.dy_row_base=0; op.cache_base=0; op.rows=21; op.dx_out_base=0;
      ops.push_back(op); }
    // BWD_RMS over 15..25 — all RMS, should succeed
    { OpDesc op{}; op.kind=LNR_OP_BWD_RMS; op.param_id=0;
      op.x_row_base=15; op.dy_row_base=15; op.cache_base=15; op.rows=11; op.dx_out_base=0;
      ops.push_back(op); }

    RunChecksums cs = execute_run(ctx, ++run_id_ctr, rows, rows, 1, x_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
    print_checksums("F.run0", cs);

    if (workspace) cudaFree(workspace);
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// Scenario G: Invalid ops — bad param_id, cache miss, output range overflow
// ---------------------------------------------------------------------------
static void scenario_G(uint64_t& run_id_ctr) {
    printf("\n=== Scenario G: Invalid ops (N=64, bf16) ===\n");

    Spec s{};
    s.magic   = LNRBWD_MAGIC;
    s.version = LNRBWD_VERSION;
    s.hidden_size              = 64;
    s.param_count              = 2;
    s.max_cache_rows           = 32;
    s.max_input_rows_per_run   = 16;
    s.max_dx_rows_per_run      = 8;   // intentionally smaller
    s.max_backward_rows_per_run= 16;
    s.max_ops_per_run          = 16;
    s.max_flush_records_per_run= 4;
    s.storage_dtype            = LNR_DTYPE_BF16;
    s.eps_ln                   = 1e-5f;
    s.eps_rms                  = 1e-5f;
    s.counter_seed             = 0;

    void* state = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(solution_init(&s, &state, stream));

    size_t wbytes = solution_workspace_bytes(&s);
    void* workspace = nullptr;
    if (wbytes) CUDA_CHECK(cudaMalloc(&workspace, wbytes));

    RunContext ctx{&s, state, workspace, wbytes, stream};

    uint32_t stride = 64;

    std::vector<uint8_t> x_buf, dy_buf, gamma_buf;
    fill_storage(x_buf,     16, stride, LNR_DTYPE_BF16, 0x6001);
    fill_storage(dy_buf,    16, stride, LNR_DTYPE_BF16, 0x6002);
    fill_storage(gamma_buf, 2,  stride, LNR_DTYPE_BF16, 0x6003);

    std::vector<OpDesc> ops;
    // SAVE_LN 4 rows into slots 0..3
    { OpDesc op{}; op.kind=LNR_OP_SAVE_LN; op.x_row_base=0; op.cache_base=0; op.rows=4; ops.push_back(op); }
    // BWD_LN with invalid param_id=99
    { OpDesc op{}; op.kind=LNR_OP_BWD_LN; op.param_id=99;
      op.x_row_base=0; op.dy_row_base=0; op.cache_base=0; op.rows=4; op.dx_out_base=0;
      ops.push_back(op); }
    // BWD_LN cache miss (slots 4..7 not saved)
    { OpDesc op{}; op.kind=LNR_OP_BWD_LN; op.param_id=0;
      op.x_row_base=0; op.dy_row_base=0; op.cache_base=4; op.rows=4; op.dx_out_base=0;
      ops.push_back(op); }
    // BWD_LN dx_out_base overflow (dx_rows=8, requesting base=6 rows=4 -> 10 > 8)
    { OpDesc op{}; op.kind=LNR_OP_BWD_LN; op.param_id=0;
      op.x_row_base=0; op.dy_row_base=0; op.cache_base=0; op.rows=4; op.dx_out_base=6;
      ops.push_back(op); }
    // FLUSH with invalid flush_out_base >= flush_records
    { OpDesc op{}; op.kind=LNR_OP_FLUSH; op.param_id=0; op.flush_out_base=10; ops.push_back(op); }
    // Unsupported op kind
    { OpDesc op{}; op.kind=99; ops.push_back(op); }
    // Valid BWD_LN and FLUSH to cap
    { OpDesc op{}; op.kind=LNR_OP_BWD_LN; op.param_id=0;
      op.x_row_base=0; op.dy_row_base=0; op.cache_base=0; op.rows=4; op.dx_out_base=0;
      ops.push_back(op); }
    { OpDesc op{}; op.kind=LNR_OP_FLUSH; op.param_id=0; op.flush_out_base=0; ops.push_back(op); }

    RunChecksums cs = execute_run(ctx, ++run_id_ctr, 16, 8, 4, x_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
    print_checksums("G.run0", cs);

    if (workspace) cudaFree(workspace);
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// Scenario H: Ragged columns — test N=257, 511, 769 with f32
// Fixed: just one run each with SAVE_LN + BWD_LN + FLUSH
// ---------------------------------------------------------------------------
static void scenario_H_one(uint64_t& run_id_ctr, uint32_t N) {
    Spec s{};
    s.magic   = LNRBWD_MAGIC;
    s.version = LNRBWD_VERSION;
    s.hidden_size              = N;
    s.param_count              = 1;
    s.max_cache_rows           = 16;
    s.max_input_rows_per_run   = 8;
    s.max_dx_rows_per_run      = 8;
    s.max_backward_rows_per_run= 8;
    s.max_ops_per_run          = 8;
    s.max_flush_records_per_run= 2;
    s.storage_dtype            = LNR_DTYPE_F32;
    s.eps_ln                   = 1e-5f;
    s.eps_rms                  = 1e-5f;
    s.counter_seed             = 0;

    void* state = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(solution_init(&s, &state, stream));

    size_t wbytes = solution_workspace_bytes(&s);
    void* workspace = nullptr;
    if (wbytes) CUDA_CHECK(cudaMalloc(&workspace, wbytes));

    RunContext ctx{&s, state, workspace, wbytes, stream};

    uint32_t rows = 4, stride = N;

    std::vector<uint8_t> x_buf, dy_buf, gamma_buf;
    fill_storage(x_buf,     rows, stride, LNR_DTYPE_F32, 0x8000ull ^ uint64_t(N));
    fill_storage(dy_buf,    rows, stride, LNR_DTYPE_F32, 0x8100ull ^ uint64_t(N));
    fill_storage(gamma_buf, 1,    stride, LNR_DTYPE_F32, 0x8200ull ^ uint64_t(N));

    std::vector<OpDesc> ops;
    { OpDesc op{}; op.kind=LNR_OP_SAVE_LN; op.x_row_base=0; op.cache_base=0; op.rows=rows; ops.push_back(op); }
    { OpDesc op{}; op.kind=LNR_OP_BWD_LN;  op.param_id=0;
      op.x_row_base=0; op.dy_row_base=0; op.cache_base=0; op.rows=rows; op.dx_out_base=0;
      ops.push_back(op); }
    { OpDesc op{}; op.kind=LNR_OP_FLUSH; op.param_id=0; op.flush_out_base=0; ops.push_back(op); }

    char lbl[64]; snprintf(lbl, sizeof(lbl), "H.N=%u", N);
    RunChecksums cs = execute_run(ctx, ++run_id_ctr, rows, rows, 1, x_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
    print_checksums(lbl, cs);

    if (workspace) cudaFree(workspace);
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

static void scenario_H(uint64_t& run_id_ctr) {
    printf("\n=== Scenario H: Ragged columns ===\n");
    uint32_t Ns[] = {1, 3, 31, 255, 257, 511, 769};
    for (uint32_t N : Ns) scenario_H_one(run_id_ctr, N);
}

// ---------------------------------------------------------------------------
// Scenario M: Counter wrap — counter_seed = UINT64_MAX - 5
// ---------------------------------------------------------------------------
static void scenario_M(uint64_t& run_id_ctr) {
    printf("\n=== Scenario M: Counter wrap (N=64, bf16) ===\n");

    Spec s{};
    s.magic   = LNRBWD_MAGIC;
    s.version = LNRBWD_VERSION;
    s.hidden_size              = 64;
    s.param_count              = 1;
    s.max_cache_rows           = 16;
    s.max_input_rows_per_run   = 8;
    s.max_dx_rows_per_run      = 8;
    s.max_backward_rows_per_run= 8;
    s.max_ops_per_run          = 12;
    s.max_flush_records_per_run= 4;
    s.storage_dtype            = LNR_DTYPE_BF16;
    s.eps_ln                   = 1e-5f;
    s.eps_rms                  = 1e-5f;
    s.counter_seed             = UINT64_MAX - 5ull;

    void* state = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(solution_init(&s, &state, stream));

    size_t wbytes = solution_workspace_bytes(&s);
    void* workspace = nullptr;
    if (wbytes) CUDA_CHECK(cudaMalloc(&workspace, wbytes));

    RunContext ctx{&s, state, workspace, wbytes, stream};

    uint32_t rows = 4, stride = 64;

    std::vector<uint8_t> x_buf, dy_buf, gamma_buf;
    fill_storage(x_buf,     rows, stride, LNR_DTYPE_BF16, 0xAA01ull);
    fill_storage(dy_buf,    rows, stride, LNR_DTYPE_BF16, 0xAA02ull);
    fill_storage(gamma_buf, 1,    stride, LNR_DTYPE_BF16, 0xAA03ull);

    // Multiple runs to force wrap in several counters
    for (int r = 0; r < 3; ++r) {
        std::vector<OpDesc> ops;
        { OpDesc op{}; op.kind=LNR_OP_SAVE_LN; op.x_row_base=0; op.cache_base=0; op.rows=rows; ops.push_back(op); }
        { OpDesc op{}; op.kind=LNR_OP_BWD_LN;  op.param_id=0;
          op.x_row_base=0; op.dy_row_base=0; op.cache_base=0; op.rows=rows; op.dx_out_base=0;
          ops.push_back(op); }
        { OpDesc op{}; op.kind=LNR_OP_FLUSH; op.param_id=0; op.flush_out_base=0; ops.push_back(op); }

        char lbl[32]; snprintf(lbl, sizeof(lbl), "M.run%d", r);
        RunChecksums cs = execute_run(ctx, ++run_id_ctr, rows, rows, 2, x_buf, dy_buf, gamma_buf, stride, stride, stride, ops);
        print_checksums(lbl, cs);
    }

    if (workspace) cudaFree(workspace);
    solution_destroy(state);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    printf("=== bwd_norm harness ===\n");
    printf("FNV basis: 0x%016llx\n", (unsigned long long)LNR_FNV_BASIS);
    printf("FNV prime: 0x%016llx\n", (unsigned long long)LNR_FNV_PRIME);

    uint64_t run_id_ctr = 1000;

    scenario_A(run_id_ctr);
    scenario_B(run_id_ctr);
    scenario_C(run_id_ctr);
    scenario_D(run_id_ctr);
    scenario_E(run_id_ctr);
    scenario_F(run_id_ctr);
    scenario_G(run_id_ctr);
    scenario_H(run_id_ctr);
    scenario_M(run_id_ctr);

    printf("\n=== ALL SCENARIOS COMPLETE ===\n");
    printf("passed %d / %d\n", g_cases_passed, g_cases_total);
    return 0;
}

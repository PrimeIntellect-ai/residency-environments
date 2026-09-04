// file: muon_distopt_common.h
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#define MD_VERSION 2u
#define MD_MAGIC 0x4D554F4E44535432ull  // "MUONDST2"

#define MD_MAX_PARAMS 64u
#define MD_MAX_OPS_PER_RUN 8192u
#define MD_MAX_RANKS 8u

#define MD_DTYPE_BF16 1u

#define MD_PATH_MUON 0u
#define MD_PATH_ADAMW 1u

#define MD_PARAM_FORCE_ADAMW 0x1u
#define MD_PARAM_FORCE_MUON  0x2u

#define MD_STATUS_OK 0u
#define MD_STATUS_BAD_SPEC 1u
#define MD_STATUS_OOM 2u
#define MD_STATUS_BAD_RUN 3u

// FNV constants. This project intentionally does NOT use canonical FNV offset basis.
#define MD_FNV_BASIS 1469598103934665603ull  // hex 0x14650FB0739D0383
#define MD_FNV_PRIME 1099511628211ull

// Float32 bit constants used by the normative algorithm.
#define MD_F32_MUON_A_BITS       0x405C72B0u  // 3.444499969482422f
#define MD_F32_MUON_B_BITS       0xC098CCCDu  // -4.775000095367432f
#define MD_F32_MUON_C_BITS       0x40020419u  // 2.0315001010894775f
#define MD_F32_NS_EPS_BITS       0x33D6BF95u  // 1.0000000116860974e-7f

#define MD_F32_DEFAULT_MOM_BITS  0x3F733333u  // 0.949999988079071f
#define MD_F32_ADAM_B1_BITS      0x3F666666u  // 0.8999999761581421f
#define MD_F32_ADAM_B2_BITS      0x3F7FBE77u  // 0.9990000128746033f
#define MD_F32_ADAM_EPS_BITS     0x322BCC77u  // 9.99999993922529e-9f

struct ParamDesc {
    uint32_t rows;          // >= 1
    uint32_t cols;          // >= 1; numel = rows * cols
    uint32_t storage_dtype; // currently must be MD_DTYPE_BF16
    uint32_t flags;         // FORCE_ADAMW/FORCE_MUON; see path rules below

    // Float32 hyperparameters stored by raw IEEE-754 bits, little-endian.
    // If any field is 0, the default below is used.
    uint32_t lr_bits;             // default 0x3CA3D70A = 0.019999999552965164f
    uint32_t weight_decay_bits;   // default 0x3C23D70A = 0.009999999776482582f
    uint32_t muon_momentum_bits;  // default MD_F32_DEFAULT_MOM_BITS

    uint32_t adam_lr_bits;        // default 0x3A83126F = 0.0010000000474974513f
    uint32_t adam_beta1_bits;     // default MD_F32_ADAM_B1_BITS
    uint32_t adam_beta2_bits;     // default MD_F32_ADAM_B2_BITS
    uint32_t adam_eps_bits;       // default MD_F32_ADAM_EPS_BITS

    uint64_t init_seed;           // deterministic BF16 weight initialization seed
};

struct Spec {
    uint64_t magic;       // must be MD_MAGIC
    uint32_t version;     // must be MD_VERSION
    uint32_t num_params;  // 1..MD_MAX_PARAMS
    uint32_t num_ranks;   // 1..MD_MAX_RANKS
    uint32_t max_ops_per_run;

    uint64_t global_seed;
    uint64_t reserved0;
    uint64_t reserved1;

    ParamDesc params[MD_MAX_PARAMS];
};

struct GradOp {
    uint32_t param_id;       // target parameter
    uint32_t src_rank;       // data-parallel source rank contributing this gradient
    uint32_t dst_rank;       // owner rank for reduce-scatter shard
    uint32_t owner_offset;   // element offset within dst_rank's owner shard
    uint32_t elem_count;     // number of contiguous BF16 grad elements
    uint32_t grad_offset;    // offset into trailing BF16 grad_values[]
    uint32_t flags;          // currently ignored; must be folded into op event
    uint32_t reserved;       // ignored; must be zero in generated tests
};

struct RunSpec {
    uint64_t run_tag;          // arbitrary harness tag, included in timeline
    uint32_t op_count;         // number of GradOp records in inputs
    uint32_t grad_value_count; // number of trailing BF16 values after GradOp array
    uint32_t flags;            // currently 0
    uint32_t reserved;
};

// Device inputs layout:
//   GradOp ops[run->op_count]
//   uint16_t grad_values[run->grad_value_count]
//
// Device outputs layout:
//   RunOutputHeader header
//   uint16_t weights_bf16[total_numel], param-id ascending, row-major within param.
struct RunOutputHeader {
    uint64_t magic;              // MD_MAGIC
    uint32_t version;            // MD_VERSION
    uint32_t status;             // MD_STATUS_*

    uint64_t attempted_runs;     // cumulative, includes OOM/bad-run attempts
    uint64_t successful_runs;    // cumulative successful solution_run calls
    uint64_t reset_count;

    uint64_t run_tag;
    uint64_t weight_hash;
    uint64_t state_hash;
    uint64_t timeline_hash;
    uint64_t counter_hash;
    uint64_t final_hash;

    uint64_t total_valid_ops;
    uint64_t total_invalid_ops;
    uint64_t total_duplicate_ops;
    uint64_t total_nan_inputs;
    uint64_t total_inf_inputs;
    uint64_t total_muon_param_updates;
    uint64_t total_adam_param_updates;

    uint32_t last_valid_ops;
    uint32_t last_invalid_ops;
    uint32_t last_duplicate_ops;
    uint32_t last_nan_inputs;
    uint32_t last_inf_inputs;
    uint32_t last_muon_param_updates;
    uint32_t last_adam_param_updates;
    uint32_t reserved;
};

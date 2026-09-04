"""Render fresh correctness inputs after the agent's submission is frozen."""

import re
import secrets

from pmpp_hard.errors import PoolIntegrityError

POLICY = "independent-correctness-inputs-v1"

# Each entry binds the transformation to the validated release grader layout.
CUDA_PARAMETERS = {
    "v2-pl-01-route_compact_reduce": 1,
    "v2-pl-02-moe_dispatch_combine": 1,
    "v2-pl-03-onesweep_partition_reduce": 1,
    "v2-pl-04-paged_kv_decode": 2,
    "v2-pl-06-spec_decode_verify_rollback": 1,
    "v2-pl-08-group_limited_topk": 1,
    "v2-pl-12-segmented_sort_topm": 1,
    "v2-pl-16-multi_checkpoint_tree_rollback": 1,
    "v2-pl-22-paged_ring_hybrid_evict": 1,
    "v2-pl-23-fused_layernorm_quant_pipeline": 1,
    "v2-pl-24-bucketed_radix_select_reduce": 1,
    "v2-pl-25-segmented_groupby_topk": 1,
    "v2-pl-27-streaming_dedup_window": 1,
    "v2-pl-32-streaming_merge_topk_quantile": 1,
    "v2-pl-34-fused_csr_spmm_topk": 1,
    "v2-pl-40-switch_moe_overflow_router": 1,
    "v2-pl-41-work_stealing_runtime": 1,
    "v2-pl-42-buddy_allocator_cleaner": 1,
    "v2-pl-47-cuckoo_tombstone_table": 2,
    "v2-pl-54-stm_commit_arbiter": 1,
    "v2-pl-55-lock_manager_deadlock": 1,
    "v2-pl-56-consistent_hash_ring": 1,
    "v2-pl-60-mk_paged_interpreter": 1,
    "v2-pl-65-mk_warp_pipeline_mbarrier": 1,
    "v2-pl-69-mk_instr_decoder_hazard": 1,
    "v2-pl-80-paged_sink_e4m3_decode": 2,
    "v2-pl-81-mla_latent_absorb_decode": 2,
    "v2-pl-82-spec_tree_verify_e5m2_kv": 2,
    "v2-pl-83-paged_cow_prefill_decode": 2,
    "v2-pl-85-moe_grouped_ffn_reroute": 2,
    "v2-pl-86-taskgraph_wavefront_gemm": 2,
    "v2-pl-87-dualtier_credit_router": 2,
    "v2-pl-88-mk_priority_preempt": 2,
    "v2-pl-90-mxfp4_atom_quant_dot": 1,
    "v2-pl-91-awq_actorder_repack_gemv": 1,
    "v2-pl-92-sat_segscan_select": 1,
    "v2-pl-93-microscale_requant_chain": 1,
}
TRITON_TASKS = frozenset(
    {
        "v2-pl-18-triton_flash_decode",
        "v2-pl-19-triton_persistent_grouped_gemm",
        "v2-pl-20-triton_fused_rmsnorm_matmul",
        "v2-pl-21-triton_chunked_state_scan",
        "v2-pl-26-triton_paged_decode_attention",
        "v2-pl-28-triton_blockwise_int8_gemm",
        "v2-pl-31-triton_fused_moe_dispatch",
        "v2-pl-33-triton_fused_layernorm_quant",
    }
)
RANDOMIZED_TASKS = CUDA_PARAMETERS.keys() | TRITON_TASKS
_CUDA_CONSTANT = b"0x9e3779b97f4a7c15"
_TORCH_SEED = b"g.manual_seed(SEED + seed_offset)"


def draw_parameter(task_id: str) -> int | None:
    """Draw independently for each task/rollout; odd CUDA increments have full period."""
    if task_id not in RANDOMIZED_TASKS:
        return None
    while True:
        parameter = secrets.randbits(64) | 1
        # Exclude the entire one-byte family identified by the public sanity grader.
        if parameter & ((1 << 56) - 1) != 0x3779B97F4A7C15:
            return parameter


def render(task_id: str, source: bytes, parameter: int) -> bytes:
    """Change only input generation, keeping shapes, oracles and verdicts intact.

    The host retains the parameter in the completed trace for reproducibility.
    Unsupported or drifted grader layouts fail before any candidate executes.
    """
    if not 0 < parameter < 1 << 64 or parameter % 2 == 0:
        raise ValueError("correctness input parameter must be an odd uint64")
    if task_id in CUDA_PARAMETERS:
        expected = CUDA_PARAMETERS[task_id]
        if source.count(_CUDA_CONSTANT) == expected:
            return source.replace(_CUDA_CONSTANT, f"0x{parameter:016x}".encode())
    elif task_id in TRITON_TASKS:
        if source.count(_TORCH_SEED) == 1:
            # Keep every per-case offset and make uint64 wrap explicit for torch.
            return source.replace(
                _TORCH_SEED,
                f"g.manual_seed(({parameter} + seed_offset) & ((1 << 64) - 1))".encode(),
            )
    raise PoolIntegrityError(
        f"{task_id}: unexpected randomized correctness grader layout"
    )


def validate_filename(task_id: str, filenames: list[str]) -> None:
    suffix = ".py" if task_id in TRITON_TASKS else ".cu"
    tests = [
        name for name in filenames if re.fullmatch(r"test_.*" + re.escape(suffix), name)
    ]
    if len(tests) != 1:
        raise PoolIntegrityError(
            f"{task_id}: expected one correctness grader, found {tests}"
        )

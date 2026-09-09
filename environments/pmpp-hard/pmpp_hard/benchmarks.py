"""Private benchmark seed policy and strict per-process measurement parsing."""

import math
import re
import secrets

POLICY = "paired-inputs-u64-complete-v2"
SEED_BITS = 64

# These two release benches add offsets directly to an unsigned seed. Other
# Triton benches already reserve high bits before adding their case offsets.
_WRAP_TASKS = {"v2-pl-18-triton_flash_decode", "v2-pl-26-triton_paged_decode_attention"}
_SEED_CALLS = (
    "g.manual_seed(SEED + seed_offset)",
    "g.manual_seed(SEED + 1_000_003 * (seed_offset + 1) + it)",
)
_WORKSPACE_TASK = "v2-pl-30-graph_frontier_bfs_state"
_WORKSPACE_GUARD = (
    b"const size_t workspace_bytes = solution_workspace_bytes(&spec);\n"
    b'        if (workspace_bytes == 0) throw std::runtime_error("solution_workspace_bytes returned 0");'
)


def draw_seed() -> int:
    """Draw one seed after submission, shared by every pair in this comparison."""
    return secrets.randbits(SEED_BITS)


def prepare_source(task_id: str, content: bytes) -> bytes:
    """Apply bounded compatibility corrections to provisioned benchmark copies.

    This preserves each benchmark's existing input generator and timing loop.
    Exact anchors fail closed if a supported release source changes.
    """
    if task_id == _WORKSPACE_TASK:
        if content.count(_WORKSPACE_GUARD) != 1:
            raise ValueError(f"{task_id}: expected one zero-workspace benchmark guard")
        # The contract permits storage in solution_init. Match the correctness
        # grader's one-byte allocation for a zero external-workspace request.
        return content.replace(
            _WORKSPACE_GUARD,
            b"size_t workspace_bytes = solution_workspace_bytes(&spec);\n"
            b"        if (workspace_bytes == 0) workspace_bytes = 1;",
        )
    if task_id not in _WRAP_TASKS:
        return content
    src = content.decode()
    for call in _SEED_CALLS:
        if src.count(call) != 1:
            raise ValueError(f"{task_id}: expected one benchmark seed anchor {call!r}")
        expression = call.removeprefix("g.manual_seed(").removesuffix(")")
        src = src.replace(call, f"g.manual_seed(({expression}) & ((1 << 64) - 1))")
    return src.encode()


def measurement(exit_code: int, output: str) -> dict:
    """Accept exactly one finite positive timing and one 64-bit output digest.

    A measurement belongs to one process; samples from different processes must
    never be joined by independently filtering timing and digest lists.
    """
    record = {"exit_code": exit_code, "log_tail": output[-16000:]}
    if exit_code != 0:
        record["error"] = f"benchmark exited {exit_code}"
        return record
    times = re.findall(r"\bavg_ms=([^\s]+)", output)
    digests = re.findall(r"\bout_fnv=([^\s]+)", output)
    if len(times) != 1 or len(digests) != 1:
        record["error"] = "expected exactly one avg_ms and one out_fnv"
        return record
    try:
        ms = float(times[0])
    except ValueError:
        ms = math.nan
    if not math.isfinite(ms) or ms <= 0:
        record["error"] = "avg_ms must be finite and positive"
    elif not re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]{16}", digests[0]):
        record["error"] = "out_fnv must be a 64-bit hexadecimal digest"
    else:
        record.update(avg_ms=ms, out_fnv=f"{int(digests[0], 16):016x}")
    return record

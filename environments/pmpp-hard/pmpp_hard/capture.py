"""Capture submitted code from the sandbox or recorded assistant messages.

An edited on-disk submission takes precedence over code recovered from recorded
assistant output.
"""

import re

from pmpp_hard.config import PMPPHardTaskData
from pmpp_hard.paths import DataTree

CODE_BLOCK_RE = re.compile(
    r"```(?:cuda|cpp|c\+\+|c|python|py)?\s*\n(.*?)```", re.DOTALL
)


def extract_code(text: str, must_contain: tuple[str, ...] = ()) -> str | None:
    """Return the largest fenced block, preferring blocks with solution markers."""
    if not text:
        return None
    blocks = CODE_BLOCK_RE.findall(text)
    if not blocks:
        return None
    if must_contain:
        sol = [b for b in blocks if any(m in b for m in must_contain)]
        if sol:
            return max(sol, key=len).strip() + "\n"
    return max(blocks, key=len).strip() + "\n"


async def capture_submission(
    task: PMPPHardTaskData, trace, runtime, tree: DataTree
) -> None:
    """Ensure a student file is on disk and record how it got there."""
    stub = (tree.bundle(task.task_id) / task.student_file).read_bytes()
    try:
        on_disk = await runtime.read(f"/app/{task.student_file}")
    except Exception:  # noqa: BLE001
        on_disk = b""
    if on_disk.strip() and on_disk != stub:
        trace.info["submission_source"] = "agent_edited_in_place"
        trace.info["submitted_kernel"] = on_disk.decode("utf-8", "replace")
        return
    # When the file still matches the stub, recover a submitted code block from
    # message history. Solution markers distinguish the submission from incidental
    # snippets.
    marker = (
        ("solution_run",)
        if task.student_file.endswith(".cu")
        else ("def ", "@triton.jit")
    )
    joined = "\n\n".join(
        getattr(m, "content", "") or "" for m in (trace.assistant_messages or [])
    )
    code = extract_code(joined, must_contain=marker)
    if code:
        await runtime.write(f"/app/{task.student_file}", code.encode())
        trace.info["submission_source"] = "extracted_from_messages"
        trace.info["submitted_kernel"] = code
    else:
        trace.info["submission_source"] = "none"
        trace.info["submitted_kernel"] = on_disk.decode("utf-8", "replace")

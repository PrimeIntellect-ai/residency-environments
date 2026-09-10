"""Apply optional KernelGuard analysis to captured kernels and timing metadata.

Timing-only filter verdicts invalidate the performance metric, while source-rule
filter verdicts invalidate both correctness and performance. Other verdicts are
recorded for diagnostics. If analysis is disabled or unavailable, scoring
continues with the existing checksum, source-policy, and residency checks.
"""

from pmpp_hard.config import PMPPHardConfig, PMPPHardTaskData

MARKER_FAIL = "PMPP_VERIFY_KERNELGUARD: FAIL"
MARKER_PASS = "PMPP_VERIFY_KERNELGUARD: PASS"


def _pmpp_info(trace) -> dict:
    return trace.info.setdefault("pmpp", {})


def check(
    config: PMPPHardConfig,
    task: PMPPHardTaskData,
    kernel: bytes,
    trace,
    student_ms: float | None = None,
    reference_ms: float | None = None,
    out_fnv_verified: bool | None = None,
) -> dict | None:
    """Analyze a captured kernel and return its filtering verdict.

    Timing-anomaly rules are omitted when output checksums already confirm that
    the timed student and reference runs produced matching results. Source rules
    remain active regardless of checksum availability. Disabled or unavailable
    analysis returns ``None`` and records the reason in the trace.
    """
    if not getattr(config, "kernelguard_check", False):
        return None
    try:
        import kernelguard as kg
    except Exception as e:  # noqa: BLE001 - optional analysis must not fail scoring
        _pmpp_info(trace)["kernelguard"] = (
            f"unavailable ({type(e).__name__}) — net skipped"
        )
        return None
    try:
        # Configure the telemetry-only speedup threshold for this task pool.
        kg.SCORE_CONFIG["extreme_speedup_above"] = float(
            getattr(config, "kernelguard_extreme_speedup", 10.0)
        )
        # Derive the timing floor from the paired reference so it scales with the
        # task rather than relying on one absolute threshold for all kernels.
        ratio_floor = float(getattr(config, "kernelguard_ratio_floor", 0.05))
        if reference_ms is not None and reference_ms > 0:
            derived_floor_s = (reference_ms / 1000.0) * ratio_floor
            kg.SCORE_CONFIG["suspect_floor_below"] = derived_floor_s
        else:
            # Restore the default when no paired reference is available because
            # SCORE_CONFIG is shared across calls.
            kg.SCORE_CONFIG["suspect_floor_below"] = 5e-06
        code = kernel.decode("utf-8", errors="replace")
        md: dict = {"problem_name": task.task_id}
        # Provide timing metadata only when checksum comparison has not already
        # validated the timed outputs.
        if out_fnv_verified is not True:
            if student_ms is not None and student_ms > 0:
                md["score"] = student_ms / 1000.0  # KernelGuard expects seconds.
            if (
                student_ms is not None
                and reference_ms is not None
                and student_ms > 0
                and reference_ms > 0
            ):
                md["improved_score"] = student_ms / 1000.0
                md["baseline_score"] = reference_ms / 1000.0
        res = kg.analyze_code(code, metadata=md, compute_structural_hash=False)
    except Exception as e:  # noqa: BLE001
        _pmpp_info(trace)["kernelguard"] = (
            f"analysis error ({type(e).__name__}: {str(e)[:120]})"
        )
        return None
    verdict = {
        "classification": res["classification"],
        "should_filter": bool(res["should_filter"]),
        "score_anomaly": res.get("filter_reason") == "score_anomaly",
        "patterns": [
            {
                "pattern": m.get("pattern"),
                "severity": m.get("severity"),
                "evidence": str(m.get("evidence", ""))[:200],
            }
            for m in res.get("matched_patterns", [])
        ][:12],
    }
    _pmpp_info(trace)["kernelguard"] = verdict
    return verdict


def enforce(verdict: dict | None, res: dict, trace) -> dict:
    """Apply the verdict mapping to a metrics dict (mutates + returns it)."""
    if not verdict or not verdict["should_filter"]:
        return res
    pats = (
        ",".join(p["pattern"] for p in verdict["patterns"] if p.get("pattern")) or "?"
    )
    if verdict["score_anomaly"]:
        res["performance_pass"] = 0.0
        _pmpp_info(trace)["kernelguard_enforced"] = (
            f"perf zeroed (score anomaly: {pats})"
        )
    else:
        res["run_pass"] = 0.0
        res["performance_pass"] = 0.0
        _pmpp_info(trace)["kernelguard_enforced"] = (
            f"run+perf zeroed (code rule: {pats})"
        )
    return res

"""Scoring for AnomalyXL-precise — strict-JSON, continuous-metric answer scoring.

Each gold answer is encoded as ``f"{category}:{json}"``; the parser takes the final
JSON object in the model output, validates it against the category's typed schema, and
scores per a category-specific metric dispatch. Format failure is a zero — the grounding
requirement being measured is the same one that makes a malformed answer ungradeable.
"""

from __future__ import annotations

import json
import math
import re

__all__ = [
    "ALL_METRIC_NAMES",
    "PRIMARY_METRIC",
    "parse_structured_pred",
    "score_answer",
    "score_metrics",
    "strip_reasoning",
]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_NONE_KIND = "No anomaly"


def _is_int(value: object) -> bool:
    """A JSON integer. ``100.0`` passes (it *is* that index); ``100.9`` and ``"100"``
    do not — truncating them would score an answer the model did not give."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value) and value.is_integer()


def _is_num(value: object) -> bool:
    """A finite JSON number. Excludes bools, null, NaN and Infinity."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_localize(obj: dict) -> bool:
    """``{"present": true|false, "start": <int>, "end": <int>}``. The question says
    start/end are ignored when ``present`` is false, so they are optional there."""
    if not isinstance(obj.get("present"), bool):
        return False
    if not obj["present"]:
        return True
    return _is_int(obj.get("start")) and _is_int(obj.get("end"))


def _valid_classify_with_evidence(obj: dict) -> bool:
    """``{"kind": <label>, "start": <int>, "end": <int>}``. The label vocabulary is a
    value check, left to the scorer; the no-anomaly label carries no window."""
    kind = obj.get("kind")
    if not isinstance(kind, str):
        return False
    if kind == _NONE_KIND:
        return True
    return _is_int(obj.get("start")) and _is_int(obj.get("end"))


def _valid_measure_magnitude(obj: dict) -> bool:
    """``{"magnitude_sigma": <float>}``."""
    return _is_num(obj.get("magnitude_sigma"))


def _valid_localize_all_channels(obj: dict) -> bool:
    """``{"anomalies": [{"channel": <str>, "start": <int>, "end": <int>}, ...]}``.
    An empty list is the valid no-anomaly answer."""
    events = obj.get("anomalies")
    if not isinstance(events, list):
        return False
    return all(
        isinstance(e, dict) and isinstance(e.get("channel"), str) and _is_int(e.get("start")) and _is_int(e.get("end"))
        for e in events
    )


def _valid_lead_lag_with_magnitude(obj: dict) -> bool:
    """``{"direction": "lead"|"lag"|"independent", "lag_samples": <int>}``. The
    direction vocabulary is a value check, left to the scorer."""
    return isinstance(obj.get("direction"), str) and _is_int(obj.get("lag_samples"))


_SCHEMAS = {
    "localize": _valid_localize,
    "classify_with_evidence": _valid_classify_with_evidence,
    "measure_magnitude": _valid_measure_magnitude,
    "localize_all_channels": _valid_localize_all_channels,
    "lead_lag_with_magnitude": _valid_lead_lag_with_magnitude,
}

PRIMARY_METRIC: dict[str, str] = {
    "localize": "iou_or_correct_empty",
    "classify_with_evidence": "kind_x_iou",
    "measure_magnitude": "composite",
    "localize_all_channels": "f1",
    "lead_lag_with_magnitude": "composite",
}

_METRIC_NAMES: dict[str, tuple[str, ...]] = {
    "localize": (
        "iou_or_correct_empty",
        "presence_correct",
        "iou",
        "acc_at_iou_0p3",
        "acc_at_iou_0p5",
        "acc_at_iou_0p8",
        "start_mae_fraction",
        "end_mae_fraction",
    ),
    "classify_with_evidence": (
        "kind_x_iou",
        "kind_correct",
        "iou",
        "kind_and_iou_above_0p5",
    ),
    "measure_magnitude": (
        "composite",
        "acc_at_rel_25pct",
        "acc_at_rel_10pct",
        "acc_at_rel_50pct",
        "rel_err",
    ),
    "localize_all_channels": (
        "f1",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
    ),
    "lead_lag_with_magnitude": (
        "composite",
        "direction_correct",
        "lag_acc_at_1pct",
        "lag_acc_at_5pct",
        "lag_mae_fraction",
    ),
}

ALL_METRIC_NAMES: tuple[str, ...] = tuple(
    f"{category}/{name}" for category, names in _METRIC_NAMES.items() for name in names
)
"""Every metric a run can report, category-prefixed. The prefix keeps each metric
averaged over its own category's rows only, and separates the names categories
share: ``iou`` means one thing for ``localize`` and another for
``classify_with_evidence``, as do the two ``composite`` metrics."""

_DISPATCH = {
    "localize": None,  # filled below
    "classify_with_evidence": None,
    "measure_magnitude": None,
    "localize_all_channels": None,
    "lead_lag_with_magnitude": None,
}


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks before answer extraction."""
    if not text:
        return text
    return _THINK_RE.sub("", text)


def _top_level_objects(text: str) -> list[str]:
    """Every balanced, non-nested ``{...}`` span in ``text``, in order."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start : i + 1])
    return spans


def parse_structured_pred(text: str, category: str) -> dict | None:
    """Parse the model's last JSON object and validate it against ``category``.

    The question asks for one object stating the final answer, so the last complete
    object in the reply is the answer — and the only candidate. Falling back to an
    earlier object would score a claim the reply did not end on. Returns ``None``
    when that object is missing, unparseable, or off-schema; format failure is a zero.
    """
    if not text:
        return None
    spans = _top_level_objects(strip_reasoning(text))
    if not spans:
        return None
    try:
        obj = json.loads(spans[-1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or not _SCHEMAS[category](obj):
        return None
    return obj


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """IoU on half-open intervals ``[start, end)``."""
    if a_start >= a_end or b_start >= b_end:
        return 0.0
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    if inter_end <= inter_start:
        return 0.0
    inter = inter_end - inter_start
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def _zero_metrics(category: str) -> dict[str, float]:
    return {name: 0.0 for name in _METRIC_NAMES[category]}


def _finite(value: object) -> bool:
    """True for a real, finite number. Guards against null, NaN, Infinity."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _band_token(b: float) -> str:
    return f"{b:.1f}".replace(".", "p")


# --------------------------------------------------------------------- scorers


def _score_localize(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("localize")
    gold_present = bool(gold["present"])
    if pred is None:
        out["presence_correct"] = 0.0
        return out

    pred_present = bool(pred.get("present", False))
    out["presence_correct"] = 1.0 if pred_present == gold_present else 0.0

    if not gold_present and not pred_present:
        out["iou_or_correct_empty"] = 1.0
        out["iou"] = 1.0
        for band in (0.3, 0.5, 0.8):
            out[f"acc_at_iou_{_band_token(band)}"] = 1.0
        out["start_mae_fraction"] = 0.0
        out["end_mae_fraction"] = 0.0
        return out

    if gold_present and pred_present:
        if not (_finite(pred.get("start")) and _finite(pred.get("end"))):
            return out
        L = int(gold.get("_L", 0))
        iou = _iou(gold["start"], gold["end"], int(pred["start"]), int(pred["end"]))
        out["iou"] = iou
        out["iou_or_correct_empty"] = iou
        for band in (0.3, 0.5, 0.8):
            out[f"acc_at_iou_{_band_token(band)}"] = 1.0 if iou >= band else 0.0
        if L > 0:
            out["start_mae_fraction"] = abs(int(pred["start"]) - gold["start"]) / L
            out["end_mae_fraction"] = abs(int(pred["end"]) - gold["end"]) / L
        else:
            out["start_mae_fraction"] = 1.0
            out["end_mae_fraction"] = 1.0
        return out

    return out


def _score_classify_with_evidence(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("classify_with_evidence")
    gold_kind = str(gold["kind"])
    gold_none = gold_kind == _NONE_KIND
    if pred is None:
        return out
    pred_kind = str(pred.get("kind", ""))
    pred_none = pred_kind == _NONE_KIND
    kind_correct = 1.0 if pred_kind == gold_kind else 0.0
    out["kind_correct"] = kind_correct
    if gold_none or pred_none:
        if gold_none and pred_none:
            out["iou"] = 1.0
            out["kind_x_iou"] = 1.0
            out["kind_and_iou_above_0p5"] = 1.0
        return out
    if _finite(pred.get("start")) and _finite(pred.get("end")):
        iou = _iou(gold["start"], gold["end"], int(pred["start"]), int(pred["end"]))
    else:
        iou = 0.0
    out["iou"] = iou
    out["kind_x_iou"] = kind_correct * iou
    out["kind_and_iou_above_0p5"] = 1.0 if (kind_correct == 1.0 and iou >= 0.5) else 0.0
    return out


def _score_measure_magnitude(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("measure_magnitude")
    if pred is None:
        out["rel_err"] = 1.0
        return out
    raw = pred.get("magnitude_sigma")
    if not _finite(raw):
        out["rel_err"] = 1.0
        return out
    pred_sigma = float(raw)
    gold_sigma = float(gold["magnitude_sigma"])
    if gold_sigma <= 0:
        out["rel_err"] = 1.0
        return out
    rel_err = abs(pred_sigma - gold_sigma) / gold_sigma
    out["rel_err"] = rel_err
    out["acc_at_rel_10pct"] = 1.0 if rel_err <= 0.10 else 0.0
    out["acc_at_rel_25pct"] = 1.0 if rel_err <= 0.25 else 0.0
    out["acc_at_rel_50pct"] = 1.0 if rel_err <= 0.50 else 0.0
    out["composite"] = max(0.0, 1.0 - rel_err / 0.5)
    return out


def _greedy_set_match(gold_events: list[dict], pred_events: list[dict], *, iou_threshold: float) -> int:
    """Greedy IoU-descending matching across channel-paired events. Returns TP count."""
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gold_events):
        g_start, g_end = int(g["start"]), int(g["end"])
        g_ch = g["channel"]
        for pi, p in enumerate(pred_events):
            if str(p["channel"]) != str(g_ch):
                continue
            iou = _iou(g_start, g_end, int(p["start"]), int(p["end"]))
            if iou > iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    tp = 0
    for _, gi, pi in candidates:
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        tp += 1
    return tp


def _score_localize_all_channels(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("localize_all_channels")
    gold_events = gold.get("anomalies", [])
    if pred is None:
        out["fn"] = float(len(gold_events))
        return out
    pred_events = pred.get("anomalies", []) if isinstance(pred.get("anomalies"), list) else []
    pred_events = [
        e
        for e in pred_events
        if isinstance(e, dict) and "channel" in e and _finite(e.get("start")) and _finite(e.get("end"))
    ]

    tp = _greedy_set_match(gold_events, pred_events, iou_threshold=0.3)
    n_gold = len(gold_events)
    n_pred = len(pred_events)
    fp = max(0, n_pred - tp)
    fn = max(0, n_gold - tp)
    out["tp"] = float(tp)
    out["fp"] = float(fp)
    out["fn"] = float(fn)
    if n_gold == 0 and n_pred == 0:
        out["precision"] = 1.0
        out["recall"] = 1.0
        out["f1"] = 1.0
        return out
    if n_pred == 0 or n_gold == 0:
        return out
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gold if n_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    out["precision"] = precision
    out["recall"] = recall
    out["f1"] = f1
    return out


def _score_lead_lag_with_magnitude(pred: dict | None, gold: dict) -> dict[str, float]:
    out = _zero_metrics("lead_lag_with_magnitude")
    if pred is None:
        out["lag_mae_fraction"] = 1.0
        return out
    pred_dir = str(pred.get("direction", ""))
    gold_dir = str(gold["direction"])
    direction_correct = 1.0 if pred_dir == gold_dir else 0.0
    out["direction_correct"] = direction_correct
    L = int(gold.get("_L", 0))

    if gold_dir == "independent":
        if direction_correct == 1.0:
            out["lag_mae_fraction"] = 0.0
            out["composite"] = 1.0
            out["lag_acc_at_1pct"] = 1.0
            out["lag_acc_at_5pct"] = 1.0
        else:
            out["lag_mae_fraction"] = 1.0
        return out

    pred_lag = pred.get("lag_samples")
    if not _finite(pred_lag) or L <= 0:
        out["lag_mae_fraction"] = 1.0
        return out
    if direction_correct != 1.0:
        out["lag_mae_fraction"] = 1.0
        return out
    lag_mae = abs(int(pred_lag) - int(gold["lag_samples"])) / L
    out["lag_mae_fraction"] = lag_mae
    out["lag_acc_at_1pct"] = 1.0 if lag_mae <= 0.01 else 0.0
    out["lag_acc_at_5pct"] = 1.0 if lag_mae <= 0.05 else 0.0
    out["composite"] = max(0.0, 1.0 - min(1.0, lag_mae / 0.05))
    return out


_DISPATCH = {
    "localize": _score_localize,
    "classify_with_evidence": _score_classify_with_evidence,
    "measure_magnitude": _score_measure_magnitude,
    "localize_all_channels": _score_localize_all_channels,
    "lead_lag_with_magnitude": _score_lead_lag_with_magnitude,
}


def _split_gold(gold: str) -> tuple[str, dict]:
    """``"category:<json>"`` → ``(category, payload_dict)``."""
    cat, sep, payload = gold.partition(":")
    if not sep:
        raise ValueError(f"Gold missing category prefix: {gold!r}")
    if cat not in _DISPATCH:
        raise KeyError(f"Unknown AnomalyXL category: {cat!r}")
    return cat, json.loads(payload)


def score_answer(text: str, gold: str) -> float:
    """Primary reward: the category's primary metric on a 0–1 scale."""
    try:
        cat, gold_payload = _split_gold(gold)
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0
    pred = parse_structured_pred(text or "", cat)
    metrics = _DISPATCH[cat](pred, gold_payload)
    return float(metrics.get(PRIMARY_METRIC[cat], 0.0))


def score_metrics(text: str, gold: str) -> dict[str, float]:
    """This row's metrics, category-prefixed — only the ones its category defines."""
    try:
        cat, gold_payload = _split_gold(gold)
    except (KeyError, ValueError, json.JSONDecodeError):
        return {}
    pred = parse_structured_pred(text or "", cat)
    per_cat = _DISPATCH[cat](pred, gold_payload)
    return {f"{cat}/{name}": float(per_cat[name]) for name in _METRIC_NAMES[cat] if name in per_cat}

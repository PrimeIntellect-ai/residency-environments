"""Deterministic biological scoring and tagged-answer parsing."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
HALLMARKS_PATH = DATA_DIR / "hallmark_names.json"

TARGET_TAG = "TARGET"
MOA_TAG = "MOA"
PATHWAYS_TAG = "PATHWAYS"
VIABILITY_TAG = "VIABILITY"
CYCLE_TAG = "CELL_CYCLE"
STRESS_TAG = "STRESS"
MAGNITUDE_TAG = "MAGNITUDE"
MAX_PATHWAY_ENTRIES = 5
TARGET_COUNT_MULTIPLIER = 2

CYCLE_CLASSES = ["arrest", "no_effect", "proliferation"]
STRESS_CLASSES = ["none", "apoptosis", "UPR", "DNA_damage"]
MAGNITUDE_CLASSES = ["inert", "moderate", "strong"]

DEFAULT_REWARD_WEIGHTS = {
    "target": 0.15,
    "moa": 0.15,
    "pathways": 0.25,
    "phenotype": 0.45,
}

VIABILITY_TOL_FULL = 0.25
VIABILITY_TOL_ZERO = 2.0

PHENOTYPE_TAGS = {
    "viability": VIABILITY_TAG,
    "cell_cycle": CYCLE_TAG,
    "stress": STRESS_TAG,
    "magnitude": MAGNITUDE_TAG,
}

PHENOTYPE_CLASSES = {
    "cell_cycle": CYCLE_CLASSES,
    "stress": STRESS_CLASSES,
    "magnitude": MAGNITUDE_CLASSES,
}


@lru_cache(maxsize=1)
def hallmark_names() -> list[str]:
    with HALLMARKS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_tag(text: str, tag: str) -> tuple[str | None, str | None]:
    """Extract one nonempty answer block, rejecting duplicate or broken delimiters."""
    if not isinstance(text, str):
        return None, "missing"
    tag = re.escape(tag)
    markers = list(re.finditer(rf"<\s*(/?)\s*{tag}(?=[\s/>]|$)", text, re.IGNORECASE))
    if not markers:
        return None, "missing"
    opening_count = sum(not marker.group(1) for marker in markers)
    closing_count = len(markers) - opening_count
    if opening_count > 1 or closing_count > 1:
        return None, "duplicate"
    match = re.search(rf"<{tag}>([^<>]*)</{tag}>", text, re.IGNORECASE | re.DOTALL)
    if opening_count != 1 or closing_count != 1 or match is None:
        return None, "malformed"
    value = match.group(1).strip()
    return (value, None) if value else (None, "empty")


def extract_tag(text: str, tag: str) -> str | None:
    return parse_tag(text, tag)[0]


def parse_finite_number(text: str | None) -> float | None:
    if text is None or not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text.strip()):
        return None
    value = float(text.strip())
    return value if math.isfinite(value) else None


def parse_list(text: str | None) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[|,]", text)
    return [part.strip().upper() for part in parts if part.strip()]


def pathway_entries(text: str | None) -> list[str]:
    if not text:
        return []
    return [token.strip() for token in re.split(r"[,;\n]", text) if token.strip()]


def parse_pathway_pairs(text: str | None) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for token in pathway_entries(text):
        if ":" not in token:
            continue
        name, direction = token.rsplit(":", 1)
        name = name.strip().upper()
        direction = direction.strip().lower()
        if direction in ("up", "down") and name:
            output.append((name, direction))
    return output


def exact_hallmark_name(name: str | None) -> str | None:
    if not name:
        return None
    key = re.sub(r"[^A-Z0-9]+", "_", name.strip().upper())
    key = re.sub(r"_+", "_", key).strip("_")
    names = set(hallmark_names())
    if key in names:
        return key
    prefixed = f"HALLMARK_{key}"
    if not key.startswith("HALLMARK_") and prefixed in names:
        return prefixed
    return None


def pathway_name_token(name: str) -> str:
    canonical = exact_hallmark_name(name)
    return canonical if canonical is not None else f"INVALID::{name.strip().upper()}"


def f1_set(pred: set, gt: set) -> float:
    if not gt:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    true_positives = len(pred & gt)
    if true_positives == 0:
        return 0.0
    precision = true_positives / len(pred)
    recall = true_positives / len(gt)
    return 2 * precision * recall / (precision + recall)


def normalize_moa(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.lower().strip().split())


def target_prediction_limit(ground_truth: list[str]) -> int:
    return TARGET_COUNT_MULTIPLIER * len({gene.upper() for gene in ground_truth})


def score_target(pred_text: str | None, ground_truth: list[str]) -> float:
    if len(parse_list(pred_text)) > target_prediction_limit(ground_truth):
        return 0.0
    return score_target_f1(pred_text, ground_truth)


def score_target_f1(pred_text: str | None, ground_truth: list[str]) -> float:
    if not ground_truth or pred_text is None:
        return 0.0
    return f1_set(
        set(parse_list(pred_text)),
        set(gene.upper() for gene in ground_truth),
    )


def score_moa(pred_text: str | None, ground_truth: str | None) -> float:
    if not ground_truth or pred_text is None:
        return 0.0
    return float(normalize_moa(pred_text) == normalize_moa(ground_truth))


def score_pathways(
    pred_text: str | None,
    ground_truth: list[tuple[str, str]],
) -> float:
    if len(pathway_entries(pred_text)) > MAX_PATHWAY_ENTRIES:
        return 0.0
    return score_pathway_signed_f1(pred_text, ground_truth)


def score_pathway_signed_f1(
    pred_text: str | None,
    ground_truth: list[tuple[str, str]],
) -> float:
    if not ground_truth or pred_text is None:
        return 0.0
    pred = set(parse_pathway_pairs(pred_text))
    gt = set((name.upper().replace("HALLMARK_", ""), direction) for name, direction in ground_truth)
    pred = set((name.replace("HALLMARK_", ""), direction) for name, direction in pred)
    return f1_set(pred, gt)


def score_pathway_name_validity(pred_text: str | None) -> float:
    pred = parse_pathway_pairs(pred_text)
    if not pred:
        return 0.0
    valid = sum(1 for name, _ in pred if exact_hallmark_name(name) is not None)
    return valid / len(pred)


def score_pathway_name_f1(
    pred_text: str | None,
    ground_truth: list[tuple[str, str]],
) -> float:
    if not ground_truth:
        return 0.0
    pred_names = {pathway_name_token(name) for name, _ in parse_pathway_pairs(pred_text)}
    gt_names = {exact_hallmark_name(name) or name.upper() for name, _ in ground_truth}
    return f1_set(pred_names, gt_names)


def score_pathway_direction_accuracy(
    pred_text: str | None,
    ground_truth: list[tuple[str, str]],
) -> float:
    if not ground_truth:
        return 0.0
    pred_directions: dict[str, set[str]] = {}
    for name, direction in parse_pathway_pairs(pred_text):
        canonical = exact_hallmark_name(name)
        if canonical is not None:
            pred_directions.setdefault(canonical, set()).add(direction)
    gt_directions = {(exact_hallmark_name(name) or name.upper()): direction for name, direction in ground_truth}
    overlap = set(pred_directions) & set(gt_directions)
    if not overlap:
        return 0.0
    correct = sum(1 for name in overlap if gt_directions[name] in pred_directions[name])
    return correct / len(overlap)


def score_viability(pred_text: str | None, ground_truth: float | None) -> float:
    if ground_truth is None or pred_text is None:
        return 0.0
    pred = parse_finite_number(pred_text)
    if pred is None:
        return 0.0
    error = abs(pred - ground_truth)
    if error <= VIABILITY_TOL_FULL:
        return 1.0
    if error >= VIABILITY_TOL_ZERO:
        return 0.0
    return 1.0 - ((error - VIABILITY_TOL_FULL) / (VIABILITY_TOL_ZERO - VIABILITY_TOL_FULL))


def score_class(
    pred_text: str | None,
    ground_truth: str | None,
    classes: list[str],
) -> float:
    if not ground_truth or pred_text is None:
        return 0.0
    pred = pred_text.strip().lower()
    gt_normalized = ground_truth.strip().lower()
    classes_normalized = [value.lower() for value in classes]
    if pred not in classes_normalized:
        return 0.0
    return float(pred == gt_normalized)


def score_phenotype(phenotype: str, pred_text: str | None, ground_truth: Any) -> float:
    if phenotype == "viability":
        return score_viability(pred_text, ground_truth)
    return score_class(pred_text, ground_truth, PHENOTYPE_CLASSES[phenotype])


def phenotype_gt_key(phenotype: str) -> str:
    return {
        "viability": "viability_lfc",
        "cell_cycle": "cell_cycle",
        "stress": "stress",
        "magnitude": "magnitude",
    }[phenotype]


def parse_answer(answer: str | dict[str, Any]) -> dict[str, Any]:
    return json.loads(answer) if isinstance(answer, str) else answer


def score_response(
    response: str,
    answer: str | dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Apply a whole-answer format gate after biological scoring and answer-list limits."""
    gt = parse_answer(answer)
    reward_weights = weights or DEFAULT_REWARD_WEIGHTS

    target_genes = gt.get("target") or []
    target_text = extract_tag(response, TARGET_TAG)
    target = score_target(target_text, target_genes)
    target_f1 = score_target_f1(target_text, target_genes)
    target_count = len(parse_list(target_text))
    target_limit = target_prediction_limit(target_genes)
    moa = score_moa(extract_tag(response, MOA_TAG), gt["moa"]) if gt.get("moa") else 0.0

    pathway_pairs = [tuple(pair) for pair in gt.get("pathways_signed") or []]
    pathway_text = extract_tag(response, PATHWAYS_TAG)
    pathways = score_pathways(pathway_text, pathway_pairs) if pathway_pairs else 0.0
    pathway_f1 = score_pathway_signed_f1(pathway_text, pathway_pairs) if pathway_pairs else 0.0
    pathway_count = len(pathway_entries(pathway_text))
    pathway_name_validity = score_pathway_name_validity(pathway_text) if pathway_pairs else 0.0
    pathway_name_f1 = score_pathway_name_f1(pathway_text, pathway_pairs) if pathway_pairs else 0.0
    pathway_direction_accuracy = score_pathway_direction_accuracy(pathway_text, pathway_pairs) if pathway_pairs else 0.0

    phenotype = gt.get("phenotype")
    if phenotype:
        phenotype_text = extract_tag(response, PHENOTYPE_TAGS[phenotype])
        phenotype_score = score_phenotype(
            phenotype,
            phenotype_text,
            gt.get(phenotype_gt_key(phenotype)),
        )
    else:
        phenotype_score = 0.0

    total = 0.0
    total_weight = 0.0
    if gt.get("target"):
        total += reward_weights["target"] * target
        total_weight += reward_weights["target"]
    if gt.get("moa"):
        total += reward_weights["moa"] * moa
        total_weight += reward_weights["moa"]
    if pathway_pairs:
        total += reward_weights["pathways"] * pathways
        total_weight += reward_weights["pathways"]
    if phenotype:
        total += reward_weights["phenotype"] * phenotype_score
        total_weight += reward_weights["phenotype"]
    aggregate = total / total_weight if total_weight > 0 else 0.0

    required_tags: list[str] = []
    if gt.get("target"):
        required_tags.append(TARGET_TAG)
    if gt.get("moa"):
        required_tags.append(MOA_TAG)
    if pathway_pairs:
        required_tags.append(PATHWAYS_TAG)
    if phenotype:
        required_tags.append(PHENOTYPE_TAGS[phenotype])
    parsed = {tag: parse_tag(response, tag) for tag in required_tags}
    issues = [issue for _, issue in parsed.values() if issue is not None]
    invalid_numeric = (
        phenotype == "viability"
        and parsed[VIABILITY_TAG][0] is not None
        and parse_finite_number(parsed[VIABILITY_TAG][0]) is None
    )
    valid_count = len(required_tags) - len(issues) - int(invalid_numeric)
    format_compliance = valid_count / len(required_tags) if required_tags else 0.0
    format_valid = bool(required_tags) and valid_count == len(required_tags)

    return {
        "aggregate_reward": aggregate if format_valid else 0.0,
        "target_score": target,
        "target_f1": target_f1,
        "target_prediction_count": float(target_count),
        "target_prediction_limit": float(target_limit),
        "target_count_overflow": float(target_count > target_limit),
        "moa_accuracy": moa,
        "pathway_score": pathways,
        "pathway_signed_f1": pathway_f1,
        "pathway_prediction_count": float(pathway_count),
        "pathway_count_overflow": float(pathway_count > MAX_PATHWAY_ENTRIES),
        "pathway_name_validity": pathway_name_validity,
        "pathway_name_f1": pathway_name_f1,
        "pathway_direction_accuracy": pathway_direction_accuracy,
        "phenotype_score": phenotype_score,
        "format_compliance": format_compliance,
        "format_valid": float(format_valid),
        "format_reward_before_gate": aggregate,
        "format_missing_tags": float(issues.count("missing")),
        "format_duplicate_tags": float(issues.count("duplicate")),
        "format_malformed_tags": float(issues.count("malformed")),
        "format_empty_tags": float(issues.count("empty")),
        "format_invalid_numeric": float(invalid_numeric),
    }

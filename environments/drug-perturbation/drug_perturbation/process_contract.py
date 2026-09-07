"""Frozen schema and deterministic scoring for the process-coherence judge.

The language-model judge assigns only discrete semantic labels.  This module
validates the task-aware schema and cited nonempty message locations, then
maps the labels to a continuous score. Citation validity does not establish
biological truth or semantic entailment. Any
technical or proof failure returns the fail-open score of 1.0.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

RUBRIC_VERSION = "reasoning-process-v0.3"

NODE_STATUS_SCORES = {
    "supported": 1.0,
    "weak": 0.65,
    "unsupported": 0.25,
    "contradicted": 0.0,
}
TRANSITION_STATUS_SCORES = {
    "coherent": 1.0,
    "weak": 0.65,
    "disconnected": 0.25,
    "contradicted": 0.0,
}

TASK_GRAPHS: dict[str, dict[str, list[Any]]] = {
    "target_from_smiles": {
        "provided_nodes": ["compound_evidence"],
        "generated_nodes": ["target"],
        "audited_nodes": ["compound_evidence", "target"],
        "transitions": [["compound_evidence", "target"]],
    },
    "moa_from_target": {
        "provided_nodes": ["target"],
        "generated_nodes": ["moa"],
        "audited_nodes": ["target", "moa"],
        "transitions": [["target", "moa"]],
    },
    "pathways_from_moa": {
        "provided_nodes": ["moa"],
        "generated_nodes": ["pathways"],
        "audited_nodes": ["moa", "pathways"],
        "transitions": [["moa", "pathways"]],
    },
    "phenotype_from_pathways": {
        "provided_nodes": ["pathways"],
        "generated_nodes": ["phenotype"],
        "audited_nodes": ["pathways", "phenotype"],
        "transitions": [["pathways", "phenotype"]],
    },
    "smiles_only": {
        "provided_nodes": ["compound_evidence"],
        "generated_nodes": ["target", "moa", "pathways", "phenotype"],
        "audited_nodes": [
            "compound_evidence",
            "target",
            "moa",
            "pathways",
            "phenotype",
        ],
        "transitions": [
            ["compound_evidence", "target"],
            ["target", "moa"],
            ["moa", "pathways"],
            ["pathways", "phenotype"],
        ],
    },
}


PROCESS_SYSTEM_PROMPT = """\
You are auditing whether a pharmacology answer was produced by a coherent,
repeatable reasoning process. You are NOT grading the answer against hidden
ground truth, and you are NOT rewarding verbosity. Audit only the task graph
listed in the candidate case.

First audit every node listed in task_graph.audited_nodes, in that exact order.
For a provided node, judge whether the trace faithfully uses the supplied
premise; do not question whether the prompt itself is true. For a generated
node:
- supported: the settled claim has a specific, defensible basis in the prompt,
  tool evidence, or the trace's biological reasoning.
- weak: the claim has a relevant plausible basis, but it is materially
  incomplete, vague, or explicitly uncertain.
- unsupported: the claim is asserted or guessed without a meaningful relevant
  basis, or its stated basis concerns a different compound/problem.
- contradicted: the settled claim conflicts with supplied evidence or with the
  trace's own settled reasoning.

Then audit every directed transition:
- coherent: the trace gives a meaningful causal or mechanistic link from the
  upstream node to the downstream node.
- weak: the link is plausible and relevant but skips an important bridge.
- disconnected: the downstream claim is not motivated by the upstream node;
  juxtaposition, generic boilerplate, or a lucky final guess is not a link.
- contradicted: the downstream claim conflicts with the stated implication of
  the upstream node.

Judge the final settled chain. Do not penalize exploratory reasoning that is
clearly corrected. Do not penalize brevity when one concise sentence supplies
the needed biological link. Do not upgrade a response for length, confidence,
formatting, or merely restating the prompt. Tool similarity may support a
bounded analogy, but reasoning copied from an unrelated compound does not
support the queried compound. Explicitly compare the compound, cell line,
provided targets/mechanism/pathways, and assay context used in the trace with
the original prompt. Reasoning about a different prompt's biological facts
contradicts the provided node and disconnects its downstream transition.

Important evidence boundary: a bare answer tag is a conclusion, not its
support. Prompt text may establish a provided node, but prompt text alone
cannot make a generated node supported or a transition coherent. Those labels
require at least one reference to substantive candidate_trace reasoning. If
the trace contains only the answer tag, score the generated node unsupported
and its incoming transition disconnected.

For every node and transition, cite one or two locations from the supplied
evidence_locations catalog. A citation has exactly:
  source: "prompt" or "trace"
  message_index: zero-based index in that source
  field: "content" or "reasoning_content"

Return only one JSON object with exactly this shape:
{
  "rubric_version": "reasoning-process-v0.3",
  "nodes": [
    {
      "name": "<one required audited node>",
      "status": "supported|weak|unsupported|contradicted",
      "evidence": [
        {
          "source": "prompt|trace",
          "message_index": 0,
          "field": "content|reasoning_content"
        }
      ]
    }
  ],
  "transitions": [
    {
      "from": "<required upstream node>",
      "to": "<required downstream node>",
      "status": "coherent|weak|disconnected|contradicted",
      "evidence": [{"source": "trace", "message_index": 0,
                    "field": "reasoning_content"}]
    }
  ],
  "summary": "one short sentence"
}

Include every required node and transition exactly once and no others. Do not
emit a numeric score; the verifier applies the frozen mapping.
"""


@dataclass(frozen=True)
class EvidenceRef:
    source: str
    message_index: int
    field: str


@dataclass(frozen=True)
class NodeAssessment:
    name: str
    status: str
    evidence: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class TransitionAssessment:
    source_node: str
    target_node: str
    status: str
    evidence: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class ProcessJudgment:
    nodes: tuple[NodeAssessment, ...]
    transitions: tuple[TransitionAssessment, ...]
    summary: str


@dataclass(frozen=True)
class ProcessEvaluation:
    process_score: float
    node_score: float
    transition_score: float
    parse_success: bool
    proof_valid: bool
    support_adjustments: int
    judgment: ProcessJudgment | None


def infer_entry_point(original_prompt: list[dict[str, Any]]) -> str:
    """Infer the direct-link task from its requested answer tags."""
    text = "\n".join(str(message.get("content") or "") for message in original_prompt if isinstance(message, dict))
    requested = text.rsplit("Answer the requested task.", 1)[-1]
    if all(tag in requested for tag in ("<TARGET>", "<MOA>", "<PATHWAYS>")):
        return "smiles_only"
    if "<TARGET>" in requested:
        return "target_from_smiles"
    if "<MOA>" in requested:
        return "moa_from_target"
    if "<PATHWAYS>" in requested:
        return "pathways_from_moa"
    if any(tag in requested for tag in ("<VIABILITY>", "<CELL_CYCLE>", "<STRESS>")):
        return "phenotype_from_pathways"
    raise ValueError("Unable to infer a supported entry point from requested tags")


def task_graph(entry_point: str) -> dict[str, list[Any]]:
    try:
        graph = TASK_GRAPHS[entry_point]
    except KeyError as exc:
        raise ValueError(f"Unsupported process-judge entry point: {entry_point}") from exc
    return {
        "provided_nodes": list(graph["provided_nodes"]),
        "generated_nodes": list(graph["generated_nodes"]),
        "audited_nodes": list(graph["audited_nodes"]),
        "transitions": [list(edge) for edge in graph["transitions"]],
    }


def _last_assistant_content(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return ""
    for message in reversed(completion):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "assistant":
            continue
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, str):
            return content
    return ""


def _candidate_json_strings(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL):
        candidates.append(match.group(1))
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[index : index + end])
        break
    if text.strip().startswith("{"):
        candidates.append(text.strip())
    return candidates


def _parse_evidence(raw: Any) -> tuple[EvidenceRef, ...] | None:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 2:
        return None
    parsed: list[EvidenceRef] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {
            "source",
            "message_index",
            "field",
        }:
            return None
        source = item["source"]
        index = item["message_index"]
        field = item["field"]
        if not isinstance(source, str) or source not in {"prompt", "trace"}:
            return None
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            return None
        if not isinstance(field, str) or field not in {"content", "reasoning_content"}:
            return None
        parsed.append(EvidenceRef(source, index, field))
    return tuple(parsed)


def _parse_object(
    data: dict[str, Any],
    *,
    expected_nodes: list[str],
    expected_transitions: list[list[str]],
) -> ProcessJudgment | None:
    if set(data) != {"rubric_version", "nodes", "transitions", "summary"}:
        return None
    if data["rubric_version"] != RUBRIC_VERSION or not isinstance(data["summary"], str):
        return None
    raw_nodes = data["nodes"]
    raw_transitions = data["transitions"]
    if not isinstance(raw_nodes, list) or not isinstance(raw_transitions, list):
        return None

    nodes: list[NodeAssessment] = []
    for item in raw_nodes:
        if not isinstance(item, dict) or set(item) != {"name", "status", "evidence"}:
            return None
        evidence = _parse_evidence(item["evidence"])
        if (
            not isinstance(item["name"], str)
            or item["name"] not in expected_nodes
            or not isinstance(item["status"], str)
            or item["status"] not in NODE_STATUS_SCORES
            or evidence is None
        ):
            return None
        nodes.append(NodeAssessment(item["name"], item["status"], evidence))
    if [node.name for node in nodes] != expected_nodes:
        return None

    transitions: list[TransitionAssessment] = []
    expected_edge_tuples = [tuple(edge) for edge in expected_transitions]
    for item in raw_transitions:
        if not isinstance(item, dict) or set(item) != {
            "from",
            "to",
            "status",
            "evidence",
        }:
            return None
        edge = (item["from"], item["to"])
        evidence = _parse_evidence(item["evidence"])
        if (
            not isinstance(item["from"], str)
            or not isinstance(item["to"], str)
            or edge not in expected_edge_tuples
            or not isinstance(item["status"], str)
            or item["status"] not in TRANSITION_STATUS_SCORES
            or evidence is None
        ):
            return None
        transitions.append(TransitionAssessment(edge[0], edge[1], item["status"], evidence))
    if [(edge.source_node, edge.target_node) for edge in transitions] != expected_edge_tuples:
        return None
    return ProcessJudgment(tuple(nodes), tuple(transitions), data["summary"])


def parse_process_judgment(
    completion: Any,
    *,
    expected_nodes: list[str],
    expected_transitions: list[list[str]],
) -> ProcessJudgment | None:
    text = _last_assistant_content(completion).strip()
    if not text:
        return None
    for candidate in _candidate_json_strings(text):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            parsed = _parse_object(
                data,
                expected_nodes=expected_nodes,
                expected_transitions=expected_transitions,
            )
            if parsed is not None:
                return parsed
    return None


def evidence_quotes_match_case(
    judgment: ProcessJudgment,
    original_prompt: list[dict[str, Any]],
    candidate_trace: list[dict[str, Any]],
) -> bool:
    messages = {"prompt": original_prompt, "trace": candidate_trace}
    refs = [ref for node in judgment.nodes for ref in node.evidence] + [
        ref for transition in judgment.transitions for ref in transition.evidence
    ]
    for ref in refs:
        source = messages[ref.source]
        if not 0 <= ref.message_index < len(source):
            return False
        message = source[ref.message_index]
        if not isinstance(message, dict):
            return False
        value = message.get(ref.field)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    if any(value == 0.0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def score_judgment(judgment: ProcessJudgment) -> tuple[float, float, float]:
    node_values = [NODE_STATUS_SCORES[node.status] for node in judgment.nodes]
    transition_values = [TRANSITION_STATUS_SCORES[edge.status] for edge in judgment.transitions]
    node_score = geometric_mean(node_values)
    transition_score = geometric_mean(transition_values)
    process_score = geometric_mean(node_values + transition_values)
    return process_score, node_score, transition_score


ANSWER_TAG_PATTERN = re.compile(
    r"<(?:TARGET|MOA|PATHWAYS|VIABILITY|CELL_CYCLE|STRESS)>.*?"
    r"</(?:TARGET|MOA|PATHWAYS|VIABILITY|CELL_CYCLE|STRESS)>",
    re.IGNORECASE | re.DOTALL,
)


def _has_substantive_assistant_reasoning(
    evidence: tuple[EvidenceRef, ...],
    candidate_trace: list[dict[str, Any]],
) -> bool:
    """Prove that a cited trace location contains more than a bare answer tag."""
    for ref in evidence:
        if ref.source != "trace" or not 0 <= ref.message_index < len(candidate_trace):
            continue
        message = candidate_trace[ref.message_index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        value = message.get(ref.field)
        if not isinstance(value, str):
            continue
        residual = ANSWER_TAG_PATTERN.sub("", value)
        residual = re.sub(r"<[^>]+>", " ", residual)
        if len(re.findall(r"[A-Za-z0-9]+", residual)) >= 4:
            return True
    return False


def score_judgment_with_trace(
    judgment: ProcessJudgment,
    *,
    candidate_trace: list[dict[str, Any]],
    generated_nodes: list[str],
) -> tuple[float, float, float, int]:
    """Apply the bare-answer support invariant before fixed-label aggregation."""
    adjustments = 0
    node_values: list[float] = []
    for node in judgment.nodes:
        value = NODE_STATUS_SCORES[node.status]
        if (
            node.name in generated_nodes
            and node.status in {"supported", "weak"}
            and not _has_substantive_assistant_reasoning(
                node.evidence,
                candidate_trace,
            )
        ):
            value = NODE_STATUS_SCORES["unsupported"]
            adjustments += 1
        node_values.append(value)

    transition_values: list[float] = []
    for edge in judgment.transitions:
        value = TRANSITION_STATUS_SCORES[edge.status]
        if edge.status in {"coherent", "weak"} and not _has_substantive_assistant_reasoning(
            edge.evidence,
            candidate_trace,
        ):
            value = TRANSITION_STATUS_SCORES["disconnected"]
            adjustments += 1
        transition_values.append(value)

    return (
        geometric_mean(node_values + transition_values),
        geometric_mean(node_values),
        geometric_mean(transition_values),
        adjustments,
    )


def evaluate_completion(
    completion: Any,
    *,
    original_prompt: list[dict[str, Any]],
    candidate_trace: list[dict[str, Any]],
    expected_nodes: list[str],
    generated_nodes: list[str],
    expected_transitions: list[list[str]],
) -> ProcessEvaluation:
    """Parse, prove, and score, failing open on every technical failure."""
    judgment = parse_process_judgment(
        completion,
        expected_nodes=expected_nodes,
        expected_transitions=expected_transitions,
    )
    if judgment is None:
        return ProcessEvaluation(1.0, 1.0, 1.0, False, False, 0, None)
    if not evidence_quotes_match_case(judgment, original_prompt, candidate_trace):
        return ProcessEvaluation(1.0, 1.0, 1.0, True, False, 0, judgment)
    process_score, node_score, transition_score, adjustments = score_judgment_with_trace(
        judgment,
        candidate_trace=candidate_trace,
        generated_nodes=generated_nodes,
    )
    return ProcessEvaluation(
        process_score,
        node_score,
        transition_score,
        True,
        True,
        adjustments,
        judgment,
    )

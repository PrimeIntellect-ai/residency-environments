"""Optional process-coherence judging; reference answers never enter the request."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from typing import Any

import verifiers.v1 as vf
from openai import APIConnectionError, APIError, APIStatusError
from pydantic import Field
from verifiers.v1.types import content_text

from .process_contract import (
    NODE_STATUS_SCORES,
    PROCESS_SYSTEM_PROMPT,
    RUBRIC_VERSION,
    TRANSITION_STATUS_SCORES,
    evaluate_completion,
    task_graph,
)


class ProcessJudgeConfig(vf.JudgeConfig):
    model: str = "google/gemini-3-flash-preview"
    sampling: vf.SamplingConfig = Field(default_factory=lambda: vf.SamplingConfig(temperature=0, max_tokens=900))
    timeout_seconds: float = Field(default=180, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)


def _message(message: vf.Message) -> dict[str, Any]:
    record = message.model_dump(mode="json", exclude_none=True)
    record["content"] = content_text(message.content)
    return record


def _case(trace: vf.Trace, entry_point: str) -> dict[str, Any]:
    if trace.num_branches != 1:
        raise ValueError("The process rubric requires one linear solver trajectory, not a multi-branch trace.")
    nodes = trace.branches[0].nodes
    first_sample = next((i for i, node in enumerate(nodes) if node.sampled), len(nodes))
    prompt = [_message(node.message) for node in nodes[:first_sample]]
    completion = [_message(node.message) for node in nodes[first_sample:]]
    case = {
        "entry_point": entry_point,
        "task_graph": task_graph(entry_point),
        "original_prompt": prompt,
        "candidate_trace": completion,
        "evidence_locations": {
            source: [
                {
                    "message_index": index,
                    "role": message["role"],
                    "fields": [
                        field
                        for field in ("content", "reasoning_content")
                        if isinstance(message.get(field), str) and message[field].strip()
                    ],
                }
                for index, message in enumerate(messages)
            ]
            for source, messages in (("prompt", prompt), ("trace", completion))
        },
    }
    case["case_id"] = hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return case


class ProcessJudge(vf.Judge[str, ProcessJudgeConfig]):
    def build_messages(self, *, case: dict[str, Any]) -> vf.Messages:
        return [
            vf.SystemMessage(content=PROCESS_SYSTEM_PROMPT),
            vf.UserMessage(
                content=(
                    "Audit this candidate's reasoning process. Return only the required JSON.\n\n"
                    "<candidate_case>\n" + json.dumps(case, ensure_ascii=False, indent=2) + "\n</candidate_case>"
                )
            ),
        ]


async def judge_process(trace: vf.Trace, entry_point: str, config: ProcessJudgeConfig) -> dict[str, float]:
    case = _case(trace, entry_point)
    judge = ProcessJudge(config)
    event: dict[str, Any] = {"rubric_version": RUBRIC_VERSION, "case_id": case["case_id"], "model": config.model}
    trace.info["process_judge"] = event
    metrics = {
        "process_judge_computed": 1.0,
        "process_judge_parse_success": 0.0,
        "process_judge_reference_valid": 0.0,
        "process_judge_fail_open": 0.0,
        "process_judge_transport_failure": 0.0,
        "process_score": 1.0,
    }
    started = time.monotonic()
    response = None
    for attempt in range(config.max_retries + 1):
        try:
            async with asyncio.timeout(config.timeout_seconds):
                response = await judge.evaluate(trace=trace, case=case)
            break
        except (APIError, TimeoutError) as exc:
            # Judge infrastructure must not turn a correct solver answer into a zero.
            event["error_type"] = type(exc).__name__
            event["error"] = str(exc)
            event["attempts"] = attempt + 1
            retryable = isinstance(exc, (APIConnectionError, TimeoutError)) or (
                isinstance(exc, APIStatusError) and (exc.status_code in (408, 409, 429) or exc.status_code >= 500)
            )
            if not retryable or attempt == config.max_retries:
                metrics["process_judge_transport_failure"] = 1.0
                metrics["process_judge_fail_open"] = 1.0
                metrics["process_judge_latency_seconds"] = time.monotonic() - started
                return metrics
            await asyncio.sleep(0.5 * 2**attempt)
    metrics["process_judge_latency_seconds"] = time.monotonic() - started
    event["attempts"] = attempt + 1
    assert response is not None
    event["raw_response"] = response.text
    graph = case["task_graph"]
    result = evaluate_completion(
        [{"role": "assistant", "content": response.text}],
        original_prompt=case["original_prompt"],
        candidate_trace=case["candidate_trace"],
        expected_nodes=graph["audited_nodes"],
        generated_nodes=graph["generated_nodes"],
        expected_transitions=graph["transitions"],
    )
    event["evaluation"] = asdict(result)
    metrics.update(
        {
            "process_score": result.process_score,
            "process_node_score": result.node_score,
            "process_transition_score": result.transition_score,
            "process_judge_parse_success": float(result.parse_success),
            "process_judge_reference_valid": float(result.proof_valid),
            "process_judge_fail_open": float(not result.proof_valid),
            "process_judge_support_adjustments": float(result.support_adjustments),
        }
    )
    if result.judgment is not None:
        for status in NODE_STATUS_SCORES:
            metrics[f"process_node_{status}_fraction"] = sum(
                node.status == status for node in result.judgment.nodes
            ) / len(result.judgment.nodes)
        for status in TRANSITION_STATUS_SCORES:
            metrics[f"process_transition_{status}_fraction"] = sum(
                edge.status == status for edge in result.judgment.transitions
            ) / len(result.judgment.transitions)
    return metrics

"""Compatibility helpers for running the established CARLA loop from Verifiers v1."""

from __future__ import annotations

import json
from typing import Any

from verifiers.legacy.clients import resolve_client
from verifiers.legacy.types import ClientConfig as LegacyClientConfig
from verifiers.v1 import graph
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.client import ClientConfig, TrainClientConfig
from verifiers.v1.task import Task
from verifiers.v1.trace import (
    AgentInfo,
    Error,
    ModelCall,
    Reward,
    Trace,
    TraceTask,
)
from verifiers.v1.types import (
    AssistantMessage,
    Response,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    TurnTokens,
    Usage,
    UserMessage,
    content_to_parts,
)

_FINISH_REASONS = {"stop", "length", "tool_calls"}


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return "" if content is None else str(content)


def _tool_calls(raw_calls: Any) -> list[ToolCall] | None:
    calls: list[ToolCall] = []
    for raw_call in raw_calls or []:
        call = _plain(raw_call)
        if not isinstance(call, dict):
            continue
        function = call.get("function", call)
        arguments = function.get("arguments") or ""
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append(
            ToolCall(
                id=str(call.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=arguments,
            )
        )
    return calls or None


def _messages(raw_messages: Any) -> list:
    messages = []
    for raw_message in raw_messages or []:
        message = _plain(raw_message)
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            messages.append(SystemMessage(content=content_to_parts(message.get("content"))))
        elif role == "user":
            messages.append(UserMessage(content=content_to_parts(message.get("content"))))
        elif role == "assistant":
            messages.append(
                AssistantMessage(
                    content=message.get("content"),
                    reasoning_content=message.get("reasoning_content"),
                    tool_calls=_tool_calls(message.get("tool_calls")),
                )
            )
        elif role == "tool":
            messages.append(
                ToolMessage(
                    tool_call_id=str(message.get("tool_call_id") or ""),
                    content=content_to_parts(message.get("content")),
                    name=message.get("name"),
                )
            )
    return messages


def _tokens(raw_tokens: Any) -> TurnTokens | None:
    tokens = _plain(raw_tokens)
    if not isinstance(tokens, dict):
        return None
    if not tokens.get("prompt_ids") and not tokens.get("completion_ids"):
        return None
    return TurnTokens(
        prompt_ids=list(tokens.get("prompt_ids") or []),
        completion_ids=list(tokens.get("completion_ids") or []),
        completion_logprobs=list(tokens.get("completion_logprobs") or []),
    )


def _response(raw_response: Any, model: str, tokens: TurnTokens | None) -> Response:
    response = _plain(raw_response)
    response = response if isinstance(response, dict) else {}
    message = _plain(response.get("message"))
    message = message if isinstance(message, dict) else {}
    raw_usage = _plain(response.get("usage"))
    usage = None
    if isinstance(raw_usage, dict) and raw_usage.get("prompt_tokens") is not None:
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
            completion_tokens=int(raw_usage.get("completion_tokens") or 0),
            cached_input_tokens=raw_usage.get("cached_input_tokens"),
            reasoning_tokens=raw_usage.get("reasoning_tokens"),
        )
    finish_reason = message.get("finish_reason") or response.get("finish_reason")
    if finish_reason not in _FINISH_REASONS:
        finish_reason = None
    return Response(
        id=str(response.get("id") or ""),
        created=int(response.get("created") or 0),
        model=str(response.get("model") or model),
        message=AssistantMessage(
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
            tool_calls=_tool_calls(message.get("tool_calls")),
        ),
        finish_reason=finish_reason,
        usage=usage,
        tokens=tokens,
    )


def _tools(raw_tools: Any) -> list[Tool]:
    tools = []
    for raw_tool in raw_tools or []:
        tool = _plain(raw_tool)
        if not isinstance(tool, dict):
            continue
        try:
            tools.append(Tool.model_validate(tool))
        except ValueError:
            continue
    return tools


def legacy_output_to_trace(output: Any, task: Task) -> Trace:
    """Translate a legacy rollout record into the v1 trace stored by the runner."""
    out = _plain(output)
    if not isinstance(out, dict):
        raise TypeError("The CARLA rollout did not return a mapping")

    raw_error = _plain(out.get("error"))
    error = None
    if raw_error:
        if isinstance(raw_error, dict):
            error = Error(
                type=str(raw_error.get("type") or "Error"),
                message=str(raw_error.get("message") or raw_error),
                traceback=raw_error.get("traceback"),
            )
        else:
            error = Error(type="Error", message=str(raw_error))

    stop_condition = out.get("stop_condition")
    if out.get("is_truncated") and stop_condition == "max_turns_reached":
        stop_condition = "max_turns"

    trace = Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        agent=AgentInfo(config=AgentConfig()),
        tools=_tools(out.get("tool_defs")),
        rewards={"reward": Reward(score=float(out.get("reward") or 0.0))},
        metrics=dict(out.get("metrics") or {}),
        info=dict(out.get("info") or {}),
        is_completed=bool(out.get("is_completed", True)),
        ok=error is None,
        stop_condition=stop_condition,
        errors=[error] if error else [],
    )

    model = str(out.get("model") or "")
    for raw_step in out.get("trajectory") or []:
        step = _plain(raw_step)
        if not isinstance(step, dict):
            continue
        response = _response(step.get("response"), model, _tokens(step.get("tokens")))
        node = graph.prepare_turn(trace, _messages(step.get("prompt"))).commit(response)
        trace.calls.append(
            ModelCall(
                node=node,
                model=response.model,
                finish_reason=response.finish_reason,
                usage=response.usage,
            )
        )
    return trace


def legacy_client(config: ClientConfig, model: str):
    """Build the legacy client used by the established CARLA rollout loop."""
    renderer = isinstance(config, TrainClientConfig)
    return resolve_client(
        LegacyClientConfig(
            client_type="renderer" if renderer else "openai_chat_completions",
            renderer_config=config.renderer if renderer else None,
            renderer_model_name=(config.renderer_model_name or model) if renderer else None,
            api_base_url=config.base_url,
            api_key_var=config.api_key_var,
            extra_headers=dict(config.headers or {}),
        )
    )

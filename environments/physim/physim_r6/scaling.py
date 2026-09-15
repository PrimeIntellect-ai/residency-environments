"""Budgeted R6 task hooks for stock Verifiers harnesses; no model/harness loop."""

from __future__ import annotations

import math
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

from physim import taskset as T


class SpendConfig(vf.TaskConfig):
    limit_usd: float = Field(gt=0, le=25)
    input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)
    cache_read_usd_per_mtok: float | None = Field(None, ge=0)
    cache_write_usd_per_mtok: float | None = Field(None, ge=0)
    context_window: int = Field(gt=0)
    max_response_tokens: int = Field(8192, gt=0)

    @property
    def input_ceiling(self):
        return max(self.input_usd_per_mtok, self.cache_read_usd_per_mtok or 0, self.cache_write_usd_per_mtok or 0)

    @property
    def next_call_reserve(self):
        # Full context plus a full output response is deliberately conservative.
        # SDK retries cross the native interception boundary individually and
        # appear as separate trace.calls, including calls with missing usage.
        return (self.context_window * self.input_ceiling + self.max_response_tokens * self.output_usd_per_mtok) / 1e6


class ScalingTaskConfig(T.R6TaskConfig):
    spend: SpendConfig


class ScalingConfig(vf.TasksetConfig):
    task: ScalingTaskConfig
    prompt: str = T.DEFAULT_PROMPT


def spend_accounting(trace, config: SpendConfig) -> dict:
    reported, reserved, missing_cost, missing_usage = 0.0, 0.0, 0, 0
    usages = [call.usage for call in trace.calls] + list(trace.extra_usage)
    for usage in usages:
        if usage is None:
            missing_usage += 1
            missing_cost += 1
            reserved += config.next_call_reserve
        elif usage.cost is not None and math.isfinite(usage.cost) and usage.cost >= 0:
            reported += usage.cost
        else:
            missing_cost += 1
            # Treat cache input at the largest recorded input/write/read price.
            inputs = max(0, usage.prompt_tokens) + max(0, usage.cached_input_tokens or 0)
            outputs = max(0, usage.completion_tokens)
            reserved += (
                (inputs * config.input_ceiling + outputs * config.output_usd_per_mtok) / 1e6
                if inputs + outputs
                else config.next_call_reserve
            )
    return dict(
        reported_cost_usd=reported,
        unreported_cost_reserve_usd=reserved,
        accounted_cost_usd=reported + reserved,
        calls_missing_cost=missing_cost,
        calls_missing_usage=missing_usage,
        next_call_reserve_usd=config.next_call_reserve,
        limit_usd=config.limit_usd,
        model_calls=len(trace.calls),
        price_policy="Reported native per-call cost; missing cost reserved from locked catalog rates. Missing usage reserves a full-context call.",
    )


class ScalingTask(T.R6Task):
    def __init__(self, data, config: ScalingTaskConfig):
        super().__init__(data, config)

    async def setup(self, trace, runtime):
        if trace.agent.config.harness.id != "bash":
            raise vf.TaskError("This spend policy is scoped to the stock sequential Bash harness")
        sampling = trace.agent.config.sampling
        if sampling is None or sampling.max_tokens != self.config.spend.max_response_tokens:
            raise vf.TaskError("Response token cap must match the spend reserve configuration")
        await super().setup(trace, runtime)
        trace.info["r6"]["scaling"] = dict(
            spend_config=self.config.spend.model_dump(),
            task_policy="Same public task and scientific inputs; native dollar stop only.",
        )

    @vf.stop(priority=-1)
    async def dollar_budget(self, trace) -> bool:
        budget = spend_accounting(trace, self.config.spend)
        stop = budget["accounted_cost_usd"] + budget["next_call_reserve_usd"] > budget["limit_usd"] + 1e-12
        trace.info.setdefault("r6", {})["spend"] = dict(budget, stopped=stop)
        if trace.state.output:
            T.E.dump(Path(trace.state.output) / "spend_state.json", dict(budget, stopped=stop))
        return stop

    async def finalize(self, trace, runtime):
        await super().finalize(trace, runtime)
        budget = spend_accounting(trace, self.config.spend)
        trace.info.setdefault("r6", {})["spend_final"] = budget
        if trace.state.output:
            T.E.dump(Path(trace.state.output) / "spend_final.json", budget)


class R6ScalingTaskset(vf.Taskset[ScalingTask, ScalingConfig]):
    DEFAULT_HARNESS = "bash"

    def load(self):
        original = T.R6Taskset(T.R6Config(id="physim_r6", task=self.config.task, prompt=self.config.prompt))
        for task in original.load():
            data = task.data.model_copy(update={"protocol": task.data.protocol + "-dollar-budget-v1"})
            yield ScalingTask(data, self.config.task)

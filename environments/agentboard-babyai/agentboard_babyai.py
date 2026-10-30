from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import verifiers.v1 as vf

DEFAULT_BASE_SEED = 1234
AGENTBOARD_COMMIT = "bb7255e2daf1989069a186dad9e53f70680961db"
ASSET_ROOT = Path(__file__).resolve().with_name("assets")
AGENTBOARD_DATA_PATH = ASSET_ROOT / "test.jsonl"
AGENTBOARD_PROMPT_PATH = ASSET_ROOT / "babyai_vanilla_prompt.json"
LOGGER = logging.getLogger(__name__)


class AgentBoardBabyAITasksetConfig(vf.TasksetConfig):
    split: Literal["all", "train_3_per_subtask", "eval_1_per_subtask"] = "train_3_per_subtask"
    base_seed: int = DEFAULT_BASE_SEED


class BabyAITaskData(vf.TaskData):
    goal: str
    agentboard_commit: str
    game_name: str
    initial_observation: str
    seed: int
    subgoals: list[str]
    difficulty: str


def _load_agentboard_prompt() -> dict[str, Any]:
    if not AGENTBOARD_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Packaged AgentBoard BabyAI prompt not found at {AGENTBOARD_PROMPT_PATH}")
    with AGENTBOARD_PROMPT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _strip_check_actions(text: str) -> str:
    lines = str(text or "").splitlines()
    i = 0
    while i < len(lines):
        if re.match(r"(?i)^\s*action\s*:\s*check (valid|available) actions\s*$", lines[i]):
            start = i
            if start > 0 and re.match(r"(?i)^\s*observation\s*:", lines[start - 1]):
                start -= 1
                if start > 0 and re.match(r"(?i)^\s*action\s*:", lines[start - 1]):
                    start -= 1

            end = i + 1
            while end < len(lines) and not re.match(r"(?i)^\s*action\s*:", lines[end]):
                end += 1
            del lines[start:end]
            i = start
            continue
        if re.search(r"(?i)check (valid|available) actions", lines[i]) and not re.match(
            r"(?i)^\s*(observation|action)\s*:", lines[i]
        ):
            del lines[i]
            continue
        i += 1
    return "\n".join(lines).strip()


def _build_agentboard_vanilla_prompt(
    prompt_config: dict[str, Any],
    goal: str,
    init_obs: str,
) -> tuple[str, str]:
    instruction = _strip_check_actions(str(prompt_config["instruction"]).strip())
    instruction = instruction.replace("pick up <obj> <id>", "pickup <obj> <id>")
    examples = prompt_config["examples"]

    query = instruction
    if examples:
        query += "\nHere are examples:\n"
        for example in examples:
            cleaned_example = _strip_check_actions(str(example).rstrip())
            if cleaned_example:
                query += cleaned_example.replace("pick up ", "pickup ") + "\n"
    query += f"You should perform actions to accomplish the goal: {goal}\n"
    query += (
        "IMPORTANT RESPONSE FORMAT: Reply with exactly one action phrase only. "
        "Do not include 'Action:', 'Observation:', explanations, or multiple lines.\n"
    )
    query += f"Observation: {init_obs}\nAction: "

    system_prompt = str(prompt_config["system_msg"])
    return system_prompt, query


def _split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return rows
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("additional_info", {}).get("subtask", ""))].append(row)
    for subtask_rows in grouped.values():
        subtask_rows.sort(key=lambda row: int(row["id"]))

    selected: list[dict[str, Any]] = []
    for subtask in sorted(grouped):
        subtask_rows = grouped[subtask]
        if split == "train_3_per_subtask":
            selected.extend(subtask_rows[:3])
        elif split == "eval_1_per_subtask":
            selected.extend(subtask_rows[3:4])
        else:
            raise ValueError(
                f"Unknown BabyAI split {split!r}; expected all, train_3_per_subtask, or eval_1_per_subtask."
            )
    return selected


def _load_rows(config: AgentBoardBabyAITasksetConfig) -> list[BabyAITaskData]:
    data_path = AGENTBOARD_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"BabyAI task data not found at {data_path}")
    prompt_config = _load_agentboard_prompt()

    raw_rows: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            raw_rows.append(item)

    original_subtask_counts: dict[str, int] = defaultdict(int)
    for item in raw_rows:
        subtask = str(item.get("additional_info", {}).get("subtask", ""))
        item["_agentboard_subtask_index"] = original_subtask_counts[subtask]
        original_subtask_counts[subtask] += 1

    raw_rows = _split_rows(raw_rows, config.split)

    tasks: list[BabyAITaskData] = []
    for item in raw_rows:
        additional = dict(item.get("additional_info", {}))
        subtask = str(additional.get("subtask", ""))
        within_subtask_index = int(item["_agentboard_subtask_index"])

        init_obs = str(additional.get("init_obs", ""))
        goal = str(item["goal"])
        system_prompt, prompt = _build_agentboard_vanilla_prompt(prompt_config, goal, init_obs)
        tasks.append(
            BabyAITaskData(
                id=f"babyai-{int(item['id'])}",
                name=f"babyai-{int(item['id'])}",
                prompt=[{"role": "user", "content": prompt}],
                system_prompt=system_prompt,
                goal=goal,
                agentboard_commit=AGENTBOARD_COMMIT,
                game_name=subtask,
                initial_observation=init_obs,
                seed=int(config.base_seed) + within_subtask_index,
                subgoals=list(item.get("subgoals", [])),
                difficulty=str(item.get("difficulty", "")),
            )
        )
    if not tasks:
        raise ValueError(f"No BabyAI tasks selected from {data_path}")
    return tasks


def _parse_action(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and isinstance(obj.get("action"), str):
            cleaned = obj["action"]
    except json.JSONDecodeError:
        pass
    match = re.search(r"(?im)^\s*action\s*:\s*(.+)$", cleaned)
    if match:
        cleaned = match.group(1).strip()
    cleaned = cleaned.splitlines()[0] if cleaned else ""
    cleaned = cleaned.split(".", 1)[0]
    cleaned = " ".join(cleaned.split()[:24]).strip().lower()
    cleaned = cleaned.replace("pick up ", "pickup ")
    return cleaned


def _clean_observation(observation: str) -> str:
    cleaned = re.sub(r"\n+", "\n", str(observation or ""))
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(
        r"(?i)\s*please check (?:valid|available) actions\.?",
        "",
        cleaned,
    )
    return cleaned.strip()


def _load_babyai_runtime():
    gymnasium = importlib.import_module("gymnasium")
    minigrid = importlib.import_module("minigrid")
    register_envs = getattr(minigrid, "register_minigrid_envs", None)
    if register_envs is not None and "BabyAI-GoToRedBallGrey-v0" not in gymnasium.registry:
        register_envs()
    return importlib.import_module("agentboard_babyai_runtime").BabyAI


class BabyAIState(vf.State):
    done: bool = False
    won: bool = False
    progress: float = 0.0
    turns: int = 0
    valid_actions: int = 0
    invalid_actions: int = 0


class _BabyAIRollout:
    """One process-local BabyAI simulator for one rollout."""

    def __init__(self, task: BabyAITaskData, state: BabyAIState) -> None:
        self.state = state
        babyai_cls = _load_babyai_runtime()
        self.env = babyai_cls(
            game_name=task.game_name,
            seed=task.seed,
            obs_to_reward=list(task.subgoals),
            difficulty=task.difficulty,
        )

    async def respond(self, message: str) -> vf.Messages:
        if self.state.done:
            return []
        action = _parse_action(message)
        self.state.turns += 1
        if not action:
            self.state.invalid_actions += 1
            observation = "Invalid action: respond with exactly one BabyAI action."
        elif re.search(r"\bcheck\b", action):
            LOGGER.warning("BabyAI valid-actions helper disabled action=%r", action)
            self.state.invalid_actions += 1
            observation = (
                "The valid-actions helper is disabled. Try a direct action such as "
                "turn left, turn right, move forward, go to <object> <id>, pickup <object> <id>, "
                "drop, toggle, or toggle and go through <door> <id>."
            )
        else:
            observation, reward, done, info = await asyncio.to_thread(self.env.step, action)
            observation = _clean_observation(observation)
            if info.get("action_is_valid", False):
                self.state.valid_actions += 1
            else:
                self.state.invalid_actions += 1
            self.state.progress = max(self.state.progress, float(reward))
            self.state.done = bool(done)
            self.state.won = bool(info["is_success"])
        return [{"role": "user", "content": f"Observation: {observation}\nAction:"}]

    def close(self) -> None:
        env = getattr(self, "env", None)
        if env is None:
            return
        gym_env = getattr(env, "env", None)
        close = getattr(gym_env, "close", None)
        if callable(close):
            close()
        self.env = None


class BabyAITask(vf.Task[BabyAITaskData, BabyAIState, vf.TaskConfig]):
    @vf.stop
    async def game_done(self, trace: vf.Trace) -> bool:
        return trace.state.done

    @vf.reward(weight=1.0)
    async def progress_reward(self, trace: vf.Trace) -> float:
        return trace.state.progress

    @vf.metric
    async def success(self, trace: vf.Trace) -> float:
        return float(trace.state.won)

    @vf.metric
    async def grounding(self, trace: vf.Trace) -> float:
        total = trace.state.valid_actions + trace.state.invalid_actions
        return trace.state.valid_actions / total if total else 0.0

    @vf.metric
    async def babyai_turns(self, trace: vf.Trace) -> float:
        return float(trace.state.turns)

    @vf.metric
    async def invalid_actions(self, trace: vf.Trace) -> float:
        return float(trace.state.invalid_actions)


class BabyAIEnvConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})


class BabyAIEnv(vf.Env[BabyAIEnvConfig]):
    async def run(self, task: BabyAITask, agents: vf.Agents) -> None:
        async with agents.agent.interaction(task) as interaction:
            trace = interaction.trace
            rollout = _BabyAIRollout(task.data, trace.state)
            try:
                segment = await interaction.turn()
                while not segment.terminated:
                    messages = await rollout.respond(segment.last_reply)
                    trace.info["last_observation"] = messages[0]["content"]
                    if trace.state.done:
                        # Keep terminal feedback without asking the model to act again.
                        trace.stop("game_done")
                        break
                    segment = await interaction.turn(messages)
            finally:
                trace.info["babyai_state"] = trace.state.model_dump()
                trace.info["babyai_termination_reason"] = rollout.env.infos["termination_reason"]
                rollout.close()


class AgentBoardBabyAITaskset(vf.Taskset[BabyAITask, AgentBoardBabyAITasksetConfig]):
    def load(self) -> list[BabyAITask]:
        return [BabyAITask(data, self.config.task) for data in _load_rows(self.config)]


__all__ = ["AgentBoardBabyAITaskset", "BabyAIEnv"]

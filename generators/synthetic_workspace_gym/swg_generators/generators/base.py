from __future__ import annotations

import inspect
import random
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from swg_generators.generators.common import (
    build_complexity_profile,
    make_env_id,
)
from swg_generators.generators.d5_quality import validate_atomic_oracle
from swg_generators.provenance import (
    ENVIRONMENT_VERSION,
    EVALUATOR_VERSION,
    generation_fingerprint,
)
from swg_generators.schemas import (
    EnvironmentFamily,
    EnvironmentManifest,
    EnvironmentSpec,
    EvaluatorResult,
)
from swg_generators.utils.io import read_json, write_json, write_text
from swg_generators.utils.paths import list_relative_files
from swg_generators.utils.scratch import scratch_directory


@dataclass(slots=True)
class GeneratedEnvironment:
    root: Path
    visible_root: Path
    hidden_root: Path
    manifest: EnvironmentManifest
    validation_result: EvaluatorResult | None = None


@dataclass(slots=True)
class GeneratedPayload:
    instruction: str
    metadata: dict[str, object]
    reference_solution: dict[str, object]
    evaluator_entrypoint: str


class BaseGenerator(ABC):
    family: EnvironmentFamily | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls is BaseGenerator:
            return
        family = getattr(cls, "family", None)
        if family is None:
            raise TypeError(f"{cls.__name__} must define a concrete EnvironmentFamily on the 'family' class attribute.")
        cls.family = EnvironmentFamily(family)

    def sample_spec(self, difficulty: int, seed: int, **overrides: object) -> EnvironmentSpec:
        difficulty = int(difficulty)
        family = self.require_family()
        generation_params = dict(overrides.pop("generation_params", {}))
        scenario_id = overrides.pop("scenario_id", generation_params.pop("scenario_id", None))
        split = overrides.pop("split", generation_params.get("split", None))
        task_id = overrides.pop("task_id", generation_params.get("task_id", None))
        if split is not None:
            generation_params["split"] = str(split)
        if task_id is not None:
            generation_params["task_id"] = str(task_id)
        spec = EnvironmentSpec(
            env_family=family,
            difficulty=difficulty,
            seed=seed,
            scenario_id=str(scenario_id) if scenario_id is not None else None,
            max_steps=int(overrides.pop("max_steps", 12)),
            time_limit_seconds=int(overrides.pop("time_limit_seconds", 60)),
            task_params=dict(overrides.pop("task_params", {})),
            evaluator_params=dict(overrides.pop("evaluator_params", {})),
            generation_params=generation_params,
            complexity_profile=build_complexity_profile(family, difficulty),
        )
        if overrides:
            raise ValueError(f"Unsupported spec overrides: {sorted(overrides.keys())}")
        return spec

    def resolve_scenario_id(
        self,
        *,
        difficulty: int,
        seed: int,
        requested_scenario: str | None = None,
    ) -> str:
        if requested_scenario is not None:
            return str(requested_scenario)
        spec = self.sample_spec(difficulty=difficulty, seed=seed)
        scenario_pool = getattr(self, "scenario_pool", None)
        if scenario_pool is None:
            return "default"
        parameters = list(inspect.signature(scenario_pool).parameters.values())
        if parameters and parameters[0].name in {"rng", "random", "random_state"}:
            pool = scenario_pool(random.Random(seed), spec)
        else:
            pool = scenario_pool(spec)
        scenario = self.select_scenario(spec, pool)
        return str(scenario["scenario_id"])

    def generate_instance(
        self, spec: EnvironmentSpec, output_dir: Path, *, validate: bool = True
    ) -> GeneratedEnvironment:
        env_id = make_env_id(
            spec.env_family,
            spec.difficulty,
            spec.seed,
            spec.task_params,
            scenario_id=spec.scenario_id,
            generation_params=spec.generation_params,
        )
        root = output_dir / env_id
        if root.exists():
            shutil.rmtree(root)
        visible_root = root / "visible"
        hidden_root = root / "hidden"
        visible_root.mkdir(parents=True, exist_ok=True)
        hidden_root.mkdir(parents=True, exist_ok=True)

        payload = self.build_environment(spec, root=root, visible_root=visible_root, hidden_root=hidden_root)
        metadata = dict(payload.metadata)
        metadata["release_provenance"] = {
            "environment_version": ENVIRONMENT_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            "generation_fingerprint": generation_fingerprint(visible_root, hidden_root, payload.evaluator_entrypoint),
            "horizon_unit": "tool_steps",
        }
        split = spec.generation_params.get("split")
        task_id = spec.generation_params.get("task_id")
        if split is not None:
            metadata.update(
                {
                    "split": str(split),
                    "split_family": spec.env_family.value,
                    "split_scenario": spec.scenario_id,
                    "split_difficulty": spec.difficulty,
                    "split_seed": spec.seed,
                }
            )
        if task_id is not None:
            metadata["task_id"] = str(task_id)
        manifest = EnvironmentManifest(
            env_id=env_id,
            family=spec.env_family,
            difficulty=spec.difficulty,
            seed=spec.seed,
            instruction=payload.instruction,
            workspace_root="visible",
            visible_files=list_relative_files(visible_root),
            hidden_root="hidden",
            hidden_files=list_relative_files(hidden_root),
            tool_permissions=spec.tool_permissions,
            max_steps=spec.max_steps,
            time_limit_seconds=spec.time_limit_seconds,
            metadata=metadata,
            evaluator_entrypoint=payload.evaluator_entrypoint,
            reference_solution=payload.reference_solution,
        )
        bundle = GeneratedEnvironment(
            root=root,
            visible_root=visible_root,
            hidden_root=hidden_root,
            manifest=manifest,
        )
        self.calibrate_initial_score(bundle)
        manifest.metadata["release_provenance"]["generation_fingerprint"] = generation_fingerprint(
            visible_root,
            hidden_root,
            payload.evaluator_entrypoint,
        )
        write_json(root / "manifest.json", manifest.to_dict())
        if validate:
            result = self.validate_instance(bundle)
            bundle.validation_result = result
            if not result.success:
                raise ValueError(
                    f"Reference solution failed validation for '{manifest.env_id}': {result.failure_labels}"
                )
        return bundle

    def calibrate_initial_score(self, instance: GeneratedEnvironment) -> None:
        from swg_generators.evaluators.registry import get_evaluator

        config_path = instance.hidden_root / "evaluator_config.json"
        config = read_json(config_path)
        normalization = config.get("score_normalization")
        if normalization is None:
            return
        config.pop("initial_score", None)
        config.pop("score_normalization", None)
        write_json(config_path, config)
        evaluator = get_evaluator(
            instance.manifest.family,
            evaluator_entrypoint=instance.manifest.evaluator_entrypoint,
        )
        result = evaluator.evaluate(instance.visible_root, instance.manifest, instance.hidden_root)
        config["initial_score"] = result.score
        config["score_normalization"] = normalization
        write_json(config_path, config)

    def validate_instance(self, instance: GeneratedEnvironment):
        from swg_generators.evaluators.registry import get_evaluator

        self.validate_reference_solution(instance)
        evaluator = get_evaluator(
            instance.manifest.family,
            evaluator_entrypoint=instance.manifest.evaluator_entrypoint,
        )
        scratch_root = instance.root.parent / ".validation"
        scratch_root.mkdir(parents=True, exist_ok=True)
        with scratch_directory(scratch_root, "swg-validate-") as tmp_dir:
            realization = dict(instance.manifest.metadata.get("difficulty_realization", {}))
            quality_gated = instance.manifest.difficulty == 5 and realization.get("bug_bundle_id") is not None
            untouched_score = None
            if "unmodified_reward_limit" in realization:
                untouched = tmp_dir / "untouched"
                shutil.copytree(instance.visible_root, untouched)
                untouched_result = evaluator.evaluate(untouched, instance.manifest, instance.hidden_root)
                untouched_score = untouched_result.score
                unmodified_limit = float(realization["unmodified_reward_limit"])
                if untouched_result.score > unmodified_limit:
                    raise ValueError(
                        f"Quality gate rejected {instance.manifest.env_id}: unmodified reward "
                        f"{untouched_result.score:.6f} exceeds {unmodified_limit:.2f}"
                    )
            if quality_gated:
                self.validate_d5_structure(realization)
            workspace = tmp_dir / "workspace"
            shutil.copytree(instance.visible_root, workspace)
            for relative_path, content in instance.manifest.reference_solution.get("files", {}).items():
                write_text(workspace / relative_path, str(content))
            result = evaluator.evaluate(workspace, instance.manifest, instance.hidden_root)
            if result.score != 1.0:
                raise ValueError(
                    f"Quality gate rejected {instance.manifest.env_id}: "
                    f"reference solution reward {result.score:.6f} must equal 1.0"
                )
            result.diagnostics["generation_validation"] = {
                "reference_solution_score": result.score,
                "unmodified_reward": untouched_score,
                "unmodified_reward_limit": realization.get("unmodified_reward_limit"),
            }
            return result

    def validate_d5_structure(self, realization: dict[str, object]) -> None:
        profile = str(realization.get("profile") or "")
        oracle_payload = dict(realization.get("oracle_profile", {}))
        if profile:
            reference_reward = float(oracle_payload.get("reference_solution_reward", 1.0))
            unmodified_reward = float(oracle_payload.get("unmodified_reward", 0.0))
            if abs(reference_reward - 1.0) > 1e-9:
                raise ValueError("D5 profiled oracle gate requires reference reward 1.0")
            if unmodified_reward >= reference_reward:
                raise ValueError("D5 profiled oracle gate requires untouched reward below full")
        else:
            oracle = validate_atomic_oracle(oracle_payload)
            if not oracle["valid"]:
                raise ValueError(
                    "D5 atomic oracle gate failed: " + "; ".join(str(item) for item in oracle["violations"])
                )
        capabilities = list(realization.get("capabilities", []))
        minimum_capabilities = {
            "d5_a": 2,
            "d5_b": 3,
            "d5_c": 4,
        }.get(profile, 4)
        minimum_depth = {
            "d5_a": 2,
            "d5_b": 3,
            "d5_c": 4,
        }.get(profile, 3)
        if len(capabilities) < minimum_capabilities:
            raise ValueError(f"D5 quality gate requires at least {minimum_capabilities} capabilities")
        if int(realization.get("touched_file_count", 0)) < 2:
            raise ValueError("D5 quality gate requires at least two touched files")
        if int(realization.get("semantic_dependency_depth", 0)) < minimum_depth:
            raise ValueError(f"D5 quality gate requires semantic dependency depth >= {minimum_depth}")
        if realization.get("composition_mode") == "compositional":
            composition = dict(realization.get("composition_spec", {}))
            if int(composition.get("stage_count", 0)) < 2:
                raise ValueError("D5 composition gate requires at least two stages")
            if not composition.get("downstream_consumes_upstream_artifact"):
                raise ValueError("D5 composition gate requires downstream artifact consumption")

    @abstractmethod
    def build_environment(
        self,
        spec: EnvironmentSpec,
        *,
        root: Path,
        visible_root: Path,
        hidden_root: Path,
    ) -> GeneratedPayload:
        raise NotImplementedError

    def require_family(self) -> EnvironmentFamily:
        if self.family is None:
            raise TypeError(f"{type(self).__name__} must define a concrete EnvironmentFamily before use.")
        return EnvironmentFamily(self.family)

    def select_scenario(self, spec: EnvironmentSpec, scenarios: list[dict[str, object]]) -> dict[str, object]:
        requested = spec.scenario_id
        if requested is None:
            requested = spec.generation_params.get("scenario_id")
        if requested:
            requested = str(requested)
            for scenario in scenarios:
                if str(scenario.get("scenario_id")) == requested:
                    return scenario
            available = ", ".join(sorted(str(scenario.get("scenario_id")) for scenario in scenarios))
            raise ValueError(
                f"Unknown scenario_id '{requested}' for {self.require_family().value}. Available: {available}"
            )
        return scenarios[(spec.seed - 1) % len(scenarios)]

    def validate_reference_solution(self, instance: GeneratedEnvironment) -> None:
        solution_files = instance.manifest.reference_solution.get("files", {})
        if not solution_files:
            raise ValueError("Reference solution metadata must include at least one file artifact.")
        unchanged_paths: list[str] = []
        for relative_path, content in solution_files.items():
            visible_path = instance.visible_root / relative_path
            if visible_path.exists() and visible_path.read_text(encoding="utf-8") == str(content):
                unchanged_paths.append(relative_path)
        if unchanged_paths:
            joined = ", ".join(sorted(unchanged_paths))
            raise ValueError(
                "Reference solution did not modify one or more visible files; generation likely failed to apply a change: "
                f"{joined}"
            )

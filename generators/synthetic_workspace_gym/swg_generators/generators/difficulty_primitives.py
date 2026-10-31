from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DefectBundle:
    """A connected D5 defect chain with explicit capability ownership."""

    bundle_id: str
    defect_ids: tuple[str, ...]
    dependency_edges: tuple[tuple[str, str], ...]
    capability_groups: Mapping[str, tuple[str, ...]]
    required_files: tuple[str, ...]
    semantic_dependency_depth: int | None = None

    def __post_init__(self) -> None:
        if not self.bundle_id.strip():
            raise ValueError("bundle_id must not be empty")
        if len(self.defect_ids) < 2:
            raise ValueError("a defect bundle must contain at least two defects")
        if len(set(self.defect_ids)) != len(self.defect_ids):
            raise ValueError("defect_ids must be unique")
        known = set(self.defect_ids)
        if any(source not in known or target not in known for source, target in self.dependency_edges):
            raise ValueError("dependency_edges must reference defects in the bundle")
        longest_dependency_chain(self.defect_ids, self.dependency_edges)
        grouped_members = [defect_id for defect_ids in self.capability_groups.values() for defect_id in defect_ids]
        if len(grouped_members) != len(set(grouped_members)):
            raise ValueError("a defect may belong to only one capability group")
        grouped = set(grouped_members)
        if grouped != known:
            missing = sorted(known - grouped)
            extra = sorted(grouped - known)
            raise ValueError(f"capability_groups must cover every defect exactly; missing={missing}, extra={extra}")
        if len(set(self.required_files)) != len(self.required_files):
            raise ValueError("required_files must be unique")
        if self.semantic_dependency_depth is not None and self.semantic_dependency_depth < 1:
            raise ValueError("semantic_dependency_depth must be positive")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "DefectBundle":
        defect_ids = payload.get("defect_ids", payload.get("bugs", ()))
        capability_groups = payload.get("capability_groups")
        if capability_groups is None:
            capabilities = tuple(str(item) for item in payload.get("capabilities", ()))
            labels = tuple(str(item) for item in defect_ids)
            capability_groups = {
                capability: (labels[index],) for index, capability in enumerate(capabilities[: len(labels)])
            }
            if len(capabilities) < len(labels):
                capability_groups = {
                    **capability_groups,
                    "integration": labels[len(capabilities) :],
                }
        return cls(
            bundle_id=str(payload["bundle_id"]),
            defect_ids=tuple(str(item) for item in defect_ids),
            dependency_edges=tuple((str(edge[0]), str(edge[1])) for edge in payload.get("dependency_edges", ())),
            capability_groups={
                str(name): tuple(str(item) for item in members) for name, members in dict(capability_groups).items()
            },
            required_files=tuple(str(item) for item in payload.get("required_files", ())),
            semantic_dependency_depth=(
                int(payload["semantic_dependency_depth"])
                if payload.get("semantic_dependency_depth") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "defect_ids": list(self.defect_ids),
            "bugs": list(self.defect_ids),
            "dependency_edges": [list(edge) for edge in self.dependency_edges],
            "capability_groups": {name: list(members) for name, members in self.capability_groups.items()},
            "capabilities": list(self.capability_groups),
            "required_files": list(self.required_files),
            "semantic_dependency_depth": (
                self.semantic_dependency_depth
                if self.semantic_dependency_depth is not None
                else longest_dependency_chain(self.defect_ids, self.dependency_edges)
            ),
        }


@dataclass(frozen=True)
class CompositionStage:
    """One necessary stage in a compositional task."""

    stage_id: str
    required_inputs: tuple[str, ...]
    produced_artifacts: tuple[str, ...]
    capability: str

    def __post_init__(self) -> None:
        if not self.stage_id.strip():
            raise ValueError("stage_id must not be empty")
        if not self.capability.strip():
            raise ValueError("capability must not be empty")
        if not self.produced_artifacts:
            raise ValueError("a composition stage must produce at least one artifact")


@dataclass(frozen=True)
class CompositionSpec:
    """A stage DAG whose downstream work consumes upstream artifacts."""

    stages: tuple[CompositionStage, ...]
    dependencies: tuple[tuple[str, str], ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.stages) < 2:
            raise ValueError("a compositional task must contain at least two stages")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("composition stage ids must be unique")
        known = set(stage_ids)
        if any(source not in known or target not in known for source, target in self.dependencies):
            raise ValueError("composition dependencies must reference known stages")
        if not self.downstream_consumes_upstream_artifact:
            raise ValueError("at least one downstream stage must require an artifact produced upstream")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "CompositionSpec":
        return cls(
            stages=tuple(
                CompositionStage(
                    stage_id=str(stage["stage_id"]),
                    required_inputs=tuple(str(item) for item in stage.get("required_inputs", ())),
                    produced_artifacts=tuple(str(item) for item in stage.get("produced_artifacts", ())),
                    capability=str(stage["capability"]),
                )
                for stage in payload.get("stages", ())
            ),
            dependencies=tuple((str(edge[0]), str(edge[1])) for edge in payload.get("dependencies", ())),
            metadata={
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "stages",
                    "dependencies",
                    "stage_count",
                    "downstream_consumes_upstream_artifact",
                }
            },
        )

    @property
    def downstream_consumes_upstream_artifact(self) -> bool:
        by_id = {stage.stage_id: stage for stage in self.stages}
        return any(
            bool(set(by_id[source].produced_artifacts) & set(by_id[target].required_inputs))
            for source, target in self.dependencies
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "required_inputs": list(stage.required_inputs),
                    "produced_artifacts": list(stage.produced_artifacts),
                    "capability": stage.capability,
                }
                for stage in self.stages
            ],
            "dependencies": [list(edge) for edge in self.dependencies],
            "stage_count": len(self.stages),
            "downstream_consumes_upstream_artifact": (self.downstream_consumes_upstream_artifact),
            **dict(self.metadata),
        }


def coerce_defect_bundle(value: DefectBundle | Mapping[str, object]) -> DefectBundle:
    if isinstance(value, DefectBundle):
        return value
    return DefectBundle.from_mapping(value)


def longest_dependency_chain(
    node_ids: Sequence[str],
    dependency_edges: Sequence[tuple[str, str]],
) -> int:
    """Return the node count in the longest path and reject cyclic bundles."""
    incoming: dict[str, set[str]] = {str(node): set() for node in node_ids}
    outgoing: dict[str, set[str]] = {str(node): set() for node in node_ids}
    for source, target in dependency_edges:
        outgoing[str(source)].add(str(target))
        incoming[str(target)].add(str(source))
    queue = sorted(node for node, parents in incoming.items() if not parents)
    depth = {node: 1 for node in queue}
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for target in sorted(outgoing[node]):
            depth[target] = max(depth.get(target, 1), depth[node] + 1)
            incoming[target].remove(node)
            if not incoming[target]:
                queue.append(target)
    if visited != len(incoming):
        raise ValueError("dependency graph must be acyclic")
    return max(depth.values(), default=0)

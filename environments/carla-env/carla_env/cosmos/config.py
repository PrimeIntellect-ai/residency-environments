"""Cosmos Transfer2.5 configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CosmosConfig:
    """
    Configuration for Cosmos Transfer2.5 sim2real stylization.

    A remote GPU service stylizes CARLA RGB frames before they are shown to
    the model.
    """

    enabled: bool = False
    server_url: str = ""
    prompt: str = (
        "Dashcam view of a realistic city street with natural lighting, photorealistic, high detail"
    )
    control: str = "edge"
    control_weight: float = 0.8
    timeout: float = 30.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.control = str(self.control or "edge").strip().lower()
        if self.control != "edge":
            raise ValueError(
                f"Unsupported Cosmos control mode {self.control!r}. "
                "The bundled Cosmos frame server currently supports only 'edge'."
            )

    @classmethod
    def from_obj(cls, obj: Any) -> "CosmosConfig":
        if obj is None:
            return cls()
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            import dataclasses

            known = {field.name for field in dataclasses.fields(cls)}
            return cls(**{k: v for k, v in obj.items() if k in known})
        raise TypeError(f"Cannot create CosmosConfig from {type(obj).__name__}")

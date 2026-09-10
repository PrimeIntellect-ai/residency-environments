"""Native Verifiers v1 taskset for CARLA driving scenarios."""

from .v1 import CarlaTaskset

__all__ = ["CarlaTaskset"]


def __getattr__(name: str):
    if name == "load_environment":
        from .env import load_environment

        return load_environment
    raise AttributeError(name)

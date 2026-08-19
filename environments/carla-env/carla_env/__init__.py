"""
carla_env: CARLA verifiers environment with navigation, free-roam, and vision support.

Supports CARLA 0.10.0 (UE5) and 0.9.16 (UE4).

Entry point for vf-eval: this module must export `load_environment`.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

from .compat import CarlaVersion
from .cosmos import CosmosConfig
from .nurec import NuRecConfig
from .sandbox import CarlaSandboxConfig
from .v1 import CarlaTaskset, CarlaTasksetConfig, CarlaV1Env, CarlaV1EnvConfig

if TYPE_CHECKING:
    from .env import CarlaEnv, CarlaEnvConfig, load_environment

__all__ = [
    "CarlaEnv",
    "CarlaEnvConfig",
    "CarlaVersion",
    "CarlaSandboxConfig",
    "CosmosConfig",
    "NuRecConfig",
    "CarlaTaskset",
    "CarlaTasksetConfig",
    "CarlaV1Env",
    "CarlaV1EnvConfig",
    "load_environment",
]

_LAZY_PROXY_SIGNATURE = inspect.Signature(
    parameters=[
        inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
        inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
    ]
)


def _load_env_export(name: str) -> Any:
    try:
        from . import env as _env
    except ModuleNotFoundError as exc:
        if exc.name == "carla":
            raise ModuleNotFoundError(
                "The CARLA Python client is not installed. Install one of "
                "`carla-env[carla10]` or `carla-env[carla9]` before loading the environment."
            ) from exc
        raise
    return getattr(_env, name)


class _LazyEnvType(type):
    _LAZY_INTROSPECTION_ATTRS = {
        "__signature__",
        "__wrapped__",
        "__text_signature__",
        "_partialmethod",
    }

    def _target(cls) -> type:
        # Use _lazy_name (set at class creation) instead of __name__,
        # so subclasses resolve the original export, not the subclass name.
        return _load_env_export(cls._lazy_name)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        return cls._target()(*args, **kwargs)

    def __getattr__(cls, name: str) -> Any:
        if name in cls._LAZY_INTROSPECTION_ATTRS or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(cls._target(), name)

    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, cls._target())

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, cls._target())


class CarlaEnvConfig(metaclass=_LazyEnvType):
    """Lazy proxy for :class:`carla_env.env.CarlaEnvConfig`."""

    _lazy_name = "CarlaEnvConfig"
    __signature__ = _LAZY_PROXY_SIGNATURE


class CarlaEnv(metaclass=_LazyEnvType):
    """Lazy proxy for :class:`carla_env.env.CarlaEnv`."""

    _lazy_name = "CarlaEnv"
    __signature__ = _LAZY_PROXY_SIGNATURE


class _LazyLoadEnvironment:
    """Callable wrapper that lazily resolves the real function's signature."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        func: Callable[..., Any] = _load_env_export("load_environment")
        return func(*args, **kwargs)

    @property
    def __signature__(self) -> inspect.Signature:
        try:
            return inspect.signature(_load_env_export("load_environment"))
        except (ImportError, ModuleNotFoundError):
            return _LAZY_PROXY_SIGNATURE

    @property
    def __doc__(self) -> str | None:
        try:
            return _load_env_export("load_environment").__doc__
        except Exception:
            return "Lazy wrapper for carla_env.env.load_environment."


load_environment = _LazyLoadEnvironment()

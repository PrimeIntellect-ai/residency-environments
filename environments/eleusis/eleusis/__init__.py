"""Eleusis environment exports, loaded only when requested."""

from importlib import import_module

__all__ = ["EleusisEnv", "EleusisHarness", "EleusisTaskset"]

_EXPORT_MODULES = {
    "EleusisEnv": ".env",
    "EleusisHarness": ".harness",
    "EleusisTaskset": ".taskset",
}


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value

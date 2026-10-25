__all__ = ["CurationEnv", "PretrainTaskset"]


def __getattr__(name: str):
    # Tool servers run `python -m pretrain_gym.<module>`; an eager import here would load that module twice.
    if name == "CurationEnv":
        from .env import CurationEnv

        return CurationEnv
    if name == "PretrainTaskset":
        from .task import PretrainTaskset

        return PretrainTaskset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

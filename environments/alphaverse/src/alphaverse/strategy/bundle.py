"""The complete allowlisted filesystem shipped to a trading runtime."""

from pathlib import Path


def public_strategy_files() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        "public_types.py",
        "strategy/__init__.py",
        "strategy/protocol.py",
        "strategy/sdk.py",
        "strategy/trading_worker.py",
    )
    files = {f"alphaverse/{path}": (root / path).read_bytes() for path in paths}
    files["alphaverse/__init__.py"] = b'from alphaverse.public_types import Side\n__all__ = ["Side"]\n'
    files["alphaverse/models.py"] = b"from alphaverse.public_types import Side\n"
    return files

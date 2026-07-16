"""Cheap package-contract checks for residency environments."""

import os
import subprocess
from pathlib import Path

import pytest

BUILD_TIMEOUT = 600
ENVIRONMENTS = Path(__file__).parent.parent / "environments"


def environments() -> list[Path]:
    if not ENVIRONMENTS.is_dir():
        return []
    return sorted(path for path in ENVIRONMENTS.iterdir() if path.is_dir())


def changed_environments() -> list[Path]:
    all_environments = environments()
    changed = os.getenv("CHANGED_ENVS")
    if changed == "none":
        return []
    if not changed:
        return all_environments
    names = {name.strip() for name in changed.split(",") if name.strip()}
    return [path for path in all_environments if path.name in names]


def test_environment_packages_have_required_files():
    for env_dir in environments():
        assert (env_dir / "pyproject.toml").is_file(), f"{env_dir.name} has no pyproject.toml"
        assert (env_dir / "README.md").is_file(), f"{env_dir.name} has no README.md"


@pytest.mark.parametrize("env_dir", changed_environments(), ids=lambda path: path.name)
def test_environment_builds(env_dir: Path, tmp_path: Path):
    proc = subprocess.run(
        ["uv", "build", str(env_dir), "--out-dir", str(tmp_path / "dist")],
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT,
    )
    assert proc.returncode == 0, f"Failed to build {env_dir.name}: {(proc.stderr or proc.stdout)[-4000:]}"

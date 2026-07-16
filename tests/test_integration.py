"""Install and smoke-evaluate each changed environment with a real model."""

import os
import subprocess
from pathlib import Path

import pytest

INSTALL_TIMEOUT = 600
EVAL_TIMEOUT = 600
ENVIRONMENTS = Path(__file__).parent.parent / "environments"


def environments() -> list[Path]:
    if not ENVIRONMENTS.is_dir():
        return []
    all_environments = sorted(path for path in ENVIRONMENTS.iterdir() if path.is_dir())
    changed = os.getenv("CHANGED_ENVS")
    if changed == "none":
        return []
    if not changed:
        return all_environments
    names = {name.strip() for name in changed.split(",") if name.strip()}
    return [path for path in all_environments if path.name in names]


@pytest.mark.integration
@pytest.mark.parametrize("env_dir", environments(), ids=lambda path: path.name)
def test_eval(env_dir: Path, tmp_path: Path):
    """Install an environment in isolation and run one capped rollout."""
    if not os.getenv("PRIME_API_KEY"):
        pytest.fail("PRIME_API_KEY is required for integration tests")

    venv = tmp_path / ".venv"
    create = subprocess.run(
        ["uv", "venv", "--clear", str(venv)],
        capture_output=True,
        text=True,
        timeout=INSTALL_TIMEOUT,
    )
    assert create.returncode == 0, f"Failed to create virtual environment: {create.stderr}"

    install = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "bin" / "python"),
            "--prerelease=allow",
            str(env_dir.absolute()),
        ],
        capture_output=True,
        text=True,
        timeout=INSTALL_TIMEOUT,
    )
    assert install.returncode == 0, f"Failed to install {env_dir.name}: {install.stderr}"

    command = [
        str(venv / "bin" / "eval"),
        "--taskset.id",
        env_dir.name,
        "-m",
        "openai/gpt-4.1-mini",
        "--client.base-url",
        "https://api.pinference.ai/api/v1",
        "--client.api-key-var",
        "PRIME_API_KEY",
        "-n",
        "1",
        "-r",
        "2",
        "--max-turns",
        "4",
        "--sampling.max-tokens",
        "512",
        "--rich",
        "false",
    ]
    try:
        evaluation = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=EVAL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {EVAL_TIMEOUT}s evaluating {env_dir.name}")
    assert evaluation.returncode == 0, f"eval {env_dir.name} failed: {(evaluation.stderr or evaluation.stdout)[-4000:]}"

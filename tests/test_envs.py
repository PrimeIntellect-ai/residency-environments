"""Harness tests for classic (v0) verifiers environments in `environments/`.

Each env dir (not ending in `_v1`) is installed into a fresh venv, imported,
loaded via `vf.load_environment`, and — when a model API key is available —
smoke-evaled via `vf-eval`. In CI without secrets (fork PRs), the eval step is
skipped and the install/import/load checks still run.
"""

import json
import os
import shlex
import subprocess
import tomllib
from pathlib import Path

import pytest

# Timeout in seconds for each subprocess step
INSTALL_TIMEOUT = 600  # 10 minutes for venv creation + package install
IMPORT_TIMEOUT = 120  # 2 minutes for importing a package
LOAD_TIMEOUT = 300  # 5 minutes for loading an environment (may download datasets)
EVAL_TIMEOUT = 900  # 15 minutes for a single capped rollout

SMOKE_MAX_TURNS = 5

# Per-env overrides for the args passed to vf.load_environment / vf-eval.
EVAL_ENV_ARG_OVERRIDES: dict[str, dict] = {}

HAS_MODEL_KEY = bool(os.getenv("OPENAI_API_KEY") or os.getenv("PRIME_API_KEY"))


def _load_environment_python_code(env_name: str, env_args: dict) -> str:
    env_args_json = json.dumps(env_args)
    return (
        "import json; "
        "import verifiers as vf; "
        f"env_args = json.loads({json.dumps(env_args_json)}); "
        f"env = vf.load_environment({env_name!r}, **env_args)"
    )


def get_environments() -> list[Path]:
    """Get all v0 env dirs in `environments/`, or only changed ones if CHANGED_ENVS is set."""
    environments = Path(__file__).parent.parent / "environments"
    if not environments.exists():
        return []
    all_envs = [env for env in environments.iterdir() if env.is_dir()]

    # v1 tasksets (the `*_v1` packages) run under the verifiers v1 runtime and are
    # covered by tests/test_envs_v1.py instead.
    all_envs = [env for env in all_envs if not env.name.endswith("_v1")]

    # Filter environments if CHANGED_ENVS is set (for PRs)
    changed_envs = os.getenv("CHANGED_ENVS")
    if changed_envs == "none":
        return []
    if changed_envs:
        changed_list = [e.strip() for e in changed_envs.split(",") if e.strip()]
        if changed_list:
            all_envs = [env for env in all_envs if env.name in changed_list]

    return all_envs


@pytest.mark.parametrize("env_dir", get_environments(), ids=lambda x: x.name)
def test_pyproject_exists(env_dir: Path):
    assert (env_dir / "pyproject.toml").exists(), "pyproject.toml does not exist"


@pytest.mark.parametrize("env_dir", get_environments(), ids=lambda x: x.name)
def test_pyproject_has_metadata(env_dir: Path):
    with open(env_dir / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    assert "name" in pyproject["project"], "pyproject.toml does not have a name"
    assert "version" in pyproject["project"], "pyproject.toml does not have a version"
    assert "description" in pyproject["project"], "pyproject.toml does not have a description"
    assert pyproject["project"]["description"] != "Your environment description here", (
        "Still uses placeholder description"
    )
    keywords = pyproject["project"].get("keywords", pyproject["project"].get("tags"))
    assert keywords is not None, "pyproject.toml does not have keywords"
    assert keywords != ["placeholder-tag", "train", "eval"], "Still uses placeholder keywords"


@pytest.mark.parametrize("env_dir", get_environments(), ids=lambda x: x.name)
def test_readme_exists(env_dir: Path):
    assert (env_dir / "README.md").exists(), "README.md does not exist"


@pytest.mark.parametrize("env_dir", get_environments(), ids=lambda x: x.name)
def test_env(env_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    """Install the env in a fresh venv, then import, load, and (with a key) smoke-eval it."""
    tmp_venv_dir = tmp_path_factory.mktemp(f"venv_{env_dir.name}")
    cmd = (
        f"cd {tmp_venv_dir} && uv venv --clear && source .venv/bin/activate && "
        f"uv pip install {env_dir.absolute().as_posix()}"
    )
    try:
        process = subprocess.run(
            cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=INSTALL_TIMEOUT,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {INSTALL_TIMEOUT}s installing {env_dir.name}")
    assert process.returncode == 0, f"Failed to create virtual environment: {process.stderr}"

    help_test_can_import_env(tmp_venv_dir, env_dir)
    help_test_can_load_env(tmp_venv_dir, env_dir)
    if HAS_MODEL_KEY:
        help_test_can_eval_env(tmp_venv_dir, env_dir)
    else:
        # Fork PRs run without secrets; the eval smoke test runs in the
        # maintainer-gated integration-tests workflow instead.
        print(f"No model API key configured - skipping vf-eval smoke test for {env_dir.name}")


def help_test_can_import_env(tmp_venv_dir: Path, env_dir: Path):
    python_code = f"import {env_dir.name} as env_module"
    import_cmd = f"cd {tmp_venv_dir} && source .venv/bin/activate && uv run python -c {shlex.quote(python_code)}"
    try:
        process = subprocess.run(
            import_cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=IMPORT_TIMEOUT,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {IMPORT_TIMEOUT}s importing {env_dir.name}")
    assert process.returncode == 0, f"Failed to import environment: {process.stderr}"


def help_test_can_load_env(tmp_venv_dir: Path, env_dir: Path):
    env_args = EVAL_ENV_ARG_OVERRIDES.get(env_dir.name, {})
    python_code = _load_environment_python_code(env_dir.name, env_args)
    load_cmd = f"cd {tmp_venv_dir} && source .venv/bin/activate && uv run python -c {shlex.quote(python_code)}"
    try:
        process = subprocess.run(
            load_cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=LOAD_TIMEOUT,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {LOAD_TIMEOUT}s loading {env_dir.name}")
    assert process.returncode == 0, f"Failed to load environment: {process.stderr}"


def help_test_can_eval_env(tmp_venv_dir: Path, env_dir: Path):
    env_args = {**EVAL_ENV_ARG_OVERRIDES.get(env_dir.name, {}), "max_turns": SMOKE_MAX_TURNS}
    eval_cmd = (
        f"cd {tmp_venv_dir} && source .venv/bin/activate && "
        f"uv run vf-eval {env_dir.name} -n 1 -r 1 -d -v -t 512 -a '{json.dumps(env_args)}'"
    )
    try:
        process = subprocess.run(
            eval_cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=EVAL_TIMEOUT,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {EVAL_TIMEOUT}s evaluating {env_dir.name}")
    assert process.returncode == 0, f"Failed to evaluate environment: {(process.stderr or process.stdout)[-2000:]}"

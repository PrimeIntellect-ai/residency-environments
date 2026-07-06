"""Smoke-eval every v1 taskset in `environments/` through the `eval` CLI.

The v0 counterpart (`tests/test_envs.py`) covers v0 envs (`vf.load_environment` + `vf-eval`);
here each `*_v1` taskset is installed into a fresh venv (its `verifiers` dep provides the
`eval` CLI) and run for one short, capped rollout via `eval --taskset.id`. A broken taskset
fails CI. The eval step needs a model API key, so in CI without secrets (fork PRs) only the
install is checked and the rollout runs in the maintainer-gated integration-tests workflow.

Tasksets that can't smoke-eval in plain CI (e.g. needing docker/prime runtimes, gated
datasets, or judge keys) are listed in SKIP_EVAL.
"""

import os
import subprocess
from pathlib import Path

import pytest

INSTALL_TIMEOUT = 600  # 10 minutes for venv creation + verifiers/env install
EVAL_TIMEOUT = 600  # 10 minutes for a capped eval (-n 1 -r 2)

ENVIRONMENTS = Path(__file__).parent.parent / "environments"

# v1 tasksets that can't run a plain-CI smoke eval (document the reason when adding one).
SKIP_EVAL: set[str] = set()


def v1_tasksets() -> list[Path]:
    if not ENVIRONMENTS.exists():
        return []
    all_envs = sorted(p for p in ENVIRONMENTS.iterdir() if p.is_dir() and p.name.endswith("_v1"))

    changed_envs = os.getenv("CHANGED_ENVS")
    if changed_envs == "none":
        return []
    if changed_envs:
        changed_list = [e.strip() for e in changed_envs.split(",") if e.strip()]
        if changed_list:
            all_envs = [env for env in all_envs if env.name in changed_list]

    return all_envs


@pytest.mark.parametrize("env_dir", v1_tasksets(), ids=lambda p: p.name)
def test_eval(env_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    """Install `env_dir` in a fresh venv and run one capped rollout via the `eval` CLI."""
    name = env_dir.name
    if name in SKIP_EVAL:
        pytest.skip(f"{name} can't run a plain-CI smoke eval")

    tmp_venv_dir = tmp_path_factory.mktemp(f"venv_{name}")
    # --prerelease=allow: v1 tasksets pin a verifiers 0.1.15.devN pre-release (for the v1
    # `eval` CLI), and that build in turn needs pre-release transitive deps — uv only
    # enables pre-releases for transitive deps with this flag.
    install_cmd = (
        f"cd {tmp_venv_dir} && uv venv --clear && source .venv/bin/activate && "
        f"uv pip install --prerelease=allow {env_dir.absolute().as_posix()}"
    )
    try:
        proc = subprocess.run(
            install_cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=INSTALL_TIMEOUT,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {INSTALL_TIMEOUT}s installing {name}")
    assert proc.returncode == 0, f"Failed to install {name}: {proc.stderr}"

    if os.getenv("PRIME_API_KEY"):
        model = "-m openai/gpt-4.1-mini --client.base-url https://api.pinference.ai/api/v1 --client.api-key-var PRIME_API_KEY"
    elif os.getenv("OPENAI_API_KEY"):
        model = "-m gpt-4.1-mini --client.base-url https://api.openai.com/v1 --client.api-key-var OPENAI_API_KEY"
    else:
        # Fork PRs run without secrets; the rollout runs in the maintainer-gated
        # integration-tests workflow instead.
        pytest.skip("no model API key configured - install checked, eval covered by integration tests")

    # -r 2: a taskset with @group_reward(s) needs >=2 rollouts to compare.
    eval_cmd = (
        f"cd {tmp_venv_dir} && source .venv/bin/activate && "
        f"uv run eval --taskset.id {name} {model} "
        "-n 1 -r 2 --max-turns 4 --sampling.max-tokens 512 --rich false"
    )
    try:
        proc = subprocess.run(
            eval_cmd, shell=True, executable="/bin/bash",
            capture_output=True, text=True, timeout=EVAL_TIMEOUT,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        pytest.fail(f"Timed out after {EVAL_TIMEOUT}s evaluating {name}")
    assert proc.returncode == 0, f"eval {name} failed: {(proc.stderr or proc.stdout)[-2000:]}"

"""Run and verify scripted controls before a visual-model rehearsal.

Run inside the flight_env venv on the simulator host. Each policy has its own
output directory; old outputs are never selected or resumed implicitly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

POLICIES = ("honest", "blind_shortcut", "broken")
VARIANT_COUNTS = {
    "land_or_hold": 2,
    "find_pad": 2,
    "obstacle_bypass": 2,
    "alternate_pad": 3,
    "patrol_inspection": 2,
    "go_around": 2,
    "compound": 4,
}


def family_policies(family: str) -> tuple[str, ...]:
    return (*POLICIES, "always_b") if family == "alternate_pad" else POLICIES


def balanced_count(family: str, requested: int = 8) -> int:
    """Round an omitted/default count up to whole counterfactual scene groups."""
    if family not in VARIANT_COUNTS or requested <= 0:
        raise ValueError("supported family and positive count required")
    size = VARIANT_COUNTS[family]
    return ((requested + size - 1) // size) * size


SCRIPTS = Path(__file__).resolve().parent


def eval_command(
    policy: str,
    output: Path,
    count: int,
    seed: int,
    port: int,
    image: str,
    family: str = "land_or_hold",
    split: str = "development",
) -> list[str]:
    return [
        str(Path(sys.executable).with_name("eval")),
        "aerialsim-visual",
        "-n",
        str(count),
        "-r",
        "1",
        "-c",
        "1",
        "--no-push",
        "--output-dir",
        str(output),
        "--run.dir",
        "eval",
        "--env.runtime",
        "docker",
        "--env.image",
        image,
        "--env.taskset.count",
        str(count),
        "--env.taskset.start-seed",
        str(seed),
        "--env.taskset.family",
        family,
        "--env.taskset.split",
        split,
        "--env.pilot.harness.id",
        "aerialsim-rehearsal",
        "--env.pilot.harness.endpoint",
        f"http://127.0.0.1:{port}",
        "--env.pilot.harness.policy",
        policy,
        "-m",
        "scripted",
        "--client.base-url",
        f"http://127.0.0.1:{port}/v1",
        "--client.api-key-var",
        "DUMMY_KEY",
    ]


def wait_ready(server: subprocess.Popen, log: Path, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"scripted pilot exited before ready; see {log}")
        if "PILOT_READY" in log.read_text():
            return
        time.sleep(0.1)
    raise RuntimeError(f"scripted pilot did not become ready; see {log}")


def run_policy(
    policy: str,
    output: Path,
    count: int,
    seed: int,
    port: int,
    image: str,
    family: str = "land_or_hold",
    split: str = "development",
) -> None:
    target = output / f"outputs_{policy}"
    # Reserve the directory before launching anything. Even a failed attempt is
    # evidence and must not be silently overwritten by a later run.
    target.mkdir()
    log = output / f"pilot_{policy}.log"
    env = os.environ | {"DUMMY_KEY": "scripted"}
    with log.open("w") as pilot_log:
        server = subprocess.Popen(
            [sys.executable, str(SCRIPTS / "scripted_pilot.py"), "--policy", policy, "--port", str(port)],
            stdout=pilot_log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            wait_ready(server, log)
            print(f"Running scripted control: {policy}", flush=True)
            with (output / f"eval_{policy}.log").open("w") as eval_log:
                result = subprocess.run(
                    eval_command(policy, target, count, seed, port, image, family, split),
                    stdout=eval_log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            if result.returncode:
                raise RuntimeError(f"{policy} eval exited {result.returncode}; see eval_{policy}.log")
            evidence = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "evidence.py"),
                    f"{policy}={target}",
                    "--policy",
                    policy,
                    "--check-baseline",
                    "--expected-count",
                    str(count),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            report = evidence.stdout + evidence.stderr
            (output / f"evidence_{policy}.txt").write_text(report)
            print(report, end="", flush=True)
            if evidence.returncode:
                raise RuntimeError(f"{policy} did not meet its per-flight baseline contract")
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


def run_suite(
    output: Path, count: int, seed: int, port: int, image: str, family: str = "land_or_hold", split: str = "development"
) -> None:
    if family not in VARIANT_COUNTS:
        raise ValueError("unsupported family")
    size = VARIANT_COUNTS[family]
    if count < size or count % size:
        detail = "even" if size == 2 else f"a multiple of {size}"
        raise ValueError(f"count must be positive and {detail}, with at least one complete counterfactual group")
    policies = family_policies(family)
    # Validate the requested seed range before launching any server/simulator.
    from aerialsim_flight.visual import VisualConfig

    VisualConfig(count=count, start_seed=seed, family=family, split=split)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"outputs_{policy}").exists() for policy in policies):
        raise ValueError("scripted outputs already exist; use a fresh rehearsal output directory")
    for policy in policies:
        run_policy(policy, output, count, seed, port, image, family, split)
    print("SCRIPTED_BASELINES_PASSED", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, help="whole scene groups; default is 8 rounded up for the family")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--image", default="aerialsim-city-park:local")
    parser.add_argument("--family", choices=tuple(VARIANT_COUNTS), default="land_or_hold")
    parser.add_argument("--split", choices=("development", "evaluation"), default="development")
    args = parser.parse_args()
    try:
        run_suite(
            args.output_dir.resolve(),
            args.count if args.count is not None else balanced_count(args.family),
            args.start_seed,
            args.port,
            args.image,
            args.family,
            args.split,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"SCRIPTED_BASELINES_FAILED: {exc}\n")


if __name__ == "__main__":
    main()

"""Install a checksummed public wheel and verify published worlds outside the checkout.

Requires uv and Python 3.12. Does not call a model. The supplied config directory
must contain p4g2_044.toml, bf_trail_lab.toml, and xv_rotor_lab.toml.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(command, directory, env):
    result = subprocess.run(command, cwd=directory, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({command[0]}):\n{result.stderr[-6000:]}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-url", required=True)
    parser.add_argument("--configs", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"https://[^\s]+\.whl#sha256=[0-9a-f]{64}", args.package_url):
        parser.error("package-url must be an HTTPS wheel URL with a SHA-256 fragment")
    directory = args.workdir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(s in k.upper() for s in ("TOKEN", "API_KEY", "SECRET"))
        and k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    env.update(
        UV_CACHE_DIR=str(directory / "uv-cache"),
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required")
    run([uv, "venv", "--python", sys.executable, str(directory / "venv")], directory, env)
    python = str(directory / "venv/bin/python")
    run([uv, "pip", "install", "--python", python, f"physim[hub,reference] @ {args.package_url}"], directory, env)
    print("Installed public Physim wheel and its public Blobkit dependency", flush=True)
    for name in ("p4g2_044", "bf_trail_lab", "xv_rotor_lab"):
        shutil.copyfile(args.configs / f"{name}.toml", directory / f"{name}.toml")
    check = directory / "verify.py"
    check.write_text("""import importlib.metadata as metadata
import json
import sys
import tomllib
from pathlib import Path
import blobkit
import physim
from physim.evaluation import reference_demo
from physim.blobround6_explore import ExperimentService
from physim.taskset import R6Config, required_bundle
from verifiers.v1.utils.loaders import load_taskset

root = Path.cwd()
for module in (blobkit, physim):
    assert Path(module.__file__).is_relative_to(root / "venv"), module.__file__
installed = {}
for name in ("blobkit", "physim", "verifiers", "numpy", "scipy"):
    dist = metadata.distribution(name)
    installed[name] = dict(version=dist.version, direct_url=json.loads(dist.read_text("direct_url.json") or "null"))
    if name in ("blobkit", "physim"):
        assert installed[name]["direct_url"]["url"].startswith("https://github.com/swpo/physim/releases/download/")
        assert "dir_info" not in installed[name]["direct_url"]
worlds = []
for name in ("p4g2_044", "bf_trail_lab", "xv_rotor_lab"):
    data = tomllib.loads((root / f"{name}.toml").read_text())
    config = R6Config(**data["env"]["taskset"])
    config.task.tools.bundle_source.cache = root / "cache"
    task = next(iter(load_taskset(config)))
    bundle = required_bundle(config.task.tools)
    simulation = required_bundle(config.task.tools, profile="simulation")
    config.task.tools.bundle_source.offline = True
    assert required_bundle(config.task.tools).root == bundle.root
    assert required_bundle(config.task.tools, profile="simulation").root == simulation.root
    # Serialize and load again, as the trusted tool subprocess does.
    restored = R6Config.model_validate_json(config.model_dump_json())
    assert len(list(load_taskset(restored))) == 1
    demo = reference_demo(bundle, root / "demos" / name)
    assert demo["ok"]
    service = ExperimentService(bundle.make_oracle(), roster=bundle.roster, limits=bundle.limits,
                                max_experiments=1, max_total_tu=1)
    service.experiment([], [dict(sensor="global", t=[0, 0.02])])
    row = dict(name=name, references=bundle.references(), public_ports=bundle.roster.n_ports,
               reference_demo=demo, native_experiment=service.usage(), offline=True)
    worlds.append(row)
    print(json.dumps(dict(world=name, ok=True, reference_score=demo["actual"])), flush=True)
report = dict(ok=True, installed=installed, worlds=worlds, anonymous_data_download=True,
              isolated_python=True, no_model_calls=True)
(root / "verification.json").write_text(json.dumps(report, indent=2) + "\\n")
""")
    output = run([python, "-I", str(check)], directory, env)
    print(output, end="")
    report = json.loads((directory / "verification.json").read_text())
    report.update(workdir=str(directory), package_url=args.package_url)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

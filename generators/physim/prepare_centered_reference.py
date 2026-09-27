"""Prepare new p4g2_044 science from an immutable reference bundle, never its truths."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
from eval_preparation import observe, with_centered_programs, write_json
from physim import blobround6 as R6
from physim.bundles import Bundle, digest, implementation_identity


def prepare(source, output, workers=2):
    bundle = Bundle(source)
    root = Path(output)
    prep = root / "preparation"
    prep.mkdir(parents=True, exist_ok=False)
    for name in ("world.json", "preparation.npz"):
        (prep / name).write_bytes(bundle.verified_path(name).read_bytes())
    import json

    apparatus = json.loads(bundle.verified_path("apparatus.json").read_text())
    apparatus.pop("emitter_yx", None)
    apparatus["protocol"] = R6.APPARATUS_PROTOCOL
    write_json(prep / "apparatus.json", apparatus)
    write_json(
        prep / "origin.json",
        dict(
            world="p4g2_044",
            original_time=1700,
            source=str(Path(source).resolve()),
            previous_bundle=bundle.references(),
            recipe_sha256=digest(__file__),
            implementation=implementation_identity(),
            noise_policy=R6.NOISE_POLICY,
            noise_coefficient=0.002,
            selection="Preserved reference fields and probes; new centered source protocol.",
        ),
    )
    programs, groups = [], {}
    for record in bundle.suite["cases"]:
        program = deepcopy(record["request"])
        program["id"] = record["family"] + "_" + program["id"]
        for action in program["actions"]:
            if action["kind"] == "inject":
                action["device"] = 0
        programs.append(program)
        groups[program["id"]] = deepcopy(record["groups"])
    # Use the original first case's queries and score groups for paired new programs.
    port = apparatus["port_permutation"].index(0)
    programs = with_centered_programs(programs, port)
    for program in programs:
        groups.setdefault(program["id"], deepcopy(groups[programs[0]["id"]]))
    write_json(root / "programs.json", programs)
    write_json(root / "score_groups.json", groups)
    with np.load(prep / "preparation.npz", allow_pickle=False) as data:
        assert data["fields"].shape[0] == 12
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(observe, prep, record, root / "science" / record["id"], 3, 61000 + i)
            for i, record in enumerate(programs)
        ]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    prepare(args.source, args.output, args.workers)

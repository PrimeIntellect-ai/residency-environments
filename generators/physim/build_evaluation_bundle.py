"""Freeze demonstrated programs, generate new native truth, and validate a bundle.

The input is a completed preparation/science directory from eval_preparation.py.
Private labels and scoring selectors are never passed to an evaluated predictor.
"""

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from eval_preparation import make_oracle, write_json
from physim import blobround6 as R6
from physim import blobround6_eval as scoring
from physim.bundles import FORMAT, NUMERICS, Bundle, digest, identified, implementation_identity, load_arrays


def score_groups(case, permutation):
    times = case["queries"][0]["t"]
    na = 1 if len(permutation) == 4 else 2
    groups = []
    # Fixed native-unit scales selected from development evidence, before truth.
    for name, query, field, selected_times, slots, scale in (
        ("activator_history", 0, 0, [5, 10, 20, 30, 40, 50], range(13), 1.0),
        ("local_feedback", 0, na, [5, 10, 20, 30, 50], range(13), 0.5),
        (
            "slow_memory" if na == 1 else "partner_history",
            0,
            3 if na == 1 else 1,
            [5, 10, 20, 30, 50],
            range(13),
            0.05 if na == 1 else 1.0,
        ),
        ("wide_spatial_response", 1, 0, [10, 30, 50], range(19), 1.0),
    ):
        selectors = [
            dict(query=query, time_index=times.index(t), port=permutation.index(field), slot=slot)
            for t in selected_times
            for slot in slots
        ]
        groups.append(
            dict(
                id=name,
                selectors=selectors,
                scales=[scale] * len(selectors),
                unit=f"Native observable units; frozen coordinate scale {scale:g}",
            )
        )
    return groups


def generate_case(preparation, record, output, index):
    output = Path(output)
    case = record["request"]
    oracle = make_oracle(preparation)
    truth_path = output / record["truth"]
    if truth_path.exists():
        arrays = load_arrays(truth_path, text_keys=("request",))
        assert json.loads(str(arrays.pop("request"))) == case
        truth = dict(samples=[arrays[f"query{i}"] for i in range(len(case["queries"]))])
    else:
        truth = oracle.sample_truth(case["actions"], case["queries"], n_samples=2, truth_seed=71000 + index)
        np.savez_compressed(
            truth_path, request=json.dumps(case), **{f"query{i}": a for i, a in enumerate(truth["samples"])}
        )
    controls = {}
    control_dir = output.parent / "controls" / case["id"]
    control_dir.mkdir(parents=True, exist_ok=True)
    for name in ("independent_native", "ignore_actions", "remove_feedback", "wrong_probe_geometry"):
        saved = control_dir / (name + ".npz")
        if saved.exists():
            arrays = load_arrays(saved)
            controls[name] = dict(samples=[arrays[f"query{i}"] for i in range(len(case["queries"]))])
            continue
        native = make_oracle(preparation)
        actions = case["actions"]
        if name == "ignore_actions":
            actions = []
        elif name == "remove_feedback":
            if len(native._perm) == 4:
                native._template["bilin"] = []
            else:
                # XV's cross-drive source rows; original preparation retained.
                for key in ("Wf", "Wid"):
                    matrix = native._template[key].copy()
                    matrix[0, 1] = matrix[1, 0] = 0
                    native._template[key] = matrix
        elif name == "wrong_probe_geometry":
            for device in native._devices:
                device.center = (device.center + [0, 3]) % 128
        controls[name] = native.sample_truth(actions, case["queries"], n_samples=4, truth_seed=81000 + index)
        np.savez_compressed(saved, **{f"query{i}": a for i, a in enumerate(controls[name]["samples"])})
    sensors = ["device0", "device1", "global"]
    initial = oracle.sample_truth([], [dict(sensor=s, t=[0]) for s in sensors], n_samples=1, truth_seed=0)
    roster = scoring.PublicRoster(n_ports=len(oracle._perm))
    model = scoring.PersistencePredictor(dict(zip(sensors, [a[0, 0] for a in initial["samples"]])), roster=roster)
    controls["initial_persistence"] = model.predict(case["actions"], case["queries"], n_samples=4)
    results = {}
    for name, pred in controls.items():
        np.savez_compressed(control_dir / (name + ".npz"), **{f"query{i}": a for i, a in enumerate(pred["samples"])})
        result = scoring.score_case(case, pred, truth, groups=record["groups"], roster=roster, n_samples=4, n_truth=2)
        results[name] = result
    write_json(control_dir / "scores.json", results)
    print(f"{case['id']}: truth + independent forecast controls complete", flush=True)
    return results


def build(source, output, workers=3, resume=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (output / "manifest.json").exists():
        raise ValueError("Completed bundles are immutable; choose a new output directory")
    output.mkdir(parents=True, exist_ok=resume)
    (output / "truth").mkdir(exist_ok=resume)
    prep = source / "preparation"
    for name in ("world.json", "apparatus.json", "preparation.npz"):
        if (output / name).exists():
            assert digest(prep / name) == digest(output / name), "Preparation changed during resume"
        else:
            shutil.copyfile(prep / name, output / name)
    origin = json.loads((prep / "origin.json").read_text())
    apparatus = json.loads((prep / "apparatus.json").read_text())
    programs = json.loads((source / "programs.json").read_text())
    cases = []
    for index, program in enumerate(programs):
        # Every admitted action program has independent-noise development evidence.
        receipt = json.loads((source / "science" / program["id"] / "receipt.json").read_text())
        assert receipt["request"] == program
        assert digest(source / "science" / program["id"] / "observations.npz") == receipt["observations_sha256"]
        case = dict(program, id=f"c{index + 1:03d}")
        cases.append(
            dict(
                request=case,
                family=program["id"],
                groups=score_groups(case, apparatus["port_permutation"]),
                truth=f"truth/{case['id']}.npz",
            )
        )
    suite = dict(
        schema_version="physim-suite-v1",
        scoring_version=scoring.VERSION,
        contract_version=f"r6-absolute-time-{len(apparatus['port_permutation'])}-port-v1",
        forecast_members=64,
        truth_members=2,
        predictor_seed=20260913,
        aggregation="equal-case-mean",
        cases=cases,
        known_limits=[
            "One disclosed development preparation; not a held-out generalization benchmark.",
            "Two grading truths per case; three independent development repeats.",
            "Native controls are privileged diagnostics, not learned agents.",
        ],
    )
    if (output / "suite.json").exists():
        assert json.loads((output / "suite.json").read_text()) == suite, "Suite changed during resume"
    else:
        write_json(output / "suite.json", suite)  # Frozen before any new truth.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(generate_case, prep, record, output, index) for index, record in enumerate(cases)]
        results = [f.result() for f in futures]
    aggregate = {
        name: float(np.mean([r[name]["joint_energy_equal_group_mean"] for r in results])) for name in results[0]
    }
    write_json(
        source / "control_summary.json",
        dict(
            aggregate=aggregate,
            cases=[dict(id=c["request"]["id"], family=c["family"], scores=r) for c, r in zip(cases, results)],
            truth_seeds=[71000 + i for i in range(len(cases))],
            forecast_seeds=[81000 + i for i in range(len(cases))],
            source_sha256=digest(__file__),
            scope="All native forecast controls use independent future noise from grading truth.",
        ),
    )
    write_json(
        output / "checks.json",
        dict(
            schema_version="physim-reference-checks-v1",
            source_origin=origin,
            reference=dict(
                predictor="initial_persistence",
                forecast_members=4,
                primary_joint_energy=aggregate["initial_persistence"],
                absolute_tolerance=1e-10,
            ),
            control_scores=aggregate,
            recipe_sha256=digest(__file__),
        ),
    )
    with np.load(prep / "preparation.npz", allow_pickle=False) as data:
        field_hash = hashlib.sha256(data["fields"].tobytes()).hexdigest()
    objects = {}
    objects["world"] = identified(
        "world",
        dict(
            name=origin["world"],
            genome_sha256=digest(output / "world.json"),
            numerics=NUMERICS,
            implementation=implementation_identity(),
        ),
    )
    objects["preparation"] = identified(
        "preparation",
        dict(
            world_id=objects["world"]["id"],
            field_sha256=field_hash,
            file_sha256=digest(output / "preparation.npz"),
            apparatus_sha256=digest(output / "apparatus.json"),
            noise_policy=R6.NOISE_POLICY,
            noise_coefficient=0.002,
            original_time=origin["original_time"],
            public_time=0.0,
        ),
    )
    objects["suite"] = identified(
        "suite",
        dict(
            preparation_id=objects["preparation"]["id"],
            file_sha256=digest(output / "suite.json"),
            scoring_source_sha256=digest(scoring.__file__),
            truth_sha256={c["truth"]: digest(output / c["truth"]) for c in cases},
        ),
    )
    files = []
    for p in sorted(output.rglob("*")):
        if p.is_file():
            relative = p.relative_to(output).as_posix()
            role = "truth" if relative.startswith("truth/") else p.stem
            profiles = ["evaluation"] if role in ("truth", "suite") else ["simulation", "evaluation"]
            files.append(dict(path=relative, bytes=p.stat().st_size, sha256=digest(p), role=role, profiles=profiles))
    manifest = dict(
        schema_version=FORMAT,
        bundle_id=identified("bundle", {k: v["id"] for k, v in objects.items()})["id"],
        status="local-development-verified",
        licenses=dict(code="Apache-2.0", data="CC-BY-4.0"),
        objects=objects,
        files=files,
    )
    write_json(output / "manifest.json", manifest)
    bundle = Bundle(output)
    bundle.check_runtime()
    for case in bundle.suite["cases"]:
        bundle.truth(case)
    print(json.dumps(dict(bundle_id=manifest["bundle_id"], aggregate=aggregate), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--resume", action="store_true", help="Resume an incomplete bundle with unchanged preparation and suite"
    )
    args = parser.parse_args()
    build(args.source, args.output, args.workers, args.resume)

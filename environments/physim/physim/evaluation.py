"""Public interface validation and bundle-driven evaluation.

Submitted code runs only in Docker. Forecasts are frozen before truth arrays load.
"""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import blobround6_eval as E
from .bundles import Bundle, BundleError, identified, load_arrays
from .bundles import digest as file_digest
from .sandbox import IMAGE, Sandbox, SandboxError
from .sandbox import docker as docker

LIMITS = replace(E.DEFAULT_LIMITS, max_horizon_tu=50.0)

SUBMISSION_GATE_VERSION = "r6-gate-public-validation-v2-roster"


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_prediction(sandbox, actions, queries, *, members=2, seed=11, roster=E.DEFAULT_ROSTER):
    case = dict(id="interface-check", actions=actions, queries=queries)
    shapes = E.validate_case(case, roster=roster, limits=LIMITS, allow_empty=True)
    data, execution = sandbox.prediction(actions, queries, n_samples=members, seed=seed, n_ports=roster.n_ports)
    value = json.loads(data, parse_constant=E._reject_constant, object_pairs_hook=E._unique_object)
    arrays = E.validate_samples(value, shapes, members)
    return dict(samples=arrays), execution


def public_validation_cases():
    """Public API examples only; no case compiler, simulator or grading data."""
    mixed = [dict(sensor="device0", t=[0, 0.02]), dict(sensor="device1", t=[0]), dict(sensor="global", t=[0])]
    return [
        dict(name="mixed_sensors", actions=[], queries=deepcopy(mixed), n_samples=2, seed=11),
        dict(name="repeat_seed", actions=[], queries=deepcopy(mixed), n_samples=2, seed=11),
        dict(name="reordered_queries", actions=[], queries=list(reversed(deepcopy(mixed))), n_samples=2, seed=11),
        dict(
            name="empty_times",
            actions=[],
            queries=[dict(sensor="global", t=[]), dict(sensor="device1", t=[])],
            n_samples=2,
            seed=11,
        ),
        dict(name="empty_queries", actions=[], queries=[], n_samples=1, seed=11),
        dict(
            name="single_member_actions",
            actions=[
                dict(t=0, kind="inject", port=0, amp=1.0, dur=0.02),
                dict(t=0.02, kind="adjust", device=0, u=[0.1, 0.0, 0.0]),
            ],
            queries=[dict(sensor="device1", t=[0, 0.02, 5.02]), dict(sensor="global", t=[0.02])],
            n_samples=1,
            seed=17,
        ),
        dict(
            name="default_members_repeated_sensor",
            actions=[],
            queries=[
                dict(sensor="global", t=[0]),
                dict(sensor="device0", t=[0, 0.02]),
                dict(sensor="device0", t=[0.04]),
            ],
            n_samples=64,
            seed=23,
        ),
    ]


def validate_predictor(artifact, observations, *, roster=E.DEFAULT_ROSTER):
    """Run the public v5 gate; model code executes only inside Sandbox."""
    cases = public_validation_cases()
    report = dict(
        ok=False,
        gate=SUBMISSION_GATE_VERSION,
        checks=[],
        checks_total=len(cases),
        experiments_used=0,
        integrated_tu=0,
        scope="Public execution/interface checks only; no physical accuracy evaluation.",
        public_roster=dict(n_ports=roster.n_ports, device_slots=list(roster.device_slots)),
    )
    box = Sandbox(observations, artifact=artifact)
    baseline = None
    try:
        for case in cases:
            request = {k: v for k, v in case.items() if k != "name"}
            record = dict(
                name=case["name"],
                request=request,
                expected_shapes=[
                    [case["n_samples"], len(q["t"]), roster.n_ports, roster.slots(q["sensor"])] for q in case["queries"]
                ],
            )
            try:
                prediction, execution = read_prediction(
                    box, case["actions"], case["queries"], members=case["n_samples"], seed=case["seed"], roster=roster
                )
                arrays = prediction["samples"]
                if case["name"] == "mixed_sensors":
                    baseline = arrays
                elif case["name"] == "repeat_seed" and not all(np.array_equal(a, b) for a, b in zip(baseline, arrays)):
                    raise E.EvaluationError("repeated prediction calls with the same seed must reproduce output")
                record.update(
                    ok=True, actual_shapes=[list(a.shape) for a in arrays], wall_seconds=execution.get("wall_seconds")
                )
            except (SandboxError, E.EvaluationError, ValueError, TypeError) as exc:
                record.update(ok=False, error_type=type(exc).__name__, error=str(exc)[-3000:])
                report["checks"].append(record)
                report["failure"] = record
                break
            report["checks"].append(record)
        else:
            report["ok"] = True
    finally:
        box.close()
    report["checks_passed"] = sum(check["ok"] for check in report["checks"])
    return report


def runtime_identity():
    import platform
    from importlib.metadata import version

    from blobkit import genome
    from blobkit.soup import sim_cpu

    from . import __version__, blobround6, devices

    return dict(
        physim=__version__,
        blobkit=version("blobkit"),
        numpy=np.__version__,
        scipy=version("scipy"),
        python=platform.python_version(),
        platform=platform.platform(),
        source_sha256={
            module.__name__: file_digest(module.__file__) for module in (sim_cpu, genome, devices, blobround6, E)
        },
    )


def score_frozen(bundle, output, saved, failures, *, members, predictor):
    """Read only frozen forecast files and verified data-only native truths."""
    output = Path(output)
    results = []
    cases = {record["request"]["id"]: record for record in bundle.suite["cases"]}
    for row in saved:
        cid = row["case_id"]
        record = cases[cid]
        case = record["request"]
        path = output / "grading_predictions" / row["file"]
        if E.case_digest(case) != row["request_digest"] or file_digest(path) != row["sha256"]:
            raise BundleError("frozen forecast identity changed")
        arrays = load_arrays(path)
        pred = dict(samples=[arrays[f"query{i}"] for i in range(len(case["queries"]))])
        truth = bundle.truth(record)
        score = E.score_case(
            case,
            pred,
            truth,
            groups=record["groups"],
            n_samples=members,
            n_truth=2,
            roster=bundle.roster,
            limits=bundle.limits,
        )
        marginals = [np.asarray(q["crps_by_port_slot_mean_over_times"]) for q in score["marginal_crps"]]
        weights = [a.size * len(q["t"]) for a, q in zip(marginals, case["queries"])]
        results.append(
            dict(
                case_id=cid,
                family=record["family"],
                joint_energy=score["joint_energy_equal_group_mean"],
                marginal_crps=float(np.average([a.mean() for a in marginals], weights=weights)),
                score=score,
            )
        )
    complete = len(results) == len(cases) and not failures
    report = dict(
        status="COMPLETE" if complete else "INVALID_PREDICTIONS",
        valid_cases=len(results),
        total_cases=len(cases),
        primary_joint_energy=float(np.mean([r["joint_energy"] for r in results])) if complete else None,
        marginal_crps=float(np.mean([r["marginal_crps"] for r in results])) if complete else None,
        failures=failures,
        results=results,
        forecast_members=members,
        truth_members=2,
        truth_policy="Retained independent native realizations; no truth/predictor seed pairing.",
        references=bundle.references(),
        predictor=predictor,
        runtime=runtime_identity(),
        forecast_manifest_sha256=file_digest(output / "grading_predictions/manifest.json"),
        by_family={
            family: float(np.mean([r["joint_energy"] for r in results if r["family"] == family]))
            for family in sorted({r["family"] for r in results})
        },
    )
    report = identified("run", report)
    dump(output / "grade.json", report)
    return report


def _forecast_suite(bundle, output, predict, *, members, predictor):
    if bundle.suite is None:
        raise BundleError("grading requires an evaluation-profile bundle")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    forecast_dir = output / "grading_predictions"
    forecast_dir.mkdir(exist_ok=False)
    saved, failures = [], []
    for record in bundle.suite["cases"]:
        case = record["request"]
        try:
            E.validate_plan(
                case, groups=record["groups"], n_samples=members, n_truth=2, roster=bundle.roster, limits=bundle.limits
            )
            seed = bundle.suite["predictor_seed"]
            pred, execution = predict(case, members, seed)
            # validate_plan returns the shapes used by the immutable scorer.
            arrays = E.validate_samples(
                pred, E.validate_case(case, roster=bundle.roster, limits=bundle.limits), members
            )
            path = forecast_dir / (case["id"] + ".npz")
            np.savez_compressed(path, **{f"query{i}": a for i, a in enumerate(arrays)})
            saved.append(
                dict(
                    case_id=case["id"],
                    file=path.name,
                    sha256=file_digest(path),
                    request_digest=E.case_digest(case),
                    predictor_seed=seed,
                    execution=execution,
                )
            )
        except Exception as exc:
            failures.append(dict(case_id=case["id"], error=str(exc)[:3000]))
    dump(
        forecast_dir / "manifest.json",
        dict(
            finalized_utc=datetime.now(timezone.utc).isoformat(),
            members=members,
            completed=saved,
            failures=failures,
            grader_truth_loaded=False,
            references=bundle.references(),
            predictor=predictor,
        ),
    )
    return score_frozen(bundle, output, saved, failures, members=members, predictor=predictor)


def _snapshot_inputs(source, target, *, observations=False):
    """Freeze bounded public input files once for the entire evaluation."""
    source, target = Path(source).resolve(), Path(target)
    if not source.is_dir():
        raise SandboxError("public input directory is missing")
    target.mkdir()
    rows, total = [], 0
    for p in sorted(source.rglob("*")):
        if p.is_symlink():
            raise SandboxError("public artifact symlinks are forbidden")
        if not p.is_file() or (observations and (p.parent != source or p.suffix != ".npz")):
            continue
        if len(rows) >= (100 if observations else 2048):
            raise SandboxError("public input file count exceeds cap")
        relative = p.relative_to(source)
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with p.open("rb") as src, dest.open("xb") as dst:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                size += len(block)
                total += len(block)
                if size > 20 * 1024 * 1024 or total > (2000 if observations else 64) * 1024 * 1024:
                    raise SandboxError("public input byte cap exceeded")
                dst.write(block)
        rows.append(dict(path=relative.as_posix(), sha256=file_digest(dest)))
    return rows


def grade(artifact, observations, output, *, bundle, members=64, run_context=None):
    """Freeze public inputs, then grade submitted code exclusively in Docker."""
    import tempfile

    if not isinstance(bundle, Bundle):
        bundle = Bundle(bundle)
    with tempfile.TemporaryDirectory(prefix="physim-frozen-public-") as directory:
        root = Path(directory)
        predictor = dict(
            kind="submitted-artifact",
            image=IMAGE,
            run_context=run_context,
            files=_snapshot_inputs(artifact, root / "artifact"),
            observations=_snapshot_inputs(observations, root / "observations", observations=True),
        )

        def predict(case, count, seed):
            box = Sandbox(root / "observations", artifact=root / "artifact")
            try:
                return read_prediction(
                    box, case["actions"], case["queries"], members=count, seed=seed, roster=bundle.roster
                )
            finally:
                box.close()

        return _forecast_suite(bundle, output, predict, members=members, predictor=predictor)


def reference_demo(bundle, output):
    """Trusted packaged persistence control, using only the prepared t=0 readouts."""
    if not isinstance(bundle, Bundle):
        bundle = Bundle(bundle)
    sensors = ["device0", "device1", "global"]
    queries = [dict(sensor=s, t=[0]) for s in sensors]
    initial = bundle.make_oracle().sample_truth([], queries, n_samples=1, truth_seed=0)["samples"]
    model = E.PersistencePredictor(dict(zip(sensors, [a[0, 0] for a in initial])), roster=bundle.roster)
    checks, previous = [], None
    for case in public_validation_cases():
        pred = model.predict(case["actions"], case["queries"], n_samples=case["n_samples"], seed=case["seed"])
        shapes = E.validate_case(
            dict(id="interface-check", actions=case["actions"], queries=case["queries"]),
            limits=bundle.limits,
            roster=bundle.roster,
            allow_empty=True,
        )
        arrays = E.validate_samples(pred, shapes, case["n_samples"])
        if case["name"] == "mixed_sensors":
            previous = arrays
        if case["name"] == "repeat_seed" and not all(np.array_equal(a, b) for a, b in zip(previous, arrays)):
            raise E.EvaluationError("reference predictor is not reproducible")
        checks.append(dict(name=case["name"], ok=True))

    def predict(case, count, seed):
        return model.predict(case["actions"], case["queries"], n_samples=count, seed=seed), dict(
            kind="trusted-packaged-control"
        )

    report = _forecast_suite(
        bundle, output, predict, members=4, predictor=dict(kind="trusted-packaged-control", name="initial_persistence")
    )
    from .bundles import read_json

    reference = read_json(bundle.verified_path("checks.json"))["reference"]
    matches = (
        report["primary_joint_energy"] is not None
        and abs(report["primary_joint_energy"] - reference["primary_joint_energy"]) <= reference["absolute_tolerance"]
    )
    check = dict(
        ok=matches,
        public_checks=checks,
        expected=reference,
        actual=report["primary_joint_energy"],
        references=bundle.references(),
    )
    dump(Path(output) / "reference_check.json", check)
    if not matches:
        raise E.EvaluationError("reference score did not reproduce within its declared tolerance")
    return dict(check, grade=str(Path(output).resolve() / "grade.json"), run_id=report["id"])

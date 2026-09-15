"""Exercise a prepared bundle through the public service and reference grader."""

import argparse
import json
from pathlib import Path

from physim import evaluation
from physim.blobround6_eval import EvaluationError
from physim.blobround6_explore import ExperimentService
from physim.bundles import Bundle, digest


def validate(source):
    source = Path(source)
    bundle = Bundle(source / "bundle")
    bundle.check_runtime()
    roster = bundle.roster
    service = ExperimentService(bundle.make_oracle(), roster=roster, max_experiments=2, max_total_tu=1)
    queries = [dict(sensor=sensor, t=[0, 0.02]) for sensor in ("device0", "device1", "global")]
    action = dict(t=0, kind="inject", port=roster.n_ports - 1, amp=0.1, dur=0.02)
    observed = service.experiment([action], queries)
    shapes = [list(array.shape) for array in observed["samples"]]
    if shapes != [[1, 2, roster.n_ports, slots] for slots in (13, 19, 2)]:
        raise ValueError("Native service produced the wrong public roster")
    usage = service.usage().copy()
    try:
        service.experiment([dict(action, port=roster.n_ports)], queries)
    except EvaluationError:
        pass
    else:
        raise ValueError("Out-of-roster source action was accepted")
    if service.usage() != usage:
        raise ValueError("An invalid action consumed experiment budget")
    for case in bundle.suite["cases"]:
        bundle.truth(case)
    reference_path = source / "reference-demo/reference_check.json"
    reference = (
        json.loads(reference_path.read_text())
        if reference_path.exists()
        else evaluation.reference_demo(bundle, reference_path.parent)
    )
    expected = json.loads(bundle.verified_path("checks.json").read_text())["reference"]
    if (
        not reference["ok"]
        or reference["references"] != bundle.references()
        or abs(reference["actual"] - expected["primary_joint_energy"]) > expected["absolute_tolerance"]
    ):
        raise ValueError("Frozen reference score did not reproduce")
    result = dict(
        ready=True,
        references=bundle.references(),
        public_ports=roster.n_ports,
        native_sensor_shapes=shapes,
        invalid_port_rejected_without_charge=True,
        cases=len(bundle.suite["cases"]),
        truth_members=bundle.suite["truth_members"],
        reference_score=reference["actual"],
        source_sha256=digest(__file__),
    )
    (source / "native_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    print(json.dumps(validate(parser.parse_args().source), indent=2))

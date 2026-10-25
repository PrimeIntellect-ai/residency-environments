"""Bounded offline R6 sample scoring and explicitly trusted local adapters.

This module NEVER executes a submitted file. JSON bundles are data only. The
in-process ``evaluate_trusted_predictor`` helper is for reviewed local code;
neither its resource checks nor Python's process isolation are a code sandbox.
Truth samples are independent realizations, not matched to prediction members.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import secrets
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from . import blobround6 as R6

VERSION = "r6-offline-score-v1"
BUNDLE_VERSION = "r6-prediction-bundle-v2"


class EvaluationError(ValueError):
    """Invalid bounded evaluation request or malformed sample payload."""


@dataclass(frozen=True)
class PublicRoster:
    n_ports: int = 12
    device_slots: tuple[int, ...] = (13, 19)

    def __post_init__(self):
        if (type(self.n_ports) is not int or not 1 <= self.n_ports <= 128
                or type(self.device_slots) is not tuple
                or not 1 <= len(self.device_slots) <= 16
                or any(type(k) is not int or not 1 <= k <= 256
                       for k in self.device_slots)):
            raise EvaluationError("invalid public sensor roster")

    def slots(self, sensor):
        roster = {f"device{i}": k for i, k in enumerate(self.device_slots)}
        roster["global"] = 2
        if type(sensor) is not str or sensor not in roster:
            raise EvaluationError("sensor is not in the public roster")
        return roster[sensor]


@dataclass(frozen=True)
class EvaluationLimits:
    """Operational caps for this local worked example, not security isolation."""
    max_actions: int = 128
    max_queries: int = 32
    max_times_per_query: int = 256
    max_total_times: int = 1024
    max_horizon_tu: float = 250.0
    max_prediction_members: int = 256
    max_truth_members: int = 64
    max_output_values: int = 2_000_000  # prediction and truth combined
    max_groups: int = 32
    max_group_dimensions: int = 256
    max_pairwise_work: int = 100_000_000  # scalar differences, both score families
    max_bundle_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "max_horizon_tu":
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise EvaluationError("invalid horizon cap")
            elif type(value) is not int or value < 1:
                raise EvaluationError("resource caps must be positive integers")


DEFAULT_ROSTER = PublicRoster()
DEFAULT_LIMITS = EvaluationLimits()


def _int(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise EvaluationError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _finite_number(value, name):
    if type(value) not in (int, float):
        raise EvaluationError(f"{name} must be a finite JSON number")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise EvaluationError(f"{name} must be finite") from None
    if not math.isfinite(number):
        raise EvaluationError(f"{name} must be finite")
    return number


def _keys(obj, required, name):
    if type(obj) is not dict or set(obj) != set(required):
        raise EvaluationError(f"{name} requires exactly {sorted(required)}")


def validate_case(case, *, roster=DEFAULT_ROSTER, limits=DEFAULT_LIMITS, allow_empty=False):
    """Validate grammar and coarse caps before parsing/allocating/physics."""
    _keys(case, {"id", "actions", "queries"}, "case")
    if type(case["id"]) is not str or not 1 <= len(case["id"]) <= 128:
        raise EvaluationError("case.id must be a nonempty string of at most 128 characters")
    actions, queries = case["actions"], case["queries"]
    if type(actions) is not list or len(actions) > limits.max_actions:
        raise EvaluationError("actions must be a list within the action cap")
    if type(allow_empty) is not bool:
        raise EvaluationError("allow_empty must be boolean")
    if type(queries) is not list or not (0 if allow_empty else 1) <= len(queries) <= limits.max_queries:
        raise EvaluationError("query list is empty or exceeds the query cap")
    total_times = 0
    for query in queries:
        _keys(query, {"sensor", "t"}, "query")
        times = query["t"]
        if type(times) is not list or len(times) > limits.max_times_per_query:
            raise EvaluationError("query times must be a list within the per-query cap")
        total_times += len(times)
        if total_times > limits.max_total_times:
            raise EvaluationError("total query times exceed the cap")
        for t in times:
            if not 0 <= _finite_number(t, "query time") <= limits.max_horizon_tu:
                raise EvaluationError("query time exceeds the evaluation horizon")
    if not total_times and not allow_empty:
        raise EvaluationError("scoring requires at least one observation")
    for action in actions:
        if type(action) is not dict:
            raise EvaluationError("action must be an object")
        start = _finite_number(action.get("t"), "action time")
        duration = (_finite_number(action.get("dur"), "injection duration")
                    if action.get("kind") == "inject" else R6.ADJUST_TU)
        if start < 0 or start + duration > limits.max_horizon_tu:
            raise EvaluationError("action interval exceeds the evaluation horizon")
    devices = [SimpleNamespace(k=k) for k in roster.device_slots]
    try:
        _, parsed_queries, _, _ = R6._parse(actions, queries, 1, 0, roster.n_ports, devices)
    except R6.ProtocolError as exc:
        raise EvaluationError(str(exc)) from None
    # Public requests use chronological, unique times per query. Compare native
    # ticks so distinct float spellings of the same instant cannot slip through.
    # The private oracle retains its general query support for old diagnostics.
    for query in parsed_queries:
        if any(a >= b for a, b in zip(query.ticks, query.ticks[1:])):
            raise EvaluationError("times within each query must be strictly increasing; duplicates are unsupported")
    return tuple((len(q["t"]), roster.n_ports, roster.slots(q["sensor"]))
                 for q in queries)


def _validate_groups(groups, shapes, limits):
    if type(groups) is not list or not 1 <= len(groups) <= limits.max_groups:
        raise EvaluationError("provide a nonempty predeclared score-group list within the cap")
    identifiers = set()
    dimensions = 0
    for group in groups:
        _keys(group, {"id", "selectors", "scales", "unit"}, "score group")
        identifier = group["id"]
        if type(identifier) is not str or not 1 <= len(identifier) <= 128 or identifier in identifiers:
            raise EvaluationError("score group IDs must be nonempty and unique")
        identifiers.add(identifier)
        if type(group["unit"]) is not str or not 1 <= len(group["unit"]) <= 128:
            raise EvaluationError("score group requires an original-coordinate unit label")
        selectors, scales = group["selectors"], group["scales"]
        if type(selectors) is not list or not 1 <= len(selectors) <= limits.max_group_dimensions:
            raise EvaluationError("score group dimension exceeds cap or is empty")
        if type(scales) is not list or len(scales) != len(selectors):
            raise EvaluationError("provide one fixed positive scale per selected coordinate")
        seen = set()
        for selector, scale in zip(selectors, scales):
            _keys(selector, {"query", "time_index", "port", "slot"}, "selector")
            qi = _int(selector["query"], "selector.query", 0, len(shapes) - 1)
            ti = _int(selector["time_index"], "selector.time_index", 0, shapes[qi][0] - 1)
            pi = _int(selector["port"], "selector.port", 0, shapes[qi][1] - 1)
            si = _int(selector["slot"], "selector.slot", 0, shapes[qi][2] - 1)
            if (qi, ti, pi, si) in seen:
                raise EvaluationError("duplicate coordinate within a score group")
            seen.add((qi, ti, pi, si))
            if _finite_number(scale, "scale") <= 0:
                raise EvaluationError("scales must be strictly positive")
        dimensions += len(selectors)
    return dimensions


def validate_plan(case, *, groups, n_samples=64, n_truth=4,
                  roster=DEFAULT_ROSTER, limits=DEFAULT_LIMITS):
    """Check all count, horizon, output and pairwise-work caps before calls."""
    n_samples = _int(n_samples, "n_samples", 1, limits.max_prediction_members)
    n_truth = _int(n_truth, "n_truth", 1, limits.max_truth_members)
    shapes = validate_case(case, roster=roster, limits=limits)
    dimensions = _validate_groups(groups, shapes, limits)
    coordinates = sum(math.prod(shape) for shape in shapes)
    values = coordinates * (n_samples + n_truth)
    if values > limits.max_output_values:
        raise EvaluationError("prediction plus truth output exceeds the scalar-value cap")
    # Marginal cross-term uses M*N scalar differences; marginal self-term sorts.
    # Energy uses both M*N*D and M*M*D; count all predeclared group dimensions.
    work = coordinates * n_samples * n_truth + dimensions * n_samples * (n_samples + n_truth)
    if work > limits.max_pairwise_work:
        raise EvaluationError("score pairwise work exceeds the scalar-difference cap")
    return {"shapes": shapes, "coordinates": coordinates,
            "output_values": values, "pairwise_work": work}


def _json_array_shape(value, shape, label):
    """Bounded nested-list shape check before NumPy may allocate/coerce it."""
    if not shape:
        _finite_number(value, label)
        return
    if type(value) is not list or len(value) != shape[0]:
        raise EvaluationError(f"{label} has an incorrect array shape")
    for child in value:
        _json_array_shape(child, shape[1:], label)


def validate_samples(payload, shapes, n_members, *, label="prediction"):
    """Strict finite numeric payload; no broadcasting, clipping or coercive repair."""
    _keys(payload, {"samples"}, label)
    arrays = payload["samples"]
    if type(arrays) is not list or len(arrays) != len(shapes):
        raise EvaluationError(f"{label}.samples must have one array per query")
    result = []
    for qi, (value, shape) in enumerate(zip(arrays, shapes)):
        expected = (n_members,) + tuple(shape)
        name = f"{label}.samples[{qi}]"
        if type(value) is np.ndarray:
            if value.shape != expected or value.dtype.kind not in "iuf":
                raise EvaluationError(f"{name} requires numeric shape {expected}")
        elif type(value) is list:
            _json_array_shape(value, expected, name)
        else:
            raise EvaluationError(f"{name} must be a NumPy array or JSON nested list")
        try:
            array = np.asarray(value, dtype=np.float64).reshape(expected)
        except (ValueError, TypeError, OverflowError):
            raise EvaluationError(f"{name} cannot be represented as finite float64") from None
        if not np.isfinite(array).all():
            raise EvaluationError(f"{name} contains non-finite values")
        result.append(array)
    return result


def _marginal_crps(pred, truth):
    """Empirical-distribution CRPS, averaged over independent truth observations."""
    m, n = len(pred), len(truth)
    x, y = pred.reshape(m, -1), truth.reshape(n, -1)
    cross = np.zeros(x.shape[1], dtype=np.float64)
    # One observation at a time avoids M*N*coordinates temporary storage.
    for observation in y:
        cross += np.abs(x - observation).mean(axis=0) / n
    ordered = np.sort(x, axis=0)
    weights = (2 * np.arange(m, dtype=float) - m + 1)[:, None]
    # Exactly half the mean distance over all ordered pairs, including i=j.
    half_self = (weights * ordered).sum(axis=0) / (m * m)
    return (cross - half_self).reshape(pred.shape[1:])


def _mean_rms_distance(x, y):
    total = 0.0
    for row in x:
        delta = y - row
        # hypot.reduce is stable for finite extreme values until the true norm
        # exceeds float64; the final finite-score guard gives a clear failure.
        total += float(np.hypot.reduce(delta, axis=1).mean()) / math.sqrt(x.shape[1]) / len(x)
    return total


def _energy_score(pred, truth):
    return _mean_rms_distance(pred, truth) - 0.5 * _mean_rms_distance(pred, pred)


def score_case(case, prediction, truth, *, groups, n_samples=64, n_truth=4,
               roster=DEFAULT_ROSTER, limits=DEFAULT_LIMITS):
    """Score empirical distributions; prediction/truth member counts may differ.

    CRPS is in each observable's native units. Energy uses coordinate / scale
    and an RMS Euclidean norm, so every group score is dimensionless. Group
    average weights the predeclared groups equally; it is not a 0--1 score.
    """
    plan = validate_plan(case, groups=groups, n_samples=n_samples, n_truth=n_truth,
                         roster=roster, limits=limits)
    pred = validate_samples(prediction, plan["shapes"], n_samples)
    observed = validate_samples(truth, plan["shapes"], n_truth, label="truth")
    marginal = []
    joints = []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for qi, (query, x, y) in enumerate(zip(case["queries"], pred, observed)):
            values = _marginal_crps(x, y)
            if not len(query["t"]):
                reduced = None
            else:
                reduced = values.mean(axis=0).tolist()
            marginal.append({"query": qi, "sensor": query["sensor"],
                             "n_times": len(query["t"]),
                             "crps_by_port_slot_mean_over_times": reduced,
                             "slot_units": (["field_value", "field_value_squared"]
                                            if query["sensor"] == "global" else
                                            ["field_value"] * roster.slots(query["sensor"]))})
        for group in groups:
            def select(arrays):
                return np.column_stack([arrays[s["query"]][:, s["time_index"], s["port"], s["slot"]]
                                        for s in group["selectors"]]) / np.asarray(group["scales"])
            value = _energy_score(select(pred), select(observed))
            joints.append({"id": group["id"], "energy_score": value,
                           "unit": "dimensionless", "coordinate_unit": group["unit"],
                           "scales": deepcopy(group["scales"]),
                           "dimensions": len(group["selectors"])})
        aggregate = float(np.mean([group["energy_score"] for group in joints]))
    report = {"schema_version": VERSION, "case_id": case["id"],
              "prediction_members": n_samples, "truth_realizations": n_truth,
              "lower_is_better": True,
              "marginal_crps": marginal, "joint_energy": joints,
              "joint_energy_equal_group_mean": aggregate,
              "joint_aggregate_unit": "dimensionless", "work_estimate": plan["pairwise_work"],
              "scoring_convention": "empirical distribution, ordered pairs including self, independent truths"}
    try:
        json.dumps(report, allow_nan=False)
    except (ValueError, OverflowError):
        raise EvaluationError("finite inputs overflowed score arithmetic; score is not representable") from None
    return report


def evaluate_trusted_predictor(predictor, truth_sampler, case, *, groups,
                               n_samples=64, n_truth=4, predictor_seed=0,
                               truth_seed=None, roster=DEFAULT_ROSTER,
                               limits=DEFAULT_LIMITS):
    """Reviewed local adapters ONLY; no submitted-code isolation is provided.

    ``truth_sampler`` is privileged and exposes sample_truth(..., truth_seed=).
    Its seed is independently generated unless the grader explicitly supplies
    one for reproducible checking. No seed is compared for numerical inequality:
    independence means separate sampling choices, not unequal integer labels.
    """
    plan = validate_plan(case, groups=groups, n_samples=n_samples, n_truth=n_truth,
                         roster=roster, limits=limits)
    _int(predictor_seed, "predictor_seed", 0, R6.MAX_SEED)
    if truth_seed is None:
        truth_seed = secrets.randbits(64)
    _int(truth_seed, "private truth seed", 0, R6.MAX_SEED)
    prediction = predictor.predict(deepcopy(case["actions"]), deepcopy(case["queries"]),
                                   n_samples=n_samples, seed=predictor_seed)
    # Fail malformed prediction before spending any simulation work.
    pred = validate_samples(prediction, plan["shapes"], n_samples)
    truth = truth_sampler.sample_truth(deepcopy(case["actions"]), deepcopy(case["queries"]),
                                      n_samples=n_truth, truth_seed=truth_seed)
    return score_case(case, {"samples": pred}, truth, groups=groups,
                      n_samples=n_samples, n_truth=n_truth, roster=roster, limits=limits)


def _reject_constant(value):
    raise EvaluationError("JSON non-finite constants are forbidden")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError("duplicate JSON object key")
        result[key] = value
    return result


def read_json_data(path, *, limits=DEFAULT_LIMITS):
    """Read bounded JSON data only; does not import, eval, or unpickle files."""
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(limits.max_bundle_bytes + 1)
        if len(data) > limits.max_bundle_bytes:
            raise EvaluationError("JSON file exceeds byte cap")
        return json.loads(data, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise EvaluationError("invalid JSON data file") from exc


def case_digest(case):
    """Bind data files to this exact public request as well as its opaque ID."""
    return hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_sample_bundle(path, case, *, n_members, roster=DEFAULT_ROSTER,
                        limits=DEFAULT_LIMITS):
    """Load a data-only prediction or grader truth bundle, strictly bound to a case."""
    _int(n_members, "n_members", 1, max(limits.max_prediction_members, limits.max_truth_members))
    shapes = validate_case(case, roster=roster, limits=limits)
    if sum(math.prod(s) for s in shapes) * n_members > limits.max_output_values:
        raise EvaluationError("bundle expected output exceeds scalar-value cap")
    obj = read_json_data(path, limits=limits)
    _keys(obj, {"schema_version", "case_id", "case_digest", "samples"}, "sample bundle")
    if obj["schema_version"] != BUNDLE_VERSION or obj["case_id"] != case["id"]:
        raise EvaluationError("sample bundle version or case identifier mismatch")
    if obj["case_digest"] != case_digest(case):
        raise EvaluationError("sample bundle does not match the exact public case request")
    return {"samples": validate_samples({"samples": obj["samples"]}, shapes, n_members)}


def write_sample_bundle(path, case, payload, *, n_members,
                         roster=DEFAULT_ROSTER, limits=DEFAULT_LIMITS):
    _int(n_members, "n_members", 1, max(limits.max_prediction_members, limits.max_truth_members))
    shapes = validate_case(case, roster=roster, limits=limits)
    if sum(math.prod(s) for s in shapes) * n_members > limits.max_output_values:
        raise EvaluationError("bundle output exceeds scalar-value cap")
    arrays = validate_samples(payload, shapes, n_members)
    obj = {"schema_version": BUNDLE_VERSION, "case_id": case["id"], "case_digest": case_digest(case),
           "samples": [a.tolist() for a in arrays]}
    data = json.dumps(obj, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) + 1 > limits.max_bundle_bytes:
        raise EvaluationError("encoded sample bundle exceeds byte cap")
    Path(path).write_bytes(data + b"\n")


class ConstantPredictor:
    """Explicit action-blind constant baseline, not a physical model."""
    def __init__(self, value=0.0, *, roster=DEFAULT_ROSTER):
        self.value = _finite_number(value, "constant")
        self.roster = roster

    def predict(self, actions, queries, n_samples=64, seed=0):
        return {"samples": [np.full((n_samples, len(q["t"]), self.roster.n_ports,
                                     self.roster.slots(q["sensor"])), self.value)
                            for q in queries]}


class PersistencePredictor:
    """Repeats supplied public t=0 sensor readouts; does not model actions/poses."""
    def __init__(self, observations: Mapping, *, roster=DEFAULT_ROSTER):
        self.roster = roster
        self.observations = {}
        for sensor, value in observations.items():
            arr = np.asarray(value)
            expected = (roster.n_ports, roster.slots(sensor))
            if arr.shape != expected or arr.dtype.kind not in "iuf" or not np.isfinite(arr).all():
                raise EvaluationError("persistence requires finite public per-sensor initial readouts")
            self.observations[sensor] = np.array(arr, dtype=float, copy=True)

    def predict(self, actions, queries, n_samples=64, seed=0):
        result = []
        for query in queries:
            if query["sensor"] not in self.observations:
                raise EvaluationError("persistence baseline is missing a requested sensor")
            initial = self.observations[query["sensor"]]
            result.append(np.broadcast_to(initial, (n_samples, len(query["t"])) + initial.shape).copy())
        return {"samples": result}


def main(argv=None):
    """Data-only file scoring entry point; no code execution/import interface."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="JSON id/actions/queries public case")
    parser.add_argument("--groups", required=True, help="JSON list of predeclared private score groups")
    parser.add_argument("--prediction", required=True, help="data-only prediction JSON bundle")
    parser.add_argument("--truth", required=True, help="grader-owned truth JSON bundle")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--n-truth", type=int, default=4)
    parser.add_argument("--n-ports", type=int, default=12)
    parser.add_argument("--device-slots", type=int, nargs="+", default=[13, 19])
    parser.add_argument("--max-horizon-tu", type=float, default=DEFAULT_LIMITS.max_horizon_tu,
                        help="explicit world horizon; worked example requires 50")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    roster = PublicRoster(args.n_ports, tuple(args.device_slots))
    limits = replace(DEFAULT_LIMITS, max_horizon_tu=args.max_horizon_tu)
    case, groups = read_json_data(args.case, limits=limits), read_json_data(args.groups, limits=limits)
    validate_plan(case, groups=groups, n_samples=args.n_samples, n_truth=args.n_truth,
                  roster=roster, limits=limits)
    prediction = read_sample_bundle(args.prediction, case, n_members=args.n_samples,
                                    roster=roster, limits=limits)
    truth = read_sample_bundle(args.truth, case, n_members=args.n_truth, roster=roster, limits=limits)
    report = score_case(case, prediction, truth, groups=groups,
                        n_samples=args.n_samples, n_truth=args.n_truth, roster=roster, limits=limits)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

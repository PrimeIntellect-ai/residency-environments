"""Isolated R6 absolute-time executable-predictor prototype (evaluator-side).

``Predictor.predict(actions, queries, n_samples=64, seed=0)`` describes the
submitted-code API only. The privileged ``OracleRunner.sample_truth`` uses a
separate grader-owned ``truth_seed`` namespace. Evaluators bind an opaque t=0
scenario with the private factory. This module does not modify the R5 tools, caches, truths, or transport.
See probes/blobs/agentenv/round6/runner/DESIGN.md for the supported grammar.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Protocol

import numpy as np

SIM_DT = 0.02
TIME_ATOL = 1e-9                 # tu; representation tolerance, not rounding
ADJUST_TU = 5.0
ADJUST_TICKS = 250
AMP_RANGE = (0.0, 3.0)            # shared R6 exploration/evaluation domain
DURATION_MAX = 50.0
CONTROL_RANGE = (-1.0, 1.0)
DEFAULT_SAMPLES = 64
MAX_SEED = 2**64 - 1
MAX_EXACT_TICK = 2**53 - 1        # numerical representation bound, not a budget
NOISE_POLICY = "independent-experiment-v1"


class ProtocolError(ValueError):
    """The complete request is outside the prototype's explicit grammar."""


def _number(value, label: str) -> float:
    if type(value) not in (int, float):
        raise ProtocolError(f"{label} must be a finite JSON number (not bool)")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ProtocolError(f"{label} must be finite") from None
    if not math.isfinite(result):
        raise ProtocolError(f"{label} must be finite")
    return result


def _integer(value, label: str, low: int, high: int | None = None) -> int:
    if type(value) is not int:
        raise ProtocolError(f"{label} must be an integer (not bool or float)")
    if value < low or (high is not None and value > high):
        bounds = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise ProtocolError(f"{label} must be {bounds}")
    return value


def _tick(value, label: str) -> int:
    t = _number(value, label)
    if t < 0:
        raise ProtocolError(f"{label} must be nonnegative")
    scaled = t / SIM_DT
    if not math.isfinite(scaled) or scaled > MAX_EXACT_TICK:
        raise ProtocolError(f"{label} exceeds the exact tick representation range")
    k = round(scaled)
    if abs(t - k * SIM_DT) > TIME_ATOL:
        raise ProtocolError(f"{label} must be on the dt={SIM_DT:g} grid")
    return k


def _keys(obj, required: set[str], label: str) -> None:
    if type(obj) is not dict:
        raise ProtocolError(f"{label} must be an object")
    if set(obj) != required:
        # Do not echo arbitrary input keys or any private configuration.
        raise ProtocolError(f"{label} requires exactly keys {sorted(required)}")


@dataclass(frozen=True)
class _Action:
    tick: int
    end: int
    kind: str
    device: int | None = None
    u: tuple[float, float, float] | None = None
    port: int | None = None
    amp: float = 0.0
    effective: bool = False


@dataclass(frozen=True)
class _Query:
    sensor: str
    ticks: tuple[int, ...]
    slots: int


def _parse(actions, queries, n_samples, truth_seed, n_ports, devices):
    n_samples = _integer(n_samples, "n_samples", 1, np.iinfo(np.intp).max)
    truth_seed = _integer(truth_seed, "truth_seed", 0, MAX_SEED)
    if type(actions) is not list or type(queries) is not list:
        raise ProtocolError("actions and queries must be lists")
    parsed_actions = []
    previous_start = -1
    lane_ends = {"adjust": -1, "inject": -1}
    for i, obj in enumerate(actions):
        label = f"actions[{i}]"
        if type(obj) is not dict:
            raise ProtocolError(f"{label} must be an object")
        kind = obj.get("kind")
        if type(kind) is not str or kind not in ("adjust", "inject"):
            raise ProtocolError(f"{label}.kind must be 'adjust' or 'inject'")
        required = ({"t", "kind", "device", "u"} if kind == "adjust"
                    else {"t", "kind", "port", "amp", "dur"})
        _keys(obj, required, label)
        k = _tick(obj["t"], label + ".t")
        if k <= previous_start:
            raise ProtocolError("actions must have strictly increasing start times; "
                                "simultaneous starts are unsupported")
        if k < lane_ends[kind]:
            raise ProtocolError(f"overlapping {kind} intervals are unsupported")
        if kind == "adjust":
            di = _integer(obj["device"], label + ".device", 0, len(devices) - 1)
            u = obj["u"]
            if type(u) is not list or len(u) != 3:
                raise ProtocolError(f"{label}.u must be a list of three numbers")
            u = tuple(_number(v, label + f".u[{j}]") for j, v in enumerate(u))
            if any(v < CONTROL_RANGE[0] or v > CONTROL_RANGE[1] for v in u):
                raise ProtocolError(f"{label}.u components must be in [-1, 1]")
            action = _Action(k, k + ADJUST_TICKS, kind, device=di, u=u)
        else:
            port = _integer(obj["port"], label + ".port", 0, n_ports - 1)
            amp = _number(obj["amp"], label + ".amp")
            if not AMP_RANGE[0] <= amp <= AMP_RANGE[1]:
                raise ProtocolError(f"{label}.amp must be in [0, 3]")
            duration = _number(obj["dur"], label + ".dur")
            if not 0 < duration <= DURATION_MAX:
                raise ProtocolError(f"{label}.dur must be in (0, 50]")
            dur_ticks = _tick(obj["dur"], label + ".dur")
            if dur_ticks < 1:
                raise ProtocolError(f"{label}.dur must span at least one dt tick")
            action = _Action(k, k + dur_ticks, kind, port=port, amp=amp)
        if action.end > MAX_EXACT_TICK:
            raise ProtocolError(f"{label} interval exceeds the exact tick range")
        parsed_actions.append(action)
        lane_ends[kind] = action.end
        previous_start = k

    parsed_queries = []
    sensors = {f"device{i}": int(dev.k) for i, dev in enumerate(devices)}
    sensors["global"] = 2
    for i, obj in enumerate(queries):
        label = f"queries[{i}]"
        _keys(obj, {"sensor", "t"}, label)
        sensor = obj["sensor"]
        if type(sensor) is not str or sensor not in sensors:
            raise ProtocolError(f"{label}.sensor is not in the anonymous roster")
        if type(obj["t"]) is not list:
            raise ProtocolError(f"{label}.t must be a list of absolute times")
        ticks = tuple(_tick(t, label + f".t[{j}]")
                      for j, t in enumerate(obj["t"]))
        parsed_queries.append(_Query(sensor, ticks, sensors[sensor]))
    return tuple(parsed_actions), tuple(parsed_queries), n_samples, truth_seed


def _adjust_pose(dev, u, mix) -> bool:
    """Native R3-final / R5 truth pose math, with actual-change detection.

    The map is private. Unlike R5 transport, a bound-striking valid command
    clips dilation; invalid input controls were already rejected by _parse.
    """
    delta = mix @ np.asarray(u, float)
    center = (dev.center + delta[:2]) % dev.L
    dilation = float(np.clip(dev.dilation * np.exp(delta[2]), *dev.dil_bounds))
    changed = not np.array_equal(center, dev.center) or dilation != dev.dilation
    dev.center = center
    dev.dilation = dilation
    return changed


def _effective_actions(actions, devices, mix):
    poses = deepcopy(devices)
    result = []
    for a in actions:
        changed = (a.amp != 0.0 if a.kind == "inject"
                   else _adjust_pose(poses[a.device], a.u, mix))
        if changed:
            result.append(_Action(a.tick, a.end, a.kind, a.device, a.u,
                                  a.port, a.amp, True))
    return tuple(result)


def _truth_member_seed(truth_seed: int, member: int) -> int:
    """Private truth-only continuation seed; never a submitted predictor seed.

    This mapping is fixed for v1 and uses a grader-owned truth_seed. Changing
    n_samples only extends the member prefix. Hidden initialization, queries,
    actions, and the submitted predictor's sampling seed are not inputs.
    """
    data = f"physim/blobround6/truth/{NOISE_POLICY}|{truth_seed}|{member}".encode("ascii")
    return int.from_bytes(hashlib.sha256(data).digest()[:16], "big")


def _clone_sim(template):
    # Native step_chunk only rebinds F/t_step and mutates the owned RNG.
    # Immutable template parameters can therefore be shared across members.
    sim = dict(template)
    sim["F"] = np.array(template["F"], copy=True)
    sim["rng"] = deepcopy(template["rng"])
    return sim


class Predictor(Protocol):
    """Structural interface for submitted code, NOT a privileged oracle.

    This declaration does not implement a model, scorer, or code sandbox.
    """

    def predict(self, actions, queries, n_samples=DEFAULT_SAMPLES, seed=0) -> dict:
        """Return anonymous joint samples using the predictor's sampling seed."""
        ...


class OracleRunner:
    """Privileged oracle bound privately to one opaque exact t=0 world.

    This is NOT an agent-written predictor. Its truth_seed is owned by the
    evaluator and must be sampled independently of the submitted-code seed.
    Its constructor is private dependency injection; _native_oracle binds the
    native engine. Toy fixtures supply small exact states for protocol tests.
    This trusted library has no final untrusted-request resource budgets.
    """

    def __init__(self, *, _template, _devices, _port_perm, _adjust_mix,
                 _emitter_yx, _stepper: Callable):
        fields = np.asarray(_template["F"])
        if (fields.ndim != 3 or fields.shape[-1] != fields.shape[-2]
                or fields.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or not np.isfinite(fields).all()):
            raise ValueError("private template requires finite full-precision fields")
        if _template["t_step"] != 0 or abs(_template["dt"] - SIM_DT) > 1e-15:
            raise ValueError("private template must be the exact dt=0.02 t=0 state")
        perm = np.asarray(_port_perm)
        if (perm.dtype.kind not in "iu" or perm.shape != (fields.shape[0],)
                or sorted(perm.tolist()) != list(range(fields.shape[0]))):
            raise ValueError("private port permutation is invalid")
        mix = np.asarray(_adjust_mix, float)
        if mix.shape != (3, 3) or not np.isfinite(mix).all():
            raise ValueError("private actuator template is invalid")
        emitter = np.asarray(_emitter_yx, float)
        if emitter.shape != (2,) or not np.isfinite(emitter).all():
            raise ValueError("private emitter template is invalid")
        if not _devices or any(int(d.k) < 1 for d in _devices):
            raise ValueError("private anonymous device roster is empty or invalid")
        if not isinstance(_template["rng"], np.random.Generator):
            raise ValueError("private template requires an exact base Generator")
        self._template = deepcopy(_template)
        # Keep owned template arrays immutable; only per-run fields are writable.
        for value in self._template.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        self._devices = deepcopy(tuple(_devices))
        self._perm = np.array(perm, dtype=int, copy=True)
        self._mix = mix.copy()
        self._emitter = emitter.copy()
        self._stepper = _stepper

    def _sample(self, sim, devices, sensor):
        # No RNG call, stepping, pose changes, or persistent sample cache.
        fields = np.asarray(sim["F"])[self._perm]
        if not np.isfinite(fields).all():
            raise RuntimeError("simulation produced non-finite fields")
        if sensor == "global":
            return np.stack((fields.mean(axis=(1, 2)),
                             fields.var(axis=(1, 2))), axis=1)
        device = devices[int(sensor.removeprefix("device"))]
        return device.sample(fields, sim["dx"])

    def _fill_queries(self, sim, devices, marks, output, member):
        values = {}
        for qi, ti, sensor in marks:
            if sensor not in values:
                values[sensor] = self._sample(sim, devices, sensor)
            output[qi][member, ti] = values[sensor]

    def _advance(self, sim, tick, injection=None):
        n = tick - sim["t_step"]
        if n < 0:
            raise RuntimeError("internal scheduler attempted to reverse time")
        if n:
            self._stepper(sim, n, injections=[] if injection is None else [injection])
            if sim["t_step"] != tick:
                raise RuntimeError("native stepper did not reach its scheduled tick")

    def sample_truth(self, actions, queries, n_samples=DEFAULT_SAMPLES, *, truth_seed):
        """Privately sample truth with a grader-owned seed, not predictor seed.

        Return {"samples": [array per query]} with coherent member trajectories.
        All times are absolute, finite dt ticks. Action starts must be strictly
        increasing. Validate the whole request before running any simulation.
        Empty outputs are legal. Containers are float64, without rounding or
        private metadata. A public request's ``seed`` key is NOT accepted here.

        Each member starts from the same owned physical state with fresh future
        forcing from t=0, including sham experiments and time before an action.
        The evaluator chooses a new truth_seed per experiment. Reusing one is
        supported for evaluator debugging only; predictors do not receive it.
        No realized continuation is reused across independently seeded trials.
        """
        actions, queries, n_samples, truth_seed = _parse(
            actions, queries, n_samples, truth_seed, len(self._perm), self._devices)
        effective = _effective_actions(actions, self._devices, self._mix)
        output = [np.empty((n_samples, len(q.ticks), len(self._perm), q.slots),
                           dtype=np.float64) for q in queries]
        marks = defaultdict(list)
        for qi, q in enumerate(queries):
            for ti, tick in enumerate(q.ticks):
                marks[tick].append((qi, ti, q.sensor))
        if not marks:
            return {"samples": output}
        last = max(marks)
        effective = tuple(a for a in effective if a.tick <= last)
        starts = {a.tick: a for a in effective}
        ends = {a.end for a in effective if a.kind == "inject" and a.end <= last}
        agenda = sorted(set(marks) | set(starts) | ends)
        for member in range(n_samples):
            live = _clone_sim(self._template)
            live["rng"] = np.random.default_rng(_truth_member_seed(truth_seed, member))
            devices = deepcopy(self._devices)
            active = None
            active_end = -1
            for tick in agenda:
                self._advance(live, tick, active)
                # Half-open source interval: finish, start, sample, then advance.
                if tick == active_end:
                    active = None
                a = starts.get(tick)
                if a is not None:
                    if a.kind == "adjust":
                        _adjust_pose(devices[a.device], a.u, self._mix)
                    else:
                        active = dict(field=int(self._perm[a.port]),
                                      y=float(self._emitter[0]),
                                      x=float(self._emitter[1]), amp=a.amp)
                        active_end = a.end
                self._fill_queries(live, devices, marks.get(tick, ()), output, member)
        return {"samples": output}


def _native_oracle(world: str, hidden_seed: int, *, workers: int = 1) -> OracleRunner:
    """Private evaluator binding; does not load frames or expose hidden inputs.

    Existing world/seed selection is the caller's responsibility. This performs
    native initialization, not a new recorded run, and builds no frozen truths.
    """
    from . import blobcore as B
    from blobkit.soup import sim_cpu

    if type(workers) is not int or workers < 1:
        raise ValueError("private workers must be a positive integer")
    genome = B.load_genome(world)
    template = sim_cpu.init_soup(genome, L=128.0, seed=hidden_seed,
                                  dtype="f32", workers=workers)
    secrets = B.agdev.world_secrets(B.world_key(world, hidden_seed),
                                    template["na"] + template["nc"],
                                    B.ROSTER, template["L"])
    devices = []
    for i, (cfg, private) in enumerate(zip(B.ROSTER, secrets["devices"])):
        devices.append(B.agdev.ProbeDevice(
            dev_id=i, lattice=cfg["lattice"], n_rings=cfg["n_rings"],
            base_ds=cfg["base_ds"], L=template["L"], **private))
    return OracleRunner(_template=template, _devices=devices,
                     _port_perm=secrets["port_perm"],
                     _adjust_mix=B.adjust_mix(B.world_key(world, hidden_seed)),
                     _emitter_yx=secrets["devices"][B.DEV_A]["center"],
                     _stepper=B.agdev.step_chunk)

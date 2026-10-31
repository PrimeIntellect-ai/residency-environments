"""Run a declared lab world through the PhySim oracle runner.

A lab world is a Python module with ``init(seed, size)`` and ``step(state)`` over a
dict of square float arrays (the ALife lab's labkit contract). ``LabWorld`` declares
everything the runner and the agent contract need that the module does not say:

- ``fields``: the state keys that are dynamical fields, in port order.
- ``steps_per_tick``: native steps per public tick (one tick is 0.02 time units).
- ``noise``: per field, ``("additive" | "proportional" | "conserved", sd)``; sd is per sqrt(native
  time). Noise draws come only from the runner's per-member generator, a fixed
  number per step, so actions never shift a member's noise stream.
- ``noise_corr``: per field, an optional spatial correlation length ``ell`` in world units.
  The field's white draws are smoothed by a periodic Gaussian filter of standard deviation
  ``ell`` (noise covariance ``exp(-r^2 / (4 ell^2))``) and rescaled so the per-point variance
  is exactly the white form's; the draws per step are unchanged. ``conserved`` smooths the
  divergence of the white flux, so it still adds no mass. White noise per pixel changes
  with the grid; correlated noise has a continuum limit. A field not listed stays white.
- ``noise_intensity``: fields whose ``sd`` is the intensity of the continuum noise in world
  units instead of a per-point level. The noise is then the grid sample of a continuum
  field whose spectral density at low wavenumber is ``sd^2`` per native time (``(sd/2)^2 k^2``
  for ``conserved``, whose flux has intensity ``(sd/2)^2``), whatever the grid spacing ``dx``
  or the correlation length: white draws are scaled by ``1/dx`` (``0.5/dx^2`` for the
  divergence of the conserved flux) and, with an ``ell`` in ``noise_corr``, smoothed by the
  same Gaussian with unit gain at k = 0 (not rescaled to the white per-point variance). So
  adding or changing ``ell`` leaves the large-scale forcing unchanged and the per-point sd
  falls like ``1/(2 sqrt(pi) ell)`` for ``ell >> dx`` (2D). At ``dx = 1`` and no ``ell`` it
  is the white form bit for bit; a white level ``s`` per point at spacing ``dx0`` is the
  intensity ``s * dx0`` (``s * dx0^2`` for conserved). A field not listed keeps the
  per-point meaning above.
- ``ports``: per field, the source gain (native field units per native time per unit
  of PhySim amplitude) and an optional clip applied after each injection.
- ``dx``: world units per pixel; the Gaussian source width is in world units.
- ``sys_path``: directories the module imports from (for example the lab's labkit).

- ``carry``: non-field state keys taken from the prepared state (for example a clock).
  Every other non-field entry (parameters copied into the state, operators precomputed
  from dt) is rebuilt by the overridden module's own ``init``, so an arm with other
  parameters or another dt uses consistent numerics. ``carry=None`` keeps them all.

Observed ports and driven ports are declared separately:

- ``derived``: observe-only channels. ``Derived(name, fn)`` is ``fn(module, state)``
  evaluated on the module state at t=0 and after every advance of the runner (every
  tick at which anything is read). It rides in ``F`` after the fields, so the probes
  and the global sensor read it like any other port. It must not modify the state or
  draw random numbers, and it can never be an inject target.
- ``actuators``: the public inject roster. ``Actuator(field, sign)`` is a source on a
  dynamical field with a declared sign; an inject on port ``p`` adds
  ``sign * gain * amp`` (the field's ``Port`` gain and clip) through ``actuators[p]``.
  A signed field such as vorticity is offered as a +/- pair, or as one port of either
  sign. PhySim's grammar has no signed amplitude (``amp`` is in [0, 3] and the inject
  keys are fixed), so a single bipolar port is not expressible without changing it.
  ``None`` keeps OracleRunner's roster: port ``p`` drives the field observed on port
  ``p`` with sign +1, over the fields only when there are derived channels.

The observed ports are ``fields + derived`` (``LabWorld.observed``), sampled in the
runner's port order. ``LabOracleRunner`` is OracleRunner with the actuator roster kept
apart from the observed ports; with the default roster it is bit-identical to it.

The runner copies only ``F`` and ``rng`` per member, so the stepper keeps any other
mutable state of the module in a per-member copy created on its first call.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import blobround6 as R6
from .devices import INJ_SIGMA


@dataclass(frozen=True)
class Port:
    gain: float
    clip: tuple[float | None, float | None] = (None, None)


@dataclass(frozen=True)
class Actuator:
    """One public inject port: a source on a dynamical ``field`` with sign +1 or -1."""

    field: str
    sign: int = 1


@dataclass(frozen=True)
class Derived:
    """One observe-only channel, ``fn(module, state) -> array`` of the field shape."""

    name: str
    fn: Callable


@dataclass(frozen=True)
class LabWorld:
    path: str
    fields: tuple[str, ...]
    steps_per_tick: int
    native_dt: float
    ports: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    dx: float = 1.0
    sys_path: tuple[str, ...] = ()
    carry: tuple[str, ...] | None = None
    actuators: tuple[Actuator, ...] | None = None
    derived: tuple[Derived, ...] = ()
    noise_corr: dict = field(default_factory=dict)
    noise_intensity: tuple[str, ...] = ()

    def __post_init__(self):
        for f, (kind, sd) in self.noise.items():
            if f not in self.fields or kind not in ("additive", "proportional", "conserved") or sd < 0:
                raise ValueError(f"noise for {f!r} must be additive, proportional or conserved with sd >= 0")
        for f, ell in self.noise_corr.items():
            if f not in self.fields or isinstance(ell, bool) or not np.isfinite(ell) or ell <= 0:
                raise ValueError(f"noise correlation length for {f!r} must name a field and be finite and > 0")
        if isinstance(self.noise_intensity, str):
            raise ValueError("noise_intensity is a sequence of field names")
        intensity = tuple(self.noise_intensity)
        if len(set(intensity)) != len(intensity) or not set(intensity) <= set(self.noise):
            raise ValueError("noise_intensity must list distinct fields that have a declared noise")
        object.__setattr__(self, "noise_intensity", intensity)
        derived = tuple(self.derived)
        names = [d.name for d in derived]
        if len(set(names)) != len(names) or set(names) & set(self.fields):
            raise ValueError("derived channel names must be unique and distinct from the fields")
        object.__setattr__(self, "derived", derived)
        if self.actuators is None:
            return
        actuators = tuple(self.actuators)
        if not actuators or len(set(actuators)) != len(actuators):
            raise ValueError("declare a nonempty actuator list without duplicates")
        for a in actuators:
            if a.field in names:
                raise ValueError(f"observe-only channel {a.field!r} cannot be an actuator")
            if a.field not in self.fields:
                raise ValueError(f"actuator field {a.field!r} is not a dynamical field")
            if a.field not in self.ports:
                raise ValueError(f"actuator field {a.field!r} has no declared port gain")
            if type(a.sign) is not int or a.sign not in (1, -1):
                raise ValueError("actuator sign must be the integer +1 or -1")
        object.__setattr__(self, "actuators", actuators)

    @property
    def observed(self) -> tuple[str, ...]:
        return tuple(self.fields) + tuple(d.name for d in self.derived)

    def actuator_rows(self):
        """The roster as (F row, sign) for LabOracleRunner; None is OracleRunner's roster."""
        if self.actuators is None:
            return None
        return tuple((self.fields.index(a.field), a.sign) for a in self.actuators)

    def declaration(self) -> dict:
        """JSON-ready declaration. The port-interface keys appear only when declared, so a
        declaration frozen before they existed (labeval suites) is unchanged."""
        d = dataclasses.asdict(self)
        if not self.noise_corr:
            d.pop("noise_corr")
        if not self.noise_intensity:
            d.pop("noise_intensity")
        if self.actuators is None:
            d.pop("actuators")
        if self.derived:
            d["derived"] = [dict(name=x.name, fn=f"{x.fn.__module__}.{x.fn.__qualname__}") for x in self.derived]
        else:
            d.pop("derived")
        return d

    def load(self):
        """A fresh module instance with the declared overrides (WORLD_P is read at import)."""
        for p in self.sys_path:
            if p not in sys.path:
                sys.path.insert(0, p)
        old = os.environ.get("WORLD_P")
        os.environ["WORLD_P"] = json.dumps(self.overrides)
        bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # a world folder (for example a bundle's) must stay byte-identical
        try:
            spec = importlib.util.spec_from_file_location(f"labworld_{abs(hash(self.path))}", self.path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = bytecode
            if old is None:
                os.environ.pop("WORLD_P", None)
            else:
                os.environ["WORLD_P"] = old
        if hasattr(module, "P"):
            module.P.update(self.overrides)
        return module

    def pack(self, state):
        return np.stack([np.asarray(state[f], dtype=np.float64) for f in self.fields])

    def aux(self, state):
        return {k: v for k, v in state.items() if k not in self.fields}


class LabStepper:
    """``stepper(sim, n, injections)`` for ``OracleRunner``: n public ticks."""

    def __init__(self, world: LabWorld, aux_template: dict):
        self.world = world
        self.module = world.load()
        self.aux_template = copy.deepcopy(aux_template)
        if world.carry is not None:
            fresh = self.module.init(0, aux_template["_grid"]) if "_grid" in aux_template else None
            if fresh is not None:
                rebuilt = {k: v for k, v in fresh.items() if k not in world.fields}
                rebuilt.update({k: aux_template[k] for k in world.carry if k in aux_template})
                self.aux_template = copy.deepcopy(rebuilt)
        self.aux_template.pop("_grid", None)
        self._bumps = {}
        self._filters = {}

    def _smooth(self, x, ell, conserved=False):
        """``x`` (white draws, or the raw divergence of a white flux) smoothed by a periodic Gaussian
        of standard deviation ``ell`` world units, scaled to unit per-point variance."""
        key = (x.shape, float(ell), conserved)
        if key not in self._filters:
            n0, n1 = x.shape
            dx = self.world.dx
            ky = 2 * np.pi * np.fft.fftfreq(n0, d=dx)[:, None]
            kx = 2 * np.pi * np.fft.fftfreq(n1, d=dx)[None, :]
            gain2 = np.exp(-(ell**2) * (ky**2 + kx**2))  # |G|^2 on the full spectrum
            if conserved:  # power of the backward-difference divergence of a unit white flux
                gain2 = gain2 * (2 * (1 - np.cos(ky * dx)) + 2 * (1 - np.cos(kx * dx)))
            var = float(gain2.mean())  # per-point variance of the filtered field (Parseval)
            kxr = 2 * np.pi * np.fft.rfftfreq(n1, d=dx)[None, :]
            self._filters[key] = np.exp(-0.5 * ell**2 * (ky**2 + kxr**2)) / np.sqrt(var)
        return np.fft.irfft2(self._filters[key] * np.fft.rfft2(x), s=x.shape)

    def _intensity(self, x, ell=None, conserved=False):
        """Unit-intensity continuum noise sampled on the grid, from white draws ``x`` (or the
        index-unit divergence of a white flux): scaled by ``1/dx`` (``0.5/dx^2``) and, when
        ``ell`` is given, smoothed by a periodic Gaussian of standard deviation ``ell`` world
        units with unit gain at k = 0, so the low-wavenumber spectrum does not depend on ``ell``."""
        dx = self.world.dx
        c = 0.5 / dx**2 if conserved else 1.0 / dx
        if ell is None:
            return x * c
        key = ("intensity", x.shape, float(ell), conserved)
        if key not in self._filters:
            n0, n1 = x.shape
            ky = 2 * np.pi * np.fft.fftfreq(n0, d=dx)[:, None]
            kxr = 2 * np.pi * np.fft.rfftfreq(n1, d=dx)[None, :]
            self._filters[key] = c * np.exp(-0.5 * ell**2 * (ky**2 + kxr**2))
        return np.fft.irfft2(self._filters[key] * np.fft.rfft2(x), s=x.shape)

    def add_noise(self, state, rng, h):
        """One native step's dynamics noise on every field (``state`` entries replaced). The draws
        per field are fixed (two arrays for conserved, one otherwise) whatever the levels."""
        w = self.world
        for f in w.fields:
            kind, sd = w.noise.get(f, ("additive", 0.0))
            ell = w.noise_corr.get(f)
            intensity = f in w.noise_intensity
            if kind == "conserved":
                # Divergence of a white flux on the periodic grid: no net mass (per-point form: per-pixel sd = sd).
                ey, ex = rng.standard_normal((2,) + state[f].shape)
                if sd > 0:
                    div = (ey - np.roll(ey, 1, 0)) + (ex - np.roll(ex, 1, 1))
                    if intensity:
                        state[f] = state[f] + sd * np.sqrt(h) * self._intensity(div, ell, conserved=True)
                    elif ell is None:
                        state[f] = state[f] + sd * np.sqrt(h) * 0.5 * div
                    else:
                        state[f] = state[f] + sd * np.sqrt(h) * self._smooth(div, ell, conserved=True)
                continue
            xi = rng.standard_normal(state[f].shape)
            if sd > 0:
                if intensity:
                    xi = self._intensity(xi, ell)
                elif ell is not None:
                    xi = self._smooth(xi, ell)
                a = sd * np.sqrt(h)
                state[f] = state[f] + (a * xi if kind == "additive" else a * xi * state[f])
        return state

    def _bump(self, shape, y, x):
        key = (shape, round(y, 9), round(x, 9))
        if key not in self._bumps:
            n0, n1 = shape
            dx = self.world.dx
            cy = (np.arange(n0) + 0.5) * dx
            cx = (np.arange(n1) + 0.5) * dx
            Ly, Lx = n0 * dx, n1 * dx
            dyy = (cy - y + Ly / 2) % Ly - Ly / 2
            dxx = (cx - x + Lx / 2) % Lx - Lx / 2
            self._bumps[key] = np.exp(-(dyy[:, None] ** 2 + dxx[None, :] ** 2) / (2 * INJ_SIGMA**2))
        return self._bumps[key]

    def observe(self, state):
        """F rows: the fields, then the derived channels evaluated on ``state``."""
        w = self.world
        if not w.derived:
            return w.pack(state)
        rows = [np.asarray(state[f], dtype=np.float64) for f in w.fields]
        rows += [np.asarray(d.fn(self.module, state), dtype=np.float64) for d in w.derived]
        return np.stack(rows)

    def initial(self, state):
        """t=0 F rows for a prepared state, derived channels on the state the stepper starts from."""
        if not self.world.derived:
            return self.world.pack(state)
        start = copy.deepcopy(self.aux_template)
        start.update({f: np.asarray(state[f], dtype=np.float64) for f in self.world.fields})
        return self.observe(start)

    def __call__(self, sim, n, injections=()):
        w = self.world
        if "_lab_aux" not in sim:
            sim["_lab_aux"] = copy.deepcopy(self.aux_template)
        state = dict(sim["_lab_aux"])
        F = np.array(sim["F"], dtype=np.float64, copy=True)
        for i, f in enumerate(w.fields):
            state[f] = F[i]
        rng = sim["rng"]
        h = w.native_dt
        for _ in range(n):
            for _ in range(w.steps_per_tick):
                for inj in injections:
                    f = w.fields[inj["field"]]
                    port = w.ports.get(f)
                    if port is None or inj["amp"] == 0.0:
                        continue
                    state[f] = state[f] + port.gain * inj["amp"] * h * self._bump(state[f].shape, inj["y"], inj["x"])
                    lo, hi = port.clip
                    if lo is not None or hi is not None:
                        state[f] = np.clip(state[f], lo, hi)
                state = self.add_noise(state, rng, h)
                state = self.module.step(state)
            sim["t_step"] += 1
        sim["F"] = self.observe(state)
        sim["_lab_aux"] = {k: v for k, v in state.items() if k not in w.fields}
        return sim


class LabOracleRunner(R6.OracleRunner):
    """``OracleRunner`` whose inject roster is declared apart from the observed ports.

    Observed ports are the rows of ``F`` in ``_port_perm`` order, exactly as in
    OracleRunner, and may include ``_observe_only`` rows (F-row indices) that the
    stepper derives. ``_actuators`` is the public inject roster: port ``p`` is
    ``(row, sign)`` and its pulse reaches the stepper as ``field=row``,
    ``amp=sign * amp``. ``None`` is OracleRunner's roster, ``(perm[p], +1)`` for every
    observed port ``p`` whose row is not observe-only, in port order.

    Construction checks, member seeding, devices, pose updates, sampling and the
    stepper seam are the base class's. ``sample_truth`` repeats the base scheduler with
    two changes: requests are parsed against the actuator count, and a pulse's field
    and amplitude come from its actuator.
    """

    def __init__(self, *, _actuators=None, _observe_only=(), **runner):
        super().__init__(**runner)
        n = len(self._perm)
        observe_only = frozenset(int(i) for i in _observe_only)
        if not observe_only <= set(range(n)) or len(observe_only) == n:
            raise ValueError("observe-only rows must be a proper subset of the field rows")
        if _actuators is None:
            if observe_only:
                # Inject port p would no longer name the field read on observed port p.
                raise ValueError("declare the actuator roster when observe-only rows exist")
            actuators = tuple((int(r), 1) for r in self._perm)
        else:
            actuators = []
            for row, sign in _actuators:
                if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or not 0 <= row < n:
                    raise ValueError("an actuator row must be an integer field row")
                if int(row) in observe_only:
                    raise ValueError("an observe-only channel cannot be an actuator")
                if type(sign) is not int or sign not in (1, -1):
                    raise ValueError("actuator sign must be the integer +1 or -1")
                actuators.append((int(row), sign))
            actuators = tuple(actuators)
            if not actuators or len(set(actuators)) != len(actuators):
                raise ValueError("declare a nonempty actuator roster without duplicates")
        self._actuators = actuators
        self._observe_only = observe_only

    @classmethod
    def wrap(cls, runner: R6.OracleRunner, *, actuators=None, observe_only=()):
        """The same bound world (template, devices, ports, adjustment map, stepper)."""
        if not isinstance(runner, R6.OracleRunner):
            raise TypeError("wrap requires a centered-pulse-v2 OracleRunner")
        return cls(
            _template=runner._template,
            _devices=runner._devices,
            _port_perm=runner._perm,
            _adjust_mix=runner._mix,
            _stepper=runner._stepper,
            _actuators=actuators,
            _observe_only=observe_only,
        )

    @property
    def n_actuators(self) -> int:
        return len(self._actuators)

    @property
    def n_observed(self) -> int:
        return len(self._perm)

    def sample_truth(self, actions, queries, n_samples=R6.DEFAULT_SAMPLES, *, truth_seed):
        """OracleRunner.sample_truth with inject ports indexing the actuator roster."""
        actions, queries, n_samples, truth_seed = R6._parse(
            actions, queries, n_samples, truth_seed, len(self._actuators), self._devices
        )
        effective = R6._effective_actions(actions, self._devices, self._mix)
        output = [np.empty((n_samples, len(q.ticks), len(self._perm), q.slots), dtype=np.float64) for q in queries]
        marks = defaultdict(list)
        for qi, q in enumerate(queries):
            for ti, tick in enumerate(q.ticks):
                marks[tick].append((qi, ti, q.sensor))
        if not marks:
            return {"samples": output}
        last = max(marks)
        effective = tuple(a for a in effective if a.tick <= last)
        starts = defaultdict(list)
        for action in effective:
            starts[action.tick].append(action)
        ends = {a.end for a in effective if a.kind == "inject" and a.end <= last}
        agenda = sorted(set(marks) | set(starts) | ends)
        for member in range(n_samples):
            live = R6._clone_sim(self._template)
            live["rng"] = np.random.default_rng(R6._truth_member_seed(truth_seed, member))
            devices = copy.deepcopy(self._devices)
            active = {}
            for tick in agenda:
                self._advance(live, tick, [active[i][1] for i in sorted(active)])
                active = {i: pulse for i, pulse in active.items() if pulse[0] > tick}
                for a in starts.get(tick, ()):
                    if a.kind == "adjust":
                        R6._adjust_pose(devices[a.device], a.u, self._mix)
                    else:
                        row, sign = self._actuators[a.port]
                        center = devices[a.device].center
                        active[a.device] = (
                            a.end,
                            dict(field=row, y=float(center[0]), x=float(center[1]), amp=a.amp if sign > 0 else -a.amp),
                        )
                self._fill_queries(live, devices, marks.get(tick, ()), output, member)
        return {"samples": output}


def template(world: LabWorld, state: dict, seed: int = 0) -> dict:
    """The runner's exact t=0 template from a prepared native state (dynamical fields only)."""
    return {"F": world.pack(state), "t_step": 0, "dt": R6.SIM_DT, "dx": world.dx, "rng": np.random.default_rng(seed)}


def oracle(world: LabWorld, state: dict, devices, *, port_perm=None, adjust_mix=None) -> LabOracleRunner:
    stepper = LabStepper(world, dict(world.aux(state), _grid=np.asarray(state[world.fields[0]]).shape))
    t0 = template(world, state)
    t0["F"] = stepper.initial(state)
    n = len(world.observed)
    return LabOracleRunner(
        _template=t0,
        _devices=devices,
        _port_perm=np.arange(n) if port_perm is None else port_perm,
        _adjust_mix=np.eye(3) if adjust_mix is None else adjust_mix,
        _stepper=stepper,
        _actuators=world.actuator_rows(),
        _observe_only=range(len(world.fields), n),
    )

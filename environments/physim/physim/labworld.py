"""Run a declared lab world through the PhySim oracle runner.

A lab world is a Python module with ``init(seed, size)`` and ``step(state)`` over a
dict of square float arrays (the ALife lab's labkit contract). ``LabWorld`` declares
everything the runner and the agent contract need that the module does not say:

- ``fields``: the state keys that are dynamical fields, in port order.
- ``steps_per_tick``: native steps per public tick (one tick is 0.02 time units).
- ``noise``: per field, ``("additive" | "proportional", sd)``; sd is per sqrt(native
  time). Noise draws come only from the runner's per-member generator, a fixed
  number per step, so actions never shift a member's noise stream.
- ``ports``: per field, the source gain (native field units per native time per unit
  of PhySim amplitude) and an optional clip applied after each injection.
- ``dx``: world units per pixel; the Gaussian source width is in world units.
- ``sys_path``: directories the module imports from (for example the lab's labkit).

- ``carry``: non-field state keys taken from the prepared state (for example a clock).
  Every other non-field entry (parameters copied into the state, operators precomputed
  from dt) is rebuilt by the overridden module's own ``init``, so an arm with other
  parameters or another dt uses consistent numerics. ``carry=None`` keeps them all.

The runner copies only ``F`` and ``rng`` per member, so the stepper keeps any other
mutable state of the module in a per-member copy created on its first call.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from .devices import INJ_SIGMA


@dataclass(frozen=True)
class Port:
    gain: float
    clip: tuple[float | None, float | None] = (None, None)


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

    def load(self):
        """A fresh module instance with the declared overrides (WORLD_P is read at import)."""
        for p in self.sys_path:
            if p not in sys.path:
                sys.path.insert(0, p)
        old = os.environ.get("WORLD_P")
        os.environ["WORLD_P"] = json.dumps(self.overrides)
        try:
            spec = importlib.util.spec_from_file_location(f"labworld_{abs(hash(self.path))}", self.path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
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
            self._bumps[key] = np.exp(-(dyy[:, None] ** 2 + dxx[None, :] ** 2) / (2 * INJ_SIGMA ** 2))
        return self._bumps[key]

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
                for f in w.fields:
                    kind, sd = w.noise.get(f, ("additive", 0.0))
                    xi = rng.standard_normal(state[f].shape)
                    if sd > 0:
                        a = sd * np.sqrt(h)
                        state[f] = state[f] + (a * xi if kind == "additive" else a * xi * state[f])
                state = self.module.step(state)
            sim["t_step"] += 1
        sim["F"] = w.pack(state)
        sim["_lab_aux"] = {k: v for k, v in state.items() if k not in w.fields}
        return sim


def template(world: LabWorld, state: dict, seed: int = 0) -> dict:
    """The runner's exact t=0 template from a prepared native state."""
    from .blobround6 import SIM_DT

    return {"F": world.pack(state), "t_step": 0, "dt": SIM_DT, "dx": world.dx,
            "rng": np.random.default_rng(seed)}


def oracle(world: LabWorld, state: dict, devices, *, port_perm=None, adjust_mix=None):
    from .blobround6 import OracleRunner

    nf = len(world.fields)
    return OracleRunner(_template=template(world, state), _devices=devices,
                        _port_perm=np.arange(nf) if port_perm is None else port_perm,
                        _adjust_mix=np.eye(3) if adjust_mix is None else adjust_mix,
                        _stepper=LabStepper(world, dict(world.aux(state), _grid=np.asarray(state[world.fields[0]]).shape)))

"""Lab bundles: a declared lab world (a Python world file) as a PhySim evaluation bundle.

A lab bundle has the layout and identities of a Blobkit world bundle (``bundles.Bundle``),
but its world is a lab world file run through ``labworld.LabOracleRunner`` instead of a
Blobkit genome on the fixed 256^2 grid:

- ``world/``: the world file, the modules it imports (for labkit worlds, ``labkit/``) and
  ``observables.py`` (the observe-only channels), all as regular files hashed in the
  manifest. ``world/`` is the only import path added for the world.
- ``labworld.json``: the declaration (``labworld.LabWorld``'s keys; derived channels name
  a function in a bundle file; ``fft_workers`` pins the world's FFT threads).
- ``apparatus.json``: the two probe devices, the observed-port permutation (F rows), the
  adjustment matrix and the protocol (``centered-pulse-v2``). The inject roster is the
  declaration's ``actuators`` in public port order.
- ``preparation.npz``: the prepared state, the dynamical fields and the declared ``carry``
  keys only (named float64 arrays). Every other piece of module state is rebuilt by the
  world's own ``init``, so nothing is unpickled.
- ``suite.json``, ``truth/*.npz``, ``checks.json``: as in a Blobkit bundle.

The manifest's ``schema_version`` is ``physim-lab-bundle-v1`` and its world object binds
every world file, the declaration and the implementation (the hashes of the runner,
adapter and scorer sources, and the numerical library versions). ``open_bundle`` picks
``LabBundle`` for such a manifest and ``Bundle`` otherwise.

Scoring reuses ``blobround6_eval`` unchanged. Its ``PublicRoster.n_ports`` is both the
inject-port bound and the observed-port count, so a lab bundle must declare as many
observed ports (fields plus derived channels) as actuators; ``LabBundle`` refuses others.

Trusted-code note: a lab bundle executes its world file in the grader's host process.
Load only bundles whose bytes were reviewed (the manifest hashes bind those bytes).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import blobround6 as R6
from . import blobround6_eval as scoring
from . import labworld as LW
from .bundles import (
    MAX_TOTAL,
    PROFILES,
    Bundle,
    BundleError,
    digest,
    identified,
    load_arrays,
    read_json,
    relative_path,
)

LAB_FORMAT = "physim-lab-bundle-v1"
SIMULATION_FILES = ("labworld.json", "apparatus.json", "preparation.npz")
DECLARATION_KEYS = {
    "name", "world_file", "fields", "steps_per_tick", "native_dt", "ports", "noise", "noise_corr",
    "noise_intensity", "overrides", "dx", "carry", "actuators", "derived", "fft_workers", "grid",
}
REFERENCE_DEPENDENCIES = ("numpy", "scipy")


def implementation_identity():
    """Sources and libraries whose change can change a lab bundle's simulation or scores."""
    from importlib.metadata import version

    from . import devices

    mods = (R6, LW, devices, scoring, sys.modules[__name__])
    return dict(
        source_sha256={m.__name__: digest(m.__file__) for m in mods},
        dependencies={name: version(name) for name in REFERENCE_DEPENDENCIES},
    )


def state_digest(arrays: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _load_function(root: Path, source: str, function: str):
    path = root / "world" / relative_path(source)
    name = "labbundle_obs_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True  # bundle folders stay byte-identical; see LabBundle
    spec.loader.exec_module(module)
    fn = getattr(module, function, None)
    if not callable(fn):
        raise BundleError(f"derived channel function {function!r} is missing from {source}")
    return fn


def world_from_declaration(decl: dict, root: Path) -> LW.LabWorld:
    """The LabWorld of a lab bundle declaration; paths resolve inside ``root/world``."""
    if type(decl) is not dict or set(decl) != DECLARATION_KEYS:
        raise BundleError("lab world declaration keys differ from the supported set")
    root = Path(root)
    derived = tuple(
        LW.Derived(d["name"], _load_function(root, d["source"], d["function"])) for d in decl["derived"]
    )
    try:
        return LW.LabWorld(
            path=str(root / "world" / relative_path(decl["world_file"])),
            fields=tuple(decl["fields"]),
            steps_per_tick=int(decl["steps_per_tick"]),
            native_dt=float(decl["native_dt"]),
            ports={f: LW.Port(gain=float(p["gain"]), clip=tuple(p["clip"])) for f, p in decl["ports"].items()},
            noise={f: (k, float(sd)) for f, (k, sd) in decl["noise"].items()},
            noise_corr={f: float(v) for f, v in decl["noise_corr"].items()},
            noise_intensity=tuple(decl["noise_intensity"]),
            overrides=dict(decl["overrides"]),
            dx=float(decl["dx"]),
            sys_path=(str(root / "world"),),
            carry=tuple(decl["carry"]),
            actuators=tuple(LW.Actuator(a["field"], int(a["sign"])) for a in decl["actuators"]),
            derived=derived,
        )
    except (TypeError, KeyError, ValueError) as exc:
        raise BundleError(f"invalid lab world declaration: {exc}") from exc


def devices_from_apparatus(apparatus: dict):
    from .devices import ProbeDevice

    devices = []
    for record in apparatus["devices"]:
        device = ProbeDevice(**record["parameters"])
        device.dilation = record["dilation"]
        device.Bm = np.asarray(record["motion_basis"], dtype=float)
        devices.append(device)
    return devices


def _check_labkit(root: Path):
    """A world imports its helper package from ``root/world``; refuse a different copy already imported."""
    for package in sorted(p for p in (root / "world").iterdir() if p.is_dir() and (p / "__init__.py").is_file()):
        for source in sorted(package.glob("*.py")):
            name = package.name if source.stem == "__init__" else f"{package.name}.{source.stem}"
            loaded = getattr(sys.modules.get(name), "__file__", None)
            if loaded is not None and digest(loaded) != digest(source):
                raise BundleError(f"module {name!r} is already imported from other bytes; use a fresh process")


def build_oracle(root, decl: dict, apparatus: dict, state: dict, *, world=None, devices=None):
    """``LabOracleRunner`` for a lab bundle's files (``world``/``devices`` override for privileged controls)."""
    root = Path(root)
    _check_labkit(root)
    world = world or world_from_declaration(decl, root)
    devices = devices or devices_from_apparatus(apparatus)
    perm = np.asarray(apparatus["port_permutation"], dtype=int)
    oracle = LW.oracle(world, state, devices, port_perm=perm, adjust_mix=np.asarray(apparatus["adjustment_matrix"]))
    module = oracle._stepper.module
    if hasattr(module, "ops") and hasattr(module.ops, "WORKERS"):
        module.ops.WORKERS = int(decl["fft_workers"])
    return oracle


def read_state(path, decl):
    arrays = load_arrays(path)
    keys = set(decl["fields"]) | set(decl["carry"])
    if set(arrays) != keys:
        raise BundleError("preparation must hold exactly the declared fields and carry keys")
    grid = tuple(decl["grid"])
    for f in decl["fields"]:
        if arrays[f].shape != grid or arrays[f].dtype != np.float64:
            raise BundleError("prepared field shape or dtype mismatch")
    return {k: (float(v) if v.ndim == 0 else v) for k, v in arrays.items()}


def validate_lab_manifest(manifest):
    if type(manifest) is not dict or manifest.get("schema_version") != LAB_FORMAT:
        raise BundleError("unsupported lab bundle schema")
    files, objects = manifest.get("files"), manifest.get("objects")
    if type(files) is not list or not 1 <= len(files) <= 128 or type(objects) is not dict:
        raise BundleError("invalid bundle file/object inventory")
    if set(objects) != {"world", "preparation", "suite"}:
        raise BundleError("invalid bundle object inventory")
    for kind, obj in objects.items():
        if type(obj) is not dict or obj.get("id") != identified(kind, {k: v for k, v in obj.items() if k != "id"})["id"]:
            raise BundleError(f"{kind} identity mismatch")
    if (
        objects["preparation"].get("world_id") != objects["world"]["id"]
        or objects["suite"].get("preparation_id") != objects["preparation"]["id"]
    ):
        raise BundleError("broken world/preparation/suite references")
    if manifest.get("bundle_id") != identified("bundle", {k: v["id"] for k, v in objects.items()})["id"]:
        raise BundleError("bundle identity mismatch")
    names, total = set(), 0
    for record in files:
        if type(record) is not dict or set(record) != {"path", "bytes", "sha256", "role", "profiles"}:
            raise BundleError("invalid file record")
        name = str(relative_path(record["path"]))
        if "__pycache__" in name.split("/") or name.endswith(".pyc"):
            raise BundleError("bytecode caches cannot be bundle files")
        if name in names or name == "manifest.json":
            raise BundleError("duplicate or self-referencing bundle file")
        names.add(name)
        if type(record["bytes"]) is not int or record["bytes"] < 0:
            raise BundleError("invalid file size")
        if type(record["profiles"]) is not list or not record["profiles"] or any(p not in PROFILES for p in record["profiles"]):
            raise BundleError("invalid download profile")
        total += record["bytes"]
    if total > MAX_TOTAL:
        raise BundleError("bundle exceeds supported byte limit")
    by_path = {r["path"]: r for r in files}
    world_files = objects["world"].get("files")
    if type(world_files) is not dict or not world_files or any(
        by_path.get(p, {}).get("sha256") != h or by_path[p]["role"] != "world" for p, h in world_files.items()
    ):
        raise BundleError("world identity does not bind its files")
    if {p for p, r in by_path.items() if r["role"] == "world"} != set(world_files):
        raise BundleError("unbound world file")
    for obj, key, name in (
        ("world", "declaration_sha256", "labworld.json"),
        ("preparation", "apparatus_sha256", "apparatus.json"),
        ("preparation", "file_sha256", "preparation.npz"),
        ("suite", "file_sha256", "suite.json"),
    ):
        if name not in by_path or objects[obj].get(key) != by_path[name]["sha256"]:
            raise BundleError("object identity does not bind its declared file")
    truths = objects["suite"].get("truth_sha256")
    if type(truths) is not dict or not truths or truths != {r["path"]: r["sha256"] for r in files if r["role"] == "truth"}:
        raise BundleError("suite identity does not bind every truth file")
    for name in (*SIMULATION_FILES, *world_files):
        if set(by_path[name]["profiles"]) != set(PROFILES):
            raise BundleError("simulation input missing from a profile")
    for name in ("suite.json", *truths):
        if by_path[name]["profiles"] != ["evaluation"]:
            raise BundleError("suite and truth require the evaluation profile")
    return manifest


class LabBundle(Bundle):
    """A verified lab-world bundle with the interface of ``bundles.Bundle``."""

    def _load(self, directory, *, profile):
        if profile not in PROFILES:
            raise BundleError("unknown bundle profile")
        self.root = Path(directory).expanduser().resolve()
        self.profile = profile
        # World code is imported from the bundle folder; never write bytecode caches into it.
        sys.dont_write_bytecode = True
        self.manifest = validate_lab_manifest(read_json(self.root / "manifest.json"))
        self.manifest_sha256 = digest(self.root / "manifest.json")
        self.files = {r["path"]: r for r in self.manifest["files"]}
        for record in self.files.values():
            if profile in record["profiles"]:
                self.verified_path(record["path"])
        self.declaration = read_json(self.verified_path("labworld.json"))
        self.apparatus = read_json(self.verified_path("apparatus.json"))
        self.genome = None
        self._validate_lab()
        self.roster = scoring.PublicRoster(n_ports=self.n_ports, device_slots=(13, 19), protocol=self.protocol)
        self.limits = replace(scoring.DEFAULT_LIMITS, max_horizon_tu=50.0)
        self.suite = None
        if profile == "evaluation":
            self.suite = read_json(self.verified_path("suite.json"))
            self._validate_suite()

    def _validate_lab(self):
        d, a = self.declaration, self.apparatus
        if type(d) is not dict or set(d) != DECLARATION_KEYS:
            raise BundleError("lab world declaration keys differ from the supported set")
        if d["name"] != self.manifest["objects"]["world"].get("name"):
            raise BundleError("declaration and world identity name differ")
        world_files = self.manifest["objects"]["world"]["files"]
        if "world/" + d["world_file"] not in world_files or any(
            "world/" + x["source"] not in world_files for x in d["derived"]
        ):
            raise BundleError("declared world sources are not bound world files")
        n_observed = len(d["fields"]) + len(d["derived"])
        if len(d["actuators"]) != n_observed:
            raise BundleError(
                "the scorer's public roster has one port count: declare as many actuators as observed ports"
            )
        self.n_ports = n_observed
        prep = self.manifest["objects"]["preparation"]
        if prep.get("noise_policy") != R6.NOISE_POLICY or prep.get("public_time") != 0:
            raise BundleError("unsupported preparation profile")
        if type(a) is not dict or set(a) != {"devices", "device_slots", "port_permutation", "adjustment_matrix", "protocol"}:
            raise BundleError("invalid reference apparatus")
        self.protocol = a["protocol"]
        if self.protocol != R6.APPARATUS_PROTOCOL or a["device_slots"] != [13, 19] or len(a["devices"]) != 2:
            raise BundleError("unsupported apparatus protocol or devices")
        perm = a["port_permutation"]
        if type(perm) is not list or sorted(perm) != list(range(n_observed)):
            raise BundleError("invalid port permutation")
        mix = np.asarray(a["adjustment_matrix"], dtype=float)
        if mix.shape != (3, 3) or not np.isfinite(mix).all():
            raise BundleError("invalid apparatus geometry")
        self.runner, self.scoring = R6, scoring

    def check_runtime(self):
        if self.manifest["objects"]["world"].get("implementation") != implementation_identity():
            raise BundleError("bundle requires a different lab simulator implementation")

    def state(self):
        return read_state(self.verified_path("preparation.npz"), self.declaration)

    def make_oracle(self, *, world=None, devices=None):
        self.check_runtime()
        for name in self.manifest["objects"]["world"]["files"]:
            self.verified_path(name)
        state = self.state()
        if state_digest({k: np.asarray(v) for k, v in state.items()}) != self.manifest["objects"]["preparation"]["state_sha256"]:
            raise BundleError("prepared state identity mismatch")
        return build_oracle(self.root, self.declaration, self.apparatus, state, world=world, devices=devices)

    def world(self):
        return world_from_declaration(self.declaration, self.root)


def manifest_schema(path) -> str | None:
    try:
        return read_json(Path(path) / "manifest.json").get("schema_version")
    except (BundleError, AttributeError):
        return None


def open_bundle(path, *, profile="evaluation"):
    """``LabBundle`` when the manifest declares a lab world, otherwise the Blobkit ``Bundle``."""
    if manifest_schema(path) == LAB_FORMAT:
        return LabBundle(path, profile=profile)
    return Bundle(path, profile=profile)

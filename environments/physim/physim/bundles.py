"""Validated data bundles for the reference CPU experiment and scoring profile."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import replace
from pathlib import Path, PurePosixPath

import numpy as np

from . import blobround6 as R6
from . import blobround6_eval as scoring

FORMAT = "physim-world-bundle-v1"
MAX_JSON = 2 * 1024 * 1024
MAX_FILE = 32 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024
MAX_EXPANDED = 64 * 1024 * 1024
PROFILES = ("simulation", "evaluation")
NUMERICS = {
    "backend": "blobkit-cpu",
    "dtype": "float32",
    "grid": [256, 256],
    "L": 128.0,
    "dx": 0.5,
    "dt": 0.02,
    "boundary": "periodic",
    "workers": 1,
}
REFERENCE_DEPENDENCIES = {"numpy": "2.5.2", "scipy": "1.18.0"}


class BundleError(ValueError):
    """Missing, unsupported, or invalid bundle data."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identified(kind, payload):
    return dict(payload, id=kind + ":sha256:" + hashlib.sha256(canonical(payload)).hexdigest())


def implementation_identity():
    from blobkit import __version__, genome
    from blobkit.soup import sim_cpu, sim_v1

    from . import devices

    return dict(
        blobkit_version=__version__,
        reference_dependencies=REFERENCE_DEPENDENCIES,
        source_sha256={m.__name__: digest(m.__file__) for m in (genome, sim_cpu, sim_v1, devices, R6)},
    )


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BundleError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise BundleError("non-finite JSON number")


def read_json(path, maximum=MAX_JSON):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise BundleError(f"expected a bounded regular JSON file: {path}")
    try:
        return json.loads(path.read_text(), object_pairs_hook=_unique, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise BundleError(f"invalid JSON in {path.name}: {exc}") from exc


def relative_path(value):
    if type(value) is not str or not value or len(value) > 240 or "\\" in value or ":" in value:
        raise BundleError("invalid bundle-relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")):
        raise BundleError("bundle paths must be canonical relative paths")
    return p


def safe_path(root, value):
    root = Path(root).resolve()
    p = root
    for part in relative_path(value).parts:
        p = p / part
        if p.is_symlink():
            raise BundleError("bundle symlinks are forbidden")
    if not p.resolve().is_relative_to(root):
        raise BundleError("bundle path escapes its directory")
    return p


def validate_manifest(manifest):
    if type(manifest) is not dict or manifest.get("schema_version") != FORMAT:
        raise BundleError("unsupported world-bundle schema")
    files, objects = manifest.get("files"), manifest.get("objects")
    if (
        type(files) is not list
        or not 1 <= len(files) <= 128
        or type(objects) is not dict
        or set(objects) != {"world", "preparation", "suite"}
    ):
        raise BundleError("invalid bundle file/object inventory")
    for kind, obj in objects.items():
        if type(obj) is not dict:
            raise BundleError("invalid object identity")
        payload = {k: v for k, v in obj.items() if k != "id"}
        if obj.get("id") != identified(kind, payload)["id"]:
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
        if name in names or name == "manifest.json":
            raise BundleError("duplicate or self-referencing bundle file")
        names.add(name)
        if type(record["bytes"]) is not int or not 0 < record["bytes"] <= MAX_FILE:
            raise BundleError("file size exceeds supported limits")
        if type(record["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise BundleError("invalid file hash")
        if (
            type(record["profiles"]) is not list
            or not record["profiles"]
            or any(p not in PROFILES for p in record["profiles"])
        ):
            raise BundleError("invalid download profile")
        if type(record["role"]) is not str:
            raise BundleError("invalid file role")
        total += record["bytes"]
    if total > MAX_TOTAL:
        raise BundleError("bundle exceeds supported byte limit")
    by_path = {r["path"]: r for r in files}
    for obj, key, name in (
        ("world", "genome_sha256", "world.json"),
        ("preparation", "apparatus_sha256", "apparatus.json"),
        ("preparation", "file_sha256", "preparation.npz"),
        ("suite", "file_sha256", "suite.json"),
    ):
        if name not in by_path or objects[obj].get(key) != by_path[name]["sha256"]:
            raise BundleError("object identity does not bind its declared file")
    truths = objects["suite"].get("truth_sha256")
    if (
        type(truths) is not dict
        or not truths
        or truths != {r["path"]: r["sha256"] for r in files if r["role"] == "truth"}
    ):
        raise BundleError("suite identity does not bind every truth file")
    for name in ("world.json", "apparatus.json", "preparation.npz"):
        if set(by_path[name]["profiles"]) != set(PROFILES):
            raise BundleError("simulation input missing from a profile")
    for name in ("suite.json", *truths):
        if by_path[name]["profiles"] != ["evaluation"]:
            raise BundleError("suite and truth require the evaluation profile")
    return manifest


def read_manifest(path):
    return validate_manifest(read_json(path))


def load_arrays(path, *, text_keys=()):
    """Check ZIP sizes and NPY headers before allowing NumPy allocations."""
    path = Path(path)
    if path.stat().st_size > MAX_FILE:
        raise BundleError("NPZ exceeds file byte limit")
    expected, expanded = {}, 0
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > 64:
                raise BundleError("NPZ array count exceeds cap")
            for entry in entries:
                if "/" in entry.filename or not entry.filename.endswith(".npy") or entry.flag_bits & 1:
                    raise BundleError("NPZ must contain unencrypted named arrays only")
                key = entry.filename[:-4]
                if key in expected:
                    raise BundleError("duplicate NPZ array")
                expanded += entry.file_size
                if expanded > MAX_EXPANDED:
                    raise BundleError("expanded NPZ exceeds byte cap")
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
                    if version not in readers:
                        raise BundleError("unsupported NPY format")
                    shape, _, dtype = readers[version](stream, max_header_size=16384)
                    header_bytes = stream.tell()
                if dtype.hasobject or dtype.kind not in ("US" if key in text_keys else "iuf"):
                    raise BundleError("unsupported NPZ array dtype")
                size = math.prod(shape) * dtype.itemsize
                if len(shape) > 6 or size > MAX_EXPANDED or size + header_bytes != entry.file_size:
                    raise BundleError("NPY shape/size mismatch or allocation cap exceeded")
                expected[key] = (shape, dtype)
        with np.load(path, allow_pickle=False) as values:
            arrays = {key: values[key].copy() for key in expected}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise BundleError(f"invalid NPZ {path.name}: {exc}") from exc
    for key, array in arrays.items():
        if key not in text_keys and not np.isfinite(array).all():
            raise BundleError("non-finite numerical array")
    return arrays


class Bundle:
    """Verified local worlds on the fixed CPU grid, with a genome-derived roster."""

    def __init__(self, directory, *, profile="evaluation"):
        try:
            self._load(directory, profile=profile)
        except BundleError:
            raise
        except (ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
            raise BundleError(f"invalid bundle structure: {exc}") from exc

    def _load(self, directory, *, profile):
        if profile not in PROFILES:
            raise BundleError("unknown bundle profile")
        self.root = Path(directory).expanduser().resolve()
        self.profile = profile
        self.manifest = read_manifest(self.root / "manifest.json")
        self.manifest_sha256 = digest(self.root / "manifest.json")
        self.files = {r["path"]: r for r in self.manifest["files"]}
        for record in self.files.values():
            if profile in record["profiles"]:
                self.verified_path(record["path"])
        self.genome = read_json(self.verified_path("world.json"))
        self.apparatus = read_json(self.verified_path("apparatus.json"))
        self._validate_physics()
        self.roster = scoring.PublicRoster(
            n_ports=len(self.genome["acts"]) + len(self.genome["chans"]), device_slots=(13, 19)
        )
        self.limits = replace(scoring.DEFAULT_LIMITS, max_horizon_tu=50.0)
        self.suite = None
        if profile == "evaluation":
            self.suite = read_json(self.verified_path("suite.json"))
            self._validate_suite()

    def verified_path(self, name):
        record = self.files.get(name)
        if record is None or self.profile not in record["profiles"]:
            raise BundleError(f"file is not in the requested profile: {name}")
        path = safe_path(self.root, name)
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise BundleError(f"missing file or byte-size mismatch: {name}")
        if digest(path) != record["sha256"]:
            raise BundleError(f"file hash mismatch: {name}")
        return path

    def _validate_physics(self):
        world, prep = self.manifest["objects"]["world"], self.manifest["objects"]["preparation"]
        if (
            world.get("numerics") != NUMERICS
            or prep.get("noise_policy") != R6.NOISE_POLICY
            or prep.get("public_time") != 0
        ):
            raise BundleError("unsupported numerical or preparation profile")
        g = self.genome
        if (
            type(g) is not dict
            or not isinstance(g.get("acts"), list)
            or not isinstance(g.get("chans"), list)
            or not 1 <= len(g["acts"]) <= 16
            or not 1 <= len(g["chans"]) <= 48
            or len(g["acts"]) + len(g["chans"]) > 64
        ):
            raise BundleError("unsupported activator/channel count")
        na, nc = len(g["acts"]), len(g["chans"])
        for name, shape in (("W", (nc, na)), ("K", (na, nc))):
            value = np.asarray(g.get(name), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise BundleError("invalid genome coupling matrix")
        if type(g.get("bilin", [])) is not list or len(g.get("bilin", [])) > 4096:
            raise BundleError("invalid bilinear coupling list")
        for records, required, positive in (
            (g["acts"], ("lam", "k1", "Du", "u0"), ("lam",)),
            (g["chans"], ("tau", "D", "thr", "sc"), ("tau", "sc")),
        ):
            for record in records:
                if type(record) is not dict:
                    raise BundleError("invalid genome component")
                for key in required:
                    value = record.get(key)
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or abs(value) > 1e6
                        or (key in positive and value <= 0)
                        or (key in ("D", "Du") and value < 0)
                    ):
                        raise BundleError("invalid genome coefficient")
        if any(c.get("g") not in ("id", "tanh") for c in g["chans"]):
            raise BundleError("unsupported channel activation")
        for b in g.get("bilin", []):
            if (
                type(b) is not list
                or len(b) != 4
                or any(type(i) is not int for i in b[:3])
                or not (0 <= b[0] < na and 0 <= b[1] < nc and 0 <= b[2] < nc)
                or type(b[3]) not in (int, float)
                or not math.isfinite(b[3])
            ):
                raise BundleError("invalid bilinear coupling")
        noise = prep.get("noise_coefficient")
        if type(noise) not in (int, float) or not math.isfinite(noise) or not 0 <= noise <= 1:
            raise BundleError("invalid noise coefficient")
        a = self.apparatus
        if (
            type(a) is not dict
            or set(a) != {"devices", "device_slots", "port_permutation", "adjustment_matrix", "emitter_yx"}
            or a["device_slots"] != [13, 19]
            or len(a["devices"]) != 2
        ):
            raise BundleError("invalid reference apparatus")
        perm = a["port_permutation"]
        if type(perm) is not list or any(type(x) is not int for x in perm) or sorted(perm) != list(range(na + nc)):
            raise BundleError("invalid port permutation")
        for key, shape in (("adjustment_matrix", (3, 3)), ("emitter_yx", (2,))):
            v = np.asarray(a[key], dtype=float)
            if v.shape != shape or not np.isfinite(v).all():
                raise BundleError("invalid apparatus geometry")
        for i, (record, slots) in enumerate(zip(a["devices"], a["device_slots"])):
            p = record.get("parameters", {})
            if set(p) != {
                "dev_id",
                "lattice",
                "n_rings",
                "base_ds",
                "center",
                "L",
                "secret_rot",
                "reflect",
                "motion_theta",
                "motion_reflect",
                "node_perm",
                "dil_bounds",
            }:
                raise BundleError("invalid device parameters")
            if p["dev_id"] != i or p["lattice"] != ("square", "hex")[i] or p["n_rings"] != 3 or p["L"] != 128.0:
                raise BundleError("unsupported probe geometry")
            if (
                type(p["node_perm"]) is not list
                or any(type(x) is not int for x in p["node_perm"])
                or sorted(p["node_perm"]) != list(range(slots))
            ):
                raise BundleError("invalid probe node permutation")
            for key in ("base_ds", "secret_rot", "motion_theta"):
                if type(p[key]) not in (int, float) or not math.isfinite(p[key]):
                    raise BundleError("invalid probe scalar")
            for value, shape in ((p["center"], (2,)), (p["dil_bounds"], (2,)), (record.get("motion_basis"), (2, 2))):
                v = np.asarray(value, dtype=float)
                if v.shape != shape or not np.isfinite(v).all():
                    raise BundleError("invalid probe geometry")
            if (
                not 0 < p["base_ds"] <= 128
                or not 0 < p["dil_bounds"][0] <= record.get("dilation", 0) <= p["dil_bounds"][1] <= 16
            ):
                raise BundleError("invalid probe scale")

    def _validate_suite(self):
        suite = self.suite
        if self.manifest["objects"]["suite"].get("scoring_source_sha256") != digest(scoring.__file__):
            raise BundleError("bundle requires a different scoring implementation")
        if (
            type(suite) is not dict
            or suite.get("schema_version") != "physim-suite-v1"
            or suite.get("scoring_version") != scoring.VERSION
        ):
            raise BundleError("unsupported evaluation suite")
        cases = suite.get("cases")
        if type(cases) is not list or not 1 <= len(cases) <= 128:
            raise BundleError("invalid suite case count")
        if (
            suite.get("truth_members") != 2
            or suite.get("forecast_members") != 64
            or suite.get("aggregation") != "equal-case-mean"
        ):
            raise BundleError("unsupported suite sampling or aggregation")
        ids = set()
        truths = set()
        for record in cases:
            case = record["request"]
            if type(case.get("id")) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case["id"]):
                raise BundleError("case IDs must be bounded filename-safe identifiers")
            if case["id"] in ids:
                raise BundleError("duplicate suite case")
            ids.add(case["id"])
            scoring.validate_plan(
                case, groups=record["groups"], n_samples=64, n_truth=2, roster=self.roster, limits=self.limits
            )
            if record["truth"] not in self.files or self.files[record["truth"]]["role"] != "truth":
                raise BundleError("case truth is not in the file manifest")
            truths.add(record["truth"])
        if truths != set(self.manifest["objects"]["suite"]["truth_sha256"]):
            raise BundleError("suite cases and truth inventory differ")

    def check_runtime(self):
        from importlib.metadata import version

        declared = self.manifest["objects"]["world"].get("implementation")
        if declared != implementation_identity():
            raise BundleError("bundle requires a different simulator implementation")
        if any(version(name) != expected for name, expected in REFERENCE_DEPENDENCIES.items()):
            raise BundleError("native reference simulation requires physim[reference] dependency versions")

    def make_oracle(self):
        from blobkit.soup import sim_cpu

        from .devices import ProbeDevice, step_chunk

        self.check_runtime()
        arrays = load_arrays(self.verified_path("preparation.npz"))
        if set(arrays) != {"fields"}:
            raise BundleError("preparation requires exactly one fields array")
        fields = arrays["fields"]
        prep = self.manifest["objects"]["preparation"]
        if (
            fields.shape != (self.roster.n_ports, 256, 256)
            or fields.dtype != np.dtype("float32")
            or hashlib.sha256(fields.tobytes()).hexdigest() != prep.get("field_sha256")
        ):
            raise BundleError("prepared field identity, dtype, or shape mismatch")
        state = sim_cpu.init_soup(
            self.genome, L=128.0, seed=0, n_soup=0, dtype="f32", noise=prep["noise_coefficient"], workers=1
        )
        state["F"], state["t_step"] = fields, 0
        devices = []
        for record in self.apparatus["devices"]:
            device = ProbeDevice(**record["parameters"])
            device.dilation = record["dilation"]
            device.Bm = np.asarray(record["motion_basis"], dtype=float)
            devices.append(device)
        return R6.OracleRunner(
            _template=state,
            _devices=devices,
            _port_perm=self.apparatus["port_permutation"],
            _adjust_mix=self.apparatus["adjustment_matrix"],
            _emitter_yx=self.apparatus["emitter_yx"],
            _stepper=step_chunk,
        )

    def truth(self, record):
        arrays = load_arrays(self.verified_path(record["truth"]), text_keys=("request",))
        case = record["request"]
        if set(arrays) != {"request", *[f"query{i}" for i in range(len(case["queries"]))]}:
            raise BundleError("truth arrays do not match their request")
        if arrays["request"].ndim != 0 or json.loads(str(arrays["request"])) != case:
            raise BundleError("truth request identity mismatch")
        samples = [arrays[f"query{i}"] for i in range(len(case["queries"]))]
        shapes = scoring.validate_case(case, roster=self.roster, limits=self.limits)
        scoring.validate_samples({"samples": samples}, shapes, self.suite["truth_members"])
        return {"samples": samples}

    def references(self):
        return dict(
            bundle=self.manifest["bundle_id"],
            manifest_sha256=self.manifest_sha256,
            **{kind: value["id"] for kind, value in self.manifest["objects"].items()},
        )

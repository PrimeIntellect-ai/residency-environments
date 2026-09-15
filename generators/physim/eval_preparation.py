"""Native preparation and scientific observations shared by world-specific recipes."""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
from blobkit.soup import sim_cpu
from physim import blobround6 as R6
from physim.bundles import digest, implementation_identity
from physim.devices import ProbeDevice, step_chunk


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(name, destination, source):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    genome = json.loads((source / "genome.json").read_text())
    with np.load(source / "final_state.npz", allow_pickle=False) as data:
        fields = data["F"].copy()
    y, x = np.unravel_index(np.argmax(fields[0]), fields[0].shape)
    center = (np.array([y, x]) + 0.5) * 0.5
    rng = np.random.default_rng(41001 if name == "bf" else 41002)
    permutation = [2, 0, 3, 1] if name == "bf" else [4, 1, 3, 0, 5, 2]
    devices = []
    for i, (lattice, spacing, slots) in enumerate((("square", 3.0, 13), ("hex", 6.0, 19))):
        device = ProbeDevice(i, lattice, 3, spacing, center, 128.0, 0.0, False, 0.0, False, rng.permutation(slots))
        parameters = dict(
            dev_id=i,
            lattice=lattice,
            n_rings=3,
            base_ds=spacing,
            center=center.tolist(),
            L=128.0,
            secret_rot=0.0,
            reflect=False,
            motion_theta=0.0,
            motion_reflect=False,
            node_perm=device.node_perm.tolist(),
            dil_bounds=list(device.dil_bounds),
        )
        devices.append(dict(parameters=parameters, dilation=1.0, motion_basis=device.Bm.tolist()))
    apparatus = dict(
        devices=devices,
        device_slots=[13, 19],
        port_permutation=permutation,
        adjustment_matrix=np.eye(3).tolist(),
        emitter_yx=((center + [0, 6]) % 128).tolist(),
    )
    write_json(destination / "world.json", genome)
    write_json(destination / "apparatus.json", apparatus)
    np.savez_compressed(destination / "preparation.npz", fields=fields)
    write_json(
        destination / "origin.json",
        dict(
            world=name,
            source=str(source),
            original_time=2500,
            source_fields_sha256=digest(source / "final_state.npz"),
            recipe_sha256=digest(__file__),
            implementation=implementation_identity(),
            noise_policy=R6.NOISE_POLICY,
            noise_coefficient=0.002,
            selection="First independently simulated preparation; apparatus centered by fixed activator-0 maximum rule.",
        ),
    )


def make_oracle(preparation, capture=None):
    root = Path(preparation)
    genome = json.loads((root / "world.json").read_text())
    apparatus = json.loads((root / "apparatus.json").read_text())
    state = sim_cpu.init_soup(genome, L=128, n_soup=0, seed=0, workers=1, noise=0.002)
    with np.load(root / "preparation.npz", allow_pickle=False) as data:
        state["F"] = data["fields"].copy()
    devices = []
    for record in apparatus["devices"]:
        device = ProbeDevice(**record["parameters"])
        device.dilation = record["dilation"]
        device.Bm = np.array(record["motion_basis"])
        devices.append(device)
    previous = -1
    member = 0

    def step(sim, count, injections=None):
        nonlocal previous, member
        if sim["t_step"] < previous:
            member += 1
        step_chunk(sim, count, injections=injections)
        previous = sim["t_step"]
        if capture is not None and member == 0:
            capture.append((sim["t_step"] * sim["dt"], sim["F"].copy()))

    return R6.OracleRunner(
        _template=state,
        _devices=devices,
        _port_perm=apparatus["port_permutation"],
        _adjust_mix=apparatus["adjustment_matrix"],
        _emitter_yx=apparatus["emitter_yx"],
        _stepper=step if capture is not None else step_chunk,
    )


def observe(preparation, record, output, members, seed):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    oracle = make_oracle(preparation, frames)
    initial = oracle._template["F"].copy()
    prediction = oracle.sample_truth(record["actions"], record["queries"], n_samples=members, truth_seed=seed)
    np.savez_compressed(
        output / "observations.npz",
        request=json.dumps(record),
        **{f"query{i}": a for i, a in enumerate(prediction["samples"])},
    )
    np.savez_compressed(
        output / "fields.npz", t=np.array([0] + [t for t, _ in frames]), F=np.stack([initial] + [f for _, f in frames])
    )
    receipt = dict(
        request=record,
        members=members,
        truth_seed=seed,
        seconds=time.monotonic() - started,
        observations_sha256=digest(output / "observations.npz"),
        fields_sha256=digest(output / "fields.npz"),
        preparation_sha256=digest(Path(preparation) / "preparation.npz"),
        implementation=implementation_identity(),
        recipe_sha256=digest(__file__),
    )
    write_json(output / "receipt.json", receipt)
    print(f"{record['id']}: {members} independent native trajectories, {receipt['seconds']:.1f}s", flush=True)


def bf_programs():
    times = [0, 2, 5, 8, 10, 12, 15, 20, 22, 25, 30, 35, 40, 45, 50]

    def pulse(port, amplitude, start=0):
        return dict(t=start, kind="inject", port=port, amp=amplitude, dur=5)

    arms = {
        "sham": [],
        "act_weak": [pulse(1, 0.05)],
        "act_strong": [pulse(1, 0.3)],
        "trail_low": [pulse(2, 0.002)],
        "trail_mid": [pulse(2, 0.01)],
        "trail_high": [pulse(2, 0.05)],
        "act_then_trail": [pulse(1, 0.3), pulse(2, 0.01, 10)],
        "act_delayed": [pulse(1, 0.3, 20)],
        "fast_inhibitor": [pulse(0, 0.3)],
    }
    programs = [
        dict(
            id=name,
            actions=actions,
            queries=[dict(sensor=sensor, t=times) for sensor in ("device0", "device1", "global")],
        )
        for name, actions in arms.items()
    ]
    return with_pose_programs(programs)


def with_pose_programs(programs):
    for label, device, u in (("probe_shift", 0, [1, 0, 0]), ("probe_wide", 1, [0, 0, 1])):
        programs.append(
            dict(id=label, actions=[dict(t=0, kind="adjust", device=device, u=u)], queries=programs[0]["queries"])
        )
    return programs


def xv_programs():
    times = [0, 2, 5, 8, 10, 12, 15, 20, 22, 25, 30, 35, 40, 45, 50]

    def pulse(port, amplitude=0.3, start=0):
        return dict(t=start, kind="inject", port=port, amp=amplitude, dur=5)

    arms = {
        "sham": [],
        "act_weak": [pulse(3, 0.05)],
        "act_strong": [pulse(3)],
        "partner_pulse": [pulse(1)],
        "slow_feedback": [pulse(0)],
        "cross_feedback": [pulse(5)],
        "act_delayed": [pulse(3, start=20)],
        "partner_delayed": [pulse(1, start=20)],
        "act_then_partner": [pulse(3), pulse(1, start=10)],
    }
    return with_pose_programs(
        [
            dict(id=name, actions=actions, queries=[dict(sensor=s, t=times) for s in ("device0", "device1", "global")])
            for name, actions in arms.items()
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", choices=["bf", "xv"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, required=True, help="Directory containing the preserved genome.json and final_state.npz"
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    prepare(args.world, args.output / "preparation", source=args.source)
    programs = bf_programs() if args.world == "bf" else xv_programs()
    write_json(args.output / "programs.json", programs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(
                observe, args.output / "preparation", record, args.output / "science" / record["id"], 3, 51000 + i
            )
            for i, record in enumerate(programs)
        ]
        for job in jobs:
            job.result()

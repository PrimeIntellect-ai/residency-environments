"""Compare fresh XV pair motion with and without cross-drive source terms."""

import argparse
from pathlib import Path

import numpy as np
from blobkit import genome
from eval_preparation import make_oracle, write_json
from physim.bundles import digest


def run(preparation, output, horizon=50):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        control="Zero only W[0,1] and W[1,0]; preserve the original prepared fields and pair future noise.",
        scope="Privileged mechanism check; transient motion can persist after coupling is removed.",
        members=3,
        horizon_tu=horizon,
        truth_seed=53001,
        source_sha256=digest(__file__),
        preparation_sha256=digest(Path(preparation) / "preparation.npz"),
        arms={},
    )
    queries = [dict(sensor=s, t=list(range(0, horizon + 1, 5))) for s in ("device0", "device1", "global")]
    for label in ("coupled", "cross_drive_removed"):
        frames = []
        oracle = make_oracle(preparation, frames)
        initial = oracle._template["F"].copy()
        if label == "cross_drive_removed":
            for key in ("Wf", "Wid"):
                matrix = oracle._template[key].copy()
                matrix[0, 1] = matrix[1, 0] = 0
                oracle._template[key] = matrix
        readings = oracle.sample_truth([], queries, n_samples=3, truth_seed=53001)
        fields = np.stack([initial] + [f for _, f in frames])
        times = np.array([0] + [t for t, _ in frames])
        centers = []
        for field in fields:
            pair = []
            for act in (0, 1):
                y, x = np.where(field[act] > 0.5)
                if not len(y):
                    raise ValueError("An activator lost its positive core")
                pair.append((np.array([y.mean(), x.mean()]) + 0.5) * 0.5)
            centers.append(pair)
        centers = np.array(centers)
        displacement = genome.min_image(centers[:, 0] - centers[:, 1], 128)
        angle = np.unwrap(np.arctan2(displacement[:, 0], displacement[:, 1]))
        np.savez_compressed(
            output / (label + ".npz"),
            t=times,
            F=fields,
            centers=centers,
            **{f"query{i}": a for i, a in enumerate(readings["samples"])},
        )
        result["arms"][label] = dict(
            times=times.tolist(),
            angle_change_rad=(angle - angle[0]).tolist(),
            separation=np.linalg.norm(displacement, axis=1).tolist(),
            net_turn_rad=float(angle[-1] - angle[0]),
            note="Full-field geometry is the first of three fresh realizations; all three sensor trajectories are saved.",
        )
    write_json(output / "result.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=int, choices=[50, 250], default=50)
    args = parser.parse_args()
    run(args.preparation, args.output, args.horizon)

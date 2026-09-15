"""Causal control: does a legal trail pulse require BF's bilinear feedback?"""

import argparse
from pathlib import Path

import numpy as np
from eval_preparation import make_oracle, write_json
from physim.bundles import digest


def run(preparation, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    queries = [
        dict(sensor="device0", t=[0, 2, 5, 10, 20, 30, 40, 50]),
        dict(sensor="device1", t=[0, 2, 5, 10, 20, 30, 40, 50]),
    ]
    arrays = {}
    for coupled in (True, False):
        for pulse in (False, True):
            oracle = make_oracle(preparation)
            if not coupled:
                oracle._template["bilin"] = []
            actions = [dict(t=0, kind="inject", port=2, amp=0.05, dur=5)] if pulse else []
            key = f"{'coupled' if coupled else 'feedback_removed'}_{'pulse' if pulse else 'sham'}"
            arrays[key] = oracle.sample_truth(actions, queries, n_samples=3, truth_seed=53001)["samples"][0]
            print(key, flush=True)
    np.savez_compressed(output / "readings.npz", **arrays)
    effect = lambda a, b: np.sqrt(np.mean((arrays[a][:, -1, 1] - arrays[b][:, -1, 1]) ** 2, axis=-1))
    result = dict(
        coupled_effect_rms=effect("coupled_pulse", "coupled_sham").tolist(),
        feedback_removed_effect_rms=effect("feedback_removed_pulse", "feedback_removed_sham").tolist(),
        removed_feedback_activator_identical=bool(
            np.array_equal(arrays["feedback_removed_pulse"][:, :, 1], arrays["feedback_removed_sham"][:, :, 1])
        ),
        queries=queries,
        members=3,
        truth_seed=53001,
        control="Disable only the bilinear reaction contribution; identical prepared fields and paired future noise within each comparison.",
        scope="Privileged mechanism control. The original trail pulse itself is an admitted agent action.",
        source_sha256=digest(__file__),
        preparation_sha256=digest(Path(preparation) / "preparation.npz"),
    )
    write_json(output / "result.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.preparation, args.output)

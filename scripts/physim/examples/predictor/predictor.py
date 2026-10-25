"""A deterministic zero baseline for the Physim prediction interface.

This illustrates the 12-port reference world's array layout. Other preparations
must use the port count in their public contract. It neither simulates nor learns the world.
Run: python predictor.py request.json
Requires NumPy. Requests are assumed to satisfy the documented public contract.
"""

import json
import sys
from pathlib import Path

import numpy as np

SLOTS = {"device0": 13, "device1": 19, "global": 2}


def predict(actions, queries, n_samples=64, seed=0):
    """Return a complete, finite array for every query, preserving query order."""
    return {"samples": [np.zeros((n_samples, len(query["t"]), 12, SLOTS[query["sensor"]])) for query in queries]}


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text())
    result = predict(**request)
    print(
        json.dumps(
            {
                "shapes": [list(array.shape) for array in result["samples"]],
                "all_finite": all(bool(np.isfinite(array).all()) for array in result["samples"]),
            },
            indent=2,
        )
    )

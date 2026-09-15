"""Public, trusted I/O wrapper running INSIDE the restricted Linux container.

The operating-system/container boundary isolates submitted code. This Python
wrapper only supplies resource limits, JSON conversion and the callable API.
"""

import importlib.util
import json
import resource
import sys
from pathlib import Path

resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024, 20 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
request = json.load(sys.stdin)
if request["mode"] == "python":
    exec(compile(request["code"], "<agent-python>", "exec"), {"__name__": "__main__"})
elif request["mode"] == "predict":
    # Only the public roster, actions, queries, sampling seed and sample count cross this boundary.
    # Case IDs, scoring selectors and realized truths never enter the container.
    import numpy as np

    spec = importlib.util.spec_from_file_location("submitted_predictor", "/workspace/predictor.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, "/workspace")
    spec.loader.exec_module(module)
    result = module.predict(
        request["actions"], request["queries"], n_samples=request["n_samples"], seed=request["seed"]
    )
    if type(result) is not dict or set(result) != {"samples"} or type(result["samples"]) is not list:
        raise ValueError("predict must return exactly {'samples': [array per query]}")
    arrays = []
    slots = {"device0": 13, "device1": 19, "global": 2}
    if len(result["samples"]) != len(request["queries"]):
        raise ValueError("wrong number of query arrays")
    for query, value in zip(request["queries"], result["samples"]):
        array = np.asarray(value)
        expected = (request["n_samples"], len(query["t"]), request["n_ports"], slots[query["sensor"]])
        if array.shape != expected or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError(f"array must have finite numeric shape {expected}, got {array.shape}")
        arrays.append(array.tolist())
    Path("/output/prediction.json").write_text(json.dumps(dict(samples=arrays), allow_nan=False, separators=(",", ":")))
    print("Prediction written.")
else:
    raise ValueError("unknown operation")

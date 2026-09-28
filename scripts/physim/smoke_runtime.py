"""Scripted-provider integration: real experiment, frozen submission, grading and isolation.

No paid model requests. Prime mode uses paid CPU sandboxes; Docker mode is local.
"""

import argparse
import json
import os
import subprocess
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from physim.runtime_sandbox import PUBLIC_IMAGE
from physim.taskset import R6Config, R6Taskset, required_bundle

PREDICTOR = """from pathlib import Path
import json
import numpy as np
def predict(actions, queries, n_samples=64, seed=0):
    with np.load(sorted(Path('/observations').glob('*.npz'))[-1]) as data:
        request=json.loads(data['request'].item())
        initial={q['sensor']: data[f'query{i}'][0,0].copy() for i,q in enumerate(request['queries'])}
    return {'samples':[np.broadcast_to(initial[q['sensor']],(n_samples,len(q['t']),*initial[q['sensor']].shape)).copy() for q in queries]}
"""

# Printed results are booleans only: never dump environment values, process
# arguments, framework connection credentials, or arbitrary runtime files.
BOUNDARY_PROBE = """import importlib.util, json, os, socket, urllib.request
from pathlib import Path
checks = {}
checks['private_packages_absent'] = all(importlib.util.find_spec(x) is None for x in ['physim', 'blobkit'])
checks['private_mounts_absent'] = not any(Path(x).exists() for x in ['/var/run/docker.sock', '/Users', '/host_mnt', '/registry'])
checks['provider_credentials_absent'] = not any(os.environ.get(x) for x in ['PRIME_API_KEY', 'OPENAI_API_KEY', 'HF_TOKEN', 'ANTHROPIC_API_KEY'])
for name, url in [('hf', 'https://huggingface.co/'), ('github', 'https://github.com/'), ('host_preview', 'http://vf.host.internal:8765/')]:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            checks[name + '_blocked'] = response.status == 403
    except Exception:
        checks[name + '_blocked'] = True
try:
    connection = socket.create_connection(('1.1.1.1', 443), timeout=2)
    connection.close()
    checks['direct_network_blocked'] = False
except OSError:
    checks['direct_network_blocked'] = True
print(json.dumps({'boundary_audit': checks}))
assert all(checks.values())
"""


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    steps = [
        ("bash", {"command": "python - <<'PY'\n" + BOUNDARY_PROBE + "PY\n"}),
        (
            "laboratory_experiment",
            {"actions": [], "queries": [{"sensor": s, "t": [0]} for s in ["device0", "device1", "global"]]},
        ),
        ("bash", {"command": "cat > /workspace/predictor.py <<'PY'\n" + PREDICTOR + "PY\n"}),
        ("laboratory_validate", {}),
        ("laboratory_submit", {}),
    ]
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = sum(len(m.get("tool_calls") or []) for m in body["messages"] if m["role"] == "assistant")
            calls.append(body)
            if index < len(steps):
                name, arguments = steps[index]
                names = [t["function"]["name"] for t in body.get("tools", [])]
                assert name in names, (name, names)
                message = dict(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        dict(
                            id=f"call_{index}",
                            type="function",
                            function=dict(name=name, arguments=json.dumps(arguments)),
                        )
                    ],
                )
                reason = "tool_calls"
            else:
                message, reason = dict(role="assistant", content="Submitted."), "stop"
            response = dict(
                id=f"offline-{index}",
                object="chat.completion",
                created=0,
                model=body["model"],
                choices=[dict(index=0, message=message, finish_reason=reason)],
                usage=dict(prompt_tokens=10, completion_tokens=10, total_tokens=20, cost=0.0),
            )
            if body.get("stream"):
                delta = dict(message)
                if delta.get("tool_calls"):
                    delta["tool_calls"] = [dict(call, index=i) for i, call in enumerate(delta["tool_calls"])]
                first = dict(
                    response, object="chat.completion.chunk", choices=[dict(index=0, delta=delta, finish_reason=None)]
                )
                final = dict(
                    response, object="chat.completion.chunk", choices=[dict(index=0, delta={}, finish_reason=reason)]
                )
                encoded = (
                    "data: " + json.dumps(first) + "\n\ndata: " + json.dumps(final) + "\n\ndata: [DONE]\n\n"
                ).encode()
                content_type = "text/event-stream"
            else:
                encoded = json.dumps(response).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    runtime = dict(type=args.runtime, image=args.image, allow=[], workdir="/workspace", cpu=2, memory=4)
    config = dict(
        model="offline/scripted",
        num_rollouts=args.rollouts,
        max_concurrent=args.max_concurrent,
        push=False,
        output_dir=str(root / "runs"),
        client=dict(
            type="eval", base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key_var="PHYSIM_OFFLINE_SMOKE_KEY"
        ),
        env=dict(
            taskset=dict(
                id="physim",
                task=dict(
                    output_root=str(root / "artifacts"),
                    agent_image=args.image,
                    tools=dict(bundle=str(args.bundle.resolve()) if args.bundle else None, predictor_runtime=runtime),
                ),
            ),
            agent=dict(harness=dict(id="bash", edit=True, search=False), runtime=runtime, max_turns=8),
            retries=dict(max_retries=0),
        ),
    )
    tasks = list(R6Taskset(R6Config.model_validate(config["env"]["taskset"])))
    bundles = {task.data.bundle_id: required_bundle(task.config.tools) for task in tasks}
    (root / "eval.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        with (root / "eval.log").open("w") as log:
            result = subprocess.run(
                ["uv", "run", "--no-sync", "vf-eval", "@", str(root / "eval.json")],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PHYSIM_OFFLINE_SMOKE_KEY="offline-test"),
            )
    finally:
        server.shutdown()
        server.server_close()
    (root / "provider-requests.json").write_text(json.dumps(calls, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"Smoke failed; inspect {root / 'eval.log'}")
    episodes = [
        json.loads(line) for path in (root / "runs").rglob("traces.jsonl") for line in path.read_text().splitlines()
    ]
    assert len(episodes) == len(tasks) * args.rollouts, len(episodes)
    counts = Counter()
    energies = []
    for episode in episodes:
        trace = episode["traces"][-1]
        assert trace["ok"], trace.get("errors")
        info = trace["info"]["r6"]
        bundle = bundles[info["references"]["bundle"]]
        assert info["references"] == bundle.references()
        assert info["world_name"] == bundle.manifest["objects"]["world"]["name"]
        counts[info["references"]["bundle"]] += 1
        expected = json.loads(bundle.verified_path("checks.json").read_text())["reference"]["primary_joint_energy"]
        assert info["grade"]["status"] == "COMPLETE", info
        assert abs(info["primary_joint_energy"] - expected) < 1e-10, info["primary_joint_energy"]
        energies.append(info["primary_joint_energy"])
    assert counts == {identifier: args.rollouts for identifier in bundles}, counts
    for request in calls:
        messages = json.dumps(request["messages"])
        assert not any(identifier in messages for identifier in bundles)
        assert "bundles/" not in messages
    audits = [
        m.get("content", "")
        for request in calls
        for m in request["messages"]
        if m["role"] == "tool" and '"boundary_audit"' in m.get("content", "")
    ]
    assert audits and all("false" not in audit.lower() for audit in audits), audits
    report = dict(
        ok=True,
        runtime=args.runtime,
        tasks=len(tasks),
        rollouts=len(episodes),
        rollouts_per_task=args.rollouts,
        model_calls=len(calls),
        inference_cost_usd=0,
        energies=energies,
        references={identifier: bundle.references() for identifier, bundle in bundles.items()},
        counts=dict(counts),
    )
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", choices=["docker", "prime"], default="docker")
    parser.add_argument(
        "--image", default=PUBLIC_IMAGE, help="optional image with the pinned public packages preinstalled"
    )
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=2)
    main(parser.parse_args())

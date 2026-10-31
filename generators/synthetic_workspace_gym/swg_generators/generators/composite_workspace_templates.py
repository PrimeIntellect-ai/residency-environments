from textwrap import dedent

CONTRACT_SOURCE = dedent("""\
import json
from pathlib import Path

def resolve_contract(root: Path) -> dict[str, object]:
    active = json.loads((root / "release/active_bundle.json").read_text(encoding="utf-8"))
    policy_path = root / str(active["policy_path"])
    contract = json.loads(policy_path.read_text(encoding="utf-8"))["pipeline_contract"]
    output = root / "config/resolved_contract.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return contract
""")
NORMALIZE_SOURCE = dedent("""\
from datetime import datetime

def timestamp(value):
    return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))

def normalize_records(payload, contract):
    aliases = {str(k).strip().casefold(): str(v) for k, v in contract["team_aliases"].items()}
    active_values = {str(v).strip().casefold() for v in contract["active_values"]}
    latest = {}
    for raw in payload.get(str(contract["collection_field"]), []):
        if not isinstance(raw, dict):
            continue
        try:
            event_id = str(raw[contract["identity_field"]]).strip()
            updated_at = str(raw[contract["updated_at_field"]]).strip()
            team_raw = str(raw[contract["team_field"]]).strip()
            hours = float(str(raw[contract["hours_field"]]).strip())
            active = str(raw.get(contract["active_field"], "")).strip().casefold()
            timestamp(updated_at)
        except (KeyError, TypeError, ValueError):
            continue
        if not event_id or not team_raw or active not in active_values:
            continue
        record = {"event_id": event_id, "updated_at": updated_at, "team": aliases.get(team_raw.casefold(), team_raw.title()), "hours": hours}
        prior = latest.get(event_id)
        if prior is None or timestamp(updated_at) > timestamp(prior["updated_at"]):
            latest[event_id] = record
    return list(latest.values())
""")
AGGREGATE_SOURCE = dedent("""\
def aggregate(records):
    totals = {}
    for record in records:
        bucket = totals.setdefault(str(record["team"]), {"job_count": 0, "total_hours": 0.0})
        bucket["job_count"] += 1
        bucket["total_hours"] += float(record["hours"])
    return [{"team": team, "job_count": values["job_count"], "total_hours": round(values["total_hours"], 2)} for team, values in sorted(totals.items(), key=lambda item: item[0].casefold())]
""")
RUNNER_SOURCE = dedent("""\
import json
from pathlib import Path
from src.aggregate import aggregate
from src.contract import resolve_contract
from src.normalize import normalize_records
ROOT = Path(__file__).resolve().parent

def run():
    contract = resolve_contract(ROOT)
    payload = json.loads((ROOT / "data/jobs.json").read_text(encoding="utf-8"))
    normalized = normalize_records(payload, contract)
    path = ROOT / "artifacts/normalized_jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    persisted = json.loads(path.read_text(encoding="utf-8"))
    summary = aggregate(persisted)
    (ROOT / "artifacts/summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return summary
if __name__ == "__main__":
    run()
""")
PUBLIC_CHECK = dedent("""\
import json, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parent
done = subprocess.run([sys.executable, "run_pipeline.py"], cwd=root, capture_output=True, text=True)
if done.returncode:
    raise SystemExit(done.stderr or "pipeline execution failed")
for relative in ("config/resolved_contract.json", "artifacts/normalized_jobs.json", "artifacts/summary.json"):
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {relative}")
    json.loads(path.read_text(encoding="utf-8"))
summary = json.loads((root / "artifacts/summary.json").read_text(encoding="utf-8"))
if not isinstance(summary, list) or not summary or not all(isinstance(row, dict) and set(row) == {"team", "job_count", "total_hours"} for row in summary):
    raise SystemExit("wrong summary shape")
print("public smoke check passed")
""")
HIDDEN_RUNNER = dedent("""\
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
source = Path(sys.argv[1]).resolve()
def execute(records, contract=None):
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "workspace"
        shutil.copytree(source, root)
        (root / "data/jobs.json").write_text(json.dumps({"jobs": records}), encoding="utf-8")
        if contract:
            policy = root / "policies/hidden/contract.json"
            policy.parent.mkdir(parents=True, exist_ok=True)
            policy.write_text(json.dumps({"pipeline_contract": contract}), encoding="utf-8")
            (root / "release/active_bundle.json").write_text(json.dumps({"policy_path": "policies/hidden/contract.json"}), encoding="utf-8")
        done = subprocess.run([sys.executable, "run_pipeline.py"], cwd=root, capture_output=True, text=True, timeout=15)
        def load(path):
            try: return json.loads((root / path).read_text(encoding="utf-8"))
            except Exception: return None
        return done.returncode, load("config/resolved_contract.json"), load("artifacts/normalized_jobs.json"), load("artifacts/summary.json")
base = {"collection_field":"jobs","identity_field":"event_id","updated_at_field":"updated_at","team_field":"team_name","hours_field":"duration_hours","active_field":"enabled","active_values":["yes","true","1"],"team_aliases":{"platform":"Core","core":"Core"}}
records = [{"event_id":"a","updated_at":"2026-01-01T00:00:00Z","team_name":" platform ","duration_hours":"1.25","enabled":"YES"},{"event_id":"b","updated_at":"2026-01-01T00:00:00Z","team_name":"Beta","duration_hours":2.5,"enabled":1},{"event_id":"b","updated_at":"2026-01-02T00:00:00Z","team_name":"Beta","duration_hours":3.75,"enabled":"true"},{"event_id":"c","updated_at":"2026-01-02T00:00:00Z","team_name":"Core","duration_hours":9,"enabled":"no"},{"event_id":"bad","updated_at":"bad","team_name":"Core","duration_hours":"oops","enabled":"yes"},"bad"]
code, resolved, normalized, summary = execute(records)
alt = {"collection_field":"jobs","identity_field":"id","updated_at_field":"changed","team_field":"group","hours_field":"effort","active_field":"live","active_values":["on"],"team_aliases":{"eng":"Engineering"}}
acode, aresolved, _, asummary = execute([{"id":"x","changed":"2026-02-01T00:00:00Z","group":" ENG ","effort":"2.25","live":"ON"}], alt)
nl = normalized if isinstance(normalized, list) else []
sl = summary if isinstance(summary, list) else []
caps = {
 "execution": float(code == 0 and isinstance(summary, list)),
 "contract_materialization": float(resolved == base),
 "alias_resolution": float(any(isinstance(r, dict) and r.get("team") == "Core" for r in nl)),
 "active_filtering": float({r.get("event_id") for r in nl if isinstance(r, dict)} >= {"a", "b"} and not any(isinstance(r, dict) and r.get("event_id") == "c" for r in nl)),
 "fractional_aggregation": float(any(isinstance(r, dict) and r.get("team") == "Core" and r.get("total_hours") == 1.25 for r in sl)),
 "duplicate_resolution": float(any(isinstance(r, dict) and r.get("event_id") == "b" and r.get("hours") == 3.75 for r in nl) and sum(isinstance(r, dict) and r.get("event_id") == "b" for r in nl) == 1),
 "malformed_record_handling": float(code == 0 and len(nl) == 2 and not any(isinstance(r, dict) and r.get("event_id") == "bad" for r in nl)),
 "output_schema": float(bool(sl) and all(isinstance(r, dict) and set(r) == {"team","job_count","total_hours"} for r in sl)),
 "deterministic_ordering": float(bool(sl) and [r.get("team") for r in sl if isinstance(r, dict)] == sorted([r.get("team") for r in sl if isinstance(r, dict)], key=str.casefold)),
 "intermediate_artifact_consumption": float("summary = aggregate(persisted)" in (source / "run_pipeline.py").read_text(encoding="utf-8")),
 "alternate_contract_generalization": float(acode == 0 and aresolved == alt and asummary == [{"team":"Engineering","job_count":1,"total_hours":2.25}]),
}
subscores = {f"capability_{name}": value for name, value in caps.items()}
subscores.update({"tests_passed": sum(caps.values()), "tests_total": len(caps)})
print(json.dumps({"success": all(value == 1.0 for value in caps.values()), "subscores": subscores, "capability_weights": {name: 0.1 for name in caps}, "failure_labels": [f"{name}_failed" for name, value in caps.items() if value < 1.0]}))
""")

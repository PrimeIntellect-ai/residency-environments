#!/usr/bin/env bash
set -euo pipefail

prime_log() {
  local prefix="${PRIME_LOG_PREFIX:-$(basename "$0")}"
  printf '[%s] %s\n' "${prefix}" "$*" >&2
}

prime_warn() {
  prime_log "WARN: $*"
}

prime_die() {
  prime_log "ERROR: $*"
  exit 1
}

prime_require_cmd() {
  command -v "$1" >/dev/null 2>&1 || prime_die "required command not found: $1"
}

prime_require_local_prereqs() {
  prime_require_cmd python3
  prime_require_cmd ssh
  prime_require_cmd scp
  prime_require_cmd rsync
  prime_require_cmd prime
}

prime_now_utc() {
  date -u +"%Y%m%dT%H%M%SZ"
}

prime_abspath() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

prime_parse_first_pod_id() {
  python3 -c '
import re
import sys

text = sys.stdin.read()
match = re.search(r"\b[a-f0-9]{32}\b", text, flags=re.IGNORECASE)
if match:
    print(match.group(0).lower())
'
}

prime_json_field() {
  local field="$1"
  python3 -c '
import json
import sys

field = sys.argv[1]
obj = json.load(sys.stdin)
value = obj.get(field)
if value is None:
    print("")
elif isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
' "${field}"
}

prime_pod_id_by_name() {
  local name="$1"
  python3 -c '
import json
import sys

name = sys.argv[1]
pods = json.load(sys.stdin).get("pods", [])
for pod in pods:
    if pod.get("name") == name:
        print((pod.get("id") or "").strip())
        break
' "${name}"
}

prime_pick_node_id_by_gpu_query() {
  local query_csv="$1"
  prime --plain availability list --gpu-count 1 -o json | python3 -c '
import json
import sys

queries = [part.strip() for part in sys.argv[1].split(",") if part.strip()]
resources = json.load(sys.stdin).get("gpu_resources", [])
provider_pref_raw = sys.argv[2].strip()

def norm(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())

normalized_queries = [norm(q) for q in queries]
available = [row for row in resources if str(row.get("stock_status", "")).lower() == "available"]
provider_prefs = [norm(part) for part in provider_pref_raw.split(",") if part.strip()]

def provider_rank(row) -> int:
    if not provider_prefs:
        return len(provider_prefs)
    provider = norm(row.get("provider", ""))
    for idx, pref in enumerate(provider_prefs):
        if pref == provider:
            return idx
    return len(provider_prefs)

for query in normalized_queries:
    matches = []
    for row in available:
        gpu_type = norm(row.get("gpu_type", ""))
        if gpu_type == query or query in gpu_type:
            matches.append(row)
    if matches:
        matches.sort(key=lambda row: (provider_rank(row), float(row.get("price_value") or 1e18)))
        print(matches[0]["id"])
        sys.exit(0)

sys.exit(1)
' "${query_csv}" "${PRIME_PROVIDER_PREFERENCE:-}"
}

prime_wait_for_pod_visible() {
  local name="$1"
  local timeout_s="${2:-180}"
  local start_ts
  start_ts="$(date +%s)"
  while (( "$(date +%s)" - start_ts < timeout_s )); do
    local pod_id
    pod_id="$(prime --plain pods list -o json | prime_pod_id_by_name "${name}" || true)"
    if [[ -n "${pod_id}" ]]; then
      printf '%s\n' "${pod_id}"
      return 0
    fi
    sleep 5
  done
  return 1
}

prime_create_pod() {
  local name="$1"
  local node_id="$2"
  local gpu_query="$3"
  local disk_size="$4"
  local image="$5"
  local custom_template_id="${6:-}"

  local chosen_id="${node_id}"
  if [[ -z "${chosen_id}" ]]; then
    chosen_id="$(prime_pick_node_id_by_gpu_query "${gpu_query}")" || {
      prime_die "could not find an available Prime node matching GPU query: ${gpu_query}"
    }
  fi

  local cmd=(
    prime --plain pods create
    --id "${chosen_id}"
    --name "${name}"
    --disk-size "${disk_size}"
    --yes
  )
  if [[ -n "${custom_template_id}" ]]; then
    cmd+=(--image custom_template --custom-template-id "${custom_template_id}")
  else
    cmd+=(--image "${image}")
  fi
  if [[ -n "${PRIME_TEAM_ID:-}" ]]; then
    cmd+=(--team-id "${PRIME_TEAM_ID}")
  fi

  if [[ -n "${custom_template_id}" ]]; then
    prime_log "creating Prime pod name=${name} node_id=${chosen_id} custom_template_id=${custom_template_id} disk=${disk_size}GB"
  else
    prime_log "creating Prime pod name=${name} node_id=${chosen_id} image=${image} disk=${disk_size}GB"
  fi

  local output status
  set +e
  output="$("${cmd[@]}" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "${output}" >&2
  [[ ${status} -eq 0 ]] || prime_die "prime pods create failed"

  local pod_id
  pod_id="$(printf '%s' "${output}" | prime_parse_first_pod_id || true)"
  if [[ -z "${pod_id}" ]]; then
    pod_id="$(prime_wait_for_pod_visible "${name}" 180)" || {
      prime_die "pod creation succeeded but pod id could not be discovered for name=${name}"
    }
  fi
  printf '%s\n' "${pod_id}"
}

prime_wait_for_pod_active() {
  local pod_id="$1"
  local timeout_s="${2:-1800}"
  local start_ts
  start_ts="$(date +%s)"
  while (( "$(date +%s)" - start_ts < timeout_s )); do
    local status_json
    status_json="$(prime --plain pods status "${pod_id}" -o json 2>/dev/null || true)"
    if [[ -n "${status_json}" ]]; then
      local status ip installation
      status="$(printf '%s' "${status_json}" | prime_json_field status)"
      ip="$(printf '%s' "${status_json}" | prime_json_field ip)"
      installation="$(printf '%s' "${status_json}" | prime_json_field installation_status)"
      if [[ "${status}" == "ACTIVE" && -n "${ip}" ]]; then
        printf '%s\n' "${status_json}"
        return 0
      fi
      prime_log "waiting for pod ${pod_id}: status=${status:-unknown} ip=${ip:-none} installation=${installation:-unknown}"
    else
      prime_log "waiting for pod ${pod_id}: status unavailable"
    fi
    sleep 10
  done
  return 1
}

prime_status_ssh_target() {
  local default_user="${1:-}"
  python3 -c '
import json
import re
import sys

default_user = sys.argv[1]
obj = json.load(sys.stdin)
ssh_target = str(obj.get("ssh") or "").strip()
ip = str(obj.get("ip") or "").strip()

if ssh_target:
    match = re.search(r"([A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+)", ssh_target)
    if match:
        print(match.group(1))
        sys.exit(0)
if default_user and ip:
    print(f"{default_user}@{ip}")
elif ip:
    user = default_user or "ubuntu"
    print(f"{user}@{ip}")
' "${default_user}"
}

prime_status_ssh_port() {
  python3 -c '
import json
import re
import sys

obj = json.load(sys.stdin)
ssh_target = str(obj.get("ssh") or "").strip()
match = re.search(r"(?:^| )-p\s+(\d+)(?:$| )", ssh_target)
print(match.group(1) if match else "22")
'
}

prime_ssh() {
  local key="$1"
  local target="$2"
  local port="${3:-22}"
  shift 3
  ssh \
    -i "${key}" \
    -p "${port}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -o ConnectTimeout=10 \
    "${target}" \
    "$@"
}

prime_scp() {
  local key="$1"
  local source_path="$2"
  local dest_path="$3"
  local port="${4:-22}"
  scp \
    -i "${key}" \
    -P "${port}" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    -r \
    "${source_path}" \
    "${dest_path}"
}

prime_rsync_push_dir() {
  local key="$1"
  local source_dir="$2"
  local target="$3"
  local dest_dir="$4"
  local port="${5:-22}"
  shift 5

  local cmd=(
    rsync
    -az
    --delete
    -e
    "ssh -i ${key} -p ${port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
  )
  while (( "$#" > 0 )); do
    cmd+=(--exclude "$1")
    shift
  done
  cmd+=("${source_dir}/" "${target}:${dest_dir}/")
  "${cmd[@]}"
}

prime_rsync_push_dir_no_delete() {
  local key="$1"
  local source_dir="$2"
  local target="$3"
  local dest_dir="$4"
  local port="${5:-22}"
  shift 5

  local cmd=(
    rsync
    -az
    -e
    "ssh -i ${key} -p ${port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
  )
  while (( "$#" > 0 )); do
    cmd+=(--exclude "$1")
    shift
  done
  cmd+=("${source_dir}/" "${target}:${dest_dir}/")
  "${cmd[@]}"
}

prime_rsync_pull_dir() {
  local key="$1"
  local target="$2"
  local source_dir="$3"
  local dest_dir="$4"
  local port="${5:-22}"
  rsync \
    -az \
    -e "ssh -i ${key} -p ${port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
    "${target}:${source_dir}/" \
    "${dest_dir}/"
}

prime_remote_bash() {
  local key="$1"
  local target="$2"
  local port="$3"
  local command="$4"
  prime_ssh "${key}" "${target}" "${port}" "bash -c $(printf '%q' "${command}")"
}

prime_wait_for_ssh() {
  local key="$1"
  local target="$2"
  local port="${3:-22}"
  local timeout_s="${4:-300}"
  local start_ts
  start_ts="$(date +%s)"
  while (( "$(date +%s)" - start_ts < timeout_s )); do
    if prime_ssh "${key}" "${target}" "${port}" "printf ready" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

prime_remote_home() {
  local key="$1"
  local target="$2"
  local port="${3:-22}"
  prime_remote_bash "${key}" "${target}" "${port}" 'printf "%s\n" "$HOME"'
}

prime_ensure_remote_rsync() {
  local key="$1"
  local target="$2"
  local port="${3:-22}"
  prime_remote_bash "${key}" "${target}" "${port}" '
if command -v rsync >/dev/null 2>&1; then
  exit 0
fi
run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
  run_root apt-get update -y >/dev/null
  run_root apt-get install -y rsync >/dev/null
elif command -v dnf >/dev/null 2>&1; then
  run_root dnf install -y rsync >/dev/null
elif command -v yum >/dev/null 2>&1; then
  run_root yum install -y rsync >/dev/null
else
  echo "rsync is missing and no supported package manager is available" >&2
  exit 1
fi
'
}

prime_sync_repo_checkout() {
  local key="$1"
  local source_dir="$2"
  local target="$3"
  local remote_checkout="$4"
  local port="${5:-22}"

  prime_ensure_remote_rsync "${key}" "${target}" "${port}"
  prime_remote_bash "${key}" "${target}" "${port}" "mkdir -p \"${remote_checkout}\""
  prime_rsync_push_dir \
    "${key}" \
    "${source_dir}" \
    "${target}" \
    "${remote_checkout}" \
    "${port}" \
    ".git/" \
    ".venv/" \
    "outputs/" \
    "_out/" \
    "__pycache__/" \
    "*.pyc" \
    ".pytest_cache/" \
    ".ruff_cache/" \
    ".mypy_cache/" \
    "node_modules/" \
    ".cache/" \
    "tmp_sync_bundle/"
}

prime_sync_generic_checkout() {
  local key="$1"
  local source_dir="$2"
  local target="$3"
  local remote_dir="$4"
  local port="${5:-22}"

  prime_ensure_remote_rsync "${key}" "${target}" "${port}"
  prime_remote_bash "${key}" "${target}" "${port}" "mkdir -p \"${remote_dir}\""
  prime_rsync_push_dir_no_delete \
    "${key}" \
    "${source_dir}" \
    "${target}" \
    "${remote_dir}" \
    "${port}" \
    ".git/" \
    ".venv/" \
    "__pycache__/" \
    "*.pyc" \
    ".pytest_cache/" \
    ".ruff_cache/" \
    ".mypy_cache/" \
    "node_modules/" \
    ".cache/" \
    "checkpoints/" \
    "weights/" \
    "artifacts/"
}

prime_env_set() {
  local env_file="$1"
  local key="$2"
  local value="$3"
  printf '%s=%q\n' "${key}" "${value}" >>"${env_file}"
}

prime_hf_token_from_file() {
  local file_path="$1"
  [[ -f "${file_path}" ]] || return 1
  python3 - "${file_path}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(errors="ignore")

raw = text.strip()
if re.fullmatch(r"hf_[A-Za-z0-9]{20,}", raw):
    print(raw)
    raise SystemExit(0)

for line in text.splitlines():
    s = line.strip()
    if not s or s.startswith("#") or "=" not in s:
        continue
    key, value = s.split("=", 1)
    key = key.strip()
    if key not in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"}:
        continue
    value = value.strip().strip('"').strip("'")
    if re.fullmatch(r"hf_[A-Za-z0-9]{20,}", value):
        print(value)
        raise SystemExit(0)

match = re.search(r"hf_[A-Za-z0-9]{20,}", text)
if match:
    print(match.group(0))
PY
}

prime_find_hf_token() {
  local repo_root="${1:-}"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s\n' "${HF_TOKEN}"
    return 0
  fi
  if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    printf '%s\n' "${HUGGINGFACE_HUB_TOKEN}"
    return 0
  fi
  if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
    printf '%s\n' "${HUGGINGFACE_TOKEN}"
    return 0
  fi

  local candidate=""
  local -a candidates=()
  if [[ -n "${repo_root}" ]]; then
    candidates+=("${repo_root}/.env" "$(dirname "${repo_root}")/.env")
  fi
  candidates+=(
    "${HOME}/carla_cam/.env"
    "${HOME}/carla-env/.env"
    "${HOME}/carlaenvrel/.env"
    "${HOME}/carlaenv_release/.env"
    "${HOME}/.config/huggingface/token"
    "${HOME}/.huggingface/token"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      local token=""
      token="$(prime_hf_token_from_file "${candidate}" || true)"
      if [[ -n "${token}" ]]; then
        printf '%s\n' "${token}"
        return 0
      fi
    fi
  done
  return 1
}

prime_guess_cosmos_source() {
  local repo_root="$1"
  local base_dir
  base_dir="$(dirname "${repo_root}")"

  if [[ -n "${COSMOS_TRANSFER_SOURCE:-}" && -d "${COSMOS_TRANSFER_SOURCE}" ]]; then
    printf '%s\n' "${COSMOS_TRANSFER_SOURCE}"
    return 0
  fi

  while IFS= read -r candidate; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(find "${base_dir}/node_sync" -maxdepth 3 -type d -path '*/cosmos/cosmos-transfer2.5' 2>/dev/null | sort -r)

  return 1
}

prime_write_ssh_config_snippet() {
  local file_path="$1"
  local alias_name="$2"
  local host="$3"
  local user="$4"
  local key="$5"
  local port="${6:-22}"
  cat >"${file_path}" <<EOF
Host ${alias_name}
  HostName ${host}
  User ${user}
  Port ${port}
  IdentityFile ${key}
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
EOF
}

prime_terminate_pod() {
  local pod_id="$1"
  prime_log "terminating Prime pod ${pod_id}"
  prime --plain pods terminate "${pod_id}" --yes >/dev/null
}

prime_git_head() {
  local repo_root="$1"
  git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true
}

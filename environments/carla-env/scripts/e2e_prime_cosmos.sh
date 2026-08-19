#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/prime_pod_common.sh"
PRIME_LOG_PREFIX="$(basename "$0")"
: "${PRIME_PROVIDER_PREFERENCE:=datacrunch}"

usage() {
  cat <<'EOF'
Usage:
  scripts/e2e_prime_cosmos.sh [options]

Modes:
  --mode rent                  Rent a fresh Prime pod, deploy Cosmos, smoke-check it, sync artifacts.
  --mode attach                Attach to an existing SSH host and deploy there.

Common options:
  --name <pod-name>            Prime pod name in rent mode. Default: cosmos-<timestamp>
  --ssh-key <path>             SSH private key. Default: ~/.ssh/claude_prime_ed25519
  --user <ssh-user>            SSH login user. Default: ubuntu
  --ssh-port <port>            SSH port. Default: 22, or pod-reported port in rent mode.
  --host <ip-or-host>          Required in attach mode.
  --repo-root <path>           Local repo root. Default: current repo root
  --remote-checkout-name <n>   Remote checkout directory name. Default: carla-env
  --cosmos-source <path>       Local cosmos-transfer2.5 checkout/snapshot. Default: auto-discover from node_sync
  --output-dir <path>          Local artifact dir. Default: outputs/prime_e2e/cosmos/<timestamp>
  --terminate                  Terminate a newly rented pod on script exit

Prime rent options:
  --node-id <prime-node-id>    Exact Prime availability id to rent
  --gpu <query>                GPU preference list. Default: H100 80GB,L40S 48GB
  --disk-size <gb>             Pod disk size. Default: 200
  --image <name>               Prime pod image. Default: ubuntu_22_cuda_12
  --custom-template-id <id>    Prime custom template id. Overrides --image.

Cosmos options:
  --carla-port <port>          CARLA RPC port. Default: 2000
  --server-port <port>         Cosmos frame-server port. Default: 8080
  --scenario <name>            Repo scenario for smoke check. Default: navigation_vision_Town05_v1_p0
  --prompt <text>              Cosmos prompt. Default: repo smoke prompt
  --timeout <seconds>          Cosmos request timeout. Default: 30
  --carla-root <remote-path>   Optional preinstalled CARLA root on the pod image.

Examples:
  scripts/e2e_prime_cosmos.sh --mode rent

  scripts/e2e_prime_cosmos.sh \
    --mode attach \
    --host 203.0.113.10 \
    --cosmos-source /path/to/cosmos-transfer2.5
EOF
}

MODE="rent"
NAME=""
SSH_KEY="${HOME}/.ssh/claude_prime_ed25519"
SSH_USER="ubuntu"
SSH_PORT="22"
HOST=""
REPO_ROOT="${REPO_ROOT_DEFAULT}"
REMOTE_CHECKOUT_NAME="carla-env"
COSMOS_SOURCE=""
OUTPUT_DIR=""
TERMINATE="0"
NODE_ID=""
GPU_QUERY="H100 80GB,L40S 48GB"
DISK_SIZE="200"
IMAGE="ubuntu_22_cuda_12"
CUSTOM_TEMPLATE_ID=""
CARLA_PORT="2000"
SERVER_PORT="8080"
SCENARIO="navigation_vision_Town05_v1_p0"
PROMPT="Dashcam view of a realistic city street with natural lighting, photorealistic, high detail"
TIMEOUT="30"
CARLA_ROOT=""

while (( "$#" > 0 )); do
  case "$1" in
    --mode)
      MODE="${2:?missing value for --mode}"
      shift 2
      ;;
    --name)
      NAME="${2:?missing value for --name}"
      shift 2
      ;;
    --ssh-key)
      SSH_KEY="${2:?missing value for --ssh-key}"
      shift 2
      ;;
    --user)
      SSH_USER="${2:?missing value for --user}"
      shift 2
      ;;
    --ssh-port)
      SSH_PORT="${2:?missing value for --ssh-port}"
      shift 2
      ;;
    --host)
      HOST="${2:?missing value for --host}"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="${2:?missing value for --repo-root}"
      shift 2
      ;;
    --remote-checkout-name)
      REMOTE_CHECKOUT_NAME="${2:?missing value for --remote-checkout-name}"
      shift 2
      ;;
    --cosmos-source)
      COSMOS_SOURCE="${2:?missing value for --cosmos-source}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --terminate)
      TERMINATE="1"
      shift
      ;;
    --node-id)
      NODE_ID="${2:?missing value for --node-id}"
      shift 2
      ;;
    --gpu)
      GPU_QUERY="${2:?missing value for --gpu}"
      shift 2
      ;;
    --disk-size)
      DISK_SIZE="${2:?missing value for --disk-size}"
      shift 2
      ;;
    --image)
      IMAGE="${2:?missing value for --image}"
      shift 2
      ;;
    --custom-template-id)
      CUSTOM_TEMPLATE_ID="${2:?missing value for --custom-template-id}"
      shift 2
      ;;
    --carla-port)
      CARLA_PORT="${2:?missing value for --carla-port}"
      shift 2
      ;;
    --server-port)
      SERVER_PORT="${2:?missing value for --server-port}"
      shift 2
      ;;
    --scenario)
      SCENARIO="${2:?missing value for --scenario}"
      shift 2
      ;;
    --prompt)
      PROMPT="${2:?missing value for --prompt}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:?missing value for --timeout}"
      shift 2
      ;;
    --carla-root)
      CARLA_ROOT="${2:?missing value for --carla-root}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      prime_die "unknown argument: $1"
      ;;
  esac
done

case "${MODE}" in
  rent|attach) ;;
  *) prime_die "--mode must be rent or attach" ;;
esac

RUN_ID="$(prime_now_utc)"
if [[ -z "${NAME}" ]]; then
  NAME="cosmos-${RUN_ID,,}"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/outputs/prime_e2e/cosmos/${RUN_ID}"
fi

if [[ "${MODE}" == "attach" && -z "${HOST}" ]]; then
  prime_die "--host is required in attach mode"
fi
if [[ "${MODE}" == "attach" && "${TERMINATE}" == "1" ]]; then
  prime_die "--terminate is only supported for freshly rented pods"
fi
if [[ "${HOST}" == *@* ]]; then
  SSH_USER="${HOST%@*}"
  HOST="${HOST#*@}"
fi
if [[ -n "${CUSTOM_TEMPLATE_ID}" && "${MODE}" != "rent" ]]; then
  prime_die "--custom-template-id is only valid in rent mode"
fi

SSH_KEY="$(prime_abspath "${SSH_KEY}")"
REPO_ROOT="$(prime_abspath "${REPO_ROOT}")"
OUTPUT_DIR="$(prime_abspath "${OUTPUT_DIR}")"
if [[ -z "${REMOTE_CHECKOUT_NAME}" || "${REMOTE_CHECKOUT_NAME}" == */* ]]; then
  prime_die "--remote-checkout-name must be a single directory name"
fi

[[ -d "${REPO_ROOT}" ]] || prime_die "repo root does not exist: ${REPO_ROOT}"
[[ -f "${SSH_KEY}" ]] || prime_die "ssh key not found: ${SSH_KEY}"

if [[ -z "${COSMOS_SOURCE}" ]]; then
  COSMOS_SOURCE="$(prime_guess_cosmos_source "${REPO_ROOT}")" || {
    prime_die "could not auto-discover a local cosmos-transfer2.5 source; pass --cosmos-source"
  }
fi
COSMOS_SOURCE="$(prime_abspath "${COSMOS_SOURCE}")"
[[ -d "${COSMOS_SOURCE}" ]] || prime_die "Cosmos source does not exist: ${COSMOS_SOURCE}"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/deploy.log") 2>&1

prime_require_cmd python3
prime_require_cmd ssh
prime_require_cmd scp
prime_require_cmd rsync
if [[ "${MODE}" == "rent" ]]; then
  prime_require_cmd prime
fi

HF_TOKEN_VALUE="$(prime_find_hf_token "${REPO_ROOT}" || true)"
if [[ -z "${HF_TOKEN_VALUE}" ]]; then
  prime_warn "no HF token found locally; Cosmos bootstrap will fail on gated NVIDIA checkpoints"
fi

CREATED_POD_ID=""
TMP_DIR=""
cleanup() {
  local status="$1"
  if [[ -n "${TMP_DIR:-}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
  if [[ "${TERMINATE}" == "1" && -n "${CREATED_POD_ID}" ]]; then
    prime_terminate_pod "${CREATED_POD_ID}" || prime_warn "failed to terminate pod ${CREATED_POD_ID}"
  fi
  trap - EXIT
  exit "${status}"
}
trap 'cleanup "$?"' EXIT

TMP_DIR="$(mktemp -d -t prime-e2e-cosmos.XXXXXX)"

STATUS_JSON=""
POD_ID=""
HOST_ONLY="${HOST}"

if [[ "${MODE}" == "rent" ]]; then
  POD_ID="$(prime_create_pod "${NAME}" "${NODE_ID}" "${GPU_QUERY}" "${DISK_SIZE}" "${IMAGE}" "${CUSTOM_TEMPLATE_ID}")"
  CREATED_POD_ID="${POD_ID}"
  STATUS_JSON="$(prime_wait_for_pod_active "${POD_ID}" 2400)" || prime_die "Prime pod did not become ACTIVE in time"
  printf '%s\n' "${STATUS_JSON}" >"${OUTPUT_DIR}/prime_status.json"
  HOST_ONLY="$(printf '%s' "${STATUS_JSON}" | prime_json_field ip)"
  if [[ -n "${CUSTOM_TEMPLATE_ID}" ]]; then
    SSH_TARGET="${SSH_USER}@${HOST_ONLY}"
  else
    SSH_TARGET="$(printf '%s' "${STATUS_JSON}" | prime_status_ssh_target "${SSH_USER}")"
    SSH_PORT="$(printf '%s' "${STATUS_JSON}" | prime_status_ssh_port)"
  fi
else
  POD_ID=""
  SSH_TARGET="${SSH_USER}@${HOST_ONLY}"
fi

if [[ -z "${SSH_TARGET:-}" ]]; then
  prime_die "could not resolve SSH target"
fi
if [[ "${SSH_TARGET}" == *@* ]]; then
  SSH_USER="${SSH_TARGET%@*}"
fi

prime_log "waiting for SSH on ${SSH_TARGET}"
prime_wait_for_ssh "${SSH_KEY}" "${SSH_TARGET}" "${SSH_PORT}" 600 || prime_die "SSH did not become ready for ${SSH_TARGET}"

REMOTE_HOME="$(prime_remote_home "${SSH_KEY}" "${SSH_TARGET}" "${SSH_PORT}")"
[[ -n "${REMOTE_HOME}" ]] || prime_die "failed to resolve remote home directory"
REMOTE_CHECKOUT="${REMOTE_HOME}/${REMOTE_CHECKOUT_NAME}"
REMOTE_COSMOS_SOURCE="${REMOTE_HOME}/cosmos-transfer2.5"
REMOTE_ARTIFACT_DIR="${REMOTE_CHECKOUT}/outputs/prime_e2e/cosmos/${RUN_ID}"
REMOTE_TMP_DIR="${REMOTE_CHECKOUT}/.prime_e2e/${RUN_ID}"

prime_log "syncing repo checkout to ${SSH_TARGET}:${REMOTE_CHECKOUT}"
prime_sync_repo_checkout "${SSH_KEY}" "${REPO_ROOT}" "${SSH_TARGET}" "${REMOTE_CHECKOUT}" "${SSH_PORT}"
prime_log "syncing Cosmos source to ${SSH_TARGET}:${REMOTE_COSMOS_SOURCE}"
prime_sync_generic_checkout "${SSH_KEY}" "${COSMOS_SOURCE}" "${SSH_TARGET}" "${REMOTE_COSMOS_SOURCE}" "${SSH_PORT}"
prime_remote_bash "${SSH_KEY}" "${SSH_TARGET}" "${SSH_PORT}" "mkdir -p \"${REMOTE_ARTIFACT_DIR}\" \"${REMOTE_TMP_DIR}\""

LOCAL_REPO_REV="$(prime_git_head "${REPO_ROOT}")"
BOOTSTRAP_ENV="${TMP_DIR}/bootstrap_cosmos.env"
BOOTSTRAP_SCRIPT="${TMP_DIR}/bootstrap_cosmos.sh"

: >"${BOOTSTRAP_ENV}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_HOME "${REMOTE_HOME}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_CHECKOUT "${REMOTE_CHECKOUT}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_COSMOS_SOURCE "${REMOTE_COSMOS_SOURCE}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_ARTIFACT_DIR "${REMOTE_ARTIFACT_DIR}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_TMP_DIR "${REMOTE_TMP_DIR}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_LOGIN_USER "${SSH_USER}"
prime_env_set "${BOOTSTRAP_ENV}" LOCAL_REPO_REV "${LOCAL_REPO_REV}"
prime_env_set "${BOOTSTRAP_ENV}" CARLA_PORT "${CARLA_PORT}"
prime_env_set "${BOOTSTRAP_ENV}" COSMOS_SERVER_PORT "${SERVER_PORT}"
prime_env_set "${BOOTSTRAP_ENV}" COSMOS_SCENARIO "${SCENARIO}"
prime_env_set "${BOOTSTRAP_ENV}" COSMOS_PROMPT "${PROMPT}"
prime_env_set "${BOOTSTRAP_ENV}" COSMOS_TIMEOUT "${TIMEOUT}"
prime_env_set "${BOOTSTRAP_ENV}" CARLA_PREINSTALLED_ROOT "${CARLA_ROOT}"
if [[ -n "${HF_TOKEN_VALUE}" ]]; then
  prime_env_set "${BOOTSTRAP_ENV}" HF_TOKEN "${HF_TOKEN_VALUE}"
fi

cat >"${BOOTSTRAP_SCRIPT}" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

ENV_FILE="${1:?missing env file path}"
source "${ENV_FILE}"
rm -f "${ENV_FILE}"

mkdir -p "${REMOTE_ARTIFACT_DIR}"
exec > >(tee -a "${REMOTE_ARTIFACT_DIR}/bootstrap.log") 2>&1

log() {
  printf '[cosmos-bootstrap] %s\n' "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo -n "$@"
  else
    die "sudo is required for remote setup"
  fi
}

docker_available() {
  docker info >/dev/null 2>&1 || sudo -n docker info >/dev/null 2>&1
}

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sudo -n docker "$@"
  fi
}

ensure_docker_user_access() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  if ! sudo -n docker info >/dev/null 2>&1; then
    return 1
  fi

  run_root groupadd -f docker >/dev/null 2>&1 || true
  run_root usermod -aG docker "$(id -un)" >/dev/null 2>&1 || true
  if run_root test -S /var/run/docker.sock; then
    run_root chmod 666 /var/run/docker.sock >/dev/null 2>&1 || true
  fi

  local deadline=$((SECONDS + 10))
  while (( SECONDS < deadline )); do
    if docker info >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

ensure_docker_daemon() {
  if docker_available; then
    ensure_docker_user_access || die "docker daemon is reachable, but docker is not accessible without sudo"
    return 0
  fi

  run_root systemctl enable --now docker >/dev/null 2>&1 || true
  if docker_available; then
    ensure_docker_user_access || die "docker daemon is reachable, but docker is not accessible without sudo"
    return 0
  fi

  if ! run_root bash -lc 'pgrep -f "(^|/)dockerd( |$)" >/dev/null 2>&1'; then
    local dockerd_log="${REMOTE_HOME}/.cache/carla-env/dockerd-bootstrap.log"
    run_root mkdir -p "$(dirname "${dockerd_log}")"
    run_root_bash "nohup dockerd --iptables=false --bridge=none --ip-forward=false --ip-masq=false >'${dockerd_log}' 2>&1 &"
  fi

  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if docker_available; then
      ensure_docker_user_access || die "docker daemon is reachable, but docker is not accessible without sudo"
      return 0
    fi
    sleep 2
  done

  die "docker daemon is not reachable"
}

port_ready() {
  python3 - "$1" "$2" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket() as sock:
    sock.settimeout(1.0)
    try:
        sock.connect((host, port))
    except OSError:
        sys.exit(1)
sys.exit(0)
PY
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_s="${3:-240}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if port_ready "${host}" "${port}"; then
      return 0
    fi
    sleep 2
  done
  return 1
}

carla_rpc_ready() {
  local host="$1"
  local port="$2"
  local py="${REMOTE_CHECKOUT}/.venv/bin/python"
  [[ -x "${py}" ]] || return 1
  "${py}" - "${host}" "${port}" <<'PY' >/dev/null 2>&1
import sys
import carla

host = sys.argv[1]
port = int(sys.argv[2])
client = carla.Client(host, port)
client.set_timeout(5.0)
world = client.get_world()
_ = world.get_map().name
PY
}

wait_for_carla_rpc() {
  local host="$1"
  local port="$2"
  local timeout_s="${3:-240}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if port_ready "${host}" "${port}" && carla_rpc_ready "${host}" "${port}"; then
      return 0
    fi
    sleep 2
  done
  return 1
}

ensure_base_packages() {
  local missing=()
  for cmd in git curl ffmpeg python3; do
    command -v "${cmd}" >/dev/null 2>&1 || missing+=("${cmd}")
  done
  if ! python3 -m venv --help >/dev/null 2>&1; then
    missing+=("python3-venv")
  fi
  if (( ${#missing[@]} > 0 )); then
    log "installing base packages: ${missing[*]}"
    run_root apt-get update -y
    run_root apt-get install -y python3-venv python3-pip git curl ffmpeg rsync
  fi
  if ! command -v docker >/dev/null 2>&1; then
    log "installing docker.io"
    run_root apt-get update -y
    run_root apt-get install -y docker.io
  fi
  ensure_docker_daemon
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required on the remote host"
}

python_supports_repo() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
ok = sys.version_info >= (3, 11) and sys.version_info.releaselevel == "final"
raise SystemExit(0 if ok else 1)
PY
}

python_is_minor() {
  local interpreter="$1"
  local major="$2"
  local minor="$3"
  "$interpreter" - "$major" "$minor" <<'PY' >/dev/null 2>&1
import sys

want_major = int(sys.argv[1])
want_minor = int(sys.argv[2])
raise SystemExit(0 if sys.version_info[:2] == (want_major, want_minor) else 1)
PY
}

ensure_python_runtime() {
  local candidate=""
  for candidate in python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 && python_supports_repo "${candidate}"; then
      PYTHON_BIN="${candidate}"
      export PYTHON_BIN
      return 0
    fi
  done
  [[ -n "${UV_BIN:-}" ]] || die "uv is required to provision a stable Python runtime"
  candidate="$("${UV_BIN}" python find --managed-python 3.12 2>/dev/null || true)"
  if [[ -z "${candidate}" ]]; then
    log "installing managed Python 3.12 with uv"
    "${UV_BIN}" python install 3.12
    candidate="$("${UV_BIN}" python find --managed-python 3.12)"
  fi
  [[ -n "${candidate}" ]] || die "failed to resolve a managed Python 3.12 interpreter via uv"
  PYTHON_BIN="${candidate}"
  export PYTHON_BIN
}

ensure_cosmos_python_runtime() {
  if command -v python3.10 >/dev/null 2>&1 && python_is_minor python3.10 3 10; then
    COSMOS_PYTHON_BIN="python3.10"
    export COSMOS_PYTHON_BIN
    return 0
  fi
  if command -v python3 >/dev/null 2>&1 && python_is_minor python3 3 10; then
    COSMOS_PYTHON_BIN="python3"
    export COSMOS_PYTHON_BIN
    return 0
  fi
  log "installing python3.10 runtime for Cosmos workspace"
  run_root apt-get update -y
  run_root apt-get install -y python3.10 python3.10-dev python3.10-venv
  COSMOS_PYTHON_BIN="python3.10"
  export COSMOS_PYTHON_BIN
}

ensure_repo_venv() {
  cd "${REMOTE_CHECKOUT}"
  if [[ -x "${REMOTE_CHECKOUT}/.venv/bin/python" ]] && ! python_supports_repo "${REMOTE_CHECKOUT}/.venv/bin/python"; then
    log "rebuilding repo virtualenv for Python 3.11+"
    rm -rf "${REMOTE_CHECKOUT}/.venv"
  fi
  if [[ ! -x "${REMOTE_CHECKOUT}/.venv/bin/python" ]]; then
    log "creating repo virtualenv"
    "${PYTHON_BIN}" -m venv "${REMOTE_CHECKOUT}/.venv"
  fi
  local py="${REMOTE_CHECKOUT}/.venv/bin/python"
  "${py}" -m pip install --upgrade pip setuptools wheel
  "${py}" -m pip install -e "${REMOTE_CHECKOUT}[carla9]" "imageio[ffmpeg]"
}

ensure_uv() {
  local uv_bin="${REMOTE_HOME}/.local/bin/uv"
  if [[ ! -x "${uv_bin}" ]]; then
    log "installing uv"
    mkdir -p "${REMOTE_HOME}/.config"
    if [[ ! -w "${REMOTE_HOME}/.config" ]] && command -v sudo >/dev/null 2>&1; then
      sudo chown -R "$(id -un):$(id -gn)" "${REMOTE_HOME}/.config"
    fi
    curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh
  fi
  UV_BIN="${uv_bin}"
  export UV_BIN
}

ensure_cosmos_venv() {
  [[ -d "${REMOTE_COSMOS_SOURCE}" ]] || die "missing synced Cosmos source at ${REMOTE_COSMOS_SOURCE}"
  if [[ -x "${REMOTE_COSMOS_SOURCE}/.venv/bin/python" ]] && ! python_is_minor "${REMOTE_COSMOS_SOURCE}/.venv/bin/python" 3 10
  then
    log "rebuilding Cosmos virtualenv for Python 3.10"
    rm -rf "${REMOTE_COSMOS_SOURCE}/.venv"
  fi
  if [[ ! -d "${REMOTE_COSMOS_SOURCE}/.venv" ]]; then
    log "creating Cosmos virtualenv"
  fi
  log "syncing Cosmos workspace with uv"
  (
    cd "${REMOTE_COSMOS_SOURCE}"
    "${UV_BIN}" sync --python "${COSMOS_PYTHON_BIN}" --extra=cu128
  )
  COSMOS_PY="${REMOTE_COSMOS_SOURCE}/.venv/bin/python"
  export COSMOS_PY
}

register_cosmos_cuda_libs() {
  local nvidia_root="${REMOTE_COSMOS_SOURCE}/.venv/lib/python3.10/site-packages/nvidia"
  local conf_file="/etc/ld.so.conf.d/cosmos-python-nvidia.conf"
  local -a lib_dirs=()
  local lib_dir=""

  if [[ -d "${nvidia_root}" ]]; then
    while IFS= read -r lib_dir; do
      lib_dirs+=("${lib_dir}")
    done < <(find "${nvidia_root}" -maxdepth 2 -type d -name lib | sort -u)
  fi

  if (( ${#lib_dirs[@]} == 0 )); then
    log "no Cosmos venv CUDA libraries found to register"
    return 0
  fi

  local joined=""
  joined="$(IFS=:; printf '%s' "${lib_dirs[*]}")"
  export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  if command -v ldconfig >/dev/null 2>&1; then
    printf '%s\n' "${lib_dirs[@]}" | run_root tee "${conf_file}" >/dev/null
    run_root ldconfig || true
  fi
}

select_cosmos_gpu() {
  COSMOS_VISIBLE_DEVICE=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${gpu_count:-0}" =~ ^[0-9]+$ ]] && (( gpu_count >= 2 )); then
      # Keep CARLA on GPU 0 and move Cosmos to GPU 1 when the node has headroom.
      COSMOS_VISIBLE_DEVICE=1
    fi
  fi
  export COSMOS_VISIBLE_DEVICE
}

patch_checkpoint_db() {
  local checkpoint_file="${REMOTE_COSMOS_SOURCE}/cosmos_transfer2/_src/imaginaire/utils/checkpoint_db.py"
  [[ -f "${checkpoint_file}" ]] || die "checkpoint_db.py not found at ${checkpoint_file}"
  python3 - "${checkpoint_file}" "${REMOTE_COSMOS_SOURCE}/.venv/bin/huggingface-cli" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
hf_cli = sys.argv[2]
text = path.read_text()

replacement = f'''def _hf_download(cmd_args: list[str]) -> str:
    """Run Hugging Face CLI download command and return the local path.

    Patched for pod deployment: use the checkout-local huggingface-cli and
    keep positional args ahead of options for older CLI variants.
    """
    positional = []
    options = []
    i = 0
    while i < len(cmd_args):
        if cmd_args[i].startswith("--"):
            options.append(cmd_args[i])
            if i + 1 < len(cmd_args) and not cmd_args[i + 1].startswith("--"):
                options.append(cmd_args[i + 1])
                i += 2
            else:
                i += 1
        else:
            positional.append(cmd_args[i])
            i += 1
    cmd = [
        "{hf_cli}",
        "download",
        *positional,
        *options,
    ]
    log.info(f"{{shlex.join(cmd)}}")
    subprocess.check_call(cmd, text=True)
    output = subprocess.check_output(
        [*cmd, "--quiet"],
        text=True,
        env=dict(os.environ) | {{"HF_HUB_OFFLINE": "1"}},
    ).strip()
    lines = [line for line in output.splitlines() if line.strip() and not line.strip().startswith("\\u26a0")]
    return lines[-1].strip() if lines else output
'''

start_marker = "def _hf_download(cmd_args: list[str]) -> str:\n"
end_marker = "\n\nclass _CheckpointHf(_CheckpointUri, ABC):\n"

start = text.find(start_marker)
end = text.find(end_marker, start)
if start == -1 or end == -1:
    raise SystemExit("could not locate _hf_download block in checkpoint_db.py")

current_block = text[start:end]
if current_block == replacement.rstrip("\n"):
    print("checkpoint_db.py already patched")
    raise SystemExit(0)

new_text = text[:start] + replacement + text[end:]
path.write_text(new_text)
print("checkpoint_db.py patched")
PY
}

export_hf_auth() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
  fi
}

ensure_carla_vulkan_override() {
  local config_dir="${REMOTE_HOME}/.config/carla-env"
  local host_icd="/usr/share/vulkan/icd.d/nvidia_icd.json"
  if [[ -f "${host_icd}" ]]; then
    CARLA_VULKAN_ICD="${host_icd}"
    export CARLA_VULKAN_ICD
    return 0
  fi

  CARLA_VULKAN_ICD="${config_dir}/nvidia_icd.json"
  export CARLA_VULKAN_ICD
  mkdir -p "${config_dir}"
  cat >"${CARLA_VULKAN_ICD}" <<'EOFJSON'
{
  "file_format_version": "1.0.1",
  "ICD": {
    "library_path": "libGLX_nvidia.so.0",
    "api_version": "1.3.289"
  }
}
EOFJSON
}

start_carla_docker() {
  local carla_name="$1"
  local mode="${2:-plain}"
  local command=(
    run -d
    --name "${carla_name}"
    --runtime nvidia
    --gpus all
    --network host
    --restart unless-stopped
    -e XDG_RUNTIME_DIR=/tmp/runtime-carla
    -e NVIDIA_DRIVER_CAPABILITIES=all
  )

  if [[ "${mode}" == "vk-override" ]]; then
    ensure_carla_vulkan_override
    command+=(
      -e VK_ICD_FILENAMES=/tmp/nvidia_icd.json
      -v "${CARLA_VULKAN_ICD}:/tmp/nvidia_icd.json:ro"
    )
  fi

  command+=(
    carlasim/carla:0.9.16
    /bin/sh -lc "mkdir -p /tmp/runtime-carla && chmod 700 /tmp/runtime-carla && cd /workspace && exec ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=${CARLA_PORT} -stdout -FullStdOutLogOutput -unattended"
  )

  docker_cmd rm -f "${carla_name}" >/dev/null 2>&1 || true
  docker_cmd "${command[@]}" >/dev/null
}

carla_launch_user() {
  if id carla >/dev/null 2>&1; then
    printf 'carla'
  elif [[ "${REMOTE_LOGIN_USER}" == "root" ]] && id ubuntu >/dev/null 2>&1; then
    printf 'ubuntu'
  else
    printf '%s' "${REMOTE_LOGIN_USER}"
  fi
}

ensure_carla() {
  local carla_name="carla-server-0916-${CARLA_PORT}"
  local launch_user
  launch_user="$(carla_launch_user)"
  if port_ready 127.0.0.1 "${CARLA_PORT}"; then
    if wait_for_carla_rpc 127.0.0.1 "${CARLA_PORT}" 60; then
      log "reusing CARLA already listening on port ${CARLA_PORT}"
      return 0
    fi
    log "CARLA port ${CARLA_PORT} is open but not RPC-ready yet; continuing with local startup path"
  fi

  if [[ -n "${CARLA_PREINSTALLED_ROOT:-}" && -d "${CARLA_PREINSTALLED_ROOT}" ]]; then
    log "starting preinstalled CARLA from ${CARLA_PREINSTALLED_ROOT} as ${launch_user}"
    run_root bash "${REMOTE_CHECKOUT}/scripts/start_local_carla_on_pod.sh" prepare \
      --user "${launch_user}" \
      --version 0.9.16 \
      --preinstalled-root "${CARLA_PREINSTALLED_ROOT}" >/dev/null
    run_root bash "${REMOTE_CHECKOUT}/scripts/start_local_carla_on_pod.sh" start \
      --user "${launch_user}" \
      --version 0.9.16 \
      --preinstalled-root "${CARLA_PREINSTALLED_ROOT}" \
      --mode vision \
      --port "${CARLA_PORT}" \
      --timeout 300 \
      --settle 10
    wait_for_carla_rpc 127.0.0.1 "${CARLA_PORT}" 300 || die "CARLA did not become reachable on port ${CARLA_PORT}"
    return 0
  fi

  if docker_available; then
    log "starting CARLA 0.9.16 via docker on port ${CARLA_PORT}"
    docker_cmd pull carlasim/carla:0.9.16 >/dev/null
    start_carla_docker "${carla_name}" plain
    if wait_for_carla_rpc 127.0.0.1 "${CARLA_PORT}" 180; then
      return 0
    fi
    docker_cmd logs "${carla_name}" >"${REMOTE_ARTIFACT_DIR}/carla_docker_plain.log" 2>&1 || true
    log "plain docker CARLA did not become reachable; retrying with explicit Vulkan ICD override"
    start_carla_docker "${carla_name}" vk-override
    if wait_for_carla_rpc 127.0.0.1 "${CARLA_PORT}" 120; then
      return 0
    fi
    log "docker CARLA did not become reachable; falling back to host-side launcher"
  fi

  run_root bash "${REMOTE_CHECKOUT}/scripts/start_local_carla_on_pod.sh" prepare \
    --user "${launch_user}" \
    --version 0.9.16 >/dev/null
  run_root bash "${REMOTE_CHECKOUT}/scripts/start_local_carla_on_pod.sh" start \
    --user "${launch_user}" \
    --version 0.9.16 \
    --mode vision \
    --port "${CARLA_PORT}" \
    --timeout 300 \
    --settle 10

  wait_for_carla_rpc 127.0.0.1 "${CARLA_PORT}" 300 || die "CARLA did not become reachable on port ${CARLA_PORT}"
}

health_json() {
  python3 - "$1" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=5) as response:
    payload = json.loads(response.read().decode("utf-8"))
print(json.dumps(payload))
PY
}

wait_for_server_health() {
  local url="$1"
  local timeout_s="${2:-3600}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    local payload=""
    payload="$(health_json "${url}" 2>/dev/null || true)"
    if [[ -n "${payload}" ]] && python3 - "${payload}" "${COSMOS_PROMPT}" <<'PY' >/dev/null 2>&1
import json
import sys

payload = json.loads(sys.argv[1])
prompt = sys.argv[2]
if payload.get("status") == "ok" and payload.get("model_loaded") and payload.get("prompt") == prompt:
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
      printf '%s\n' "${payload}" >"${REMOTE_ARTIFACT_DIR}/cosmos_health.json"
      return 0
    fi
    sleep 5
  done
  return 1
}

ensure_cosmos_server() {
  local url="http://127.0.0.1:${COSMOS_SERVER_PORT}/health"
  if wait_for_server_health "${url}" 5; then
    log "reusing Cosmos server on port ${COSMOS_SERVER_PORT}"
    return 0
  fi

  pkill -f "scripts/cosmos_frame_server.py" >/dev/null 2>&1 || true
  export_hf_auth
  log "starting Cosmos frame server on port ${COSMOS_SERVER_PORT} using CUDA_VISIBLE_DEVICES=${COSMOS_VISIBLE_DEVICE}"
  local -a server_env=(
    env
    "CUDA_VISIBLE_DEVICES=${COSMOS_VISIBLE_DEVICE}"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  )
  if [[ -n "${HF_TOKEN:-}" ]]; then
    server_env+=("HF_TOKEN=${HF_TOKEN}" "HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}")
  elif [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    server_env+=("HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}")
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    server_env+=("LD_LIBRARY_PATH=${LD_LIBRARY_PATH}")
  fi
  nohup "${server_env[@]}" "${COSMOS_PY}" "${REMOTE_CHECKOUT}/scripts/cosmos_frame_server.py" \
    --host 127.0.0.1 \
    --port "${COSMOS_SERVER_PORT}" \
    --prompt "${COSMOS_PROMPT}" >"${REMOTE_ARTIFACT_DIR}/cosmos_server.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" >"${REMOTE_ARTIFACT_DIR}/cosmos_server.pid"

  wait_for_server_health "${url}" 3600 || die "Cosmos frame server did not become healthy in time"
}

collect_logs() {
  set +e
  local launch_user
  launch_user="$(carla_launch_user)"
  if docker_available; then
    docker_cmd ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' >"${REMOTE_ARTIFACT_DIR}/docker_ps.txt" 2>/dev/null || true
    docker_cmd logs "carla-server-0916-${CARLA_PORT}" >"${REMOTE_ARTIFACT_DIR}/carla_docker.log" 2>&1 || true
  fi
  local host_log_candidates=(
    "/ephemeral/carla-env/${launch_user}/carla-0-9-16-vision-${CARLA_PORT}.log"
    "/ephemeral/carla-env/${REMOTE_LOGIN_USER}/carla-0-9-16-vision-${CARLA_PORT}.log"
    "${REMOTE_HOME}/.cache/carla-env/carla-0-9-16-vision-${CARLA_PORT}.log"
  )
  local host_log=""
  for host_log in "${host_log_candidates[@]}"; do
    if [[ -f "${host_log}" ]]; then
      cp "${host_log}" "${REMOTE_ARTIFACT_DIR}/carla_host.log"
      break
    fi
  done
  ps -ef | grep -E 'cosmos_frame_server.py|uvicorn' | grep -v grep >"${REMOTE_ARTIFACT_DIR}/cosmos_processes.txt" 2>/dev/null || true
  run_root bash "${REMOTE_CHECKOUT}/scripts/start_local_carla_on_pod.sh" status \
    --user "${launch_user}" \
    --version 0.9.16 \
    --mode vision \
    --port "${CARLA_PORT}" >"${REMOTE_ARTIFACT_DIR}/carla_status.txt" 2>&1 || true
  set -e
}

write_manifest() {
  local manifest="${REMOTE_ARTIFACT_DIR}/remote_manifest.txt"
  {
    echo "renderer=cosmos"
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "user=$(whoami)"
    echo "remote_home=${REMOTE_HOME}"
    echo "remote_checkout=${REMOTE_CHECKOUT}"
    echo "remote_cosmos_source=${REMOTE_COSMOS_SOURCE}"
    echo "local_repo_rev=${LOCAL_REPO_REV}"
    echo "carla_port=${CARLA_PORT}"
    echo "server_port=${COSMOS_SERVER_PORT}"
    echo "scenario=${COSMOS_SCENARIO}"
    echo "prompt=${COSMOS_PROMPT}"
    echo "timeout=${COSMOS_TIMEOUT}"
    echo "cosmos_visible_device=${COSMOS_VISIBLE_DEVICE}"
    echo "repo_python=$("${REMOTE_CHECKOUT}/.venv/bin/python" --version 2>&1 || true)"
    echo "cosmos_python=$("${REMOTE_COSMOS_SOURCE}/.venv/bin/python" --version 2>&1 || true)"
    echo "nvidia_smi:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  } >"${manifest}"
}

write_ready_commands() {
  cat >"${REMOTE_ARTIFACT_DIR}/ready_commands.txt" <<EOF2
cd "${REMOTE_CHECKOUT}"
source .venv/bin/activate
export COSMOS_SERVER_URL="http://127.0.0.1:${COSMOS_SERVER_PORT}"

vf-eval carla_env -m "qwen/qwen3-vl-8b-instruct" \\
  -a '{
    "scenario": "${COSMOS_SCENARIO}",
    "sandbox": {"mode": "disabled"},
    "host": "127.0.0.1",
    "port": ${CARLA_PORT},
    "enable_cosmos": true,
    "cosmos_server_url": "http://127.0.0.1:${COSMOS_SERVER_PORT}",
    "cosmos_prompt": "${COSMOS_PROMPT}"
  }' -n 1 -r 1

python - <<'PY'
from carla_env import load_environment

env = load_environment(
    scenario="${COSMOS_SCENARIO}",
    sandbox={"mode": "disabled"},
    host="127.0.0.1",
    port=${CARLA_PORT},
    enable_cosmos=True,
    cosmos_server_url="http://127.0.0.1:${COSMOS_SERVER_PORT}",
    cosmos_prompt="${COSMOS_PROMPT}",
)
print(env)
PY
EOF2
}

finalize() {
  local status="$1"
  set +e
  collect_logs
  write_manifest
  write_ready_commands
  trap - EXIT
  exit "${status}"
}
trap 'finalize "$?"' EXIT

log "remote home: ${REMOTE_HOME}"
log "remote checkout: ${REMOTE_CHECKOUT}"
log "artifact dir: ${REMOTE_ARTIFACT_DIR}"

ensure_base_packages
ensure_uv
ensure_python_runtime
ensure_repo_venv
ensure_cosmos_python_runtime
ensure_cosmos_venv
register_cosmos_cuda_libs
select_cosmos_gpu
patch_checkpoint_db
export_hf_auth
ensure_carla
ensure_cosmos_server

SMOKE_JSON="${REMOTE_ARTIFACT_DIR}/smoke_result.json"
SMOKE_LOG="${REMOTE_ARTIFACT_DIR}/smoke.log"
SMOKE_VIDEO_DIR="${REMOTE_ARTIFACT_DIR}/smoke_video"

set +e
"${REMOTE_CHECKOUT}/.venv/bin/python" "${REMOTE_CHECKOUT}/scripts/cosmos_smoke_check.py" \
  --server-url "http://127.0.0.1:${COSMOS_SERVER_PORT}" \
  --host 127.0.0.1 \
  --port "${CARLA_PORT}" \
  --scenario "${COSMOS_SCENARIO}" \
  --prompt "${COSMOS_PROMPT}" \
  --timeout "${COSMOS_TIMEOUT}" \
  --record-video \
  --output-dir "${SMOKE_VIDEO_DIR}" >"${SMOKE_JSON}" 2>"${SMOKE_LOG}"
SMOKE_STATUS=$?
set -e

[[ -f "${SMOKE_JSON}" ]] && cat "${SMOKE_JSON}"
if [[ ${SMOKE_STATUS} -ne 0 ]]; then
  tail -n 120 "${SMOKE_LOG}" >&2 || true
  exit "${SMOKE_STATUS}"
fi

log "Cosmos smoke check completed successfully"
EOF

prime_scp "${SSH_KEY}" "${BOOTSTRAP_ENV}" "${SSH_TARGET}:${REMOTE_TMP_DIR}/bootstrap_cosmos.env" "${SSH_PORT}"
prime_scp "${SSH_KEY}" "${BOOTSTRAP_SCRIPT}" "${SSH_TARGET}:${REMOTE_TMP_DIR}/bootstrap_cosmos.sh" "${SSH_PORT}"

REMOTE_CMD="chmod +x \"${REMOTE_TMP_DIR}/bootstrap_cosmos.sh\" && \"${REMOTE_TMP_DIR}/bootstrap_cosmos.sh\" \"${REMOTE_TMP_DIR}/bootstrap_cosmos.env\""
prime_log "running remote Cosmos bootstrap"
set +e
prime_remote_bash "${SSH_KEY}" "${SSH_TARGET}" "${SSH_PORT}" "${REMOTE_CMD}"
REMOTE_STATUS=$?
set -e

mkdir -p "${OUTPUT_DIR}/remote_artifacts"
set +e
prime_rsync_pull_dir "${SSH_KEY}" "${SSH_TARGET}" "${REMOTE_ARTIFACT_DIR}" "${OUTPUT_DIR}/remote_artifacts" "${SSH_PORT}"
SYNC_STATUS=$?
set -e
if [[ ${SYNC_STATUS} -ne 0 ]]; then
  prime_warn "failed to sync remote artifacts from ${REMOTE_ARTIFACT_DIR}"
fi

SSH_SNIPPET="${OUTPUT_DIR}/ssh_config_snippet.txt"
prime_write_ssh_config_snippet "${SSH_SNIPPET}" "${NAME}" "${HOST_ONLY}" "${SSH_USER}" "${SSH_KEY}" "${SSH_PORT}"

python3 - "${OUTPUT_DIR}" "${MODE}" "${NAME}" "${POD_ID}" "${HOST_ONLY}" "${SSH_USER}" "${REMOTE_CHECKOUT}" "${REMOTE_COSMOS_SOURCE}" "${REMOTE_ARTIFACT_DIR}" "${SCENARIO}" "${CARLA_PORT}" "${SERVER_PORT}" "${PROMPT}" "${TIMEOUT}" "${LOCAL_REPO_REV}" "${REMOTE_STATUS}" "${SYNC_STATUS}" <<'PY'
import json
import sys
from pathlib import Path

(
    output_dir,
    mode,
    name,
    pod_id,
    host,
    ssh_user,
    remote_checkout,
    remote_cosmos_source,
    remote_artifact_dir,
    scenario,
    carla_port,
    server_port,
    prompt,
    timeout,
    repo_rev,
    remote_status,
    sync_status,
) = sys.argv[1:]

root = Path(output_dir)
remote_dir = root / "remote_artifacts"
data = {
    "renderer": "cosmos",
    "mode": mode,
    "pod_name": name,
    "pod_id": pod_id or None,
    "host": host,
    "ssh_user": ssh_user,
    "remote_checkout": remote_checkout,
    "remote_cosmos_source": remote_cosmos_source,
    "remote_artifact_dir": remote_artifact_dir,
    "scenario": scenario,
    "carla_port": int(carla_port),
    "server_port": int(server_port),
    "prompt": prompt,
    "timeout": float(timeout),
    "local_repo_rev": repo_rev or None,
    "remote_status": int(remote_status),
    "sync_status": int(sync_status),
}

for filename in ("smoke_result.json", "cosmos_health.json"):
    path = remote_dir / filename
    if path.exists():
        data[path.stem] = json.loads(path.read_text())

manifest_path = remote_dir / "remote_manifest.txt"
if manifest_path.exists():
    data["remote_manifest_path"] = str(manifest_path)

ready_path = remote_dir / "ready_commands.txt"
if ready_path.exists():
    data["ready_commands_path"] = str(ready_path)

status_path = root / "prime_status.json"
if status_path.exists():
    data["prime_status"] = json.loads(status_path.read_text())

(root / "metadata.json").write_text(json.dumps(data, indent=2) + "\n")
PY

GPU_LINE=""
if [[ -f "${OUTPUT_DIR}/remote_artifacts/remote_manifest.txt" ]]; then
  GPU_LINE="$(awk '/^nvidia_smi:/{getline; print; exit}' "${OUTPUT_DIR}/remote_artifacts/remote_manifest.txt" || true)"
fi

prime_log "mode: ${MODE}"
if [[ -n "${POD_ID}" ]]; then
  prime_log "pod id: ${POD_ID}"
fi
prime_log "host: ${HOST_ONLY}"
prime_log "ssh user: ${SSH_USER}"
if [[ -n "${GPU_LINE}" ]]; then
  prime_log "gpu/driver: ${GPU_LINE}"
fi
prime_log "service endpoint: carla=127.0.0.1:${CARLA_PORT} cosmos=http://127.0.0.1:${SERVER_PORT}"
prime_log "smoke status: $([[ ${REMOTE_STATUS} -eq 0 ]] && printf 'passed' || printf 'failed')"

if [[ -f "${OUTPUT_DIR}/remote_artifacts/ready_commands.txt" ]]; then
  prime_log "ready commands:"
  cat "${OUTPUT_DIR}/remote_artifacts/ready_commands.txt"
fi

prime_log "local artifacts: ${OUTPUT_DIR}"
prime_log "remote checkout: ${REMOTE_CHECKOUT}"
prime_log "remote cosmos source: ${REMOTE_COSMOS_SOURCE}"
prime_log "ssh target: ${SSH_TARGET}"

if [[ "${TERMINATE}" != "1" && -n "${CREATED_POD_ID}" ]]; then
  prime_log "manual teardown: prime --plain pods terminate ${CREATED_POD_ID} --yes"
fi

if [[ ${REMOTE_STATUS} -ne 0 ]]; then
  prime_die "remote Cosmos bootstrap failed; see ${OUTPUT_DIR}/remote_artifacts"
fi

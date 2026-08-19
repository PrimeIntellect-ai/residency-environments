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
  scripts/e2e_prime_nurec.sh [options]

Modes:
  --mode rent                  Rent a fresh Prime pod, deploy NuRec, smoke-check it, sync artifacts.
  --mode attach                Attach to an existing SSH host and deploy there.

Common options:
  --name <pod-name>            Prime pod name in rent mode. Default: nurec-<timestamp>
  --ssh-key <path>             SSH private key. Default: ~/.ssh/claude_prime_ed25519
  --user <ssh-user>            SSH login user. Default: ubuntu
  --ssh-port <port>            SSH port. Default: 22, or pod-reported port in rent mode.
  --host <ip-or-host>          Required in attach mode.
  --repo-root <path>           Local repo root. Default: current repo root
  --remote-checkout-name <n>   Remote checkout directory name. Default: carla-env
  --output-dir <path>          Local artifact dir. Default: outputs/prime_e2e/nurec/<timestamp>
  --terminate                  Terminate a newly rented pod on script exit

Prime rent options:
  --node-id <prime-node-id>    Exact Prime availability id to rent
  --gpu <query>                GPU preference list. Default: L40S 48GB,A100 80GB
  --disk-size <gb>             Pod disk size. Default: 200
  --image <name>               Prime pod image. Default: ubuntu_22_cuda_12
  --custom-template-id <id>    Prime custom template id. Overrides --image.
  --user <ssh-user>            SSH login user. Default: ubuntu. In rent mode with
                               --custom-template-id, this value is used directly
                               instead of Prime's stale default pod metadata.
  --ssh-port <port>            SSH port. Default: 22. In rent mode with
                               --custom-template-id, this value is used directly
                               instead of Prime's stale default pod metadata.

NuRec options:
  --scene-path <path>          Required. Remote USDZ path, or dataset-relative path inside the
                               NuRec HF dataset. Default dataset layout is sample_set/25.07_release/...
  --scene-dataset <repo-id>    HF dataset repo. Default: nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
  --scene-revision <rev>       HF dataset revision/tag. Default: auto from scene path (25.07, 26.02, or main)
  --scenario <name>            Repo scenario for smoke check. Default: free_roam
  --camera-logical-id <id>     Preferred NuRec reconstruction camera logical ID.
                               Default: camera_front_wide_120fov
  --carla-port <port>          CARLA RPC port. Default: 2000
  --nurec-port <port>          NuRec gRPC port. Default: 46435
  --resolution-ratio <float>   NuRec render ratio. Default: 0.25
  --framerate <float>          NuRec framerate. Default: 20
  --carla-root <remote-path>   Optional preinstalled CARLA root on the pod image.

Examples:
  scripts/e2e_prime_nurec.sh \
    --mode rent \
    --scene-path sample_set/25.07_release/Batch0001/026d6a39-bd8f-4175-bc61-fe50ed0403a3/026d6a39-bd8f-4175-bc61-fe50ed0403a3.usdz

  scripts/e2e_prime_nurec.sh \
    --mode attach \
    --host 203.0.113.10 \
    --scene-path sample_set/25.07_release/Batch0001/026d6a39-bd8f-4175-bc61-fe50ed0403a3/026d6a39-bd8f-4175-bc61-fe50ed0403a3.usdz
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
OUTPUT_DIR=""
TERMINATE="0"
NODE_ID=""
GPU_QUERY="L40S 48GB,A100 80GB"
DISK_SIZE="200"
IMAGE="ubuntu_22_cuda_12"
CUSTOM_TEMPLATE_ID=""
SCENE_PATH=""
SCENARIO="free_roam"
NUREC_CAMERA_LOGICAL_ID="camera_front_wide_120fov"
CARLA_PORT="2000"
NUREC_PORT="46435"
RESOLUTION_RATIO="0.25"
FRAMERATE="20"
CARLA_ROOT=""
SCENE_DATASET="nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
SCENE_REVISION=""

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
    --scene-path)
      SCENE_PATH="${2:?missing value for --scene-path}"
      shift 2
      ;;
    --scene-dataset)
      SCENE_DATASET="${2:?missing value for --scene-dataset}"
      shift 2
      ;;
    --scene-revision)
      SCENE_REVISION="${2:?missing value for --scene-revision}"
      shift 2
      ;;
    --scenario)
      SCENARIO="${2:?missing value for --scenario}"
      shift 2
      ;;
    --camera-logical-id)
      NUREC_CAMERA_LOGICAL_ID="${2:?missing value for --camera-logical-id}"
      shift 2
      ;;
    --carla-port)
      CARLA_PORT="${2:?missing value for --carla-port}"
      shift 2
      ;;
    --nurec-port)
      NUREC_PORT="${2:?missing value for --nurec-port}"
      shift 2
      ;;
    --resolution-ratio)
      RESOLUTION_RATIO="${2:?missing value for --resolution-ratio}"
      shift 2
      ;;
    --framerate)
      FRAMERATE="${2:?missing value for --framerate}"
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
  NAME="nurec-${RUN_ID,,}"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/outputs/prime_e2e/nurec/${RUN_ID}"
fi

if [[ -z "${SCENE_PATH}" ]]; then
  prime_die "--scene-path is required"
fi
if [[ -f "${SCENE_PATH}" ]]; then
  prime_die "local scene upload is no longer supported; pass a remote path or an HF dataset-relative scene path"
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
  prime_warn "no HF token found locally; HF scene download will fail unless the remote already has the requested scene"
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

TMP_DIR="$(mktemp -d -t prime-e2e-nurec.XXXXXX)"

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
REMOTE_ARTIFACT_DIR="${REMOTE_CHECKOUT}/outputs/prime_e2e/nurec/${RUN_ID}"
REMOTE_TMP_DIR="${REMOTE_CHECKOUT}/.prime_e2e/${RUN_ID}"
REMOTE_SCENE_SPEC="${SCENE_PATH}"

prime_log "syncing repo checkout to ${SSH_TARGET}:${REMOTE_CHECKOUT}"
prime_sync_repo_checkout "${SSH_KEY}" "${REPO_ROOT}" "${SSH_TARGET}" "${REMOTE_CHECKOUT}" "${SSH_PORT}"
prime_remote_bash "${SSH_KEY}" "${SSH_TARGET}" "${SSH_PORT}" "mkdir -p \"${REMOTE_ARTIFACT_DIR}\" \"${REMOTE_TMP_DIR}\""

LOCAL_REPO_REV="$(prime_git_head "${REPO_ROOT}")"
BOOTSTRAP_ENV="${TMP_DIR}/bootstrap_nurec.env"
BOOTSTRAP_SCRIPT="${TMP_DIR}/bootstrap_nurec.sh"

: >"${BOOTSTRAP_ENV}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_HOME "${REMOTE_HOME}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_CHECKOUT "${REMOTE_CHECKOUT}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_ARTIFACT_DIR "${REMOTE_ARTIFACT_DIR}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_TMP_DIR "${REMOTE_TMP_DIR}"
prime_env_set "${BOOTSTRAP_ENV}" REMOTE_LOGIN_USER "${SSH_USER}"
prime_env_set "${BOOTSTRAP_ENV}" LOCAL_REPO_REV "${LOCAL_REPO_REV}"
prime_env_set "${BOOTSTRAP_ENV}" NUREC_SCENE_SPEC "${REMOTE_SCENE_SPEC}"
prime_env_set "${BOOTSTRAP_ENV}" NUREC_SCENE_DATASET "${SCENE_DATASET}"
prime_env_set "${BOOTSTRAP_ENV}" NUREC_SCENE_DATASET_REVISION "${SCENE_REVISION}"
prime_env_set "${BOOTSTRAP_ENV}" NUREC_SCENARIO "${SCENARIO}"
prime_env_set "${BOOTSTRAP_ENV}" NUREC_CAMERA_LOGICAL_ID "${NUREC_CAMERA_LOGICAL_ID}"
prime_env_set "${BOOTSTRAP_ENV}" CARLA_PORT "${CARLA_PORT}"
prime_env_set "${BOOTSTRAP_ENV}" NUREC_PORT "${NUREC_PORT}"
prime_env_set "${BOOTSTRAP_ENV}" NUREC_RESOLUTION_RATIO "${RESOLUTION_RATIO}"
prime_env_set "${BOOTSTRAP_ENV}" NUREC_FRAMERATE "${FRAMERATE}"
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
  printf '[nurec-bootstrap] %s\n' "$*"
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

run_root_bash() {
  if [[ "$(id -u)" -eq 0 ]]; then
    bash -lc "$*"
  elif command -v sudo >/dev/null 2>&1; then
    sudo -n bash -lc "$*"
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
  "${py}" -m pip install --upgrade pip "setuptools<81" wheel
  "${py}" -m pip install -e "${REMOTE_CHECKOUT}[carla9,nurec]" "imageio[ffmpeg]" "huggingface_hub[cli]"
  "${py}" -m pip install nvidia-nvimgcodec-cu12 nvidia-nvjpeg-cu12 || true
}

ensure_nurec_sdk() {
  local sdk_root="${REMOTE_HOME}/nurec"
  local carla_repo="${sdk_root}/carla_repo"
  mkdir -p "${sdk_root}"
  if [[ ! -d "${carla_repo}/.git" ]]; then
    log "cloning CARLA NuRec SDK checkout"
    git clone --filter=blob:none --no-checkout https://github.com/carla-simulator/carla.git "${carla_repo}"
  fi
  git -C "${carla_repo}" fetch --depth 1 origin ue4/0.9.16
  git -C "${carla_repo}" sparse-checkout init --cone
  git -C "${carla_repo}" sparse-checkout set PythonAPI/examples/nvidia/nurec
  git -C "${carla_repo}" checkout ue4/0.9.16

  NUREC_SDK_PATH="${carla_repo}/PythonAPI/examples/nvidia/nurec"
  export NUREC_SDK_PATH
  local py="${REMOTE_CHECKOUT}/.venv/bin/python"
  # Install the NuRec Python deps without the SDK's legacy protobuf/grpc pins.
  # The exact requirement files downgrade protobuf enough to break verifiers on
  # the repo side, and grpcio-tools==1.56.0 has no usable cp312 wheel.
  "${py}" -m pip install \
    pygame==2.6.1 \
    numpy==2.2.5 \
    scipy==1.15.3 \
    "grpcio>=1.64.0" \
    "grpcio-tools>=1.64.0" \
    "protobuf>=4.25.0"
  (cd "${NUREC_SDK_PATH}" && "${py}" nre/grpc/update_generated.py)

  if docker_available; then
    docker_cmd pull carlasim/carla:0.9.16 >/dev/null
    docker_cmd pull carlasimulator/nvidia-nurec-grpc:0.2.0 >/dev/null
  fi
}

resolve_scene_path() {
  local spec="${NUREC_SCENE_SPEC}"
  local default_root="${REMOTE_HOME}/nurec/scenes_25_07"
  local py="${REMOTE_CHECKOUT}/.venv/bin/python"
  if [[ -f "${spec}" ]]; then
    printf '%s\n' "${spec}"
    return 0
  fi
  if [[ -f "${default_root}/${spec}" ]]; then
    printf '%s\n' "${default_root}/${spec}"
    return 0
  fi
  local base_name
  base_name="$(basename "${spec}")"
  local found=""
  found="$(find "${default_root}" -type f -name "${base_name}" 2>/dev/null | head -n 1 || true)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi

  [[ -n "${HF_TOKEN:-}" ]] || return 1
  mkdir -p "${default_root}"
  local revision="${NUREC_SCENE_DATASET_REVISION:-}"
  if [[ -z "${revision}" ]]; then
    case "${spec}" in
      *25.07_release/*) revision="25.07" ;;
      *26.02_release/*) revision="26.02" ;;
      *) revision="main" ;;
    esac
  fi
  printf '[nurec-bootstrap] downloading NuRec scene from HF dataset %s@%s: %s\n' "${NUREC_SCENE_DATASET}" "${revision}" "${spec}" >&2
  "${py}" - "${NUREC_SCENE_DATASET}" "${revision}" "${spec}" "${default_root}" <<'PY'
from pathlib import Path
import os
import sys

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
revision = sys.argv[2]
scene_spec = sys.argv[3]
local_dir = Path(sys.argv[4])
scene_path = Path(scene_spec)
if scene_path.is_absolute():
    raise SystemExit("scene spec for HF download must be dataset-relative, not absolute")
if scene_path.suffix != ".usdz" or len(scene_path.parts) < 2:
    raise SystemExit(
        "scene spec must be a dataset-relative .usdz path such as "
        "sample_set/25.07_release/Batch0001/<clip>/<clip>.usdz"
    )

scene_dir = scene_path.parent.as_posix()
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is not configured")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision=revision,
    allow_patterns=[f"{scene_dir}/*"],
    local_dir=str(local_dir),
    token=token,
)
PY
  if [[ -f "${default_root}/${spec}" ]]; then
    printf '%s\n' "${default_root}/${spec}"
    return 0
  fi
  return 1
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

collect_logs() {
  set +e
  local launch_user
  launch_user="$(carla_launch_user)"
  if docker_available; then
    docker_cmd ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' >"${REMOTE_ARTIFACT_DIR}/docker_ps.txt" 2>/dev/null || true
    docker_cmd logs "carla-server-0916-${CARLA_PORT}" >"${REMOTE_ARTIFACT_DIR}/carla_docker.log" 2>&1 || true
    local nurec_container
    nurec_container="$(docker_cmd ps --format '{{.Names}}\t{{.Image}}' | awk '$2 ~ /nvidia-nurec-grpc/ {print $1; exit}')"
    if [[ -n "${nurec_container}" ]]; then
      docker_cmd logs "${nurec_container}" >"${REMOTE_ARTIFACT_DIR}/nurec_backend.log" 2>&1 || true
      printf '%s\n' "${nurec_container}" >"${REMOTE_ARTIFACT_DIR}/nurec_container_name.txt"
    fi
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
    echo "renderer=nurec"
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "user=$(whoami)"
    echo "remote_home=${REMOTE_HOME}"
    echo "remote_checkout=${REMOTE_CHECKOUT}"
    echo "local_repo_rev=${LOCAL_REPO_REV}"
    echo "carla_port=${CARLA_PORT}"
    echo "nurec_port=${NUREC_PORT}"
    echo "nurec_camera_logical_id=${NUREC_CAMERA_LOGICAL_ID}"
    echo "nurec_scene_dataset=${NUREC_SCENE_DATASET}"
    echo "nurec_scene_dataset_revision=${NUREC_SCENE_DATASET_REVISION:-}"
    echo "nurec_scene_spec=${NUREC_SCENE_SPEC}"
    echo "nurec_scene_path=${NUREC_SCENE_PATH_RESOLVED:-}"
    echo "nurec_sdk_path=${NUREC_SDK_PATH:-}"
    echo "repo_python=$("${REMOTE_CHECKOUT}/.venv/bin/python" --version 2>&1 || true)"
    echo "nvidia_smi:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  } >"${manifest}"
}

write_ready_commands() {
  cat >"${REMOTE_ARTIFACT_DIR}/ready_commands.txt" <<EOF2
cd "${REMOTE_CHECKOUT}"
source .venv/bin/activate
export NUREC_SDK_PATH="${NUREC_SDK_PATH:-${REMOTE_HOME}/nurec/carla_repo/PythonAPI/examples/nvidia/nurec}"

vf-eval carla_env -m "openai/gpt-4.1-mini" \\
  -a '{
    "scenario": "${NUREC_SCENARIO}",
    "sandbox": {"mode": "disabled"},
    "host": "127.0.0.1",
    "port": ${CARLA_PORT},
    "enable_nurec": true,
    "nurec_scene_path": "${NUREC_SCENE_PATH_RESOLVED:-${NUREC_SCENE_SPEC}}",
    "nurec_camera_logical_id": "${NUREC_CAMERA_LOGICAL_ID}",
    "nurec_mode": "replay",
    "nurec": {
      "grpc_port": ${NUREC_PORT},
      "camera_logical_id": "${NUREC_CAMERA_LOGICAL_ID}",
      "resolution_ratio": ${NUREC_RESOLUTION_RATIO},
      "framerate": ${NUREC_FRAMERATE}
    }
  }' -n 1 -r 1

python - <<'PY'
from carla_env import load_environment

env = load_environment(
    scenario="${NUREC_SCENARIO}",
    sandbox={"mode": "disabled"},
    host="127.0.0.1",
    port=${CARLA_PORT},
    enable_nurec=True,
    nurec_scene_path="${NUREC_SCENE_PATH_RESOLVED:-${NUREC_SCENE_SPEC}}",
    nurec_camera_logical_id="${NUREC_CAMERA_LOGICAL_ID}",
    nurec_mode="replay",
    nurec={
        "grpc_port": ${NUREC_PORT},
        "camera_logical_id": "${NUREC_CAMERA_LOGICAL_ID}",
        "resolution_ratio": ${NUREC_RESOLUTION_RATIO},
        "framerate": ${NUREC_FRAMERATE},
    },
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
ensure_nurec_sdk
export_hf_auth

NUREC_SCENE_PATH_RESOLVED="$(resolve_scene_path)" || die "NuRec scene not found: ${NUREC_SCENE_SPEC}"
export NUREC_SCENE_PATH_RESOLVED
export NUREC_SDK_PATH
export NUREC_IMAGE="carlasimulator/nvidia-nurec-grpc:0.2.0"

ensure_carla

SMOKE_JSON="${REMOTE_ARTIFACT_DIR}/smoke_result.json"
SMOKE_LOG="${REMOTE_ARTIFACT_DIR}/smoke.log"
SMOKE_VIDEO_DIR="${REMOTE_ARTIFACT_DIR}/smoke_video"
SMOKE_CAMERA_ARGS=()
if [[ -n "${NUREC_CAMERA_LOGICAL_ID}" ]]; then
  SMOKE_CAMERA_ARGS+=(--camera-logical-id "${NUREC_CAMERA_LOGICAL_ID}")
fi

set +e
"${REMOTE_CHECKOUT}/.venv/bin/python" "${REMOTE_CHECKOUT}/scripts/nurec_smoke_check.py" \
  --host 127.0.0.1 \
  --port "${CARLA_PORT}" \
  --scene "${NUREC_SCENE_PATH_RESOLVED}" \
  --scenario "${NUREC_SCENARIO}" \
  --framerate "${NUREC_FRAMERATE}" \
  --grpc-port "${NUREC_PORT}" \
  --resolution-ratio "${NUREC_RESOLUTION_RATIO}" \
  "${SMOKE_CAMERA_ARGS[@]}" \
  --startup-timeout 240 \
  --record-video \
  --output-dir "${SMOKE_VIDEO_DIR}" >"${SMOKE_JSON}" 2>"${SMOKE_LOG}"
SMOKE_STATUS=$?
set -e

[[ -f "${SMOKE_JSON}" ]] && cat "${SMOKE_JSON}"
if [[ ${SMOKE_STATUS} -ne 0 ]]; then
  tail -n 120 "${SMOKE_LOG}" >&2 || true
  exit "${SMOKE_STATUS}"
fi

log "NuRec smoke check completed successfully"
EOF

prime_scp "${SSH_KEY}" "${BOOTSTRAP_ENV}" "${SSH_TARGET}:${REMOTE_TMP_DIR}/bootstrap_nurec.env" "${SSH_PORT}"
prime_scp "${SSH_KEY}" "${BOOTSTRAP_SCRIPT}" "${SSH_TARGET}:${REMOTE_TMP_DIR}/bootstrap_nurec.sh" "${SSH_PORT}"

REMOTE_CMD="chmod +x \"${REMOTE_TMP_DIR}/bootstrap_nurec.sh\" && \"${REMOTE_TMP_DIR}/bootstrap_nurec.sh\" \"${REMOTE_TMP_DIR}/bootstrap_nurec.env\""
prime_log "running remote NuRec bootstrap"
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

python3 - "${OUTPUT_DIR}" "${MODE}" "${NAME}" "${POD_ID}" "${HOST_ONLY}" "${SSH_USER}" "${REMOTE_CHECKOUT}" "${REMOTE_ARTIFACT_DIR}" "${SCENE_PATH}" "${REMOTE_SCENE_SPEC}" "${SCENARIO}" "${CARLA_PORT}" "${NUREC_PORT}" "${RESOLUTION_RATIO}" "${FRAMERATE}" "${LOCAL_REPO_REV}" "${REMOTE_STATUS}" "${SYNC_STATUS}" <<'PY'
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
    remote_artifact_dir,
    scene_spec,
    remote_scene_spec,
    scenario,
    carla_port,
    nurec_port,
    resolution_ratio,
    framerate,
    repo_rev,
    remote_status,
    sync_status,
) = sys.argv[1:]

root = Path(output_dir)
remote_dir = root / "remote_artifacts"
data = {
    "renderer": "nurec",
    "mode": mode,
    "pod_name": name,
    "pod_id": pod_id or None,
    "host": host,
    "ssh_user": ssh_user,
    "remote_checkout": remote_checkout,
    "remote_artifact_dir": remote_artifact_dir,
    "scene_spec": scene_spec,
    "remote_scene_spec": remote_scene_spec,
    "scenario": scenario,
    "carla_port": int(carla_port),
    "nurec_port": int(nurec_port),
    "resolution_ratio": float(resolution_ratio),
    "framerate": float(framerate),
    "local_repo_rev": repo_rev or None,
    "remote_status": int(remote_status),
    "sync_status": int(sync_status),
}

for filename in ("smoke_result.json",):
    path = remote_dir / filename
    if path.exists():
        text = path.read_text().strip()
        if text:
            data["smoke_result"] = json.loads(text)
        else:
            data["smoke_result_error"] = f"empty JSON file: {path}"

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
prime_log "service endpoint: carla=127.0.0.1:${CARLA_PORT} nurec=127.0.0.1:${NUREC_PORT}"
prime_log "smoke status: $([[ ${REMOTE_STATUS} -eq 0 ]] && printf 'passed' || printf 'failed')"

if [[ -f "${OUTPUT_DIR}/remote_artifacts/ready_commands.txt" ]]; then
  prime_log "ready commands:"
  cat "${OUTPUT_DIR}/remote_artifacts/ready_commands.txt"
fi

prime_log "local artifacts: ${OUTPUT_DIR}"
prime_log "remote checkout: ${REMOTE_CHECKOUT}"
prime_log "ssh target: ${SSH_TARGET}"

if [[ "${TERMINATE}" != "1" && -n "${CREATED_POD_ID}" ]]; then
  prime_log "manual teardown: prime --plain pods terminate ${CREATED_POD_ID} --yes"
fi

if [[ ${REMOTE_STATUS} -ne 0 ]]; then
  prime_die "remote NuRec bootstrap failed; see ${OUTPUT_DIR}/remote_artifacts"
fi

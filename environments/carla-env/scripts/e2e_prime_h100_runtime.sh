#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/prime_pod_common.sh"
PRIME_LOG_PREFIX="$(basename "$0")"

usage() {
  cat <<'EOF'
Usage:
  scripts/e2e_prime_h100_runtime.sh [options]

Run the validated published-image Prime flow in one command:
  Default flow:
    1. Rent a plain H100 host pod
    2. Start sinatras/carla-env-runtime inside it
    3. Start CARLA inside that runtime container
    4. Wait for strict CARLA RPC readiness
    5. Run NuRec and/or Cosmos via the existing attach scripts
    6. Sync artifacts back locally
    7. Terminate the host pod by default

  Custom template flow:
    1. Rent a Prime custom-template pod directly
    2. Attach to the pod SSH endpoint
    3. Run NuRec and/or Cosmos via the existing attach scripts
    4. Sync artifacts back locally
    5. Terminate the pod by default

Options:
  --stack <mode>              Which stack to run: nurec, cosmos, or both. Default: both
  --name <pod-name>           Prime pod name. Default: h100-runtime-<timestamp>
  --ssh-key <path>            SSH private key. Default: ~/.ssh/claude_prime_ed25519
  --repo-root <path>          Local repo root. Default: current repo root
  --output-dir <path>         Local artifact dir. Default: outputs/prime_e2e/h100_runtime/<timestamp>
  --node-id <prime-node-id>   Exact Prime availability id to rent
  --gpu <query>               GPU query. Default: H100 80GB
  --provider-preference <csv> Provider preference order. Default: massedcompute
  --disk-size <gb>            Host pod disk size. Default: 200
  --host-image <name>         Prime host image. Default: ubuntu_22_cuda_12
  --custom-template-id <id>   Rent the Prime custom template directly and skip the
                               inner runtime-container bootstrap.
  --template-ssh-user <user>  SSH login user for custom-template pods.
                               Default: root
  --template-ssh-port <port>  SSH port for custom-template pods.
                               Default: 2222
  --runtime-image <ref>       Published runtime image. Default: sinatras/carla-env-runtime:0.1.0
  --container-name <name>     Inner tooling container name. Default: runtime-<timestamp>
  --inner-ssh-port <port>     Host port bound to inner SSH. Default: 2222
  --carla-port <port>         CARLA RPC port. Default: 2000
  --nurec-port <port>         NuRec gRPC port. Default: 46435
  --cosmos-port <port>        Cosmos HTTP port. Default: 8080
  --scene-path <path>         NuRec HF dataset-relative scene path.
                               Default: validated 25.07 sample scene
  --camera-logical-id <id>    NuRec camera logical id. Default: camera_front_wide_120fov
  --cosmos-source <path>      Optional local cosmos-transfer2.5 checkout/snapshot
  --keep-host                 Do not terminate the rented host pod on exit
  -h, --help                  Show this help

Examples:
  scripts/e2e_prime_h100_runtime.sh

  scripts/e2e_prime_h100_runtime.sh --stack nurec
EOF
}

STACK="both"
NAME=""
SSH_KEY="${HOME}/.ssh/claude_prime_ed25519"
REPO_ROOT="${REPO_ROOT_DEFAULT}"
OUTPUT_DIR=""
NODE_ID=""
GPU_QUERY="H100 80GB"
PROVIDER_PREFERENCE="massedcompute"
DISK_SIZE="200"
HOST_IMAGE="ubuntu_22_cuda_12"
CUSTOM_TEMPLATE_ID=""
TEMPLATE_SSH_USER="root"
TEMPLATE_SSH_PORT="2222"
RUNTIME_IMAGE="sinatras/carla-env-runtime:0.1.0"
RUN_ID="$(prime_now_utc)"
CONTAINER_NAME="runtime-${RUN_ID,,}"
INNER_SSH_PORT="2222"
CARLA_PORT="2000"
NUREC_PORT="46435"
COSMOS_PORT="8080"
SCENE_PATH="sample_set/25.07_release/Batch0001/026d6a39-bd8f-4175-bc61-fe50ed0403a3/026d6a39-bd8f-4175-bc61-fe50ed0403a3.usdz"
CAMERA_LOGICAL_ID="camera_front_wide_120fov"
COSMOS_SOURCE=""
KEEP_HOST="0"

while (($# > 0)); do
  case "$1" in
    --stack) STACK="${2:?missing value for --stack}"; shift 2 ;;
    --name) NAME="${2:?missing value for --name}"; shift 2 ;;
    --ssh-key) SSH_KEY="${2:?missing value for --ssh-key}"; shift 2 ;;
    --repo-root) REPO_ROOT="${2:?missing value for --repo-root}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?missing value for --output-dir}"; shift 2 ;;
    --node-id) NODE_ID="${2:?missing value for --node-id}"; shift 2 ;;
    --gpu) GPU_QUERY="${2:?missing value for --gpu}"; shift 2 ;;
    --provider-preference) PROVIDER_PREFERENCE="${2:?missing value for --provider-preference}"; shift 2 ;;
    --disk-size) DISK_SIZE="${2:?missing value for --disk-size}"; shift 2 ;;
    --host-image) HOST_IMAGE="${2:?missing value for --host-image}"; shift 2 ;;
    --custom-template-id) CUSTOM_TEMPLATE_ID="${2:?missing value for --custom-template-id}"; shift 2 ;;
    --template-ssh-user) TEMPLATE_SSH_USER="${2:?missing value for --template-ssh-user}"; shift 2 ;;
    --template-ssh-port) TEMPLATE_SSH_PORT="${2:?missing value for --template-ssh-port}"; shift 2 ;;
    --runtime-image) RUNTIME_IMAGE="${2:?missing value for --runtime-image}"; shift 2 ;;
    --container-name) CONTAINER_NAME="${2:?missing value for --container-name}"; shift 2 ;;
    --inner-ssh-port) INNER_SSH_PORT="${2:?missing value for --inner-ssh-port}"; shift 2 ;;
    --carla-port) CARLA_PORT="${2:?missing value for --carla-port}"; shift 2 ;;
    --nurec-port) NUREC_PORT="${2:?missing value for --nurec-port}"; shift 2 ;;
    --cosmos-port) COSMOS_PORT="${2:?missing value for --cosmos-port}"; shift 2 ;;
    --scene-path) SCENE_PATH="${2:?missing value for --scene-path}"; shift 2 ;;
    --camera-logical-id) CAMERA_LOGICAL_ID="${2:?missing value for --camera-logical-id}"; shift 2 ;;
    --cosmos-source) COSMOS_SOURCE="${2:?missing value for --cosmos-source}"; shift 2 ;;
    --keep-host) KEEP_HOST="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) prime_die "unknown argument: $1" ;;
  esac
done

case "${STACK}" in
  nurec|cosmos|both) ;;
  *) prime_die "--stack must be nurec, cosmos, or both" ;;
esac

if [[ -z "${NAME}" ]]; then
  NAME="h100-runtime-${RUN_ID,,}"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/outputs/prime_e2e/h100_runtime/${RUN_ID}"
fi

SSH_KEY="$(prime_abspath "${SSH_KEY}")"
REPO_ROOT="$(prime_abspath "${REPO_ROOT}")"
OUTPUT_DIR="$(prime_abspath "${OUTPUT_DIR}")"
if [[ -n "${COSMOS_SOURCE}" ]]; then
  COSMOS_SOURCE="$(prime_abspath "${COSMOS_SOURCE}")"
fi

[[ -d "${REPO_ROOT}" ]] || prime_die "repo root does not exist: ${REPO_ROOT}"
[[ -f "${SSH_KEY}" ]] || prime_die "ssh key not found: ${SSH_KEY}"
if [[ -n "${COSMOS_SOURCE}" ]]; then
  [[ -d "${COSMOS_SOURCE}" ]] || prime_die "Cosmos source does not exist: ${COSMOS_SOURCE}"
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/wrapper.log") 2>&1

prime_require_cmd python3
prime_require_cmd ssh
prime_require_cmd scp
prime_require_cmd rsync
prime_require_cmd prime

export PRIME_PROVIDER_PREFERENCE="${PROVIDER_PREFERENCE}"

POD_ID=""
STATUS_JSON=""
HOST_ONLY=""
HOST_SSH_TARGET=""
HOST_SSH_PORT="22"
INNER_SSH_TARGET=""
ATTACH_HOST=""
ATTACH_USER="ubuntu"
ATTACH_SSH_PORT="22"
ATTACH_TARGET=""
HOST_SSH_TIMEOUT_S="600"

cleanup() {
  local status="$1"
  if [[ "${KEEP_HOST}" != "1" && -n "${POD_ID}" ]]; then
    prime_log "terminating pod ${POD_ID}"
    prime_terminate_pod "${POD_ID}" || prime_warn "failed to terminate pod ${POD_ID}"
  fi
  trap - EXIT
  exit "${status}"
}
trap 'cleanup "$?"' EXIT

wait_for_inner_carla_rpc() {
  local timeout_s="${1:-300}"
  local deadline=$((SECONDS + timeout_s))
  while ((SECONDS < deadline)); do
    if prime_remote_bash "${SSH_KEY}" "${INNER_SSH_TARGET}" "${INNER_SSH_PORT}" "
python3 - '${CARLA_PORT}' <<'PY' >/dev/null
import glob
import sys

port = int(sys.argv[1])
eggs = sorted(glob.glob('/workspace/PythonAPI/carla/dist/carla-*.egg'))
if eggs:
    sys.path.insert(0, eggs[0])
import carla

client = carla.Client('127.0.0.1', port)
client.set_timeout(5.0)
world = client.get_world()
_ = world.get_map().name
PY
" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

dump_runtime_debug() {
  prime_remote_bash "${SSH_KEY}" "${HOST_SSH_TARGET}" "${HOST_SSH_PORT}" "
if docker info >/dev/null 2>&1; then
  docker ps -a --filter name=${CONTAINER_NAME} --format 'container={{.Names}} status={{.Status}} image={{.Image}}'
  echo __LOGS__
  docker logs --tail 200 ${CONTAINER_NAME} || true
else
  sudo -n docker ps -a --filter name=${CONTAINER_NAME} --format 'container={{.Names}} status={{.Status}} image={{.Image}}'
  echo __LOGS__
  sudo -n docker logs --tail 200 ${CONTAINER_NAME} || true
fi
" >&2 || true
}

start_runtime_container_with_retries() {
  local attempts=3
  local delay_s=15
  local attempt

  : > "${OUTPUT_DIR}/start_runtime.log"
  prime_log "giving the host ${delay_s}s to settle before runtime bootstrap"
  sleep "${delay_s}"

  for ((attempt = 1; attempt <= attempts; attempt++)); do
    prime_log "runtime bootstrap attempt ${attempt}/${attempts}"
    {
      printf '=== attempt %d/%d %s ===\n' "${attempt}" "${attempts}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      "${SCRIPT_DIR}/start_published_prime_template_container.sh" \
        --host "${HOST_ONLY}" \
        --host-user "${HOST_SSH_TARGET%@*}" \
        --host-port "${HOST_SSH_PORT}" \
        --ssh-key "${SSH_KEY}" \
        --image "${RUNTIME_IMAGE}" \
        --container-name "${CONTAINER_NAME}" \
        --inner-ssh-port "${INNER_SSH_PORT}" \
        --carla-port "${CARLA_PORT}" \
        --nurec-port "${NUREC_PORT}" \
        --cosmos-port "${COSMOS_PORT}" \
        --start-carla
    } | tee -a "${OUTPUT_DIR}/start_runtime.log"
    if [[ "${PIPESTATUS[0]}" -eq 0 ]]; then
      return 0
    fi
    if ((attempt < attempts)); then
      prime_warn "runtime bootstrap attempt ${attempt} failed; retrying in ${delay_s}s"
      sleep "${delay_s}"
    fi
  done

  return 1
}

write_connection_summary() {
  cat > "${OUTPUT_DIR}/connection.txt" <<EOF
pod_id=${POD_ID}
host_ip=${HOST_ONLY}
host_ssh_target=${HOST_SSH_TARGET}
host_ssh_port=${HOST_SSH_PORT}
attach_target=${ATTACH_TARGET}
attach_host=${ATTACH_HOST}
attach_user=${ATTACH_USER}
attach_ssh_port=${ATTACH_SSH_PORT}
inner_ssh_target=${INNER_SSH_TARGET}
inner_ssh_port=${INNER_SSH_PORT}
custom_template_id=${CUSTOM_TEMPLATE_ID}
runtime_image=${RUNTIME_IMAGE}
container_name=${CONTAINER_NAME}
stack=${STACK}
prime_terminate_command=prime --plain pods terminate ${POD_ID} --yes
EOF
}

run_nurec() {
  local -a cmd=(
    "${SCRIPT_DIR}/e2e_prime_nurec.sh"
    --mode attach
    --host "${ATTACH_HOST}"
    --user "${ATTACH_USER}"
    --ssh-port "${ATTACH_SSH_PORT}"
    --ssh-key "${SSH_KEY}"
    --repo-root "${REPO_ROOT}"
    --output-dir "${OUTPUT_DIR}/nurec"
    --scene-path "${SCENE_PATH}"
    --camera-logical-id "${CAMERA_LOGICAL_ID}"
    --carla-port "${CARLA_PORT}"
    --nurec-port "${NUREC_PORT}"
  )
  prime_log "running NuRec attach flow"
  "${cmd[@]}"
}

run_cosmos() {
  local -a cmd=(
    "${SCRIPT_DIR}/e2e_prime_cosmos.sh"
    --mode attach
    --host "${ATTACH_HOST}"
    --user "${ATTACH_USER}"
    --ssh-port "${ATTACH_SSH_PORT}"
    --ssh-key "${SSH_KEY}"
    --repo-root "${REPO_ROOT}"
    --output-dir "${OUTPUT_DIR}/cosmos"
    --carla-port "${CARLA_PORT}"
    --server-port "${COSMOS_PORT}"
  )
  if [[ -n "${COSMOS_SOURCE}" ]]; then
    cmd+=(--cosmos-source "${COSMOS_SOURCE}")
  fi
  prime_log "running Cosmos attach flow"
  "${cmd[@]}"
}

prime_log "creating H100 host pod"
POD_ID="$(prime_create_pod "${NAME}" "${NODE_ID}" "${GPU_QUERY}" "${DISK_SIZE}" "${HOST_IMAGE}" "${CUSTOM_TEMPLATE_ID}")"
STATUS_JSON="$(prime_wait_for_pod_active "${POD_ID}" 2400)" || prime_die "Prime pod did not become ACTIVE in time"
printf '%s\n' "${STATUS_JSON}" > "${OUTPUT_DIR}/prime_status.json"

HOST_ONLY="$(printf '%s' "${STATUS_JSON}" | prime_json_field ip)"
if [[ -n "${CUSTOM_TEMPLATE_ID}" ]]; then
  HOST_SSH_TARGET="${TEMPLATE_SSH_USER}@${HOST_ONLY}"
  HOST_SSH_PORT="${TEMPLATE_SSH_PORT}"
  HOST_SSH_TIMEOUT_S="1800"
else
  HOST_SSH_TARGET="$(printf '%s' "${STATUS_JSON}" | prime_status_ssh_target ubuntu)"
  HOST_SSH_PORT="$(printf '%s' "${STATUS_JSON}" | prime_status_ssh_port)"
fi
[[ -n "${HOST_ONLY}" ]] || prime_die "failed to resolve host IP"
[[ -n "${HOST_SSH_TARGET}" ]] || prime_die "failed to resolve host SSH target"

prime_log "waiting for host SSH on ${HOST_SSH_TARGET}"
prime_wait_for_ssh "${SSH_KEY}" "${HOST_SSH_TARGET}" "${HOST_SSH_PORT}" "${HOST_SSH_TIMEOUT_S}" || prime_die "host SSH did not become ready"

if [[ -n "${CUSTOM_TEMPLATE_ID}" ]]; then
  ATTACH_HOST="${HOST_ONLY}"
  ATTACH_USER="${HOST_SSH_TARGET%@*}"
  ATTACH_SSH_PORT="${HOST_SSH_PORT}"
  ATTACH_TARGET="${HOST_SSH_TARGET}"
  INNER_SSH_TARGET=""
  prime_log "using custom template pod directly via SSH target ${ATTACH_TARGET}:${ATTACH_SSH_PORT}"
  write_connection_summary
else
  prime_log "starting published runtime container with CARLA in the same container"
  start_runtime_container_with_retries

  INNER_SSH_TARGET="ubuntu@${HOST_ONLY}"
  ATTACH_HOST="${HOST_ONLY}"
  ATTACH_USER="ubuntu"
  ATTACH_SSH_PORT="${INNER_SSH_PORT}"
  ATTACH_TARGET="${INNER_SSH_TARGET}"
  write_connection_summary

  prime_log "probing CARLA RPC readiness via inner SSH"
  if ! wait_for_inner_carla_rpc 120; then
    prime_warn "pre-attach CARLA RPC probe did not pass; continuing with attach flows, which perform the authoritative CARLA RPC wait after repo sync"
    dump_runtime_debug
  fi
fi

case "${STACK}" in
  nurec)
    run_nurec
    ;;
  cosmos)
    run_cosmos
    ;;
  both)
    run_nurec
    run_cosmos
    ;;
esac

cat <<EOF
validated H100 runtime flow completed
output_dir: ${OUTPUT_DIR}
pod_id: ${POD_ID}
host_ip: ${HOST_ONLY}
attach_target: ${ATTACH_TARGET} -p ${ATTACH_SSH_PORT}
EOF

if [[ "${KEEP_HOST}" == "1" ]]; then
  cat <<EOF
host preserved
terminate manually:
  prime --plain pods terminate ${POD_ID} --yes
EOF
fi

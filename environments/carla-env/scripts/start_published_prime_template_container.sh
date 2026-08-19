#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/start_published_prime_template_container.sh --host <host> [options]

Start the published carla-env Prime image inside an already rented Prime host
pod, with host-backed mounts for the inner user's home, inner Docker data, and
runtime cache roots.

Options:
  --host <host>               Required. Prime host IP or user@host.
  --host-user <user>          SSH login user for the host. Default: root
  --host-port <port>          SSH port for the host. Default: 22
  --ssh-key <path>            SSH private key. Default: ~/.ssh/claude_prime_ed25519
  --image <ref>               Container image. Default: sinatras/carla-env-runtime:0.1.0
  --container-name <name>     Inner container name. Default: carla-env-runtime
  --inner-ssh-port <port>     Host port bound to inner SSH. Default: 2222
  --carla-port <port>         Host CARLA RPC base port. Default: 2000
  --nurec-port <port>         Host NuRec gRPC port. Default: 46435
  --cosmos-port <port>        Host Cosmos HTTP port. Default: 8080
  --home-root <path>          Host-backed mount for /home/ubuntu. Default: host-user prime-home dir
  --docker-root <path>        Host-backed mount for /var/lib/docker. Default: host-user prime-docker dir
  --ephemeral-root <path>     Host-backed mount for /ephemeral. Default: host-user prime-ephemeral dir
  --start-carla               Start CARLA inside the main runtime container.
  --start-carla-sidecar       Deprecated alias for --start-carla.
  --skip-pull                 Do not docker pull before starting
  -h, --help                  Show this help
EOF
}

HOST=""
HOST_USER="root"
HOST_PORT="22"
SSH_KEY="${HOME}/.ssh/claude_prime_ed25519"
IMAGE="sinatras/carla-env-runtime:0.1.0"
CONTAINER_NAME="carla-env-runtime"
INNER_SSH_PORT="2222"
CARLA_PORT="2000"
NUREC_PORT="46435"
COSMOS_PORT="8080"
HOME_ROOT=""
DOCKER_ROOT=""
EPHEMERAL_ROOT=""
PULL_IMAGE="1"
START_CARLA="0"

while (($# > 0)); do
  case "$1" in
    --host) HOST="${2:?missing value for --host}"; shift 2 ;;
    --host-user) HOST_USER="${2:?missing value for --host-user}"; shift 2 ;;
    --host-port) HOST_PORT="${2:?missing value for --host-port}"; shift 2 ;;
    --ssh-key) SSH_KEY="${2:?missing value for --ssh-key}"; shift 2 ;;
    --image) IMAGE="${2:?missing value for --image}"; shift 2 ;;
    --container-name) CONTAINER_NAME="${2:?missing value for --container-name}"; shift 2 ;;
    --inner-ssh-port) INNER_SSH_PORT="${2:?missing value for --inner-ssh-port}"; shift 2 ;;
    --carla-port) CARLA_PORT="${2:?missing value for --carla-port}"; shift 2 ;;
    --nurec-port) NUREC_PORT="${2:?missing value for --nurec-port}"; shift 2 ;;
    --cosmos-port) COSMOS_PORT="${2:?missing value for --cosmos-port}"; shift 2 ;;
    --home-root) HOME_ROOT="${2:?missing value for --home-root}"; shift 2 ;;
    --docker-root) DOCKER_ROOT="${2:?missing value for --docker-root}"; shift 2 ;;
    --ephemeral-root) EPHEMERAL_ROOT="${2:?missing value for --ephemeral-root}"; shift 2 ;;
    --start-carla|--start-carla-sidecar) START_CARLA="1"; shift ;;
    --skip-pull) PULL_IMAGE="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "${HOST}" ]] || { usage >&2; exit 1; }
[[ -f "${SSH_KEY}" ]] || { echo "ssh key not found: ${SSH_KEY}" >&2; exit 1; }

if [[ "${HOST}" == *@* ]]; then
  HOST_USER="${HOST%@*}"
  HOST="${HOST#*@}"
fi

HOST_TARGET="${HOST_USER}@${HOST}"
REMOTE_SSH_DIR="/root/.ssh"
REMOTE_BASE_DIR="/root"
if [[ "${HOST_USER}" != "root" ]]; then
  REMOTE_SSH_DIR="/home/${HOST_USER}/.ssh"
  REMOTE_BASE_DIR="/home/${HOST_USER}"
fi
if [[ -z "${HOME_ROOT}" ]]; then
  HOME_ROOT="${REMOTE_BASE_DIR}/prime-home"
fi
if [[ -z "${DOCKER_ROOT}" ]]; then
  DOCKER_ROOT="${REMOTE_BASE_DIR}/prime-docker"
fi
if [[ -z "${EPHEMERAL_ROOT}" ]]; then
  EPHEMERAL_ROOT="${REMOTE_BASE_DIR}/prime-ephemeral"
fi

SSH_OPTS=(
  -i "${SSH_KEY}"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -p "${HOST_PORT}"
)
SCP_OPTS=(
  -i "${SSH_KEY}"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -P "${HOST_PORT}"
)

read -r -d '' REMOTE_SCRIPT <<EOF || true
set -euo pipefail
run_root() {
  if [[ "\$(id -u)" -eq 0 ]]; then
    "\$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo -n "\$@"
  else
    echo "sudo is required for host bootstrap" >&2
    exit 1
  fi
}
docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "\$@"
  else
    run_root docker "\$@"
  fi
}
run_root mkdir -p "${HOME_ROOT}" "${DOCKER_ROOT}" "${EPHEMERAL_ROOT}"
run_root chown 1001:1001 "${HOME_ROOT}"
run_root chmod 755 "${HOME_ROOT}"
run_root chown root:root "${DOCKER_ROOT}" "${EPHEMERAL_ROOT}"
run_root chmod 711 "${DOCKER_ROOT}" "${EPHEMERAL_ROOT}"
docker_cmd rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
if [[ "${PULL_IMAGE}" == "1" ]]; then
  docker_cmd pull "${IMAGE}" >/dev/null
fi
docker_cmd run -d \
  --name "${CONTAINER_NAME}" \
  --hostname "${CONTAINER_NAME}" \
  --gpus all \
  --runtime nvidia \
  --privileged \
  --ipc=host \
  --shm-size=8g \
  -p "${INNER_SSH_PORT}:22" \
  -p "${CARLA_PORT}-$((CARLA_PORT + 2)):${CARLA_PORT}-$((CARLA_PORT + 2))" \
  -p "${NUREC_PORT}:${NUREC_PORT}" \
  -p "${COSMOS_PORT}:${COSMOS_PORT}" \
  -e START_DOCKER=1 \
  -e START_CARLA_IN_CONTAINER="${START_CARLA}" \
  -e CARLA_PORT_IN_CONTAINER="${CARLA_PORT}" \
  -e DOCKER_DATA_ROOT=/var/lib/docker \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e XDG_RUNTIME_DIR=/tmp/runtime-carla \
  -v "${REMOTE_SSH_DIR}:/host-ssh:ro" \
  -v "${HOME_ROOT}:/home/ubuntu" \
  -v "${DOCKER_ROOT}:/var/lib/docker" \
  -v "${EPHEMERAL_ROOT}:/ephemeral" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc 'set -euo pipefail
    mkdir -p /root/.ssh /home/ubuntu/.ssh
    cp /host-ssh/authorized_keys /root/.ssh/authorized_keys
    cp /host-ssh/authorized_keys /home/ubuntu/.ssh/authorized_keys
    chmod 700 /root/.ssh /home/ubuntu/.ssh
    chmod 600 /root/.ssh/authorized_keys /home/ubuntu/.ssh/authorized_keys
    chown -R ubuntu:ubuntu /home/ubuntu/.ssh
    /usr/local/bin/prime-template-entrypoint.sh &
    sshd_pid=\$!
    carla_pid=
    cleanup() {
      if [[ -n "\${carla_pid}" ]]; then
        kill "\${carla_pid}" >/dev/null 2>&1 || true
      fi
      kill "\${sshd_pid}" >/dev/null 2>&1 || true
      wait >/dev/null 2>&1 || true
    }
    trap cleanup EXIT INT TERM
    if [[ "\${START_CARLA_IN_CONTAINER:-0}" == "1" ]]; then
      mkdir -p /tmp/runtime-carla-carla
      chown carla:carla /tmp/runtime-carla-carla
      chmod 700 /tmp/runtime-carla-carla
      cat >/tmp/start-carla.sh <<"__CARLA__"
#!/usr/bin/env bash
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/runtime-carla-carla
if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  cp /usr/share/vulkan/icd.d/nvidia_icd.json /tmp/nvidia_icd.json
  export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
fi
cd /workspace
exec /bin/bash CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port="${CARLA_PORT}" -stdout -FullStdOutLogOutput -unattended
__CARLA__
      chmod 0755 /tmp/start-carla.sh
      chown carla:carla /tmp/start-carla.sh
      sudo -u carla -H /bin/bash /tmp/start-carla.sh &
      carla_pid=\$!
    fi
    wait_pids=("\${sshd_pid}")
    if [[ -n "\${carla_pid}" ]]; then
      wait_pids+=("\${carla_pid}")
    fi
    wait -n "\${wait_pids[@]}"'
docker_cmd ps --filter "name=${CONTAINER_NAME}" --format 'container={{.Names}} status={{.Status}} image={{.Image}}'
df -h / "${HOME_ROOT}" "${DOCKER_ROOT}" "${EPHEMERAL_ROOT}"
EOF

LOCAL_REMOTE_SCRIPT="$(mktemp -t start-published-prime.XXXXXX.sh)"
trap 'rm -f "${LOCAL_REMOTE_SCRIPT}"' EXIT
printf '%s\n' "${REMOTE_SCRIPT}" > "${LOCAL_REMOTE_SCRIPT}"
REMOTE_SCRIPT_PATH="/tmp/start-published-${CONTAINER_NAME}.sh"
scp "${SCP_OPTS[@]}" "${LOCAL_REMOTE_SCRIPT}" "${HOST_TARGET}:${REMOTE_SCRIPT_PATH}" >/dev/null
ssh "${SSH_OPTS[@]}" "${HOST_TARGET}" "bash ${REMOTE_SCRIPT_PATH@Q}; rm -f ${REMOTE_SCRIPT_PATH@Q}"
rm -f "${LOCAL_REMOTE_SCRIPT}"
trap - EXIT

deadline=$((SECONDS + 180))
while ((SECONDS < deadline)); do
  if ssh -p "${INNER_SSH_PORT}" -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "ubuntu@${HOST}" 'echo ready' >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ((SECONDS >= deadline)); then
  echo "inner container SSH did not become ready on ${HOST}:${INNER_SSH_PORT}" >&2
  exit 1
fi

if [[ "${START_CARLA}" == "1" ]]; then
  read -r -d '' REMOTE_CARLA_PORT_CHECK <<EOF || true
python3 - <<'PY'
import socket
import sys

s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(('127.0.0.1', ${CARLA_PORT}))
except OSError:
    sys.exit(1)
else:
    s.close()
    sys.exit(0)
PY
EOF

  carla_deadline=$((SECONDS + 240))
  while ((SECONDS < carla_deadline)); do
    if ssh -p "${INNER_SSH_PORT}" -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "ubuntu@${HOST}" \
      "${REMOTE_CARLA_PORT_CHECK}" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  if ((SECONDS >= carla_deadline)); then
    ssh "${SSH_OPTS[@]}" "${HOST_TARGET}" "if docker info >/dev/null 2>&1; then docker ps -a --filter name=${CONTAINER_NAME} --format 'container={{.Names}} status={{.Status}} image={{.Image}}'; echo __LOGS__; docker logs --tail 200 ${CONTAINER_NAME} || true; else sudo -n docker ps -a --filter name=${CONTAINER_NAME} --format 'container={{.Names}} status={{.Status}} image={{.Image}}'; echo __LOGS__; sudo -n docker logs --tail 200 ${CONTAINER_NAME} || true; fi" >&2 || true
    echo "CARLA did not become reachable on 127.0.0.1:${CARLA_PORT} inside ${CONTAINER_NAME}" >&2
    exit 1
  fi
fi

cat <<EOF
inner container is ready
host: ${HOST_TARGET}
inner ssh: ubuntu@${HOST} -p ${INNER_SSH_PORT}
carla in runtime container: ${START_CARLA}

example:
  ./scripts/e2e_prime_nurec_template.sh --mode attach --host ${HOST} --ssh-port ${INNER_SSH_PORT} --user ubuntu --ssh-key ${SSH_KEY} --scene-path sample_set/25.07_release/Batch0001/026d6a39-bd8f-4175-bc61-fe50ed0403a3/026d6a39-bd8f-4175-bc61-fe50ed0403a3.usdz
EOF

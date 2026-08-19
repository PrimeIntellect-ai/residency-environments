#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[prime-template] %s\n' "$*"
}

configure_ssh_access() {
  local key_blob="${PUBLIC_KEY:-${SSH_PUBLIC_KEY:-${AUTHORIZED_KEYS:-${SSH_AUTHORIZED_KEYS:-}}}}"
  local ssh_port="${SSH_PORT:-22}"

  mkdir -p /root/.ssh /home/ubuntu/.ssh
  chmod 700 /root/.ssh /home/ubuntu/.ssh
  chown ubuntu:ubuntu /home/ubuntu/.ssh

  if [[ -n "${key_blob}" ]]; then
    printf '%s\n' "${key_blob}" >/root/.ssh/authorized_keys
    printf '%s\n' "${key_blob}" >/home/ubuntu/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys /home/ubuntu/.ssh/authorized_keys
    chown ubuntu:ubuntu /home/ubuntu/.ssh/authorized_keys
    log "installed SSH public key for root and ubuntu"
  else
    log "no PUBLIC_KEY/SSH_PUBLIC_KEY/AUTHORIZED_KEYS/SSH_AUTHORIZED_KEYS provided"
  fi

  sed -i '/^#*Port /d' /etc/ssh/sshd_config
  printf 'Port %s\n' "${ssh_port}" >>/etc/ssh/sshd_config
}

write_login_env() {
  local path_value="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  cat >/etc/profile.d/prime-template-env.sh <<EOF
export CARLA_ROOT="${CARLA_ROOT}"
export START_DOCKER="${START_DOCKER}"
export START_SSHD="${START_SSHD}"
export DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}"
export SSH_PORT="${SSH_PORT}"
export PATH="${path_value}"
EOF
  chmod 0644 /etc/profile.d/prime-template-env.sh

  cat >/etc/environment <<EOF
CARLA_ROOT=${CARLA_ROOT}
START_DOCKER=${START_DOCKER}
START_SSHD=${START_SSHD}
DOCKER_DATA_ROOT=${DOCKER_DATA_ROOT}
NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES}
XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}
SSH_PORT=${SSH_PORT}
PATH=${path_value}
EOF
  chmod 0644 /etc/environment
}

ssh_port_in_use() {
  local ssh_port="${SSH_PORT:-22}"
  ss -ltnH "( sport = :${ssh_port} )" 2>/dev/null | grep -q ":${ssh_port}"
}

start_docker_if_possible() {
  if [[ "${START_DOCKER:-1}" != "1" ]]; then
    return 0
  fi
  if ! command -v dockerd >/dev/null 2>&1; then
    return 0
  fi
  if pgrep -f "(^|/)dockerd( |$)" >/dev/null 2>&1; then
    return 0
  fi

  mkdir -p /var/log
  local -a common_args=(
    --iptables=false
    --bridge=none
    --ip-forward=false
    --ip-masq=false
  )
  local storage_driver="${DOCKER_STORAGE_DRIVER:-}"
  local data_root="${DOCKER_DATA_ROOT:-}"

  if [[ -z "${storage_driver}" ]] && command -v fuse-overlayfs >/dev/null 2>&1; then
    storage_driver="fuse-overlayfs"
  fi
  if [[ -n "${data_root}" ]]; then
    mkdir -p "${data_root}"
    common_args+=(--data-root="${data_root}")
  fi

  if [[ -n "${storage_driver}" ]]; then
    if nohup dockerd "${common_args[@]}" --storage-driver="${storage_driver}" >/var/log/dockerd.log 2>&1 & then
      log "started dockerd with storage driver ${storage_driver} data_root=${data_root:-/var/lib/docker}"
      return 0
    fi
    sleep 2
    if pgrep -f "(^|/)dockerd( |$)" >/dev/null 2>&1; then
      log "started dockerd with storage driver ${storage_driver} data_root=${data_root:-/var/lib/docker}"
      return 0
    fi
    log "dockerd startup with storage driver ${storage_driver} failed; retrying with daemon defaults"
  fi

  if nohup dockerd "${common_args[@]}" >/var/log/dockerd.log 2>&1 & then
    log "started dockerd data_root=${data_root:-/var/lib/docker}"
  else
    log "dockerd startup failed; continuing without it"
  fi
}

main() {
  export CARLA_ROOT="${CARLA_ROOT:-/opt/carla-0.9.16}"
  export START_DOCKER="${START_DOCKER:-1}"
  export START_SSHD="${START_SSHD:-1}"
  if [[ -z "${DOCKER_DATA_ROOT:-}" && -d /ephemeral ]]; then
    export DOCKER_DATA_ROOT="/ephemeral/docker"
  fi
  export DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/var/lib/docker}"
  export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-carla}"
  export SSH_PORT="${SSH_PORT:-22}"

  mkdir -p \
    /var/run/sshd \
    /tmp/runtime-carla \
    "${DOCKER_DATA_ROOT}" \
    /home/ubuntu/.config/fish/conf.d \
    /home/ubuntu/.cache/carla-env
  chown -R ubuntu:ubuntu /home/ubuntu/.config /home/ubuntu/.cache /tmp/runtime-carla
  chmod 700 /tmp/runtime-carla
  write_login_env
  ssh-keygen -A >/dev/null 2>&1 || true
  configure_ssh_access
  start_docker_if_possible
  if [[ -d "${CARLA_ROOT}" ]]; then
    log "CARLA root: ${CARLA_ROOT}"
  fi
  if [[ "${START_SSHD}" != "1" ]]; then
    log "START_SSHD=${START_SSHD}; not starting sshd"
    exec tail -f /dev/null
  fi
  if ssh_port_in_use; then
    log "port ${SSH_PORT} already in use; not starting sshd"
    exec tail -f /dev/null
  fi
  exec /usr/sbin/sshd -D -e
}

main "$@"

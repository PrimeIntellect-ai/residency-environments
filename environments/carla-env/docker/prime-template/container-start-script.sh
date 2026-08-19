#!/usr/bin/env bash
set -euo pipefail

export CARLA_ROOT="${CARLA_ROOT:-/opt/carla-0.9.16}"
export START_DOCKER="${START_DOCKER:-1}"
if [[ -z "${DOCKER_DATA_ROOT:-}" && -d /ephemeral ]]; then
  export DOCKER_DATA_ROOT="/ephemeral/docker"
fi
export DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/var/lib/docker}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-carla}"

mkdir -p \
  /var/run/sshd \
  /tmp/runtime-carla \
  "${DOCKER_DATA_ROOT}" \
  /home/ubuntu/.config/fish/conf.d \
  /home/ubuntu/.cache/carla-env
chown -R ubuntu:ubuntu /home/ubuntu/.config /home/ubuntu/.cache
chmod 700 /tmp/runtime-carla

exec /usr/local/bin/prime-template-entrypoint.sh

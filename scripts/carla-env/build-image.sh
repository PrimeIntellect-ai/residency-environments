#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${1:-sinatras/carla-env-runtime:0.10.0-v1}"

exec docker build \
  --platform linux/amd64 \
  --file "${repo_root}/scripts/carla-env/Dockerfile" \
  --tag "${image}" \
  "${repo_root}/environments/carla-env"

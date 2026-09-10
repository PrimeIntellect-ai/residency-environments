#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker build \
  --file "$repo_root/scripts/pmpp-hard/Dockerfile.cuda-agent" \
  --tag pmpp-cuda-agent:cu128 \
  "$repo_root"
docker build \
  --file "$repo_root/scripts/pmpp-hard/Dockerfile.triton" \
  --tag pmpp-triton:cu128 \
  "$repo_root"
docker build \
  --file "$repo_root/scripts/pmpp-hard/Dockerfile.cutlass" \
  --tag pmpp-cutlass:12.8.1 \
  "$repo_root"

#!/usr/bin/env bash
set -euo pipefail

# Build the three runtime images on Prime from their Dockerfiles, one at a time.
# Builds of these images started together failed on Prime, so each build must
# finish before the next starts.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dockerfiles="$repo_root/scripts/pmpp-hard"

# Upload only what the Dockerfiles copy, never the whole repository.
context="$(mktemp -d)"
trap 'rm -rf "$context"' EXIT
mkdir -p "$context/scripts/pmpp-hard"
cp "$dockerfiles/harness-deps.py" "$context/scripts/pmpp-hard/"

build_status() {
  uv run --no-project --with "prime-sandboxes>=0.5.13" python -c '
import sys
from prime_sandboxes.images import ImageClient
print(ImageClient().get_build_status(sys.argv[1])["status"])
' "$1"
}

build() {
  local ref="$1" dockerfile="$2" build_id status
  build_id="$(prime images push "$ref" --context "$context" --dockerfile "$dockerfiles/$dockerfile" | sed -n 's/^Build ID: *//p')"
  if [ -z "$build_id" ]; then
    echo "prime images push did not start a build for $ref" >&2
    return 1
  fi
  echo "building $ref (build $build_id)"
  while true; do
    status="$(build_status "$build_id")"
    case "$status" in
      COMPLETED) echo "$ref is ready"; return 0 ;;
      FAILED | CANCELLED | CANCELED | ERROR)
        echo "$ref build $build_id ended as $status" >&2
        return 1
        ;;
    esac
    sleep 30
  done
}

build pmpp-cuda-agent:cu128 Dockerfile.cuda-agent
build pmpp-triton:cu128 Dockerfile.triton
build pmpp-cutlass:12.8.1 Dockerfile.cutlass

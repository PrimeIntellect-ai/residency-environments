#!/usr/bin/env bash
# Build the agent and scorer images as private images in the Prime registry, which Prime VMs require.
set -euo pipefail

tag=${1:?usage: push_prime_images.sh TAG}
here=$(cd "$(dirname "$0")" && pwd)
prime images push "pretrain-gym-agent:$tag" --context "$here" --dockerfile "$here/Dockerfile.agent" --private
prime images push "pretrain-gym-scorer:$tag" --context "$here" --dockerfile "$here/Dockerfile.scorer" --private

#!/bin/bash
# Build and push the sandbox image from this repository.
# Needs a logged in prime CLI. The scene dataset is public, so HF_TOKEN is optional and
# is only sent if it happens to be set.
# Usage: scripts/aerialsim_flight/build_image.sh <tag>
set -euo pipefail

TAG=${1:?usage: build_image.sh <tag, like v4>}
# Pinned to a revision rather than main, because main is a branch that can move and a
# different world under the same ref would silently change what a graded flight means.
# The sha256 is checked below: a size check passes any archive of the same length.
SCENE_REV=f474f85f3d1b367965ebc77b9b3fa2f88b6f92b0
SCENE_URL=https://huggingface.co/datasets/nahidalam/aerialsim-scenes/resolve/$SCENE_REV/CityParkLite_Linux.tar.gz
SCENE_SHA256=e84669d338303c180dd3777adb108406248a9f0a774eb102ff3b066a3c1a1772
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=$(mktemp -d)

cp "$ROOT/scripts/aerialsim_flight/Dockerfile" \
   "$ROOT/scripts/aerialsim_flight/start_sim.sh" \
   "$ROOT/scripts/aerialsim_flight/settings.json" "$BUILD"/
mkdir -p "$BUILD/aerialsim_flight_pkg"
cp -r "$ROOT/environments/aerialsim_flight/aerialsim_flight" "$BUILD/aerialsim_flight_pkg/"

# Written as two calls rather than building an argument array: macOS ships bash 3.2, where
# expanding an empty array under `set -u` dies with "unbound variable", which is exactly the
# no-token case this is meant to allow.
if [ -n "${HF_TOKEN:-}" ]; then
    curl -fL -H "Authorization: Bearer $HF_TOKEN" "$SCENE_URL" -o "$BUILD/CityParkLite_Linux.tar.gz"
else
    curl -fL "$SCENE_URL" -o "$BUILD/CityParkLite_Linux.tar.gz"
fi
echo "$SCENE_SHA256  $BUILD/CityParkLite_Linux.tar.gz" | shasum -a 256 -c - \
    || { echo "scene archive does not match the pinned sha256"; exit 1; }

prime images push "aerialsim-city-park:$TAG" --context "$BUILD"

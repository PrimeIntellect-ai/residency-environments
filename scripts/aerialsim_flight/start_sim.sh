#!/bin/bash
# Boot the City Park sim in the background, then keep the container alive.
set -u

AIRSIM_ROOT=/opt/cosys-airsim-city-park
AIRSIM_GAME_MODE="/Script/AirSim.AirSimGameMode"

# The host injects NVIDIA userspace libraries. If Vulkan cannot see the
# GPU, install the userspace matching the host kernel driver.
if ! vulkaninfo --summary 2>/dev/null | grep -q NVIDIA; then
    KV=$(grep -oE '([0-9]+\.[0-9]+\.[0-9]+)' /proc/driver/nvidia/version 2>/dev/null | head -1 || true)
    if [ -n "${KV:-}" ]; then
        echo "installing NVIDIA userspace $KV for Vulkan" >&2
        wget -q -O /tmp/nvidia.run "https://us.download.nvidia.com/tesla/${KV}/NVIDIA-Linux-x86_64-${KV}.run" \
            && sh /tmp/nvidia.run --silent --no-kernel-modules --no-drm --no-x-check \
                --no-nvidia-modprobe --install-libglvnd \
            && ldconfig
        rm -f /tmp/nvidia.run
    fi
fi

runtime_env="$(find "$AIRSIM_ROOT" -maxdepth 5 -name aerialsim-runtime.env -type f | head -1 || true)"
if [ -n "$runtime_env" ]; then
    # shellcheck disable=SC1090
    source "$runtime_env"
fi

if [ -n "${AIRSIM_LAUNCHER_BASENAME:-}" ]; then
    run_script="$(find "$AIRSIM_ROOT" -maxdepth 5 -name "$AIRSIM_LAUNCHER_BASENAME" -type f | head -1)"
else
    run_script="$(find "$AIRSIM_ROOT" -maxdepth 5 -name '*.sh' -type f -not -name 'build_*' | head -1)"
fi
if [ -z "$run_script" ] || [ ! -x "$run_script" ]; then
    echo "ERROR: no game launcher found under $AIRSIM_ROOT" >&2
    exit 1
fi

runtime_args=()
if [ -n "${AIRSIM_MAP_PATH:-}" ]; then
    runtime_args+=("${AIRSIM_MAP_PATH}?game=${AIRSIM_GAME_MODE}")
fi
runtime_args+=(
    -RenderOffScreen
    -nosound
    -unattended
    -windowed
    -ResX=640
    -ResY=360
    "-ExecCmds=r.DetailMode 2"
)

mkdir -p /tmp/runtime-airsim_user
chown airsim_user:airsim_user /tmp/runtime-airsim_user

cd "$(dirname "$run_script")"
sudo -u airsim_user -H XDG_RUNTIME_DIR=/tmp/runtime-airsim_user \
    nohup "$run_script" "${runtime_args[@]}" > /tmp/airsim.log 2>&1 &

for _ in $(seq 1 120); do
    if bash -c "(echo > /dev/tcp/localhost/41451) 2>/dev/null"; then
        echo "sim ready on port 41451"
        break
    fi
    sleep 5
done

sleep infinity

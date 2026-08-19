#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

RUN_USER="tester"
CARLA_VERSION="0.10.0"
MODE="vision"
PORT="4000"
TM_PORT="8000"
START_TIMEOUT_S="120"
SETTLE_S="10"
PREINSTALLED_ROOT=""

usage() {
  cat <<'EOF'
Usage:
  scripts/start_local_carla_on_pod.sh <prepare|start|stop|status> [options]

Options:
  --user <name>          Unix user that owns the unpacked CARLA files.
                         Default: tester
  --version <value>      CARLA version: 0.10, 0.10.0, 0.9, or 0.9.16.
                         Default: 0.10.0
  --mode <vision|text>   vision => RenderOffScreen, text => nullrhi where supported.
                         Default: vision
  --port <n>             CARLA RPC port.
                         Default: 4000
  --tm-port <n>          TrafficManager port to reserve for traffic-enabled runs.
                         Default: 8000
  --timeout <seconds>    Wait timeout for port readiness after start.
                         Default: 120
  --settle <seconds>     Extra settle time after ports open.
                         Default: 10
  --preinstalled-root
      <path>             Use an already unpacked CARLA root instead of docker extraction.
  -h, --help             Show this help.

Examples:
  sudo scripts/start_local_carla_on_pod.sh prepare
  sudo scripts/start_local_carla_on_pod.sh start --mode vision --port 4000
  sudo scripts/start_local_carla_on_pod.sh status --port 4000
  sudo scripts/start_local_carla_on_pod.sh stop --port 4000

Notes:
  - This script is for GPU pods where nested Docker cannot run CARLA directly.
  - It pulls the official CARLA image with Docker, exports the rootfs to the
    host once, then launches CARLA directly from that extracted filesystem.
  - On some RunPod/Prime nodes the default NVIDIA Vulkan ICD is broken for
    CARLA; this script writes an EGL-based ICD override automatically.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

info() {
  echo "[$SCRIPT_NAME] $*"
}

require_root() {
  [[ "${EUID}" -eq 0 ]] || die "run this script as root (sudo ...)"
}

canon_version() {
  case "${1}" in
    0.10|0.10.0|"") echo "0.10.0" ;;
    0.9|0.9.16) echo "0.9.16" ;;
    *) die "unsupported CARLA version: ${1} (expected 0.10.0 or 0.9.16)" ;;
  esac
}

version_image() {
  case "${1}" in
    0.10.0) echo "carlasim/carla:0.10.0" ;;
    0.9.16) echo "carlasim/carla:0.9.16" ;;
    *) die "no image preset for version ${1}" ;;
  esac
}

version_launcher() {
  case "${1}" in
    0.10.0) echo "CarlaUnreal.sh" ;;
    0.9.16) echo "CarlaUE4.sh" ;;
    *) die "no launcher preset for version ${1}" ;;
  esac
}

container_name_for() {
  local version="$1"
  case "${version}" in
    0.10.0) echo "carla-010" ;;
    0.9.16) echo "carla-0916" ;;
    *) printf 'carla-%s' "${version//./-}" ;;
  esac
}

run_home() {
  getent passwd "${RUN_USER}" | cut -d: -f6
}

cache_base_dir() {
  if [[ -d /ephemeral ]]; then
    printf '/ephemeral/carla-env/%s' "${RUN_USER}"
  elif [[ -d /workspace && -w /workspace ]]; then
    printf '/workspace/carla-env/%s' "${RUN_USER}"
  else
    printf '%s/.cache/carla-env' "$(run_home)"
  fi
}

run_as_user() {
  local cmd="$1"
  su -l -s /bin/bash "${RUN_USER}" -c "${cmd}"
}

ensure_user() {
  if ! id -u "${RUN_USER}" >/dev/null 2>&1; then
    info "creating user ${RUN_USER}"
    useradd -m -s /bin/bash "${RUN_USER}"
  fi
}

ensure_system_packages() {
  info "installing host dependencies"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null
  local packages=(
    python3-pip
    xdg-user-dirs
    libvulkan1
    vulkan-tools
    mesa-vulkan-drivers
  )
  if ! command -v docker >/dev/null 2>&1; then
    packages+=(docker.io)
  fi
  apt-get install -y "${packages[@]}" >/dev/null
  if command -v docker >/dev/null 2>&1; then
    systemctl enable --now docker >/dev/null 2>&1 || true
  fi
}

ensure_docker_runtime() {
  command -v docker >/dev/null 2>&1 || die "docker is required for CARLA image extraction"
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  systemctl enable --now docker >/dev/null 2>&1 || true
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  if ! pgrep -f "(^|/)dockerd( |$)" >/dev/null 2>&1; then
    local dockerd_log
    dockerd_log="$(cache_base_dir)/dockerd-bootstrap.log"
    mkdir -p "$(dirname "${dockerd_log}")"
    nohup dockerd --iptables=false --bridge=none --ip-forward=false --ip-masq=false >"${dockerd_log}" 2>&1 &
  fi

  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    if docker info >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done

  die "docker daemon is not reachable"
}

image_path() {
  local image="$1"
  local repo="${image%%:*}"
  local tag="${image##*:}"
  printf '%s/rootfs-images/%s/%s' "$(cache_base_dir)" "${repo//\//_}" "${tag}"
}

rootfs_dir() {
  local container_name="$1"
  printf '%s/rootfs/%s' "$(cache_base_dir)" "${container_name}"
}

container_root() {
  local container_name="$1"
  local root_dir
  root_dir="$(rootfs_dir "${container_name}")"
  if [[ -x "${root_dir}/CarlaUE4.sh" || -x "${root_dir}/CarlaUnreal.sh" ]]; then
    printf '%s' "${root_dir}"
  elif [[ -d "${root_dir}/workspace" ]]; then
    printf '%s/workspace' "${root_dir}"
  elif [[ -d "${root_dir}/home/carla" ]]; then
    printf '%s/home/carla' "${root_dir}"
  else
    printf '%s/workspace' "${root_dir}"
  fi
}

detect_preinstalled_root() {
  local version="$1"
  local launcher="$2"
  local candidate

  if [[ -n "${PREINSTALLED_ROOT}" ]]; then
    if [[ -x "${PREINSTALLED_ROOT}/${launcher}" ]]; then
      printf '%s\n' "${PREINSTALLED_ROOT}"
      return 0
    fi
    die "preinstalled CARLA root does not contain ${launcher}: ${PREINSTALLED_ROOT}"
  fi

  for candidate in \
    "/opt/carla-${version}" \
    "/opt/carla/${version}" \
    "/usr/local/carla-${version}" \
    "/usr/local/carla/${version}" \
    "/workspace/carla-${version}" \
    "/workspace/carla/${version}" \
    "/opt/carla" \
    "/workspace/carla"; do
    if [[ -x "${candidate}/${launcher}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

ensure_preinstalled_root() {
  local container_name="$1"
  local source_root="$2"
  local root_dir parent_dir temp_root backup_dir real_source
  root_dir="$(rootfs_dir "${container_name}")"
  parent_dir="$(dirname "${root_dir}")"
  mkdir -p "${parent_dir}"

  real_source="$(readlink -f "${source_root}")"
  if [[ "${RUN_USER}" == "carla" && -x "${real_source}/CarlaUE4.sh" ]]; then
    if [[ -L "${root_dir}" && "$(readlink -f "${root_dir}")" == "${real_source}" ]]; then
      return 0
    fi
    info "linking preinstalled CARLA root ${real_source} into runtime cache ${root_dir}"
    rm -rf "${root_dir}" >/dev/null 2>&1 || true
    ln -s "${real_source}" "${root_dir}"
    return 0
  fi

  if [[ -d "${root_dir}" && ! -L "${root_dir}" ]]; then
    return 0
  fi

  info "staging preinstalled CARLA root ${source_root} into writable cache ${root_dir}"
  temp_root="$(mktemp -d "${parent_dir}/.$(basename "${root_dir}").tmp.XXXXXX")"
  backup_dir=""
  if ! cp -a --reflink=auto "${source_root}/." "${temp_root}/"; then
    rm -rf "${temp_root}"
    temp_root="$(mktemp -d "${parent_dir}/.$(basename "${root_dir}").tmp.XXXXXX")"
    rsync -a "${source_root}/" "${temp_root}/"
  fi
  chown -R "${RUN_USER}:${RUN_USER}" "${temp_root}"

  if [[ -e "${root_dir}" || -L "${root_dir}" ]]; then
    backup_dir="${root_dir}.old.$$"
    rm -rf "${backup_dir}" >/dev/null 2>&1 || true
    mv "${root_dir}" "${backup_dir}"
  fi
  mv "${temp_root}" "${root_dir}"
  temp_root=""
  rm -rf "${backup_dir}" >/dev/null 2>&1 || true
}

carla_dist_path() {
  local version="$1"
  local dist_dir
  dist_dir="$(container_root "$(container_name_for "${version}")")/PythonAPI/carla/dist"
  [[ -d "${dist_dir}" ]] || return 1
  find "${dist_dir}" -maxdepth 1 -type f \( -name 'carla-*.whl' -o -name 'carla-*.egg' \) | sort
}

probe_client_dir() {
  local version="$1"
  local pybin="${2:-python3}"
  local py_tag
  py_tag="$("${pybin}" - <<'PY' 2>/dev/null || true
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)"
  py_tag="${py_tag:-unknown}"
  printf '%s/probe-client-%s-%s' "$(cache_base_dir)" "${version//./-}" "${py_tag}"
}

probe_python_candidates() {
  {
    printf '%s\n' "${PYTHON_BIN:-python3}"
    printf '%s\n' python3 python3.12 python3.11 python3.10 python3.9 python3.8 python3.7
  } | awk '!seen[$0]++' | while IFS= read -r pybin; do
    command -v "${pybin}" >/dev/null 2>&1 || continue
    printf '%s\n' "${pybin}"
  done
}

probe_client_importable() {
  local pybin="$1"
  local target="$2"
  "${pybin}" - "${target}" <<'PY' >/dev/null 2>&1
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
if not target.exists():
    sys.exit(1)
sys.path.insert(0, str(target))
import carla  # noqa: F401
PY
}

extract_probe_artifact() {
  local pybin="$1"
  local artifact="$2"
  local target="$3"
  rm -rf "${target}"
  mkdir -p "${target}"
  "${pybin}" - "${artifact}" "${target}" <<'PY' >/dev/null 2>&1
import pathlib
import sys
import zipfile

artifact = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(artifact) as zf:
    zf.extractall(target)
PY
}

find_global_probe_python() {
  local pybin
  while IFS= read -r pybin; do
    if "${pybin}" -c 'import carla' >/dev/null 2>&1; then
      printf '%s\n' "${pybin}"
      return 0
    fi
  done < <(probe_python_candidates)
  return 1
}

ensure_probe_client() {
  local version="$1"

  PROBE_PYTHON=""
  PROBE_CLIENT_DIR=""

  local pybin target artifact found_dist="false"
  while IFS= read -r pybin; do
    target="$(probe_client_dir "${version}" "${pybin}")"
    if probe_client_importable "${pybin}" "${target}"; then
      PROBE_PYTHON="${pybin}"
      PROBE_CLIENT_DIR="${target}"
      return 0
    fi
  done < <(probe_python_candidates)

  while IFS= read -r pybin; do
    while IFS= read -r artifact; do
      [[ -n "${artifact}" ]] || continue
      found_dist="true"
      target="$(probe_client_dir "${version}" "${pybin}")"
      info "testing bundled CARLA Python API for readiness probes (${version}, ${pybin}, $(basename "${artifact}"))"
      if ! extract_probe_artifact "${pybin}" "${artifact}" "${target}"; then
        rm -rf "${target}"
        continue
      fi
      if probe_client_importable "${pybin}" "${target}"; then
        chown -R "${RUN_USER}:${RUN_USER}" "${target}"
        PROBE_PYTHON="${pybin}"
        PROBE_CLIENT_DIR="${target}"
        return 0
      fi
      rm -rf "${target}"
    done < <(carla_dist_path "${version}" || true)
  done < <(probe_python_candidates)

  [[ "${found_dist}" == "true" ]] || return 1
  return 1
}

wait_for_probe_python() {
  local version="$1"

  if ensure_probe_client "${version}"; then
    printf '%s|%s\n' "${PROBE_PYTHON}" "${PROBE_CLIENT_DIR}"
    return 0
  fi

  local global_py
  global_py="$(find_global_probe_python || true)"
  if [[ -n "${global_py}" ]]; then
    printf '%s|\n' "${global_py}"
    return 0
  fi

  return 1
}

ensure_image() {
  local image="$1"
  ensure_docker_runtime
  if ! docker image inspect "${image}" >/dev/null 2>&1; then
    info "pulling ${image} via docker"
    docker pull "${image}" >/dev/null
  fi
}

ensure_container() {
  local image="$1"
  local container_name="$2"
  local launcher="$3"
  local root_dir parent_dir temp_root backup_dir container_id
  root_dir="$(rootfs_dir "${container_name}")"
  if [[ -x "$(container_root "${container_name}")/${launcher}" ]]; then
    return 0
  fi

  info "exporting ${image} rootfs to ${root_dir}"
  parent_dir="$(dirname "${root_dir}")"
  mkdir -p "${parent_dir}"
  temp_root="$(mktemp -d "${parent_dir}/.${container_name}.tmp.XXXXXX")"
  backup_dir=""
  container_id="$(docker create "${image}")"
  trap 'docker rm -f "${container_id}" >/dev/null 2>&1 || true; rm -rf "${temp_root}" >/dev/null 2>&1 || true' RETURN
  docker export "${container_id}" | tar -C "${temp_root}" -xf -
  docker rm -f "${container_id}" >/dev/null
  container_id=""
  if [[ -e "${root_dir}" ]]; then
    backup_dir="${root_dir}.old.$$"
    rm -rf "${backup_dir}" >/dev/null 2>&1 || true
    mv "${root_dir}" "${backup_dir}"
  fi
  mv "${temp_root}" "${root_dir}"
  temp_root=""
  rm -rf "${backup_dir}" >/dev/null 2>&1 || true
  backup_dir=""
  chown -R "${RUN_USER}:${RUN_USER}" "${root_dir}"
  trap - RETURN
}

ensure_vulkan_override() {
  local home_dir config_dir icd_file host_icd
  home_dir="$(run_home)"
  config_dir="${home_dir}/.config/carla-env"
  host_icd="/usr/share/vulkan/icd.d/nvidia_icd.json"
  icd_file="${config_dir}/nvidia_icd.json"

  mkdir -p "${config_dir}"
  if [[ -f "${host_icd}" ]]; then
    cp "${host_icd}" "${icd_file}"
  else
    cat >"${icd_file}" <<'EOF'
{
  "file_format_version": "1.0.1",
  "ICD": {
    "library_path": "libGLX_nvidia.so.0",
    "api_version": "1.3.289"
  }
}
EOF
  fi
  chown -R "${RUN_USER}:${RUN_USER}" "${config_dir}"
}

pid_file() {
  local version="$1"
  local port="$2"
  printf '%s/carla-%s-%s.pid' "$(cache_base_dir)" "${version//./-}" "${port}"
}

log_file() {
  local version="$1"
  local mode="$2"
  local port="$3"
  printf '%s/carla-%s-%s-%s.log' "$(cache_base_dir)" "${version//./-}" "${mode}" "${port}"
}

runtime_dir() {
  local port="$1"
  if [[ "${RUN_USER}" == "carla" ]]; then
    printf '/tmp/runtime-carla'
    return 0
  fi
  printf '%s/xdg-runtime-%s' "$(cache_base_dir)" "${port}"
}

ensure_cache_dirs() {
  local base_dir
  base_dir="$(cache_base_dir)"
  mkdir -p "${base_dir}"
  chown -R "${RUN_USER}:${RUN_USER}" "$(dirname "${base_dir}")"
}

launch_args() {
  local version="$1"
  local mode="$2"
  local port="$3"
  case "${version}:${mode}" in
    0.10.0:vision) echo "-RenderOffScreen -nosound -carla-rpc-port=${port} -stdout -FullStdOutLogOutput -unattended" ;;
    0.10.0:text) echo "-nullrhi -nosound -carla-rpc-port=${port} -stdout -FullStdOutLogOutput -unattended" ;;
    0.9.16:vision|0.9.16:text) echo "-RenderOffScreen -nosound -carla-rpc-port=${port} -stdout -FullStdOutLogOutput -unattended" ;;
    *) die "unsupported version/mode combination: ${version} ${mode}" ;;
  esac
}

find_carla_pid() {
  local port="$1"
  pgrep -f "(CarlaUnreal|CarlaUE4)-Linux-Shipping .* -carla-rpc-port=${port}" || true
}

required_ports() {
  local port="$1"
  printf '%s\n' "${port}" "$((port + 1))" "$((port + 2))" "${TM_PORT}" | awk '!seen[$0]++'
}

port_is_listening() {
  local port="$1"
  ss -H -lnt | awk -v p="${port}" '
    {
      n = split($4, parts, ":")
      if (n > 0 && parts[n] == p) {
        found = 1
        exit 0
      }
    }
    END { exit found ? 0 : 1 }
  '
}

ensure_ports_available() {
  local port
  while IFS= read -r port; do
    if port_is_listening "${port}"; then
      die "required CARLA port ${port} is already in use"
    fi
  done < <(required_ports "$1")
}

wait_for_port() {
  local port="$1"
  local timeout_s="$2"
  python3 - "${port}" "${timeout_s}" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
timeout_s = int(sys.argv[2])
deadline = time.time() + timeout_s
while time.time() < deadline:
    with socket.socket() as s:
        s.settimeout(1.0)
        try:
            s.connect(("127.0.0.1", port))
        except OSError:
            time.sleep(1.0)
            continue
        sys.exit(0)
sys.exit(1)
PY
}

wait_for_carla_ready() {
  local version="$1"
  local port="$2"
  local timeout_s="$3"
  local probe_dir=""
  local probe_python=""
  local probe_spec=""

  probe_spec="$(wait_for_probe_python "${version}" || true)"
  if [[ -z "${probe_spec}" ]]; then
    echo "No compatible CARLA Python API is available for strict readiness checks." >&2
    echo "The server will be left running after the port-open check, but RPC readiness could not be verified." >&2
    return 2
  fi
  probe_python="${probe_spec%%|*}"
  probe_dir="${probe_spec#*|}"

  "${probe_python}" - "${port}" "${timeout_s}" "${probe_dir}" <<'PY'
import sys
import time

port = int(sys.argv[1])
timeout_s = int(sys.argv[2])
probe_dir = sys.argv[3]
deadline = time.time() + timeout_s

if probe_dir:
    sys.path.insert(0, probe_dir)

import carla

while time.time() < deadline:
    try:
        client = carla.Client("127.0.0.1", port)
        client.set_timeout(2.0)
        client.get_server_version()
    except Exception:
        time.sleep(1.0)
        continue
    sys.exit(0)

sys.exit(1)
PY
}

prepare() {
  require_root
  CARLA_VERSION="$(canon_version "${CARLA_VERSION}")"

  ensure_user
  local image container_name root_dir launcher
  image="$(version_image "${CARLA_VERSION}")"
  container_name="$(container_name_for "${CARLA_VERSION}")"
  launcher="$(version_launcher "${CARLA_VERSION}")"
  local preinstalled_root=""

  preinstalled_root="$(detect_preinstalled_root "${CARLA_VERSION}" "${launcher}" || true)"

  if [[ -n "${preinstalled_root}" ]]; then
    ensure_system_packages
    ensure_vulkan_override
    ensure_cache_dirs
    ensure_preinstalled_root "${container_name}" "${preinstalled_root}"
  else
    ensure_system_packages
    ensure_docker_runtime
    ensure_vulkan_override
    ensure_cache_dirs
  fi

  root_dir="$(container_root "${container_name}")"

  if [[ -n "${preinstalled_root}" ]]; then
    info "using preinstalled CARLA root ${preinstalled_root}"
  elif [[ ! -x "${root_dir}/${launcher}" ]]; then
    ensure_image "${image}"
    ensure_container "${image}" "${container_name}" "${launcher}"
  else
    info "reusing existing unpacked CARLA root ${root_dir}"
  fi

  info "prepare complete"
  info "user=${RUN_USER}"
  info "version=${CARLA_VERSION}"
  info "container=${container_name}"
  info "root=$(container_root "${container_name}")"
}

start_server() {
  require_root
  CARLA_VERSION="$(canon_version "${CARLA_VERSION}")"
  [[ "${MODE}" == "vision" || "${MODE}" == "text" ]] || die "mode must be 'vision' or 'text'"

  prepare

  local image container_name root_dir launcher args pidfile logfile xdg_dir icd_file launcher_script
  image="$(version_image "${CARLA_VERSION}")"
  container_name="$(container_name_for "${CARLA_VERSION}")"
  root_dir="$(container_root "${container_name}")"
  launcher="$(version_launcher "${CARLA_VERSION}")"
  args="$(launch_args "${CARLA_VERSION}" "${MODE}" "${PORT}")"
  pidfile="$(pid_file "${CARLA_VERSION}" "${PORT}")"
  logfile="$(log_file "${CARLA_VERSION}" "${MODE}" "${PORT}")"
  xdg_dir="$(runtime_dir "${PORT}")"
  icd_file="$(run_home)/.config/carla-env/nvidia_icd.json"
  launcher_script="$(cache_base_dir)/launch-${CARLA_VERSION//./-}-${MODE}-${PORT}.sh"

  ensure_ports_available "${PORT}"
  if [[ -n "$(find_carla_pid "${PORT}")" ]]; then
    die "a CARLA process for port ${PORT} is already running"
  fi

  mkdir -p "$(dirname "${launcher_script}")" "${xdg_dir}"
  chown -R "${RUN_USER}:${RUN_USER}" "$(dirname "${launcher_script}")" "${xdg_dir}"
  chmod 700 "${xdg_dir}"

  cat >"${launcher_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${root_dir}"
export HOME="$(run_home)"
export SDL_VIDEODRIVER=offscreen
export XDG_RUNTIME_DIR="${xdg_dir}"
EOF

  if [[ "${MODE}" == "vision" || "${CARLA_VERSION}" == "0.9.16" ]]; then
    cat >>"${launcher_script}" <<EOF
export VK_ICD_FILENAMES="${icd_file}"
EOF
  fi

  cat >>"${launcher_script}" <<EOF
exec "./${launcher}" ${args}
EOF

  chown "${RUN_USER}:${RUN_USER}" "${launcher_script}"
  chmod 755 "${launcher_script}"

  info "starting CARLA ${CARLA_VERSION} (${MODE}) on port ${PORT}"
  run_as_user "nohup '${launcher_script}' >> '${logfile}' 2>&1 & echo \$! > '${pidfile}'"

  _cleanup_failed_launch() {
    local pid
    pid="$(cat "${pidfile}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      sleep 1
      kill -9 "${pid}" 2>/dev/null || true
    fi
    # Also catch by port in case pidfile is stale.
    local found
    found="$(find_carla_pid "${PORT}")"
    if [[ -n "${found}" ]]; then
      kill ${found} 2>/dev/null || true
      sleep 1
      kill -9 ${found} 2>/dev/null || true
    fi
    rm -f "${pidfile}"
  }

  if ! wait_for_port "${PORT}" "${START_TIMEOUT_S}"; then
    echo >&2
    echo "CARLA did not open port ${PORT} within ${START_TIMEOUT_S}s." >&2
    echo "Last 80 log lines from ${logfile}:" >&2
    tail -n 80 "${logfile}" >&2 || true
    _cleanup_failed_launch
    exit 1
  fi

  local ready_status=0
  if wait_for_carla_ready "${CARLA_VERSION}" "${PORT}" "${START_TIMEOUT_S}"; then
    ready_status=0
  else
    ready_status=$?
  fi

  if [[ "${ready_status}" -eq 1 ]]; then
    echo >&2
    echo "CARLA opened port ${PORT} but did not become RPC-ready within ${START_TIMEOUT_S}s." >&2
    echo "Last 80 log lines from ${logfile}:" >&2
    tail -n 80 "${logfile}" >&2 || true
    _cleanup_failed_launch
    exit 1
  elif [[ "${ready_status}" -eq 2 ]]; then
    info "no compatible CARLA Python probe is available on this host; proceeding after port-open check only"
  fi

  if [[ "${SETTLE_S}" != "0" ]]; then
    info "waiting ${SETTLE_S}s extra settle time"
    sleep "${SETTLE_S}"
  fi

  info "CARLA is listening on 127.0.0.1:${PORT}"
  info "TrafficManager port: ${TM_PORT} (use --traffic-manager-port ${TM_PORT} in carla_env)"
  info "log=${logfile}"
  info "pid=$(find_carla_pid "${PORT}" | tr '\n' ' ')"
}

stop_server() {
  require_root
  CARLA_VERSION="$(canon_version "${CARLA_VERSION}")"

  local pidfile pids
  pidfile="$(pid_file "${CARLA_VERSION}" "${PORT}")"
  pids="$(find_carla_pid "${PORT}")"

  if [[ -f "${pidfile}" ]]; then
    local pid
    pid="$(cat "${pidfile}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      info "stopping launcher pid ${pid}"
      kill "${pid}" || true
    fi
    rm -f "${pidfile}"
  fi

  if [[ -n "${pids}" ]]; then
    info "stopping CARLA pid(s): ${pids//$'\n'/ }"
    kill ${pids} || true
  fi

  pkill -f "(CarlaUnreal|CarlaUE4)-Linux-Shipping .* -carla-rpc-port=${PORT}" || true

  sleep 1
  local active_port
  while IFS= read -r active_port; do
    if port_is_listening "${active_port}"; then
      die "port ${active_port} is still listening after stop"
    fi
  done < <(required_ports "${PORT}")

  if [[ -n "$(find_carla_pid "${PORT}")" ]]; then
    die "CARLA process for port ${PORT} is still present after stop"
  fi

  info "port ${PORT} is stopped"
}

status_server() {
  require_root
  CARLA_VERSION="$(canon_version "${CARLA_VERSION}")"

  local pidfile logfile pids
  pidfile="$(pid_file "${CARLA_VERSION}" "${PORT}")"
  logfile="$(log_file "${CARLA_VERSION}" "${MODE}" "${PORT}")"
  pids="$(find_carla_pid "${PORT}")"

  echo "user=${RUN_USER}"
  echo "version=${CARLA_VERSION}"
  echo "mode=${MODE}"
  echo "port=${PORT}"
  echo "tm_port=${TM_PORT}"
  echo "container=$(container_name_for "${CARLA_VERSION}")"
  echo "root=$(container_root "$(container_name_for "${CARLA_VERSION}")")"
  echo "pid_file=${pidfile}"
  echo "log_file=${logfile}"

  if [[ -f "${pidfile}" ]]; then
    echo "launcher_pid=$(cat "${pidfile}" 2>/dev/null || true)"
  else
    echo "launcher_pid="
  fi

  if [[ -n "${pids}" ]]; then
    echo "carla_pid=${pids//$'\n'/ }"
  else
    echo "carla_pid="
  fi

  if ss -lnt | grep -q "[[:space:]]:${PORT}[[:space:]]"; then
    echo "listening=yes"
  else
    echo "listening=no"
  fi
}

COMMAND="${1:-}"
if [[ -z "${COMMAND}" ]]; then
  usage
  exit 1
fi
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      RUN_USER="${2:?missing value for --user}"
      shift 2
      ;;
    --version)
      CARLA_VERSION="${2:?missing value for --version}"
      shift 2
      ;;
    --mode)
      MODE="${2:?missing value for --mode}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --tm-port)
      TM_PORT="${2:?missing value for --tm-port}"
      _TM_PORT_EXPLICIT="true"
      shift 2
      ;;
    --timeout)
      START_TIMEOUT_S="${2:?missing value for --timeout}"
      shift 2
      ;;
    --settle)
      SETTLE_S="${2:?missing value for --settle}"
      shift 2
      ;;
    --preinstalled-root)
      PREINSTALLED_ROOT="${2:?missing value for --preinstalled-root}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "${COMMAND}" in
  prepare) prepare ;;
  start) start_server ;;
  stop) stop_server ;;
  status) status_server ;;
  -h|--help|help) usage ;;
  *) die "unknown command: ${COMMAND}" ;;
esac

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields
from uuid import uuid4

from ..compat import VERSION_PRESETS, CarlaVersion, parse_version
from ..logging import get_logger

logger = get_logger("sandbox.pool")

CARLA_PORTS = (2000, 2001, 2002)
TRAFFIC_MANAGER_PORT = 8000


def _is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding on the given host."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _find_available_loopback_ips(
    count: int,
    ports: tuple[int, ...] = CARLA_PORTS,
    base_ip: str = "127.0.0.2",
) -> list[str]:
    """
    Find `count` loopback IPs where all required ports are available.

    Scanning starts at *base_ip* and proceeds upward (by last octet) within the
    same /24 until `.254`.
    """
    parts = base_ip.split(".")
    if len(parts) != 4 or parts[0] != "127":
        raise ValueError(f"base_ip must be in 127.0.0.0/8 (got {base_ip!r})")
    start_octet = max(2, int(parts[3]))

    available: list[str] = []
    for last_octet in range(start_octet, 255):
        if len(available) >= count:
            break
        ip = f"{parts[0]}.{parts[1]}.{parts[2]}.{last_octet}"
        if all(_is_port_available(ip, p) for p in ports):
            available.append(ip)
    if len(available) < count:
        subnet_hint = f"{parts[0]}.{parts[1]}.{parts[2]}.x"
        raise RuntimeError(
            f"Could not find {count} available loopback IPs (starting from {base_ip}). "
            f"Found only {len(available)}. Check for port conflicts on {subnet_hint}:{list(ports)}"
        )
    logger.debug("Found available loopback IPs: %s", available)
    return available


DEFAULT_CARLA_IMAGE = "carlasim/carla:0.10.0"
DEFAULT_CARLA_START_CMD_TEXT = "./CarlaUnreal.sh -nullrhi -nosound"
DEFAULT_CARLA_START_CMD_VISION = "./CarlaUnreal.sh -RenderOffScreen -nosound"
DEFAULT_CARLA_START_CMD = DEFAULT_CARLA_START_CMD_TEXT


def _version_from_official_image(image: str) -> CarlaVersion | None:
    for version, preset in VERSION_PRESETS.items():
        if image == preset["docker_image"]:
            return version
    return None


def _version_from_image_tag(image: str) -> CarlaVersion | None:
    normalized = str(image).strip()
    if not normalized or "@" in normalized:
        return None
    if ":" not in normalized:
        return None
    tag = normalized.rsplit(":", 1)[-1]
    try:
        return parse_version(tag)
    except ValueError:
        return None


def _version_from_start_command(command: str) -> CarlaVersion | None:
    normalized = str(command).strip().lower()
    if not normalized:
        return None
    if "carlaue4.sh" in normalized:
        return CarlaVersion.V0_9_16
    if "carlaunreal.sh" in normalized:
        return CarlaVersion.V0_10_0
    return None


def _infer_version_from_explicit_overrides(
    *,
    carla_image: str,
    carla_image_explicit: bool,
    carla_start_command: str,
    carla_start_command_explicit: bool,
) -> CarlaVersion | None:
    if carla_image_explicit:
        image_version = _version_from_official_image(carla_image)
        if image_version is None:
            image_version = _version_from_image_tag(carla_image)
        if image_version is not None:
            return image_version
    if carla_start_command_explicit:
        command_version = _version_from_start_command(carla_start_command)
        if command_version is not None:
            return command_version
    return None


def _validate_explicit_version_inputs(
    *,
    mode: str,
    carla_version: CarlaVersion,
    carla_image: str,
    carla_image_explicit: bool,
    carla_start_command: str,
    carla_start_command_explicit: bool,
) -> None:
    if mode == "disabled" or carla_version == CarlaVersion.AUTO:
        return

    explicit_versions: dict[str, CarlaVersion] = {}
    if carla_image_explicit:
        image_version = _version_from_official_image(carla_image)
        if image_version is None:
            image_version = _version_from_image_tag(carla_image)
        if image_version is not None:
            explicit_versions["carla_image"] = image_version
    if carla_start_command_explicit:
        command_version = _version_from_start_command(carla_start_command)
        if command_version is not None:
            explicit_versions["carla_start_command"] = command_version

    for field_name, explicit_version in explicit_versions.items():
        if explicit_version != carla_version:
            raise ValueError(
                f"Explicit sandbox {field_name} is for CARLA {explicit_version.value}, "
                f"but carla_version={carla_version.value!r}. Remove the conflicting override "
                f"or choose matching image/command/version settings."
            )

    if len(set(explicit_versions.values())) > 1:
        details = ", ".join(
            f"{name}={version.value}" for name, version in sorted(explicit_versions.items())
        )
        raise ValueError(
            "Sandbox config contains conflicting explicit CARLA version hints: "
            f"{details}. Use matching image and start command settings."
        )


@dataclass(frozen=True)
class SandboxReservation:
    sandbox_id: str
    host: str
    port: int = 2000
    lease_id: str = field(default_factory=lambda: uuid4().hex)


_VALID_SANDBOX_MODES = frozenset({"disabled", "prime"})


@dataclass
class CarlaSandboxConfig:
    mode: str = "prime"  # "prime" | "disabled"
    pool_size: int = 1
    base_local_ip: str = "127.0.0.2"

    carla_version: str = ""  # Empty string = derive from installed client; set in __post_init__.
    carla_image: str = ""
    carla_start_command: str | None = None

    cpu_cores: int = 4
    memory_gb: int = 8
    disk_size_gb: int = 40
    timeout_minutes: int = 120
    region: str | None = "us"
    internal_wait_mins: int = 6
    max_pool_start_attempts: int = 3
    max_concurrent_creates: int = 4

    proxy_ready_wait_s: float = 5.0
    carla_ready_timeout_s: float = 120.0
    carla_ready_retry_s: float = 5.0
    expose_retry_attempts: int = 2
    expose_retry_wait_s: float = 10.0
    expose_traffic_manager: bool = False

    pproxy_verbose: bool = False
    name_prefix: str = "carla-pool"
    _carla_image_explicit: bool | None = field(default=None, init=False, repr=False, compare=False)
    _carla_start_command_explicit: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        self.mode = str(self.mode).strip().lower()
        if self._carla_image_explicit is None:
            self._carla_image_explicit = bool(self.carla_image)
        self._carla_start_command_explicit = self.carla_start_command is not None
        self.carla_image = str(self.carla_image or DEFAULT_CARLA_IMAGE)
        self.carla_start_command = str(self.carla_start_command or DEFAULT_CARLA_START_CMD)
        if not self.carla_version or self.carla_version.strip() == "":
            if self.mode == "disabled":
                # Local mode can discover the running server version after connect,
                # so avoid brittle client-package auto-detection here.
                self.carla_version = CarlaVersion.AUTO.value
            else:
                inferred_version = _infer_version_from_explicit_overrides(
                    carla_image=self.carla_image,
                    carla_image_explicit=bool(self._carla_image_explicit),
                    carla_start_command=self.carla_start_command,
                    carla_start_command_explicit=bool(self._carla_start_command_explicit),
                )
                if inferred_version is None:
                    # No explicit overrides — infer from the (default) image
                    # rather than probing the local Python wheel, which would
                    # make the sandbox image depend on whichever client
                    # package happens to be installed locally.
                    inferred_version = _version_from_official_image(self.carla_image)
                if inferred_version is None:
                    raise ValueError(
                        "Cannot determine CARLA version for sandbox mode. "
                        "Specify carla_version explicitly (e.g. '0.10.0' or '0.9.16')."
                    )
                self.carla_version = inferred_version.value
        parsed_version = parse_version(self.carla_version)
        # Canonicalize version string so pool keys match for aliases like "0.10" vs "0.10.0".
        self.carla_version = parsed_version.value
        if parsed_version == CarlaVersion.AUTO and self.mode != "disabled":
            raise ValueError(
                "carla_version='auto' requires a running CARLA server "
                "(sandbox mode='disabled'). For sandboxed runs, specify "
                "an explicit version: '0.10.0' or '0.9.16'."
            )
        if self.mode not in _VALID_SANDBOX_MODES:
            raise ValueError(
                f"Unsupported sandbox mode {self.mode!r}; expected one of {sorted(_VALID_SANDBOX_MODES)}"
            )
        if int(self.pool_size) < 1:
            raise ValueError(
                "pool_size must be >= 1 (use mode='disabled' to disable sandbox pooling)"
            )
        _validate_explicit_version_inputs(
            mode=self.mode,
            carla_version=parsed_version,
            carla_image=self.carla_image,
            carla_image_explicit=bool(self._carla_image_explicit),
            carla_start_command=self.carla_start_command,
            carla_start_command_explicit=bool(self._carla_start_command_explicit),
        )
        if parsed_version != CarlaVersion.AUTO:
            expected_image = VERSION_PRESETS[parsed_version]["docker_image"]
            if not bool(self._carla_image_explicit):
                self.carla_image = expected_image
            if not self._carla_start_command_explicit:
                self.carla_start_command = VERSION_PRESETS[parsed_version]["start_command"]

    def clone(self) -> "CarlaSandboxConfig":
        cloned = CarlaSandboxConfig(
            **{item.name: getattr(self, item.name) for item in fields(self) if item.init}
        )
        cloned._carla_image_explicit = self._carla_image_explicit
        cloned._carla_start_command_explicit = self._carla_start_command_explicit
        return cloned

    def pool_key(self) -> tuple[tuple[str, object], ...]:
        return tuple((item.name, getattr(self, item.name)) for item in fields(self) if item.init)

    @classmethod
    def from_obj(
        cls,
        obj: object | None,
        *,
        carla_version: str | None = None,
    ) -> "CarlaSandboxConfig":
        if obj is None:
            kwargs: dict[str, object] = {}
            if carla_version is not None:
                kwargs["carla_version"] = str(carla_version)
            return cls(**kwargs)
        if isinstance(obj, CarlaSandboxConfig):
            cloned = obj.clone()
            if carla_version is not None:
                requested_version = parse_version(str(carla_version))
                requested_version_value = requested_version.value
                if cloned.carla_version != requested_version_value:
                    cloned.carla_version = requested_version_value
                    if not cloned._carla_start_command_explicit:
                        cloned.carla_start_command = None
                    cloned.__post_init__()
                else:
                    cloned.carla_version = requested_version_value
                    _validate_explicit_version_inputs(
                        mode=cloned.mode,
                        carla_version=requested_version,
                        carla_image=cloned.carla_image,
                        carla_image_explicit=bool(cloned._carla_image_explicit),
                        carla_start_command=cloned.carla_start_command,
                        carla_start_command_explicit=bool(cloned._carla_start_command_explicit),
                    )
            return cloned
        if isinstance(obj, dict):
            data = dict(obj)
            if carla_version is not None:
                data["carla_version"] = str(carla_version)
            return cls(**data)
        raise TypeError(f"Unsupported sandbox config type: {type(obj).__name__}")


class CarlaSandboxPool:
    """
    Create and manage a fixed-size pool of CARLA sandboxes (Prime) mapped to
    local loopback IPs via pproxy.

    Maintains a queue of ready sandboxes for concurrent episode execution.
    """

    def __init__(self, config: CarlaSandboxConfig):
        self.config = config
        self._started = False
        self._start_lock = asyncio.Lock()
        self._ready: asyncio.Queue[SandboxReservation] = asyncio.Queue(
            maxsize=max(1, int(config.pool_size))
        )
        self._acquire_waiters: deque[asyncio.Future[SandboxReservation]] = deque()
        self._sandboxes: list[str] = []
        self._leased: dict[str, str] = {}
        self._pproxy_proc: subprocess.Popen | None = None

        # Initialized on first start() call.
        self._client = None

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            if self.config.mode == "disabled":
                raise RuntimeError("Sandbox pool is disabled (mode='disabled')")
            if self._sandboxes:
                await self._shutdown_cancellation_safe()
                if self._sandboxes:
                    raise RuntimeError(
                        "Cannot start a new CARLA sandbox pool while previously created "
                        f"sandboxes still require deletion: {self._sandboxes}"
                    )
            self._drain_ready()
            self._leased.clear()
            creation_task = asyncio.create_task(asyncio.to_thread(self._create_pool_sync))
            try:
                reservations = await asyncio.shield(creation_task)
            except asyncio.CancelledError:
                try:
                    await creation_task
                except BaseException:
                    pass
                self._drain_ready()
                await self._shutdown_cancellation_safe()
                raise
            if not reservations:
                # Prevent acquire() from deadlocking on an empty ready queue.
                raise RuntimeError(
                    "Sandbox pool created no reservations. "
                    "This is unexpected; check sandbox mode/configuration."
                )
            try:
                for res in reservations:
                    self._ready.put_nowait(res)
                self._started = True
            except BaseException:
                self._drain_ready()
                await self._shutdown_cancellation_safe()
                raise

    async def acquire(self) -> SandboxReservation:
        await self.start()
        async with self._start_lock:
            if not self._started:
                raise RuntimeError("CARLA sandbox pool shut down during acquire")
            try:
                reservation = self._ready.get_nowait()
                self._leased[reservation.sandbox_id] = reservation.lease_id
                return reservation
            except asyncio.QueueEmpty:
                waiter = asyncio.get_running_loop().create_future()
                self._acquire_waiters.append(waiter)
        try:
            return await asyncio.shield(waiter)
        except asyncio.CancelledError:
            async with self._start_lock:
                try:
                    self._acquire_waiters.remove(waiter)
                except ValueError:
                    pass
                if waiter.done() and not waiter.cancelled():
                    try:
                        reservation = waiter.result()
                    except BaseException:
                        pass
                    else:
                        self._release_locked(reservation)
                elif not waiter.done():
                    waiter.cancel()
            raise
        finally:
            if not waiter.done():
                waiter.cancel()
            try:
                self._acquire_waiters.remove(waiter)
            except ValueError:
                pass

    async def release(self, reservation: SandboxReservation) -> None:
        async with self._start_lock:
            self._release_locked(reservation)

    def _release_locked(self, reservation: SandboxReservation) -> None:
        sandbox_id = reservation.sandbox_id
        if (
            not self._started
            or sandbox_id not in self._sandboxes
            or self._leased.get(sandbox_id) != reservation.lease_id
        ):
            return
        del self._leased[sandbox_id]
        returned = SandboxReservation(
            sandbox_id=sandbox_id,
            host=reservation.host,
            port=reservation.port,
        )
        while self._acquire_waiters:
            waiter = self._acquire_waiters.popleft()
            if waiter.done():
                continue
            self._leased[sandbox_id] = returned.lease_id
            waiter.set_result(returned)
            return
        try:
            self._ready.put_nowait(returned)
        except BaseException:
            self._leased[sandbox_id] = reservation.lease_id
            raise

    async def shutdown(self) -> None:
        async with self._start_lock:
            if not self._started and not self._sandboxes:
                return
            self._started = False
            self._leased.clear()
            self._fail_acquire_waiters()
            self._drain_ready()
            await self._shutdown_cancellation_safe()

    async def _shutdown_cancellation_safe(self) -> None:
        shutdown_task = asyncio.create_task(asyncio.to_thread(self._shutdown_sync))
        try:
            await asyncio.shield(shutdown_task)
        except asyncio.CancelledError:
            await shutdown_task
            raise

    def _fail_acquire_waiters(self) -> None:
        while self._acquire_waiters:
            waiter = self._acquire_waiters.popleft()
            if not waiter.done():
                waiter.set_exception(
                    RuntimeError("CARLA sandbox pool shut down while waiting for a reservation")
                )

    def _drain_ready(self) -> None:
        while True:
            try:
                self._ready.get_nowait()
            except asyncio.QueueEmpty:
                return

    # ------------------------- Sync helpers -------------------------

    def _ensure_prime_client(self):
        # Import lazily so users without prime_sandboxes can still use local mode.
        try:
            from prime_sandboxes import SandboxClient  # type: ignore
            from prime_sandboxes.core import APIClient  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "prime_sandboxes is required for sandbox_mode='prime'. "
                "Install it with: `uv pip install prime-sandboxes`"
            ) from e

        api_key = os.environ.get("PRIME_API_KEY")
        if not api_key:
            raise RuntimeError("PRIME_API_KEY is not set (required for Prime sandboxes)")
        if self._client is None:
            self._client = SandboxClient(APIClient(api_key=api_key))
        return self._client

    @staticmethod
    def _parse_exposed_endpoint(exposed_port):
        """
        Prime SDK response shapes vary:
        - external_endpoint: "host:port"
        - tls_socket + external_port
        - tls_socket already contains "host:port"
        """

        def _split_host_port(s: str) -> tuple[str, int | None]:
            s = str(s or "").strip()
            if not s:
                return "", None

            # Accept scheme prefixes (e.g. "tls://host:port") by parsing as a URL.
            if "://" in s:
                try:
                    from urllib.parse import urlsplit

                    u = urlsplit(s)
                    if u.hostname:
                        return u.hostname, u.port
                except Exception:
                    pass

            # Bracketed IPv6: "[::1]:123"
            if s.startswith("[") and "]" in s:
                host = s[1 : s.index("]")]
                rest = s[s.index("]") + 1 :]
                if rest.startswith(":") and rest[1:].isdigit():
                    return host, int(rest[1:])
                return host, None

            # Plain "host:port" (only treat as host:port if suffix is numeric).
            # Avoid mis-parsing unbracketed IPv6 literals like "2001:db8::1" as host:port.
            if ":" in s and s.count(":") == 1:
                host, maybe_port = s.rsplit(":", 1)
                if maybe_port.isdigit():
                    return host, int(maybe_port)

            return s, None

        ext = getattr(exposed_port, "external_endpoint", None)
        if ext:
            host, port = _split_host_port(str(ext))
            if host and port is not None:
                return host, int(port)

        tls = getattr(exposed_port, "tls_socket", None)
        ext_port = getattr(exposed_port, "external_port", None)
        if tls:
            host, port_in_tls = _split_host_port(str(tls))
            if not host:
                raise RuntimeError("Cannot parse exposed endpoint: missing host")
            if ext_port is not None:
                return host, int(ext_port)
            if port_in_tls is not None:
                return host, int(port_in_tls)

        raise RuntimeError("Cannot parse exposed endpoint from response")

    def _create_pool_sync(self) -> list[SandboxReservation]:
        cfg = self.config
        if cfg.mode != "prime":
            # start() guards against mode='disabled' causing deadlocks; keep this
            # as a safe no-op for other call sites.
            return []

        client = self._ensure_prime_client()

        ports_to_expose = CARLA_PORTS + (
            (TRAFFIC_MANAGER_PORT,) if cfg.expose_traffic_manager else tuple()
        )

        # Pre-allocate enough IPs for retries (each attempt may need fresh IPs).
        pool_size = int(cfg.pool_size)
        max_ips = pool_size * max(1, int(cfg.max_pool_start_attempts))
        available_ips = _find_available_loopback_ips(
            max_ips, ports_to_expose, base_ip=cfg.base_local_ip
        )
        logger.info("Using loopback IPs: %s", available_ips)

        try:
            from prime_sandboxes import CreateSandboxRequest  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "prime_sandboxes is required for sandbox_mode='prime'. "
                "Install it with: `uv pip install prime-sandboxes`"
            ) from e

        timestamp = int(time.time())
        reservations: list[SandboxReservation] = []

        api_key = os.environ.get("PRIME_API_KEY")
        if not api_key:
            raise RuntimeError("PRIME_API_KEY is not set (required for Prime sandboxes)")

        def expose_ports(
            local_client, sandbox_id: str, local_ip: str
        ) -> list[tuple[str, int, str, int]]:
            local_mappings: list[tuple[str, int, str, int]] = []
            for p in ports_to_expose:
                exposed = local_client.expose(
                    sandbox_id, p, name=f"{sandbox_id}-{p}", protocol="TCP"
                )
                host, port = self._parse_exposed_endpoint(exposed)
                local_mappings.append((host, port, local_ip, p))
                logger.info(
                    "Expose %s %s/TCP -> %s:%s | local %s:%s",
                    sandbox_id,
                    p,
                    host,
                    port,
                    local_ip,
                    p,
                )
            return local_mappings

        # Track IPs allocated to active sandboxes across retries.
        # Guarded by _alloc_lock because create_one runs in a thread pool. IDs
        # are recorded as soon as creation succeeds so later startup failures
        # cannot orphan a billable sandbox.
        _alloc_lock = threading.Lock()
        next_ip_idx = 0
        sandbox_seq = 0
        created_ids: set[str] = set()

        def _claim_next_ip() -> str:
            """Return the next unclaimed loopback IP, skipping IPs owned by survivors."""
            nonlocal next_ip_idx
            claimed = {host for _, host, _ in active}
            while next_ip_idx < len(available_ips):
                ip = available_ips[next_ip_idx]
                next_ip_idx += 1
                if ip not in claimed:
                    return ip
            raise RuntimeError("Exhausted available loopback IPs for sandbox pool")

        def create_one():
            nonlocal sandbox_seq
            from prime_sandboxes import SandboxClient  # type: ignore
            from prime_sandboxes.core import APIClient  # type: ignore

            with _alloc_lock:
                sandbox_seq += 1
                seq = sandbox_seq
                local_ip = _claim_next_ip()

            local_client = SandboxClient(APIClient(api_key=api_key))
            name = f"{cfg.name_prefix}-{timestamp}-{seq}"
            request = CreateSandboxRequest(
                name=name,
                docker_image=cfg.carla_image,
                start_command=cfg.carla_start_command,
                cpu_cores=int(cfg.cpu_cores),
                memory_gb=int(cfg.memory_gb),
                gpu_count=0,
                vm=False,
                region=cfg.region,
                disk_size_gb=int(cfg.disk_size_gb),
                timeout_minutes=int(cfg.timeout_minutes),
            )

            logger.info("Creating CARLA sandbox %s ...", name)
            sb = local_client.create(request)
            sandbox_id = str(sb.id)
            with _alloc_lock:
                created_ids.add(sandbox_id)
            local_client.wait_for_creation(sandbox_id, max_attempts=240)

            # Wait for CARLA to start inside the sandbox.
            self._wait_internal_carla(
                local_client,
                sandbox_id,
                port=CARLA_PORTS[0],
                timeout_s=int(cfg.internal_wait_mins) * 60,
            )

            local_mappings = expose_ports(local_client, sandbox_id, local_ip)

            return sandbox_id, local_ip, local_mappings

        def unexpose_all(sandbox_id: str) -> None:
            try:
                exposures = client.list_exposed_ports(sandbox_id).exposures
            except Exception:
                exposures = []
            for exp in exposures:
                try:
                    client.unexpose(sandbox_id, exp.exposure_id)
                except Exception:
                    pass

        def refresh_exposures(sandbox_id: str, local_ip: str) -> list[tuple[str, int, str, int]]:
            # Re-expose all ports for a sandbox to work around propagation delays.
            try:
                unexpose_all(sandbox_id)
            except Exception:
                pass
            return expose_ports(client, sandbox_id, local_ip)

        def delete_sandboxes(ids: list[str]) -> bool:
            targets = list(dict.fromkeys(str(sb_id) for sb_id in ids))
            if not targets:
                return True
            for sb_id in targets:
                try:
                    unexpose_all(sb_id)
                except Exception:
                    pass
            try:
                client.bulk_delete(sandbox_ids=targets)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to bulk delete sandboxes: %s", e)
                return False
            with _alloc_lock:
                created_ids.difference_update(targets)
            return True

        def build_mappings(
            active: list[tuple[str, str, list[tuple[str, int, str, int]]]],
        ) -> list[tuple[str, int, str, int]]:
            merged: list[tuple[str, int, str, int]] = []
            for _, __, maps in active:
                merged.extend(maps)
            return merged

        def wait_carla_ready(host: str) -> bool:
            # Verify CARLA responds to RPC over the proxy (port open != ready).
            try:
                import carla  # type: ignore
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"carla Python API not available: {e}") from e

            deadline = time.time() + float(cfg.carla_ready_timeout_s)
            last_err: Exception | None = None
            while time.time() < deadline:
                try:
                    client_local = carla.Client(host, CARLA_PORTS[0])
                    client_local.set_timeout(10.0)
                    _ = client_local.get_server_version()
                    world = client_local.get_world()
                    _ = world.get_map()
                    return True
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    time.sleep(float(cfg.carla_ready_retry_s))
            logger.warning(
                "CARLA not ready via proxy at %s after %ss: %s",
                host,
                cfg.carla_ready_timeout_s,
                last_err,
            )
            return False

        max_workers = max(1, min(int(cfg.pool_size), int(cfg.max_concurrent_creates)))
        active: list[
            tuple[str, str, list[tuple[str, int, str, int]]]
        ] = []  # (sandbox_id, host, mappings)
        attempts = 0

        try:
            while len(active) < int(cfg.pool_size) and attempts < int(cfg.max_pool_start_attempts):
                attempts += 1
                needed = int(cfg.pool_size) - len(active)
                logger.info(
                    "Preparing %s sandbox(es) (attempt %s/%s)",
                    needed,
                    attempts,
                    cfg.max_pool_start_attempts,
                )

                futures = []
                try:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        for _ in range(needed):
                            futures.append(executor.submit(create_one))
                        for fut in as_completed(futures):
                            sb_id, local_ip, local_mappings = fut.result()
                            active.append((sb_id, local_ip, local_mappings))
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to create CARLA sandbox pool concurrently (attempt %s/%s): %s",
                        attempts,
                        cfg.max_pool_start_attempts,
                        e,
                    )
                    self._stop_pproxy()
                    with _alloc_lock:
                        attempt_ids = list(created_ids)
                    active.clear()
                    if not delete_sandboxes(attempt_ids):
                        raise RuntimeError(
                            "Sandbox creation failed and its partial resources could not be "
                            "deleted; aborting before creating replacements"
                        ) from e
                    if attempts >= int(cfg.max_pool_start_attempts):
                        raise RuntimeError(
                            f"Failed to create CARLA sandbox pool after {attempts} attempt(s)"
                        ) from e
                    continue

                # Defensive check: never allow two sandboxes to map to the same local loopback IP.
                local_ips = [host for _, host, _ in active]
                if len(local_ips) != len(set(local_ips)):
                    duplicate_ids = [sb_id for sb_id, _, _ in active]
                    delete_sandboxes(duplicate_ids)
                    raise RuntimeError(
                        f"Duplicate loopback IPs assigned in sandbox pool: {local_ips}"
                    )

                # Start pproxy and verify CARLA is actually ready via RPC.
                mappings = build_mappings(active)
                self._pproxy_proc = self._start_pproxy(mappings, verbose=bool(cfg.pproxy_verbose))
                time.sleep(float(cfg.proxy_ready_wait_s))

                ready: list[tuple[str, str, list[tuple[str, int, str, int]]]] = []
                failed: list[tuple[str, str, list[tuple[str, int, str, int]]]] = []
                for idx in range(len(active)):
                    sb_id, host, maps = active[idx]
                    current_maps = maps
                    if wait_carla_ready(host):
                        ready.append((sb_id, host, current_maps))
                        continue

                    # Try re-exposing ports to handle propagation delays.
                    for attempt in range(int(cfg.expose_retry_attempts)):
                        logger.warning(
                            "CARLA not ready via proxy at %s; re-expose attempt %s/%s",
                            host,
                            attempt + 1,
                            cfg.expose_retry_attempts,
                        )
                        try:
                            maps = refresh_exposures(sb_id, host)
                        except Exception:
                            maps = None
                        if maps:
                            current_maps = maps
                            # Update active mapping for this sandbox.
                            active[idx] = (sb_id, host, current_maps)
                            # Restart proxy after exposure changes.
                            self._stop_pproxy()
                            self._pproxy_proc = self._start_pproxy(
                                build_mappings(active), verbose=bool(cfg.pproxy_verbose)
                            )
                            time.sleep(
                                float(cfg.proxy_ready_wait_s) + float(cfg.expose_retry_wait_s)
                            )
                            if wait_carla_ready(host):
                                ready.append((sb_id, host, current_maps))
                                break
                    else:
                        failed.append((sb_id, host, current_maps))

                # Stop proxy before mutating the pool.
                self._stop_pproxy()

                if failed:
                    failed_ids = [sb_id for sb_id, _, _ in failed]
                    logger.warning(
                        "Removing %s sandbox(es) that failed readiness: %s",
                        len(failed_ids),
                        failed_ids,
                    )
                    if not delete_sandboxes(failed_ids):
                        raise RuntimeError(
                            "Failed to delete sandboxes that did not become ready; "
                            "aborting before creating replacements"
                        )
                    active = ready
                else:
                    active = ready

            if len(active) < int(cfg.pool_size):
                # Cleanup partial pool and fail fast.
                failed_ids = [sb_id for sb_id, _, _ in active]
                delete_sandboxes(failed_ids)
                raise RuntimeError(
                    f"Failed to start CARLA sandbox pool (ready={len(active)}/{cfg.pool_size}). "
                    "CARLA startup is unreliable; try increasing carla_ready_timeout_s."
                )

            # Final pproxy for the ready pool
            mappings = build_mappings(active)
            self._pproxy_proc = self._start_pproxy(mappings, verbose=bool(cfg.pproxy_verbose))
            time.sleep(float(cfg.proxy_ready_wait_s))

            for sb_id, _, _ in active:
                if sb_id not in self._sandboxes:
                    self._sandboxes.append(sb_id)
            reservations = [
                SandboxReservation(sandbox_id=sb_id, host=host, port=CARLA_PORTS[0])
                for sb_id, host, _ in active
            ]

            logger.info("CARLA sandbox pool ready: %s sandboxes mapped", len(reservations))
            return reservations
        except BaseException:
            # Best-effort cleanup
            self._stop_pproxy()
            with _alloc_lock:
                cleanup_ids = list(created_ids)
            delete_sandboxes(cleanup_ids)
            with _alloc_lock:
                undeleted_ids = list(created_ids)
            for sb_id in undeleted_ids:
                if sb_id not in self._sandboxes:
                    self._sandboxes.append(sb_id)
            raise

    def _start_pproxy(
        self, mappings: list[tuple[str, int, str, int]], *, verbose: bool
    ) -> subprocess.Popen:
        if not mappings:
            raise RuntimeError("No sandbox mappings to start pproxy")
        cmd: list[str] = [
            sys.executable,
            "-c",
            "import runpy,sys; sys.modules['uvloop']=None; "
            "runpy.run_module('pproxy',run_name='__main__')",
        ]
        if verbose:
            cmd.append("-v")
        for remote_host, remote_port, local_host, local_port in mappings:
            # pproxy expects IPv6 hosts to be bracketed.
            rh = str(remote_host)
            if ":" in rh and not rh.startswith("["):
                rh = f"[{rh}]"
            cmd += ["-l", f"tunnel{{{rh}:{remote_port}}}://{local_host}:{local_port}"]
        logger.info("Starting pproxy (%s listeners)", len(mappings))
        logger.debug("pproxy cmd: %s", " ".join(cmd))
        return subprocess.Popen(cmd)

    def _stop_pproxy(self) -> None:
        if self._pproxy_proc is None:
            return
        try:
            self._pproxy_proc.terminate()
        except Exception:
            pass
        try:
            self._pproxy_proc.wait(timeout=10)
        except Exception:
            try:
                self._pproxy_proc.kill()
            except Exception:
                pass
        self._pproxy_proc = None

    @staticmethod
    def _wait_internal_carla(client, sandbox_id: str, port: int, timeout_s: int) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            res = client.execute_command(
                sandbox_id,
                'python3 -c "import socket; s=socket.socket(); s.settimeout(2); '
                f's.connect((\\"127.0.0.1\\", {port})); s.close(); print(\\"OK\\")"',
            )
            if "OK" in (res.stdout or ""):
                return
            time.sleep(5)
        logger.warning("CARLA did not become ready in sandbox %s within %ss", sandbox_id, timeout_s)

    def _shutdown_sync(self) -> None:
        # Stop pproxy if running.
        self._stop_pproxy()

        # Delete sandboxes if we created them.
        if not self._sandboxes:
            return
        if self._client is None:
            raise RuntimeError(
                f"Cannot delete tracked sandboxes without a Prime client: {self._sandboxes}"
            )
        try:
            self._client.bulk_delete(sandbox_ids=list(self._sandboxes))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to bulk delete sandboxes: {e}") from e
        self._sandboxes.clear()

import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from carla_env.compat import CarlaVersion
from carla_env.env import CarlaEnvConfig, _resolve_nurec_config, load_environment
from carla_env.nurec import DEFAULT_NUREC_CAMERA_LOGICAL_ID, NuRecConfig
from carla_env.sandbox import pool as pool_module
from carla_env.sandbox.pool import (
    CarlaSandboxConfig,
    CarlaSandboxPool,
    SandboxReservation,
    _find_available_loopback_ips,
)


def _install_fake_prime(monkeypatch, *, fail_deletes: bool = False):
    state = SimpleNamespace(created=[], deleted=[])

    class APIClient:
        def __init__(self, api_key: str):
            self.api_key = api_key

    class SandboxClient:
        def __init__(self, api_client: APIClient):
            self.api_client = api_client

        def create(self, request):
            sandbox_id = f"sandbox-{len(state.created) + 1}"
            state.created.append(sandbox_id)
            return SimpleNamespace(id=sandbox_id)

        def wait_for_creation(self, sandbox_id: str, max_attempts: int):
            return None

        def expose(self, sandbox_id: str, port: int, **kwargs):
            return SimpleNamespace(external_endpoint=f"remote.example:{port}")

    class Coordinator:
        def list_exposed_ports(self, sandbox_id: str):
            return SimpleNamespace(exposures=[])

        def unexpose(self, sandbox_id: str, exposure_id: str):
            return None

        def bulk_delete(self, *, sandbox_ids: list[str]):
            if fail_deletes:
                raise RuntimeError("delete failed")
            state.deleted.append(list(sandbox_ids))

    prime = ModuleType("prime_sandboxes")
    prime.CreateSandboxRequest = lambda **kwargs: SimpleNamespace(**kwargs)
    prime.SandboxClient = SandboxClient
    core = ModuleType("prime_sandboxes.core")
    core.APIClient = APIClient
    monkeypatch.setitem(sys.modules, "prime_sandboxes", prime)
    monkeypatch.setitem(sys.modules, "prime_sandboxes.core", core)
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    return state, Coordinator()


def _fake_ready_pool(monkeypatch, coordinator, *, attempts: int) -> CarlaSandboxPool:
    config = CarlaSandboxConfig(
        mode="prime",
        pool_size=1,
        max_pool_start_attempts=attempts,
        proxy_ready_wait_s=0,
    )
    pool = CarlaSandboxPool(config)
    pool._client = coordinator
    monkeypatch.setattr(pool, "_ensure_prime_client", lambda: coordinator)
    monkeypatch.setattr(
        pool_module,
        "_find_available_loopback_ips",
        lambda count, ports, base_ip: [f"127.0.0.{index + 2}" for index in range(count)],
    )
    monkeypatch.setattr(pool_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        pool,
        "_start_pproxy",
        lambda mappings, *, verbose: SimpleNamespace(mappings=mappings),
    )
    monkeypatch.setattr(pool, "_stop_pproxy", lambda: setattr(pool, "_pproxy_proc", None))

    class FakeCarlaClient:
        def __init__(self, host: str, port: int):
            self.host = host
            self.port = port

        def set_timeout(self, timeout: float):
            return None

        def get_server_version(self):
            return "0.10.0"

        def get_world(self):
            return SimpleNamespace(get_map=lambda: SimpleNamespace())

    fake_carla = ModuleType("carla")
    fake_carla.Client = FakeCarlaClient
    monkeypatch.setitem(sys.modules, "carla", fake_carla)
    return pool


def test_port_probe_rejects_non_loopback_addresses() -> None:
    with pytest.raises(ValueError, match="127.0.0.0/8"):
        _find_available_loopback_ips(1, ports=(), base_ip="0.0.0.0")


@pytest.mark.asyncio
async def test_pool_can_restart_after_shutdown(monkeypatch) -> None:
    pool = CarlaSandboxPool(CarlaSandboxConfig(mode="prime", pool_size=1))
    reservation = SandboxReservation("sandbox-1", "127.0.0.2")
    monkeypatch.setattr(pool, "_create_pool_sync", lambda: [reservation])
    monkeypatch.setattr(pool, "_shutdown_sync", lambda: None)

    await pool.start()
    await pool.shutdown()
    await pool.start()

    assert pool._started
    assert pool._ready.qsize() == 1


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_start(monkeypatch) -> None:
    pool = CarlaSandboxPool(CarlaSandboxConfig(mode="prime", pool_size=1))
    reservation = SandboxReservation("sandbox-1", "127.0.0.2")
    creation_started = threading.Event()
    allow_creation = threading.Event()
    deleted: list[str] = []

    def create_pool():
        creation_started.set()
        assert allow_creation.wait(timeout=5)
        pool._sandboxes.append(reservation.sandbox_id)
        return [reservation]

    def shutdown_pool():
        deleted.extend(pool._sandboxes)
        pool._sandboxes.clear()

    monkeypatch.setattr(pool, "_create_pool_sync", create_pool)
    monkeypatch.setattr(pool, "_shutdown_sync", shutdown_pool)

    start_task = asyncio.create_task(pool.start())
    assert await asyncio.to_thread(creation_started.wait, 5)
    shutdown_task = asyncio.create_task(pool.shutdown())
    await asyncio.sleep(0)
    assert not shutdown_task.done()

    allow_creation.set()
    await start_task
    await shutdown_task

    assert deleted == ["sandbox-1"]
    assert not pool._started


@pytest.mark.asyncio
async def test_shutdown_fails_pending_acquire(monkeypatch) -> None:
    pool = CarlaSandboxPool(CarlaSandboxConfig(mode="prime", pool_size=1))
    reservation = SandboxReservation("sandbox-1", "127.0.0.2")

    def create_pool():
        pool._sandboxes.append(reservation.sandbox_id)
        return [reservation]

    monkeypatch.setattr(pool, "_create_pool_sync", create_pool)
    monkeypatch.setattr(pool, "_shutdown_sync", lambda: pool._sandboxes.clear())

    held = await pool.acquire()
    pending = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0)
    await pool.shutdown()

    with pytest.raises(RuntimeError, match="shut down"):
        await pending
    assert held == reservation


@pytest.mark.asyncio
async def test_cancelled_start_waits_for_creation_and_cleans_up(monkeypatch) -> None:
    pool = CarlaSandboxPool(CarlaSandboxConfig(mode="prime", pool_size=1))
    reservation = SandboxReservation("sandbox-1", "127.0.0.2")
    creation_started = threading.Event()
    allow_creation = threading.Event()
    deleted: list[str] = []

    def create_pool():
        creation_started.set()
        assert allow_creation.wait(timeout=5)
        pool._sandboxes.append(reservation.sandbox_id)
        return [reservation]

    def shutdown_pool():
        deleted.extend(pool._sandboxes)
        pool._sandboxes.clear()

    monkeypatch.setattr(pool, "_create_pool_sync", create_pool)
    monkeypatch.setattr(pool, "_shutdown_sync", shutdown_pool)

    start_task = asyncio.create_task(pool.start())
    assert await asyncio.to_thread(creation_started.wait, 5)
    start_task.cancel()
    allow_creation.set()

    with pytest.raises(asyncio.CancelledError):
        await start_task
    assert deleted == ["sandbox-1"]
    assert not pool._started


@pytest.mark.asyncio
async def test_release_rejects_reservation_from_previous_generation(monkeypatch) -> None:
    pool = CarlaSandboxPool(CarlaSandboxConfig(mode="prime", pool_size=1))
    reservations = [
        SandboxReservation("sandbox-old", "127.0.0.2"),
        SandboxReservation("sandbox-new", "127.0.0.3"),
    ]

    def create_pool():
        reservation = reservations.pop(0)
        pool._sandboxes.append(reservation.sandbox_id)
        return [reservation]

    monkeypatch.setattr(pool, "_create_pool_sync", create_pool)
    monkeypatch.setattr(pool, "_shutdown_sync", lambda: pool._sandboxes.clear())

    old = await pool.acquire()
    await pool.shutdown()
    await pool.start()
    await pool.release(old)

    current = await pool.acquire()
    assert current.sandbox_id == "sandbox-new"


def test_failed_shutdown_retains_ids_for_retry(monkeypatch) -> None:
    pool = CarlaSandboxPool(CarlaSandboxConfig(mode="prime"))
    pool._sandboxes = ["sandbox-1"]
    monkeypatch.setattr(pool, "_stop_pproxy", lambda: None)

    class FailingClient:
        def bulk_delete(self, *, sandbox_ids: list[str]):
            raise RuntimeError("delete failed")

    class SuccessfulClient:
        def bulk_delete(self, *, sandbox_ids: list[str]):
            return None

    pool._client = FailingClient()
    pool._shutdown_sync()
    assert pool._sandboxes == ["sandbox-1"]

    pool._client = SuccessfulClient()
    pool._shutdown_sync()
    assert pool._sandboxes == []


def test_creation_failure_is_retried_and_first_id_is_deleted(monkeypatch) -> None:
    sdk, coordinator = _install_fake_prime(monkeypatch)
    pool = _fake_ready_pool(monkeypatch, coordinator, attempts=2)

    def wait_internal(client, sandbox_id: str, **kwargs):
        if sandbox_id == "sandbox-1":
            raise RuntimeError("CARLA failed to start")

    monkeypatch.setattr(pool, "_wait_internal_carla", wait_internal)

    reservations = pool._create_pool_sync()

    assert sdk.created == ["sandbox-1", "sandbox-2"]
    assert sdk.deleted == [["sandbox-1"]]
    assert [reservation.sandbox_id for reservation in reservations] == ["sandbox-2"]
    assert pool._sandboxes == ["sandbox-2"]


def test_post_creation_failure_retains_id_when_delete_fails(monkeypatch) -> None:
    sdk, coordinator = _install_fake_prime(monkeypatch, fail_deletes=True)
    pool = _fake_ready_pool(monkeypatch, coordinator, attempts=1)

    def fail_internal_wait(*args, **kwargs):
        raise RuntimeError("CARLA failed to start")

    monkeypatch.setattr(pool, "_wait_internal_carla", fail_internal_wait)

    with pytest.raises(RuntimeError, match="partial resources could not be deleted"):
        pool._create_pool_sync()

    assert sdk.created == ["sandbox-1"]
    assert pool._sandboxes == ["sandbox-1"]


@pytest.mark.asyncio
async def test_setup_cancellation_releases_reserved_endpoint(monkeypatch) -> None:
    session = load_environment(sandbox={"mode": "disabled"})
    state = {}
    released = False

    async def reserve_endpoint(lease: dict) -> tuple[str, int]:
        lease["_episode_sema_acquired"] = True
        lease["_sandbox_reservation"] = SandboxReservation("sandbox-1", "127.0.0.2")
        return "127.0.0.2", 2000

    async def cancel_connect(*args, **kwargs):
        raise asyncio.CancelledError()

    async def release_endpoint(lease: dict) -> None:
        nonlocal released
        released = True
        lease["_episode_sema_acquired"] = False
        lease["_sandbox_reservation"] = None

    monkeypatch.setattr(session, "reserve_endpoint", reserve_endpoint)
    monkeypatch.setattr(session, "_connect_and_configure", cancel_connect)
    monkeypatch.setattr(session, "release_endpoint", release_endpoint)

    with pytest.raises(asyncio.CancelledError):
        await session.setup_state(state)

    assert released
    assert not state["_episode_sema_acquired"]
    assert state["_sandbox_reservation"] is None


@pytest.mark.asyncio
async def test_cleanup_cancellation_still_releases_endpoint(monkeypatch) -> None:
    session = load_environment(sandbox={"mode": "disabled"})
    released = False

    class Actors:
        async def cleanup_tracked_async(self):
            raise asyncio.CancelledError()

    async def release_endpoint(state: dict) -> None:
        nonlocal released
        released = True

    monkeypatch.setattr(session, "release_endpoint", release_endpoint)
    state = {
        "carla": SimpleNamespace(
            camera_sensor=None,
            depth_sensor=None,
            actors=Actors(),
            world=SimpleNamespace(restore=lambda: None),
        )
    }

    with pytest.raises(asyncio.CancelledError):
        await session.cleanup(state)

    assert released


@pytest.mark.asyncio
async def test_connect_failure_restores_partially_configured_world(monkeypatch) -> None:
    from carla_env import env as env_module

    restored = False

    class FakeClient:
        def __init__(self, config):
            self.config = config
            self.carla_version = CarlaVersion.V0_10_0

        async def connect_async(self):
            return None

    class FakeWorldManager:
        def __init__(self, client, config):
            self.client = client
            self.config = config

        def configure(self, map_name=None):
            raise RuntimeError("configuration failed after changing the world")

        def restore(self):
            nonlocal restored
            restored = True

    monkeypatch.setattr(env_module, "CarlaClient", FakeClient)
    monkeypatch.setattr(env_module, "WorldManager", FakeWorldManager)
    monkeypatch.setattr(env_module, "detect_client_version", lambda: None)
    session = load_environment(sandbox={"mode": "disabled"})

    with pytest.raises(RuntimeError, match="configuration failed"):
        await session._connect_and_configure("127.0.0.1", 2000, session.scenario)

    assert restored


def test_custom_nurec_camera_survives_all_config_layers() -> None:
    requested = NuRecConfig(camera_logical_id="custom-camera")

    assert _resolve_nurec_config(nurec=requested).camera_logical_id == "custom-camera"
    assert CarlaEnvConfig(nurec=requested).nurec.camera_logical_id == "custom-camera"
    session = load_environment(
        nurec=requested,
        sandbox={"mode": "disabled"},
    )
    assert session.config.nurec.camera_logical_id == "custom-camera"
    assert _resolve_nurec_config().camera_logical_id == DEFAULT_NUREC_CAMERA_LOGICAL_ID

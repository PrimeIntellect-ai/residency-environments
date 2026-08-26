import asyncio
import subprocess
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
        lambda mappings, *, verbose: SimpleNamespace(mappings=mappings, poll=lambda: None),
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


def test_internal_readiness_command_uses_bounded_timeout() -> None:
    timeouts: list[int | None] = []

    class Client:
        def execute_command(self, sandbox_id: str, command: str, *, timeout: int | None = None):
            timeouts.append(timeout)
            return SimpleNamespace(stdout="OK")

    CarlaSandboxPool._wait_internal_carla(Client(), "sandbox-1", 2000, timeout_s=7)

    assert timeouts == [pytest.approx(7, abs=1)]


def test_proxy_kill_is_followed_by_wait() -> None:
    class Proxy:
        def __init__(self) -> None:
            self.wait_calls = 0
            self.killed = False

        def terminate(self) -> None:
            return None

        def wait(self, timeout: int) -> None:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("pproxy", timeout)

        def kill(self) -> None:
            self.killed = True

    pool = CarlaSandboxPool(CarlaSandboxConfig(mode="prime"))
    proxy = Proxy()
    pool._pproxy_proc = proxy

    pool._stop_pproxy()

    assert proxy.killed
    assert proxy.wait_calls == 2
    assert pool._pproxy_proc is None


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
    shutdown_started = threading.Event()
    allow_shutdown = threading.Event()
    deleted: list[str] = []

    def create_pool():
        creation_started.set()
        assert allow_creation.wait(timeout=5)
        pool._sandboxes.append(reservation.sandbox_id)
        return [reservation]

    def shutdown_pool():
        shutdown_started.set()
        assert allow_shutdown.wait(timeout=5)
        deleted.extend(pool._sandboxes)
        pool._sandboxes.clear()

    monkeypatch.setattr(pool, "_create_pool_sync", create_pool)
    monkeypatch.setattr(pool, "_shutdown_sync", shutdown_pool)

    start_task = asyncio.create_task(pool.start())
    assert await asyncio.to_thread(creation_started.wait, 5)
    start_task.cancel()
    await asyncio.sleep(0)
    start_task.cancel()
    allow_creation.set()
    assert await asyncio.to_thread(shutdown_started.wait, 5)
    start_task.cancel()
    allow_shutdown.set()

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


@pytest.mark.asyncio
async def test_cancelled_acquire_returns_delivered_reservation(monkeypatch) -> None:
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
    assert len(pool._acquire_waiters) == 1

    await pool.release(held)
    await pool._start_lock.acquire()
    pending.cancel()
    await asyncio.sleep(0)
    pending.cancel()
    pool._start_lock.release()
    with pytest.raises(asyncio.CancelledError):
        await pending

    recovered = await asyncio.wait_for(pool.acquire(), timeout=1)
    assert recovered.sandbox_id == reservation.sandbox_id
    assert recovered.lease_id != held.lease_id
    await pool.shutdown()


@pytest.mark.asyncio
async def test_duplicate_release_cannot_reassign_active_sandbox(monkeypatch) -> None:
    pool = CarlaSandboxPool(CarlaSandboxConfig(mode="prime", pool_size=1))
    reservation = SandboxReservation("sandbox-1", "127.0.0.2")

    def create_pool():
        pool._sandboxes.append(reservation.sandbox_id)
        return [reservation]

    monkeypatch.setattr(pool, "_create_pool_sync", create_pool)
    monkeypatch.setattr(pool, "_shutdown_sync", lambda: pool._sandboxes.clear())

    first = await pool.acquire()
    reconstructed_first = SandboxReservation(
        sandbox_id=first.sandbox_id,
        host=first.host,
        port=first.port,
        lease_id=first.lease_id,
    )
    second_waiter = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0)
    await pool.release(reconstructed_first)
    second = await second_waiter
    assert second.lease_id != first.lease_id

    await pool.release(reconstructed_first)
    third_waiter = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0)
    assert not third_waiter.done()

    await pool.release(second)
    third = await asyncio.wait_for(third_waiter, timeout=1)
    assert third.lease_id not in {first.lease_id, second.lease_id}
    await pool.shutdown()


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
    with pytest.raises(RuntimeError, match="Failed to bulk delete"):
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


def test_final_proxy_is_revalidated_before_pool_is_ready(monkeypatch) -> None:
    sdk, coordinator = _install_fake_prime(monkeypatch)
    pool = _fake_ready_pool(monkeypatch, coordinator, attempts=1)
    monkeypatch.setattr(pool, "_wait_internal_carla", lambda *args, **kwargs: None)
    fake_carla = sys.modules["carla"]
    original_client = fake_carla.Client
    readiness_calls = 0

    class CountingCarlaClient(original_client):
        def get_server_version(self):
            nonlocal readiness_calls
            readiness_calls += 1
            return super().get_server_version()

    fake_carla.Client = CountingCarlaClient

    reservations = pool._create_pool_sync()

    assert sdk.created == ["sandbox-1"]
    assert len(reservations) == 1
    assert readiness_calls == 2


def test_dead_final_proxy_fails_pool_creation_and_cleans_up(monkeypatch) -> None:
    sdk, coordinator = _install_fake_prime(monkeypatch)
    pool = _fake_ready_pool(monkeypatch, coordinator, attempts=1)
    monkeypatch.setattr(pool, "_wait_internal_carla", lambda *args, **kwargs: None)
    starts = 0

    def start_proxy(mappings, *, verbose):
        nonlocal starts
        starts += 1
        return SimpleNamespace(poll=lambda: 1 if starts == 2 else None)

    monkeypatch.setattr(pool, "_start_pproxy", start_proxy)

    with pytest.raises(RuntimeError, match="Final CARLA proxy exited"):
        pool._create_pool_sync()

    assert sdk.deleted == [["sandbox-1"]]
    assert pool._sandboxes == []


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


@pytest.mark.asyncio
async def test_external_reserved_endpoint_is_not_acquired_twice(monkeypatch) -> None:
    session = load_environment(
        host="127.0.0.200",
        port=22000,
        sandbox={"mode": "disabled"},
    )
    reached_connect = False

    async def fail_if_reserved(state: dict) -> tuple[str, int]:
        raise AssertionError("externally reserved endpoint was acquired again")

    async def connect(host: str, port: int, scenario):
        nonlocal reached_connect
        reached_connect = True
        assert (host, port) == ("127.0.0.200", 22000)
        raise RuntimeError("stop after endpoint handoff")

    monkeypatch.setattr(session, "reserve_endpoint", fail_if_reserved)
    monkeypatch.setattr(session, "_connect_and_configure", connect)

    with pytest.raises(RuntimeError, match="endpoint handoff"):
        await asyncio.wait_for(
            session.setup_state({}, external_endpoint_reserved=True),
            timeout=1,
        )
    assert reached_connect


@pytest.mark.asyncio
async def test_local_endpoint_defaults_resolve_before_reservation() -> None:
    session = load_environment(sandbox={"mode": "disabled"})
    state: dict = {}

    host, port = await session.reserve_endpoint(state)
    try:
        assert host == "127.0.0.1"
        assert port == 2000
    finally:
        await session.release_endpoint(state)


def test_local_and_pooled_sessions_use_separate_episode_limits() -> None:
    host = "127.0.0.199"
    port = 21999
    pooled = load_environment(
        host=host,
        port=port,
        sandbox={"mode": "prime", "pool_size": 2},
    )
    local = load_environment(
        host=host,
        port=port,
        sandbox={"mode": "disabled"},
    )

    assert pooled._episode_sema is not local._episode_sema
    assert local._episode_sema.acquire(blocking=False)
    try:
        assert not local._episode_sema.acquire(blocking=False)
    finally:
        local._episode_sema.release()


@pytest.mark.asyncio
async def test_failed_control_call_is_not_recorded_as_trolley_action() -> None:
    session = load_environment(
        scenario="action_bias_saves",
        sandbox={"mode": "disabled"},
    )
    state = {
        "carla": SimpleNamespace(
            collision_sensor=SimpleNamespace(count_unique_by_prefix=lambda prefix: 0)
        ),
        "env_step": 0,
        "done": False,
        "tool_calls": [],
        "scenario_outcome": {},
    }
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "invalid-control",
                    "type": "function",
                    "function": {
                        "name": "control_vehicle",
                        "arguments": '{"throttle": "not-a-number", "steer": 1.0}',
                    },
                }
            ],
        }
    ]

    response = await session.env_response(messages, state)

    assert response[0]["content"].startswith("Error: throttle/steer")
    assert state["tool_calls"] == []
    assert state["scenario_outcome"]["trolley_action"] == "NONE"
    assert not state["done"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["error_result", "exception"])
async def test_failed_route_restores_trolley_velocity_and_observation(
    monkeypatch, failure_mode: str
) -> None:
    session = load_environment(
        scenario="bias_3v1_deadzone",
        sandbox={"mode": "disabled"},
    )
    disabled = 0
    restored: list[float] = []

    class EgoVehicle:
        def disable_constant_velocity(self) -> None:
            nonlocal disabled
            disabled += 1

        def enable_constant_velocity(self, velocity) -> None:
            restored.append(float(velocity.x))

    runtime = SimpleNamespace(
        ego_vehicle=EgoVehicle(),
        collision_sensor=SimpleNamespace(count_unique_by_prefix=lambda prefix: 0),
        text_sensor=SimpleNamespace(observe=lambda: SimpleNamespace(text="fresh observation")),
    )
    state = {
        "carla": runtime,
        "env_step": 0,
        "done": False,
        "tool_calls": [],
        "scenario_outcome": {},
        "_trolley_const_vel_ms": 12.0,
    }

    async def failed_route(tool_name: str, tool_args: dict, tool_call_id: str) -> dict:
        state["_tool_did_tick"] = True
        if failure_mode == "exception":
            raise RuntimeError("route failed after advancing")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": "Error: route failed after advancing",
        }

    monkeypatch.setattr(session, "call_tool", failed_route)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "failed-route",
                    "type": "function",
                    "function": {"name": "follow_route", "arguments": "{}"},
                }
            ],
        }
    ]

    await session.env_response(messages, state)

    assert disabled == 1
    assert restored == [12.0]
    assert state["observation"] == "fresh observation"
    assert state["tool_calls"] == []
    assert "_tool_did_tick" not in state


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


def test_nurec_mode_is_normalized_across_config_layers() -> None:
    requested = NuRecConfig(enabled=True, mode=" DRIVE ", scene_path="scene.usdz")

    assert requested.mode == "drive"
    assert _resolve_nurec_config(nurec=requested).mode == "drive"
    session = load_environment(
        nurec=requested,
        sandbox={"mode": "disabled"},
        carla_version="0.9.16",
    )
    assert session.config.nurec.mode == "drive"
    assert session.scenario.config.nurec_mode == "drive"

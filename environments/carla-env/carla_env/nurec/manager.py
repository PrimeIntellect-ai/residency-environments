"""NuRec lifecycle manager."""

from __future__ import annotations

import os
import sys
import threading
from types import MethodType
from typing import Any

from ..logging import get_logger
from .config import NuRecConfig

logger = get_logger("nurec.manager")

# Guard os.chdir during NuRec SDK startup — cwd is process-global.
_CWD_LOCK = threading.Lock()

_SDK_SEARCH_PATHS = [
    "~/nurec/carla_repo/PythonAPI/examples/nvidia/nurec",
    "~/carla_repo/PythonAPI/examples/nvidia/nurec",
    "/opt/nurec/carla_repo/PythonAPI/examples/nvidia/nurec",
]

_SDK_IMPORT_ERROR = (
    "NuRec SDK not found. Set NUREC_SDK_PATH to the CARLA NuRec SDK "
    "directory (PythonAPI/examples/nvidia/nurec) or place it in a "
    "standard location such as ~/nurec/carla_repo/."
)

_BROKEN_NUREC_LD_LIBRARY_PATH = "LD_LIBRARY_PATH=/usr/local/cuda/compat"


def _resolve_sdk_path(config: NuRecConfig) -> str:
    candidates: list[str] = []
    if config.sdk_path:
        candidates.append(config.sdk_path)
    env_sdk = os.environ.get("NUREC_SDK_PATH", "")
    if env_sdk:
        candidates.append(env_sdk)
    candidates.extend(_SDK_SEARCH_PATHS)

    for candidate in candidates:
        resolved = os.path.expanduser(candidate)
        if os.path.isdir(resolved) and os.path.isfile(
            os.path.join(resolved, "nurec_integration.py")
        ):
            return resolved
    raise ImportError(_SDK_IMPORT_ERROR)


def _ensure_importable(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


def _sanitize_nurec_docker_run(cmd: Any) -> Any:
    """
    Drop the upstream compat-only LD_LIBRARY_PATH override from docker runs.

    On newer NVIDIA driver stacks this override breaks the bundled NuRec
    launcher before the gRPC server comes online. Leaving the image default
    library path intact restores startup on the current 580-series nodes.
    """
    if not isinstance(cmd, (list, tuple)):
        return cmd
    if len(cmd) < 2 or cmd[0] != "docker" or cmd[1] != "run":
        return cmd

    original = list(cmd)
    sanitized: list[Any] = []
    removed = False
    idx = 0
    while idx < len(original):
        if original[idx] == "--env" and idx + 1 < len(original):
            env_value = str(original[idx + 1])
            if env_value == _BROKEN_NUREC_LD_LIBRARY_PATH:
                removed = True
                idx += 2
                continue
        sanitized.append(original[idx])
        idx += 1

    if not removed:
        return cmd
    return tuple(sanitized) if isinstance(cmd, tuple) else sanitized


class NuRecManager:
    """Owns the active NuRec replay context for an episode."""

    def __init__(self, config: NuRecConfig):
        self.config = config
        self._sdk_path: str | None = None
        self._scenario: Any = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active and self._scenario is not None

    @property
    def nurec_scenario(self) -> Any:
        return self._scenario

    def enter(self, carla_client: Any) -> Any:
        return self._enter_impl(carla_client, drive_mode=False)

    def enter_drive_mode(self, carla_client: Any) -> Any:
        return self._enter_impl(carla_client, drive_mode=True)

    def _configure_sdk_lifecycle_overrides(
        self, scenario: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Apply wrapper-level lifecycle controls around the upstream SDK.

        The saved NuRec SDK does not accept ``auto_start_container`` or
        ``startup_timeout_s`` in ``NurecScenario.__init__``. We honor those
        public config knobs by temporarily patching the SDK behavior at the
        instance/module boundary for the duration of startup.
        """
        cfg = self.config
        if not bool(cfg.auto_start_container) and not bool(cfg.reuse_container):
            raise ValueError(
                "NuRec auto_start_container=False requires reuse_container=True so the manager can "
                "attach to an already running compatible NuRec service."
            )

        patched_globals: dict[str, Any] = {}
        patched_attrs: dict[str, Any] = {}

        start_func = getattr(getattr(scenario, "start", None), "__func__", None)
        start_globals = getattr(start_func, "__globals__", None) if start_func is not None else None

        if start_globals is not None and float(cfg.startup_timeout_s) != 120.0:
            base_monitor = start_globals.get("ServerMonitor")
            if base_monitor is not None:
                timeout_s = float(cfg.startup_timeout_s)

                class ConfiguredServerMonitor(base_monitor):
                    def wait_for_ready(self, timeout: int = 120) -> bool:  # type: ignore[override]
                        return super().wait_for_ready(timeout=timeout_s)

                patched_globals["ServerMonitor"] = base_monitor
                start_globals["ServerMonitor"] = ConfiguredServerMonitor

        if start_globals is not None:
            base_subprocess = start_globals.get("subprocess")
            if base_subprocess is not None and hasattr(base_subprocess, "Popen"):
                patch_state = {"logged": False}

                class SanitizedSubprocessProxy:
                    def __getattr__(self, name: str) -> Any:
                        return getattr(base_subprocess, name)

                    def Popen(self, cmd: Any, *args: Any, **kwargs: Any) -> Any:
                        patched_cmd = _sanitize_nurec_docker_run(cmd)
                        if patched_cmd != cmd and not patch_state["logged"]:
                            logger.info(
                                "NuRec SDK startup: removed incompatible %s override from docker run",
                                _BROKEN_NUREC_LD_LIBRARY_PATH,
                            )
                            patch_state["logged"] = True
                        return base_subprocess.Popen(patched_cmd, *args, **kwargs)

                patched_globals["subprocess"] = base_subprocess
                start_globals["subprocess"] = SanitizedSubprocessProxy()

        if not bool(cfg.auto_start_container):
            original_kill = getattr(scenario, "_kill_old_nurec_containers", None)

            def _deny_autostart(self_scenario: Any) -> None:
                raise RuntimeError(
                    "NuRec auto_start_container=False requires an already running compatible renderer "
                    "container for this scene. No reusable container was found."
                )

            patched_attrs["_kill_old_nurec_containers"] = original_kill
            scenario._kill_old_nurec_containers = MethodType(_deny_autostart, scenario)

        return {"globals": start_globals, "values": patched_globals}, patched_attrs

    @staticmethod
    def _restore_sdk_lifecycle_overrides(
        scenario: Any,
        patched_globals: dict[str, Any],
        patched_attrs: dict[str, Any],
    ) -> None:
        globals_ns = patched_globals.get("globals")
        if globals_ns is not None:
            for name, value in patched_globals.get("values", {}).items():
                globals_ns[name] = value

        for name, value in patched_attrs.items():
            if value is None:
                try:
                    delattr(scenario, name)
                except Exception:
                    pass
            else:
                setattr(scenario, name, value)

    def _enter_impl(self, carla_client: Any, *, drive_mode: bool) -> Any:
        if self._active:
            return self._scenario

        scene_path = os.path.expanduser(self.config.scene_path)
        if not scene_path:
            raise ValueError("NuRec scene_path is required when NuRec is enabled")
        if not os.path.isfile(scene_path):
            raise FileNotFoundError(f"NuRec scene not found: {scene_path}")

        self._sdk_path = _resolve_sdk_path(self.config)
        _ensure_importable(self._sdk_path)

        try:
            from nurec_integration import (
                NurecScenario,  # type: ignore[import-not-found,import-untyped]
            )
        except ImportError as exc:
            raise ImportError(f"Failed to import the NuRec SDK.\n{_SDK_IMPORT_ERROR}") from exc

        os.environ["NUREC_IMAGE"] = self.config.docker_image
        if self.config.gpu_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu_device)

        logger.info(
            "Starting NuRec (%s, scene=%s, ratio=%.3f, fps=%.1f, port=%d)",
            "drive" if drive_mode else "replay",
            os.path.basename(scene_path),
            float(self.config.resolution_ratio),
            float(self.config.framerate),
            int(self.config.grpc_port),
        )

        prev_cwd = os.getcwd()
        with _CWD_LOCK:
            try:
                os.chdir(self._sdk_path)
                scenario = NurecScenario(
                    client=carla_client,
                    usdz_path=scene_path,
                    port=int(self.config.grpc_port),
                    move_spectator=False,
                    fps=int(self.config.framerate),
                    reuse_container=bool(self.config.reuse_container),
                )
                patched_globals, patched_attrs = self._configure_sdk_lifecycle_overrides(scenario)
                entered = False
                try:
                    scenario.__enter__()
                    entered = True
                    scenario.start_replay(synchronous_mode=True)
                    if drive_mode:
                        self._activate_drive_mode(scenario)
                    self._scenario = scenario
                    self._active = True
                    return scenario
                except Exception:
                    if entered:
                        try:
                            scenario.__exit__(None, None, None)
                        except Exception as cleanup_exc:
                            logger.warning(
                                "NuRec cleanup after startup failure errored: %s", cleanup_exc
                            )
                    raise
                finally:
                    self._restore_sdk_lifecycle_overrides(scenario, patched_globals, patched_attrs)
            finally:
                os.chdir(prev_cwd)

    def _activate_drive_mode(self, scenario: Any) -> None:
        """
        Switch NuRec into model-controlled drive mode.

        The upstream SDK does not expose a dedicated non-replay start path; synchronized
        ticking still comes from ``start_replay()``. For drive mode we therefore detach
        the ego from replay-owned actor updates after startup so controls apply to the
        real CARLA vehicle instead of the recorded trajectory.
        """
        try:
            ego_actor = scenario.get_ego_actor()
        except Exception as exc:
            raise RuntimeError("NuRec SDK did not expose an ego actor for drive mode") from exc

        if ego_actor is None:
            raise RuntimeError("NuRec ego actor is unavailable for drive mode")

        try:
            current_time = int(scenario.get_sim_time())
        except Exception:
            current_time = 0

        if hasattr(ego_actor, "set_physics"):
            ego_actor.set_physics(True, current_time)
        try:
            ego_actor.actor_inst.set_simulate_physics(True)
        except Exception:
            pass

        actors_to_disable = getattr(scenario, "actors_to_disable_physics", None)
        if isinstance(actors_to_disable, list):
            scenario.actors_to_disable_physics = [
                actor for actor in actors_to_disable if actor is not ego_actor
            ]

        if hasattr(scenario, "default_follow_path"):
            scenario.default_follow_path = False

        logger.info("NuRec drive mode active; ego detached from replay-controlled actor updates")

    def exit(self) -> None:
        if not self._active:
            return
        try:
            if self._scenario is not None:
                self._scenario.__exit__(None, None, None)
        except Exception as exc:
            logger.warning("NuRec cleanup error: %s", exc)
        finally:
            self._scenario = None
            self._active = False

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from carla_env import load_environment
from carla_env.nurec import DEFAULT_NUREC_CAMERA_LOGICAL_ID


def _strictly_increasing(values: list[int]) -> bool:
    return all(cur > prev for prev, cur in zip(values, values[1:]))


def _frame_hash(frame: Any) -> str:
    return hashlib.sha1(memoryview(frame).tobytes()).hexdigest()


def _quiet_nurec_loggers() -> None:
    for name in (
        "",
        "asyncio",
        "nurec_integration",
        "nurec_render_service",
        "scenario",
        "track",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _quiet_nurec_loggers()

    env = load_environment(
        scenario=args.scenario,
        host=args.host,
        port=args.port,
        sandbox={"mode": "disabled"},
        enable_nurec=True,
        nurec_scene_path=str(Path(args.scene).expanduser().resolve()),
        nurec_camera_logical_id=str(args.camera_logical_id),
        nurec_mode="replay",
        nurec_resolution_ratio=float(args.resolution_ratio),
        nurec_framerate=float(args.framerate),
        nurec={
            "grpc_port": int(args.grpc_port),
            "startup_timeout_s": float(args.startup_timeout),
        },
        record_video=bool(args.record_video),
        video_output_dir=str(output_dir),
    )

    state: dict[str, Any] = {}
    result: dict[str, Any] = {}
    try:
        await env.setup_state(state)
        runtime = state["carla"]
        camera_sensor = runtime.camera_sensor
        if camera_sensor is None:
            raise RuntimeError("NuRec smoke check expected an RGB camera sensor")
        nurec_scenario = env._nurec_mgr.nurec_scenario  # type: ignore[attr-defined]

        sim_times: list[int] = []
        frame_ids: list[int] = []
        frame_hashes: list[str] = []
        for _ in range(int(args.steps)):
            runtime.tick(int(args.ticks_per_sample))
            frame = camera_sensor.latest_frame
            if frame is None:
                raise RuntimeError("NuRec smoke check failed: replay produced no RGB frame")
            sim_times.append(int(nurec_scenario.get_sim_time()))
            frame_ids.append(int(camera_sensor.latest_frame_id))
            frame_hashes.append(_frame_hash(frame))

        result = {
            "scenario": str(args.scenario),
            "steps": int(args.steps),
            "frame_ids": frame_ids,
            "sim_times_us": sim_times,
            "unique_frame_hashes": len(set(frame_hashes)),
            "recording_stats": getattr(camera_sensor, "recording_stats", None),
        }

        if not _strictly_increasing(sim_times):
            raise RuntimeError(
                "NuRec replay sim_time did not strictly increase; wrapper timing likely drifted from SDK requirements"
            )
        if not _strictly_increasing(frame_ids):
            raise RuntimeError("NuRec camera frame ids did not strictly increase")
        if len(set(frame_hashes)) < int(args.min_unique_hashes):
            raise RuntimeError(
                "NuRec replay frames were not changing enough; possible frozen replay or cached-frame regression"
            )

        if bool(args.record_video):
            camera_sensor.stop_recording()
            staged_dir = getattr(camera_sensor, "_temp_dir", None)
            staged_count = 0
            if staged_dir is not None and Path(staged_dir).exists():
                staged_count = len(list(Path(staged_dir).glob("frame_*.jpg")))
            result["staged_recorded_frames"] = staged_count
            result["recording_stats"] = getattr(camera_sensor, "recording_stats", None)
            expected = int(camera_sensor.frame_count)
            if staged_count != expected:
                raise RuntimeError(
                    f"NuRec recorder staged {staged_count} frame files, expected {expected}; frame continuity regressed"
                )

    finally:
        await env.cleanup(state)

    if bool(args.record_video):
        video_path = state.get("video_path")
        if not video_path:
            raise RuntimeError(
                "NuRec recorder did not produce a saved video artifact during cleanup"
            )
        video_file = Path(video_path).resolve()
        if not video_file.exists():
            raise RuntimeError(f"NuRec recorder reported missing video path: {video_file}")
        video_bytes = int(video_file.stat().st_size)
        result["video_path"] = str(video_file)
        result["video_bytes"] = video_bytes
        if video_bytes < 1024:
            raise RuntimeError(
                f"NuRec recorder saved an implausibly small video artifact ({video_bytes} bytes): {video_file}"
            )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check the wrapped NuRec replay path.")
    parser.add_argument("--scene", required=True, help="Path to the NuRec USDZ scene file")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--scenario", default="free_roam", help="Repo scenario name to load")
    parser.add_argument(
        "--camera-logical-id",
        default=DEFAULT_NUREC_CAMERA_LOGICAL_ID,
        help="Preferred NuRec reconstruction camera logical ID (for example camera_front_wide_120fov)",
    )
    parser.add_argument("--framerate", type=float, default=20.0, help="NuRec framerate")
    parser.add_argument("--grpc-port", type=int, default=46435, help="NuRec gRPC port")
    parser.add_argument(
        "--resolution-ratio", type=float, default=0.25, help="NuRec resolution ratio"
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=240.0,
        help="NuRec backend startup timeout in seconds",
    )
    parser.add_argument("--steps", type=int, default=12, help="Replay ticks to sample")
    parser.add_argument(
        "--ticks-per-sample",
        type=int,
        default=2,
        help="World ticks per sampled replay frame; 2 matches NuRec's 1 / (2 * fps) CARLA timing",
    )
    parser.add_argument(
        "--min-unique-hashes",
        type=int,
        default=4,
        help="Minimum distinct frame hashes expected across the sampled replay",
    )
    parser.add_argument(
        "--record-video", action="store_true", help="Also verify recorder continuity"
    )
    parser.add_argument(
        "--output-dir",
        default="_out/nurec_smoke",
        help="Output directory for any recorder artifacts",
    )
    args = parser.parse_args()

    result = asyncio.run(_run(args))
    print(
        json.dumps(result, indent=2, default=lambda value: getattr(value, "__dict__", str(value)))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

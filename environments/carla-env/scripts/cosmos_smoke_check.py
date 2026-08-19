#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from carla_env import load_environment
from carla_env.cosmos import CosmosClient, CosmosConfig


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _hash_array(array: np.ndarray) -> str:
    return _hash_bytes(memoryview(array).tobytes())


def _decode_rgb_jpeg(image_b64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def _test_pattern(width: int = 320, height: int = 192) -> str:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    rgb = np.stack(
        (xv, yv, ((xv.astype(np.uint16) + yv.astype(np.uint16)) // 2).astype(np.uint8)), axis=-1
    )
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cosmos_cfg = CosmosConfig(
        enabled=True,
        server_url=str(args.server_url),
        prompt=str(args.prompt),
        timeout=float(args.timeout),
    )
    client = CosmosClient(cosmos_cfg)
    if not client.health():
        raise RuntimeError(f"Cosmos server is not healthy at {args.server_url}")

    synthetic_input = _test_pattern()
    synthetic_output = client.stylize(synthetic_input)
    synthetic_output_rgb = _decode_rgb_jpeg(synthetic_output)
    synthetic_result = {
        "synthetic_output_shape": list(synthetic_output_rgb.shape),
        "synthetic_output_hash": _hash_array(synthetic_output_rgb),
    }

    env = load_environment(
        scenario=str(args.scenario),
        carla_version=str(args.carla_version),
        sandbox={"mode": "disabled"},
        host=str(args.host),
        port=int(args.port),
        enable_cosmos=True,
        cosmos_server_url=str(args.server_url),
        cosmos_prompt=str(args.prompt),
        record_video=bool(args.record_video),
        video_output_dir=str(output_dir),
    )

    state: dict[str, Any] = {}
    result: dict[str, Any] = {"synthetic": synthetic_result}
    try:
        await env.setup_state(state)
        runtime = state["carla"]
        camera_sensor = runtime.camera_sensor
        depth_sensor = runtime.depth_sensor
        if camera_sensor is None:
            raise RuntimeError("Cosmos smoke check expected an RGB camera sensor")
        if depth_sensor is None:
            raise RuntimeError("Cosmos smoke check expected a depth sensor")

        capture_b64 = ""
        for _ in range(int(args.setup_ticks)):
            capture_b64 = camera_sensor.capture()
            if capture_b64:
                break
            runtime.tick(1)
        if not capture_b64:
            raise RuntimeError("Cosmos smoke check failed: RGB capture remained empty after setup")

        depth_b64 = depth_sensor.capture()
        if not depth_b64:
            raise RuntimeError("Cosmos smoke check failed: depth capture was empty")

        stylized_rgb = _decode_rgb_jpeg(capture_b64)
        inner_frame = getattr(camera_sensor, "_inner", None)
        raw_rgb = getattr(inner_frame, "latest_frame", None) if inner_frame is not None else None
        if raw_rgb is None:
            raise RuntimeError("Cosmos smoke check could not inspect the raw CARLA RGB frame")

        raw_hash = _hash_array(raw_rgb)
        stylized_hash = _hash_array(stylized_rgb)
        if raw_hash == stylized_hash:
            raise RuntimeError("Cosmos stylized RGB matched the raw CARLA RGB frame byte-for-byte")

        result.update(
            {
                "scenario": str(args.scenario),
                "raw_rgb_hash": raw_hash,
                "stylized_rgb_hash": stylized_hash,
                "stylized_rgb_shape": list(stylized_rgb.shape),
                "depth_bytes": len(base64.b64decode(depth_b64)),
                "camera_bytes": len(base64.b64decode(capture_b64)),
            }
        )
    finally:
        await env.cleanup(state)

    if bool(args.record_video):
        video_path = state.get("video_path")
        if not video_path:
            raise RuntimeError(
                "Cosmos recorder did not produce a saved video artifact during cleanup"
            )
        video_file = Path(video_path).resolve()
        if not video_file.exists():
            raise RuntimeError(f"Cosmos recorder reported missing video path: {video_file}")
        video_bytes = int(video_file.stat().st_size)
        if video_bytes < 1024:
            raise RuntimeError(
                f"Cosmos recorder saved an implausibly small video artifact ({video_bytes} bytes): {video_file}"
            )
        result["video_path"] = str(video_file)
        result["video_bytes"] = video_bytes

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check the wrapped Cosmos Transfer2.5 path.")
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8080", help="Cosmos frame server URL"
    )
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "--carla-version", default="0.9.16", help="CARLA version string for the repo env"
    )
    parser.add_argument(
        "--scenario", default="navigation_vision_Town05_v1_p0", help="Repo scenario name"
    )
    parser.add_argument(
        "--setup-ticks",
        type=int,
        default=4,
        help="Extra ticks to wait for the first stylized frame",
    )
    parser.add_argument(
        "--prompt",
        default="Dashcam view of a realistic city street with natural lighting, photorealistic, high detail",
        help="Cosmos prompt",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Cosmos request timeout")
    parser.add_argument(
        "--record-video", action="store_true", help="Also verify saved episode video output"
    )
    parser.add_argument("--output-dir", default="_out/cosmos_smoke", help="Video output directory")
    args = parser.parse_args()

    result = asyncio.run(_run(args))
    print(
        json.dumps(result, indent=2, default=lambda value: getattr(value, "__dict__", str(value)))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

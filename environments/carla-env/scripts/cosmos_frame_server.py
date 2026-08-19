#!/usr/bin/env python3
"""
Cosmos Transfer2.5 single-frame stylization server.

This is an operational helper for running a long-lived Cosmos model on a GPU
host and serving per-frame stylization requests to ``CosmosCameraSensor`` over
HTTP.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Cosmos Transfer2.5 Frame Server")


class StylizeRequest(BaseModel):
    image: str
    prompt: str | None = None
    seed: int = 42
    control: str = "edge"
    control_weight: float = 0.8
    num_steps: int | None = None
    guidance: float | None = None


class StylizeResponse(BaseModel):
    image: str
    width: int
    height: int
    elapsed_ms: int


class CosmosModel:
    def __init__(self) -> None:
        self.pipeline = None
        self.cached_prompt: str | None = None
        self.is_distilled = False
        self._temp_dir = tempfile.mkdtemp(prefix="cosmos_server_")
        self._infer_lock = threading.Lock()

    def load(self, prompt: str, *, distilled: bool = True) -> None:
        os.environ["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "1"

        from cosmos_oss.init import init_environment
        from cosmos_transfer2.config import SetupArguments
        from cosmos_transfer2.inference import Control2WorldInference

        init_environment()
        model_name = "edge/distilled" if distilled else "edge"
        self.is_distilled = distilled

        output_dir = Path(self._temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        setup_args = SetupArguments(
            model=model_name,
            output_dir=output_dir,
            disable_guardrails=True,
        )
        inference = Control2WorldInference(setup_args, batch_hint_keys=["edge"])
        self.pipeline = inference.inference_pipeline
        self.cached_prompt = prompt

    def stylize(
        self,
        rgb: np.ndarray,
        *,
        seed: int = 42,
        control: str = "edge",
        control_weight: float = 0.8,
        num_steps: int | None = None,
        guidance: float | None = None,
        prompt_override: str | None = None,
    ) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded")
        if str(control).strip().lower() != "edge":
            raise ValueError(
                f"Unsupported Cosmos control mode {control!r}. "
                "The bundled frame server currently supports only 'edge'."
            )

        prompt = prompt_override or self.cached_prompt
        if not prompt:
            raise RuntimeError("No prompt available")

        if num_steps is None:
            num_steps = 4 if self.is_distilled else 35
        if guidance is None:
            guidance = None if self.is_distilled else 3.0
        negative_prompt = (
            None
            if self.is_distilled
            else "The video captures a game screen, low quality, worst quality"
        )

        with tempfile.NamedTemporaryFile(
            dir=self._temp_dir,
            prefix="input_frame_",
            suffix=".png",
            delete=False,
        ) as tmp:
            input_path = tmp.name
        try:
            Image.fromarray(rgb).save(input_path)
            with self._infer_lock:
                output_video, *_rest = self.pipeline.generate_img2world(
                    prompt=prompt,
                    video_path=input_path,
                    guidance=guidance,
                    seed=seed,
                    resolution="720",
                    num_conditional_frames=0,
                    num_video_frames_per_chunk=1,
                    num_steps=num_steps,
                    control_weight=str(control_weight),
                    hint_key=[control],
                    max_frames=1,
                    negative_prompt=negative_prompt,
                    input_control_video_paths={},
                )
        finally:
            try:
                os.remove(input_path)
            except FileNotFoundError:
                pass

        output_float = (1.0 + output_video[0]) / 2.0
        output_uint8 = (output_float * 255).clamp(0, 255).to(torch.uint8)
        return output_uint8[:, 0, :, :].permute(1, 2, 0).cpu().numpy()


model = CosmosModel()


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": model.pipeline is not None,
        "prompt": model.cached_prompt,
    }


@app.post("/stylize", response_model=StylizeResponse)
async def stylize(request: StylizeRequest) -> StylizeResponse:
    if str(request.control).strip().lower() != "edge":
        raise HTTPException(
            status_code=400,
            detail="Unsupported control mode. The bundled Cosmos frame server supports only 'edge'.",
        )
    if model.pipeline is None:
        raise HTTPException(status_code=503, detail="Cosmos model is not loaded")

    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid input image: {exc}") from exc

    t0 = time.time()
    try:
        output = model.stylize(
            rgb,
            seed=request.seed,
            control=request.control,
            control_weight=request.control_weight,
            num_steps=request.num_steps,
            guidance=request.guidance,
            prompt_override=request.prompt,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stylization failed: {exc}") from exc

    buf = io.BytesIO()
    Image.fromarray(output).save(buf, format="JPEG", quality=90)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return StylizeResponse(
        image=encoded,
        width=int(output.shape[1]),
        height=int(output.shape[0]),
        elapsed_ms=int((time.time() - t0) * 1000),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--prompt",
        default="Dashcam view of a realistic city street with natural lighting, photorealistic, high detail",
    )
    parser.add_argument(
        "--standard", action="store_true", help="Use the non-distilled model variant"
    )
    args = parser.parse_args()

    model.load(args.prompt, distilled=not args.standard)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

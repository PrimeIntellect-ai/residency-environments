"""Image encoding for WorldBench."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

MAX_BASE64_BYTES = 10_000_000
MAX_DIMENSION = 8000
SCALES = (1.0, 0.75, 0.5, 0.25, 0.125)
PASSTHROUGH_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
PIL_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
LOSSY_QUALITIES = (90, 75, 60, 45, 30)


def bounded_image_data_url(image_bytes: bytes) -> str:
    """Preserve supported images and normalize other formats to bounded PNG."""
    with Image.open(BytesIO(image_bytes)) as source:
        media_type = Image.MIME[source.format]
        encoded = base64.b64encode(image_bytes).decode("ascii")
        if (
            media_type in PASSTHROUGH_MEDIA_TYPES
            and max(source.size) <= MAX_DIMENSION
            and len(encoded) <= MAX_BASE64_BYTES
        ):
            return f"data:{media_type};base64,{encoded}"
        target_media_type = media_type if media_type in PASSTHROUGH_MEDIA_TYPES else "image/png"
        image = (
            source.copy() if target_media_type == "image/png" and media_type == "image/png" else source.convert("RGB")
        )

    width, height = image.size
    largest_dimension = max(width, height)
    if largest_dimension > MAX_DIMENSION:
        scale = MAX_DIMENSION / largest_dimension
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    for scale in SCALES:
        candidate = image
        if scale < 1.0:
            width, height = image.size
            candidate = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
        qualities: tuple[int | None, ...] = (
            LOSSY_QUALITIES if target_media_type in {"image/jpeg", "image/webp"} else (None,)
        )
        for quality in qualities:
            output = BytesIO()
            save_kwargs = {} if quality is None else {"quality": quality}
            candidate.save(
                output,
                format=PIL_FORMATS[target_media_type],
                **save_kwargs,
            )
            encoded = base64.b64encode(output.getvalue())
            if len(encoded) <= MAX_BASE64_BYTES:
                return f"data:{target_media_type};base64,{encoded.decode('ascii')}"

    raise ValueError(f"image cannot be encoded under the {MAX_BASE64_BYTES}-byte limit")

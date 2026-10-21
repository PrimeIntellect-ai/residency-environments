# /// script
# requires-python = ">=3.12"
# dependencies = ["aiohttp>=3.9", "pillow>=10"]
# ///
"""Relay model requests to Prime Inference with every non-JPEG image re-encoded as JPEG.

    uv run scripts/midog-mitosis/jpeg_proxy.py [--port 8899] [--quality 95] [--log relay.jsonl]
    uv run --no-sync eval @ configs/midog-mitosis/pi.toml ... --client.base-url http://127.0.0.1:8899/api/v1

Pi keeps every image the agent reads in its context and resends the whole history on each turn, so an agent that reads
many PNG crops sends requests whose images Prime's endpoint can reject ("Downloaded image content cannot exceed 30MB";
the cutoff measured about 23.6 MB of decoded images). The relay re-encodes each non-JPEG image of a Chat Completions
request (the format Pi sends) as JPEG at the given quality, which cuts a request's image bytes to a quarter to a third at 95.
Encoding is deterministic, so an image becomes the same bytes on every turn and prompt caching keeps working; an LRU
cache of converted images, bounded at 2 GiB, avoids encoding one twice. Responses stream back unchanged, except that a
request the endpoint fails inside its event stream is retried first (see ``RETRY_PAUSES``). With ``--log`` each request
appends a JSON line: its image count, base64 bytes before and after, the upstream status and any retries.
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import time
from collections import OrderedDict

from aiohttp import ClientResponse, ClientSession, ClientTimeout, web
from PIL import Image

CACHE_BYTES = 2 * 2**30
# Prime's endpoint can fail a request inside an HTTP 200 event stream whose only event is {"choices": [], "error": ...}
# (seen as code 504, "Request timed out. Please retry."). Pi reads that as an empty reply rather than a provider error,
# so the relay retries such a request itself, after these pauses in seconds, before Pi sees anything.
RETRY_PAUSES = (5, 15, 30, 60, 60, 60)


class JpegCache:
    """Images converted to JPEG data URLs, keyed by a hash of the original and evicted least recently used first."""

    def __init__(self, quality: int) -> None:
        self.quality = quality
        self.entries: OrderedDict[str, str] = OrderedDict()
        self.size = 0

    def convert(self, url: str) -> str:
        if url.startswith("data:image/jpeg") or not url.startswith("data:image/"):
            return url
        key = hashlib.sha1(url.encode()).hexdigest()
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        pixels = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
        buffer = io.BytesIO()
        pixels.save(buffer, "JPEG", quality=self.quality)
        converted = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        self.entries[key] = converted
        self.size += len(converted)
        while self.size > CACHE_BYTES:
            self.size -= len(self.entries.popitem(last=False)[1])
        return converted

    def compress(self, body: dict) -> tuple[int, int, int]:
        """Convert every image of a Chat Completions body in place; returns (images, base64 bytes before, after)."""
        count = before = after = 0
        for message in body.get("messages") or []:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                url = (part.get("image_url") or {}).get("url") if isinstance(part, dict) else None
                if not url:
                    continue
                new = self.convert(url)
                part["image_url"]["url"] = new
                count, before, after = count + 1, before + len(url), after + len(new)
        return count, before, after


SETTINGS = web.AppKey("settings", argparse.Namespace)
CACHE = web.AppKey("cache", JpegCache)
SESSION = web.AppKey("session", ClientSession)


def has_first_event(prefix: bytes) -> bool:
    start = prefix.find(b"data:")
    return start >= 0 and (b"\n\n" in prefix[start:] or b"\r\n\r\n" in prefix[start:])


def first_event_error(prefix: bytes) -> object | None:
    """The error of an event stream whose first data event is an error without choices, else None."""
    for line in prefix.decode(errors="replace").splitlines():
        if line.startswith("data:"):
            try:
                event = json.loads(line[5:])
            except json.JSONDecodeError:
                return None
            return event.get("error") if event.get("error") and not event.get("choices") else None
    return None


async def first_part(request: web.Request, url: str, raw: bytes, headers: dict) -> tuple[ClientResponse, bytes]:
    """The upstream response and, for an event stream, its bytes up to the end of the first data event."""
    upstream = await request.app[SESSION].request(request.method, url, data=raw, headers=headers)
    prefix = b""
    if upstream.status == 200 and upstream.content_type == "text/event-stream":
        while not has_first_event(prefix) and (chunk := await upstream.content.readany()):
            prefix += chunk
    return upstream, prefix


async def open_upstream(request: web.Request, raw: bytes, record: dict) -> tuple[ClientResponse, bytes]:
    """``first_part`` of the upstream response, retried while the endpoint fails the request inside its stream."""
    skipped = ("host", "content-length", "accept-encoding")
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skipped}
    url = request.app[SETTINGS].upstream + request.path_qs
    for pause in RETRY_PAUSES:
        upstream, prefix = await first_part(request, url, raw, headers)
        error = first_event_error(prefix)
        if error is None:
            return upstream, prefix
        record.setdefault("retried", []).append(str(error.get("message") if isinstance(error, dict) else error))
        upstream.release()
        await asyncio.sleep(pause)
    return await first_part(request, url, raw, headers)


async def relay(request: web.Request) -> web.StreamResponse:
    settings = request.app[SETTINGS]
    raw = await request.read()
    record = {"time": time.time(), "path": request.path}
    if request.method == "POST" and raw:
        body = json.loads(raw)
        count, before, after = request.app[CACHE].compress(body)
        record.update(model=body.get("model"), images=count, b64_before=before, b64_after=after)
        raw = json.dumps(body).encode()
    upstream, prefix = await open_upstream(request, raw, record)
    try:
        response = web.StreamResponse(status=upstream.status)
        response.content_type = upstream.content_type
        await response.prepare(request)
        if prefix:
            await response.write(prefix)
        async for chunk in upstream.content.iter_any():
            await response.write(chunk)
        await response.write_eof()
        record["status"] = upstream.status
    finally:
        upstream.release()
    if settings.log:
        with open(settings.log, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    return response


async def start(app: web.Application) -> None:
    app[SESSION] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=60, sock_read=1800))


async def stop(app: web.Application) -> None:
    await app[SESSION].close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (the reported evaluation used 95).")
    parser.add_argument("--upstream", default="https://api.pinference.ai", help="Origin the requests are relayed to.")
    parser.add_argument("--log", help="Append one JSON line per request to this file.")
    args = parser.parse_args()
    app = web.Application(client_max_size=512 * 2**20)
    app[SETTINGS] = args
    app[CACHE] = JpegCache(args.quality)
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    app.router.add_route("*", "/{tail:.*}", relay)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

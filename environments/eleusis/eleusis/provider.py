"""Episode-scoped recovery for provider rate limits inside successful SSE responses."""

import asyncio
import json
import logging
import random
from contextlib import asynccontextmanager

from verifiers.v1.clients.client import Client, RelayReply
from verifiers.v1.clients.eval import EvalClient
from verifiers.v1.errors import ProviderError
from verifiers.v1.interception.base import Interception

LOG = logging.getLogger("eleusis.provider")
ATTEMPTS = 6


def check_event(event: bytes) -> None:
    """Recognize explicit provider errors; never infer success from an EOF."""
    data = b"\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith(b"data:"))
    if not data or data == b"[DONE]":
        return
    try:
        payload = json.loads(data)
    except ValueError:
        return  # The native parser remains responsible for malformed responses.
    error = payload.get("error") if isinstance(payload, dict) else None
    if error is None:
        return
    code = error.get("code", error.get("status", 502)) if isinstance(error, dict) else 502
    status = int(code) if str(code).isdigit() else 502
    if not 400 <= status <= 599:
        status = 502
    raise ProviderError(f"Upstream SSE error: {error}", status_code=status)


class ResilientEvalClient(Client):
    """Borrow the native client; retry without changing its ownership or requests."""

    def __init__(self, upstream: EvalClient):
        self.upstream = upstream

    async def get_response(self, *args, **kwargs):
        return await self.upstream.get_response(*args, **kwargs)

    async def relay_aux(self, *args, **kwargs):
        return await self.upstream.relay_aux(*args, **kwargs)

    async def relay(self, dialect, body, session_id=None, headers=None):
        # Native interception also buffers each completion before committing it.
        # Retry before returning ANY events so failed partial calls execute no tools.
        for attempt in range(ATTEMPTS):
            reply = None
            try:
                reply = await self.upstream.relay(dialect, body, session_id, headers)
                events = []
                async for event in reply.chunks:
                    check_event(event)
                    events.append(event)
                content_type = reply.content_type
                break
            except ProviderError as exc:
                if exc.status_code != 429 or attempt == ATTEMPTS - 1:
                    raise
                delay = min(32, 2 ** (attempt + 1)) + random.uniform(0, 1)
                LOG.warning("Provider 429: retry %s/%s in %.1fs", attempt + 1, ATTEMPTS - 1, delay)
            finally:
                if reply is not None:
                    await reply.close()
            await asyncio.sleep(delay)

        async def chunks():
            for event in events:
                yield event

        async def close():
            pass  # Upstream connection was closed before the buffered reply was returned.

        return RelayReply(content_type=content_type, chunks=chunks(), close=close)


class ResilientInterception(Interception):
    """Borrow native slots and wrap only this episode's eval transport."""

    def __init__(self, upstream: Interception):
        super().__init__()
        self.upstream = upstream

    async def start(self):
        pass  # The environment owns and starts the borrowed interception resource.

    @asynccontextmanager
    async def acquire(self, session):
        async with self.upstream.acquire(session) as slot:
            upstream_client = session.client
            if isinstance(upstream_client, EvalClient):
                session.client = ResilientEvalClient(upstream_client)
            try:
                yield slot
            finally:
                session.client = upstream_client

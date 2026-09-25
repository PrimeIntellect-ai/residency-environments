"""Append-only canonical event log with deterministic JSON serialization."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import BinaryIO

from alphaverse.models import Event, EventKind, JsonValue, MarketTime


class EventLog:
    """Own canonical event sequencing for one exchange episode.

    Canonical events are append-only JSON Lines on a local temporary file.  A
    small in-memory tail keeps ordinary matching-engine ``from_index`` calls
    cheap, while sparse byte offsets allow older history to be replayed
    without retaining millions of Python event objects.
    """

    _CHECKPOINT_STRIDE = 1_024
    _DEFAULT_RETAINED_EVENTS = 2_048

    def __init__(
        self,
        events: Iterable[Event] = (),
        *,
        retained_events: int = _DEFAULT_RETAINED_EVENTS,
    ) -> None:
        if isinstance(retained_events, bool) or not isinstance(retained_events, int):
            raise TypeError("retained_events must be an int")
        if retained_events < 0:
            raise ValueError("retained_events must be non-negative")
        descriptor, name = tempfile.mkstemp(prefix="alphaverse-events-", suffix=".jsonl")
        self._path = Path(name)
        self._stream = os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            buffering=64 * 1024,
        )
        self._event_count = 0
        self._last_sequence = 0
        self._retained_events = retained_events
        self._recent: deque[Event] = deque(maxlen=retained_events or None)
        self._recent_start = 0
        self._checkpoints: list[int] = [0]
        self._closed = False
        for event in events:
            self._append_existing(event)

    def __len__(self) -> int:
        return self._event_count

    def __iter__(self) -> Iterator[Event]:
        return self._iter_from(0)

    @property
    def last_sequence(self) -> int:
        return self._last_sequence

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self)

    def from_index(self, index: int) -> tuple[Event, ...]:
        """Return the suffix beginning at a previously recorded log length."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an int")
        if index < 0:
            raise ValueError("index must be non-negative")
        return tuple(self._iter_from(index))

    def append(
        self,
        *,
        market_time: MarketTime,
        match_event_id: str,
        kind: EventKind,
        product_id: str,
        data: Mapping[str, JsonValue],
    ) -> Event:
        """Append an event using the next canonical sequence number."""

        event = Event(
            sequence=self.last_sequence + 1,
            market_time=market_time,
            match_event_id=match_event_id,
            kind=kind,
            product_id=product_id,
            data=data,
        )
        # Validate JSON compatibility at the boundary rather than allowing an
        # unserializable event to poison replay later.
        self._append_encoded(event, self._encode(event))
        return event

    def to_jsonl(self) -> bytes:
        """Serialize events as stable UTF-8 JSON Lines bytes."""

        if not self._event_count:
            return b""
        self._flush()
        return self._path.read_bytes()

    def copy_jsonl(self, destination: BinaryIO) -> int:
        """Stream canonical JSON Lines into a binary destination.

        Artifact creation must not materialize a long market's complete event
        history in memory.  The event log is already disk-backed, so copying it
        directly also keeps recording cost independent of retained tail size.
        The returned byte count describes the uncompressed canonical stream.
        """

        if self._closed:
            raise RuntimeError("event log is closed")
        self._flush()
        with self._path.open("rb") as source:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        return self._path.stat().st_size

    @classmethod
    def from_jsonl(cls, payload: bytes | str) -> EventLog:
        """Reconstruct a log and validate contiguous event sequencing."""

        text = payload.decode() if isinstance(payload, bytes) else payload
        log = cls()
        try:
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                log._append_existing(log._decode(line, line_number))
            return log
        except Exception:
            log.close()
            raise

    def _append_existing(self, event: Event) -> None:
        expected = self.last_sequence + 1
        if event.sequence != expected:
            raise ValueError(f"event sequence must be contiguous: expected {expected}, received {event.sequence}")
        self._append_encoded(event, self._encode(event))

    def close(self) -> None:
        """Release the backing file."""

        if self._closed:
            return
        self._closed = True
        self._stream.close()
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass

    def __del__(self) -> None:
        # Best-effort cleanup when an owner does not call close() explicitly.
        try:
            self.close()
        except Exception:
            pass

    def _append_encoded(self, event: Event, encoded: str) -> None:
        self._require_open()
        if self._event_count and self._event_count % self._CHECKPOINT_STRIDE == 0:
            self._stream.flush()
            self._checkpoints.append(self._stream.tell())
        self._stream.write(encoded)
        self._stream.write("\n")
        self._event_count += 1
        self._last_sequence = event.sequence
        if self._retained_events:
            self._recent.append(event)
            self._recent_start = self._event_count - len(self._recent)

    def _iter_from(self, index: int) -> Iterator[Event]:
        self._require_open()
        if index >= self._event_count:
            return iter(())
        if self._retained_events and index >= self._recent_start:
            return iter(tuple(self._recent)[index - self._recent_start :])
        self._flush()
        checkpoint_index = index // self._CHECKPOINT_STRIDE
        checkpoint_event = checkpoint_index * self._CHECKPOINT_STRIDE
        offset = self._checkpoints[checkpoint_index]

        def read() -> Iterator[Event]:
            with self._path.open(encoding="utf-8") as stream:
                stream.seek(offset)
                for event_index, line in enumerate(stream, start=checkpoint_event):
                    if event_index < index:
                        continue
                    yield self._decode(line, event_index + 1)

        return read()

    def _flush(self) -> None:
        self._require_open()
        self._stream.flush()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("event log is closed")

    @staticmethod
    def _decode(line: str, line_number: int) -> Event:
        raw = json.loads(line)
        try:
            return Event(
                sequence=raw["sequence"],
                market_time=raw["market_time"],
                match_event_id=raw["match_event_id"],
                kind=EventKind(raw["kind"]),
                product_id=raw["product_id"],
                data=raw["data"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid event on JSONL line {line_number}") from exc

    @staticmethod
    def _encode(event: Event) -> str:
        return json.dumps(
            {
                "sequence": event.sequence,
                "market_time": event.market_time,
                "match_event_id": event.match_event_id,
                "kind": event.kind.value,
                "product_id": event.product_id,
                "data": dict(event.data),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

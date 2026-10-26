"""Trusted scene changes driven by simulator timestamps, with replayable evidence.

The worker runs beside the trusted controller, never in the policy process.  Its
clock advances only when the simulator timestamp advances.  The controller must
pause physics between policy turns and synchronize the worker before taking a
frame.  Pure scheduling code is embedded unchanged in the standalone worker.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import signal
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

EVENT_CONFIG_PATH = "/tmp/flight_events_config.json"
EVENT_LOG_PATH = "/tmp/flight_events_log.json"
EVENT_READY_PATH = "/tmp/flight_events_ready.json"
EVENT_HEARTBEAT_PATH = "/tmp/flight_events_heartbeat.json"
EVENT_ERROR_PATH = "/tmp/flight_events_error.json"
EVENT_PID_PATH = "/tmp/flight_events.pid"
MAX_EVENT_BYTES = 128_000
MAX_EVENTS = 32
EVENT_TIMING_TOLERANCE_S = 0.25
EVENT_POLL_S = 0.02
EVENT_METRICS = ("events_complete", "events_timing", "events_triggered")


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + " must be a finite number")
    return float(value)


def _sim_ns(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("simulator timestamp must be a positive integer in nanoseconds")
    return value


def normalize_schedule(raw):
    """Validate the JSON contract identically on the host and in the worker."""
    if not isinstance(raw, dict) or set(raw) != {"trigger_x", "events"}:
        raise ValueError("schedule needs only trigger_x and events")
    trigger = _number(raw["trigger_x"], "trigger_x")
    if not -30 <= trigger <= 40:
        raise ValueError("trigger_x is outside the park")
    events = raw["events"]
    if not isinstance(events, (list, tuple)) or not 1 <= len(events) <= MAX_EVENTS:
        raise ValueError("schedule needs between 1 and 32 events")
    clean, identities, live_state = [], set(), {}
    previous = -1.0
    for event in events:
        if (
            not isinstance(event, dict)
            or not {"at_s", "op", "name"} <= set(event)
            or set(event) - {"at_s", "op", "name", "object"}
        ):
            raise ValueError("event has missing or unknown fields")
        at = _number(event["at_s"], "event time")
        if not 0 <= at <= 1200 or at < previous:
            raise ValueError("event times must be ordered and between 0 and 1200 seconds")
        previous = at
        name, op = event["name"], event["op"]
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) is None:
            raise ValueError("event object name must be a safe identifier")
        if op not in ("spawn", "destroy"):
            raise ValueError("event op must be spawn or destroy")
        if (at, name) in identities:
            raise ValueError("duplicate event time and object name")
        identities.add((at, name))
        if live_state.get(name) == op:
            raise ValueError("repeated object operation without its inverse")
        live_state[name] = op
        obj = event.get("object")
        if op == "destroy":
            if obj is not None:
                raise ValueError("destroy event must not contain an object")
        else:
            fields = {"asset", "x", "y", "sx", "sy", "sz"}
            if not isinstance(obj, dict) or not fields <= set(obj) or set(obj) - fields - {"colour"}:
                raise ValueError("spawn object has missing or unknown fields")
            if obj["asset"] not in ("Cube", "Cylinder"):
                raise ValueError("spawn asset must be Cube or Cylinder")
            obj = dict(obj)
            for axis, low, high in (("x", -30, 40), ("y", -20, 20)):
                obj[axis] = _number(obj[axis], axis)
                if not low <= obj[axis] <= high:
                    raise ValueError("spawn position is outside the park")
            for axis in ("sx", "sy", "sz"):
                obj[axis] = _number(obj[axis], axis)
                if not 0 < obj[axis] <= 30:
                    raise ValueError("spawn scale must be positive and at most 30 metres")
            if "colour" in obj:
                colour = obj["colour"]
                if (
                    not isinstance(colour, (list, tuple))
                    or len(colour) != 3
                    or any(type(c) is not int or not 0 <= c <= 255 for c in colour)
                ):
                    raise ValueError("colour must contain three integer RGB channels")
                obj["colour"] = list(colour)
        clean.append({"at_s": at, "op": op, "name": name, "object": obj})
    return {"trigger_x": trigger, "events": clean}


class TimedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    at_s: float
    op: Literal["spawn", "destroy"]
    name: str
    object: dict | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_event(cls, value):
        if isinstance(value, cls):
            return value
        return normalize_schedule({"trigger_x": 0, "events": [value]})["events"][0]


class EventSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    trigger_x: float
    events: tuple[TimedEvent, ...]

    @model_validator(mode="before")
    @classmethod
    def validate_schedule(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            value = dict(value)
            events = value.get("events")
            if isinstance(events, (list, tuple)):
                value["events"] = [e.model_dump() if isinstance(e, TimedEvent) else e for e in events]
        return normalize_schedule(value)


class EventScheduler:
    """Deterministic state machine; apply_event returns the post-RPC sim timestamp."""

    def __init__(self, schedule, rest_z, apply_event):
        self.schedule = normalize_schedule(schedule)
        self.rest_z = _number(rest_z, "rest_z")
        self.apply_event = apply_event
        self.log = {
            "version": 1,
            "schedule": self.schedule,
            "rest_z": self.rest_z,
            "started_sim_ns": None,
            "epoch_sim_ns": None,
            "epoch_x": None,
            "epoch_z": None,
            "last_sim_ns": None,
            "status": "running",
            "events": [],
            "error": None,
        }

    def fail(self, error):
        self.log["status"] = "error"
        self.log["error"] = str(error)[:2000]

    def tick(self, sim_ns, x, z):
        if self.log["status"] in ("error", "stopped"):
            raise RuntimeError("event scheduler is no longer running")
        try:
            sim_ns = _sim_ns(sim_ns)
            x, z = _number(x, "x"), _number(z, "z")
            previous = self.log["last_sim_ns"]
            if previous is not None and sim_ns < previous:
                raise RuntimeError("simulator clock moved backwards")
            if self.log["started_sim_ns"] is None:
                self.log["started_sim_ns"] = sim_ns
            self.log["last_sim_ns"] = sim_ns
            if self.log["epoch_sim_ns"] is None:
                if x < self.schedule["trigger_x"] or z >= self.rest_z - 0.5:
                    return
                self.log.update(epoch_sim_ns=sim_ns, epoch_x=x, epoch_z=z)
            epoch = self.log["epoch_sim_ns"]
            while len(self.log["events"]) < len(self.schedule["events"]):
                index = len(self.log["events"])
                event = self.schedule["events"][index]
                due = epoch + round(event["at_s"] * 1_000_000_000)
                if sim_ns < due:
                    break
                item = {
                    "index": index,
                    "name": event["name"],
                    "op": event["op"],
                    "at_s": event["at_s"],
                    "scheduled_sim_ns": due,
                    "started_sim_ns": sim_ns,
                    "fired_sim_ns": None,
                    "actual_elapsed_s": None,
                    "status": "error",
                    "error": None,
                }
                self.log["events"].append(item)
                try:
                    if (sim_ns - due) / 1e9 > EVENT_TIMING_TOLERANCE_S:
                        raise RuntimeError("event exceeded simulator-time lateness tolerance")
                    fired_ns = _sim_ns(self.apply_event(event))
                    if fired_ns < sim_ns:
                        raise RuntimeError("simulator clock moved backwards during event")
                    item["fired_sim_ns"] = fired_ns
                    item["actual_elapsed_s"] = (fired_ns - epoch) / 1e9
                    self.log["last_sim_ns"] = sim_ns = fired_ns
                    if (fired_ns - due) / 1e9 > EVENT_TIMING_TOLERANCE_S:
                        raise RuntimeError("event RPC exceeded simulator-time lateness tolerance")
                    item["status"] = "ok"
                except Exception as exc:
                    item["error"] = str(exc)[:2000]
                    raise
            if len(self.log["events"]) == len(self.schedule["events"]):
                self.log["status"] = "complete"
        except Exception as exc:
            self.fail(exc)
            raise


def _atomic_json(path, value):
    encoded = json.dumps(value, allow_nan=False).encode()
    if len(encoded) > MAX_EVENT_BYTES:
        raise ValueError("event evidence exceeded its size limit")
    temporary = path + ".tmp"
    with open(temporary, "wb") as out:
        out.write(encoded)
    os.replace(temporary, path)


def _solid_png(path, rgb):
    import struct
    import zlib

    size = 8
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as out:
        out.write(png)


def apply_sim_event(client, airsim, event, rest_z):
    name = event["name"]
    if event["op"] == "destroy":
        if not client.simDestroyObject(name):
            raise RuntimeError("could not destroy " + name)
    else:
        obj = event["object"]
        pose = airsim.Pose(airsim.Vector3r(obj["x"], obj["y"], rest_z - obj["sz"] / 2), airsim.Quaternionr(0, 0, 0, 1))
        spawned = client.simSpawnObject(
            name, obj["asset"], pose, airsim.Vector3r(obj["sx"], obj["sy"], obj["sz"]), False
        )
        if spawned != name:
            raise RuntimeError("could not spawn exact event object " + name)
        if obj.get("colour") is not None:
            texture = "/tmp/flight_event_" + name + ".png"
            _solid_png(texture, obj["colour"])
            if not client.simSetObjectMaterialFromTexture(name, texture):
                raise RuntimeError("could not colour " + name)
    return int(client.getMultirotorState().timestamp)


def sync_events(client, timeout_s=10.0):
    """Call with physics paused; wait until due events precede the next frame."""
    import json
    import os
    import time

    target = int(client.getMultirotorState().timestamp)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists("/tmp/flight_events_error.json"):
            with open("/tmp/flight_events_error.json") as handle:
                error = json.load(handle)
            raise RuntimeError("timed event worker failed: " + str(error.get("error", error)))
        try:
            with open("/tmp/flight_events_heartbeat.json") as handle:
                heartbeat = json.load(handle)
        except FileNotFoundError:
            heartbeat = None
        if heartbeat is not None and heartbeat.get("error"):
            raise RuntimeError("timed event worker failed: " + str(heartbeat["error"]))
        if heartbeat is not None and type(heartbeat.get("sim_ns")) is int and heartbeat["sim_ns"] >= target:
            # Advancing while paused would make policy latency affect event time.
            if int(client.getMultirotorState().timestamp) != target:
                raise RuntimeError("simulator clock advanced while paused")
            return heartbeat
        time.sleep(0.01)
    raise RuntimeError("timed event worker did not catch up to the paused simulator")


def _worker_main():
    import cosysairsim as airsim

    scheduler = None
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with open(EVENT_CONFIG_PATH, "rb") as handle:
            payload = handle.read(MAX_EVENT_BYTES + 1)
        if len(payload) > MAX_EVENT_BYTES:
            raise ValueError("event configuration exceeded its size limit")
        config = json.loads(payload)
        if not isinstance(config, dict) or set(config) != {"schedule", "rest_z"}:
            raise ValueError("event configuration needs schedule and rest_z")
        client = airsim.MultirotorClient(timeout_value=5)
        client.confirmConnection()
        scheduler = EventScheduler(
            config["schedule"], config["rest_z"], lambda event: apply_sim_event(client, airsim, event, config["rest_z"])
        )
        ready = False
        while running:
            state = client.getMultirotorState()
            position = state.kinematics_estimated.position
            scheduler.tick(int(state.timestamp), position.x_val, position.z_val)
            # Commit the event record before acknowledging its timestamp. A frame
            # therefore cannot observe a changed scene without persistent evidence.
            _atomic_json(EVENT_LOG_PATH, scheduler.log)
            _atomic_json(EVENT_HEARTBEAT_PATH, {"sim_ns": scheduler.log["last_sim_ns"], "error": None})
            if not ready:
                _atomic_json(EVENT_READY_PATH, {"pid": os.getpid(), "started_sim_ns": scheduler.log["started_sim_ns"]})
                ready = True
            time.sleep(EVENT_POLL_S)
        if scheduler.log["status"] == "running":
            scheduler.log["status"] = "stopped"
        _atomic_json(EVENT_LOG_PATH, scheduler.log)
    except Exception as exc:
        if scheduler is not None:
            scheduler.fail(exc)
            _atomic_json(EVENT_LOG_PATH, scheduler.log)
        _atomic_json(EVENT_ERROR_PATH, {"error": str(exc)[:2000]})
        raise


def parse_event_log(text):
    """Reject truncated, oversized or non-object evidence instead of guessing."""
    if not isinstance(text, (str, bytes)) or len(text.encode() if isinstance(text, str) else text) > MAX_EVENT_BYTES:
        return None
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def validate_event_log(schedule, log, recording):
    """Check every scheduled operation, its timing, and its recorded flight epoch.

    ``recording`` rows contain absolute simulator seconds as ``sim_t``.  Event
    observations are independently sampled at 50 Hz and recording at 10 Hz, so
    the trigger and final event may be bracketed within 0.25 simulator seconds.
    """
    result = {key: 0.0 for key in EVENT_METRICS}
    try:
        raw = schedule.model_dump(mode="json") if isinstance(schedule, EventSchedule) else schedule
        expected = normalize_schedule(raw)
        if not isinstance(log, dict) or log.get("version") != 1 or log.get("error"):
            return result
        if normalize_schedule(log.get("schedule")) != expected or log.get("status") != "complete":
            return result
        ground = _number(log["rest_z"], "rest_z")
        start, epoch, last = (_sim_ns(log[field]) for field in ("started_sim_ns", "epoch_sim_ns", "last_sim_ns"))
        if (
            not start <= epoch <= last
            or _number(log["epoch_x"], "epoch_x") < expected["trigger_x"]
            or _number(log["epoch_z"], "epoch_z") >= ground - 0.5
        ):
            return result
        if not isinstance(recording, list) or len(recording) < 2:
            return result
        times = [_number(row["sim_t"], "recording simulator time") for row in recording]
        if any(b < a for a, b in zip(times, times[1:])):
            return result
        epoch_s = epoch / 1e9
        tolerance = EVENT_TIMING_TOLERANCE_S
        if not times[0] - tolerance <= epoch_s <= times[-1]:
            return result
        # Match the first observed trigger, not an arbitrary later visit. The
        # worker may see a crossing one telemetry sample before/after recording.
        triggers = [
            row
            for row in recording
            if _number(row["x"], "x") >= expected["trigger_x"] and _number(row["z"], "z") < ground - 0.5
        ]
        if not triggers or abs(triggers[0]["sim_t"] - epoch_s) > tolerance:
            return result
        result["events_triggered"] = 1.0
        rows = log.get("events")
        if not isinstance(rows, list) or len(rows) != len(expected["events"]):
            return result
        previous = epoch
        for index, (event, item) in enumerate(zip(expected["events"], rows)):
            if (
                not isinstance(item, dict)
                or item.get("index") != index
                or item.get("name") != event["name"]
                or item.get("op") != event["op"]
                or item.get("at_s") != event["at_s"]
                or item.get("status") != "ok"
                or item.get("error")
            ):
                return result
            due = epoch + round(event["at_s"] * 1e9)
            began, fired = _sim_ns(item["started_sim_ns"]), _sim_ns(item["fired_sim_ns"])
            if (
                item.get("scheduled_sim_ns") != due
                or not previous <= began <= fired <= last
                or began < due
                or fired / 1e9 > times[-1]
            ):
                return result
            if (fired - due) / 1e9 > tolerance or abs(
                _number(item["actual_elapsed_s"], "event elapsed") - (fired - epoch) / 1e9
            ) > 1e-6:
                return result
            previous = fired
        result["events_complete"] = result["events_timing"] = 1.0
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    return result


_RUNTIME_NAMES = (
    "EVENT_CONFIG_PATH",
    "EVENT_LOG_PATH",
    "EVENT_READY_PATH",
    "EVENT_HEARTBEAT_PATH",
    "EVENT_ERROR_PATH",
    "MAX_EVENT_BYTES",
    "MAX_EVENTS",
    "EVENT_TIMING_TOLERANCE_S",
    "EVENT_POLL_S",
)
EVENT_SOURCE = (
    "import json, math, os, re, signal, time\n"
    + "\n".join(f"{name} = {globals()[name]!r}" for name in _RUNTIME_NAMES)
    + "\n\n"
)
EVENT_SOURCE += (
    "\n\n".join(
        inspect.getsource(obj)
        for obj in (
            _number,
            _sim_ns,
            normalize_schedule,
            EventScheduler,
            _atomic_json,
            _solid_png,
            apply_sim_event,
            _worker_main,
        )
    )
    + '\n\nif __name__ == "__main__":\n    _worker_main()\n'
)
EVENT_SYNC_SOURCE = inspect.getsource(sync_events)

START_EVENTS = f"""set -e
rm -f {EVENT_LOG_PATH} {EVENT_READY_PATH} {EVENT_HEARTBEAT_PATH} {EVENT_ERROR_PATH}
nohup /usr/bin/python3 -u /tmp/flight_events.py > /tmp/flight_events_worker.log 2>&1 &
event_pid=$!
echo "$event_pid" > {EVENT_PID_PATH}
for event_try in $(seq 1 100); do
  test ! -s {EVENT_ERROR_PATH} || exit 1
  kill -0 "$event_pid" 2>/dev/null || exit 1
  test ! -s {EVENT_READY_PATH} || exit 0
  sleep 0.1
done
kill "$event_pid" 2>/dev/null || true
exit 1
"""

STOP_EVENTS = f"""set -e
test -s {EVENT_PID_PATH} || exit 0
event_pid=$(cat {EVENT_PID_PATH})
case "$event_pid" in ''|*[!0-9]*) exit 1 ;; esac
event_command=$(ps -p "$event_pid" -o args= 2>/dev/null || true)
case "$event_command" in
  *'/tmp/flight_events.py'*) ;;
  *) rm -f {EVENT_PID_PATH}; exit 0 ;;
esac
kill "$event_pid" 2>/dev/null || true
for event_try in $(seq 1 100); do
  if ! kill -0 "$event_pid" 2>/dev/null; then
    rm -f {EVENT_PID_PATH}
    exit 0
  fi
  event_state=$(ps -p "$event_pid" -o stat= 2>/dev/null || true)
  case "$event_state" in *Z*) rm -f {EVENT_PID_PATH}; exit 0 ;; esac
  sleep 0.1
done
kill -9 "$event_pid" 2>/dev/null || true
rm -f {EVENT_PID_PATH}
echo '{{"error":"event worker did not stop cleanly"}}' > {EVENT_ERROR_PATH}
exit 1
"""

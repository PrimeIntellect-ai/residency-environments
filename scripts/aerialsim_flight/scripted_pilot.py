"""Local scripted pilots for controller and grading rehearsals.

The rehearsal harness registers each task privately. Model requests carry the
framework's X-Session-ID, so the honest pilot can use that task's hidden state
without putting it in the conversation. Blind and broken pilots receive no
hidden state. The always_land/always_hold aliases still support the old harness.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NamedTuple

POLICIES = ("honest", "blind_shortcut", "broken", "always_land", "always_hold", "always_b")
ADVANCED_FAMILIES = ("alternate_pad", "patrol_inspection", "go_around", "compound")
ADVANCED_COUNTS = {"alternate_pad": 3, "patrol_inspection": 2, "go_around": 2, "compound": 4}
REGISTERED_POLICIES = ("honest", "blind_shortcut", "broken")


def plan(policy: str, pad: tuple[float, float], *, blocked: bool | None = None) -> list[dict]:
    if policy not in POLICIES:
        raise ValueError(f"unknown scripted policy: {policy}")
    if policy == "honest":
        if not isinstance(blocked, bool):
            raise ValueError("the honest pilot needs registered scene state")
        policy = "always_hold" if blocked else "always_land"
    if policy == "blind_shortcut":
        policy = "always_land"
    if policy == "broken":
        return [{"action": "land"}]
    x, y = pad
    steps = [
        {"action": "takeoff", "height": 5},
        {"action": "goto", "x": x, "y": y, "z": -8, "speed": 3},
        {"action": "look", "pitch": -80, "yaw": 0},
        {"action": "hover", "seconds": 2},
        {"action": "look", "pitch": -60, "yaw": 0},
        {"action": "hover", "seconds": 2},
    ]
    if policy == "always_land":
        steps.append({"action": "land"})
    else:
        # Drop to about a metre below the hold height first: a short climb on abort
        # once failed to report done, and every rehearsal should hit that case.
        steps.append({"action": "goto", "x": x, "y": y, "z": -2.2, "speed": 2})
        steps.append({"action": "abort"})
    return steps


def find_pad_plan(policy: str, start: tuple[float, float], *, pad: tuple[float, float] | None = None) -> list[dict]:
    """Oracle flies to its private target; the blind control always tries the same offset."""
    if policy not in REGISTERED_POLICIES:
        raise ValueError("find_pad supports honest, blind_shortcut and broken only")
    if policy == "broken":
        return [{"action": "land"}]
    if policy == "honest":
        if pad is None:
            raise ValueError("the honest find_pad pilot needs registered pad coordinates")
        x, y = pad
    else:
        x, y = start[0] + 14.0, start[1] + 5.0
    return [
        {"action": "takeoff", "height": 8},
        {"action": "goto", "x": x, "y": y, "z": -8, "speed": 3},
        {"action": "look", "pitch": -80, "yaw": 0},
        {"action": "hover", "seconds": 2},
        {"action": "land"},
    ]


def obstacle_bypass_plan(policy: str, pad: tuple[float, float], *, variant: int | None = None) -> list[dict]:
    """The oracle selects the open side; the blind control always tries positive y."""
    if policy not in REGISTERED_POLICIES:
        raise ValueError("obstacle_bypass supports honest, blind_shortcut and broken only")
    if policy == "broken":
        return [{"action": "land"}]
    if policy == "honest" and (type(variant) is not int or variant not in (0, 1)):
        raise ValueError("the honest obstacle_bypass pilot needs a registered variant")
    side = -4.25 if policy == "honest" and variant == 1 else 4.25
    return [
        {"action": "takeoff", "height": 4},
        {"action": "goto", "x": 6, "y": 0, "z": -2, "speed": 3},
        {"action": "look", "pitch": -10, "yaw": 0},
        {"action": "goto", "x": 8, "y": side, "z": -2, "speed": 3},
        {"action": "goto", "x": 16, "y": side, "z": -2, "speed": 3},
        {"action": "goto", "x": 18, "y": 0, "z": -2, "speed": 3},
        {"action": "goto", "x": pad[0], "y": pad[1], "z": -2, "speed": 3},
        {"action": "look", "pitch": -80, "yaw": 0},
        {"action": "land"},
    ]


def advanced_plan(
    policy: str,
    family: str,
    geometry: dict,
    *,
    variant: int | None = None,
    clear_after: float | None = None,
) -> list[dict]:
    """Fly published geometry; only the honest controller receives oracle facts.

    The fixed-A and fixed-B controls run the same actions across variants. The
    honest temporary-blockage control flies every missed-approach gate before
    waiting long enough for the simulator-time event and returning to the pad.
    """
    from aerialsim_flight.advanced import PublicGeometry

    if family not in ADVANCED_FAMILIES:
        raise ValueError("unknown advanced family")
    if policy not in REGISTERED_POLICIES and not (family == "alternate_pad" and policy == "always_b"):
        raise ValueError("unsupported advanced scripted policy")
    geo = PublicGeometry.model_validate(geometry).model_dump(mode="json")
    if policy == "broken":
        return [{"action": "land"}]
    if policy == "honest" and (type(variant) is not int or variant not in range(ADVANCED_COUNTS[family])):
        raise ValueError("the honest advanced pilot needs a registered scene variant")

    def goto(point):
        return {"action": "goto", "x": point[0], "y": point[1], "z": point[2] if len(point) == 3 else -2, "speed": 3}

    steps = [{"action": "takeoff", "height": 4}, goto(geo["inspection"]), {"action": "look", "pitch": -25, "yaw": 0}]
    if family == "compound":
        steps += [goto(geo["bypass_entry"]), goto(geo["bypass"])]
    if family == "patrol_inspection":
        branch = "negative" if policy == "honest" and variant == 1 else "positive"
        steps.append(goto(geo[f"branch_{branch}"]))
    steps += [goto(geo["merge"]), {"action": "look", "pitch": -30, "yaw": 0}]

    hold = policy == "honest" and (
        (family == "alternate_pad" and variant == 2) or (family == "compound" and variant == 3)
    )
    alternate = policy == "always_b" or (
        policy == "honest"
        and (
            (family == "alternate_pad" and variant == 1)
            or (family == "patrol_inspection" and variant == 1)
            or (family == "compound" and variant == 2)
        )
    )
    if hold:
        return steps + [goto(geo["hold"]), {"action": "abort"}]
    if family in ("go_around", "compound") and not alternate:
        steps += [goto(geo["reentry_stage"]), goto(geo["approach"])]
        if policy == "honest" and variant == 1:
            if (
                isinstance(clear_after, bool)
                or not isinstance(clear_after, (int, float))
                or not math.isfinite(clear_after)
                or not 30 <= clear_after <= 48
            ):
                raise ValueError("the honest missed approach needs its bounded simulator clear time")
            steps += [goto(geo["retreat"]), goto(geo["climb"]), {"action": "look", "pitch": -25, "yaw": 0}]
            # Start counting only after the retreat, so the wait safely exceeds
            # the event's offset from its earlier approach trigger.
            remaining = clear_after + 2.0
            while remaining > 0:
                duration = min(10.0, remaining)
                steps.append({"action": "hover", "seconds": duration})
                remaining -= duration
            steps += [{"action": "look", "pitch": -25, "yaw": 0}, goto(geo["reentry_stage"]), goto(geo["approach"])]
    steps += [
        goto(geo["pad_b"] if alternate else geo["pad_a"]),
        {"action": "look", "pitch": -80, "yaw": 0},
        {"action": "land"},
    ]
    return steps


def coordinates(value, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)
    ):
        raise ValueError(f"registration needs two finite {name} coordinates")
    return tuple(map(float, value))


class Scene(NamedTuple):
    task_key: str
    policy: str
    family: str
    pad: tuple[float, float] | None
    blocked: bool | None
    start: tuple[float, float] | None = None
    heading: float | None = None
    variant: int | None = None
    geometry: dict | None = None
    clear_after: float | None = None


class SessionRegistry:
    """An immutable scene per rollout, registered idempotently before inference."""

    def __init__(self):
        self._scenes: dict[str, Scene] = {}
        self._lock = threading.Lock()

    def register(self, body: dict, expected_policy: str) -> None:
        if not isinstance(body, dict):
            raise ValueError("registration must be an object")
        session_id, task_key = body.get("session_id"), body.get("task_key")
        if not isinstance(session_id, str) or not session_id or not isinstance(task_key, str) or not task_key:
            raise ValueError("registration needs a session_id and task_key")
        if body.get("policy") != expected_policy or expected_policy not in POLICIES:
            raise ValueError("registered policy does not match this server")
        family = body.get("family")
        if (
            family not in ("land_or_hold", "find_pad", "obstacle_bypass", *ADVANCED_FAMILIES)
            or body.get("path") is not None
        ):
            raise ValueError("unsupported visual family or private path payload in registration")
        start, heading, blocked, variant, geometry, clear_after = None, None, None, None, None, None
        if family == "land_or_hold":
            if expected_policy == "always_b":
                raise ValueError("always_b is only supported for alternate_pad")
            pad = coordinates(body.get("pad"), "pad")
            blocked = body.get("blocked")
            if expected_policy == "honest":
                if not isinstance(blocked, bool):
                    raise ValueError("honest registration needs a boolean blocked fact")
            elif "blocked" in body:
                raise ValueError("blind and broken pilots must not receive hidden scene state")
        elif family == "find_pad":
            if expected_policy not in REGISTERED_POLICIES:
                raise ValueError("find_pad supports honest, blind_shortcut and broken only")
            allowed = {"session_id", "task_key", "policy", "family", "path", "start", "heading"}
            if expected_policy == "honest":
                allowed.add("pad")
            if set(body) - allowed:
                raise ValueError("find_pad registration must not receive extra hidden scene state")
            start = coordinates(body.get("start"), "start")
            if math.hypot(*start) > 2.0:
                raise ValueError("find_pad start must be within 2 m of launch")
            heading = body.get("heading")
            if (
                isinstance(heading, bool)
                or not isinstance(heading, (int, float))
                or not math.isfinite(heading)
                or not -180 <= heading <= 180
            ):
                raise ValueError("find_pad heading must be finite and between -180 and 180 degrees")
            heading = float(heading)
            pad = coordinates(body.get("pad"), "pad") if expected_policy == "honest" else None
        elif family == "obstacle_bypass":
            if expected_policy not in REGISTERED_POLICIES:
                raise ValueError("obstacle_bypass supports honest, blind_shortcut and broken only")
            allowed = {"session_id", "task_key", "policy", "family", "pad"}
            if expected_policy == "honest":
                allowed.add("variant")
            if set(body) - allowed:
                raise ValueError("obstacle_bypass registration must not receive extra hidden scene state")
            pad = coordinates(body.get("pad"), "pad")
            if pad != (22.0, 0.0):
                raise ValueError("obstacle_bypass requires its public pad at (22, 0)")
            if expected_policy == "honest":
                variant = body.get("variant")
                if type(variant) is not int or variant not in (0, 1):
                    raise ValueError("honest obstacle_bypass registration needs variant 0 or 1")
        else:
            from aerialsim_flight.advanced import PublicGeometry

            allowed = {"session_id", "task_key", "policy", "family", "geometry"}
            if expected_policy == "honest":
                allowed |= {"variant", "clear_after"}
            if set(body) - allowed:
                raise ValueError("advanced registration must not receive extra hidden scene state")
            geometry = PublicGeometry.model_validate(body.get("geometry")).model_dump(mode="json")
            pad = tuple(geometry["pad_a"])
            if expected_policy == "honest":
                variant, clear_after = body.get("variant"), body.get("clear_after")
            # Validate the complete contract now, before accepting the session.
            advanced_plan(expected_policy, family, geometry, variant=variant, clear_after=clear_after)
        scene = Scene(task_key, expected_policy, family, pad, blocked, start, heading, variant, geometry, clear_after)
        with self._lock:
            existing = self._scenes.get(session_id)
            if existing is not None and existing != scene:
                raise ValueError("session is already registered for a different task or policy")
            self._scenes[session_id] = scene

    def scene(self, session_id: str | None) -> Scene:
        with self._lock:
            scene = self._scenes.get(session_id)
        if scene is None:
            raise ValueError("unknown or missing scripted session; use the rehearsal harness")
        return scene

    def is_registered(self, session_id: str | None) -> bool:
        with self._lock:
            return session_id in self._scenes


def read_turn(messages: list[dict]) -> tuple[tuple[float, float] | None, int]:
    """The pad coordinates and the zero based turn number from the user messages so far."""
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    texts = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("each message must be an object")
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if not isinstance(c, (str, list)) or isinstance(c, list) and any(not isinstance(p, dict) for p in c):
            raise ValueError("user content must be text or content parts")
        texts.append(c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if p.get("type") == "text"))
    joined = "\n".join(texts)
    pad = re.search(r"x=(-?\d+(?:\.\d+)?), y=(-?\d+(?:\.\d+)?)", joined)
    turns = re.findall(r"Turn (\d+) of", joined)
    n = int(turns[-1]) - 1 if turns else 0
    return ((float(pad.group(1)), float(pad.group(2))) if pad else None), n


def reply_for_request(policy: str, body: dict, session_id: str | None, registry: SessionRegistry) -> str:
    pad, n = read_turn(body.get("messages", []))
    if policy in (*REGISTERED_POLICIES, "always_b") or registry.is_registered(session_id):
        if body.get("model") != "scripted":
            raise ValueError("registered pilots require the synthetic scripted model")
        scene = registry.scene(session_id)
        if scene.policy != policy or scene.family in ("land_or_hold", "obstacle_bypass") and pad != scene.pad:
            raise ValueError("request policy or observed pad does not match the registered task")
        if n < 0:
            raise ValueError("turn number must be positive")
        if scene.family in ADVANCED_FAMILIES:
            steps = advanced_plan(
                policy, scene.family, scene.geometry, variant=scene.variant, clear_after=scene.clear_after
            )
        elif scene.family == "find_pad":
            # Coordinates in text are never used as a hidden-target fallback.
            steps = find_pad_plan(policy, scene.start, pad=scene.pad)
        elif scene.family == "obstacle_bypass":
            steps = obstacle_bypass_plan(policy, scene.pad, variant=scene.variant)
        else:
            steps = plan(policy, scene.pad, blocked=scene.blocked)
    else:
        # Compatibility for historical always_land/always_hold runs only.
        steps = plan(policy, pad or (20.0, -5.0))
    return json.dumps(steps[min(n, len(steps) - 1)])


MODEL_NAME = "scripted"
"""The model id the eval is run with, and the one this server reports it serves."""


class Handler(BaseHTTPRequestHandler):
    policy = "always_land"
    registry = SessionRegistry()
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802
        """The two reads an inference server has to answer, and 404 for anything else.

        This pilot stands in for an inference server, so the eval treats it as one. It
        waits for /health before starting, and it lists /v1/models to check the model it
        was asked for exists. The default handler answers an unimplemented GET with 501,
        and the eval then sat on "Waiting for inference server to start up" forever, so no
        flight ever began. Answering everything with 404 got past the wait and failed the
        model listing instead. Both are answered properly here.
        """
        if self.path.rstrip("/") == "/health":
            self._send(200, b'{"status": "ok"}', "application/json")
            return
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            body = json.dumps(
                {
                    "object": "list",
                    "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "aerialsim"}],
                }
            ).encode()
            self._send(200, body, "application/json")
            return
        self._send(404, b'{"error": "not found"}', "application/json")

    def do_POST(self):  # noqa: N802
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            if self.path == "/register":
                self.registry.register(body, self.policy)
                self._send(200, b'{"registered": true}', "application/json")
                return
            if self.path not in ("/chat/completions", "/v1/chat/completions"):
                self._send(404, b'{"error": "unknown endpoint"}', "application/json")
                return
            reply = reply_for_request(self.policy, body, self.headers.get("X-Session-ID"), self.registry)
            print(f"policy={self.policy} stream={body.get('stream', False)} -> {reply}", file=sys.stderr, flush=True)
            model = body.get("model", "scripted")
            usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            if body.get("stream"):
                head = {"id": "scripted", "object": "chat.completion.chunk", "created": 0, "model": model}
                chunks = [
                    {
                        **head,
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": None}
                        ],
                    },
                    {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
                ]
                data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                self._send(200, data.encode(), "text/event-stream")
                return
            out = {
                "id": "scripted",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                "usage": usage,
            }
            self._send(200, json.dumps(out).encode(), "application/json")
        except (ValueError, TypeError, KeyError) as exc:
            print(f"request failed: {exc!r}", file=sys.stderr, flush=True)
            self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # noqa: D102
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=POLICIES, default="always_land")
    ap.add_argument("--port", type=int, default=8123)
    a = ap.parse_args()
    Handler.policy = a.policy
    Handler.registry = SessionRegistry()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"PILOT_READY policy={a.policy} port={a.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

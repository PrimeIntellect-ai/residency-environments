"""Visual flight: the model sees camera frames during the graded flight and answers one action a turn.

The environment is the trusted controller. It owns the simulator box, spawns the
scene, takes each frame, runs each action and scores the recording. The model
never gets a box and never learns the hidden fact except through the frame.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field, model_validator
from verifiers.v1.configs.env import TimeoutConfig as EnvTimeoutConfig
from verifiers.v1.runtimes import DockerConfig, PrimeConfig, SubprocessConfig, provision_runtime
from verifiers.v1.utils.loaders import harness_config_type

from aerialsim_flight import advanced
from aerialsim_flight.obstacle_bypass import (
    barrier_hit,
    barrier_object,
    choice_score,
    route_rules,
)
from aerialsim_flight.obstacle_bypass import (
    route_choice as classify_route_choice,
)
from aerialsim_flight.path_grading import PATH_METRICS, PathRules, grade_path
from aerialsim_flight.start_pose import PLACE_START_SOURCE
from aerialsim_flight.taskset import (
    CHECK_SIM,
    CLEAR_SIM_LOGS,
    IMAGE,
    MAX_RECORDING_BYTES,
    MAX_SAMPLE_GAP_S,
    RECORDER_SOURCE,
    RECORDING_PATH,
    RESET_SIM,
    SAVE_SIM_LOGS,
    SIM_ALIVE,
    SIM_BOOT_TRIES,
    SIM_BOOT_WAIT_S,
    SIM_DIAGNOSTICS,
    START_RECORDER,
    STOP_RECORDER,
    STOP_SIM,
    TOUCHDOWN_SPEED_MPS,
    collided,
    covers_flight,
    landed_on_ground,
    max_altitude,
    parse_recording,
    path_length,
    resting_z,
)
from aerialsim_flight.timed_events import (
    EVENT_CONFIG_PATH,
    EVENT_LOG_PATH,
    EVENT_SOURCE,
    MAX_EVENT_BYTES,
    START_EVENTS,
    STOP_EVENTS,
    parse_event_log,
    validate_event_log,
)
from aerialsim_flight.visual_runtime import advanced_controller, advanced_recorder

MAX_TURNS = 12
NULL_HARNESS = harness_config_type("null").model_validate({"id": "null"})
SIM_BOOT_STAGGER_S = 3.0
"""Least time between two simulator launches.

Simultaneous cold starts on one card can exhaust device memory. Three seconds against a
30 second boot costs almost nothing in concurrency and keeps the expensive moments from
overlapping. Set to 0 to launch without spacing."""

SIM_DEAD_AFTER_POLLS = 12
"""Polls to allow before checking whether the game process still exists. At five seconds a
poll that is one minute, twice the median successful boot of 30 seconds."""
MAX_LEG_M = 30.0
MAX_SPEED_MPS = 5.0
MAX_TAKEOFF_M = 10.0
MAX_HOVER_S = 10.0
MAX_ALTITUDE_M = 25.0
BOUNDS = {"x": (-30.0, 40.0), "y": (-20.0, 20.0)}
HOLD_HEIGHT_M = 5.0
HOLD_S = 3.0
HOLD_TOLERANCE_M = 1.5
HOLD_SETTLED_S = 1.5
"""How long the drone must be stopped at the end for the hold to count.

The old rule wanted three seconds of stillness, but a drone that arrives and stops spends
the first second of that decelerating, so the window never came out clean. What is being
asked is that it came to a stop and stayed stopped until the flight ended, which this
measures directly."""
HOLD_STANDOFF_M = 15.0
"""How far from the pad a hold may settle, once the drone has been over it.

Backing away from a pad you have refused is correct, so the end position is judged on
whether the drone stayed in the area rather than on whether it parked on the spot.
Fifteen metres is a standoff; flying off across the park is not."""
HOLD_MIN_M = 2.5
"""Lowest a hold may sit and still count as having avoided the landing.

Above the two metre post, so the drone is clear of whatever is on the pad."""
HOLD_MAX_M = MAX_ALTITUDE_M
"""Highest a hold may sit: anywhere inside the flight envelope the mission already allows."""
STRICT_HOLD_HEIGHT = False
"""Whether a hold must sit within HOLD_TOLERANCE_M of exactly HOLD_HEIGHT_M.

It used to, and that made the score measure one exact manoeuvre instead of the outcome.
Claude and Codex each read all eight pads correctly on land or hold and each scored 4 of
8, the same as a model that always lands: they avoided the landing and did not park at
five metres. A correct decision earned nothing, so training had nothing to climb.

A hold counts when the drone stayed off the ground, near the pad, steady, anywhere
between HOLD_MIN_M and HOLD_MAX_M. Set this true for the older, narrower rule, which is
the only way to compare against runs scored under it."""
PAD_RADIUS_M = 2.5
ACTION_TIMEOUT_S = 90.0
ACTION_LIMIT_IN_BOX_S = 60.0
EPISODE_TIMEOUT_S = 1500.0
CAMERA_PITCH_DEG = -70.0
POST_HEIGHT_M = 2.0
POST_WIDTH_M = 1.0
PAD_MARK_WIDTH_M = 3.0
PAD_MARK_HEIGHT_M = 0.02
"""Thin enough that a landing on its edge cannot skid."""
PAD_MARK_COLOUR = (35, 35, 40)
"""Dark, so a white block on it stands out from above. The block keeps the default white."""
FRAME_WIDTH = 640
FRAME_JPEG_QUALITY = 80
"""Frames go to the model as JPEG: a 28 turn mission with PNG frames passed the
provider's 30 MB request limit and errored."""
PLACE_START_TRIES = 4
FIND_PAD_EVALUATION_SEED_BASE = 1_000_000
FIND_PAD_SEEDS_PER_SPLIT = 10_000
FIND_PAD_FORWARD_M = 14.0
FIND_PAD_SIDE_M = 5.0
OBSTACLE_EVALUATION_SEED_BASE = 2_000_000
OBSTACLE_X_RANGES = {"development": (10.0, 11.75), "evaluation": (12.25, 14.0)}

ACTIONS = {
    "takeoff": ("height",),
    "goto": ("x", "y", "z", "speed"),
    "look": ("pitch", "yaw"),
    "hover": ("seconds",),
    "land": (),
    "abort": (),
}

SYSTEM_PROMPT = f"""You fly a drone in a photoreal park from what its camera shows you.

Each turn you get the drone's state and one camera frame. Answer with exactly one
action as one JSON object and nothing else:

  {{"action": "takeoff", "height": 5}}            climb to a height in metres, at most {MAX_TAKEOFF_M:.0f}
  {{"action": "goto", "x": 20, "y": -5, "z": -8, "speed": 3}}   fly one leg, at most {MAX_LEG_M:.0f} m, at most {MAX_SPEED_MPS:.0f} m/s
  {{"action": "look", "pitch": -50, "yaw": 0}}    point the camera, degrees, negative pitch looks down
  {{"action": "hover", "seconds": 2}}             hold position, at most {MAX_HOVER_S:.0f} s
  {{"action": "land"}}                            descend here until the ground stops the drone
  {{"action": "abort"}}                           climb to {HOLD_HEIGHT_M:.0f} m above the ground and hold there; ends the mission

Positions are NED: x north, y east, z down, so 8 metres up is z = -8. Landing and
abort end the mission. You have {MAX_TURNS} turns. The pad is a flat dark round
marker on the grass. Look before you decide: the mission text never says what is
on the pad."""

FIND_PAD_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

For this mission, find the marker's location from the camera. Your position and
heading describe the drone, not the target. Heading is world NED: 0 faces +x,
90 faces +y. A look action's yaw is relative to the drone's heading, not world
yaw. Initially the camera pitch is -70 degrees and its relative yaw is 0.
The camera has a 90-degree horizontal field of view. A pitch of -90 looks
straight down. Goto coordinates remain world NED.

How to search: take off to 10 m, the highest allowed, and look straight down.
From 10 m the camera sees a square about 20 m across under the drone. The
marker is within 20 m of your launch point, so fly to a spot about 10 m from
the launch in one direction, look straight down, and if the marker is not in
view try another direction. Only report the marker when a dark round shape is
clearly in the frame.

How to centre on it: your heading is 0, so with the camera straight down and
look yaw 0 the top of the frame points to +x and the right of the frame points
to +y. A shape a third of the way from the centre to the frame edge is about a
third of your altitude away in that direction: seen up and to the right from
10 m, it is roughly 3 m further in x and 3 m further in y. Fly there, then
descend to 5 m with a goto at the same x and y, look straight down again and
correct until the marker is in the middle of the frame. Land only when it is
centred at 5 m or lower; landing from 10 m usually misses by several metres."""
)

OBSTACLE_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """

For this mission, use the camera to find which passage around the white barrier
is open. The landing pad is clear. Inspect before committing to a passage;
the text gives both permitted corridors but never says which one is blocked.
Look yaw is relative to the drone's heading, while goto coordinates use world
NED. Initially the camera pitch is -70 degrees and relative yaw is 0.

How to fly it: from the inspection checkpoint look straight ahead with pitch
-10 and yaw 0. The wall covers one half of the view and the other half is open
grass. The open corridor is on the grass side: if the grass is on the right of
the frame the +y corridor is open, if it is on the left the -y corridor is
open. Decide from that one look. Then move sideways first, while still short of
x = 8: go to y = +4 for the +y corridor or y = -4 for the -y corridor, keeping
x at 7. Only then fly forward along that corridor to x = 17 at the same y. Never
aim diagonally at a corridor from the centre line: that path runs into the
wall. From x = 17 go to the merge checkpoint, then to the pad, then land."""
)


def parse_action(text: str) -> tuple[dict | None, str]:
    """One action from the model's reply, checked against the contract."""
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None, "reply had no JSON object"
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, "reply was not valid JSON"
    if not isinstance(raw, dict) or not isinstance(raw.get("action"), str) or raw["action"] not in ACTIONS:
        return None, f"action must be one of {', '.join(ACTIONS)}"
    name = raw["action"]
    action: dict[str, Any] = {"do": name}
    for field in ACTIONS[name]:
        if field not in raw:
            return None, f"{name} needs {field}"
        try:
            action[field] = float(raw[field])
        except (TypeError, ValueError):
            return None, f"{field} must be a number"
    if name == "takeoff" and not 0.5 <= action["height"] <= MAX_TAKEOFF_M:
        return None, f"takeoff height must be between 0.5 and {MAX_TAKEOFF_M:.0f}"
    if name == "goto":
        if not BOUNDS["x"][0] <= action["x"] <= BOUNDS["x"][1] or not BOUNDS["y"][0] <= action["y"] <= BOUNDS["y"][1]:
            return None, "goto is outside the park"
        if not -MAX_ALTITUDE_M <= action["z"] <= 2.0:
            return None, f"goto z must be between -{MAX_ALTITUDE_M:.0f} and 2"
        if not 0.5 <= action["speed"] <= MAX_SPEED_MPS:
            return None, f"speed must be between 0.5 and {MAX_SPEED_MPS:.0f}"
    if name == "hover" and not 0.0 < action["seconds"] <= MAX_HOVER_S:
        return None, f"hover seconds must be between 0 and {MAX_HOVER_S:.0f}"
    if name == "look" and not (-90.0 <= action["pitch"] <= 30.0 and -180.0 <= action["yaw"] <= 180.0):
        return None, "look pitch must be between -90 and 30, yaw between -180 and 180"
    return action, ""


def leg_too_long(action: dict, state: dict) -> bool:
    if action["do"] != "goto":
        return False
    return math.dist([state["x"], state["y"], state["z"]], [action["x"], action["y"], action["z"]]) > MAX_LEG_M


# Runs inside the simulator box for the whole episode: one process, one simulator
# connection. Actions arrive as files; each answer is the state and a frame.
# Between actions an altitude hold task keeps the drone up. A finished move leaves
# no goal behind, and a move to the spot the drone is already at issues none, so
# without the hold the drone falls the moment nothing is commanding it.
CONTROLLER_SOURCE = r"""import base64, json, math, os, sys, threading, time
import cosysairsim as airsim
from aerialsim_flight.drone import connect

ctl = sys.argv[1]
HOLD_TASK_S = 3600.0
SHORT_MOVE_S = 15.0
COAST_S, COAST_M = 0.45, 0.3
FINE_LOOKAHEAD_M = 0.5
LAND_STEP_S, LAND_STEP_M, LAND_MAX_STEPS, LAND_STABLE_M = 1.0, 1.0, 40, 0.05
LAND_NEAR_GROUND_M, LAND_STALLS_ALLOWED = 0.4, 4
LAND_SETTLE_S = 8.0
KEEP_HEADING = airsim.YawMode(True, 0.0)
d = connect()
c = d._client
# Measured on the first setup call, by ground_level(), not read here. See that function.
rest_z = None
deadline = None
timed_out = False
current = -1
lock = threading.Lock()
cancelling = threading.Lock()


def watchdog():
    # The RPC client is bound to the thread that made it, so the watchdog gets its own.
    global timed_out, deadline
    w = airsim.MultirotorClient()
    w.confirmConnection()
    while True:
        time.sleep(0.2)
        with lock:
            late = deadline is not None and time.time() > deadline
            mine = current
        if not late:
            continue
        # The cancel belongs to one action. The main loop waits on this lock before
        # starting the next action, and the cancel is dropped if that action is gone.
        with cancelling:
            with lock:
                if mine != current or deadline is None:
                    continue
                timed_out = True
                deadline = None
            try:
                w.cancelLastTask()
                z = w.getMultirotorState().kinematics_estimated.position.z_val
                w.moveByVelocityZAsync(0.0, 0.0, z, HOLD_TASK_S, yaw_mode=KEEP_HEADING)
            except Exception as exc:
                print("watchdog could not stop the drone:", repr(exc), file=sys.stderr, flush=True)


def step(fn):
    # One simulator command. Nothing runs after the watchdog has fired.
    if timed_out:
        return
    fn()


def hold(z):
    # Altitude hold, xy velocity zero, until the next command replaces it.
    if timed_out:
        return
    c.moveByVelocityZAsync(0.0, 0.0, z, HOLD_TASK_S, yaw_mode=KEEP_HEADING)


def airborne():
    return rest_z - d.state()["z"] > 0.5


def short_of(x, y, z, back):
    # The point on the way to (x, y, z) that stops back metres before it, or
    # None when the whole move is about that short and only the slow leg makes sense.
    here = d.state()
    dx, dy, dz = x - here["x"], y - here["y"], z - here["z"]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist <= back + 0.5:
        return None
    f = (dist - back) / dist
    return (here["x"] + f * dx, here["y"] + f * dy, here["z"] + f * dz)


def solid_png(path, rgb, size=64):
    # A one colour PNG written with the standard library, for the texture API.
    import struct, zlib
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)


def settle(target_z=None, max_s=5.0):
    # Wait until the drone has stopped moving, and reached target_z when given.
    t0 = time.time()
    while time.time() - t0 < max_s and not timed_out:
        s = d.state()
        still = math.sqrt(s["vx"] ** 2 + s["vy"] ** 2 + s["vz"] ** 2) < 0.3
        there = target_z is None or abs(s["z"] - target_z) < 0.3
        if still and there:
            return
        time.sleep(0.1)


GROUND_SETTLE_S = 5.0
GROUND_CONFIRM_S = 2.0
GROUND_AGREE_M = 0.25
GROUND_BUDGET_S = 40.0


def ground_level():
    # Where the ground is, measured rather than read once.
    #
    # Every altitude in a flight is measured from this number, so one bad reading sends
    # the whole flight wrong. The simulator drops the drone onto the park at reset and the
    # drone can end up inside it. It is ejected back to the surface about a second later,
    # so a read taken the moment the controller starts can catch it sunk. That reads as a
    # ground level several metres too low, and the landing then descends to it, coming to
    # rest metres under the park and failing the touchdown, the collision, the coverage
    # and the corridor on a flight that was flown correctly.
    #
    # So wait until the drone is still, read, pause, and read again, and require the two
    # to agree. A drone ejected between the reads gives two that do not, and the loop
    # takes the later one.
    previous = None
    deadline = time.time() + GROUND_BUDGET_S
    while True:
        settle(max_s=GROUND_SETTLE_S)
        z = d.state()["z"]
        if previous is not None and abs(z - previous) <= GROUND_AGREE_M:
            return z
        previous = z
        if time.time() >= deadline:
            return z
        time.sleep(GROUND_CONFIRM_S)


def run(spec):
    global rest_z
    do = spec["do"]
    hold_z = None
    if rest_z is None:
        rest_z = ground_level()
    # Where the drone came to rest, read at touchdown. A landing is graded from this,
    # not from the state half a second later: disarming drops the drone through the
    # ground, so the later reading is a fall, not a landing. See the land branch.
    at_rest = None
    if do == "setup":
        c.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
            airsim.utils.euler_to_quaternion(0.0, math.radians(spec["pitch"]), 0.0)))
        for obj in spec.get("spawn", []):
            pose = airsim.Pose(airsim.Vector3r(obj["x"], obj["y"], obj["z"]), airsim.Quaternionr(0, 0, 0, 1))
            spawned = c.simSpawnObject(obj["name"], obj["asset"], pose, airsim.Vector3r(obj["sx"], obj["sy"], obj["sz"]), False)
            if not spawned:
                raise RuntimeError("could not spawn %s" % obj["name"])
            if obj.get("colour"):
                # The look of the object is part of the task, so a texture that does not apply is an error.
                path = "/tmp/%s.png" % obj["name"]
                solid_png(path, obj["colour"])
                if not c.simSetObjectMaterialFromTexture(obj["name"], path):
                    raise RuntimeError("could not colour %s" % obj["name"])
    elif do == "takeoff":
        # Climb with a position move first: switching to the altitude hold straight
        # out of the takeoff call sent the drone into the ground half the time. The
        # hold takes over once the drone is still, and the wait is on height, which
        # also covers a short climb that never reports done.
        #
        # Leave no moment with nothing commanding the drone. The takeoff call climbs at
        # half a metre a second and then returns. In the gap before the position move is
        # accepted the drone sinks through the park and can scrape the ground on the way
        # out fast enough to count as a crash. The hold is issued first and costs nothing:
        # it does not wait, and the position move replaces it immediately.
        hold_z = rest_z - abs(spec["height"])
        step(lambda: c.takeoffAsync().join())
        hold(hold_z)
        step(lambda: c.moveToZAsync(hold_z, 2.0, timeout_sec=SHORT_MOVE_S, yaw_mode=KEEP_HEADING).join())
        settle()
        hold(hold_z)
        settle(hold_z, max_s=12.0)
    elif do == "goto":
        # The fast leg returns while the drone still carries speed, and the drone
        # coasts about half a metre per metre per second past where the leg ended.
        # So the fast leg stops short of the target by that much, and a slow leg
        # with a small lookahead finishes it; the default lookahead treats a move
        # of a metre or so as already done and never engages. Then settle under
        # the position goal, then the hold, then wait on height.
        hold_z = spec["z"]
        near = short_of(spec["x"], spec["y"], spec["z"], COAST_S * spec["speed"] + COAST_M)
        if near is not None:
            step(lambda: c.moveToPositionAsync(near[0], near[1], near[2], spec["speed"], yaw_mode=KEEP_HEADING).join())
        step(lambda: c.moveToPositionAsync(spec["x"], spec["y"], spec["z"], 1.0, timeout_sec=SHORT_MOVE_S,
                                           yaw_mode=KEEP_HEADING, lookahead=FINE_LOOKAHEAD_M, adaptive_lookahead=0).join())
        # Settle under the position goal first: the hold issued while the drone
        # is still moving cancels that goal and lets it coast past the target.
        settle()
        hold(hold_z)
        settle(hold_z)
    elif do == "look":
        step(lambda: d.camera_control(spec["pitch"], spec["yaw"]))
        hold_z = d.state()["z"]
    elif do == "hover":
        hold_z = d.state()["z"]
        step(lambda: c.moveByVelocityZAsync(0.0, 0.0, hold_z, spec["seconds"], yaw_mode=KEEP_HEADING).join())
    elif do == "land":
        # Descend on height targets, a metre a second, with sideways velocity zero:
        # a plain downward velocity let the drone drift a metre before touchdown.
        # Never command a target below the ground. On "the drone stopped moving" alone, a
        # descent that never registers contact keeps asking for another metre and ends tens
        # of metres under the park, which fails the path corridor. rest_z is the height the
        # drone sat at before takeoff, so it is the floor by definition.
        prev_z = d.state()["z"]
        stalls = 0
        for _ in range(LAND_MAX_STEPS):
            if timed_out:
                break
            if prev_z >= rest_z - LAND_STABLE_M:
                break
            target = min(prev_z + LAND_STEP_M, rest_z)
            c.moveByVelocityZAsync(0.0, 0.0, target, LAND_STEP_S, yaw_mode=KEEP_HEADING).join()
            z = d.state()["z"]
            if abs(z - prev_z) < LAND_STABLE_M:
                # Nothing moved. Near the ground that is the ground holding the drone up,
                # which is the landing. Metres above it, it only means one second of the
                # altitude hold did not cover the step, so keep asking. The same loop in
                # the drone API gave up on the first short step and left a descent 2.09 m
                # up reporting success, which happens when land follows a stationary
                # hover rather than a move.
                stalls += 1
                if z >= rest_z - LAND_NEAR_GROUND_M or stalls >= LAND_STALLS_ALLOWED:
                    break
            else:
                stalls = 0
            prev_z = z
        # Hold ground level, wait for the drone to stop, then read it, and leave the
        # motors running.
        #
        # The descent settles to within a few centimetres of ground level. Releasing the
        # motors stops the simulator holding the drone up and it free-falls through the
        # solid park, which scores as a crash and leaves the fall in the recording, failing
        # the collision, the coverage and the path corridor at once. The drone sitting
        # still at ground level is the landing. Abort holds the same way.
        #
        # The descent loop can also hand back while the drone is still moving, when it
        # gives up after several steps that covered little ground. The hold finishes the
        # descent, so reading the state the instant the loop returns catches the drone
        # mid-air and fails the touchdown, and fails coverage too, because the graded state
        # no longer matches the end of the recording.
        #
        # The hold is what keeps the drone down afterwards: with nothing commanding it, it
        # sinks and puts unphysical samples back in the recording.
        hold(rest_z)
        settle(rest_z, max_s=LAND_SETTLE_S)
        at_rest = d.state()
    elif do == "abort":
        hold_z = spec["hold_z"]
        hold(hold_z)
        settle(hold_z, max_s=12.0)
        step(lambda: time.sleep(spec["hold_s"]))
    if hold_z is not None and airborne():
        hold(hold_z)
    time.sleep(0.5)
    # FINAL READING. The scenario controller cuts the flight here and takes its own
    # snapshot after pausing the world, so this marker is its anchor: keep the line.
    s = at_rest if at_rest is not None else d.state()
    frame = None
    for _ in range(3):
        frame = d.camera()
        if frame:
            break
        time.sleep(0.3)
    return {"state": s, "frame": base64.b64encode(frame).decode() if frame else "",
            "timed_out": timed_out, "rest_z": rest_z}


threading.Thread(target=watchdog, daemon=True).start()
os.makedirs(ctl, exist_ok=True)
n = 0
while True:
    req = os.path.join(ctl, "req_%d.json" % n)
    if not os.path.exists(req):
        time.sleep(0.05)
        continue
    spec = json.loads(open(req).read())
    with cancelling:
        pass
    started = time.time()
    with lock:
        current = n
        timed_out = False
        deadline = started + float(spec.get("limit", 60.0))
    try:
        out = run(spec)
    except Exception as exc:
        out = {"error": repr(exc), "timed_out": timed_out}
    with lock:
        deadline = None
    out["seconds"] = round(time.time() - started, 1)
    tmp = os.path.join(ctl, "res_%d.tmp" % n)
    open(tmp, "w").write(json.dumps(out))
    os.replace(tmp, os.path.join(ctl, "res_%d.json" % n))
    n += 1
"""

CONTROL_DIR = "/tmp/visual_ctl"
START_CONTROLLER = (
    f"rm -rf {CONTROL_DIR}; mkdir -p {CONTROL_DIR}; nohup python3 /tmp/visual_controller.py {CONTROL_DIR} "
    "> /tmp/controller.log 2>&1 & echo $! > /tmp/controller.pid; sleep 0.5; test -s /tmp/controller.pid"
)
STOP_CONTROLLER = "kill $(cat /tmp/controller.pid) 2>/dev/null; pkill -f visual_controller.py; true"


def wait_for_result(n: int, timeout_s: float) -> str:
    """Shell that moves request n into place, then waits for the controller's answer."""
    ticks = int(timeout_s / 0.2)
    return (
        f"mv {CONTROL_DIR}/req_{n}.tmp {CONTROL_DIR}/req_{n}.json && "
        f"for i in $(seq 1 {ticks}); do "
        f"if [ -f {CONTROL_DIR}/res_{n}.json ]; then cat {CONTROL_DIR}/res_{n}.json; exit 0; fi; "
        f"if ! kill -0 $(cat /tmp/controller.pid) 2>/dev/null; then echo controller gone >&2; exit 4; fi; "
        f"sleep 0.2; done; exit 3"
    )


class VisualData(vf.TaskData):
    family: str
    seed: int
    pad: list[float]
    blocked: bool
    """Hidden from the model. The frame is the only way to learn it."""
    path: PathRules | None = None
    """Optional route requirements. The mission must explain any required maneuvers."""
    start: list[float] | None = None
    """Optional grounded launch XY, applied by trusted setup before recording."""
    heading: float = 0.0
    variant: int | None = Field(default=None, ge=0, le=3)
    """Hidden pad placement or open corridor; never sent to the model."""
    split: Literal["development", "evaluation"] = "development"
    obstacle_x: float | None = None
    """Hidden longitudinal wall position for obstacle_bypass."""
    scenario: advanced.AdvancedScenario | None = None
    """Trusted mission geometry and hidden facts for families 4–7."""

    @model_validator(mode="after")
    def validate_find_pad(self):
        if self.family != "find_pad":
            return self
        if (
            self.start is None
            or len(self.start) != 2
            or not all(math.isfinite(v) for v in self.start)
            or math.hypot(*self.start) > 2.0
            or not math.isfinite(self.heading)
            or not -180 <= self.heading <= 180
        ):
            raise ValueError("find_pad requires a finite launch XY within 2 m of origin and heading within +/-180")
        if self.blocked or self.path is not None or self.variant not in (0, 1):
            raise ValueError("find_pad requires a clear pad, a placement variant and no route constraints")
        base = FIND_PAD_EVALUATION_SEED_BASE if self.split == "evaluation" else 0
        if not base <= self.seed < base + FIND_PAD_SEEDS_PER_SPLIT:
            raise ValueError("find_pad seed must belong to its reserved split")
        if len(self.pad) != 2 or not all(math.isfinite(v) for v in self.pad):
            raise ValueError("find_pad requires a finite pad XY")
        if not all(BOUNDS[axis][0] <= value <= BOUNDS[axis][1] for axis, value in zip(("x", "y"), self.pad)):
            raise ValueError("find_pad target must be inside the park")
        if math.dist(self.start, self.pad) <= PAD_RADIUS_M:
            raise ValueError("find_pad target must be separate from the launch site")
        return self

    @model_validator(mode="after")
    def validate_obstacle_bypass(self):
        if self.family != "obstacle_bypass":
            return self
        if self.variant not in (0, 1) or self.blocked or self.pad != [22.0, 0.0]:
            raise ValueError("obstacle_bypass requires a placement variant and the clear pad at (22, 0)")
        if self.start is not None or self.heading != 0 or self.path != route_rules():
            raise ValueError("obstacle_bypass requires the shared public route and original launch pose")
        low, high = OBSTACLE_X_RANGES[self.split]
        if self.obstacle_x is None or not math.isfinite(self.obstacle_x) or not low <= self.obstacle_x <= high:
            raise ValueError("obstacle_bypass wall x must be finite and within its reserved split range")
        base = OBSTACLE_EVALUATION_SEED_BASE if self.split == "evaluation" else 0
        if not base <= self.seed < base + FIND_PAD_SEEDS_PER_SPLIT:
            raise ValueError("obstacle_bypass seed must belong to its reserved split")
        return self

    @model_validator(mode="after")
    def validate_advanced(self):
        if self.family not in advanced.ADVANCED_FAMILIES:
            if self.scenario is not None:
                raise ValueError("scenario is only supported by the advanced mission families")
            return self
        scenario = self.scenario
        if scenario is None or (scenario.family, scenario.seed, scenario.split, scenario.variant) != (
            self.family,
            self.seed,
            self.split,
            self.variant,
        ):
            raise ValueError("advanced mission identity must match its trusted scenario")
        if self.start is not None or self.heading != 0 or self.blocked or self.obstacle_x is not None:
            raise ValueError("advanced missions require their standard launch and scenario obstacles")
        if self.pad != list(advanced.public_geometry(scenario)["pad_a"]) or self.path != advanced.route_rules(scenario):
            raise ValueError("advanced mission pad and route must match its scenario")
        return self


class VisualTask(vf.Task[VisualData, vf.State, vf.TaskConfig]):
    pass


class VisualConfig(vf.TasksetConfig):
    image: str | None = None
    count: int = Field(default=8, ge=1)
    start_seed: int = Field(default=0, ge=0)
    family: Literal[
        "land_or_hold", "find_pad", "obstacle_bypass", "alternate_pad", "patrol_inspection", "go_around", "compound"
    ] = "land_or_hold"
    split: Literal["development", "evaluation"] = "development"

    @model_validator(mode="after")
    def reserve_evaluation_seeds(self):
        if (
            self.family in ("find_pad", "obstacle_bypass")
            and self.start_seed + (self.count + 1) // 2 > FIND_PAD_SEEDS_PER_SPLIT
        ):
            raise ValueError(
                f"{self.family} start_seed is an offset within its reserved split; requested pairs exceed the split"
            )
        if self.family in advanced.ADVANCED_FAMILIES:
            size = advanced.VARIANT_COUNTS[self.family]
            if self.start_seed + (self.count + size - 1) // size > advanced.SEEDS_PER_SPLIT:
                raise ValueError(f"{self.family} requested scene groups exceed its reserved split")
        return self


class VisualTaskset(vf.Taskset[VisualTask, VisualConfig]):
    """Counterfactual pairs with hidden occupancy, pad location or open corridor.
    Count is the number of tasks, rounded up to complete each pair."""

    def load(self) -> Iterator[VisualTask]:
        if self.config.family in advanced.ADVANCED_FAMILIES:
            yield from self._advanced()
            return
        if self.config.family == "find_pad":
            yield from self._find_pad()
            return
        if self.config.family == "obstacle_bypass":
            yield from self._obstacle_bypass()
            return
        count = self.config.count + (self.config.count % 2)
        for i in range(count):
            pair = self.config.start_seed + i // 2
            rng = random.Random(pair * 7919 + 17)
            pad = [round(rng.uniform(12.0, 24.0), 1), round(rng.uniform(-6.0, 6.0), 1)]
            blocked = (i + pair) % 2 == 1
            yield VisualTask(
                VisualData(
                    idx=i,
                    name=f"land_or_hold_{pair:04d}_{'blocked' if blocked else 'clear'}",
                    prompt=None,
                    system_prompt=SYSTEM_PROMPT,
                    image=None,
                    family="land_or_hold",
                    seed=pair,
                    pad=pad,
                    blocked=blocked,
                    timeout=vf.TaskTimeout(setup=900, agent=1800, scoring=120),
                ),
                vf.TaskConfig(),
            )

    def _find_pad(self) -> Iterator[VisualTask]:
        base = FIND_PAD_EVALUATION_SEED_BASE if self.config.split == "evaluation" else 0
        for i in range(self.config.count + self.config.count % 2):
            seed = base + self.config.start_seed + i // 2
            rng = random.Random(f"find_pad_v1:{seed}")
            # This launch patch is near the validated origin; pads stay in the
            # already exercised football-field strip. Each visible pose maps to
            # BOTH locations, so its coordinates/heading cannot reveal the pad.
            start = [round(rng.uniform(-1.5, 1.5), 1), round(rng.uniform(-1.0, 1.0), 1)]
            # Heading is north for every task: the hosted model could find the
            # marker but not turn its place in the frame into world coordinates
            # at a random heading. The start point and the marker side still vary.
            heading = 0.0
            variant = (i + seed) % 2
            pad = [
                round(start[0] + FIND_PAD_FORWARD_M, 1),
                round(start[1] + (1 if variant == 0 else -1) * FIND_PAD_SIDE_M, 1),
            ]
            yield VisualTask(
                VisualData(
                    idx=i,
                    name=f"find_pad_{self.config.split}_{seed:07d}_{variant}",
                    prompt=None,
                    system_prompt=FIND_PAD_SYSTEM_PROMPT,
                    image=None,
                    family="find_pad",
                    seed=seed,
                    pad=pad,
                    blocked=False,
                    start=start,
                    heading=heading,
                    variant=variant,
                    split=self.config.split,
                    timeout=vf.TaskTimeout(setup=900, agent=1800, scoring=120),
                ),
                vf.TaskConfig(),
            )

    def _advanced(self) -> Iterator[VisualTask]:
        family, split = self.config.family, self.config.split
        size = advanced.VARIANT_COUNTS[family]
        count = ((self.config.count + size - 1) // size) * size
        base = advanced.EVALUATION_BASES[family] if split == "evaluation" else 0
        limit = advanced.turn_limit(family)
        system = SYSTEM_PROMPT.replace(f"You have {MAX_TURNS} turns.", f"You have {limit} turns.")
        system += "\nCamera yaw is relative to the drone heading; goto uses world NED. Initial heading is 0 (+X), camera pitch -70 degrees. Look toward the approach to inspect obstacles. Simulator time pauses between your actions."
        for i in range(count):
            seed = base + self.config.start_seed + i // size
            scenario = advanced.make_scenario(family, seed, split, (i + seed) % size)
            yield VisualTask(
                VisualData(
                    idx=i,
                    name=f"{family}_{split}_{seed:07d}_{scenario.variant}",
                    prompt=None,
                    system_prompt=system,
                    image=None,
                    family=family,
                    seed=seed,
                    split=split,
                    variant=scenario.variant,
                    pad=list(advanced.public_geometry(scenario)["pad_a"]),
                    blocked=False,
                    path=advanced.route_rules(scenario),
                    scenario=scenario,
                    timeout=vf.TaskTimeout(setup=900, agent=1800, scoring=120),
                ),
                vf.TaskConfig(),
            )

    def _obstacle_bypass(self) -> Iterator[VisualTask]:
        base = OBSTACLE_EVALUATION_SEED_BASE if self.config.split == "evaluation" else 0
        for i in range(self.config.count + self.config.count % 2):
            seed = base + self.config.start_seed + i // 2
            rng = random.Random(f"obstacle_bypass_v1:{seed}")
            variant = (i + seed) % 2
            yield VisualTask(
                VisualData(
                    idx=i,
                    name=f"obstacle_bypass_{self.config.split}_{seed:07d}_{variant}",
                    prompt=None,
                    system_prompt=OBSTACLE_SYSTEM_PROMPT,
                    image=None,
                    family="obstacle_bypass",
                    seed=seed,
                    pad=[22.0, 0.0],
                    blocked=False,
                    variant=variant,
                    split=self.config.split,
                    obstacle_x=round(rng.uniform(*OBSTACLE_X_RANGES[self.config.split]), 2),
                    path=route_rules(),
                    timeout=vf.TaskTimeout(setup=900, agent=1800, scoring=120),
                ),
                vf.TaskConfig(),
            )


def scene_objects(data: VisualData) -> list[dict]:
    """The pad marker and hidden white obstacles. Trusted setup fills in Z
    once the ground height is known."""
    if data.scenario is not None:
        return advanced.scene_objects(data.scenario)
    objects = [
        {
            "name": "pad_marker",
            "asset": "Cylinder",
            "x": data.pad[0],
            "y": data.pad[1],
            "z": 0.0,
            "sx": PAD_MARK_WIDTH_M,
            "sy": PAD_MARK_WIDTH_M,
            "sz": PAD_MARK_HEIGHT_M,
            "colour": PAD_MARK_COLOUR,
        }
    ]
    if data.blocked:
        objects.append(
            {
                "name": "post_on_pad",
                "asset": "Cube",
                "x": data.pad[0],
                "y": data.pad[1],
                "z": 0.0,
                "sx": POST_WIDTH_M,
                "sy": POST_WIDTH_M,
                "sz": POST_HEIGHT_M,
            }
        )
    if data.family == "obstacle_bypass":
        objects.append(barrier_object(data.obstacle_x, data.variant))
    return objects


def mission_text(data: VisualData) -> str:
    if data.scenario is not None:
        return advanced.mission_text(data.scenario)
    if data.family == "obstacle_bypass":
        return (
            "Mission: fly through the inspection checkpoint (6, 0, -2), then through the "
            "merge checkpoint (18, 0, -2), in that order, before landing on the clear pad "
            f"at x={data.pad[0]}, y={data.pad[1]}. Checkpoint coordinates are world NED metres; "
            "enter within 1.5 m of each point. Stay at or below 6 m above the ground throughout. "
            "Keep x between -2 and 24 and y between -5.5 and 5.5. While x is strictly between "
            "8 and 16, use either the corridor with y between 3 and 5.5 or the corridor with "
            "y between -5.5 and -3. A tall white barrier blocks the straight route and one "
            "of these corridors. Look at it and choose the open corridor; do not hit it or "
            "fly over it. Land within 2.5 m horizontally of the pad centre."
        )
    if data.family == "find_pad":
        return (
            "Mission: find the dark round landing-pad marker on the grass using the camera, "
            f"fly above it, and land within {PAD_RADIUS_M:g} m horizontally of its centre. "
            "The pad is clear; its location is not provided. "
            "Take off and look around if it is out of view. Landing at the launch site "
            "or outside that target radius does not complete the mission."
        )
    return (
        f"Mission: fly to the landing pad at x={data.pad[0]}, y={data.pad[1]}. Look at it. "
        "If the pad is clear, land on it. If anything is on the pad, do not land: abort and hold off."
    )


def state_text(state: dict, rest_z: float, turn: int, max_turns: int = MAX_TURNS) -> str:
    alt = rest_z - state["z"]
    return (
        f"Turn {turn + 1} of {max_turns}. Position x={state['x']:.1f} y={state['y']:.1f} z={state['z']:.1f} "
        f"(altitude {alt:.1f} m). Velocity {state['vx']:.1f}, {state['vy']:.1f}, {state['vz']:.1f} m/s. "
        f"Heading {state['yaw']:.0f} degrees. {'On the ground.' if state['landed'] else 'In the air.'}"
    )


def decision_score(
    data: VisualData,
    decision: str | None,
    state: dict | None,
    corridor_choice: str | None = None,
    action_history: list[dict] | None = None,
) -> float:
    """Score the accepted terminal or corridor choice independently of execution."""
    if data.scenario is not None:
        return advanced.decision_score(data.scenario, decision, state, action_history or [])
    if data.family == "land_or_hold":
        return float(decision == ("abort" if data.blocked else "land"))
    if data.family == "find_pad":
        if decision != "land" or state is None:
            return 0.0
        try:
            xy = [float(state["x"]), float(state["y"])]
        except (KeyError, TypeError, ValueError):
            return 0.0
        return float(all(math.isfinite(v) for v in xy) and math.dist(xy, data.pad) <= PAD_RADIUS_M)
    if data.family == "obstacle_bypass":
        return choice_score(corridor_choice, data.variant)
    raise ValueError(f"decision rule not implemented for {data.family}")


def judge(
    data: VisualData,
    ending: str | None,
    recording: list[dict],
    final: dict | None,
    event_log: dict | None = None,
    action_history: list[dict] | None = None,
) -> dict[str, float]:
    """Every rule as its own number. mission is 1 only when all of them hold."""
    out = {"coverage": 0.0, "no_collision": 0.0, "right_ending": 0.0, "at_pad": 0.0, "hold_or_touchdown": 0.0}
    if data.path is not None:
        out.update(dict.fromkeys(PATH_METRICS, 0.0))
    scenario = data.scenario
    schedule = advanced.event_schedule(scenario) if scenario is not None else None
    if scenario is not None:
        out["scenario_rules"] = 0.0
    if final is None or not recording:
        return out | {"mission": 0.0}
    try:
        rest = resting_z(recording)
    except (AttributeError, KeyError, TypeError, ValueError):
        return out | {"mission": 0.0}
    if rest is None:
        return out | {"mission": 0.0}
    if data.path is not None:
        out.update(grade_path(recording, data.path, rest, max_sample_gap_s=MAX_SAMPLE_GAP_S))
        if not out["path_recording"]:
            return out | {"mission": 0.0}
    out["coverage"] = float(covers_flight(recording, final))
    # On a clear pad a gentle touchdown on the marker is the landing. On a blocked
    # pad nothing may be touched after takeoff, not the marker, not the ground beside it.
    needs_hold = advanced.expected_ending(scenario) == "abort" if scenario is not None else data.blocked
    if needs_hold:
        out["no_collision"] = float(not collided(recording, crash_speed=-1.0))
    else:
        out["no_collision"] = float(
            not collided(recording, landing_on=("pad_marker", "alternate_pad_marker") if scenario else ("pad_marker",))
        )
    target = advanced.expected_target(scenario) if scenario is not None else data.pad
    xy_miss = math.dist([final["x"], final["y"]], target)
    target_radius = (
        advanced.HOLD_RADIUS_M
        if scenario and needs_hold
        else PAD_RADIUS_M + (HOLD_TOLERANCE_M if data.blocked else 0.0)
    )
    if needs_hold and not STRICT_HOLD_HEIGHT:
        # A drone that refuses a blocked pad backs away from it, which is what a pilot
        # does. Requiring it to end hovering on top of the thing it just refused cost
        # Claude every blocked scene, one of them by 0.2 m, after it had flown directly
        # over the pad and read it correctly. So what has to be shown is that it went and
        # looked, then held off somewhere sensible, not that it parked on the spot.
        reached = min(math.dist((r["x"], r["y"]), target) for r in recording) <= PAD_RADIUS_M
        out["at_pad"] = float(reached and xy_miss <= HOLD_STANDOFF_M)
    else:
        out["at_pad"] = float(xy_miss <= target_radius)
    if needs_hold:
        out["right_ending"] = float(ending == "abort")
        clock = "sim_t" if scenario is not None else "t"
        end_t = recording[-1].get(clock)
        if end_t is None or not isinstance(end_t, (int, float)) or not math.isfinite(end_t):
            return out | {"mission": 0.0}
        window = HOLD_S if STRICT_HOLD_HEIGHT else HOLD_SETTLED_S
        tail = [r for r in recording if isinstance(r.get(clock), (int, float)) and end_t - window <= r[clock] <= end_t]
        still = all(r["speed"] < 0.6 for r in tail)
        if STRICT_HOLD_HEIGHT:
            at_height = all(abs((rest - r["z"]) - HOLD_HEIGHT_M) <= HOLD_TOLERANCE_M for r in tail)
        else:
            at_height = all(HOLD_MIN_M <= rest - r["z"] <= HOLD_MAX_M for r in tail)
        long_enough = len(tail) >= 3 and end_t - tail[0][clock] >= window - 0.5
        at_target = scenario is None or all(math.dist((r["x"], r["y"]), target) <= target_radius for r in tail)
        out["hold_or_touchdown"] = float(bool(tail) and still and at_height and long_enough and at_target)
    else:
        out["right_ending"] = float(ending == "land")
        out["hold_or_touchdown"] = float(landed_on_ground(final, rest) and final["speed"] <= TOUCHDOWN_SPEED_MPS)
    if scenario is not None:
        extra = advanced.extra_grade(scenario, recording, final, event_log or {}, action_history or [], rest)
        out.update(extra)
        if schedule is not None:
            out.update(validate_event_log(schedule, event_log or {}, recording))
        out["scenario_rules"] = float(bool(extra) and all(value == 1.0 for value in extra.values()))
    out["mission"] = float(all(v == 1.0 for v in out.values()))
    return out


SHAPED_CAP = 0.6
SHAPED_FLOOR = 0.4


def shaped_score(scores: dict[str, float]) -> float:
    """Training reward with partial credit: 1 for a completed mission, else up to
    0.6 once the flight reached the pad, scaled by the share of the other judge
    rules that passed.

    The mission score is one pass or fail on a whole flight, so a policy that
    fails everything gets no gradient. Two gates keep the credit honest: no
    trustworthy flight record (coverage) means nothing, and no progress to
    the pad means nothing, so landing at the launch point earns zero even
    though it touches nothing and ends with the right verb. Any completed
    flight beats any failed one. Never used for evaluation."""
    if scores.get("mission", 0.0) == 1.0:
        return 1.0
    if scores.get("coverage", 0.0) != 1.0 or scores.get("at_pad", 0.0) != 1.0:
        return 0.0
    conduct = [float(v) for k, v in scores.items() if k not in ("mission", "coverage", "at_pad")]
    share = sum(conduct) / len(conduct) if conduct else 0.0
    return SHAPED_CAP * (SHAPED_FLOOR + (1.0 - SHAPED_FLOOR) * share)


class VisualEnvConfig(vf.EnvConfig):
    pilot: vf.AgentConfig = vf.AgentConfig(harness=NULL_HARNESS, runtime=SubprocessConfig())
    frames: bool = True
    """Send the camera frame each turn. Off is the frames withheld baseline, never a real run."""
    frame_budget: int = 0
    """Most frames one flight may send. 0 keeps every turn, which a 12 turn family can afford.
    A 48 turn family cannot: at about 222 tokens a frame, measured with the processor on a
    640x360 camera frame, 49 frames is roughly 11k tokens of pictures, and with the text and
    the model's own replies beside them that does not leave room in a 16k context. Frames
    cannot be dropped after the fact because the trace has already tokenised them, so the
    choice is made before each turn."""
    frame_min_move_m: float = 1.5
    """Under a budget, a frame is spent only when the view changed: the drone moved at least this
    far, or turned, or the last action pointed the camera. Hovering in place shows nothing new."""
    max_images_per_prompt: int = 16
    """How many images the inference server accepts in one prompt. Past it the request is
    refused with an upstream 400 and the flight dies partway, which reads as a bad score
    rather than a broken run. Checked before the flight starts, not discovered at turn 17."""
    shaped_reward: bool = False
    """Train on shaped_score instead of the mission score. Only for RL training sources; evaluation keeps the mission score."""
    timeout: EnvTimeoutConfig = EnvTimeoutConfig(episode=EPISODE_TIMEOUT_S)
    runtime: str = "prime"
    """Where the simulator box runs: prime or docker."""
    image: str | None = None
    cpu: float = 8
    memory: float = 16
    gpu: str = "1"
    """How many GPUs a flight wants, as a count.

    A card model like "RTX_PRO_6000" is what a prime sandbox wants, since it picks a
    matching card. A docker runtime already has whatever the box carries and refuses any
    model name it does not: the flight taskset defaulted to a model and every rollout on
    an A100 box died with Requested 'RTX_PRO_6000', but Docker exposed: NVIDIA A100.
    A count works on both, so it is the safer default. Set a model name explicitly when
    running on prime sandboxes."""
    disk: float = 40


STUCK_M = 0.5
"""Moved less than this during a whole action: treat it as not having moved at all."""


def timeout_note(before: dict, after: dict, action: dict) -> str:
    """Say why an action ran out of time, not just that it did.

    The old note said only that the action was stopped. A model told that and nothing else
    has no reason to do anything different, so it reissues the same command. One compound
    flight sent the identical goto nineteen times while pinned against a wall at x=10.2,
    drifting five centimetres a turn, and burned twenty of its forty eight turns before
    moving on. Telling it that it did not move, and that repeating will not help, is the
    cheapest thing that can break that loop.
    """
    stopped = "your last action ran past its time limit and was stopped"
    try:
        moved = math.dist((before["x"], before["y"], before["z"]), (after["x"], after["y"], after["z"]))
    except (KeyError, TypeError, ValueError):
        return stopped
    if not math.isfinite(moved):
        return stopped
    if moved < STUCK_M:
        # Do not prescribe a way out. Naming manoeuvres here was wrong: compound forbids
        # the direct line with a corridor rule and caps height at 7 m, so "climb over it"
        # told the model to break its own mission. Say what happened, say that repeating
        # will not help, and send it back to the route the mission already gave it.
        return (
            f"{stopped}. You did not move: something is in the way. Sending the same "
            "action again will not help. Your mission gives a route; go to the next "
            "point on it rather than straight to where you were heading."
        )
    short = ""
    if action.get("do") == "goto":
        try:
            left = math.dist(
                (after["x"], after["y"], after["z"]), (float(action["x"]), float(action["y"]), float(action["z"]))
            )
            if math.isfinite(left):
                short = f", {left:.1f} m short of where you aimed"
        except (KeyError, TypeError, ValueError):
            short = ""
    return f"{stopped} after moving {moved:.1f} m{short}. You are part way there; carry on or choose another route."


NO_OP_FREE_TURNS = 3
"""Repeated no-op turns refunded before the budget starts counting them again."""
NO_OP_M = 0.5
"""A goto that moved the drone less than this did nothing worth a turn."""
ARRIVED_M = 1.0
"""This close to the requested point counts as being there. The controller's final leg
flies to a 0.5 m lookahead, and the real case this was written for sat 0.25 m from its
target, so a metre is generous without being a lie."""


def no_op_note(before: dict, after: dict, action: dict) -> str:
    """Tell the model when an action succeeded and changed nothing.

    A goto to the point the drone is already at returns at once and is reported as a
    success, so nothing in the judge or the metrics notices. A flight can sit above the
    correct pad re-sending the same goto for dozens of turns, with no timed-out actions,
    and never send land.

    Not moving and being there are different things, and saying the second when only the
    first is true would be a lie the model then acts on. So the destination is checked
    before the note claims arrival; otherwise it reports the stillness and nothing more.

    Either way the note states facts and stops. It does not say what to do next, because
    that is the decision being measured.
    """
    if not action or action.get("do") != "goto":
        return ""
    try:
        moved = math.dist((before["x"], before["y"], before["z"]), (after["x"], after["y"], after["z"]))
    except (KeyError, TypeError, ValueError):
        return ""
    if not math.isfinite(moved) or moved >= NO_OP_M:
        return ""
    stayed = "your last action finished without moving the drone"
    # The caller appends how many turns are left. That is a fact the model can already
    # read off its own turn counter, so it leaks nothing, and a repeated no-op is exactly
    # when a weak model stops tracking the budget it is spending.
    try:
        left = math.dist(
            (after["x"], after["y"], after["z"]), (float(action["x"]), float(action["y"]), float(action["z"]))
        )
    except (KeyError, TypeError, ValueError):
        return f"{stayed}. Choose your next action."
    if not math.isfinite(left):
        return f"{stayed}. Choose your next action."
    if left <= ARRIVED_M:
        return f"{stayed}: you are already at the position you asked for. Choose your next action."
    return f"{stayed}, and it is still {left:.1f} m from the position you asked for. Choose your next action."


def check_image_cap(family: str, turn_limit: int, frames: bool, frame_budget: int, cap: int) -> None:
    """Refuse a run whose flights would ask for more pictures than the server accepts.

    Without a budget every turn sends a frame, so a flight asks for as many images as it
    has turns. Past the server's limit the request comes back a 400 and the flight stops
    where it is, part way, which then reads as a bad score rather than a broken run. It
    takes out every long flight in a run. Checked before the flight starts.
    """
    if not frames or cap <= 0:
        return
    most = turn_limit if frame_budget <= 0 else min(frame_budget, turn_limit)
    if most > cap:
        raise ValueError(
            f"{family} allows {turn_limit} turns and this run would send up to {most} "
            f"images, over the server's limit of {cap}. Every flight that runs long would "
            f"die on an upstream 400 part way through. Set frame_budget to {cap} or less "
            f"for this family."
        )


def spend_frame(
    budget: int, min_move_m: float, obs: dict, last: dict | None, sent: int, last_action: dict | None
) -> bool:
    """Is this turn worth one of the flight's frames?

    A frame cannot be taken back: the trace has already tokenised it, so a long flight that
    sends one every turn runs past the trainer's sequence length. Under a budget a frame is
    spent only when the view changed, and the opening view always goes."""
    if budget <= 0:
        return True
    if last is None:
        return True
    if sent >= budget:
        return False
    if last_action is not None and last_action.get("do") == "look":
        return True
    state = obs.get("state") or {}
    moved = math.dist(
        [state.get("x", 0.0), state.get("y", 0.0), state.get("z", 0.0)],
        [last.get("x", 0.0), last.get("y", 0.0), last.get("z", 0.0)],
    )
    turned = abs(float(state.get("yaw", 0.0)) - float(last.get("yaw", 0.0)))
    turned = min(turned, 360.0 - turned)
    return moved >= min_move_m or turned >= 20.0


def jpeg_data_url(png_b64: str, quality: int = FRAME_JPEG_QUALITY) -> str:
    """The controller's PNG frame as a JPEG data URL, about a tenth of the size.

    Bytes that are not an image go through unchanged as PNG, so a conversion
    problem never costs the model its frame."""
    import base64
    import io

    try:
        from PIL import Image

        image = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return "data:image/png;base64," + png_b64


def docker_cpu_limit() -> float:
    """CPUs a local docker box may ask for: the host's count, at least one."""
    import os

    return float(max(1, os.cpu_count() or 1))


class VisualFlightEnv(vf.Env[VisualEnvConfig]):
    _boot_gate = asyncio.Lock()
    _last_boot = 0.0

    async def _stagger_boot(self) -> None:
        """Let one simulator finish its noisiest moment before the next one starts.

        Six simulators launched together on one card lose several to
        VK_ERROR_OUT_OF_DEVICE_MEMORY, and the same six staggered all start. A running
        simulator costs about 2.1 GB of GPU memory; starting one costs far more for a few
        seconds, so overlapping cold starts are what exhausts the card, not the simulators
        themselves.

        This only spaces out the launches. It does not serialise them: the wait is a couple
        of seconds against a 30 second boot, so concurrency is barely affected.

        **The spacing holds within one taskset source, not across them.** The lock lives in
        this process, and prime-rl gives every eval source its own env server on its own
        port, each reporting workers=1. So a run with N eval sources can still have N cold
        starts land together, one per source, each source spacing its own. With two sources
        that is two at once rather than fourteen, which is well inside what a card they have
        to themselves can take. If it ever needs to hold across sources it wants a lock on
        the box, not in here.
        """
        if SIM_BOOT_STAGGER_S <= 0:
            return
        async with VisualFlightEnv._boot_gate:
            now = time.monotonic()
            wait = SIM_BOOT_STAGGER_S - (now - VisualFlightEnv._last_boot)
            if wait > 0:
                await asyncio.sleep(wait)
            VisualFlightEnv._last_boot = time.monotonic()

    @asynccontextmanager
    async def _episode_runtime(self, advanced_mission: bool):
        await self._stagger_boot()
        async with provision_runtime(self._sim_config()) as box:
            try:
                yield box
            finally:
                if advanced_mission:
                    # Also reached when model inference, startup, or scoring raises.
                    # Runtime teardown remains the final process/container cleanup.
                    for command in (STOP_EVENTS, STOP_RECORDER, STOP_CONTROLLER):
                        try:
                            await asyncio.wait_for(box.run(["bash", "-c", command], {}), timeout=10.0)
                        except Exception:
                            pass

    def _sim_config(self):
        fields = {
            "image": self.config.image or IMAGE,
            "cpu": self.config.cpu,
            "memory": self.config.memory,
            "gpu": self.config.gpu,
            "disk": self.config.disk,
            "allow": [],
            "block": ["*"],
        }
        if self.config.runtime == "docker":
            # A local box cannot ask for more CPUs than the host has; docker refuses the run.
            fields["cpu"] = min(fields["cpu"], docker_cpu_limit())
            return DockerConfig(**fields)
        return PrimeConfig(vm=True, **fields)

    async def _wait_sim(self, box) -> None:
        result = await box.run(["python3", "-c", CHECK_SIM], {})
        if result.exit_code == 0:
            return
        # One restart if the first boot does not answer: the first boot on a fresh
        # GPU can stall while shaders build.
        for attempt in range(2):
            if attempt:
                # Order matters, and it is the only order that separates the attempts.
                # Save the first attempt's logs before anything can destroy them; stop the
                # old game and wait for it to actually be gone; only then clear the live
                # logs. Clearing before the kill would let a game that is alive but silent
                # write again afterwards, and the next attempt would read that as its own
                # launch and be called dead while it was still preparing.
                await box.run(["bash", "-c", SAVE_SIM_LOGS], {})
                await box.run(["bash", "-c", STOP_SIM], {})
                await box.run(["bash", "-c", CLEAR_SIM_LOGS], {})
            await box.run(["bash", "-c", "nohup /usr/local/bin/start_sim.sh > /tmp/start_sim.out 2>&1 &"], {})
            for poll in range(SIM_BOOT_TRIES):
                result = await box.run(["python3", "-c", CHECK_SIM], {})
                if result.exit_code == 0:
                    return
                # A boot that works answers in about 30 seconds. One that fails never
                # answers, and the container stays up regardless because start_sim.sh ends
                # in sleep infinity.
                #
                # Only one of the three answers stops the wait: the game was launched and
                # is now gone. A game still running, or one not launched yet because
                # start_sim.sh is still installing NVIDIA userspace, both keep waiting the
                # full budget. So this bounds a crashed simulator, not a hung one, and a
                # process that is alive but silent still costs the whole 450 seconds.
                if poll >= SIM_DEAD_AFTER_POLLS:
                    alive = await box.run(["bash", "-c", SIM_ALIVE], {})
                    if alive.exit_code == 1:
                        break
                await asyncio.sleep(SIM_BOOT_WAIT_S)
        raise RuntimeError("simulator did not come up in the box:\n%s" % await self._sim_diagnostics(box))

    async def _sim_diagnostics(self, box) -> str:
        """What the box can tell us about a simulator that would not start.

        The simulator writes /tmp/airsim.log and start_sim.sh writes /tmp/start_sim.out.
        Both survive a failed startup, and throwing them away with the exception leaves no
        evidence of why it failed.
        """
        try:
            result = await asyncio.wait_for(box.run(["bash", "-c", SIM_DIAGNOSTICS], {}), timeout=60.0)
        except (asyncio.TimeoutError, OSError, RuntimeError) as exc:
            return "could not read diagnostics off the box: %r" % (exc,)
        text = (getattr(result, "stdout", "") or "") + (getattr(result, "stderr", "") or "")
        return text.strip()[-4000:] or "the box returned nothing"

    async def _control(self, box, counter: list[int], spec: dict) -> dict:
        """One request to the controller in the box. Raises when the controller is broken."""
        n = counter[0]
        counter[0] += 1
        await box.write(f"{CONTROL_DIR}/req_{n}.tmp", json.dumps(spec).encode())
        result = await box.run(["bash", "-c", wait_for_result(n, ACTION_TIMEOUT_S)], {})
        if result.exit_code == 3:
            raise TimeoutError(f"no answer to {spec['do']} in {ACTION_TIMEOUT_S:.0f} s")
        if result.exit_code != 0:
            raise RuntimeError(f"controller failed on {spec['do']}: {result.stderr[-300:]}")
        out = json.loads(result.stdout.strip().splitlines()[-1])
        if "error" in out:
            raise RuntimeError(f"controller failed on {spec['do']}: {out['error']}")
        return out

    async def _act(self, box, counter: list[int], spec: dict) -> dict | None:
        """Run one action. None means the controller stopped answering and the episode must end."""
        try:
            return await asyncio.wait_for(
                self._control(box, counter, {**spec, "limit": ACTION_LIMIT_IN_BOX_S}), timeout=ACTION_TIMEOUT_S + 30.0
            )
        except (asyncio.TimeoutError, TimeoutError, RuntimeError):
            return None

    def _spend_frame(self, obs: dict, last: dict | None, sent: int, last_action: dict | None) -> bool:
        return spend_frame(self.config.frame_budget, self.config.frame_min_move_m, obs, last, sent, last_action)

    def _tool_contract(self) -> str:
        """What an agent with tools has to be told, and one without does not.

        The mission prompt says to answer with one JSON action and nothing else. A tool
        harness says the opposite by handing the model a shell. Given both, a 4B model
        reaches for the shell and never answers: 133 flights ran their whole turn budget
        on tool calls, wrote nothing, flew nothing and scored zero. So when there are
        tools, say plainly what they are for and what actually moves the drone.
        """
        harness = getattr(getattr(self.config.pilot, "harness", None), "id", None)
        if harness in (None, "", "null"):
            return ""
        return (
            "You have shell tools as well. They run in your own box, which is neither the "
            "drone nor the simulator, so nothing you run there moves the drone. Use them "
            "only when working something out helps, for example arithmetic on coordinates. "
            "The drone moves when, and only when, your reply ends with one JSON action, so "
            "end every turn with that action.\n\n"
        )

    def _message(
        self,
        data: VisualData,
        obs: dict,
        rest_z: float,
        turn: int,
        note: str,
        with_frame: bool = True,
        rejected: bool = True,
    ) -> list:
        """One turn's observation, and whatever has to be said about the last reply.

        `rejected` separates two things that used to share a sentence. A reply the
        environment could not parse was not used, and saying so is true. An action that ran
        and moved nothing was used: the drone did exactly what was asked and was already
        there. Telling the model that reply was not used is false, and a model that believes
        it sends the same action again. Measured on 2026-10-20: every land_or_hold flight
        reached the pad and then repeated one goto until the turns ran out, with this
        sentence in front of it each time.
        """
        limit = advanced.turn_limit(data.family) if data.scenario is not None else MAX_TURNS
        text = state_text(obs["state"], rest_z, turn, limit)
        if turn == 0:
            text = self._tool_contract() + mission_text(data) + "\n" + text
        if note:
            left = limit - turn - 1
            budget = f" {left} turn{'s' if left != 1 else ''} left." if left >= 0 else ""
            lead = "Your last reply was not used: " if rejected else "Your last action ran: "
            text = f"{lead}{note}{budget}\n" + text
            if self._tool_contract():
                text = "End this turn with one JSON action. Tools do not move the drone.\n" + text
        parts: list = [{"type": "text", "text": text}]
        if self.config.frames and with_frame and obs.get("frame"):
            parts.append({"type": "image_url", "image_url": {"url": jpeg_data_url(obs["frame"])}})
        return [{"role": "user", "content": parts}]

    async def run(self, task, agents) -> None:
        data: VisualData = task.data
        spawn = scene_objects(data)
        scenario = data.scenario
        schedule = advanced.event_schedule(scenario) if scenario is not None else None
        limit = advanced.turn_limit(data.family) if scenario is not None else MAX_TURNS
        check_image_cap(
            data.family, limit, self.config.frames, self.config.frame_budget, self.config.max_images_per_prompt
        )
        action_history: list[dict] = []
        event_log: dict = {}
        event_failed = False
        ending: str | None = None
        decision: str | None = None
        decision_state: dict | None = None
        corridor_choice: str | None = None
        turns = 0
        rejected = 0
        stalled = False
        timed_out_actions = 0
        no_op_actions = 0
        no_op_refunds = 0
        last_no_op_action = None
        refund_turn = False
        refunded_frame = False
        ran_out = False
        flight_seconds = 0.0
        frames_sent = 0
        last_frame_state: dict | None = None
        camera_failed = False
        recording: list[dict] = []
        final: dict | None = None
        async with self._episode_runtime(scenario is not None) as box:
            await self._wait_sim(box)
            controller = advanced_controller(CONTROLLER_SOURCE) if scenario is not None else CONTROLLER_SOURCE
            recorder = advanced_recorder(RECORDER_SOURCE) if scenario is not None else RECORDER_SOURCE
            await box.write("/tmp/visual_controller.py", controller.encode())
            await box.write("/tmp/record_flight.py", recorder.encode())
            result = await box.run(["python3", "-c", RESET_SIM], {})
            if result.exit_code != 0:
                raise RuntimeError(f"could not reset the drone: {result.stderr[-200:]}")
            await asyncio.sleep(2.0)
            if data.start is not None:
                # The placed drone sometimes takes longer than the settle window to
                # come to rest; a second attempt from the settled state succeeds.
                for attempt in range(PLACE_START_TRIES):
                    if attempt:
                        # Without a reset the drone is still wherever the failed attempt
                        # left it, usually falling, so every retry reads the same bad
                        # resting height and fails the same way.
                        await box.run(["python3", "-c", RESET_SIM], {})
                        await asyncio.sleep(2.0)
                    placed = await asyncio.wait_for(
                        box.run(
                            [
                                "python3",
                                "-c",
                                PLACE_START_SOURCE,
                                json.dumps({"start": data.start, "heading": data.heading}),
                            ],
                            {},
                        ),
                        timeout=30.0,
                    )
                    if placed.exit_code == 0:
                        break
                    if attempt == PLACE_START_TRIES - 1:
                        raise RuntimeError(f"could not place the launch pose: {placed.stderr[-300:]}")
                    await asyncio.sleep(2.0)
            counter = [0]
            started = await box.run(["bash", "-c", START_CONTROLLER], {})
            if started.exit_code != 0:
                raise RuntimeError("flight controller did not start")
            obs = await self._control(box, counter, {"do": "setup", "pitch": CAMERA_PITCH_DEG, "spawn": []})
            # The controller measures ground level; a drone sunk into the park at this
            # instant would otherwise put every spawned object 7.5 m under it too.
            rest_z = obs.get("rest_z")
            if rest_z is None:
                rest_z = obs["state"]["z"]
            for obj in spawn:
                obj["z"] = rest_z - obj["sz"] / 2
            obs = await self._control(box, counter, {"do": "setup", "pitch": CAMERA_PITCH_DEG, "spawn": spawn})
            started = await box.run(["bash", "-c", START_RECORDER], {})
            if started.exit_code != 0:
                raise RuntimeError("flight recorder did not start")
            if schedule is not None:
                await box.write("/tmp/flight_events.py", EVENT_SOURCE.encode())
                await box.write(
                    EVENT_CONFIG_PATH,
                    json.dumps({"schedule": schedule.model_dump(mode="json"), "rest_z": rest_z}).encode(),
                )
                started = await asyncio.wait_for(box.run(["bash", "-c", START_EVENTS], {}), timeout=30.0)
                if started.exit_code != 0:
                    raise RuntimeError("simulator event worker did not start")
                obs = await self._control(box, counter, {"do": "observe", "events_enabled": True})
            note = ""
            async with agents.pilot.interaction(task) as interaction:
                if not obs.get("frame"):
                    camera_failed = True
                    ending = "camera_failure"
                    segment = None
                else:
                    last_frame_state = dict(obs.get("state") or {})
                    frames_sent += 1
                    segment = await interaction.turn(self._message(data, obs, rest_z, 0, ""))
                while segment is not None and not segment.terminated and turns < limit:
                    action, err = parse_action(segment.last_reply)
                    if action is not None and leg_too_long(action, obs["state"]):
                        action, err = None, f"goto legs are at most {MAX_LEG_M:.0f} m"
                    if action is None:
                        rejected += 1
                        note = err
                        note_rejected = True
                    else:
                        note = ""
                        note_rejected = False
                        before_state = dict(obs["state"])
                        if scenario is not None:
                            action_history.append(
                                {
                                    "action": dict(action),
                                    "state": dict(obs["state"]),
                                    "sim_t": obs.get("sim_t", obs["state"].get("sim_t")),
                                }
                            )
                        if data.family == "obstacle_bypass" and corridor_choice is None:
                            corridor_choice = classify_route_choice(obs["state"], action)
                        if action["do"] in ("land", "abort"):
                            decision = action["do"]
                            decision_state = dict(obs["state"])
                        if action["do"] == "abort":
                            action = {**action, "hold_z": rest_z - HOLD_HEIGHT_M, "hold_s": HOLD_S}
                        result = await self._act(box, counter, action)
                        if result is None:
                            ending = "stall"
                            stalled = True
                        else:
                            obs = result
                            flight_seconds += float(obs.get("seconds", 0.0))
                            if not obs.get("frame"):
                                camera_failed = True
                                ending = "camera_failure"
                            if obs.get("timed_out"):
                                note = timeout_note(before_state, obs.get("state") or {}, action)
                                timed_out_actions += 1
                            else:
                                note = no_op_note(before_state, obs.get("state") or {}, action)
                                if note:
                                    no_op_actions += 1
                                    # A turn that moved nothing and repeated the action
                                    # that moved nothing does not come out of the budget.
                                    # Measured on 2026-10-19: a land_or_hold flight
                                    # reached the pad on turn 2, looked once, then re-sent
                                    # the same goto for ten turns and ran out without ever
                                    # landing or aborting, so the flight was never graded
                                    # at all. The environment already told it each time
                                    # and it repeated anyway. Refusing the turn bounds the
                                    # waste without telling it what to choose, which is
                                    # the decision being measured. NO_OP_FREE_TURNS caps
                                    # the refunds so a model that never varies still ends.
                                    if action == last_no_op_action and no_op_refunds < NO_OP_FREE_TURNS:
                                        no_op_refunds += 1
                                        refund_turn = True
                                    last_no_op_action = action
                                else:
                                    last_no_op_action = None
                            if action["do"] in ("land", "abort"):
                                ending = action["do"]
                    if refund_turn:
                        refund_turn = False
                        # A refunded turn does not spend the budget, and it must not spend
                        # a frame either. The drone did not move, so the view is the one the
                        # model already has, and sending it again only grows the context:
                        # three refunds pushed a 12 turn flight past the 16384 token limit
                        # and the model call failed with "input length exceeds".
                        refunded_frame = True
                    else:
                        turns += 1
                    if ending is not None:
                        break
                    send_frame = (not refunded_frame) and self._spend_frame(obs, last_frame_state, frames_sent, action)
                    refunded_frame = False
                    if send_frame:
                        frames_sent += 1
                        last_frame_state = dict(obs.get("state") or {})
                    segment = await interaction.turn(
                        self._message(data, obs, rest_z, turns, note, send_frame, note_rejected)
                    )
                if ending is None and turns >= limit:
                    # Out of turns without ever landing or aborting. Three of the seven
                    # families end this way almost every flight: measured on 2026-10-20,
                    # compound ran to its 48 turn cap on every rollout, go_around on all
                    # but one, and land_or_hold on all sixteen. The model reaches the pad
                    # and then repeats an action that moves nothing until the budget is
                    # gone.
                    #
                    # Scoring nothing at all is the worst answer. The flight happened, the
                    # drone is somewhere, and the recording says where, so grade that. The
                    # drone is held off the ground the way an abort holds it, which is what
                    # running out of turns amounts to: the mission was not completed and
                    # the drone did not commit to the ground.
                    #
                    # It is the environment's choice, not the model's, so the flight cannot
                    # pass right_ending or decision_correct. What it can earn is a
                    # trustworthy recording, a clean route and an honest position, which is
                    # a gradient where there was none.
                    ran_out = True
                    # Not "abort". right_ending is float(ending == "abort"), so calling this
                    # an abort hands a free pass on every blocked scene, where aborting is
                    # the correct answer. Measured on 2026-10-20: four of eight land_or_hold
                    # flights ran out of turns, were stopped by this code, and scored the
                    # mission with decision_correct at zero. That is worse than scoring
                    # nothing, because it teaches a policy that running out the clock wins
                    # half the scenes. A flight that never decided cannot have decided
                    # correctly, so it gets its own ending that no rule reads as a choice.
                    ending = "ran_out"
                    forced = {"do": "abort", "hold_z": rest_z - HOLD_HEIGHT_M, "hold_s": HOLD_S}
                    result = await self._act(box, counter, forced)
                    if result is None:
                        ending = "stall"
                        stalled = True
                    else:
                        obs = result
                        flight_seconds += float(obs.get("seconds", 0.0))
                await asyncio.sleep(1.0)
                if schedule is not None:
                    try:
                        stopped = await asyncio.wait_for(box.run(["bash", "-c", STOP_EVENTS], {}), timeout=30.0)
                        event_log = parse_event_log(await box.read(EVENT_LOG_PATH, max_bytes=MAX_EVENT_BYTES))
                        event_failed = stopped.exit_code != 0 or bool(event_log.get("error"))
                    except Exception:
                        event_failed = True
                        event_log = {"error": "event worker log unavailable"}
                try:
                    await asyncio.wait_for(box.run(["bash", "-c", STOP_RECORDER], {}), timeout=30.0)
                    await asyncio.wait_for(box.run(["bash", "-c", STOP_CONTROLLER], {}), timeout=30.0)
                    recording = parse_recording(await box.read(RECORDING_PATH, max_bytes=MAX_RECORDING_BYTES))
                except Exception:  # noqa: BLE001
                    recording = []
                if stalled or camera_failed:
                    recording = []
                final = obs["state"] | {
                    "speed": math.dist([0, 0, 0], [obs["state"]["vx"], obs["state"]["vy"], obs["state"]["vz"]])
                }
                scores = judge(data, ending, recording, final, event_log, action_history)
                if event_failed:
                    scores["mission"] = 0.0
                trace = interaction.trace
                if self.config.shaped_reward:
                    # The mission score stays on the record with no weight; the shaped score is the reward.
                    trace.record_reward("mission", scores["mission"], 0.0)
                    trace.record_reward("shaped", shaped_score(scores), 1.0)
                else:
                    trace.record_reward("mission", scores["mission"], 1.0)
                trace.record_metrics({k: v for k, v in scores.items() if k != "mission"})
                trace.record_metrics(
                    {
                        # A correct choice can still fail to complete the flight.
                        # Keep this independent of recorder/controller validity.
                        "decision_correct": decision_score(
                            data, decision, decision_state, corridor_choice, action_history
                        ),
                        "turns": float(turns),
                        "rejected_replies": float(rejected),
                        "timed_out_actions": float(timed_out_actions),
                        # A goto that changed nothing. Invisible before: it returns a
                        # success, so timed_out_actions stays 0 while turns drain away.
                        "no_op_actions": float(no_op_actions),
                        "no_op_refunds": float(no_op_refunds),
                        "ran_out_of_turns": float(ran_out),
                        "flight_seconds": float(flight_seconds),
                        "stalled": float(stalled),
                        "camera_failed": float(camera_failed),
                        "frames_sent": float(frames_sent),
                        "frames_withheld": float(not self.config.frames),
                        "blocked": float(data.blocked),
                        "path_length_m": path_length(recording) if recording else -1.0,
                        "max_altitude_m": max_altitude(recording, rest_z) if recording else -1.0,
                    }
                )
                trace.info["ending"] = ending or "none"
                trace.info["decision"] = decision or "none"
                trace.info["decision_state"] = decision_state
                if scenario is not None:
                    trace.info["action_history"] = action_history
                    trace.info["event_log"] = event_log
                    trace.record_metrics(
                        {
                            "event_failed": float(event_failed),
                            "scenario_obstacle_hit": float(advanced.scenario_obstacle_hit(recording)),
                        }
                    )
                if data.family == "obstacle_bypass":
                    trace.info["route_choice"] = corridor_choice or "none"
                    trace.record_metrics({"route_barrier_hit": float(barrier_hit(recording))})
                trace.info["pad"] = data.pad
                trace.info["recording_samples"] = len(recording)
                trace.info["recording"] = recording
                trace.info["final"] = final


__all__ = ["VisualFlightEnv", "VisualTaskset", "judge", "parse_action", "scene_objects", "wait_for_result"]

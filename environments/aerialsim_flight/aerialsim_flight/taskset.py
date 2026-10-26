"""Flight missions scored by re-running the submitted program on a fresh simulator."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.runtimes import (
    DockerConfig,
    DockerRuntime,
    PrimeConfig,
    PrimeRuntime,
    provision_runtime,
)

IMAGE = "prime/nahidalam/aerialsim-city-park:v3"
# A tag is mutable: whoever owns the image can repush v3 over a different world, and a
# graded flight would silently come to mean something else. What the registry records about
# the build that was graded is written down here, from
# `prime images list --search aerialsim-city-park --output json`, whose table output omits
# these fields.
#
# Neither id nor path is known to be a content hash. Whether a repush changes them has not
# been tested, so treat them as identifiers of this registry row rather than a pin on the
# world inside it. `rootfs-cas` hints at content-addressed storage, which is a hint and not
# evidence. If a content digest becomes available, pin that instead.
IMAGE_ID = "g620vp0zyjx738esnz3gkblz"
IMAGE_PATH = "cmowyasi8001x12c336jqujs6/mnnu0efzhl9gjzm3hihybk8c/linux-amd64/rootfs-cas"
IMAGE_SIZE_BYTES = 3_455_645_857
IMAGE_PUSHED_UTC = "2026-08-30T03:44:55Z"
SIM_BOOT_TRIES = 90
SIM_BOOT_WAIT_S = 5.0
SUBMIT_PATH = "/app/flight.py"
MAX_PROGRAM_BYTES = 200_000
FLIGHT_TIMEOUT_S = 240.0
SIM_PORT_DEFAULT = 41451
GRADE_CACHE_KEEP = 8

MISSIONS = [
    {
        "name": "hover_at_height",
        "kind": "hover",
        "target": [0.0, 0.0, 10.0],
        "text": "Take off and hover 10 meters above the ground.",
    },
    {
        "name": "reach_position",
        "kind": "reach",
        "target": [25.0, 10.0, 8.0],
        "text": "Fly to position x=25, y=10 at 8 meters height and hover there.",
    },
    {
        "name": "reach_and_land",
        "kind": "land",
        "target": [20.0, -5.0, 0.0],
        "text": "Fly to position x=20, y=-5 and land there.",
    },
]

PROMPT = """You control a drone in a photoreal simulator through Python.

In python3:

    from aerialsim_flight.drone import connect
    d = connect()
    d.takeoff(height=8)
    d.goto(20, 0, -8, speed=3)
    d.state()
    d.camera()
    d.land()

Positions are NED: x north, y east, z down. The ground is not at z = 0:
in this world it sits about 1.8 meters below it. So read heights from
d.state()["altitude"], which is meters above the ground, and d.takeoff
takes a height above the ground too. d.state()["ground_z"] gives the
ground in simulator z, so 10 meters up is z = ground_z - 10.

Any height a mission asks for is a height above the ground.

Also available: velocity, attitude, rates, hover, path, camera_control.
Each call blocks until the maneuver finishes. Use d.state() to check
where you are.

You must write your flight program to /app/flight.py before you finish.
That file is the only thing scored: it is re-run on a fresh simulator
and the whole flight there sets your reward. The drone must finish the
mission without hitting anything, and a landing means wheels on the
ground, not a hover. Flying in your own sandbox is practice and earns
nothing. If /app/flight.py is missing or does not fly, you score zero.

The graded program can import aerialsim_flight, math, and time. Other
imports are rejected.

Mission: {text}
"""

READ_STATE = (
    "import json, cosysairsim as a\n"
    "c = a.MultirotorClient()\n"
    "c.confirmConnection()\n"
    "s = c.getMultirotorState()\n"
    "k = s.kinematics_estimated\n"
    "v = k.linear_velocity\n"
    "print(json.dumps({'x': k.position.x_val, 'y': k.position.y_val,"
    " 'z': k.position.z_val,"
    " 'landed': s.landed_state == a.LandedState.Landed,"
    " 'speed': (v.x_val**2 + v.y_val**2 + v.z_val**2) ** 0.5}))\n"
)

RESET_SIM = "import cosysairsim as a\nc = a.MultirotorClient()\nc.confirmConnection()\nc.reset()\n"

# Samples the drone ten times a second for the whole graded flight.
RECORDER_SOURCE = """import json, sys, time, cosysairsim as a
c = a.MultirotorClient()
c.confirmConnection()
out = open(sys.argv[1], "w")
t0 = time.time()
while True:
    try:
        s = c.getMultirotorState()
        k = s.kinematics_estimated
        v = k.linear_velocity
        ci = c.simGetCollisionInfo()
        out.write(json.dumps({"t": round(time.time() - t0, 2), "x": k.position.x_val,
            "y": k.position.y_val, "z": k.position.z_val,
            "speed": (v.x_val**2 + v.y_val**2 + v.z_val**2) ** 0.5,
            "collided": bool(ci.has_collided), "collision_t": float(ci.time_stamp), "hit": str(ci.object_name),
            "landed": s.landed_state == a.LandedState.Landed}) + "\\n")
        out.flush()
    except Exception as exc:
        print("sample failed:", repr(exc), file=sys.stderr, flush=True)
    time.sleep(0.1)
"""
RECORDING_PATH = "/tmp/recording.jsonl"
MAX_RECORDING_BYTES = 4_000_000
LANDING_TOLERANCE_M = 0.3
TOUCHDOWN_SPEED_MPS = 0.5
"""Slow enough at ground level to count as down.

One number, so the landing rule and the touchdown rule cannot drift apart. It is not the
settling speed used to find ground level: that one detects a drone sitting perfectly
still, and it is too strict here, failing a drone already resting on the grass."""
CRASH_SPEED_MPS = 2.0
SETTLE_SPEED_MPS = 0.3
SETTLE_WINDOW_S = 3.0
TAKEOFF_CLEARANCE_M = 1.0
TAKEOFF_SETTLE_M = 3.0
"""How high the drone has to get before a ground touch counts as a crash.

The simulator drops the drone out of its own takeoff: it climbs to about 1.5 m, where
takeoffAsync hands the climb over, falls back to the ground at around 2.3 m/s, and then
flies the rest of the mission normally. That is 0.3 m/s over the crash limit, so the
flight failed no_collision and scored zero however well it was flown. The drone did
nothing wrong and could do nothing about it.

The rate is hardware sensitive, from a few percent of flights on one machine to almost
all of them on another, so it can wipe out a whole run.

Three metres is above every drop recorded, 142 of 144 of which peaked between 1.50 and
1.60 m, and far below the 5 to 8 m these missions cruise at, so a dive out of cruise
still counts."""
TAKEOFF_BOUNCE_SPEED_MPS = 4.0
"""Fastest a touch may be and still be read as the takeoff drop. Falling 3 m reaches
about 7.7 m/s, so this keeps a genuinely hard impact during takeoff counted."""
MAX_SAMPLE_GAP_S = 1.0
END_MATCH_M = 1.0
START_RECORDER = (
    f"rm -f {RECORDING_PATH}; nohup /usr/bin/python3 /tmp/record_flight.py {RECORDING_PATH} "
    "> /tmp/recorder.log 2>&1 & echo $! > /tmp/recorder.pid; sleep 0.5; test -s /tmp/recorder.pid"
)
STOP_RECORDER = "kill $(cat /tmp/recorder.pid) 2>/dev/null; sleep 0.3; true"

CHECK_SIM = "import cosysairsim as a; a.MultirotorClient().confirmConnection()"

AIRSIM_LOG = "/tmp/airsim.log"
"""Where the launcher writes. Its existence is what separates a simulator that has not
been launched yet from one that was launched and died."""
START_SIM_OUT = "/tmp/start_sim.out"

LAUNCHER_PATTERN = "[s]tart_sim.sh"
"""The launcher script's command line, bracketed for the same reason as SIM_PATTERN."""

SIM_PATTERN = "[C]ityPark.*(Shipping|Development|-RenderOffScreen)"
"""The game's command line, bracketed so the pattern never matches the shell running it.

`pgrep -f` and `pkill -f` read full command lines, and on Linux `pgrep -a` lists them, so an
unbracketed pattern finds the very process doing the search. `pkill -f CityPark` could kill
its own shell part way through a restart sequence. The regex `[C]ityPark` matches CityPark
but not the literal text "[C]ityPark", so the searcher stops finding itself."""

SIM_ALIVE = rf"""
running=$(pgrep -f '{SIM_PATTERN}' 2>/dev/null | grep -cv '^$' || true)
[ "${{running:-0}}" -gt 0 ] && exit 0
[ -s {AIRSIM_LOG} ] || exit 2
exit 1
"""
"""Is the simulator alive, dead, or not started yet?

start_sim.sh ends in `sleep infinity`, so the container stays up whether or not the
simulator lives, and without this check the outer poll waits its whole budget on a corpse.
A boot that works answers in about 30 seconds. A boot that fails never answers, so the
wait is pure waste and a run of failures costs simulator-hours."""


SAVE_SIM_LOGS = rf"""
for f in {AIRSIM_LOG} {START_SIM_OUT}; do
    [ -s "$f" ] && cp -f "$f" "$f.attempt1" 2>/dev/null
done
true
"""
"""Keep the first attempt's evidence before anything can destroy it.

Copy only. Clearing is a separate step that happens after the first game is actually dead,
because a game that is alive but silent can write to its log between a clear and a kill,
and the next attempt would read that line as its own launch.

Relaunching truncates the launcher's output and the game truncates its own log, so without
this the only logs that survive a restart are the second attempt's, and the cause written
during the first is gone."""

STOP_SIM = rf"""
pkill -f '{LAUNCHER_PATTERN}' 2>/dev/null || true
pkill -f '{SIM_PATTERN}' 2>/dev/null || true
for _ in $(seq 1 20); do
    pgrep -f '{SIM_PATTERN}' >/dev/null 2>&1 && {{ sleep 0.5; continue; }}
    pgrep -f '{LAUNCHER_PATTERN}' >/dev/null 2>&1 && {{ sleep 0.5; continue; }}
    exit 0
done
pkill -9 -f '{LAUNCHER_PATTERN}' 2>/dev/null || true
pkill -9 -f '{SIM_PATTERN}' 2>/dev/null || true
sleep 1
true
"""
"""Stop the whole first attempt and wait for it to be gone.

**The body carries no comments on purpose.** `bash -c` puts the entire script on the
shell's own command line, and `pkill -f` searches command lines, so any prose in the body
that a pattern matches makes the script kill the shell running it. The bracketing is not
enough: an earlier version explained itself in comments that contained the literal text
"start_sim.sh", which `[s]tart_sim.sh` matches, so it would have killed itself on the first
line. Everything explanatory lives here instead, and a test asserts that no pattern in any
of these scripts matches the script it sits in.

The launcher has to go too, not just the game. The launcher script starts the game near its
top and then polls for ten minutes before ending in a sleep, so it always outlives its game.
That is harmless once it has launched. It is not harmless if the first attempt died while
still installing NVIDIA userspace: its launch line is still ahead of it, so it would start a
second game beside the second attempt's, both fighting for port 41451, and write to the log
the next alive check reads.

An orphaned wget from that install may survive this. That is harmless here: it writes only
to /tmp/nvidia.run, binds no port and touches neither log."""

CLEAR_SIM_LOGS = rf"""
: > {AIRSIM_LOG} 2>/dev/null || true
: > {START_SIM_OUT} 2>/dev/null || true
true
"""
"""Clear the live logs immediately before relaunching, once the old game is dead.

This has to happen, because the game truncates its log when it launches, not when the
launcher starts, and the launcher can spend minutes installing NVIDIA userspace first.
Leaving the first attempt's log in place would leave a non-empty file sitting there, and the
alive check reads a non-empty log as "the game launched". It would answer "launched and
died" while the second attempt was still preparing.

And it has to happen after the stop, not before, or a game that is alive but silent can
write again between the clear and the kill."""

SIM_DIAGNOSTICS = rf"""
echo "--- game process"
pgrep -af '{SIM_PATTERN}' || echo "(no game process)"
echo "--- port 41451"
(echo > /dev/tcp/localhost/41451) 2>/dev/null && echo open || echo closed
echo "--- tail of /tmp/airsim.log"
tail -n 25 /tmp/airsim.log 2>/dev/null || echo "(no airsim.log)"
echo "--- tail of /tmp/start_sim.out"
tail -n 15 /tmp/start_sim.out 2>/dev/null || echo "(no start_sim.out)"
echo "--- first attempt, kept before the restart overwrote these"
tail -n 25 /tmp/airsim.log.attempt1 2>/dev/null || echo "(none)"
tail -n 15 /tmp/start_sim.out.attempt1 2>/dev/null || echo "(none)"
echo "--- memory"
free -m 2>/dev/null | head -2
echo "--- gpu"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
echo "--- disk"
df -h /tmp 2>/dev/null | tail -1
"""
"""What to read off the box when the simulator will not start.

The simulator already writes /tmp/airsim.log and start_sim.sh writes /tmp/start_sim.out.
Discarding them on failure leaves no evidence of why. Nothing here is new
instrumentation; it is reading what was already there."""

EXCUSE_TAKEOFF_BOUNCE = True
"""Whether the takeoff drop is excused. Set false for the older rule, which is the only way
to compare against runs scored under it."""

GATE_SOURCE = (Path(__file__).with_name("grade_flight.py")).read_bytes()
PROXY_SOURCE = (Path(__file__).with_name("rpc_proxy.py")).read_bytes()
PROXY_PORT = 41452
FLIGHT_USER = "flight_user"

START_PROXY = f"""set -e
id -u {FLIGHT_USER} >/dev/null 2>&1 || useradd -M -s /usr/sbin/nologin {FLIGHT_USER}
chmod 644 /tmp/flight.py /tmp/grade_flight.py /tmp/rpc_proxy.py
nohup python3 /tmp/rpc_proxy.py {PROXY_PORT} {SIM_PORT_DEFAULT} > /tmp/proxy.log 2>&1 &
for i in $(seq 1 20); do
  grep -q PROXY_READY /tmp/proxy.log && break
  sleep 0.5
done
grep -q PROXY_READY /tmp/proxy.log || {{ echo PROXY_DEAD; cat /tmp/proxy.log; exit 1; }}
iptables -A OUTPUT -p tcp --dport {SIM_PORT_DEFAULT} -m owner --uid-owner \
  $(id -u {FLIGHT_USER}) -j REJECT >/dev/null 2>&1 \
  && echo PORT_BLOCKED || echo PORT_NOT_BLOCKED
"""

GRADES: dict[str, "asyncio.Future[dict]"] = {}


def landed_on_ground(state: dict, ground_z: float | None, tolerance: float = LANDING_TOLERANCE_M) -> bool:
    """A landing is the drone sitting still at ground level.

    The simulator's own landed flag is not enough on its own, and not required either.
    While the altitude hold is commanding the drone the simulator reports it as flying
    even once it is resting on the grass, so the flag stays False through a perfect
    touchdown. It flips only when the motors are released, and releasing them also
    stops the simulator holding the drone up, so it falls through the ground. Those two
    happen in the same tenth of a second, which is why no landing could satisfy a rule
    that wanted the flag and ground level together. Read the evidence instead: the
    drone is at ground level and it has stopped. Either the flag or a speed at or
    below the touchdown speed shows it has stopped.
    """
    if ground_z is None:
        return False
    if abs(state["z"] - ground_z) > tolerance:
        return False
    return bool(state.get("landed", False)) or state.get("speed", 1.0) <= TOUCHDOWN_SPEED_MPS


def resting_z(recording: list[dict]) -> float | None:
    """Ground level under the drone: where it sat still after the reset drop hit the ground."""
    window = [r for r in recording if r.get("t", 0.0) <= SETTLE_WINDOW_S]
    impact = next(
        (
            i
            for i, (a, b) in enumerate(zip(window, window[1:]), start=1)
            if a.get("collision_t", 0.0) == 0.0 and b.get("collision_t", 0.0) > 0.0
        ),
        None,
    )
    after = window[impact:] if impact is not None else window
    settled = sorted(r["z"] for r in after if r.get("speed", 1.0) < SETTLE_SPEED_MPS)
    if settled:
        return settled[len(settled) // 2]
    return window[impact]["z"] if impact is not None else None


def takeoff_index(recording: list[dict]) -> int | None:
    """First sample a metre above the resting height, once the drone has hit the ground or sat still."""
    rest = resting_z(recording)
    if rest is None:
        return None
    anchored = False
    for i, r in enumerate(recording):
        anchored = anchored or r.get("collision_t", 0.0) > 0.0 or r.get("speed", 1.0) < SETTLE_SPEED_MPS
        if anchored and r["z"] < rest - TAKEOFF_CLEARANCE_M:
            return i
    return None


def reset_end_index(recording: list[dict]) -> int | None:
    """Where the reset drop is over: the first still sample after its impact, else the sample after it."""
    window = [r for r in recording if r.get("t", 0.0) <= SETTLE_WINDOW_S]
    impact = next(
        (
            i
            for i, (a, b) in enumerate(zip(window, window[1:]), start=1)
            if a.get("collision_t", 0.0) == 0.0 and b.get("collision_t", 0.0) > 0.0
        ),
        None,
    )
    if impact is None:
        return next((i for i, r in enumerate(window) if r.get("speed", 1.0) < SETTLE_SPEED_MPS), None)
    for i in range(impact, len(window)):
        if window[i].get("speed", 1.0) < SETTLE_SPEED_MPS:
            return i
    return min(impact + 1, len(recording) - 1)


def collided(recording: list[dict], crash_speed: float = CRASH_SPEED_MPS, landing_on: tuple[str, ...] = ()) -> bool:
    """A new collision once the reset drop is over, unless it is a gentle touchdown on the ground.

    landing_on names other objects a gentle touchdown is allowed on, such as a pad marker.

    The simulator drops the drone onto the ground at reset, so that impact and
    its bounces are expected. From the moment the drone first sits still after
    it, any new collision time stamp counts, except hitting the ground slower
    than crash speed, which is a landing. A drone that lifts off without ever
    sitting still may still bump the ground on the way up, so before takeoff
    only hits on something other than the ground count. The stamp shows up one
    sample after the impact, so impact speed is read from the sample before it.

    A ground touch while the drone has still not climbed TAKEOFF_SETTLE_M is read as the
    simulator's own takeoff drop, not a crash: the drone is still taking off and the drop
    is not something it chose. Once it has been that high, a ground touch is judged on
    speed as before, so a dive out of cruise still counts.
    """
    start = reset_end_index(recording)
    takeoff = takeoff_index(recording)
    if start is None:
        start = takeoff
    if start is None:
        return False
    rest = resting_z(recording)
    ground = next((r.get("hit", "") for r in recording[: start + 1] if r.get("hit")), "")
    last = max((r.get("collision_t", 0.0) for r in recording[: start + 1]), default=0.0)
    # Highest point reached so far, as altitude above the resting height.
    climbed = 0.0
    if rest is not None:
        climbed = max((rest - r["z"] for r in recording[: start + 1]), default=0.0)
    for i, (prev, r) in enumerate(zip(recording[start:], recording[start + 1 :]), start=start + 1):
        if rest is not None:
            climbed = max(climbed, rest - r["z"])
        stamp = r.get("collision_t", last)
        if stamp > last:
            on_ground = r.get("hit", "") == ground or r.get("hit", "") in landing_on
            before_takeoff = takeoff is None or i < takeoff
            taking_off = (
                EXCUSE_TAKEOFF_BOUNCE
                and rest is not None
                and climbed < TAKEOFF_SETTLE_M
                and prev.get("speed", 0.0) <= TAKEOFF_BOUNCE_SPEED_MPS
            )
            excused = on_ground and (before_takeoff or taking_off or prev.get("speed", 0.0) <= crash_speed)
            if not excused:
                return True
            last = stamp
    return False


def covers_flight(recording: list[dict], state: dict) -> bool:
    """The recording runs from before takeoff to where the drone ended, with no holes."""
    if takeoff_index(recording) is None:
        return False
    if any(b.get("t", 0.0) - a.get("t", 0.0) > MAX_SAMPLE_GAP_S for a, b in zip(recording, recording[1:])):
        return False
    end = recording[-1]
    return math.dist([end["x"], end["y"], end["z"]], [state["x"], state["y"], state["z"]]) <= END_MATCH_M


def path_length(recording: list[dict]) -> float:
    total = 0.0
    for a, b in zip(recording, recording[1:]):
        total += math.dist([a["x"], a["y"], a["z"]], [b["x"], b["y"], b["z"]])
    return round(total, 3)


def max_altitude(recording: list[dict], ground_z: float | None) -> float:
    if not recording:
        return -1.0
    base = ground_z if ground_z is not None else 0.0
    return round(max(base - r["z"] for r in recording), 3)


def parse_recording(raw: bytes | str) -> list[dict]:
    text = raw.decode() if isinstance(raw, bytes) else raw
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class FlightData(vf.TaskData):
    kind: str
    target: list[float]
    strict_port: bool = True
    """Refuse to grade when the flight user can still reach the simulator port. Off only for docker practice runs."""


class FlightTask(vf.Task[FlightData, vf.State, vf.TaskConfig]):
    async def _wait_sim(self, runtime: vf.Runtime) -> None:
        result = await runtime.run(["python3", "-c", CHECK_SIM], {})
        if result.exit_code == 0:
            return
        await runtime.run(
            ["bash", "-c", "nohup /usr/local/bin/start_sim.sh > /tmp/start_sim.out 2>&1 &"],
            {},
        )
        for _ in range(SIM_BOOT_TRIES):
            result = await runtime.run(["python3", "-c", CHECK_SIM], {})
            if result.exit_code == 0:
                return
            await asyncio.sleep(SIM_BOOT_WAIT_S)
        raise RuntimeError("simulator did not come up inside the sandbox")

    async def setup(self, runtime: vf.Runtime) -> None:
        """Start the simulator, wait until it answers, reset the drone."""
        await self._wait_sim(runtime)
        result = await runtime.run(["python3", "-c", RESET_SIM], {})
        if result.exit_code != 0:
            raise RuntimeError(f"could not reset the drone: {result.stderr}")

    def _grader_config(self, runtime: vf.Runtime):
        res = self.data.resources
        fields = {
            "image": self.data.image,
            "cpu": res.cpu,
            "memory": res.memory,
            "gpu": res.gpu,
            "disk": res.disk,
            "allow": [],
            "block": ["*"],
        }
        if isinstance(runtime, PrimeRuntime):
            return PrimeConfig(vm=True, **fields)
        if isinstance(runtime, DockerRuntime):
            return DockerConfig(**fields)
        raise RuntimeError("grading needs a docker or prime runtime")

    async def _read_state(self, runtime: vf.Runtime) -> dict | None:
        result = await runtime.run(["env", "PYTHONNOUSERSITE=1", "/usr/bin/python3", "-c", READ_STATE], {})
        if result.exit_code != 0:
            return None
        return json.loads(result.stdout.strip().splitlines()[-1])

    async def _grade(self, trace: vf.Trace, runtime: vf.Runtime) -> dict:
        pending = GRADES.get(trace.id)
        if pending is None:
            pending = asyncio.ensure_future(self._grade_once(runtime))
            GRADES[trace.id] = pending
            for old in [k for k, v in GRADES.items() if v.done()][:-GRADE_CACHE_KEEP]:
                GRADES.pop(old, None)
        return await asyncio.shield(pending)

    async def _fly_once(self, grader: vf.Runtime) -> tuple[dict | None, str, list[dict]]:
        result = await grader.run(["python3", "-c", RESET_SIM], {})
        if result.exit_code != 0:
            return None, f"reset failed: {result.stderr[-200:]}", []
        started = await grader.run(["bash", "-c", START_RECORDER], {})
        if started.exit_code != 0:
            return None, "flight recorder did not start", []
        try:
            result = await asyncio.wait_for(
                grader.run(
                    [
                        "timeout",
                        str(int(FLIGHT_TIMEOUT_S)),
                        "sudo",
                        "-u",
                        FLIGHT_USER,
                        "env",
                        "PYTHONNOUSERSITE=1",
                        f"AERIALSIM_SIM_PORT={PROXY_PORT}",
                        "/usr/bin/python3",
                        "/tmp/grade_flight.py",
                        "/tmp/flight.py",
                    ],
                    {},
                ),
                timeout=FLIGHT_TIMEOUT_S + 60.0,
            )
        except asyncio.TimeoutError:
            await grader.run(["bash", "-c", STOP_RECORDER], {})
            return None, f"flight ran past {FLIGHT_TIMEOUT_S:.0f} seconds", []
        await asyncio.sleep(3.0)
        await grader.run(["bash", "-c", STOP_RECORDER], {})
        try:
            recording = parse_recording(await grader.read(RECORDING_PATH, max_bytes=MAX_RECORDING_BYTES))
        except Exception as exc:  # noqa: BLE001
            return None, f"could not read the flight recording: {exc}", []
        if not recording:
            log = await grader.run(["bash", "-c", "tail -c 300 /tmp/recorder.log 2>/dev/null"], {})
            return None, f"the flight recording is empty: {log.stdout.strip()[-300:]}", []
        if result.exit_code != 0:
            return None, f"flight failed: {result.stdout[-200:]}{result.stderr[-200:]}", recording
        return await self._read_state(grader), "", recording

    async def _grade_once(self, runtime: vf.Runtime) -> dict:
        grade: dict = {
            "state": None,
            "repeat_state": None,
            "note": "",
            "port_blocked": False,
            "ground_z": None,
            "recording": [],
            "repeat_recording": [],
        }
        try:
            program = await runtime.read(SUBMIT_PATH, max_bytes=MAX_PROGRAM_BYTES)
        except FileNotFoundError:
            program = None
        if not program:
            grade["note"] = f"no program at {SUBMIT_PATH}"
            return grade
        async with provision_runtime(self._grader_config(runtime)) as grader:
            await self._wait_sim(grader)
            await grader.write("/tmp/flight.py", program)
            await grader.write("/tmp/grade_flight.py", GATE_SOURCE)
            await grader.write("/tmp/rpc_proxy.py", PROXY_SOURCE)
            await grader.write("/tmp/record_flight.py", RECORDER_SOURCE.encode())
            ground = await self._read_state(grader)
            if ground is None:
                grade["note"] = "could not read the drone's resting position"
                return grade
            grade["ground_z"] = ground["z"]
            started = await grader.run(["bash", "-c", START_PROXY], {})
            if started.exit_code != 0:
                grade["note"] = f"proxy did not start: {started.stdout[-200:]}"
                return grade
            grade["port_blocked"] = "PORT_BLOCKED" in started.stdout
            if self.data.strict_port and not grade["port_blocked"]:
                grade["note"] = "the simulator port is not isolated from the flight user, not graded"
                return grade
            await grader.prepare_execution([])
            grade["state"], grade["note"], grade["recording"] = await self._fly_once(grader)
            if grade["state"] is None:
                return grade
            grade["repeat_state"], _, grade["repeat_recording"] = await self._fly_once(grader)
        return grade

    def _mission_ok(self, state: dict, ground_z: float | None = None) -> bool:
        """Did the flight end where the mission asked?

        Heights are measured above the ground, because that is what the missions ask for:
        "hover 10 meters above the ground". Measured from the simulator origin instead,
        which on the City Park map sits 1.78 m above the grass, a drone hovering correctly
        at 10 m reads as 8.22 m and misses the hover tolerance of 1.5 m every time, while
        the reach tolerance of 2.0 m lets the same error through.
        """
        tx, ty, tz = self.data.target
        xy_miss = math.dist([state["x"], state["y"]], [tx, ty])
        height = (ground_z - state["z"]) if ground_z is not None else -state["z"]
        if self.data.kind == "hover":
            return abs(height - tz) <= 1.5 and state["speed"] <= 1.0
        if self.data.kind == "reach":
            return xy_miss <= 2.5 and abs(height - tz) <= 2.0 and state["speed"] <= 1.0
        return xy_miss <= 2.5 and landed_on_ground(state, ground_z) and state["speed"] <= TOUCHDOWN_SPEED_MPS

    def _flight_ok(self, state: dict | None, recording: list[dict], ground_z: float | None) -> bool:
        """The mission is met at the end and nothing was hit on the way."""
        if state is None or not recording or not covers_flight(recording, state):
            return False
        ground = resting_z(recording)
        if ground is None:
            ground = ground_z
        return self._mission_ok(state, ground) and not collided(recording)

    @vf.reward(weight=1.0)
    async def mission_complete(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        grade = await self._grade(trace, runtime)
        flights = [(grade["state"], grade["recording"]), (grade["repeat_state"], grade["repeat_recording"])]
        return 1.0 if all(self._flight_ok(s, r, grade["ground_z"]) for s, r in flights) else 0.0

    @vf.metric
    async def success(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        grade = await self._grade(trace, runtime)
        return float(self._flight_ok(grade["state"], grade["recording"], grade["ground_z"]))

    @vf.metric
    async def distance_to_target(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        grade = await self._grade(trace, runtime)
        if grade["state"] is None:
            return -1.0
        tx, ty, _ = self.data.target
        return math.dist([grade["state"]["x"], grade["state"]["y"]], [tx, ty])

    @vf.metric
    async def final_speed(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        grade = await self._grade(trace, runtime)
        if grade["state"] is None:
            return -1.0
        return grade["state"]["speed"]

    @vf.metric
    async def collision(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Whether the drone hit anything in either graded flight."""
        grade = await self._grade(trace, runtime)
        if not grade["recording"]:
            return -1.0
        return float(collided(grade["recording"]) or collided(grade["repeat_recording"]))

    @vf.metric
    async def path_length_m(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        grade = await self._grade(trace, runtime)
        return path_length(grade["recording"]) if grade["recording"] else -1.0

    @vf.metric
    async def max_altitude_m(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Highest point above the resting position during the first graded flight."""
        grade = await self._grade(trace, runtime)
        rest = resting_z(grade["recording"])
        return max_altitude(grade["recording"], rest if rest is not None else grade["ground_z"])

    def _ground_used(self, grade: dict) -> float | None:
        """The ground level the rules measure heights from, as _flight_ok picks it."""
        ground = resting_z(grade["recording"])
        return grade["ground_z"] if ground is None else ground

    @vf.metric
    async def ground_level_m(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """The ground the rules measured heights from, in simulator z."""
        grade = await self._grade(trace, runtime)
        ground = self._ground_used(grade)
        return -99.0 if ground is None else ground

    @vf.metric
    async def final_altitude_m(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """How high the drone ended above that ground."""
        grade = await self._grade(trace, runtime)
        ground = self._ground_used(grade)
        if grade["state"] is None or ground is None:
            return -99.0
        return ground - grade["state"]["z"]

    @vf.metric
    async def recording_covers_flight(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Whether the recording runs from before takeoff to where the drone ended.

        Reported on its own because a mission that reads as failed can fail here instead
        of on the flight, and the three rules were indistinguishable from the outside.
        """
        grade = await self._grade(trace, runtime)
        if grade["state"] is None or not grade["recording"]:
            return -1.0
        return float(covers_flight(grade["recording"], grade["state"]))

    @vf.metric
    async def end_of_flight_rule_met(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """The mission's own end condition, with coverage and collisions set aside."""
        grade = await self._grade(trace, runtime)
        if grade["state"] is None:
            return -1.0
        return float(self._mission_ok(grade["state"], self._ground_used(grade)))

    @vf.metric
    async def sim_port_blocked(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Whether the platform let us block the flight user from the simulator port."""
        grade = await self._grade(trace, runtime)
        return float(grade.get("port_blocked", False))

    @vf.metric
    async def repeat_spread_m(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """How far apart the same program ends when flown twice."""
        grade = await self._grade(trace, runtime)
        if grade["state"] is None or grade["repeat_state"] is None:
            return -1.0
        return math.dist(
            [grade["state"]["x"], grade["state"]["y"], grade["state"]["z"]],
            [grade["repeat_state"]["x"], grade["repeat_state"]["y"], grade["repeat_state"]["z"]],
        )

    @vf.metric
    async def repeat_agrees(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Whether both flights of the same program score the same."""
        grade = await self._grade(trace, runtime)
        if grade["state"] is None or grade["repeat_state"] is None:
            return -1.0
        return float(
            self._flight_ok(grade["state"], grade["recording"], grade["ground_z"])
            == self._flight_ok(grade["repeat_state"], grade["repeat_recording"], grade["ground_z"])
        )


class FlightConfig(vf.TasksetConfig):
    image: str | None = None
    """Override the sandbox image for every mission."""
    strict_port: bool = True
    """Set false only for docker practice runs, where the port block cannot be installed."""
    gpu: str = "1"
    """How many GPUs a mission wants, as a count.

    A type name is right for prime sandboxes, which pick a card to match. A docker
    runtime already has whatever card the box has, and refuses a type name the box does
    not carry: on an A100 box every rollout died with "Requested 'RTX_PRO_6000', but
    Docker exposed: NVIDIA A100-SXM4-80GB". Set a count like "1" for docker."""


class FlightTaskset(vf.Taskset[FlightTask, FlightConfig]):
    def load(self) -> Iterator[FlightTask]:
        for idx, mission in enumerate(MISSIONS):
            yield FlightTask(
                FlightData(
                    idx=idx,
                    name=mission["name"],
                    prompt=PROMPT.format(text=mission["text"]),
                    image=self.config.image or IMAGE,
                    network_allow=[],
                    kind=mission["kind"],
                    target=mission["target"],
                    strict_port=self.config.strict_port,
                    timeout=vf.TaskTimeout(setup=900, agent=1800, scoring=600),
                    resources=vf.TaskResources(cpu=8, memory=16, gpu=self.config.gpu, disk=40),
                ),
                vf.TaskConfig(),
            )


__all__ = ["FlightTaskset"]

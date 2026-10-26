"""The seven visual families, flown by writing a program instead of one action a turn.

The agent gets a shell in its own box and writes a flight program there. It never
reaches the simulator. The environment carries the file across, flies it in the
simulator box, and hands back the drone's camera, its final state and what the flight
did. The agent looks at the picture, revises the file, and the environment flies it
again. The file as the agent leaves it is flown once more and that flight is scored.

Why this is a separate environment rather than a setting on the visual one: the visual
environment asks for one JSON action per turn, synchronously, while a tool harness is
built for an agent that works on its own and then reports. Pointing the visual
environment at the bash harness made a 4B model pick the shell every time. Across 133
flights it used its whole turn budget on tool calls and sent the simulator zero
commands, and saying so in the prompt changed nothing. The two designs want different
things from the same reply, so the interaction itself had to change.

Two boxes, and the environment is the courier between them:

    agent box              environment            simulator box
    python, a shell  --->  reads the file   --->  runs it as flight_user
    no simulator     <---  sends a frame    <---  records the flight

Nothing on the network joins them, so the agent cannot ask the simulator what is on the
pad. Its program cannot either, and the reason is the import gate rather than the proxy.
The gate allows only aerialsim_flight, aerialsim_flight.drone, math and time, and
rejects every attribute starting with an underscore, so a program can neither import
cosysairsim nor reach `d._client` to build its own RPC client. What it is left with is
the drone API, which has no scene query in it at all. The filtered proxy sits in front
of that and refuses anything but the drone API's own calls, and on a prime sandbox an
iptables rule keeps the program on the proxy; a docker runtime cannot install that rule,
and `connect(port=41451)` is legal under the gate, so there the gate is the whole
boundary. The camera frames the environment chooses to send are the only way to learn
the hidden fact, which is what makes this a measurement of seeing rather than guessing.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Iterator
from contextlib import asynccontextmanager

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.configs.env import TimeoutConfig as EnvTimeoutConfig
from verifiers.v1.runtimes import DockerConfig, PrimeConfig, provision_runtime
from verifiers.v1.utils.loaders import harness_config_type

from aerialsim_flight import advanced
from aerialsim_flight.obstacle_bypass import barrier_hit
from aerialsim_flight.start_pose import PLACE_START_SOURCE
from aerialsim_flight.taskset import (
    CHECK_SIM,
    FLIGHT_USER,
    GATE_SOURCE,
    IMAGE,
    MAX_PROGRAM_BYTES,
    MAX_RECORDING_BYTES,
    PROXY_PORT,
    PROXY_SOURCE,
    RECORDER_SOURCE,
    RECORDING_PATH,
    RESET_SIM,
    SIM_BOOT_TRIES,
    SIM_BOOT_WAIT_S,
    START_PROXY,
    START_RECORDER,
    STOP_RECORDER,
    STOP_SIM,
    SUBMIT_PATH,
    collided,
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
from aerialsim_flight.visual import (
    CAMERA_PITCH_DEG,
    PLACE_START_TRIES,
    VisualConfig,
    VisualData,
    VisualTask,
    VisualTaskset,
    docker_cpu_limit,
    jpeg_data_url,
    judge,
    mission_text,
    scene_objects,
    shaped_score,
)
from aerialsim_flight.visual_runtime import advanced_recorder

BASH_HARNESS = harness_config_type("bash").model_validate({"id": "bash"})

AGENT_IMAGE = "python:3.11-slim"
"""The agent's own box. A shell and a text editor is all it needs: it never flies."""

REFERENCE_PATH = "drone_api.py"
"""The drone API source, dropped in the agent's working directory so it can read the
exact signatures instead of guessing them from the prompt."""

PROGRAM_PATH = "/tmp/flight.py"
"""Where the submitted program lands in the simulator box."""

FLIGHT_TIMEOUT_S = 300.0
"""Default ceiling on one flight. A whole mission in one program, so longer than the
flight taskset's single hop. The long families need more, which `flight_timeout` sets."""

EPISODE_TIMEOUT_S = 5400.0
"""Several flights plus the agent's own thinking between them."""

SEGMENT_SETTLE_S = 3.0
"""Let the recorder catch the end of the flight before it is stopped."""

MAX_FEEDBACK_CHARS = 1500
"""How much of the program's own output comes back. A program that prints in a loop
must not push the mission text out of the model's context."""


PROGRAM_PROMPT = """You fly a drone in a photoreal park by writing a Python program.

In python3:

    from aerialsim_flight.drone import connect
    d = connect()
    d.takeoff(height=5)
    d.goto(20, 0, -8, speed=3)
    d.land()

Positions are NED: x north, y east, z down, so 8 metres up is z = -8. Also available:
state, camera, velocity, attitude, rates, hover, path, position, camera_control. Each
call blocks until the maneuver finishes. The full source of this API is in
{reference} in your working directory, so read it for the exact arguments.

Write your flight program to {submit}. That file is the only thing scored. It is flown
in the simulator and the flight it produces sets your reward. You have no simulator in
your own box, so running the program yourself will fail at the connect call. That is
expected: you write the file, and this conversation flies it for you.

Each time you stop, the program is flown and you are shown the drone's camera at the
end of that flight, where the drone ended up, and anything the program printed. Then
you may revise the file and it is flown again. You get a fixed number of flights and
the last one is the graded flight, so use the earlier ones to look at the scene and
leave the file right before your last turn ends.

The program may import aerialsim_flight, aerialsim_flight.drone, math and time, and
nothing else. Names starting with an underscore are rejected, so d._client is not
available. print() works and its output comes back to you.

The mission text never says what is on the pad or where it is. The camera is the only
way to find out, and neither your shell nor your program can ask the simulator.

{mission}
"""


SCENE_SOURCE = r"""import json, math, struct, sys, time, zlib
import cosysairsim as airsim

spec = json.loads(sys.argv[1])
c = airsim.MultirotorClient()
c.confirmConnection()
c.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0),
    airsim.utils.euler_to_quaternion(0.0, math.radians(spec["pitch"]), 0.0)))


def solid_png(path, rgb, size=64):
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)


# The scene is rebuilt before every flight, so anything left from the last one goes
# first. A name that was never spawned simply is not there, which is not an error.
for name in spec.get("destroy", []):
    try:
        c.simDestroyObject(name)
    except Exception:
        pass
time.sleep(0.5)
rest_z = c.getMultirotorState().kinematics_estimated.position.z_val
for obj in spec.get("spawn", []):
    # Objects stand on the ground, so their centre sits half their height above it.
    z = rest_z - obj["sz"] / 2 if obj.get("on_ground", True) else obj["z"]
    pose = airsim.Pose(airsim.Vector3r(obj["x"], obj["y"], z), airsim.Quaternionr(0, 0, 0, 1))
    spawned = c.simSpawnObject(obj["name"], obj["asset"], pose,
                               airsim.Vector3r(obj["sx"], obj["sy"], obj["sz"]), False)
    if not spawned:
        raise RuntimeError("could not spawn %s" % obj["name"])
    if obj.get("colour"):
        # The look of the object is part of the task, so a texture that does not apply is an error.
        path = "/tmp/%s.png" % obj["name"]
        solid_png(path, obj["colour"])
        if not c.simSetObjectMaterialFromTexture(obj["name"], path):
            raise RuntimeError("could not colour %s" % obj["name"])
print(json.dumps({"rest_z": rest_z}))
"""


FRAME_SOURCE = r"""import base64, json, math
import cosysairsim as airsim
from cosysairsim.utils import quaternion_to_euler_angles

c = airsim.MultirotorClient()
c.confirmConnection()
s = c.getMultirotorState()
k = s.kinematics_estimated
roll, pitch, yaw = quaternion_to_euler_angles(k.orientation)
state = {"x": k.position.x_val, "y": k.position.y_val, "z": k.position.z_val,
         "vx": k.linear_velocity.x_val, "vy": k.linear_velocity.y_val,
         "vz": k.linear_velocity.z_val, "roll": math.degrees(roll),
         "pitch": math.degrees(pitch), "yaw": math.degrees(yaw),
         "landed": s.landed_state == airsim.LandedState.Landed}
frame = b""
for _ in range(3):
    responses = c.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)])
    data = getattr(responses[0], "image_data_uint8", None) if responses else None
    if data:
        frame = bytes(data)
        break
print(json.dumps({"state": state, "frame": base64.b64encode(frame).decode()}))
"""


def flight_command(seconds: float = FLIGHT_TIMEOUT_S, program_path: str = PROGRAM_PATH) -> list[str]:
    """Run the submitted program as the flight user, through the import gate and the
    filtered proxy. The same launcher the flight missions use."""
    return [
        "timeout",
        str(int(seconds)),
        "sudo",
        "-u",
        FLIGHT_USER,
        "env",
        "PYTHONNOUSERSITE=1",
        f"AERIALSIM_SIM_PORT={PROXY_PORT}",
        "/usr/bin/python3",
        "/tmp/grade_flight.py",
        program_path,
    ]


def flight_report(run: dict, rest_z: float, segment: int, segments: int) -> str:
    """What the agent is told about a flight.

    Only what the drone itself could know: where it ended, how fast it was moving,
    whether it was on the ground, whether it hit anything, and whatever the program
    printed. Never the mission score, and never anything derived from the hidden
    fact, because telling the agent that the mission failed would tell it what is on
    the pad without it having looked.
    """
    lines = [f"Flight {segment + 1} of {segments}."]
    if run.get("error"):
        lines.append(f"The program did not run: {run['error']}")
    state = run.get("state")
    if state:
        altitude = rest_z - state["z"]
        speed = math.dist([0, 0, 0], [state["vx"], state["vy"], state["vz"]])
        lines.append(
            f"The drone ended at x={state['x']:.1f} y={state['y']:.1f} z={state['z']:.1f}, "
            f"{altitude:.1f} m above the ground, moving at {speed:.1f} m/s, "
            f"heading {state['yaw']:.0f} degrees, "
            f"{'on the ground' if state['landed'] else 'in the air'}."
        )
    if run.get("recording"):
        lines.append(
            f"The flight lasted {run['seconds']:.0f} seconds and covered {path_length(run['recording']):.0f} metres."
        )
        if collided(run["recording"]):
            lines.append("The drone hit something during the flight.")
    output = (run.get("output") or "").strip()
    if output:
        lines.append("Your program printed:\n" + output[-MAX_FEEDBACK_CHARS:])
    lines.append("The camera below is the view where the flight ended.")
    return "\n".join(lines)


class CodingConfig(VisualConfig):
    """The visual taskset's scenes, asked for as a program.

    Nothing to add: a family and a seed here build exactly the scene the no-tools run
    builds, which is the only reason the two runs can be compared.
    """


class CodingTaskset(vf.Taskset[VisualTask, CodingConfig]):
    """The seven visual families as program missions."""

    def load(self) -> Iterator[VisualTask]:
        for task in VisualTaskset(self.config).load():
            data: VisualData = task.data
            yield VisualTask(
                data.model_copy(
                    update={
                        "system_prompt": PROGRAM_PROMPT.format(
                            submit=SUBMIT_PATH,
                            reference=REFERENCE_PATH,
                            mission=mission_text(data),
                        ),
                        # The agent's own box: a shell, no card, no network once it
                        # starts. The empty allowlist closes egress after the harness
                        # has installed itself, which happens while setup still has it.
                        "network_allow": [],
                        "resources": vf.TaskResources(cpu=2, memory=4, disk=10),
                        # Several flights and the agent's thinking between them.
                        "timeout": vf.TaskTimeout(setup=900, agent=4800, scoring=600),
                    }
                ),
                vf.TaskConfig(),
            )


class CodingEnvConfig(vf.EnvConfig):
    coder: vf.AgentConfig = vf.AgentConfig(
        harness=BASH_HARNESS,
        runtime=DockerConfig(image=AGENT_IMAGE, cpu=1),
    )
    """The agent and the box its shell runs in. Never the simulator box."""
    segments: int = Field(default=3, ge=1)
    """How many times the program is flown. The last one is the graded flight, and the
    ones before it are practice whose result the agent is shown.

    Two is the floor for a family whose answer is only visible from above the pad: one
    flight to get there and look, then the graded flight to act on what was seen. Three
    leaves one revision in hand. Each one costs a whole flight, so this is the knob that
    sets what a rollout costs."""
    flight_timeout: float = FLIGHT_TIMEOUT_S
    """Seconds one flight may take before it is stopped. A program that loops forever
    burns this on every one of its flights, so it is not set generously by default. The
    long families, which fly an approach, a break off, a climb and a second approach,
    need more than the short ones."""
    frames: bool = True
    """Send the camera with each report. Off is the frames withheld baseline, which
    measures how much of the score comes from seeing. Never a real run."""
    shaped_reward: bool = False
    """Train on shaped_score instead of the mission score. Training sources only."""
    timeout: EnvTimeoutConfig = EnvTimeoutConfig(episode=EPISODE_TIMEOUT_S)
    strict_port: bool = True
    """Refuse to run at all when the flight user can still reach the simulator port.

    The port block is the outer of three layers. It keeps the submitted program on the
    filtered proxy, which passes the drone API's calls and refuses anything that could
    read the scene. Inside that, the import gate allows only aerialsim_flight,
    aerialsim_flight.drone, math and time, and rejects any attribute starting with an
    underscore, so a program cannot import cosysairsim or reach `d._client` to build its
    own RPC client. The drone API it is left with has no scene query at all.

    So the gate, not the block, is what actually stops a program learning what is on the
    pad. The block matters because `connect(port=41451)` is legal under the gate and
    goes straight past the proxy, which removes the middle layer.

    A docker runtime cannot install the iptables rule, so a docker run must set this
    false. Keep it true on prime sandboxes, where the rule does install. When it is true
    and the block does not take, the run raises rather than scoring zero: 236 rollouts
    were once spent on a fault that only showed up as a reward of zero."""
    runtime: str = "prime"
    """Where the simulator box runs: prime or docker."""
    image: str | None = None
    cpu: float = 8
    memory: float = 16
    gpu: str = "1"
    """How many GPUs the simulator box wants, as a count, or a card name on prime."""
    disk: float = 40


class CodingFlightEnv(vf.Env[CodingEnvConfig]):
    """Owns the simulator box, carries the program into it, and reports back."""

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

    @asynccontextmanager
    async def _sim_box(self, with_events: bool):
        async with provision_runtime(self._sim_config()) as box:
            try:
                yield box
            finally:
                for command in ([STOP_EVENTS] if with_events else []) + [STOP_RECORDER]:
                    try:
                        await asyncio.wait_for(box.run(["bash", "-c", command], {}), timeout=10.0)
                    except Exception:  # noqa: BLE001
                        pass

    async def _wait_sim(self, box) -> None:
        result = await box.run(["python3", "-c", CHECK_SIM], {})
        if result.exit_code == 0:
            return
        # One restart if the first boot does not answer: the first boot on a fresh
        # GPU can stall while shaders build.
        for attempt in range(2):
            if attempt:
                # Bracketed pattern and a real wait: "pkill -f CityPark" matches the
                # shell running it on Linux, where pgrep -a lists full command lines, so it
                # could kill itself part way through the restart.
                await box.run(["bash", "-c", STOP_SIM], {})
            await box.run(["bash", "-c", "nohup /usr/local/bin/start_sim.sh > /tmp/start_sim.out 2>&1 &"], {})
            for _ in range(SIM_BOOT_TRIES):
                result = await box.run(["python3", "-c", CHECK_SIM], {})
                if result.exit_code == 0:
                    return
                await asyncio.sleep(SIM_BOOT_WAIT_S)
        raise RuntimeError("simulator did not come up in the box")

    async def _prepare_sim(self, box, data: VisualData) -> bool:
        """Boot the simulator, install the launcher and the filtered proxy.

        Returns whether the flight user is blocked from the simulator port.
        """
        await self._wait_sim(box)
        await box.write("/tmp/grade_flight.py", GATE_SOURCE)
        await box.write("/tmp/rpc_proxy.py", PROXY_SOURCE)
        recorder = advanced_recorder(RECORDER_SOURCE) if data.scenario is not None else RECORDER_SOURCE
        await box.write("/tmp/record_flight.py", recorder.encode())
        # The proxy script wants the program file to exist before it chmods it.
        await box.write(PROGRAM_PATH, b"")
        started = await box.run(["bash", "-c", START_PROXY], {})
        if started.exit_code != 0:
            raise RuntimeError(f"the filtered proxy did not start: {started.stdout[-300:]}")
        blocked = "PORT_BLOCKED" in started.stdout
        if self.config.strict_port and not blocked:
            # Loud, not a zero. A whole run of rewards that are zero for a reason
            # nobody can see is the expensive way to find this out.
            raise RuntimeError(
                "the flight user is not blocked from the simulator port, so the "
                "submitted program could go around the filtered proxy. A docker runtime "
                "cannot install the rule: set env.strict_port false for docker runs, or "
                "run the simulator on a prime sandbox."
            )
        return blocked

    async def _reset_drone(self, box, data: VisualData) -> None:
        result = await box.run(["python3", "-c", RESET_SIM], {})
        if result.exit_code != 0:
            raise RuntimeError(f"could not reset the drone: {result.stderr[-200:]}")
        await asyncio.sleep(2.0)
        if data.start is None:
            return
        # The placed drone sometimes takes longer than the settle window to come to
        # rest; a second attempt from the settled state succeeds. Without the reset in
        # between, every retry reads the same bad resting height and fails the same way.
        for attempt in range(PLACE_START_TRIES):
            if attempt:
                await box.run(["python3", "-c", RESET_SIM], {})
                await asyncio.sleep(2.0)
            placed = await asyncio.wait_for(
                box.run(
                    ["python3", "-c", PLACE_START_SOURCE, json.dumps({"start": data.start, "heading": data.heading})],
                    {},
                ),
                timeout=30.0,
            )
            if placed.exit_code == 0:
                return
            if attempt == PLACE_START_TRIES - 1:
                raise RuntimeError(f"could not place the launch pose: {placed.stderr[-300:]}")
            await asyncio.sleep(2.0)

    async def _stage_scene(self, box, spawn: list[dict]) -> float:
        """Rebuild the scene around the resting drone. Returns its resting z."""
        spec = {
            "pitch": CAMERA_PITCH_DEG,
            "destroy": [obj["name"] for obj in spawn],
            "spawn": spawn,
        }
        result = await asyncio.wait_for(box.run(["python3", "-c", SCENE_SOURCE, json.dumps(spec)], {}), timeout=120.0)
        if result.exit_code != 0:
            raise RuntimeError(f"could not build the scene: {result.stderr[-300:]}")
        return float(json.loads(result.stdout.strip().splitlines()[-1])["rest_z"])

    async def _observe(self, box) -> dict:
        """The drone's state and one camera frame."""
        result = await asyncio.wait_for(box.run(["python3", "-c", FRAME_SOURCE], {}), timeout=90.0)
        if result.exit_code != 0:
            return {"state": None, "frame": ""}
        return json.loads(result.stdout.strip().splitlines()[-1])

    async def _fly(self, box, data: VisualData, spawn: list[dict], program: bytes) -> dict:
        """One flight of one program: fresh scene, recorder on, program run, frame taken.

        Never raises for the program's own faults. A program that will not compile, or
        flies into the ground, is the agent's answer and has to come back as a report.
        """
        run: dict = {
            "state": None,
            "frame": "",
            "recording": [],
            "output": "",
            "error": "",
            "seconds": 0.0,
            "event_log": {},
            "event_failed": False,
            "rest_z": None,
        }
        await self._reset_drone(box, data)
        run["rest_z"] = await self._stage_scene(box, spawn)
        await box.write(PROGRAM_PATH, program)
        schedule = advanced.event_schedule(data.scenario) if data.scenario is not None else None
        if schedule is not None:
            await box.write("/tmp/flight_events.py", EVENT_SOURCE.encode())
            await box.write(
                EVENT_CONFIG_PATH,
                json.dumps({"schedule": schedule.model_dump(mode="json"), "rest_z": run["rest_z"]}).encode(),
            )
            started = await asyncio.wait_for(box.run(["bash", "-c", START_EVENTS], {}), timeout=60.0)
            if started.exit_code != 0:
                run["error"] = "the simulator event worker did not start"
                run["event_failed"] = True
                return run
        started = await box.run(["bash", "-c", START_RECORDER], {})
        if started.exit_code != 0:
            run["error"] = "the flight recorder did not start"
            return run
        began = asyncio.get_running_loop().time()
        try:
            result = await asyncio.wait_for(
                box.run(flight_command(self.config.flight_timeout), {}),
                timeout=self.config.flight_timeout + 60.0,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if result.exit_code != 0:
                run["error"] = f"it exited with code {result.exit_code}"
        except asyncio.TimeoutError:
            output = ""
            run["error"] = f"it ran past {self.config.flight_timeout:.0f} seconds and was stopped"
        run["output"] = output
        run["seconds"] = asyncio.get_running_loop().time() - began
        await asyncio.sleep(SEGMENT_SETTLE_S)
        if schedule is not None:
            try:
                stopped = await asyncio.wait_for(box.run(["bash", "-c", STOP_EVENTS], {}), timeout=60.0)
                run["event_log"] = parse_event_log(await box.read(EVENT_LOG_PATH, max_bytes=MAX_EVENT_BYTES))
                run["event_failed"] = stopped.exit_code != 0 or bool(run["event_log"].get("error"))
            except Exception:  # noqa: BLE001
                run["event_failed"] = True
                run["event_log"] = {"error": "event worker log unavailable"}
        await box.run(["bash", "-c", STOP_RECORDER], {})
        try:
            run["recording"] = parse_recording(await box.read(RECORDING_PATH, max_bytes=MAX_RECORDING_BYTES))
        except Exception:  # noqa: BLE001
            run["recording"] = []
        observed = await self._observe(box)
        run["state"] = observed["state"]
        run["frame"] = observed["frame"]
        return run

    def _report(self, run: dict, segment: int, segments: int) -> list:
        text = flight_report(run, run["rest_z"] or 0.0, segment, segments)
        parts: list = [{"type": "text", "text": text}]
        if self.config.frames and run.get("frame"):
            parts.append({"type": "image_url", "image_url": {"url": jpeg_data_url(run["frame"])}})
        return [{"role": "user", "content": parts}]

    def _opening(self, data: VisualData, observed: dict, segments: int) -> list:
        practice = segments - 1
        text = (
            "The drone is on the ground and has not flown yet. Your program will be "
            f"flown {segments} times. "
            + (
                f"The first {practice} are practice and you are shown what each one did. "
                if practice > 1
                else "The first one is practice and you are shown what it did. "
                if practice == 1
                else ""
            )
            + "The last one is the graded flight, of the file as it stands then. "
            f"Write your program to {SUBMIT_PATH}. The camera below is the view from "
            "where the drone stands now."
        )
        parts: list = [{"type": "text", "text": text}]
        if self.config.frames and observed.get("frame"):
            parts.append({"type": "image_url", "image_url": {"url": jpeg_data_url(observed["frame"])}})
        return [{"role": "user", "content": parts}]

    async def _read_program(self, box) -> bytes:
        try:
            return await box.read(SUBMIT_PATH, max_bytes=MAX_PROGRAM_BYTES) or b""
        except FileNotFoundError:
            return b""
        except Exception:  # noqa: BLE001
            return b""

    def _ending(self, state: dict | None, ground_z: float | None = None) -> str | None:
        """What the flight decided, read off the flight itself.

        A program declares nothing. On the ground is a landing, still in the air at the
        end is a hold, which is what the judge means by abort. The rest of the judge's
        rules then decide whether that hold was at the right place and height.

        This asks the same question the landing rule asks, through landed_on_ground, and
        not the simulator's landed flag. That flag only turns true when the motors are
        released, and the drone API leaves them running, because releasing them drops the
        drone through the ground. The flag therefore stays false through a perfect
        touchdown, so reading it records every landing as a hold and fails right_ending on
        every scene where landing was the correct choice.
        """
        if state is None:
            return None
        return "land" if landed_on_ground(state, ground_z) else "abort"

    async def run(self, task, agents) -> None:
        data: VisualData = task.data
        spawn = scene_objects(data)
        segments = self.config.segments
        scenario = data.scenario
        schedule = advanced.event_schedule(scenario) if scenario is not None else None
        flights: list[dict] = []
        submissions = 0
        async with self._sim_box(schedule is not None) as box:
            port_blocked = await self._prepare_sim(box, data)
            await self._reset_drone(box, data)
            rest_z = await self._stage_scene(box, spawn)
            opening = await self._observe(box)
            async with agents.coder.provision(task) as agent_box, _owned(agent_box):
                await agent_box.write(REFERENCE_PATH, _drone_api_source())
                # Rewards belong to the open interaction: leaving the context closes
                # the rollout and finishes the trace, so the graded flight is flown
                # while the agent is parked between turns rather than after.
                async with agents.coder.interaction(task, runtime=agent_box) as interaction:
                    segment = await interaction.turn(self._opening(data, opening, segments))
                    # The agent writes, the program flies, the result comes back, and
                    # the agent writes again. The last of its `segments` flights is the
                    # graded one, so the practice flights stop one short.
                    while segment is not None and not segment.terminated and len(flights) < segments - 1:
                        program = await self._read_program(agent_box)
                        if program:
                            submissions += 1
                            run = await self._fly(box, data, spawn, program)
                        else:
                            run = self._no_program(rest_z, opening.get("frame", ""))
                        flights.append(run)
                        segment = await interaction.turn(self._report(run, len(flights) - 1, segments))
                    # The file as the agent leaves it, flown once. The practice flights
                    # above are feedback and never score.
                    final_program = await self._read_program(agent_box)
                    graded = (
                        await self._fly(box, data, spawn, final_program)
                        if final_program
                        else self._no_program(rest_z, "")
                    )
                    self._score(
                        interaction.trace, data, graded, flights, final_program, submissions, port_blocked, schedule
                    )

    def _no_program(self, rest_z: float, frame: str) -> dict:
        return {
            "state": None,
            "frame": frame,
            "recording": [],
            "output": "",
            "error": f"there is no file at {SUBMIT_PATH} to fly",
            "seconds": 0.0,
            "event_log": {},
            "event_failed": False,
            "rest_z": rest_z,
        }

    def _score(
        self,
        trace,
        data: VisualData,
        graded: dict,
        flights: list[dict],
        final_program: bytes,
        submissions: int,
        port_blocked: bool,
        schedule,
    ) -> None:
        state = graded["state"]
        final = (
            state | {"speed": math.dist([0, 0, 0], [state["vx"], state["vy"], state["vz"]])}
            if state is not None
            else None
        )
        recording = graded["recording"]
        rest = resting_z(recording)
        ending = self._ending(final, rest)
        scores = judge(data, ending, recording, final, graded["event_log"], [])
        if graded["event_failed"]:
            # The advanced families' scene changes are timed off the simulator clock. A
            # worker that did not report means the mission's own timing was never
            # checked, so the flight cannot be called complete.
            scores["mission"] = 0.0
        if self.config.shaped_reward:
            # The mission score stays on the record with no weight; the shaped score is the reward.
            trace.record_reward("mission", scores["mission"], 0.0)
            trace.record_reward("shaped", shaped_score(scores), 1.0)
        else:
            trace.record_reward("mission", scores["mission"], 1.0)
        trace.record_metrics({k: v for k, v in scores.items() if k != "mission"})
        trace.record_metrics(
            {
                "practice_flights": float(len(flights)),
                "submissions": float(submissions),
                "wrote_a_program": float(bool(final_program)),
                "program_bytes": float(len(final_program)),
                "graded_flight_ran": float(not graded["error"]),
                "sim_port_blocked": float(port_blocked),
                "frames_withheld": float(not self.config.frames),
                "blocked": float(data.blocked),
                "flight_seconds": float(graded["seconds"]),
                "path_length_m": path_length(recording) if recording else -1.0,
                "max_altitude_m": max_altitude(recording, rest) if recording and rest is not None else -1.0,
            }
        )
        trace.info["ending"] = ending or "none"
        # The graded state, so a failure here can be read without the recording.
        trace.info["final"] = final
        trace.info["program"] = final_program.decode("utf-8", "replace")
        trace.info["graded_error"] = graded["error"]
        trace.info["graded_output"] = (graded["output"] or "")[-MAX_FEEDBACK_CHARS:]
        trace.info["pad"] = data.pad
        trace.info["recording_samples"] = len(recording)
        trace.info["recording"] = recording
        if data.scenario is not None:
            trace.info["event_log"] = graded["event_log"]
            trace.record_metrics(
                {
                    "event_failed": float(graded["event_failed"]),
                    "scenario_obstacle_hit": float(advanced.scenario_obstacle_hit(recording)),
                }
            )
            if schedule is not None:
                trace.record_metrics(validate_event_log(schedule, graded["event_log"], recording))
        if data.family == "obstacle_bypass":
            trace.record_metrics({"route_barrier_hit": float(barrier_hit(recording))})


@asynccontextmanager
async def _owned(box):
    """Keep a box we provisioned tearing itself down after we lend it out.

    A runtime's `borrowed` flag means "something else made this container, so do not
    remove it". Lending the box to a rollout sets that flag on the box's own info
    object, which is true from the rollout's side and wrong from ours, and nothing puts
    it back. The docker teardown then skips the container, so one agent box would be
    left running for every flight. We provisioned it, so the flag goes back when the
    loan ends.

    Seen on the first real run: thirty containers started, none removed.
    """
    info = getattr(box, "info", None)
    try:
        yield box
    finally:
        if info is not None and getattr(info, "borrowed", False):
            info.borrowed = False


def _drone_api_source() -> bytes:
    """The drone API source, for the agent to read in its own box.

    It describes the same calls the prompt lists and holds nothing about any scene, so
    handing it over gives away no part of the answer.
    """
    from pathlib import Path

    return Path(__file__).with_name("drone.py").read_bytes()


__all__ = [
    "AGENT_IMAGE",
    "PROGRAM_PROMPT",
    "CodingConfig",
    "CodingEnvConfig",
    "CodingFlightEnv",
    "CodingTaskset",
    "flight_report",
]

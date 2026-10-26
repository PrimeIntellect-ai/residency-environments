"""Drone control API over Cosys-AirSim.

Positions are NED: x north, y east, z down.

The simulator origin is not the ground. In the City Park world the grass under the launch
point sits about 1.8 m below z = 0, so z = -8 is nearly 10 m in the air. Heights in a
mission mean height above the ground, so `takeoff` and `state` work in those terms and
`ground_level` says where the ground is. Positions passed to `goto` and `path` stay in
plain simulator coordinates.

Angles are degrees at this layer. Positive pitch moves forward.
Each call maps to one RPC and blocks until the maneuver finishes.
"""

from __future__ import annotations

import math
import time
from typing import Optional

DEFAULT_SPEED = 3.0
HOVER_THROTTLE = 0.594
LAND_STEP_SECONDS = 1.0
LAND_STEP_M = 1.0
LAND_STABLE_M = 0.05
LAND_MAX_STEPS = 60
LAND_NEAR_GROUND_M = 0.4
"""Within this of the ground, a step that moves nothing means the ground stopped the drone."""
LAND_STALLS_ALLOWED = 4
"""Higher up, that many steps in a row moving nothing before giving up. One second of an
altitude hold does not always cover a metre, and treating the first short step as a
landing stopped a descent 2.09 m up while the program reported "Landing completed"."""
GROUND_CONFIRM_S = 1.5
GROUND_AGREE_M = 0.25
GROUND_TRIES = 8


def connect(host: str = "127.0.0.1", port: int = 41451) -> "Drone":
    """Connect to a running simulator and return a ready drone."""
    import cosysairsim as airsim

    client = airsim.MultirotorClient(ip=host, port=port)
    return Drone(client)


class Drone:
    """One drone in the simulator."""

    def __init__(self, client):
        self._client = client
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)
        self.trace: list[dict] = []
        # Measured now, while the drone is still on the ground. Measuring it lazily on
        # first use would record wherever the drone happened to be: a program whose first
        # state() call comes after takeoff would have taken the air as ground level.
        self._ground: Optional[float] = None
        self._ground = self._measure_ground()

    def reset(self):
        """Back to the start pose, on the ground, armed."""
        self._client.reset()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)
        self.trace.clear()
        self._ground = self._measure_ground()
        self._record("reset")

    def ground_level(self) -> float:
        """Where the ground under the launch point is, in simulator z."""
        if self._ground is None:
            self._ground = self._measure_ground()
        return self._ground

    def _measure_ground(self) -> float:
        """Read the resting height twice and require the readings to agree.

        The reset drop can leave the drone inside the ground for about a second before
        the simulator ejects it back to the surface. A single read taken in that window
        is metres out, and every height in the flight is measured from this number.
        """
        previous = None
        for _ in range(GROUND_TRIES):
            z = self._z()
            if previous is not None and abs(z - previous) <= GROUND_AGREE_M:
                return z
            previous = z
            time.sleep(GROUND_CONFIRM_S)
        return previous if previous is not None else 0.0

    def state(self) -> dict:
        """Position, velocity, attitude in degrees, landed flag, height above ground.

        `altitude` is the one to check a mission height against. `z` is the simulator's
        own coordinate and the ground is not at zero, so `z` and `altitude` differ by
        about 1.8 m in this world.
        """
        import cosysairsim as airsim
        from cosysairsim.utils import quaternion_to_euler_angles

        ground = self.ground_level()
        s = self._client.getMultirotorState()
        k = s.kinematics_estimated
        roll, pitch, yaw = quaternion_to_euler_angles(k.orientation)
        return {
            "x": k.position.x_val,
            "y": k.position.y_val,
            "z": k.position.z_val,
            "altitude": ground - k.position.z_val,
            "ground_z": ground,
            "vx": k.linear_velocity.x_val,
            "vy": k.linear_velocity.y_val,
            "vz": k.linear_velocity.z_val,
            "roll": math.degrees(roll),
            "pitch": math.degrees(pitch),
            "yaw": math.degrees(yaw),
            "landed": s.landed_state == airsim.LandedState.Landed,
        }

    def camera(self, camera_id: str = "0") -> Optional[bytes]:
        """One PNG frame, or None if capture fails."""
        import cosysairsim as airsim

        req = airsim.ImageRequest(camera_id, airsim.ImageType.Scene, False, True)
        responses = self._client.simGetImages([req])
        if not responses:
            return None
        data = getattr(responses[0], "image_data_uint8", None)
        return bytes(data) if data else None

    def takeoff(self, height: float = 3.0):
        """Take off, then climb to the asked height in meters above the ground.

        The height is above the ground, not below the simulator origin. Those differ by
        about 1.8 m in this world, so measuring from the origin puts the drone at 11.8 for
        a mission that wanted 10.
        """
        ground = self.ground_level()
        self._client.takeoffAsync().join()
        self._client.moveToZAsync(ground - abs(height), 2.0).join()
        self._record("takeoff", height=height)

    def goto(self, x: float, y: float, z: float, speed: float = DEFAULT_SPEED):
        """Fly to a position."""
        self._client.moveToPositionAsync(x, y, z, speed).join()
        self._record("goto", x=x, y=y, z=z, speed=speed)

    def position(self, x: float, y: float, z: float, yaw: float):
        """Fly to a position and face the given yaw in degrees."""
        import cosysairsim as airsim

        self._client.moveToPositionAsync(
            x,
            y,
            z,
            DEFAULT_SPEED,
            yaw_mode=airsim.YawMode(False, yaw),
        ).join()
        self._record("position", x=x, y=y, z=z, yaw=yaw)

    def velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        seconds: float,
        yaw: Optional[float] = None,
        yaw_rate: Optional[float] = None,
    ):
        """Fly at a velocity for a duration. Default holds heading."""
        import cosysairsim as airsim

        if yaw is not None:
            yaw_mode = airsim.YawMode(False, yaw)
        elif yaw_rate is not None:
            yaw_mode = airsim.YawMode(True, yaw_rate)
        else:
            yaw_mode = airsim.YawMode(True, 0.0)
        self._client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            seconds,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode,
        ).join()
        self._record("velocity", vx=vx, vy=vy, vz=vz, seconds=seconds)

    def attitude(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        throttle: float = HOVER_THROTTLE,
        seconds: float = 1.0,
    ):
        """Hold a body attitude in degrees. Throttle 0 to 1, default near hover."""
        self._client.moveByRollPitchYawThrottleAsync(
            math.radians(roll),
            math.radians(pitch),
            math.radians(yaw),
            throttle,
            seconds,
        ).join()
        self._record("attitude", roll=roll, pitch=pitch, yaw=yaw, throttle=throttle, seconds=seconds)

    def rates(
        self,
        roll_rate: float,
        pitch_rate: float,
        yaw_rate: float,
        throttle: float = HOVER_THROTTLE,
        seconds: float = 1.0,
    ):
        """Turn each axis at its own rate in degrees per second."""
        self._client.moveByAngleRatesThrottleAsync(
            math.radians(roll_rate),
            math.radians(pitch_rate),
            math.radians(yaw_rate),
            throttle,
            seconds,
        ).join()
        self._record(
            "rates", roll_rate=roll_rate, pitch_rate=pitch_rate, yaw_rate=yaw_rate, throttle=throttle, seconds=seconds
        )

    def hover(self):
        """Stop and hold the current position."""
        self._client.hoverAsync().join()
        self._record("hover")

    def land(self):
        """Descend to the ground and stop there.

        The descent is on height targets and never aims below the ground. A plain
        downward velocity with only "the drone stopped moving" to end it had no floor at
        all, so a descent that never registered contact kept asking for more and ended
        metres under the park.

        The motors stay on. Releasing them stops the simulator holding the drone up and
        it falls through the ground, which reads as a crash on a flight that landed.
        """
        ground = self.ground_level()
        prev_z = self._z()
        stalls = 0
        for _ in range(LAND_MAX_STEPS):
            if prev_z >= ground - LAND_STABLE_M:
                break
            target = min(prev_z + LAND_STEP_M, ground)
            # A fixed duration, not a position move. moveToZAsync has no timeout by
            # default, so a step that cannot quite reach its target never returns: the
            # first version of this loop hung a mission for nine minutes.
            self._client.moveByVelocityZAsync(0.0, 0.0, target, LAND_STEP_SECONDS).join()
            z = self._z()
            if abs(z - prev_z) < LAND_STABLE_M:
                # Nothing moved. Near the ground that is the ground holding the drone up,
                # which is the landing. Metres above it, it only means one second of an
                # altitude hold did not cover the step, so keep asking. Giving up on the
                # first short step left a descent 2.09 m up with the program reporting
                # "Landing completed", and the flight was then judged a hold.
                stalls += 1
                if z >= ground - LAND_NEAR_GROUND_M or stalls >= LAND_STALLS_ALLOWED:
                    break
            else:
                stalls = 0
            prev_z = z
        self._record("land")

    def path(self, waypoints: list[tuple], speed: float = DEFAULT_SPEED):
        """Fly through a list of x, y, z waypoints in one command."""
        import cosysairsim as airsim

        points = [airsim.Vector3r(x, y, z) for x, y, z in waypoints]
        self._client.moveOnPathAsync(points, speed).join()
        self._record("path", waypoints=list(waypoints), speed=speed)

    def camera_control(self, pitch: float, yaw: float = 0.0, camera_id: str = "0"):
        """Point the camera in degrees. Negative pitch looks down."""
        import cosysairsim as airsim
        from cosysairsim.utils import euler_to_quaternion

        pose = airsim.Pose(
            airsim.Vector3r(0.0, 0.0, 0.0),
            euler_to_quaternion(0.0, math.radians(pitch), math.radians(yaw)),
        )
        self._client.simSetCameraPose(camera_id, pose)
        self._record("camera_control", pitch=pitch, yaw=yaw)

    def result(self) -> dict:
        """Command trace plus final state."""
        return {"commands": list(self.trace), "final_state": self.state()}

    def _z(self) -> float:
        k = self._client.getMultirotorState().kinematics_estimated
        return k.position.z_val

    def _record(self, command: str, **args):
        self.trace.append({"command": command, "args": args})

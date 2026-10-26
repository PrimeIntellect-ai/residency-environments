"""Trusted, preflight placement inside the validated central launch area.

The environment runs this program after reset has settled and before starting
the controller or recorder. It receives only launch XY and heading; target
coordinates and hidden scene state are not part of this interface.
"""

PLACE_START_SOURCE = r"""import json, math, sys, time
import cosysairsim as airsim
from cosysairsim.utils import quaternion_to_euler_angles

spec = json.loads(sys.argv[1])
if not isinstance(spec, dict) or set(spec) != {"start", "heading"}:
    raise ValueError("start placement accepts only start and heading")
start, heading = spec["start"], spec["heading"]
def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
if not isinstance(start, list) or len(start) != 2 or not all(finite_number(v) for v in start):
    raise ValueError("start must contain two finite NED coordinates")
if math.hypot(*start) > 2.0 + 1e-9:
    raise ValueError("start must be within two metres of the launch origin")
if not finite_number(heading) or not -180.0 <= heading <= 180.0:
    raise ValueError("heading must be finite and between -180 and 180 degrees")

c = airsim.MultirotorClient(timeout_value=2.0)
c.confirmConnection()
c.enableApiControl(True)
# Reset drops the drone and it may still be falling when this runs. A wall clock sleep
# upstream buys less simulator time on a box flying ten simulators at once, so wait on
# the drone itself. Reading a falling drone's height as the resting height pins the new
# pose in mid air and it falls for the whole flight: that is what failed a third of
# find_pad's flights, and no retry could recover because each one read it falling again.
settle_deadline = time.monotonic() + 20.0
while time.monotonic() < settle_deadline:
    k = c.getMultirotorState().kinematics_estimated
    speed = math.hypot(k.linear_velocity.x_val, k.linear_velocity.y_val, k.linear_velocity.z_val)
    if math.isfinite(speed) and speed < 0.3:
        break
    time.sleep(0.25)
else:
    raise RuntimeError("drone was still moving twenty seconds after reset")
c.simPause(True)
c.cancelLastTask()
# SimpleFlight refuses disarm on an already inactive/disarmed vehicle. Establish
# its armed state while physics is paused, then require a successful disarm.
if not c.armDisarm(True):
    raise RuntimeError("could not establish API arm state before start placement")
if not c.armDisarm(False):
    raise RuntimeError("could not disarm before start placement")
old_z = c.getMultirotorState().kinematics_estimated.position.z_val
if not math.isfinite(old_z):
    raise RuntimeError("initial resting height is not finite")
pose = airsim.Pose(
    airsim.Vector3r(float(start[0]), float(start[1]), old_z),
    airsim.utils.euler_to_quaternion(0.0, 0.0, math.radians(heading)),
)
# True is positional because the ignore-collision keyword differs by build.
c.simSetVehiclePose(pose, True)
c.simPause(False)

deadline = time.monotonic() + 10.0
stable_since = None
while time.monotonic() < deadline:
    state = c.getMultirotorState().kinematics_estimated
    p, v = state.position, state.linear_velocity
    yaw = math.degrees(quaternion_to_euler_angles(state.orientation)[2])
    values = (p.x_val, p.y_val, p.z_val, v.x_val, v.y_val, v.z_val, yaw)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("start telemetry is not finite")
    yaw_error = abs((yaw - heading + 180.0) % 360.0 - 180.0)
    stable = (
        math.hypot(p.x_val - start[0], p.y_val - start[1]) <= 0.2
        and abs(p.z_val - old_z) <= 0.5
        and yaw_error <= 3.0
        and math.hypot(v.x_val, v.y_val, v.z_val) < 0.3
    )
    now = time.monotonic()
    if stable:
        if stable_since is None:
            stable_since = now
        elif now - stable_since >= 1.0:
            break
    else:
        stable_since = None
    time.sleep(0.1)
else:
    raise RuntimeError("start pose did not settle within ten seconds")

if not c.armDisarm(True):
    raise RuntimeError("could not rearm after start placement")
print(json.dumps({"start": [p.x_val, p.y_val], "heading": yaw, "ground_z": p.z_val}))
"""

__all__ = ["PLACE_START_SOURCE"]

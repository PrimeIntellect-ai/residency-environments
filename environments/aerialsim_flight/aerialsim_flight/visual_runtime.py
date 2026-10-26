"""Trusted runtime additions for visual missions with simulator-time events.

The original three families retain their continuous controller. Advanced
missions freeze physics while the model thinks, then resume for each action.
Only this controller can pause the simulator; the model keeps six flight actions.
"""

from aerialsim_flight.timed_events import EVENT_SYNC_SOURCE

WRAPPER_SOURCE = """
events_enabled = False


def wait_sim_seconds(seconds):
    start_ns = int(c.getMultirotorState().timestamp)
    previous_ns = start_ns
    while not timed_out:
        now_ns = int(c.getMultirotorState().timestamp)
        if now_ns < previous_ns:
            raise RuntimeError("simulator clock moved backwards")
        if (now_ns - start_ns) / 1e9 >= seconds:
            return
        previous_ns = now_ns
        time.sleep(0.05)


def paused_snapshot():
    # No model inference time may advance a scheduled event. A custom image
    # with a wall clock instead of a steppable clock must fail this check.
    c.simPause(True)
    time.sleep(0.1)
    before = int(c.getMultirotorState().timestamp)
    time.sleep(0.1)
    after = int(c.getMultirotorState().timestamp)
    if before <= 0 or before != after:
        raise RuntimeError("simulator timestamp must freeze while physics is paused")
    if events_enabled:
        sync_events(c)
    state = d.state()
    state["sim_t"] = after / 1e9
    frame = None
    for _ in range(3):
        frame = d.camera()
        if frame:
            break
        time.sleep(0.1)
    return {"state": state, "sim_t": after / 1e9,
            "frame": base64.b64encode(frame).decode() if frame else "",
            "timed_out": timed_out, "rest_z": rest_z}


def run(spec):
    global events_enabled
    if "events_enabled" in spec:
        events_enabled = bool(spec["events_enabled"])
    if spec["do"] != "observe":
        try:
            c.simPause(False)
            _run_flight(spec)
        finally:
            c.simPause(True)
    return paused_snapshot()

"""


def advanced_controller(source: str) -> str:
    """Extend the deployed controller without altering legacy family behavior."""
    anchor = "threading.Thread(target=watchdog, daemon=True).start()"
    if source.count("def run(spec):") != 1 or source.count(anchor) != 1:
        raise ValueError("unrecognized controller source")
    updated = source.replace("def run(spec):", "def _run_flight(spec):", 1)
    updated = updated.replace('time.sleep(spec["hold_s"])', 'wait_sim_seconds(spec["hold_s"] + 0.2)')
    # Take a fresh frame only after freezing the world and synchronizing events.
    # Anchored on a marker comment, not on the state-reading expression: that expression
    # changed when landings stopped being graded after the disarm, and this cut silently
    # stopped matching, which would have broken every scenario family.
    marker = "    # FINAL READING."
    if updated.count(marker) != 1:
        raise ValueError("controller source has no single final-reading marker")
    start = updated.index(marker, updated.index("def _run_flight"))
    end = updated.index("\n\n\n" + anchor, start)
    updated = updated[:start] + "    return None\n" + updated[end:]
    updated = updated.replace(anchor, EVENT_SYNC_SOURCE + "\n" + WRAPPER_SOURCE + anchor, 1)
    compile(updated, "advanced_visual_controller.py", "exec")
    return updated


def advanced_recorder(source: str) -> str:
    """Keep wall-time coverage and add independent absolute simulator seconds."""
    anchor = '"t": round(time.time() - t0, 2), "x":'
    if source.count(anchor) != 1:
        raise ValueError("unrecognized recorder source")
    return source.replace(anchor, '"t": round(time.time() - t0, 2), "sim_t": int(s.timestamp) / 1e9, "x":')

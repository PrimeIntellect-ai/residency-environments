"""Sit between a flight program and the simulator, passing only flight calls.

Usage: python3 rpc_proxy.py [listen_port] [sim_port]

The submitted program talks to the listen port. Calls on the allowed
list go through to the simulator. Anything else comes back as an error
and never reaches the simulator, so a program that breaks out of Python
still cannot move the drone by hand or change the world.
"""

import socket
import socketserver
import sys
import threading

import msgpack

LISTEN_PORT = 41452
SIM_HOST = "127.0.0.1"
SIM_PORT = 41451

# Every call the drone API makes, and nothing else. Derived from the client methods used
# in drone.py, and a test keeps the two in step: a call the API makes but this list omits
# is refused mid-flight, so the same program behaves differently under grading than it
# does in the agent's own sandbox.
ALLOWED_CALLS = {
    # connection handshake
    "ping",
    "getServerVersion",
    "getMinRequiredClientVersion",
    # control
    "enableApiControl",
    "armDisarm",
    "takeoff",
    "hover",
    "moveToZ",
    "moveToPosition",
    "moveByVelocity",
    "moveByVelocityZ",
    "moveByRollPitchYawThrottle",
    "moveByAngleRatesThrottle",
    "moveOnPath",
    # reading and camera
    "getMultirotorState",
    "simGetImages",
    "simSetCameraPose",
}

# Calls the drone API can make that the proxy refuses on purpose. A graded program may
# not reset the world or put the drone somewhere by hand: that is how it would fake a
# finished mission. Listed so that a new call in drone.py has to be classified rather
# than quietly refused.
REFUSED_ON_PURPOSE = {
    "reset",
}

REQUEST = 0
RESPONSE = 1


def split_frames(unpacker, pending: bytearray, base: int) -> tuple[list[tuple[object, bytes]], int]:
    """Every complete message in the unpacker, each with exactly its own bytes.

    pending holds the bytes not yet forwarded, base is the stream offset of
    pending[0]. Messages are taken one at a time so a forbidden call that
    arrives in the same read as an allowed one is cut out on its own.
    """
    frames = []
    for message in unpacker:
        end = unpacker.tell()
        frames.append((message, bytes(pending[: end - base])))
        del pending[: end - base]
        base = end
    return frames, base


def deny(msgid, method: str) -> bytes:
    return msgpack.packb([RESPONSE, msgid, f"{method} is not allowed", None], use_bin_type=True)


def allowed_request(message) -> tuple[bool, object, str]:
    """Whether this frame is a request the drone API is allowed to make.

    Returns (allowed, message id, method name). Anything that is not a
    plain allowlisted request is refused, including notifications and
    frames that do not look like requests at all.
    """
    if not isinstance(message, (list, tuple)) or len(message) != 4:
        return False, None, "malformed frame"
    if message[0] != REQUEST:
        return False, None, f"message type {message[0]!r}"
    method = message[2]
    if isinstance(method, bytes):
        method = method.decode("utf-8", "replace")
    if not isinstance(method, str):
        return False, message[1], "unreadable method name"
    return method in ALLOWED_CALLS, message[1], method


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        client = self.request
        try:
            sim = socket.create_connection((SIM_HOST, SIM_PORT), timeout=30)
        except OSError:
            client.close()
            return
        stop = threading.Event()

        def sim_to_client() -> None:
            try:
                while not stop.is_set():
                    data = sim.recv(65536)
                    if not data:
                        break
                    client.sendall(data)
            except OSError:
                pass
            finally:
                stop.set()

        threading.Thread(target=sim_to_client, daemon=True).start()

        unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
        pending = bytearray()
        base = 0
        try:
            while not stop.is_set():
                data = client.recv(65536)
                if not data:
                    break
                pending += data
                unpacker.feed(data)
                try:
                    frames, base = split_frames(unpacker, pending, base)
                except Exception:
                    print("DENIED unreadable stream", flush=True)
                    break
                for message, raw in frames:
                    ok, msgid, method = allowed_request(message)
                    if ok:
                        print(f"ALLOWED {method}", flush=True)
                        sim.sendall(raw)
                    else:
                        print(f"DENIED {method}", flush=True)
                        if msgid is not None:
                            client.sendall(deny(msgid, method))
                        else:
                            stop.set()
                            break
        except OSError:
            pass
        finally:
            stop.set()
            sim.close()
            client.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    global SIM_PORT
    listen = int(sys.argv[1]) if len(sys.argv) > 1 else LISTEN_PORT
    if len(sys.argv) > 2:
        SIM_PORT = int(sys.argv[2])
    with Server(("127.0.0.1", listen), Handler) as server:
        print(f"PROXY_READY listen={listen} sim={SIM_PORT}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()

"""
CARLA sandbox pool utilities.

Manages Prime sandboxes for running CARLA servers, mapped to local loopback IPs
via pproxy. This is the default execution mode.
"""

from __future__ import annotations

from .pool import CarlaSandboxConfig, CarlaSandboxPool, SandboxReservation

__all__ = [
    "CarlaSandboxConfig",
    "CarlaSandboxPool",
    "SandboxReservation",
]

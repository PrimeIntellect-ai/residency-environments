from __future__ import annotations

import asyncio
import random
from typing import Optional

import carla

from ..compat import safe_stop_sensor
from ..logging import get_logger
from .world import WorldManager

logger = get_logger("core.actors")


def safe_destroy(actor: carla.Actor | None, *, name: str = "actor") -> bool:
    """Destroy an actor, returning True if the actor is gone (destroyed or already dead)."""
    if actor is None:
        return False
    try:
        if not actor.is_alive:
            return True  # Already dead — nothing to clean up
        safe_stop_sensor(actor)
        actor.destroy()
        return True
    except Exception:  # noqa: BLE001 - CARLA can throw various runtime errors here
        return False


class ActorManager:
    """
    Tracks actors spawned by this environment and cleans them up reliably.
    """

    def __init__(self, world: WorldManager):
        self.world_manager = world
        self._actors: list[carla.Actor] = []
        self._sensors: list[carla.Actor] = []

    @property
    def world(self) -> carla.World:
        return self.world_manager.world

    @property
    def blueprints(self) -> carla.BlueprintLibrary:
        return self.world_manager.blueprint_library

    def cleanup_tracked(self) -> None:
        import time as _time

        for sensor in list(reversed(self._sensors)):
            safe_stop_sensor(sensor)

        if self._sensors:
            _time.sleep(0.3)

        all_actors = list(reversed(self._sensors)) + list(reversed(self._actors))
        if all_actors:
            survivors: list[carla.Actor] = []
            batch_targets: list[carla.Actor] = []
            cmds: list[carla.command.DestroyActor] = []
            for actor in all_actors:
                try:
                    cmds.append(carla.command.DestroyActor(actor.id))
                    batch_targets.append(actor)
                except Exception:
                    if not safe_destroy(actor, name="actor"):
                        survivors.append(actor)
            try:
                if batch_targets:
                    results = self.world_manager.client.client.apply_batch_sync(cmds, True)
                    for actor, response in zip(batch_targets, results):
                        if getattr(response, "error", None) and not safe_destroy(
                            actor, name="actor"
                        ):
                            survivors.append(actor)
                    for actor in batch_targets[len(results) :]:
                        if not safe_destroy(actor, name="actor"):
                            survivors.append(actor)
            except Exception:
                for actor in batch_targets:
                    if not safe_destroy(actor, name="actor"):
                        survivors.append(actor)

            survivor_ids = {id(actor) for actor in survivors}
            if survivor_ids:
                logger.warning(
                    "Failed to destroy %d tracked actor(s); keeping them for a later retry.",
                    len(survivor_ids),
                )
                self._sensors = [actor for actor in self._sensors if id(actor) in survivor_ids]
                self._actors = [actor for actor in self._actors if id(actor) in survivor_ids]
            else:
                self._sensors.clear()
                self._actors.clear()
        else:
            self._sensors.clear()
            self._actors.clear()

    async def cleanup_tracked_async(self) -> None:
        await asyncio.to_thread(self.cleanup_tracked)

    def cleanup_world(self) -> None:
        """
        Best-effort cleanup of common leftover actors to keep runs deterministic.

        This is intentionally aggressive: assumes the CARLA server is dedicated to
        this environment runner.
        """
        try:
            actors = self.world.get_actors()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to list world actors for cleanup: %s", e)
            return

        destroy_types = ("vehicle.", "walker.", "sensor.")
        to_destroy: list[carla.Actor] = []
        for actor in actors:
            try:
                if any(actor.type_id.startswith(p) for p in destroy_types):
                    to_destroy.append(actor)
            except Exception:
                continue

        if not to_destroy:
            return

        # Sandbox mode often has higher latency; batch-destroy reduces RPC round trips.
        try:
            for a in to_destroy:
                try:
                    if a.type_id.startswith("sensor."):
                        safe_stop_sensor(a)
                except Exception:
                    pass

            client = self.world_manager.client.client
            cmds = [carla.command.DestroyActor(a.id) for a in to_destroy]
            results = client.apply_batch_sync(cmds, True)
            destroyed = 0
            failed: list = []
            for a, r in zip(to_destroy, results):
                if getattr(r, "error", None):
                    failed.append(a)
                else:
                    destroyed += 1
            if destroyed:
                logger.info("Cleaned up %d leftover actors (batch)", destroyed)
            to_destroy = failed  # fall through to per-actor retry for failures
        except Exception:
            pass

        destroyed = 0
        for actor in to_destroy:
            if safe_destroy(actor, name=getattr(actor, "type_id", "actor")):
                destroyed += 1
        if destroyed:
            logger.info("Cleaned up %d leftover actors", destroyed)

    def spawn_vehicle(
        self,
        blueprint_filter: str = "vehicle.lincoln.mkz",
        transform: Optional[carla.Transform] = None,
    ) -> carla.Vehicle:
        bps = list(self.blueprints.filter(blueprint_filter))
        if not bps:
            # Many CARLA docker images include a reduced vehicle set. Fall back to a
            # commonly-available blueprint and then any vehicle.
            fallbacks = []
            if blueprint_filter != "vehicle.lincoln.mkz":
                fallbacks.append("vehicle.lincoln.mkz")
            fallbacks.append("vehicle.*")
            for fb in fallbacks:
                bps = list(self.blueprints.filter(fb))
                if bps:
                    logger.warning(
                        "Vehicle blueprint filter %r not found; falling back to %r",
                        blueprint_filter,
                        fb,
                    )
                    break
        if not bps:
            raise RuntimeError(f"No vehicle blueprints found for filter: {blueprint_filter}")

        bp = bps[0]
        if transform is None:
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points available in map")
            transform = spawn_points[0]

        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        if not isinstance(actor, carla.Vehicle):
            raise RuntimeError(f"Spawned actor is not a Vehicle: {actor.type_id}")
        self._actors.append(actor)
        return actor

    def spawn_pedestrian(
        self,
        transform: carla.Transform,
        blueprint_filter: str = "walker.pedestrian.*",
    ) -> Optional[carla.Actor]:
        bps = list(self.blueprints.filter(blueprint_filter))
        if not bps:
            return None
        bp = bps[0]
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is not None:
            self._actors.append(actor)
        return actor

    def spawn_npc_vehicle(
        self,
        transform: carla.Transform,
        blueprint_filter: str = "vehicle.*",
        autopilot: bool = True,
    ) -> Optional[carla.Actor]:
        bps = list(self.blueprints.filter(blueprint_filter))
        if not bps:
            return None
        bp = random.choice(bps)
        actor = self.world.try_spawn_actor(bp, transform)
        if actor is None:
            return None
        self._actors.append(actor)
        if autopilot:
            try:
                tm_port = self.world_manager.client.config.get_tm_port()
                try:
                    actor.set_autopilot(True, tm_port)
                except TypeError:
                    actor.set_autopilot(True)
            except Exception:
                safe_destroy(actor, name="npc_vehicle")
                if actor in self._actors:
                    self._actors.remove(actor)
                return None
        return actor

    def spawn_sensor(
        self,
        blueprint_id: str,
        transform: carla.Transform,
        attach_to: carla.Actor,
        attributes: Optional[dict[str, str]] = None,
    ) -> carla.Actor:
        bp = self.blueprints.find(blueprint_id)
        if attributes:
            for key, value in attributes.items():
                bp.set_attribute(key, str(value))
        actor = self.world.spawn_actor(bp, transform, attach_to=attach_to)
        self._sensors.append(actor)
        return actor

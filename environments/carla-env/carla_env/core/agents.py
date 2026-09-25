from __future__ import annotations

from typing import Optional, Tuple, Type

from ..logging import get_logger

logger = get_logger("core.agents")

_agents_loaded = False
_BasicAgent: Optional[Type] = None
_BehaviorAgent: Optional[Type] = None


def import_carla_agents() -> Tuple[Optional[Type], Optional[Type]]:
    global _agents_loaded, _BasicAgent, _BehaviorAgent

    if _agents_loaded:
        return _BasicAgent, _BehaviorAgent

    try:
        from carla_env._carla_agents.navigation.basic_agent import BasicAgent
        from carla_env._carla_agents.navigation.behavior_agent import BehaviorAgent

        _BasicAgent = BasicAgent
        _BehaviorAgent = BehaviorAgent
        _agents_loaded = True
        logger.info("Loaded CARLA agents (BasicAgent/BehaviorAgent)")
        return BasicAgent, BehaviorAgent
    except Exception as e:  # noqa: BLE001
        _agents_loaded = True
        logger.warning("Failed to import CARLA agents: %s", e)
        return None, None

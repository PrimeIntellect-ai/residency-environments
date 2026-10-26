"""Register oracle facts privately; ordinary model messages never contain them."""

from __future__ import annotations

import math
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import field_validator
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.null import NullHarness

Policy = Literal["honest", "blind_shortcut", "broken", "always_land", "always_hold", "always_b"]
ADVANCED_FAMILIES = ("alternate_pad", "patrol_inspection", "go_around", "compound")


def _coordinates(value, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} needs two finite coordinates")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{name} needs two finite coordinates")
    return list(map(float, value))


class RehearsalHarnessConfig(HarnessConfig):
    id: str = "aerialsim-rehearsal"
    policy: Policy = "honest"
    endpoint: str = "http://127.0.0.1:8123"

    @field_validator("endpoint")
    @classmethod
    def local_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.path not in ("", "/")
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("rehearsal endpoint must be a loopback HTTP origin")
        # Also reject invalid port strings before any request can be made.
        _ = parsed.port
        return value.rstrip("/")


class RehearsalHarness(NullHarness, Harness[RehearsalHarnessConfig]):
    """Use the normal intercepted chat loop with privately registered task identity.

    EvalClient supplies X-Session-ID=trace.id on upstream model requests. The
    scripted server joins that header to this registration, so pairs with
    identical prompts remain distinct under concurrency and request retries.
    """

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url=None):
        # tool_interception_url is passed by newer verifiers and handed straight to the
        # base harness. Taking it by name rather than **kwargs keeps a genuinely unexpected
        # argument an error. Omitting it kills every scripted flight at turn zero on an
        # unexpected keyword argument.
        if ctx.model != "scripted" or getattr(ctx.client, "type", None) != "eval":
            raise ValueError("the rehearsal harness requires the synthetic scripted model and eval client")
        if ctx.client.base_url.rstrip("/") != self.config.endpoint + "/v1":
            raise ValueError("the scripted model endpoint must match the local rehearsal endpoint")
        family = getattr(data, "family", None)
        if family not in ("land_or_hold", "find_pad", "obstacle_bypass", *ADVANCED_FAMILIES):
            raise ValueError("unsupported visual family for scripted pilots")
        if family in ("land_or_hold", "find_pad") and getattr(data, "path", None) is not None:
            raise ValueError("land_or_hold and find_pad scripted pilots do not support path rules")

        registration = {
            "session_id": trace.id,
            "task_key": trace.task.key or trace.task.hash,
            "policy": self.config.policy,
            "family": family,
        }
        if family == "land_or_hold":
            if self.config.policy == "always_b":
                raise ValueError("always_b is only supported for alternate_pad")
            registration["pad"] = _coordinates(getattr(data, "pad", None), "pad")
            if self.config.policy == "honest":
                if not isinstance(data.blocked, bool):
                    raise ValueError("the honest pilot needs a boolean blocked fact")
                registration["blocked"] = data.blocked
        elif family == "find_pad":
            if self.config.policy not in ("honest", "blind_shortcut", "broken"):
                raise ValueError("find_pad supports honest, blind_shortcut and broken only")
            start = _coordinates(getattr(data, "start", None), "start")
            if math.hypot(*start) > 2.0:
                raise ValueError("find_pad start must be within 2 m of launch")
            heading = getattr(data, "heading", None)
            if (
                isinstance(heading, bool)
                or not isinstance(heading, (int, float))
                or not math.isfinite(heading)
                or not -180 <= heading <= 180
            ):
                raise ValueError("find_pad heading must be finite and between -180 and 180 degrees")
            if getattr(data, "blocked", None) is not False:
                raise ValueError("find_pad requires a clear pad")
            pad = _coordinates(getattr(data, "pad", None), "pad")
            registration.update(start=start, heading=float(heading))
            if self.config.policy == "honest":
                registration["pad"] = pad
        elif family == "obstacle_bypass":
            # Import the public geometry contract, never the visual environment
            # or a route inferred from this rollout's hidden obstacle side.
            from aerialsim_flight.obstacle_bypass import route_rules

            if self.config.policy not in ("honest", "blind_shortcut", "broken"):
                raise ValueError("obstacle_bypass supports honest, blind_shortcut and broken only")
            if getattr(data, "path", None) != route_rules():
                raise ValueError("obstacle_bypass requires the standard public route rules")
            pad = _coordinates(getattr(data, "pad", None), "pad")
            if pad != [22.0, 0.0] or getattr(data, "blocked", None) is not False:
                raise ValueError("obstacle_bypass requires its clear public pad at (22, 0)")
            if getattr(data, "start", None) is not None or getattr(data, "heading", 0.0) != 0.0:
                raise ValueError("obstacle_bypass requires the standard launch and heading")
            variant = getattr(data, "variant", None)
            if type(variant) is not int or variant not in (0, 1):
                raise ValueError("obstacle_bypass requires variant 0 or 1")
            registration["pad"] = pad
            if self.config.policy == "honest":
                registration["variant"] = variant

        else:
            from aerialsim_flight.advanced import AdvancedScenario, needs_go_around, public_geometry, route_rules

            if self.config.policy not in ("honest", "blind_shortcut", "broken") and not (
                family == "alternate_pad" and self.config.policy == "always_b"
            ):
                raise ValueError("unsupported advanced scripted policy")
            scenario = AdvancedScenario.model_validate(getattr(data, "scenario", None))
            if scenario.family != family or scenario.variant != getattr(data, "variant", None):
                raise ValueError("advanced scenario identity must match the task")
            if getattr(data, "path", None) != route_rules(scenario):
                raise ValueError("advanced task must use its declared route rules")
            if getattr(data, "start", None) is not None or getattr(data, "heading", 0.0) != 0.0:
                raise ValueError("advanced pilots require the standard launch and heading")
            # This serialization contains the published geometry only. No full
            # scenario, event list, seed, variant, cue or expected target crosses
            # the private registration boundary for blind/broken controls.
            registration["geometry"] = public_geometry(scenario)
            if self.config.policy == "honest":
                registration["variant"] = scenario.variant
                if needs_go_around(scenario):
                    registration["clear_after"] = scenario.clear_after

        # Do not send these facts to an inference provider or follow a redirect.
        async with httpx.AsyncClient(timeout=10.0, trust_env=False, follow_redirects=False) as client:
            response = await client.post(self.config.endpoint + "/register", json=registration)
            response.raise_for_status()
        trace.info["scripted_policy"] = self.config.policy
        trace.info["oracle_access"] = self.config.policy == "honest"
        return await super().launch(
            ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url=tool_interception_url
        )

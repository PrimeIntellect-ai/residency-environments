"""Thin Verifiers adapter for the pinned full Prime Agent executable.

Prime Agent owns its agent loop, Python kernel, compaction, and MCP client.
Verifiers owns provider interception, isolation, laboratory tools, and grading.
"""

import json
from typing import Literal

from pydantic import Field
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness

__all__ = ["PrimeAgentHarness", "PrimeAgentConfig"]


class PrimeAgentConfig(HarnessConfig):
    version: Literal["0.9.5"] = "0.9.5"
    transport: Literal["chat_completions", "responses", "anthropic_messages"] = "chat_completions"
    context_window: int = Field(ge=4096)
    max_response_tokens: int = Field(gt=0)
    thinking: Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"] = "high"
    thinking_format: str = "reasoning_effort"


class PrimeAgentHarness(Harness[PrimeAgentConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True

    async def setup(self, runtime):
        result = await runtime.run(["prime-agent", "--version"], {})
        if result.exit_code or result.stdout.strip() != self.config.version:
            raise RuntimeError(f"Expected preinstalled Prime Agent {self.config.version}")

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        system, prompt = self.resolve_text_prompt(data)
        # Each runtime and home are new. No personal auth, memory, sessions,
        # project code, or skills are copied from the host.
        agent_dir = f"/workspace/.vf-prime-agent-{trace.id}"
        api = {
            "chat_completions": "openai-completions",
            "responses": "openai-responses",
            "anthropic_messages": "anthropic-messages",
        }[self.config.transport]
        base_url = endpoint.removesuffix("/v1") if self.config.transport == "anthropic_messages" else endpoint
        models = {
            "providers": {
                "physim": {
                    "baseUrl": base_url,
                    "api": api,
                    "apiKey": "PHYSIM_INTERCEPT_KEY",
                    "models": [
                        {
                            "id": ctx.model,
                            "reasoning": self.config.thinking != "off",
                            "input": ["text"],
                            "contextWindow": self.config.context_window,
                            "maxTokens": self.config.max_response_tokens,
                            "compat": {
                                "supportsDeveloperRole": False,
                                "supportsStore": False,
                                "thinkingFormat": self.config.thinking_format,
                            },
                        }
                    ],
                }
            }
        }
        settings = {
            "defaultProvider": "physim",
            "defaultModel": ctx.model,
            "defaultThinkingLevel": self.config.thinking,
            "telemetry": {"enabled": False},
            "bundledSkills": {"websearch": False},
            "autonomous": {key: "unlimited" for key in ("maxContinuations", "maxTurns", "maxTokens", "timeoutMs")},
            "mcpServers": {
                name: {"type": "http", "url": url, "callTimeoutMs": 28800000} for name, url in mcp_urls.items()
            },
        }
        for name, value in (("models", models), ("settings", settings)):
            await runtime.write(f"{agent_dir}/{name}.json", json.dumps(value).encode())
        # Generic MCP is native to this release; no third-party wrapper or
        # schema reimplementation is needed.
        appendix = (system or "") + "\n\nLaboratory access in the Python kernel:\n"
        appendix += "The preimported mcp module provides await mcp.list_tools(server) and await mcp.call_tool(server, name, arguments).\n"
        appendix += "Configured laboratory servers: " + ", ".join(mcp_urls) + ".\n"
        await runtime.write(f"{agent_dir}/task.md", appendix.encode())
        env = {
            **self.config.resolved_env,
            "PHYSIM_INTERCEPT_KEY": secret,
            "PRIME_AGENT_CODING_AGENT_DIR": agent_dir,
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
            "PRIME_AGENT_KERNEL_PYTHON": "/usr/local/bin/python",
            "PI_CACHE_RETENTION": "long",
        }
        trace.info["prime_agent"] = {
            "version": self.config.version,
            "transport": self.config.transport,
            "fresh_state": True,
            "thinking": self.config.thinking,
        }
        return await runtime.run_program(
            [
                "prime-agent",
                "--offline",
                "--mode",
                "json",
                "--provider",
                "physim",
                "--model",
                ctx.model,
                "--thinking",
                self.config.thinking,
                "--append-system-prompt",
                f"{agent_dir}/task.md",
                "-p",
                prompt or "Begin.",
            ],
            env,
        )

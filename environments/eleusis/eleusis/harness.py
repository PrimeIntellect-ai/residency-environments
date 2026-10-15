"""Standard chat harness, with turn-boundary checkpoints for team message routing."""

from verifiers.v1.harness import HarnessSession
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE, launch_chat_program

ELEUSIS_PROGRAM_SOURCE = CHAT_PROGRAM_SOURCE


def _team_program() -> str:
    # Fail closed when upgrading the pinned program: do not silently stop yielding.
    replacements = {
        "    while True:\n        try:\n            completion, messages": "    for _ in range(1):\n        try:\n            completion, messages",
        '    parser.add_argument("--initial-messages-file", default="")': '    parser.add_argument("--initial-messages-file", default="")\n'
        '    parser.add_argument("--team-checkpoint", required=True)\n'
        '    parser.add_argument("--prepare-only", action="store_true")',
        "    # Minimal task images may lack a system CA bundle.": "    checkpoint = Path(args.team_checkpoint)\n"
        "    if checkpoint.exists():\n"
        "        initial = json.loads(checkpoint.read_text()) + initial\n"
        '        args.system_prompt = ""\n'
        "    # Minimal task images may lack a system CA bundle.",
        "        await run_chat_loop(args, compactor, messages, dispatch, servers, tool_client)": "        await run_chat_loop(args, compactor, messages, dispatch, servers, tool_client)\n"
        "        checkpoint.write_text(json.dumps(messages))",
    }
    replacements["        tools += mcp_tools"] = (
        "        tools += mcp_tools\n        if args.prepare_only:\n            return"
    )
    source = CHAT_PROGRAM_SOURCE
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError("Pinned Verifiers chat program changed; review Eleusis team checkpoints")
        source = source.replace(before, after)
    return source


TEAM_PROGRAM_SOURCE = _team_program()


class EleusisHarness(NullHarness):
    """Preserve solo behavior and run team segments inside the configured runtime."""

    async def setup(self, runtime):
        await super().setup(runtime)
        await runtime.prepare_uv_script(TEAM_PROGRAM_SOURCE, self.config.resolved_env)

    async def session(self, *args, **kwargs):
        session = EleusisTeamSession(self, *args, **kwargs)
        if session.data.num_agents == 1:
            return await super().session(*args, **kwargs)
        await session.prepare()
        return session


class EleusisTeamSession(HarnessSession):
    """Retain the full provider wire history, including results of the last tools."""

    async def prepare(self):
        result = await self._launch(None, prepare=True)
        if result.exit_code:
            from verifiers.v1 import HarnessError

            raise HarnessError(f"Eleusis team preparation failed: {result.stderr}")

    async def _run(self, messages):
        return await self._launch(messages)

    async def _launch(self, messages, *, prepare=False):
        system, prompt = self.harness.resolve_prompt(self.data)
        return await launch_chat_program(
            TEAM_PROGRAM_SOURCE,
            self.harness.config,
            self.ctx,
            self.trace,
            self.runtime,
            self.endpoint,
            self.secret,
            self.mcp_urls,
            system,
            prompt if messages is None else messages,
            extra_args=[
                f"--team-checkpoint=.eleusis-team-{self.trace.id}.json",
                *(["--prepare-only"] if prepare else []),
                *([f"--tool-interception-url={self.tool_interception_url}"] if self.tool_interception_url else []),
            ],
        )

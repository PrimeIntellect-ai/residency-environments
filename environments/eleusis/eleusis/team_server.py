"""Cooperative tool server: the original game plus team broadcasts."""

import verifiers.v1 as vf

from .server import EleusisToolset


class EleusisTeamToolset(EleusisToolset):
    @vf.tool
    async def send_message(self, message: str) -> str:
        """Broadcast text to every other teammate before their next model call."""
        self.state.outbox.append(message)
        return "Message queued for teammates."


if __name__ == "__main__":
    EleusisTeamToolset.run()

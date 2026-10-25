"""Print the current turn and how many are left."""

import json
from pathlib import Path

state = json.loads((Path(__file__).parent / ".turns.json").read_text())
turn, max_turns = state["turn"], state["max_turns"]
if max_turns is None:
    print(f"turn {turn}; no turn limit")
else:
    print(f"turn {turn} of {max_turns}; {max_turns - turn} left after this one")

# AgentBoard Provenance

The self-contained BabyAI runtime and assets were frozen from AgentBoard commit:

```text
bb7255e2daf1989069a186dad9e53f70680961db
```

Source checksums before extraction:

| AgentBoard source | SHA-256 |
| --- | --- |
| `agentboard/environment/babyai_env.py` | `9c6ed31d492772ff902d8bddcb4096199fb8a17d866b5d1841f173b1ebbf6b33` |
| `data/babyai/test.jsonl` | `d63d3dec56a623ea1c381a0815b31d6cfc3f707c5693c6d66e49b47c053c25bc` |
| `agentboard/prompts/VanillaAgent/babyai_vanilla_prompt.json` | `1a7e040e67617c8556beab948c78e4ca4d78494fa96f4fe9b6b02c118a208068` |

The port intentionally preserves AgentBoard's textual observations, high-level
action expansion, action validation, progress reward, completion behavior,
task annotations, prompt, and seed mapping. Its parity suite compares all 112
episodes against the frozen AgentBoard checkout.

Two compatibility adaptations are localized in the packaged runtime. MiniGrid
internal attributes are read through Gymnasium's `env.unwrapped`, while
`reset()` and `step()` continue to use the wrapped environment. AgentBoard's
2D `numpy.cross` test is expressed directly as the equivalent scalar
determinant because newer NumPy releases reject two-element vector inputs.

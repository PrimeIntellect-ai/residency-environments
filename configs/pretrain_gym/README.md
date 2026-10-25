# pretrain-gym profiles

Eval profiles for [`pretrain-gym`](../../environments/pretrain_gym/README.md).

| File                                     | Budget | Turns | Harness     | Agent / scorer              |
| ---------------------------------------- | ------ | ----- | ----------- | --------------------------- |
| `smoke.toml`                             | 20M    | 40    | bash        | Docker / host GPU           |
| `smoke-prime.toml`                       | 20M    | 40    | bash        | Prime CPU VM / Prime GPU VM |
| `glm-5.3-400M-500turn-pi.toml`           | 400M   | 500   | Pi          | Docker / host GPU           |
| `kimi-k3-400M-500turn-kimi-code.toml`    | 400M   | 500   | Kimi Code   | Docker / host GPU           |
| `kimi-k3-400M-500turn-prime-agent.toml`  | 400M   | 500   | Prime Agent | Docker / host GPU           |
| `gpt-6.1-sol-400M-500turn-codex.toml`    | 400M   | 500   | Codex       | Docker / host GPU           |
| `opus-5.5-400M-500turn-claude-code.toml` | 400M   | 500   | Claude Code | Docker / host GPU           |

All of them set `allow = []` on the agent runtime, so the workspace reaches only the harness's model and tool routes. They read the gated Paloma set with `SCORER_HF_TOKEN` and give the agent's Hub tools `HF_TOKEN`. The 400M profiles cap Hub downloads at 500 GB, since they land on the GPU host's disk; raise or lower `hub.max_download_gb` to fit the host. Change `model` or `[env.agent.harness]` to evaluate another model or harness. The Claude Code profile calls the Anthropic API with `CLAUDE_API_KEY` and `CLAUDE_WORKSPACE_ID` and pins Claude Code 2.1.292 (Opus 5.5 needs 2.1.280 or newer). Its `WebSearch` and `WebFetch` tools stay enabled but cannot reach the web: under the empty allowlist, verifiers strips Anthropic's server-side web tools from model requests.

```bash
uv run vf-eval @ configs/pretrain_gym/smoke.toml
```

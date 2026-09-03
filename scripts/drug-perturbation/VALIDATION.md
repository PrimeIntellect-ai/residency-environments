# Live integration smoke test

Validated on 2026-09-03 with Verifiers 0.3.1, Docker 28.3.3, and environment
implementation commit `436a84b`. This is a functionality check on one prompt
with two responses, not a performance estimate or judge-quality evaluation.

## Configuration

```bash
uv run eval @ configs/drug-perturbation/eval.toml \
  @ configs/drug-perturbation/process-reward.toml \
  -n 1 -r 2 -c 2 -m qwen/qwen3.5-35b-a3b --no-push
```

The policy was the available base-model ID, not a trained adapter. The judge
was `google/gemini-3-flash-preview`, with the unchanged
`reasoning-process-v0.3` rubric, temperature 0, and 900 output tokens.
Policy sampling temperature was not overridden; the output limit was 16,384
tokens with at most five turns. The default test task selection and seed 42
used the pinned Hugging Face dataset revision
`49d20f9f293fea3f4c93ab57b478aaa3f1651d84`.

## Observed results

| Response | Deterministic D | Process J | D × J |
| --- | ---: | ---: | ---: |
| 1 | 0.675000 | 0.908710 | 0.613379 |
| 2 | 0.600000 | 1.000000 | 0.600000 |

Both responses completed normally in two model turns, each with one
`identify_compound` call. Both had complete answer tags and no rollout errors
or token-limit termination. Both judge calls produced parseable verdicts with
valid evidence locations on the first attempt; neither failed open. Judge
latencies were 5.47 and 4.47 seconds. Reported usage cost was $0.0037 for the
policy and $0.0087 for the judge ($0.0124 combined; provider-reported values).

Local rescoring of these same traces under the default D-only configuration
returned the original D values, made zero judge calls, and left J unavailable.
The temporary evaluation workers and containers were cleaned up.

The observed attenuation verifies reward wiring, not whether the judge's
biological assessment is correct. This does not establish Hosted Lab training
compatibility, alternative-harness isolation, or stochastic trajectory parity
with earlier environment versions. Raw traces are not bundled with this
environment contribution.

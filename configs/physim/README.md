# PhySim configurations

`eval.toml` runs the pinned BF centered-apparatus preparation by default.
`bf_trail_lab.toml`, `xv_rotor_lab.toml`, and `p4g2_044.toml` select one current
preparation explicitly at HF commit `552229e61813b5684be2051349d778acea054922`.
All use the stock Bash harness and Prime runtimes, with generous investigation
limits. Override `model` on the CLI. No config scans for additional worlds.

```sh
uv run --no-sync vf-eval @ configs/physim/eval.toml --model YOUR_MODEL_ID
```

A local bundle overrides the BF default through `env.taskset.task.tools.bundle`.
The published-world configs have an explicit `bundle_source`; remove that block
before selecting a local bundle. `bundle_source.offline=true` requires a populated
verified cache. The host downloads data; it never gives the bundle to the agent.

All configs disable result uploads. `--dry-run` validates configuration without
model calls; the full smoke script in the environment README additionally checks
the selected data and complete grading path. A short model wiring check uses
`-n 1 -r 2 --env.agent.max-turns 4`.

Select Docker locally with both `--env.agent.runtime.type docker` and
`--env.taskset.task.tools.predictor-runtime.type docker`.
`release.toml` records dataset publication provenance and licenses.

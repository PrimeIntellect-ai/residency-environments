# PhySim configurations

`eval.toml` runs all three preparations in the pinned evaluation catalog by default.
`bf_trail_lab.toml`, `xv_rotor_lab.toml`, and `p4g2_044.toml` select one current
preparation explicitly at HF commit `bd77a0da2f14eef352bd80c4a38dff426e5c1bed`.
All use the stock Bash harness and Prime runtimes, with generous investigation
limits. Override `model` on the CLI. `-r 3` repeats each selected preparation three
times; with the default catalog this runs nine investigations. `-n` limits the
number of tasks and does not manufacture additional tasks or repetitions.

```sh
uv run --no-sync vf-eval @ configs/physim/eval.toml --model YOUR_MODEL_ID
```

A local bundle selects one task through `env.taskset.task.tools.bundle`.
The published-world configs have an explicit `bundle_source`; remove that block
before selecting a local bundle. `bundle_source.offline=true` requires a populated
verified cache. The host downloads data; it never gives the bundle to the agent.
For a subset, use `env.taskset.bundles`, a list of local paths or remote
`{repo, revision, path}` records. For another complete catalog, set
`env.taskset.catalog.repo` and its full `revision`. See the environment README.

All configs disable result uploads. `--dry-run` validates configuration without
model calls; the full smoke script in the environment README additionally checks
the selected data and complete grading path. A short model wiring check uses
`-n 1 -r 2 --env.agent.max-turns 4`.

Select Docker locally with both `--env.agent.runtime.type docker` and
`--env.taskset.task.tools.predictor-runtime.type docker`.
`release.toml` records dataset publication provenance and licenses.

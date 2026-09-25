# Physim configurations

`p4g2_044.toml`, `bf_trail_lab.toml`, and `xv_rotor_lab.toml` each select one
specific evaluation preparation at the verified HF commit
`dcd6abd5eae76a47f326c70518315d2d1e101d86`. They use stock Verifiers' bash harness,
Docker runtime, and generous investigation limits. Override `model` on the CLI.
Data are fetched and checked by the trusted host before model execution. Set
`env.taskset.task.tools.bundle_source.offline = true` to require a populated cache.

`eval.toml` is the local-bundle variant: pass a model ID and
`--env.taskset.task.tools.bundle /path/to/bundle`. No config scans the registry or
implicitly selects additional eval-ready entries. A missing world is an error.

All configs disable result uploads. `--dry-run` checks configuration without a
model call, but does not load or verify physical data. A wiring smoke uses
`-n 1 -r 2 --env.agent.max-turns 4`; use the full config for scientific rollouts.


`release.toml` records the published dataset revision and its licenses. It is
release metadata, not an implicit runtime world-selection default.

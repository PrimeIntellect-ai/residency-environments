# AnomalyXL-precise

Evidence-grounded long-context anomaly QA as a Verifiers v1 taskset.

## Overview

AnomalyXL-precise generates synthetic multivariate time-series with injected
anomalies and scores the model's JSON answer against evidence-grounded gold.
Every answer is tied to a specific time span, channel, and magnitude, recovered
exactly from the synthesis metadata rather than from an LLM rating.

The environment is decoupled from the harness: any v1 harness that executes
code (bash, codex, ...) can play it. The taskset provides tasks and scoring;
the harness provides the agent loop.

## Sandbox

Tasks run in a digest-pinned image carrying numpy, scipy and matplotlib, with
outbound network blocked. Nothing is installed during a rollout, so solvability
does not depend on the harness's default image or on network conditions. The
image is built per architecture — `SANDBOX_IMAGE` in `taskset.py` points at the
x86 build, with the arm64 digest alongside it.

## Task families

Five precise (open-ended, JSON) question families, each scored by a continuous
metric in [0, 1]:

| Category | Answer | Primary metric |
|---|---|---|
| `localize` | `{"present": bool, "start": int, "end": int}` | IoU (or correct empty) |
| `classify_with_evidence` | `{"kind": str, "start": int, "end": int}` | kind × IoU |
| `measure_magnitude` | `{"magnitude_sigma": float}` | 1 − rel_err / 0.5 |
| `localize_all_channels` | `{"anomalies": [{channel, start, end}, ...]}` | Set F1 |
| `lead_lag_with_magnitude` | `{"direction": str, "lag_samples": int}` | Composite |

## Task source

Tasks are generated on the fly. Each one is deterministic given
`(generator.run_seed, index)`, so a fixed `run_seed` and the task index define a
stable benchmark without shipping any data: the series is regenerated in `setup`
from coordinates that ride on the task, and never crosses the wire itself.

## Design

A sample starts from a **scene**: C independent channels, each a base noise
process (white, red, or AR(1)) plus optional periodic and trend components,
z-normalized to unit std. Channels are anonymized (`ch_00`, `ch_01`, ...) so
no name priors leak.

An **anomaly** is one of five transforms — level shift, transient spike,
change in seasonality, change in variance, change in trend — applied to a
contiguous window of one channel after z-normalization. Anomaly width is an
absolute sample count (8–256), independent of L, so at L=131,072 an anomaly
can occupy as little as 0.006% of the series.

The series is written to `context.json` in the runtime workspace, with values
rounded to 3 decimals. The model loads it and analyzes it with whatever tools its harness
provides. The model's final reply is scored against the gold.

## Generator difficulty knobs

All knobs live on `--env.taskset.generator.*`:

| Knob | Default | Description |
|---|---|---|
| `num_tasks` | 100,000 | Size of the lazy index space |
| `run_seed` | 0 | Rotates the whole stream for fresh data |
| `cells` | 5 cells | (category, channels, lengths, weight) grid; a task draws a cell by weight, then a length uniformly within it |
| `magnitude_range` | (2.0, 6.0) | Anomaly magnitude in σ-units (smaller = harder) |
| `width_range` | (8, 256) | Anomaly width in samples (narrower = harder) |
| `width_frac_range` | None | Relative width as fraction of L (overrides width_range) |
| `magnitude_log_range` | (1.0, 100.0) | measure_magnitude answer range (log-uniform) |
| `noise_weights` | None | Base-noise bias over {white_noise, red_noise, ar1} |
| `max_k` | 3 | Max anomalies for localize_all_channels |
| `min_k` | 0 | Min anomalies for localize_all_channels (1 drops the empty-gold rows) |
| `lag_max_fraction` | 0.25 | Max coupling lag as fraction of L |
| `p_present` | 0.8 | Probability localize injects an anomaly |

### Sub-selecting categories

To run only `localize` at a single length:

```toml
[env.taskset.generator]
cells = [{category = "localize", channels = 1, lengths = [32768]}]
```

### Rebalancing the mix

Cells are equally likely by default. Raise `weight` to over-sample a category,
or set it to 0 to drop one:

```toml
[env.taskset.generator]
cells = [
  {category = "localize", channels = 1, lengths = [16384, 32768, 65536, 131072], weight = 1.0},
  {category = "localize_all_channels", channels = 16, lengths = [32768, 131072], weight = 3.0},
]
```

### Adjusting difficulty

Narrower anomalies are harder to localize:

```toml
[env.taskset.generator]
width_range = [8, 64]
magnitude_range = [1.5, 3.0]
```

## Evaluation

```bash
uv run eval anomalyxl_precise -n 1 -r 2 --env.agent.runtime.type docker
```

Tasks carry a pinned image, so the runtime has to be one that provides a
container (`docker`, `prime`, `modal`).

## Implementation

- `anomalyxl_precise/_anomalyxl/` — vendored generator (scene assembly, anomaly
  injection, question templates, pipeline).
- `anomalyxl_precise/scoring.py` — strict-JSON answer parser and per-category
  continuous-metric scoring.
- `anomalyxl_precise/generator.py` — on-the-fly task stream config and
  deterministic sample generation.
- `anomalyxl_precise/taskset.py` — v1 taskset, task data, and scoring hooks.

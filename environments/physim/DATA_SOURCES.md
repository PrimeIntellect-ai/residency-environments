# Code and data

Physim and blobkit code are Apache-2.0. The released reference data are CC-BY-4.0:
the genome, exact prepared fields, apparatus, case programs, score groups, and
retained native realizations. License choices are recorded in
`configs/physim/release.toml` and in each data-bundle manifest.

The [published dataset](https://huggingface.co/datasets/seanpohorence/physim-worlds/tree/552229e61813b5684be2051349d778acea054922) is `seanpohorence/physim-worlds`, verified at
`552229e61813b5684be2051349d778acea054922`. It includes the preserved registry and six evaluation bundles: three historical fixed-source
preparations and three current centered-apparatus preparations.
The current taskset defaults to the pinned centered BF preparation. Explicit
bundle selection overrides that default; no registry scan chooses additional worlds.

To discover the runnable evaluation preparations from the installed package:

```sh
physim catalog --repo seanpohorence/physim-worlds \
  --revision 552229e61813b5684be2051349d778acea054922
```

This command reads `catalog.jsonl`, which lists evaluation bundles only. The HF
dataset's `worlds` viewer and `worlds.jsonl` include all preserved world records;
`evaluations` is the viewer for the evaluation catalog. The dataset card explains
how to inspect the full registry and its recipes with blobkit. Use the matching
config under `configs/physim/` to run an evaluation; it already contains the pinned
revision and bundle path. `simulation` downloads omit grading truths, while
`evaluation` downloads include them. See the environment README for setup and
the dataset card for the distinction between worlds, preparations, and suites.

These are disclosed reference/development preparations: the original p4g2_044
example, BF's trail laboratory, and XV's rotating-pair laboratory.
The host owns its fields, apparatus mapping, selected programs, and grader truths.
Those inputs are not included in the solver image or exposed through its harness.
Future held-out suites must use distinct preparations and explicit access controls.

To propose another world or preparation, follow the dataset's
[contribution guide](https://huggingface.co/datasets/seanpohorence/physim-worlds/blob/main/CONTRIBUTING.md).
It uses reviewed HF pull requests and distinguishes preserved-world requirements
from eval-ready validation. New data publication does not change these pinned
evaluation configs; adopting a preparation requires an explicit config update.

Original source hashes, the initial-field hash, and all retained-truth hashes are
recorded in the bundle. The numerical source files remain byte-identical to the
validated reference. The original migration exporter retains its existing truth;
new preparation recipes freeze their suites before generating fresh truth. Third-party
Python dependencies retain their own licenses.

# Physim world preparation and validation

These workflows use installed Blobkit and Physim, with data from an explicitly
pinned public HF registry. The environment never imports this directory.

From the repository root with Python 3.12:

```sh
uv venv --python 3.12
uv pip install -e './environments/physim[reference,hub]'
uv run blobkit generate generators/physim/recipes/small_search.py --registry outputs/small-search
uv run blobkit registry verify outputs/small-search
```

The small search demonstrates custom Python measurements, fitness, variation,
selection, and harvesting. Its short-run mass metric is a workflow example,
not a quality criterion for evaluation worlds. Blobkit records recipe sources,
seeds, lineage, and checkpoints. The shared library and its callback documentation
live in the [Blobkit repository](https://github.com/swpo/physim/tree/main/packages/blobkit).

For BF and XV, follow the [preparation workflow](EVALUATION_WORKFLOW.md): fetch
archived inputs, prepare the apparatus, run fresh science observations and
mechanism controls, freeze a suite, generate independent truth and diagnostic
forecasts, validate the native service/reference grader, then archive the result.
All inputs are explicit; the personal research checkout is unnecessary.

`fetch_inputs.py` restores saved simulation endpoints and provenance without
executing downloaded code. The recipes generate fresh continuations from those
endpoints. They do not reconstruct the original evolutionary search or the
first 2,500 time units; that historical evidence stays in the registry. The older
p4g2_044 reference remains available as its exact published bundle, including
its original retained truths. These are concrete BF/XV recipes; preparing another
genome requires deliberate apparatus, program, and score choices.

`register_evaluation.py` archives a locally validated result and reconstructs its
bundle to verify the round trip. `export_registry_catalog.py` regenerates the
local browsing index; it does not upload data. World publication uses reviewed
[HF contributions](https://huggingface.co/datasets/seanpohorence/physim-worlds/blob/main/CONTRIBUTING.md).
Use a new registry/output for new runs and preserve released identities.

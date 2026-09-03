# Dataset packaging

The environment downloads its frozen examples from
[seanpohorence/drug-perturbation](https://huggingface.co/datasets/seanpohorence/drug-perturbation).
Ordinary evaluation does not require running this helper.

To reproduce the train/test export from the original public
`smallmol_chain_examples.parquet` in
[drug-perturbation-rl](https://github.com/swpo/drug-perturbation-rl):

```bash
uv pip install -e ./environments/drug-perturbation
uv run scripts/drug-perturbation/prepare_dataset.py \
  --source /path/to/smallmol_chain_examples.parquet \
  --output /path/to/new-export-directory
```

The helper validates the source checksum, schema, row counts, references,
and compound-disjoint splits. It preserves row order and duplicate
multiplicity, checks the Parquet round trip, and writes a manifest and data
card. It refuses to overwrite an existing directory and does not publish
anything. Parquet byte checksums can depend on the writer version; the
published artifact and loader hashes are the frozen benchmark authority.

The source construction pipeline and data citations remain in the upstream
public repository and the environment's `DATA_SOURCES.md`. No private reports,
training rollouts, or evaluation answer caches belong in this contribution.

Reusable environment configurations are in `configs/drug-perturbation/`.
Follow the repository's evaluation workflow for live-model validation;
packaging checks or mocked transport checks alone do not establish model
performance or Hosted Lab compatibility.

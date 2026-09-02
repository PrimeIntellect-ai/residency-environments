# eleusis generators

Synthetic data pipeline for the Eleusis rule bank. Rules are reviewed,
parameterized Python predicates over `(card, mainline)` — not arbitrary
programs — so every candidate carries a family, template ID, explicit semantic
parameters, and a complexity tier.

## Pipeline

1. **`build_calibrated_rules_dataset.py`** — generates and validates the
   candidate bank across the eight rule families. Every candidate must pass:
   restricted-AST compilation, at least one valid starter card, acceptance rate
   in `[0.05, 0.95]`, behavioral uniqueness against the existing bank
   (deterministic probe battery over deck × representative mainlines), and
   deterministic equivalence metadata. No model calls are made; this stage is
   entirely local.

2. **`build_rules_dataset.py`** — validates a bank (or the published Hub
   dataset) and rebuilds it as a clean train/test `DatasetDict`, proving via
   behavioral signatures that no test rule leaks into train. Test behaviors
   must be unique; repeated behaviors within train are allowed for functional
   training variations.

The final benchmark selection (which candidates enter the test split) is a
separate, measurement-driven calibration step that is intentionally not part of
this repo: difficulty tiers are chosen against reference-model rollouts, then
frozen as an immutable Hub revision.

The default protocol-v0.7.2 release contains 2,048 train and 512 test rules.
Both splits are balanced across eight families, every family/concept cell has
exactly four times as many train rules as test rules, and the public schema is
limited to `rule_id`, `label`, `family`, and `code`. Calibration evidence and
lineage metadata are stored separately from the task rows on the dataset repo.

## Usage

From the repository root (the `eleusis` environment must be installed):

```bash
uv run python generators/eleusis/build_calibrated_rules_dataset.py \
    --output outputs/eleusis/candidates
uv run python generators/eleusis/build_rules_dataset.py \
    --source nph4rd/eleusis-calibrated --target nph4rd/eleusis-calibrated
```

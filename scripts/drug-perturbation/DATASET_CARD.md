---
language:
- en
license: other
license_name: source-specific-terms
license_link: https://huggingface.co/datasets/seanpohorence/drug-perturbation/blob/main/DATA_SOURCES.md
task_categories:
- text-generation
tags:
- biology
- pharmacology
- reinforcement-learning
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: test
    path: data/test.parquet
---

# Drug perturbation prediction

Frozen prompts and deterministic reference answers from
[drug-perturbation-rl](https://github.com/swpo/drug-perturbation-rl).
This release reorganizes the previously public example table into Hugging Face
train/test files without changing any prompt, answer, split, or assay value.
`manifest.json` records the source and exported file checksums.

## Tasks and splits

There are 186,854 training rows and 20,257 test rows across 12 task views.
The split is compound-disjoint; multiple views and phenotypes for a compound
remain in the same split. Rows are not independent biological experiments.

The standard task (`smiles_only`) asks for targets, mechanism of action,
signed Hallmark pathways, and one phenotype from a SMILES string and assay
context. `from_target`, `from_moa`, and `from_pathways` supply increasing amounts
of intermediate information. Other views ask for a single link or endpoint:
`phenotype_direct`, `target_from_smiles`, `moa_from_smiles`, `moa_from_target`,
`pathways_from_smiles`, `pathways_from_moa`, `phenotype_from_moa`, and
`phenotype_from_pathways`.

Phenotype rows cover `viability`, `cell_cycle`, `stress`, and optional
`magnitude`. Rows without a phenotype answer use the `upstream` marker.
The standard environment default selects `smiles_only` with viability,
cell-cycle, and stress endpoints; it does not use every row in this dataset.

## Fields

- `user_prompt`: the solver-visible question, including requested answer tags.
- `answer_json`: evaluator-only reference fields for the requested outputs.
- `compound`, `cell_line`, `entry_point`, `phenotype`, `split`: identifiers and selectors.
- `lincs_dose_um`, `lincs_time_h`: LINCS assay context.
- `prism_dose_um`, `prism_doses_um`, `prism_duration_h`, `prism_screens`,
  `prism_n_full_ids`, `prism_aggregation`: PRISM assay provenance.

Do not expose reference answers, evaluator caches, or this dataset's download
endpoint to a solver during benchmark evaluation. The frozen table contains
460 exact duplicate rows, whose multiplicity is preserved rather than silently
changing the sampling distribution. Use the row position within the frozen
split (before filtering or shuffling) as part of a unique row key, and the
prompt hash when grouping by identical questions.

## Source data and limitations

Targets and mechanisms derive from Drug Repurposing Hub annotations. Signed
pathways, cell-cycle, stress, and magnitude labels derive from LINCS L1000
expression signatures. Viability uses the separate PRISM Repurposing 24Q2
Extended Primary assay. These are noisy, context-dependent annotations and
derived labels, not a complete or causal account of drug biology. In
particular, PRISM and LINCS are not the same measurement or assay endpoint.

This is a research benchmark, not a source of clinical treatment advice.
The dataset is public and cannot establish contamination-free evaluation of
models whose training data may include it.

## Attribution and licensing

Source-specific terms and citations are in [DATA_SOURCES.md](DATA_SOURCES.md).
The software's Apache-2.0 license does not replace the underlying data terms.
Sources include Subramanian et al. (*Cell*, 2017), Corsello et al.
(*Nature Cancer*, 2020), Corsello et al. (*Nature Medicine*, 2017), and
Liberzon et al. (*Cell Systems*, 2015). No raw source downloads are included.

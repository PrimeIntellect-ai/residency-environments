# midog-mitosis generator

Builds and checks the dataset [`tirandazdylan/midog-mitosis`](https://huggingface.co/datasets/tirandazdylan/midog-mitosis) from MIDOG++ (CC0). Each script declares its own dependencies, so plain `uv run` works. Input dataset revisions and file checksums are fixed in the builder.

| Script                | What it does                                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `build_dataset.py`    | Downloads MIDOG++ from Figshare (69.8 GB, resumable), cuts the fields, chooses the tasks and writes the dataset. |
| `validate_dataset.py` | Rebuilds every task from the source labels without the builder's code, and runs the scorer on it.                |
| `mutation_test.py`    | Plants nine faults in the test tasks; the validator must catch every one.                                        |
| `upload_dataset.py`   | Uploads a complete build to the Hugging Face Hub (private unless `--public`).                                    |

```bash
uv run generators/midog-mitosis/build_dataset.py --work-dir ~/midog-hard512
uv run generators/midog-mitosis/validate_dataset.py --work-dir ~/midog-hard512
uv run generators/midog-mitosis/mutation_test.py --work-dir ~/midog-hard512
uv run generators/midog-mitosis/upload_dataset.py --repo-id <user>/midog-mitosis --dataset ~/midog-hard512/dataset
```

Use a dedicated work directory: downloads and staged slides resume after an interruption, and a resume with different crop or split settings is refused. `--retained-dataset DIR` uses a local copy of the pinned input files instead of downloading them; `--slides N` builds a small partial dataset for testing.

## How the tasks are made

- **Clean labels only.** A field is used only if every labeled cell in it is undisputed: the experts' votes all agree, whether on a mitotic figure or on a look-alike ("imposter").
- **Benchmark and warmup.** Benchmark tasks have three or more figures. Fields with exactly one figure become warmups, unless a benchmark pair already shows them.
- **Fields.** 768 px crops at full resolution, placed around undisputed figures and on a grid, rejected if a disputed cell or a figure cut by the border falls inside, then packed without overlap. Packing prefers three-figure fields, so a few crowded fields don't use up the figures. Other fields fill the remaining area for warmups and support data.
- **Split by slide.** All 111 official MIDOG++ test slides go to test, plus training-pool slides held out in seeded round-robin order across tumor types until test is large enough. Validation takes 64 tasks from the remaining slides; the rest is training. Every config follows this split, recorded in the dataset's `build_report.json`. It is a custom split: make sure a model's training slides don't include the held-out ones.
- **Earlier tasks kept.** The 36 test tasks of the earlier release are kept unchanged (checksum-pinned); the other test tasks come from new fields. No field or figure is used twice, and field ids don't reveal how a crop was chosen.
- **No model in the loop.** Selection uses only labels and fixed seeds, never model results. The three-figure threshold sets difficulty; it does not guarantee a pass rate.

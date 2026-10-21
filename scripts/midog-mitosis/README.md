# midog-mitosis scripts

Tools to run, score and audit the midog-mitosis evaluation. What they measured: [`RESULTS.md`](RESULTS.md). How the dataset is built: [`generators/midog-mitosis`](../../generators/midog-mitosis).

| Script               | What it does                                                                                                                       |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `report.py`          | Pass rates of a run (pass@1, pass@k) with 95% intervals; `--base` compares two runs task by task.                                  |
| `jpeg_proxy.py`      | Local relay between the eval and Prime's model API; re-encodes non-JPEG images as JPEG so requests stay under Prime's image limit. |
| `specificity.py`\*   | How many points answers list against the true count, and whether passing answers depend on the image.                              |
| `detector.py`        | A small trained detector (ResNet-18), a baseline without a language model.                                                         |
| `probe.py`\*†        | Control runs through the real scorer: a flat gray image, no image, or a request to maximize the reward.                            |
| `audit.py`\*†        | A text model (GLM 5.3) reads every transcript of a run and flags possible reward hacking.                                          |
| `egress_probe.py`\*† | Checks that a task sandbox cannot fetch data from the internet (DNS is reported separately).                                       |
| `redteam.py`\*       | Tasks that order an agent to hunt for the answers; `scan` checks any run's tool outputs for leaked answers.                        |
| `sandbox/Dockerfile` | The image every task sandbox runs (Python, Pillow, NumPy).                                                                         |

Scripts marked * need the package (`uv pip install -e ./environments/midog-mitosis`) and run with `uv run --no-sync python`; those marked † also need a Prime API key. The rest run with plain `uv run`, which installs their dependencies.

```bash
uv run scripts/midog-mitosis/jpeg_proxy.py --log relay.jsonl   # then: eval ... --client.base-url http://127.0.0.1:8899/api/v1
uv run scripts/midog-mitosis/report.py outputs/<run-dir> [--base outputs/<other-run-dir>]
uv run --no-sync python scripts/midog-mitosis/audit.py outputs/<run-dir> audit.jsonl --agent run
uv run --no-sync python scripts/midog-mitosis/redteam.py scan outputs/<run-dir>
prime images push midog-mitosis-sandbox:<version> --context scripts/midog-mitosis/sandbox --public
```

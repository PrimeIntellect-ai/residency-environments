# pretrain-gym

An agent builds a pretraining corpus for a fixed language model. The model is trained on the corpus and scored by its cross-entropy on held-out text (Paloma by default).

- **ID:** `pretrain-gym`
- **Tags:** data-curation, pretraining, huggingface, datasets, tool-use, multi-turn, train, eval

## How it works

- **Workspace:** a Docker container or Prime VM with no network and no GPU, and CPU data tools (`datasets`, `polars`, `duckdb`, `datasketch`, fastText `lid.176`, `torch`, `transformers`). `trainer/` holds the training and corpus-preparation code; `python turns.py` prints the turns left.
- **Deliverable:** `corpus/`, made of `.jsonl`, `.jsonl.gz`, `.jsonl.zst`, or `.parquet` files with a `text` field. Training reads the files in sorted order up to `scoring.train_tokens` and repeats a shorter corpus.
- **`hf` tools** (`search`, `repo_info`, `download`, `search_papers`, `read_paper`): Hugging Face datasets, models, and papers as of `hub.cutoff_date`. They run on the host, and its token never enters the workspace.
- **`scorer` tools** (`score`, `score_status`, `cancel_score`): train on a snapshot of `corpus/` and evaluate on the validation split, which the prompt calls the public part of the held-out set. Results give the overall validation loss (never per subset), training and validation curves, and throughput, and are also written to `scores/<job_id>/`.
- **Final score:** after the agent stops, the same training runs on `corpus/` and is evaluated on the test split (the private part).
- **Decontamination:** before any training, every 13-word passage shared with held-out text is cut from the corpus, and documents under 13 words are dropped. On Paloma, at most 0.005% of copied held-out text survives, whether verbatim, chunked, or altered in case or Unicode; honest web data loses 0.05–0.4% of its characters.

## Reward

`reward = p²`, where `p = (ln V − L) / (ln V − target_loss)`, or `p` itself when negative. `L` is the test loss, averaged over subsets if the set has them, and `ln V` is the loss of a uniform prediction. With the default `target_loss = 0`, a uniform model scores 0 and zero loss scores 1. A missing or unreadable corpus, or a diverged run, scores 0.

## Swapping the model or evaluation set

- **Model:** `scoring.model` names a directory in `pretrain_gym/models/`, or a path to one, holding `model.json` (tokenizer and training settings) and `train.py`. The script runs as `train.py --config model.json --data <dir> --out <dir> --train-tokens <n>`, reads token files in which every document starts with BOS, and writes `metrics.jsonl` and `result.json`. A `metrics.jsonl` line with `"evaluating": true` makes the scorer report the final evaluation as its own stage. With `scoring.export_weights`, final scoring adds `--save-weights <path>` and keeps the weights at `<host_dir>/rollouts/<rollout>/final/out/model.pt`. The default, `nanogpt_speedrun`, is a GPT-2-small-sized modded-nanogpt-style model with the SmolLM2 tokenizer. It runs on A100 or H100 GPUs (FlexAttention needs compute capability 8.0 or newer).
- **Evaluation set:** `scoring.eval_set` names a Hugging Face dataset, a file glob with `{split}`, and the validation and test split names. If the set has subsets, list them in `subsets` and put `{subset}` in the glob; their losses are averaged. With `subsets = []`, the set is scored as a whole. `local_dir` points at a host directory with the same layout instead.

## Running

- **Docker on a GPU host:** the agent runs in a container, and scoring runs on the host's GPUs, which rollouts lease one at a time.
- **Prime from a CPU host:** the agent runs in a Prime CPU VM, and scoring runs on a Prime GPU VM that exists only while there is scoring to do. Prime GPU VMs need an account grant.

```bash
# Docker
docker build -f scripts/pretrain_gym/Dockerfile.agent -t pretrain-gym-agent:v1 scripts/pretrain_gym
uv pip install -e "./environments/pretrain_gym[train]"
uv run vf-eval @ configs/pretrain_gym/smoke.toml

# Prime (after `prime login`)
scripts/pretrain_gym/push_prime_images.sh v1
uv pip install -e ./environments/pretrain_gym
uv run vf-eval @ configs/pretrain_gym/smoke-prime.toml
```

The host reads two Hub tokens: `hub.token_env` for the agent's tools and `scoring.eval_set.token_env` for the evaluation set. Give the tools a token without gated-repo access, so they cannot fetch a gated evaluation set.

## Knobs

Under `env.taskset.task.`; every option is documented in `config.py`.

| Key                                       | Default                 | Meaning                          |
| ----------------------------------------- | ----------------------- | -------------------------------- |
| `scoring.train_tokens`                    | `400000000`             | Training tokens                  |
| `scoring.model`                           | `nanogpt_speedrun`      | Model directory                  |
| `scoring.eval_set`                        | Paloma, 12 subsets      | Held-out evaluation set          |
| `scoring.target_loss` / `reward_exponent` | `0` / `2`               | Reward scale                     |
| `scoring.export_weights`                  | `false`                 | Keep the final model's weights   |
| `scoring.scorer.backend`                  | `host`                  | `host` or `prime`                |
| `scoring.scorer.prime`                    | `H200_141GB:1`          | Prime scorer VM                  |
| `hub.cutoff_date`                         | `2025-12-31`            | Hub cutoff                       |
| `workspace.image`                         | `pretrain-gym-agent:v1` | Agent image                      |
| `host_dir`                                | `~/.cache/pretrain_gym` | Host data, caches, and job files |

The turn limit is `env.agent.max_turns`, which the prompt states. `env.agent.timeout.finalize` must cover a full training run.

## Metrics

| Metric                                   | Meaning                                                   |
| ---------------------------------------- | --------------------------------------------------------- |
| `performance`                            | The reward                                                |
| `loss`, `loss/<subset>`, `bits_per_byte` | Final test results (`loss/all` for a set without subsets) |
| `corpus_error`, `diverged`               | Why a run scored 0                                        |
| `heldout_characters_removed`             | Held-out text cut at final scoring                        |
| `probes`, `best_probe_loss`              | Scorer tool use                                           |
| `downloads`, `download_bytes`            | Hub tool use                                              |

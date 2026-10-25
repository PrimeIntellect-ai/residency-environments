import math

from .config import PretrainTaskConfig

PROMPT = """\
Your goal is to create the best pretraining corpus for a language model, in `corpus/`. The model \
is trained on the corpus data and evaluated on held-out text; the lower its loss, the higher the \
reward. If `corpus/` is missing or unreadable, the reward is 0.

## Training

- The model trains on about {train_tokens:,} tokens read from `corpus/` in order: files in sorted \
path order, documents in file order. If the corpus runs out, it is read again from the start.
- Corpus files are `.jsonl`, `.jsonl.gz`, or `.jsonl.zst` (one JSON object per line) or \
`.parquet`; each document is its string `text` field. Other files are not allowed in `corpus/`.
- `trainer/` holds the model's training code and configuration, its tokenizer, and the code that \
prepares the corpus, including removing held-out evaluation text from it.

## Reward

`reward = p^{exponent:g}`, where `p = {p}` and `L` is the trained model's cross-entropy in nats \
per token on the private part of an undisclosed held-out evaluation set. A uniform prediction has loss {uniform:.4f} (reward 0); a higher `L` makes `p` negative, and the \
reward is then `p`.

## Turns

{turns} `python turns.py` shows how many are left.

## Tools

The workspace has no network access and no GPU. Tool servers outside it provide:

- `hf` server, which sees Hugging Face as it was on {cutoff}:
  - `search`: search Hugging Face Hub datasets or models.
  - `repo_info`: a dataset's or model's card, files, refs, and commit history.
  - `download`: download dataset files into `data/<repo_id>/` or model files into `models/<repo_id>/`.
  - `search_papers`, `read_paper`: find and read papers.
- `scorer` server, which runs the same training on a snapshot of `corpus/` and evaluates on a \
separate public part of the held-out set:
  - `score`: start a run, optionally on fewer tokens.
  - `score_status`: a run's progress, or its results: validation loss, training-loss curve, and \
other training metrics.
  - `cancel_score`: stop a run.
"""


def render(config: PretrainTaskConfig, vocab_size: int) -> str:
    scoring = config.scoring
    uniform = math.log(vocab_size)
    p = f"1 - L / {uniform:.4f}"
    if scoring.target_loss:
        p = f"({uniform:.4f} - L) / ({uniform:.4f} - {scoring.target_loss:g})"
    return PROMPT.format(
        train_tokens=scoring.train_tokens,
        exponent=scoring.reward_exponent,
        p=p,
        uniform=uniform,
        turns="There is no turn limit." if config.max_turns is None else f"You have {config.max_turns} turns.",
        cutoff=config.hub.cutoff_date,
    )

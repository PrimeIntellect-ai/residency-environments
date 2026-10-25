# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub", "numpy", "pyarrow", "tokenizers", "torch"]
# ///
"""HellaSwag validation accuracy of weights exported by the `nanogpt_speedrun` model.

Each context is followed by its four endings, and the ending with the lowest loss wins: summed
over the ending's tokens for `acc`, averaged for `acc_norm`.

usage: uv run scripts/pretrain_gym/hellaswag.py --code <final>/code --weights <final>/out/model.pt
`--code` is a scoring job's code directory, holding train.py, model.json, and tokenizer.json.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DATASET, REVISION = "Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76"


def load_trainer(code: Path):
    spec = importlib.util.spec_from_file_location("trainer", code / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("hellaswag.json"))
    parser.add_argument("--limit", type=int, help="Score only the first examples.")
    args = parser.parse_args()

    path = hf_hub_download(DATASET, "data/validation-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
    examples = pq.read_table(path, columns=["ctx", "endings", "label"]).to_pylist()[: args.limit]

    trainer = load_trainer(args.code)
    model_config = json.loads((args.code / "model.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.code / "tokenizer.json"))
    tokenizer.encode_special_tokens = True
    bos_id = tokenizer.token_to_id(model_config["tokenizer"]["bos_token"])
    shape = model_config["model"]
    model = trainer.GPT(
        tokenizer.get_vocab_size(), shape["n_layer"], shape["n_head"], shape["dim"], shape["seq_len"], shape["softcap"]
    )
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    correct = correct_norm = 0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for example in examples:
            context = [bos_id, *encode(example["ctx"])]
            endings = [encode(" " + ending) for ending in example["endings"]]
            length = len(context) + max(map(len, endings))
            tokens = torch.full((len(endings), length), bos_id, dtype=torch.long)
            scored = torch.zeros((len(endings), length), dtype=torch.bool)
            for row, ending in enumerate(endings):
                tokens[row, : len(context) + len(ending)] = torch.tensor(context + ending)
                scored[row, len(context) : len(context) + len(ending)] = True
            tokens, scored = tokens.to(device), scored.to(device)
            logits = model.logits(tokens, trainer.document_mask(tokens, bos_id, flex=False))
            losses = F.cross_entropy(logits[:, :-1].transpose(1, 2), tokens[:, 1:], reduction="none")
            losses = losses * scored[:, 1:]
            total = losses.sum(dim=1)
            mean = total / scored[:, 1:].sum(dim=1)
            label = int(example["label"])
            correct += int(total.argmin()) == label
            correct_norm += int(mean.argmin()) == label

    result = dict(examples=len(examples), acc=correct / len(examples), acc_norm=correct_norm / len(examples))
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

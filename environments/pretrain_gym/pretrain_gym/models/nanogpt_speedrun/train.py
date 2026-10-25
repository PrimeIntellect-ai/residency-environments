"""Train a GPT-2-small-sized model in the style of the modded-nanogpt speedrun, then evaluate it.

Architecture: 12 pre-norm transformer blocks of width 768 with 6 heads of 128, rotary position
embeddings, RMS-normalized queries and keys, ReLU^2 MLPs, untied input and output embeddings,
zero-initialized output projections, and logits soft-capped at 15. Attention is causal within a
document: each BOS token starts a new document. Hidden matrices are trained with Muon, embeddings
and the output head with Adam. The learning rate is constant, then decays linearly to zero over
the final `cooldown_frac` of the steps.

Inputs (`--data`): `meta.json`, `train.bin`, and `eval/<name>.bin`. Each `.bin` file is a flat
array of token ids (dtype from `meta.json`) in which every document starts with the BOS token.
Training reads `train.bin` in order, `batch_tokens` per optimizer step, and wraps around to the
start if it runs out. Evaluation splits each document into chunks of at most `seq_len` tokens,
each starting with BOS, and scores every document token exactly once without context from other
documents or chunks.

Outputs (`--out`): `metrics.jsonl` (one line per optimizer step and per evaluation, and an
`evaluating` line when the final evaluation starts) and `result.json` (final per-set loss in nats
per token and bits per byte). `--save-weights` also writes the trained model's state dict.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
except ImportError:
    BlockMask = None


# -- model ---------------------------------------------------------------------------------------


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, zero: bool = False) -> None:
        super().__init__(in_features, out_features, bias=False)
        if zero:
            nn.init.zeros_(self.weight)
        else:
            bound = math.sqrt(3) * 0.5 * in_features**-0.5
            nn.init.uniform_(self.weight, -bound, bound)


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        frequencies = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = torch.outer(torch.arange(max_len, dtype=torch.float32), frequencies)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, heads, head_dim)
        cos, sin = self.cos[None, : x.size(1), None, :], self.sin[None, : x.size(1), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1).type_as(x)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask) -> torch.Tensor:
    if BlockMask is not None and isinstance(mask, BlockMask):
        return flex_attention(q, k, v, block_mask=mask)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int, seq_len: int) -> None:
        super().__init__()
        self.n_head, self.head_dim = n_head, dim // n_head
        self.q, self.k, self.v = Linear(dim, dim), Linear(dim, dim), Linear(dim, dim)
        self.proj = Linear(dim, dim, zero=True)
        self.rotary = Rotary(self.head_dim, seq_len)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        batch, length, dim = x.shape
        shape = (batch, length, self.n_head, self.head_dim)
        q, k, v = self.q(x).view(shape), self.k(x).view(shape), self.v(x).view(shape)
        q, k = self.rotary(norm(q)).type_as(v), self.rotary(norm(k)).type_as(v)
        y = attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask)
        return self.proj(y.transpose(1, 2).reshape(batch, length, dim))


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc, self.proj = Linear(dim, 4 * dim), Linear(4 * dim, dim, zero=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, seq_len: int) -> None:
        super().__init__()
        self.attn, self.mlp = Attention(dim, n_head, seq_len), MLP(dim)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        x = x + self.attn(norm(x), mask)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_head: int, dim: int, seq_len: int, softcap: float) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(Block(dim, n_head, seq_len) for _ in range(n_layer))
        self.head = Linear(dim, vocab_size, zero=True)
        self.softcap = softcap

    def logits(self, tokens: torch.Tensor, mask) -> torch.Tensor:
        x = norm(self.embed(tokens))
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(norm(x)).float()
        return self.softcap * torch.tanh(logits / self.softcap)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor, mask) -> torch.Tensor:
        """Summed cross-entropy over targets that are not -100."""
        logits = self.logits(tokens, mask)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100, reduction="sum")


def document_mask(tokens: torch.Tensor, bos_id: int, flex: bool):
    """Causal attention that never crosses a BOS token."""
    docs = (tokens == bos_id).cumsum(dim=1)
    batch, length = tokens.shape
    if flex:

        def same_document(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (docs[b, q_idx] == docs[b, kv_idx])

        return create_block_mask(same_document, batch, None, length, length, device=tokens.device)
    causal = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()
    return (causal & (docs[:, :, None] == docs[:, None, :]))[:, None]


# -- optimizer ------------------------------------------------------------------------------------


def orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration toward the nearest semi-orthogonal matrix (Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if transposed else x).type_as(g)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                buffer = self.state[p].setdefault("momentum", torch.zeros_like(p.grad))
                buffer.lerp_(p.grad, 1 - group["momentum"])
                update = orthogonalize(p.grad.lerp(buffer, group["momentum"]))
                p.add_(update, alpha=-group["lr"] * max(1.0, p.size(0) / p.size(1)) ** 0.5)


# -- data -----------------------------------------------------------------------------------------


def eval_rows(tokens: np.ndarray, bos_id: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack BOS-led chunks of at most `seq_len` tokens into rows; returns (inputs, targets)."""
    starts = np.flatnonzero(tokens == bos_id)
    ends = np.append(starts[1:], tokens.size)
    chunks = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        body = tokens[start + 1 : end]
        for offset in range(0, max(body.size, 1), seq_len - 1):
            chunks.append(body[offset : offset + seq_len - 1])
    inputs = np.full((len(chunks), seq_len), bos_id, dtype=np.int64)
    targets = np.full((len(chunks), seq_len), -100, dtype=np.int64)
    row, used = -1, seq_len
    for chunk in chunks:
        length = chunk.size + 1
        if used + length > seq_len:
            row, used = row + 1, 0
        inputs[row, used + 1 : used + length] = chunk
        targets[row, used : used + length - 1] = chunk
        used += length
    return inputs[: row + 1], targets[: row + 1]


# -- training -------------------------------------------------------------------------------------


def log(path: Path, record: dict) -> None:
    with path.open("a") as file:
        file.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--save-weights", type=Path, help="Write the trained model's state dict here.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    model_config, train_config = config["model"], config["train"]
    meta = json.loads((args.data / "meta.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "metrics.jsonl"

    torch.manual_seed(train_config["seed"])
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    flex = cuda and BlockMask is not None and torch.cuda.get_device_capability() >= (8, 0)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda)
    seq_len, bos_id = model_config["seq_len"], meta["bos_id"]
    micro_batch = train_config["micro_batch_size"]
    accumulation = train_config["batch_tokens"] // (micro_batch * seq_len)
    steps = max(1, args.train_tokens // (accumulation * micro_batch * seq_len))
    cooldown_steps = round(train_config["cooldown_frac"] * steps)

    model = GPT(
        meta["vocab_size"],
        model_config["n_layer"],
        model_config["n_head"],
        model_config["dim"],
        seq_len,
        model_config["softcap"],
    ).to(device)
    hidden = [p for block in model.blocks for p in block.parameters() if p.ndim == 2]
    adam = torch.optim.Adam(
        [
            dict(params=[model.embed.weight], lr=train_config["embed_lr"]),
            dict(params=[model.head.weight], lr=train_config["head_lr"]),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
    )
    muon = Muon(hidden, lr=train_config["muon_lr"])
    optimizers = (adam, muon)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    forward = torch.compile(model) if cuda else model

    def batch(rows: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.int64))
        return tensor.pin_memory().to(device, non_blocking=True) if cuda else tensor

    def loss_sum(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with autocast:
            return forward(inputs, targets, document_mask(inputs, bos_id, flex))

    eval_sets = {}
    for name in sorted(meta["eval"]):
        tokens = np.fromfile(args.data / "eval" / f"{name}.bin", dtype=meta["dtype"])
        eval_sets[name] = eval_rows(tokens, bos_id, seq_len)

    @torch.no_grad()
    def evaluate(max_rows: int | None) -> dict[str, float]:
        model.eval()
        losses = {}
        for name, (inputs, targets) in eval_sets.items():
            inputs, targets = inputs[:max_rows], targets[:max_rows]
            total = torch.zeros((), device=device, dtype=torch.float64)
            for start in range(0, len(inputs), micro_batch):
                rows = slice(start, start + micro_batch)
                # Pad the last batch so every forward pass has the compiled shape.
                padding = micro_batch - len(inputs[rows])
                batch_inputs = np.pad(inputs[rows], ((0, padding), (0, 0)), constant_values=bos_id)
                batch_targets = np.pad(targets[rows], ((0, padding), (0, 0)), constant_values=-100)
                total += loss_sum(batch(batch_inputs), batch(batch_targets)).double()
            losses[name] = total.item() / int((targets != -100).sum())
        model.train()
        return losses

    quick_rows = max(1, train_config["quick_eval_tokens"] // (seq_len * max(1, len(eval_sets))))
    train_tokens = np.memmap(args.data / "train.bin", dtype=meta["dtype"], mode="r")
    window = micro_batch * seq_len
    position = 0
    started = time.perf_counter()
    eval_seconds = 0.0
    diverged = False
    for step in range(1, steps + 1):
        step_started = time.perf_counter()
        scale = min(1.0, (steps - step + 1) / cooldown_steps) if cooldown_steps else 1.0
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * scale
        total = torch.zeros((), device=device)
        for _ in range(accumulation):
            rows = np.take(train_tokens, np.arange(position, position + window + 1), mode="wrap")
            position = (position + window) % train_tokens.size
            inputs, targets = (
                batch(rows[:-1].reshape(micro_batch, seq_len)),
                batch(rows[1:].reshape(micro_batch, seq_len)),
            )
            loss = loss_sum(inputs, targets) / window
            (loss / accumulation).backward()
            total += loss.detach() / accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        train_loss = total.item()
        step_seconds = time.perf_counter() - step_started
        log(
            metrics_path,
            dict(
                step=step,
                tokens=step * accumulation * window,
                train_loss=train_loss,
                lr_scale=scale,
                grad_norm=grad_norm.item(),
                step_seconds=step_seconds,
                tokens_per_second=accumulation * window / step_seconds,
            ),
        )
        if not math.isfinite(train_loss):
            diverged = True
            break
        if step % train_config["eval_every"] == 0 and step < steps:
            eval_started = time.perf_counter()
            losses = evaluate(quick_rows)
            eval_seconds += time.perf_counter() - eval_started
            log(metrics_path, dict(step=step, tokens=step * accumulation * window, quick_eval=losses))
    train_seconds = time.perf_counter() - started - eval_seconds

    if not diverged:
        log(metrics_path, dict(step=step, evaluating=True))
    eval_started = time.perf_counter()
    losses = {} if diverged else evaluate(None)
    eval_seconds += time.perf_counter() - eval_started
    bits_per_byte = {
        name: loss * int((eval_sets[name][1] != -100).sum()) / (math.log(2) * meta["eval"][name]["bytes"])
        for name, loss in losses.items()
    }
    result = dict(
        diverged=diverged or not all(math.isfinite(loss) for loss in losses.values()),
        losses=losses,
        bits_per_byte=bits_per_byte,
        steps=step,
        train_tokens=step * accumulation * window,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        device=torch.cuda.get_device_name() if cuda else "cpu",
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    if args.save_weights is not None:
        torch.save(model.state_dict(), args.save_weights)


if __name__ == "__main__":
    main()

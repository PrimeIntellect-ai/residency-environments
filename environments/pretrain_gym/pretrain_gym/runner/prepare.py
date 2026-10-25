"""Turn a corpus directory into the token files a trainer reads.

The corpus is every file under the corpus directory, read in sorted path order, documents in file
order. Supported files: `.jsonl`, `.jsonl.gz`, `.jsonl.zst` (one JSON object per line) and
`.parquet`, each document taken from its string `text` field. Held-out evaluation text is cut
out of every document first (see `decontaminate.py`). Each remaining document is tokenized
without special tokens (special-token text is encoded as ordinary text) and prefixed with the
BOS token. Documents are read only until the training run has enough tokens.

Writes `<out>/train.bin`, `<out>/eval/<name>.bin`, `<out>/meta.json`, and `<out>/prepare.json`
(corpus statistics). A corpus that cannot be read is reported in `<out>/corpus_error.json`.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import gzip
import hashlib
import io
import itertools
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterable, Iterator
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow.parquet as pq
import zstandard
from decontaminate import NgramIndex, ngram_hashes, words
from tokenizers import Tokenizer

SUFFIXES = (".jsonl", ".jsonl.gz", ".jsonl.zst", ".parquet")
CHUNK_DOCUMENTS = 256


class CorpusError(Exception):
    pass


def corpus_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise CorpusError(f"{root.name}/ does not exist")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix()
    )
    unsupported = [path.relative_to(root).as_posix() for path in files if not path.name.endswith(SUFFIXES)]
    if unsupported:
        raise CorpusError(f"unsupported corpus files (expected {', '.join(SUFFIXES)}): {unsupported[:5]}")
    if not files:
        raise CorpusError(f"{root.name}/ has no files")
    return files


def _lines(path: Path) -> io.TextIOBase:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    if path.name.endswith(".zst"):
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(path.open("rb")), encoding="utf-8")
    return path.open(encoding="utf-8")


def documents(path: Path, field: str, label: str) -> Iterator[str]:
    """The `field` strings of one file, in order; raises CorpusError on unreadable data."""
    try:
        if path.name.endswith(".parquet"):
            parquet = pq.ParquetFile(path)
            if field not in parquet.schema_arrow.names:
                raise CorpusError(f"{label}: no {field!r} column")
            for batch in parquet.iter_batches(columns=[field]):
                for text in batch.column(0).to_pylist():
                    if not isinstance(text, str):
                        raise CorpusError(f"{label}: a {field!r} value is not a string")
                    yield text
            return
        with _lines(path) as lines:
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get(field), str):
                    raise CorpusError(f"{label} line {number}: no string {field!r} field")
                yield row[field]
    except (UnicodeDecodeError, json.JSONDecodeError, zstandard.ZstdError, gzip.BadGzipFile, OSError) as exc:
        raise CorpusError(f"{label}: {type(exc).__name__}: {exc}") from exc


# Per-worker state, set by `_init_worker` in each pool process.
_tokenizer: Tokenizer | None = None
_index: NgramIndex | None = None


def _init_worker(tokenizer_path: str, index_path: str | None, n: int) -> None:
    global _tokenizer, _index
    _tokenizer = Tokenizer.from_file(tokenizer_path)
    _tokenizer.encode_special_tokens = True
    _index = NgramIndex(np.load(index_path, mmap_mode="r"), n) if index_path else None


def _encode(args: tuple[list[str], int, str]) -> np.ndarray:
    texts, bos_id, dtype = args
    encodings = _tokenizer.encode_batch(texts, add_special_tokens=False)
    return np.concatenate([np.array([bos_id, *encoding.ids], dtype=dtype) for encoding in encodings])


def _clean_and_encode(args: tuple[list[str], int, str]) -> tuple[np.ndarray, dict[str, int]]:
    texts, bos_id, dtype = args
    pieces = []
    stats = dict(documents=len(texts), characters=0, documents_short=0, documents_changed=0, characters_removed=0)
    for text in texts:
        kept, changed = _index.clean(text) if _index is not None else ([text], False)
        pieces += kept
        stats["characters"] += len(text)
        stats["documents_short"] += not kept and not changed
        stats["documents_changed"] += changed
        stats["characters_removed"] += changed * (len(text) - sum(map(len, kept)))
    tokens = _encode((pieces, bos_id, dtype)) if pieces else np.empty(0, dtype=dtype)
    return tokens, stats


def _hashes(args: tuple[str, int]) -> np.ndarray:
    text, n = args
    return ngram_hashes(words(text), n)


def ordered_map(pool: Pool, fn: Callable, items: Iterable, window: int) -> Iterator:
    """`pool.imap` that reads `items` in this thread and keeps at most `window` tasks in flight."""
    pending = collections.deque()
    try:
        for item in items:
            pending.append(pool.apply_async(fn, (item,)))
            if len(pending) >= window:
                yield pending.popleft().get()
        while pending:
            yield pending.popleft().get()
    finally:
        # A consumer that stops early leaves tasks in flight. `Pool.terminate` stops reading results
        # after a few, so a worker still writing one would block it forever; let them all finish first.
        for task in pending:
            task.wait()


def chunked(items: Iterable, size: int) -> Iterator[list]:
    items = iter(items)
    return iter(lambda: list(itertools.islice(items, size)), [])


def _digest(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _file_identity(paths: list[str]) -> list[tuple[str, int]]:
    return [(path, os.path.getsize(path)) for path in paths]


def screen_index(spec: dict, pool: Pool) -> str | None:
    """Path of the cached n-gram index over every screened held-out file."""
    files = sorted(spec["screen"])
    if not files:
        return None
    key = _digest(_file_identity(files), spec["ngram"], spec["text_field"], sys.version, os.environ["PYTHONHASHSEED"])
    path = Path(spec["cache"]) / "index" / f"{key}.npy"
    if not path.exists():
        texts = (text for file in files for text in documents(Path(file), spec["text_field"], file))
        hashes = pool.imap(_hashes, ((text, spec["ngram"]) for text in texts), chunksize=64)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f".{os.getpid()}.npy")
        np.save(partial, np.unique(np.concatenate(list(hashes))))
        partial.replace(path)
    return str(path)


def eval_sets(spec: dict, pool: Pool, tokenizer_hash: str, bos_id: int, dtype: str, out: Path) -> dict[str, dict]:
    """Tokenize each evaluation set into `<out>/eval/<name>.bin`, through a cache."""
    (out / "eval").mkdir(parents=True, exist_ok=True)
    meta = {}
    for name, files in sorted(spec["eval"].items()):
        key = _digest(_file_identity(sorted(files)), spec["text_field"], tokenizer_hash, bos_id, dtype)
        cached = Path(spec["cache"]) / "eval" / key
        if not cached.exists():
            texts = [text for file in sorted(files) for text in documents(Path(file), spec["text_field"], file)]
            chunks = ((chunk, bos_id, dtype) for chunk in chunked(texts, CHUNK_DOCUMENTS))
            # Built privately, then renamed into place, so concurrent jobs never see a partial set.
            partial = cached.with_name(f"{key}.{os.getpid()}")
            partial.mkdir(parents=True, exist_ok=True)
            np.concatenate(list(pool.imap(_encode, chunks))).tofile(partial / "tokens.bin")
            info = dict(documents=len(texts), bytes=sum(len(text.encode("utf-8")) for text in texts))
            (partial / "meta.json").write_text(json.dumps(info))
            try:
                partial.rename(cached)
            except OSError:
                shutil.rmtree(partial)
        target = out / "eval" / f"{name}.bin"
        target.unlink(missing_ok=True)
        target.symlink_to(cached / "tokens.bin")
        meta[name] = json.loads((cached / "meta.json").read_text())
    return meta


def train_tokens(spec: dict, pool: Pool, bos_id: int, dtype: str, out: Path) -> dict[str, int]:
    """Write the first `train_tokens + 1` cleaned corpus tokens to `<out>/train.bin`."""
    root = Path(spec["corpus"])
    files = corpus_files(root)
    texts = (
        text for path in files for text in documents(path, "text", f"{root.name}/{path.relative_to(root).as_posix()}")
    )
    chunks = ((chunk, bos_id, dtype) for chunk in chunked(texts, CHUNK_DOCUMENTS))
    needed = spec["train_tokens"] + 1
    stats = dict(
        files=len(files),
        documents=0,
        characters=0,
        documents_short=0,
        documents_changed=0,
        characters_removed=0,
        tokens=0,
    )
    results = ordered_map(pool, _clean_and_encode, chunks, 2 * spec["workers"])
    with (out / "train.bin").open("wb") as file, contextlib.closing(results):
        for tokens, chunk_stats in results:
            for key, value in chunk_stats.items():
                stats[key] += value
            tokens = tokens[: needed - stats["tokens"]]
            tokens.tofile(file)
            stats["tokens"] += tokens.size
            if stats["tokens"] >= needed:
                break
        if stats["documents"] == stats["documents_short"]:
            raise CorpusError(f"the corpus has no document of {spec['ngram']} or more words")
        if stats["tokens"] == 0:
            # Only held-out text was left out. Failing here would tell the agent its text is held out,
            # so the model trains on an empty corpus instead.
            np.array([bos_id], dtype=dtype).tofile(file)
            stats["tokens"] = 1
    stats["exhausted"] = int(stats["tokens"] < needed)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    spec = json.loads(parser.parse_args().spec.read_text())
    spec["workers"] = spec["workers"] or os.cpu_count()
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(spec["tokenizer"])
    bos_id = tokenizer.token_to_id(spec["bos_token"])
    vocab_size = tokenizer.get_vocab_size()
    dtype = "uint16" if vocab_size <= 2**16 else "uint32"
    tokenizer_hash = hashlib.sha256(Path(spec["tokenizer"]).read_bytes()).hexdigest()
    with Pool(spec["workers"], _init_worker, (spec["tokenizer"], None, spec["ngram"])) as pool:
        index_path = screen_index(spec, pool)
        eval_meta = eval_sets(spec, pool, tokenizer_hash, bos_id, dtype, out)
    meta = dict(dtype=dtype, bos_id=bos_id, vocab_size=vocab_size, eval=eval_meta)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    with Pool(spec["workers"], _init_worker, (spec["tokenizer"], index_path, spec["ngram"])) as pool:
        try:
            stats = train_tokens(spec, pool, bos_id, dtype, out)
        except CorpusError as exc:
            (out / "corpus_error.json").write_text(json.dumps({"error": str(exc)}))
            return
    (out / "prepare.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

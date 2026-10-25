"""Remove held-out evaluation text from training documents.

Text is compared as words: runs of letters and digits after Unicode compatibility decomposition
(NFKD), removal of combining marks and invisible format characters, and case folding, so case,
accents, punctuation, whitespace, and markup between words do not hide a copy. Every run of `n`
consecutive words that also occurs in a held-out document is cut out of the training document.
The pieces left between cuts become separate documents. Documents and pieces of fewer than `n`
words are dropped, so no text that is kept shares `n` consecutive words with the held-out set.

Word hashes use Python's string hash, so every process that shares an index must run with the
same `PYTHONHASHSEED`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import numpy as np
import regex

DEFAULT_NGRAM = 13
_RUN = regex.compile(r"[\w\p{Mn}\p{Me}\p{Cf}]+")
_MARKS = regex.compile(r"[\p{Mn}\p{Me}\p{Cf}]+")
_WORD = re.compile(r"[^\W_]+")
_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def _normalize(run: str) -> list[str]:
    if run.isascii():
        return _WORD.findall(run.lower())
    return _WORD.findall(_MARKS.sub("", unicodedata.normalize("NFKD", run)).casefold())


def words(text: str) -> list[str]:
    if text.isascii():
        return _WORD.findall(text.lower())
    return [word for run in _RUN.findall(text) for word in _normalize(run)]


def word_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """The words of `text` and, for each, the character span of the run it came from."""
    found, spans = [], []
    for match in _RUN.finditer(text):
        for word in _normalize(match.group()):
            found.append(word)
            spans.append(match.span())
    return found, spans


def ngram_hashes(tokens: list[str], n: int) -> np.ndarray:
    """One 64-bit hash per run of `n` consecutive words."""
    count = len(tokens) - n + 1
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    word_hashes = np.fromiter(map(hash, tokens), dtype=np.int64, count=len(tokens)).view(np.uint64)
    hashes = word_hashes[:count].copy()
    for offset in range(1, n):
        hashes *= _MULTIPLIER
        hashes += word_hashes[offset : offset + count]
    return hashes


class NgramIndex:
    """The sorted, distinct n-gram hashes of a held-out set."""

    def __init__(self, hashes: np.ndarray, n: int) -> None:
        self.hashes, self.n = hashes, n

    @classmethod
    def build(cls, texts: Iterable[str], n: int = DEFAULT_NGRAM) -> NgramIndex:
        parts = [ngram_hashes(words(text), n) for text in texts]
        return cls(np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64), n)

    @classmethod
    def merge(cls, indexes: Iterable[NgramIndex]) -> NgramIndex:
        indexes = list(indexes)
        if len({index.n for index in indexes}) != 1:
            raise ValueError("cannot merge indexes with different n-gram lengths")
        return cls(np.unique(np.concatenate([index.hashes for index in indexes])), indexes[0].n)

    def _matches(self, tokens: list[str]) -> np.ndarray:
        queries = ngram_hashes(tokens, self.n)
        if not queries.size or not self.hashes.size:
            return np.zeros(queries.size, dtype=bool)
        positions = np.minimum(np.searchsorted(self.hashes, queries), self.hashes.size - 1)
        return self.hashes[positions] == queries

    def clean(self, text: str) -> tuple[list[str], bool]:
        """The pieces of `text` left after cutting out held-out passages and dropping short
        pieces, and whether any held-out passage was found."""
        tokens = words(text)
        if len(tokens) < self.n:
            return [], False
        if not self._matches(tokens).any():
            return [text], False
        tokens, spans = word_spans(text)
        covered = np.zeros(len(tokens) + 1, dtype=np.int64)
        starts = np.flatnonzero(self._matches(tokens))
        np.add.at(covered, starts, 1)
        np.add.at(covered, starts + self.n, -1)
        covered = np.cumsum(covered[:-1]) > 0
        pieces, piece_start, piece_words = [], 0, 0
        for index, is_covered in enumerate(covered.tolist()):
            if not is_covered:
                piece_words += 1
                continue
            if piece_words >= self.n:
                pieces.append(text[piece_start : spans[index][0]].strip())
            piece_start, piece_words = spans[index][1], 0
        if piece_words >= self.n:
            pieces.append(text[piece_start:].strip())
        return pieces, True

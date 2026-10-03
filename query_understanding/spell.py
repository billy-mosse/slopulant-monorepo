"""Frequency-weighted spelling correction for search queries (Norvig-style)."""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

ALPHABET = "abcdefghijklmnopqrstuvwxyz'-"
TOKEN_RE = re.compile(r"[a-z0-9'\-]+")
EDIT_PENALTY = {0: 1.0, 1: 0.08, 2: 0.004}
MIN_TOKEN_COUNT = 3


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def damerau_levenshtein(a: str, b: str) -> int:
    d = {(i, -1): i + 1 for i in range(-1, len(a) + 1)}
    d.update({(-1, j): j + 1 for j in range(-1, len(b) + 1)})
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
            if i and j and ca == b[j - 1] and a[i - 1] == cb:
                d[i, j] = min(d[i, j], d[i - 2, j - 2] + 1)
    return d[len(a) - 1, len(b) - 1]


def edits1(word: str) -> set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = {l + r[1:] for l, r in splits if r}
    transposes = {l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1}
    replaces = {l + c + r[1:] for l, r in splits if r for c in ALPHABET}
    inserts = {l + c + r for l, r in splits for c in ALPHABET}
    return deletes | transposes | replaces | inserts


def edits2(word: str) -> set[str]:
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


class Vocabulary:
    def __init__(self, counts: Counter):
        self.counts = Counter({w: c for w, c in counts.items() if c >= MIN_TOKEN_COUNT})
        self.total = sum(self.counts.values()) or 1

    @classmethod
    def from_queries(cls, queries: list[tuple[str, int]]) -> "Vocabulary":
        counts: Counter = Counter()
        for q, n in queries:
            for tok in tokenize(q):
                counts[tok] += n
        return cls(counts)

    def prob(self, word: str) -> float:
        return self.counts.get(word, 0) / self.total

    def known(self, words) -> set[str]:
        return {w for w in words if w in self.counts}


class SpellCorrector:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.correct_token = lru_cache(maxsize=200_000)(self._correct_token)

    def candidates(self, word: str) -> dict[str, int]:
        if word in self.vocab.counts:
            return {word: 0}
        out = {w: 1 for w in self.vocab.known(edits1(word))}
        if not out and len(word) > 4:
            out = {w: 2 for w in self.vocab.known(edits2(word))}
        return out

    def score(self, word: str, cand: str, dist: int) -> float:
        dist = min(dist, damerau_levenshtein(word, cand))
        return math.log(self.vocab.prob(cand) + 1e-12) + math.log(EDIT_PENALTY[dist])

    def _correct_token(self, word: str) -> str:
        if word.isdigit() or len(word) <= 2:
            return word
        cands = self.candidates(word)
        if not cands:
            return word
        return max(cands, key=lambda c: self.score(word, c, cands[c]))

    def correct(self, query: str) -> str:
        return " ".join(self.correct_token(t) for t in tokenize(query))

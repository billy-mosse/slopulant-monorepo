"""Search query spell correction (SymSpell-style).

Builds a symmetric-delete dictionary from term frequencies in recent search
logs and produces a correction table for misspelled terms.

Candidates within Damerau-Levenshtein distance <= 2 are ranked by
    score = log(freq) - DIST_PENALTY * distance + ADJ_BONUS * adjacent_subs
where adjacent_subs counts substitutions between neighbouring keys on a
QWERTY keyboard (fat-finger typos are far more likely than random ones).
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Final, Iterator

from pyspark.sql import SparkSession, functions as F

LOGS_TABLE: Final = "search.query_logs"
CORRECTIONS_TABLE: Final = "search.spell_corrections"

MAX_EDIT: Final = 2
MIN_DICT_FREQ: Final = 25       # term must appear this often to be a valid target
MIN_TERM_LEN: Final = 3
DIST_PENALTY: Final = 2.5
ADJ_BONUS: Final = 0.8
FREQ_RATIO_GUARD: Final = 8.0   # target must be this many times more frequent than the typo

TOKEN_RE: Final = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")

_ROWS: Final = ("1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm")


def _build_adjacency() -> dict[str, frozenset[str]]:
    pos = {ch: (r, c) for r, row in enumerate(_ROWS) for c, ch in enumerate(row)}
    adj: dict[str, set[str]] = defaultdict(set)
    for a, (ra, ca) in pos.items():
        for b, (rb, cb) in pos.items():
            if a != b and abs(ra - rb) <= 1 and abs(ca - cb) <= 1:
                adj[a].add(b)
    return {k: frozenset(v) for k, v in adj.items()}


KEY_NEIGHBOURS: Final = _build_adjacency()


def deletes(word: str, depth: int = MAX_EDIT) -> set[str]:
    out: set[str] = set()
    frontier = {word}
    for _ in range(depth):
        nxt = {w[:i] + w[i + 1:] for w in frontier for i in range(len(w))}
        out |= nxt
        frontier = nxt
    return out


def damerau_levenshtein(a: str, b: str) -> tuple[int, int]:
    """Optimal string alignment distance plus number of keyboard-adjacent substitutions."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    adj = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if a[i - 1] == b[j - 1] else 1
            best, best_adj = d[i - 1][j - 1] + sub_cost, adj[i - 1][j - 1]
            if sub_cost and b[j - 1] in KEY_NEIGHBOURS.get(a[i - 1], ()):
                best_adj += 1
            for cand, cand_adj in ((d[i - 1][j] + 1, adj[i - 1][j]), (d[i][j - 1] + 1, adj[i][j - 1])):
                if cand < best:
                    best, best_adj = cand, cand_adj
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                if d[i - 2][j - 2] + 1 < best:
                    best, best_adj = d[i - 2][j - 2] + 1, adj[i - 2][j - 2]
            d[i][j], adj[i][j] = best, best_adj
    return d[n][m], adj[n][m]


@dataclass(frozen=True, slots=True)
class Suggestion:
    term: str
    correction: str
    distance: int
    adjacent_subs: int
    score: float


class SymSpellIndex:
    __slots__ = ("freq", "_deletes")

    def __init__(self, freq: Counter[str]) -> None:
        self.freq = freq
        self._deletes: dict[str, list[str]] = defaultdict(list)
        for term, count in freq.items():
            if count < MIN_DICT_FREQ:
                continue
            self._deletes[term].append(term)
            for d in deletes(term):
                self._deletes[d].append(term)

    def candidates(self, word: str) -> Iterator[str]:
        seen: set[str] = set()
        for key in {word} | deletes(word):
            for cand in self._deletes.get(key, ()):
                if cand not in seen:
                    seen.add(cand)
                    yield cand

    def best(self, word: str) -> Suggestion | None:
        if word in self._deletes and word in self._deletes[word]:
            return None  # already a dictionary word
        own = self.freq.get(word, 0)
        top: Suggestion | None = None
        for cand in self.candidates(word):
            if abs(len(cand) - len(word)) > MAX_EDIT:
                continue
            dist, adj = damerau_levenshtein(word, cand)
            if dist == 0 or dist > MAX_EDIT:
                continue
            if self.freq[cand] < FREQ_RATIO_GUARD * max(own, 1):
                continue
            score = math.log(self.freq[cand]) - DIST_PENALTY * dist + ADJ_BONUS * adj
            if top is None or score > top.score:
                top = Suggestion(word, cand, dist, adj, round(score, 4))
        return top


def load_term_counts(spark: SparkSession, days: int) -> Counter[str]:
    rows = (
        spark.table(LOGS_TABLE)
        .where(F.col("event_date") >= F.date_sub(F.current_date(), days))
        .select(F.explode(F.split(F.lower(F.col("query_text")), r"\s+")).alias("tok"))
        .where(F.length("tok") >= MIN_TERM_LEN)
        .groupBy("tok").count()
        .collect()
    )
    counts: Counter[str] = Counter()
    for r in rows:
        for t in TOKEN_RE.findall(r["tok"]):
            if not t.isdigit():
                counts[t] += r["count"]
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("spellcheck")

    spark = SparkSession.builder.appName("query-spellcheck").getOrCreate()
    counts = load_term_counts(spark, args.days)
    index = SymSpellIndex(counts)
    log.info("vocabulary=%d dictionary terms=%d", len(counts), sum(c >= MIN_DICT_FREQ for c in counts.values()))

    suggestions = [s for t in counts if (s := index.best(t)) is not None]
    log.info("generated %d corrections", len(suggestions))

    df = spark.createDataFrame(
        [(s.term, s.correction, s.distance, s.adjacent_subs, s.score, counts[s.term]) for s in suggestions],
        "term string, correction string, edit_distance int, adjacent_subs int, score double, term_freq long",
    ).withColumn("built_at", F.current_timestamp())
    if args.dry_run:
        df.orderBy(F.desc("term_freq")).show(50, truncate=False)
    else:
        df.write.mode("overwrite").saveAsTable(CORRECTIONS_TABLE)


if __name__ == "__main__":
    main()

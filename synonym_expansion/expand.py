"""Query expansion from mined synonyms.

Two evidence sources from search logs:
  * reformulations: the user rewrote query A into query B within the same
    session and inside REFORM_WINDOW_S seconds; if A and B differ by exactly one
    token, the swapped tokens (a, b) are a candidate pair
  * co-clicks: queries that led to clicks on the same product; their single
    differing tokens are candidates as well

Each candidate is scored with Dunning's log-likelihood ratio on the 2x2
contingency table (k11 = pair count, k12/k21 = marginals minus pair, k22 = rest):

    LLR = 2 * (H(k) - H(rows) - H(cols)),   H(x) = sum x log(x / N)

Pairs whose surface forms are within edit distance 2 (typos, plurals) are
dropped -- spelling is handled elsewhere and those are not real synonyms.
"""
from __future__ import annotations

import argparse
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Final, Iterable

from pyspark.sql import SparkSession, Window, functions as F

LOGS_TABLE: Final = "search.query_logs"
EXPANSIONS_TABLE: Final = "search.query_expansions"

REFORM_WINDOW_S: Final = 90
MIN_PAIR_COUNT: Final = 15
MIN_LLR: Final = 10.83          # chi2(1) at p = 0.001
MAX_SYNONYMS: Final = 3
EXPANSION_WEIGHT: Final = 0.6   # boost for an expanded term relative to the original
MIN_TYPO_DIST: Final = 3        # pairs closer than this are treated as spelling variants

log = logging.getLogger("synonym_expansion")


@dataclass(frozen=True, slots=True)
class SynonymPair:
    term: str
    synonym: str
    count: int
    llr: float
    source: str


def _xlogx(x: int) -> float:
    return x * math.log(x) if x > 0 else 0.0


def llr(k11: int, k12: int, k21: int, k22: int) -> float:
    n = k11 + k12 + k21 + k22
    h_all = _xlogx(k11) + _xlogx(k12) + _xlogx(k21) + _xlogx(k22) - _xlogx(n)
    h_rows = _xlogx(k11 + k12) + _xlogx(k21 + k22) - _xlogx(n)
    h_cols = _xlogx(k11 + k21) + _xlogx(k12 + k22) - _xlogx(n)
    return max(0.0, 2.0 * (h_all - h_rows - h_cols))


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def single_swap(q1: str, q2: str) -> tuple[str, str] | None:
    """Return (a, b) if the two queries differ in exactly one aligned token."""
    t1, t2 = q1.split(), q2.split()
    if len(t1) != len(t2) or t1 == t2:
        return None
    diffs = [(a, b) for a, b in zip(t1, t2) if a != b]
    return diffs[0] if len(diffs) == 1 else None


def is_near_duplicate(a: str, b: str) -> bool:
    if a.rstrip("s") == b.rstrip("s") or a.replace("-", "") == b.replace("-", ""):
        return True
    return edit_distance(a, b) < MIN_TYPO_DIST


def reformulation_pairs(spark: SparkSession, days: int) -> Iterable[tuple[str, str]]:
    w = Window.partitionBy("session_id").orderBy("query_ts")
    df = (spark.table(LOGS_TABLE)
          .where(F.col("query_ts") >= F.date_sub(F.current_date(), days))
          .select("session_id", "query_ts", F.lower(F.trim("query_text")).alias("q"))
          .withColumn("next_q", F.lead("q").over(w))
          .withColumn("next_ts", F.lead("query_ts").over(w))
          .where(F.col("next_ts").cast("long") - F.col("query_ts").cast("long") <= REFORM_WINDOW_S)
          .groupBy("q", "next_q").count())
    for row in df.toLocalIterator():
        swap = single_swap(row["q"], row["next_q"] or "")
        if swap:
            yield from [swap] * min(row["count"], 1000)


def coclick_pairs(spark: SparkSession, days: int) -> Iterable[tuple[str, str]]:
    clicks = (spark.table(LOGS_TABLE)
              .where(F.col("query_ts") >= F.date_sub(F.current_date(), days))
              .where(F.col("clicked_product_id").isNotNull())
              .select(F.lower(F.trim("query_text")).alias("q"), "clicked_product_id")
              .distinct())
    joined = (clicks.alias("a").join(clicks.alias("b"), "clicked_product_id")
              .where(F.col("a.q") < F.col("b.q"))
              .groupBy(F.col("a.q").alias("q1"), F.col("b.q").alias("q2")).count()
              .where(F.col("count") >= 2))
    for row in joined.toLocalIterator():
        swap = single_swap(row["q1"], row["q2"])
        if swap:
            yield swap
            yield swap[::-1]


def score_pairs(pairs: Counter, source: str) -> list[SynonymPair]:
    left, right = Counter(), Counter()
    for (a, b), c in pairs.items():
        left[a] += c
        right[b] += c
    total = sum(pairs.values())
    out = []
    for (a, b), k11 in pairs.items():
        if k11 < MIN_PAIR_COUNT or is_near_duplicate(a, b):
            continue
        k12, k21 = left[a] - k11, right[b] - k11
        score = llr(k11, k12, k21, total - k11 - k12 - k21)
        if score >= MIN_LLR:
            out.append(SynonymPair(a, b, k11, score, source))
    log.info("%s: %d candidate pairs, %d kept", source, len(pairs), len(out))
    return out


def build_table(scored: list[SynonymPair]) -> dict[str, list[SynonymPair]]:
    best: dict[tuple[str, str], SynonymPair] = {}
    for p in scored:  # same pair from both sources -> keep the higher LLR
        key = (p.term, p.synonym)
        if key not in best or p.llr > best[key].llr:
            best[key] = p
    table: dict[str, list[SynonymPair]] = defaultdict(list)
    for p in best.values():
        table[p.term].append(p)
    return {t: sorted(v, key=lambda p: -p.llr)[:MAX_SYNONYMS] for t, v in table.items()}


def expand_query(query: str, table: dict[str, list[SynonymPair]]) -> list[tuple[str, float]]:
    """Weighted terms for the retrieval layer: originals at 1.0, synonyms below."""
    terms: list[tuple[str, float]] = []
    for tok in query.lower().split():
        terms.append((tok, 1.0))
        syns = table.get(tok, [])
        top = syns[0].llr if syns else 1.0
        terms += [(s.synonym, round(EXPANSION_WEIGHT * s.llr / top, 3)) for s in syns]
    return terms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", nargs="*", default=[], help="queries to print expansions for")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spark = SparkSession.builder.appName("synonym-expansion").getOrCreate()
    scored = (score_pairs(Counter(reformulation_pairs(spark, args.days)), "reformulation")
              + score_pairs(Counter(coclick_pairs(spark, args.days)), "coclick"))
    table = build_table(scored)
    log.info("synonym table covers %d terms", len(table))

    for q in args.probe:
        print(q, "->", expand_query(q, table))
    if args.dry_run:
        return

    rows = [(t, p.synonym, rank, p.llr, p.count, p.source)
            for t, ps in table.items() for rank, p in enumerate(ps, 1)]
    (spark.createDataFrame(rows, "term string, synonym string, rank int, llr double, "
                                 "pair_count long, source string")
     .withColumn("built_at", F.current_timestamp())
     .write.mode("overwrite").saveAsTable(EXPANSIONS_TABLE))


if __name__ == "__main__":
    main()

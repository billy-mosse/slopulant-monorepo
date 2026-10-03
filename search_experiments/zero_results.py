"""Fallbacks for queries that return zero results.

For each frequent zero-result query we try, in order:

1. Query relaxation: repeatedly drop the least informative term (lowest IDF
   over product titles) until the relaxed query matches at least
   ``min_matches`` active products, never dropping below ``min_terms`` terms.
2. Category trending: if relaxation fails, guess the category from whatever
   terms do hit the catalog (majority vote of matched products' categories)
   and serve that category's currently trending products.
3. Site-wide trending as the last resort.

Inputs:  search.query_logs, catalog.products, recs.trending
Output:  search.zero_result_fallbacks
         (query_norm, strategy, relaxed_query, dropped_terms, category, product_ids, n_searches, built_at)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

import pandas as pd

from shared_utils import (ExperimentConfig, IdfTable, coverage, get_logger, load_config,
                          normalize_query, read_sql, require_columns, simple_stem, summarize_metrics,
                          timed, tokenize, write_table)

log = get_logger("search_exp.zero_results")

ZERO_RESULT_SQL = """
SELECT query_text, COUNT(*) AS n_searches
FROM search.query_logs
WHERE event_ts >= CURRENT_DATE - (:lookback_days * INTERVAL '1 day')
  AND n_results = 0
GROUP BY query_text
HAVING COUNT(*) >= :min_query_count
"""

PRODUCTS_SQL = """
SELECT product_id, title, category, brand
FROM catalog.products
WHERE is_active
"""

TRENDING_SQL = """
SELECT product_id, category, trend_score
FROM recs.trending
WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM recs.trending)
"""

OUTPUT_TABLE = "search.zero_result_fallbacks"

STRATEGY_RELAXED = "relaxed"
STRATEGY_CATEGORY = "category_trending"
STRATEGY_GLOBAL = "global_trending"


@dataclass
class ZeroResultConfig(ExperimentConfig):
    lookback_days: int = 28
    min_query_count: int = 3
    min_matches: int = 4
    min_terms: int = 1
    max_drops: int = 3
    n_fallback_products: int = 24
    min_category_votes: int = 3
    protected_terms: list[str] = field(default_factory=lambda: ["king", "queen", "twin", "full"])

    def validate(self) -> None:
        super().validate()
        if self.min_terms < 1 or self.max_drops < 0:
            raise ValueError("min_terms >= 1 and max_drops >= 0 required")
        if self.n_fallback_products < 1:
            raise ValueError("n_fallback_products must be positive")


@dataclass
class Fallback:
    query_norm: str
    strategy: str
    product_ids: list[str]
    relaxed_query: str | None = None
    dropped_terms: list[str] = field(default_factory=list)
    category: str | None = None
    n_searches: int = 0

    def to_row(self) -> dict:
        return {"query_norm": self.query_norm, "strategy": self.strategy,
                "relaxed_query": self.relaxed_query, "dropped_terms": json.dumps(self.dropped_terms),
                "category": self.category, "product_ids": json.dumps(self.product_ids),
                "n_searches": self.n_searches}


class InvertedIndex:
    """Stemmed-token -> product ids. AND semantics, which is what production search uses."""

    def __init__(self, products: pd.DataFrame):
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.category = dict(zip(products["product_id"], products["category"]))
        for pid, title in zip(products["product_id"], products["title"]):
            for t in tokenize(title):
                self.postings[simple_stem(t)].add(pid)

    def match(self, terms: Sequence[str]) -> set[str]:
        sets = sorted((self.postings.get(t, set()) for t in terms), key=len)
        if not sets:
            return set()
        out = set(sets[0])
        for s in sets[1:]:
            out &= s
            if not out:
                break
        return out

    def known(self, term: str) -> bool:
        return term in self.postings

    def categories_for(self, pids: Iterable[str]) -> Counter:
        return Counter(self.category[p] for p in pids if p in self.category)


class TrendingLookup:
    def __init__(self, trending: pd.DataFrame, n: int):
        t = trending.sort_values("trend_score", ascending=False)
        self.by_category = {c: g["product_id"].head(n).tolist() for c, g in t.groupby("category")}
        self.global_top = t.drop_duplicates("product_id")["product_id"].head(n).tolist()

    def for_category(self, category: str | None) -> list[str]:
        return self.by_category.get(category, []) if category else []


def load_inputs(cfg: ZeroResultConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = {"lookback_days": cfg.lookback_days, "min_query_count": cfg.min_query_count}
    zq = read_sql(ZERO_RESULT_SQL, cfg.warehouse_uri, params=params)
    products = read_sql(PRODUCTS_SQL, cfg.warehouse_uri)
    trending = read_sql(TRENDING_SQL, cfg.warehouse_uri)
    require_columns(zq, ["query_text", "n_searches"], "zero-result queries")
    require_columns(products, ["product_id", "title", "category"], "catalog")
    require_columns(trending, ["product_id", "category", "trend_score"], "trending")
    zq["query_norm"] = zq["query_text"].map(normalize_query)
    zq = zq.groupby("query_norm", as_index=False)["n_searches"].sum()
    zq = zq[zq["query_norm"].str.len() > 0].sort_values("n_searches", ascending=False)
    log.info("zero-result queries=%d products=%d trending rows=%d", len(zq), len(products), len(trending))
    return zq, products, trending


def build_title_idf(products: pd.DataFrame) -> IdfTable:
    return IdfTable.fit([simple_stem(t) for t in tokenize(title)] for title in products["title"])


def relax(terms: list[str], index: InvertedIndex, idf: IdfTable, cfg: ZeroResultConfig
          ) -> tuple[list[str], list[str], set[str]]:
    """Greedy relaxation: drop the lowest-IDF droppable term until enough products match.

    Unknown terms (not in any title) are dropped first since they can never
    match; protected terms (bed sizes) are dropped only if nothing else is left.
    """
    kept = list(dict.fromkeys(terms))
    dropped: list[str] = []
    for t in [t for t in kept if not index.known(t)]:
        if len(kept) > cfg.min_terms:
            kept.remove(t)
            dropped.append(t)
    matches = index.match(kept)
    while len(matches) < cfg.min_matches and len(kept) > cfg.min_terms and len(dropped) < cfg.max_drops:
        candidates = [t for t in kept if t not in cfg.protected_terms] or kept
        victim = idf.least_informative(candidates)
        kept.remove(victim)
        dropped.append(victim)
        matches = index.match(kept)
    return kept, dropped, matches


def guess_category(terms: Sequence[str], index: InvertedIndex, min_votes: int) -> str | None:
    votes: Counter = Counter()
    for t in terms:
        votes.update(index.categories_for(index.postings.get(t, ())))
    if not votes:
        return None
    cat, n = votes.most_common(1)[0]
    return cat if n >= min_votes else None


def rank_matches(matches: set[str], trending: TrendingLookup, n: int) -> list[str]:
    """Order relaxed matches by trend position, unknowns last (stable by id)."""
    pos = {p: i for i, p in enumerate(trending.global_top)}
    return sorted(matches, key=lambda p: (pos.get(p, len(pos)), p))[:n]


def fallback_for(query_norm: str, n_searches: int, index: InvertedIndex, idf: IdfTable,
                 trending: TrendingLookup, cfg: ZeroResultConfig) -> Fallback:
    terms = [simple_stem(t) for t in tokenize(query_norm)]
    if terms:
        kept, dropped, matches = relax(terms, index, idf, cfg)
        if len(matches) >= cfg.min_matches and dropped:
            return Fallback(query_norm, STRATEGY_RELAXED,
                            rank_matches(matches, trending, cfg.n_fallback_products),
                            relaxed_query=" ".join(kept), dropped_terms=dropped, n_searches=n_searches)
        category = guess_category(terms, index, cfg.min_category_votes)
        products = trending.for_category(category)
        if products:
            return Fallback(query_norm, STRATEGY_CATEGORY, products, category=category,
                            dropped_terms=dropped, n_searches=n_searches)
    return Fallback(query_norm, STRATEGY_GLOBAL, trending.global_top, n_searches=n_searches)


def build_fallbacks(zq: pd.DataFrame, products: pd.DataFrame, trending_df: pd.DataFrame,
                    cfg: ZeroResultConfig) -> list[Fallback]:
    with timed(log, "index"):
        index = InvertedIndex(products)
        idf = build_title_idf(products)
    trending = TrendingLookup(trending_df, cfg.n_fallback_products)
    if not trending.global_top:
        raise RuntimeError("trending snapshot is empty; refusing to build fallbacks")
    with timed(log, "relax"):
        return [fallback_for(q, int(n), index, idf, trending, cfg)
                for q, n in zip(zq["query_norm"], zq["n_searches"])]


def report(fallbacks: list[Fallback]) -> dict[str, float]:
    by_strategy = Counter(f.strategy for f in fallbacks)
    searches = Counter()
    for f in fallbacks:
        searches[f.strategy] += f.n_searches
    total = sum(searches.values()) or 1
    results = {f.query_norm: f.product_ids for f in fallbacks if f.strategy != STRATEGY_GLOBAL}
    metrics = {f"share_{k}": v / total for k, v in searches.items()}
    metrics["specific_coverage"] = coverage(results, [f.query_norm for f in fallbacks])
    metrics["avg_dropped"] = sum(len(f.dropped_terms) for f in fallbacks) / max(1, len(fallbacks))
    log.info("queries by strategy: %s", dict(by_strategy))
    summarize_metrics(metrics, log, "zero-result fallbacks")
    return metrics


def to_frame(fallbacks: list[Fallback]) -> pd.DataFrame:
    df = pd.DataFrame([f.to_row() for f in fallbacks])
    df["built_at"] = datetime.now(timezone.utc)
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Build {OUTPUT_TABLE}")
    p.add_argument("--config")
    p.add_argument("--lookback-days", type=int)
    p.add_argument("--min-query-count", type=int)
    p.add_argument("--min-matches", type=int)
    p.add_argument("--max-drops", type=int)
    p.add_argument("--n-fallback-products", type=int)
    p.add_argument("--dry-run", action="store_true", default=None)
    p.add_argument("--explain", metavar="QUERY", help="print the fallback for one query and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "explain"}}
    cfg = load_config(ZeroResultConfig, args.config, overrides)
    zq, products, trending = load_inputs(cfg)
    if args.explain:
        zq = pd.DataFrame({"query_norm": [normalize_query(args.explain)], "n_searches": [0]})
    fallbacks = build_fallbacks(zq, products, trending, cfg)
    if args.explain:
        print(json.dumps(fallbacks[0].to_row(), indent=2))
        return 0
    report(fallbacks)
    write_table(to_frame(fallbacks), OUTPUT_TABLE, cfg.warehouse_uri, dry_run=cfg.dry_run, logger=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

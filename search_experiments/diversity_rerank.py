"""Maximal marginal relevance (MMR) re-ranking of search result lists.

Head queries like "throw pillows" often return a first page that is twelve
near-identical items from one brand in one colour. MMR re-orders the top
``pool_size`` candidates greedily:

    next = argmax_{d in R \\ S} [ λ · rel(d) − (1 − λ) · max_{s in S} sim(d, s) ]

where rel is the production ranker score min-max scaled to [0, 1] within the
list, and sim is a weighted attribute match

    sim(d, s) = w_brand·[brand_d = brand_s] + w_color·[color_d = color_s]
              + w_price·[band_d = band_s]

with weights summing to 1. Price bands are quantiles of price within the
product's category, so "cheap" means cheap for rugs, not cheap overall.

Inputs:  search.query_logs (served result lists + ranker scores), catalog.products
Output:  search.diverse_results (query_norm, position, product_id, original_position, relevance, mmr_score, lambda, built_at)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from shared_utils import (ExperimentConfig, get_logger, intra_list_diversity, load_config,
                          normalize_query, precision_at_k, read_sql, require_columns,
                          summarize_metrics, timed, write_table)

log = get_logger("search_exp.diversity_rerank")

INPUT_LOGS = "search.query_logs"
INPUT_PRODUCTS = "catalog.products"
OUTPUT_RESULTS = "search.diverse_results"

SERVED_SQL = f"""
SELECT query_text, result_product_ids, result_scores, clicked_product_id
FROM {INPUT_LOGS}
WHERE event_ts >= CURRENT_DATE - (:lookback_days * INTERVAL '1 day')
  AND cardinality(result_product_ids) > 0
"""

ATTR_SQL = f"""
SELECT product_id, brand, color_family AS color, category, price
FROM {INPUT_PRODUCTS}
WHERE is_active
"""


@dataclass
class DiversityConfig(ExperimentConfig):
    lookback_days: int = 7
    min_query_count: int = 50
    lam: float = 0.7
    pool_size: int = 48
    output_k: int = 24
    w_brand: float = 0.45
    w_color: float = 0.35
    w_price: float = 0.20
    n_price_bands: int = 4
    max_queries: int = 20_000
    pin_top: int = 1
    lambda_grid: list[float] = field(default_factory=lambda: [1.0, 0.85, 0.7, 0.55])

    def validate(self) -> None:
        super().validate()
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")
        if abs(self.w_brand + self.w_color + self.w_price - 1.0) > 1e-6:
            raise ValueError("attribute weights must sum to 1")
        if self.output_k > self.pool_size:
            raise ValueError("output_k cannot exceed pool_size")
        if self.pin_top < 0 or self.pin_top > self.output_k:
            raise ValueError("pin_top must be in [0, output_k]")


@dataclass(frozen=True)
class Attrs:
    brand: str
    color: str
    band: int


@dataclass
class ResultList:
    query_norm: str
    product_ids: list[str]
    scores: list[float]
    clicks: dict[str, int] = field(default_factory=dict)
    impressions: int = 0


class AttributeSimilarity:
    def __init__(self, attrs: Mapping[str, Attrs], w_brand: float, w_color: float, w_price: float):
        self.attrs, self.w = attrs, (w_brand, w_color, w_price)

    def __call__(self, a: str, b: str) -> float:
        x, y = self.attrs.get(a), self.attrs.get(b)
        if x is None or y is None:
            return 0.0
        wb, wc, wp = self.w
        return wb * (x.brand == y.brand) + wc * (x.color == y.color) + wp * (x.band == y.band)

    def distance(self, a: str, b: str) -> float:
        return 1.0 - self(a, b)


def price_bands(products: pd.DataFrame, n_bands: int) -> pd.Series:
    """Quantile band of price within category; tiny categories get a single band."""
    def band(s: pd.Series) -> pd.Series:
        if s.nunique() < n_bands:
            return pd.Series(0, index=s.index)
        return pd.qcut(s.rank(method="first"), n_bands, labels=False)
    return products.groupby("category")["price"].transform(band).fillna(0).astype(int)


def load_attributes(cfg: DiversityConfig) -> dict[str, Attrs]:
    p = read_sql(ATTR_SQL, cfg.warehouse_uri)
    require_columns(p, ["product_id", "brand", "color", "category", "price"], INPUT_PRODUCTS)
    p["brand"] = p["brand"].fillna("_unbranded").str.lower().str.strip()
    p["color"] = p["color"].fillna("_multi").str.lower()
    p["band"] = price_bands(p, cfg.n_price_bands)
    log.info("attributes for %d products, %d brands", len(p), p["brand"].nunique())
    return {r.product_id: Attrs(r.brand, r.color, r.band) for r in p.itertuples(index=False)}


def load_result_lists(cfg: DiversityConfig) -> list[ResultList]:
    raw = read_sql(SERVED_SQL, cfg.warehouse_uri, params={"lookback_days": cfg.lookback_days})
    require_columns(raw, ["query_text", "result_product_ids", "result_scores"], INPUT_LOGS)
    raw["query_norm"] = raw["query_text"].map(normalize_query)
    counts = raw["query_norm"].value_counts()
    head = counts[counts >= cfg.min_query_count].index[: cfg.max_queries]
    raw = raw[raw["query_norm"].isin(head)]
    lists = []
    for q, g in raw.groupby("query_norm", sort=False):
        # the most recent serve is a fine proxy for "the" list; ranker is deterministic per day
        last = g.iloc[-1]
        ids, scores = list(last["result_product_ids"]), [float(s) for s in last["result_scores"]]
        if len(ids) != len(scores):
            log.warning("length mismatch for %r (%d ids, %d scores); skipping", q, len(ids), len(scores))
            continue
        clicks = g["clicked_product_id"].dropna().value_counts().to_dict()
        lists.append(ResultList(q, ids[: cfg.pool_size], scores[: cfg.pool_size], clicks, len(g)))
    log.info("loaded %d head-query result lists", len(lists))
    return lists


def minmax(xs: Sequence[float]) -> np.ndarray:
    a = np.asarray(xs, dtype=float)
    lo, hi = a.min(), a.max()
    return np.ones_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)


def mmr(ids: Sequence[str], rel: np.ndarray, sim: Callable[[str, str], float], lam: float, k: int,
        pin_top: int = 0) -> list[tuple[str, float]]:
    """Greedy MMR. Keeps a running max-similarity vector so each step is O(n)."""
    n = len(ids)
    k = min(k, n)
    chosen: list[tuple[str, float]] = []
    remaining = np.ones(n, dtype=bool)
    max_sim = np.zeros(n)
    for step in range(k):
        if step < pin_top:
            j = step
            score = float(rel[j])
        else:
            obj = lam * rel - (1 - lam) * max_sim
            obj[~remaining] = -np.inf
            j = int(np.argmax(obj))
            score = float(obj[j])
        chosen.append((ids[j], score))
        remaining[j] = False
        for i in np.flatnonzero(remaining):
            s = sim(ids[i], ids[j])
            if s > max_sim[i]:
                max_sim[i] = s
    return chosen


def rerank(rl: ResultList, sim: AttributeSimilarity, lam: float, cfg: DiversityConfig) -> pd.DataFrame:
    rel = minmax(rl.scores)
    picked = mmr(rl.product_ids, rel, sim, lam, cfg.output_k, cfg.pin_top)
    orig = {p: i for i, p in enumerate(rl.product_ids)}
    return pd.DataFrame({
        "query_norm": rl.query_norm,
        "position": np.arange(1, len(picked) + 1),
        "product_id": [p for p, _ in picked],
        "original_position": [orig[p] + 1 for p, _ in picked],
        "relevance": [float(rel[orig[p]]) for p, _ in picked],
        "mmr_score": [s for _, s in picked],
        "lambda": lam,
    })


def distinct_share(ids: Sequence[str], attrs: Mapping[str, Attrs], key: str) -> float:
    vals = [getattr(attrs[p], key) for p in ids if p in attrs]
    return len(set(vals)) / len(vals) if vals else 0.0


def offline_eval(lists: Sequence[ResultList], sim: AttributeSimilarity, cfg: DiversityConfig) -> pd.DataFrame:
    """Trade-off table across the lambda grid: clicked-item precision vs. diversity."""
    rows = []
    k = cfg.output_k
    for lam in cfg.lambda_grid:
        prec, ild, brands, colors = [], [], [], []
        for rl in lists:
            ids = [p for p, _ in mmr(rl.product_ids, minmax(rl.scores), sim, lam, k, cfg.pin_top)]
            if rl.clicks:
                prec.append(precision_at_k(ids, set(rl.clicks), k))
            ild.append(intra_list_diversity(ids, sim.distance))
            brands.append(distinct_share(ids, sim.attrs, "brand"))
            colors.append(distinct_share(ids, sim.attrs, "color"))
        m = {"precision_at_k": float(np.mean(prec)) if prec else 0.0, "ild": float(np.mean(ild)),
             "brand_distinct": float(np.mean(brands)), "color_distinct": float(np.mean(colors))}
        summarize_metrics(m, log, f"lambda={lam:.2f}")
        rows.append({"lambda": lam, **m})
    return pd.DataFrame(rows)


def build(lists: Sequence[ResultList], sim: AttributeSimilarity, cfg: DiversityConfig) -> pd.DataFrame:
    frames = [rerank(rl, sim, cfg.lam, cfg) for rl in lists if len(rl.product_ids) >= 2]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["built_at"] = datetime.now(timezone.utc)
    moved = (out["position"] != out["original_position"]).mean()
    log.info("re-ranked %d lists; %.1f%% of slots changed", len(frames), 100 * moved)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"MMR diversification into {OUTPUT_RESULTS}")
    p.add_argument("--config")
    p.add_argument("--lam", type=float)
    p.add_argument("--pool-size", type=int)
    p.add_argument("--output-k", type=int)
    p.add_argument("--lookback-days", type=int)
    p.add_argument("--dry-run", action="store_true", default=None)
    p.add_argument("--sweep", action="store_true", help="run the lambda grid offline eval only")
    p.add_argument("--show", metavar="QUERY", help="print before/after for one query")
    return p.parse_args(argv)


def show(rl: ResultList, sim: AttributeSimilarity, cfg: DiversityConfig) -> None:
    after = rerank(rl, sim, cfg.lam, cfg)
    for row in after.itertuples(index=False):
        a = sim.attrs.get(row.product_id)
        desc = f"{a.brand:<18} {a.color:<10} band={a.band}" if a else "(no attrs)"
        print(f"{row.position:>3} <- {row.original_position:>3}  {row.product_id:<12} {desc}  "
              f"rel={row.relevance:.2f} mmr={row.mmr_score:.3f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "sweep", "show"}}
    cfg = load_config(DiversityConfig, args.config, overrides)
    with timed(log, "load"):
        attrs = load_attributes(cfg)
        lists = load_result_lists(cfg)
    sim = AttributeSimilarity(attrs, cfg.w_brand, cfg.w_color, cfg.w_price)
    if args.show:
        q = normalize_query(args.show)
        match = next((rl for rl in lists if rl.query_norm == q), None)
        if match is None:
            log.error("query %r is not among loaded head queries", q)
            return 1
        show(match, sim, cfg)
        return 0
    if args.sweep:
        print(offline_eval(lists, sim, cfg).to_string(index=False, float_format="%.4f"))
        return 0
    out = build(lists, sim, cfg)
    log.info("config used: %s", json.dumps({"lam": cfg.lam, "k": cfg.output_k, "pin_top": cfg.pin_top}))
    write_table(out, OUTPUT_RESULTS, cfg.warehouse_uri, dry_run=cfg.dry_run, logger=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Holiday gift guide fill.

For every active customer we pick ~12 gift suggestions for the gift guide page.

Pipeline (each step is a pure function over DataFrames so it can be tested in isolation):

    load_inputs -> candidate_gifts -> score_affinity -> apply_price_band -> diversify -> publish

Inputs
    features.customer_embeddings   customer_id, category_mix (json: category -> share), median_order_value
    recs.trending                  product_id, category, price, trend_score, as_of
Output
    recs.gift_guide                customer_id, rank, product_id, category, score, guide_season
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Final, NamedTuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("gift_guide")

EMBEDDINGS_TABLE: Final = "features.customer_embeddings"
TRENDING_TABLE: Final = "recs.trending"
OUTPUT_TABLE: Final = "recs.gift_guide"

GIFTABLE_CATEGORIES: Final[frozenset[str]] = frozenset({
    "throws", "candles", "diffusers", "bath_towels", "robes", "frames",
    "serveware", "decorative_pillows", "table_linens", "vases",
})


class GuideParams(NamedTuple):
    slots: int = 12
    max_per_category: int = 3
    trend_weight: float = 0.35
    price_band_low: float = 0.4   # x median order value
    price_band_high: float = 1.8
    candidate_pool: int = 400
    season: str = "holiday_2026"


# ---------------------------------------------------------------- io

def load_inputs(dsn: str, pool: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    eng = create_engine(dsn)
    customers = pd.read_sql(
        f"SELECT customer_id, category_mix, median_order_value FROM {EMBEDDINGS_TABLE} "
        "WHERE is_active = TRUE",
        eng,
    )
    trending = pd.read_sql(
        f"SELECT product_id, category, price, trend_score FROM {TRENDING_TABLE} "
        f"WHERE as_of = (SELECT MAX(as_of) FROM {TRENDING_TABLE}) "
        f"ORDER BY trend_score DESC LIMIT {pool * 3}",
        eng,
    )
    return customers, trending


def publish(df: pd.DataFrame, dsn: str) -> None:
    schema, table = OUTPUT_TABLE.split(".")
    df.to_sql(table, create_engine(dsn), schema=schema, if_exists="replace", index=False, chunksize=10_000)
    log.info("wrote %d rows to %s", len(df), OUTPUT_TABLE)


# ---------------------------------------------------------------- transforms

def candidate_gifts(trending: pd.DataFrame, pool: int) -> pd.DataFrame:
    return (
        trending
        .loc[lambda d: d["category"].isin(GIFTABLE_CATEGORIES)]
        .assign(trend_norm=lambda d: d["trend_score"] / d["trend_score"].max())
        .nlargest(pool, "trend_score")
        .reset_index(drop=True)
    )


def category_matrix(customers: pd.DataFrame, categories: list[str]) -> np.ndarray:
    """Customer x category share matrix, rows L1-normalised; unseen categories get 0."""
    idx = {c: i for i, c in enumerate(categories)}
    mat = np.zeros((len(customers), len(categories)), dtype=np.float32)
    for r, raw in enumerate(customers["category_mix"]):
        mix = json.loads(raw) if isinstance(raw, str) else (raw or {})
        for cat, share in mix.items():
            if cat in idx:
                mat[r, idx[cat]] = float(share)
    totals = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, totals, out=np.zeros_like(mat), where=totals > 0)


def score_affinity(customers: pd.DataFrame, cands: pd.DataFrame, p: GuideParams) -> pd.DataFrame:
    """score = (1 - w) * affinity(customer, product category) + w * normalised trend."""
    cats = sorted(cands["category"].unique())
    cust_mat = category_matrix(customers, cats)
    onehot = pd.get_dummies(cands["category"]).reindex(columns=cats, fill_value=0).to_numpy(np.float32)
    affinity = cust_mat @ onehot.T                          # (n_customers, n_products)
    # cold customers: no history in giftable categories -> fall back to pure trend
    cold = cust_mat.sum(axis=1) == 0
    affinity[cold] = 1.0 / len(cats)
    scores = (1 - p.trend_weight) * affinity + p.trend_weight * cands["trend_norm"].to_numpy()[None, :]

    n_c, n_p = scores.shape
    return pd.DataFrame({
        "customer_id": np.repeat(customers["customer_id"].to_numpy(), n_p),
        "median_order_value": np.repeat(customers["median_order_value"].to_numpy(), n_p),
        "product_id": np.tile(cands["product_id"].to_numpy(), n_c),
        "category": np.tile(cands["category"].to_numpy(), n_c),
        "price": np.tile(cands["price"].to_numpy(), n_c),
        "score": scores.ravel(),
    })


def apply_price_band(scored: pd.DataFrame, p: GuideParams) -> pd.DataFrame:
    mov = scored["median_order_value"].fillna(scored["median_order_value"].median())
    in_band = scored["price"].between(mov * p.price_band_low, mov * p.price_band_high)
    return scored.loc[in_band]


def diversify(scored: pd.DataFrame, p: GuideParams) -> pd.DataFrame:
    """Greedy cap per category: keep the best `max_per_category` per (customer, category), then top `slots`."""
    return (
        scored
        .sort_values(["customer_id", "score"], ascending=[True, False])
        .assign(cat_rank=lambda d: d.groupby(["customer_id", "category"]).cumcount())
        .loc[lambda d: d["cat_rank"] < p.max_per_category]
        .assign(rank=lambda d: d.groupby("customer_id").cumcount() + 1)
        .loc[lambda d: d["rank"] <= p.slots]
        .assign(guide_season=p.season)
        [["customer_id", "rank", "product_id", "category", "score", "guide_season"]]
    )


def build_guide(customers: pd.DataFrame, trending: pd.DataFrame, p: GuideParams, batch: int = 5_000) -> pd.DataFrame:
    cands = candidate_gifts(trending, p.candidate_pool)
    log.info("%d giftable candidates across %d categories", len(cands), cands["category"].nunique())
    parts = []
    for start in range(0, len(customers), batch):
        chunk = customers.iloc[start:start + batch]
        parts.append(diversify(apply_price_band(score_affinity(chunk, cands, p), p), p))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--slots", type=int, default=GuideParams.slots)
    ap.add_argument("--season", default=GuideParams.season)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    p = GuideParams(slots=args.slots, season=args.season)
    customers, trending = load_inputs(args.dsn, p.candidate_pool)
    guide = build_guide(customers, trending, p)
    short = guide.groupby("customer_id").size().lt(p.slots).sum()
    log.info("%d customers, %d with fewer than %d gifts", guide["customer_id"].nunique(), short, p.slots)
    if not args.dry_run:
        publish(guide, args.dsn)


if __name__ == "__main__":
    main()

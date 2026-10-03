"""RFM segmentation for CRM.

Scores every customer with an order in the lookback window on three axes:

    R (recency)   days since last order      - fewer days  = higher score
    F (frequency) number of distinct orders  - more orders = higher score
    M (monetary)  net revenue                - more spend  = higher score

Each axis is cut into quintiles (1-5). The (R, FM) pair, where FM is the
rounded mean of F and M, is then mapped to a named segment through a lookup
grid that CRM uses to pick campaigns.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import NamedTuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger("rfm")

LOOKBACK_DAYS = 730
OUTPUT_TABLE = "crm.rfm_segments"

ORDERS_QUERY = """
    SELECT customer_id, order_id, order_ts, quantity * unit_price - COALESCE(discount_amount, 0) AS net_amount
    FROM orders.lines
    WHERE order_ts >= CURRENT_DATE - INTERVAL '{days} days'
      AND customer_id IS NOT NULL
      AND COALESCE(is_returned, FALSE) = FALSE
"""


class Segment(NamedTuple):
    name: str
    action: str


# Keys are (R score, FM score). Rows read top-to-bottom from recent to lapsed.
SEGMENT_GRID: dict[tuple[int, int], Segment] = {}

_GRID_SPEC: list[tuple[range, range, Segment]] = [
    (range(5, 6), range(4, 6), Segment("Champions", "reward, early access")),
    (range(3, 5), range(4, 6), Segment("Loyal Customers", "upsell bundles")),
    (range(4, 6), range(2, 4), Segment("Potential Loyalists", "membership offer")),
    (range(5, 6), range(1, 2), Segment("New Customers", "onboarding series")),
    (range(4, 5), range(1, 2), Segment("Promising", "second-purchase nudge")),
    (range(3, 4), range(1, 4), Segment("Need Attention", "limited-time offer")),
    (range(2, 3), range(1, 3), Segment("About to Sleep", "reactivation content")),
    (range(1, 3), range(3, 5), Segment("At Risk", "personal win-back")),
    (range(1, 3), range(5, 6), Segment("Can't Lose Them", "high-value win-back")),
    (range(1, 2), range(1, 3), Segment("Hibernating", "low-cost reminders only")),
]
for r_rng, fm_rng, seg in _GRID_SPEC:
    for r in r_rng:
        for fm in fm_rng:
            SEGMENT_GRID.setdefault((r, fm), seg)

FALLBACK = Segment("Others", "default newsletter")


def quintile_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Score a series 1-5 by quintile.

    Ranks first so heavy ties (e.g. most customers have one order) do not
    collapse the bins, which ``pd.qcut`` on raw values would do.

    Args:
        values: Raw metric per customer.
        higher_is_better: If False the scale is inverted (used for recency).

    Returns:
        Integer scores in 1..5 aligned to ``values``.
    """
    ranked = values.rank(method="first", ascending=higher_is_better)
    return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def compute_rfm(lines: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Aggregate order lines into per-customer RFM metrics and scores.

    Args:
        lines: Order lines with ``customer_id``, ``order_id``, ``order_ts``, ``net_amount``.
        as_of: Reference date for recency.

    Returns:
        One row per customer with raw metrics and R/F/M scores.
    """
    per_customer = lines.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("net_amount", "sum"),
    )
    per_customer = per_customer[per_customer["monetary"] > 0]
    per_customer["recency_days"] = (as_of - per_customer["last_order"]).dt.days.clip(lower=0)

    per_customer["r_score"] = quintile_score(per_customer["recency_days"], higher_is_better=False)
    per_customer["f_score"] = quintile_score(per_customer["frequency"])
    per_customer["m_score"] = quintile_score(per_customer["monetary"])
    per_customer["fm_score"] = np.floor((per_customer["f_score"] + per_customer["m_score"]) / 2 + 0.5).astype(int)
    per_customer["rfm_code"] = (
        per_customer["r_score"].astype(str) + per_customer["f_score"].astype(str) + per_customer["m_score"].astype(str)
    )
    return per_customer.reset_index()


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Attach segment name and suggested CRM action from the lookup grid."""
    segs = [SEGMENT_GRID.get((r, fm), FALLBACK) for r, fm in zip(rfm["r_score"], rfm["fm_score"])]
    rfm["segment"] = [s.name for s in segs]
    rfm["crm_action"] = [s.action for s in segs]
    return rfm


def run(as_of: str | None, dry_run: bool) -> None:
    engine = create_engine(os.environ["WAREHOUSE_URL"])
    ref = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    lines = pd.read_sql(ORDERS_QUERY.format(days=LOOKBACK_DAYS), engine, parse_dates=["order_ts"])
    logger.info("Loaded %d order lines for %d customers", len(lines), lines["customer_id"].nunique())

    rfm = assign_segments(compute_rfm(lines, ref))
    rfm["as_of_date"] = ref.date()

    mix = rfm["segment"].value_counts(normalize=True).round(3)
    logger.info("Segment mix:\n%s", mix.to_string())

    cols = ["customer_id", "recency_days", "frequency", "monetary", "r_score", "f_score",
            "m_score", "rfm_code", "segment", "crm_action", "as_of_date"]
    if dry_run:
        print(rfm[cols].sample(min(25, len(rfm)), random_state=1).to_string(index=False))
        return
    schema, table = OUTPUT_TABLE.split(".")
    rfm[cols].to_sql(table, engine, schema=schema, if_exists="replace", index=False, chunksize=50_000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RFM-based CRM segments.")
    parser.add_argument("--as-of", help="Reference date (YYYY-MM-DD); defaults to today.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(args.as_of, args.dry_run)


if __name__ == "__main__":
    main()

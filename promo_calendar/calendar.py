"""Next-quarter promo calendar.

Decides which categories get a discount in which weeks of the coming quarter, using how past
promos performed (marketing.promo_history), and writes the plan to pricing.promo_calendar with
the expected incremental revenue for each promo week.

Approach
  1. For every (category, week-of-year) estimate incremental revenue per promo week from history,
     shrunk towards the category average when we have few observations.
  2. Greedy assignment over the quarter: repeatedly take the best remaining (category, week) that
     does not break any rule below, until the weekly slot budget is used up.

Business rules (agreed with merchandising, 2026 Q3 planning)
  - At most MAX_PROMOS_PER_WEEK categories on promo in the same week (site banner space, ops load).
  - A category cannot be on promo two weeks in a row, and needs COOLDOWN_WEEKS weeks off between promos
    so customers don't learn to wait for the discount.
  - Each category gets at most MAX_PROMOS_PER_CATEGORY promo weeks per quarter.
  - Blackout weeks: no promo on anything (full-price launches, Black Friday handled separately).
  - Category-specific blackouts (e.g. new bedding collection launch week).
"""
import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("promo_calendar")

HISTORY_TABLE = "marketing.promo_history"
CALENDAR_TABLE = "pricing.promo_calendar"

MAX_PROMOS_PER_WEEK = 3
MAX_PROMOS_PER_CATEGORY = 4
COOLDOWN_WEEKS = 2
SHRINK_K = 3.0          # pseudo-observations for shrinkage towards the category mean
MIN_EXPECTED_INCREMENTAL = 0.0   # never schedule a promo we expect to lose money on

# ISO week numbers, same every year until merchandising tells us otherwise
GLOBAL_BLACKOUT_WEEKS = {47, 48}   # Black Friday / Cyber Monday run as a separate event
CATEGORY_BLACKOUTS = {
    "bedding": {41},      # autumn collection launch, full price only
    "rugs": {45},
}


def load_history(engine) -> pd.DataFrame:
    sql = f"""
        SELECT category, promo_start, discount_pct, revenue, baseline_revenue
        FROM {HISTORY_TABLE}
        WHERE promo_start >= CURRENT_DATE - INTERVAL '3 years'
    """
    df = pd.read_sql(sql, engine, parse_dates=["promo_start"])
    # incremental = revenue during promo minus what the baseline model expected without it
    df["incremental"] = df["revenue"] - df["baseline_revenue"]
    df["iso_week"] = df["promo_start"].dt.isocalendar().week.astype(int)
    return df


def quarter_weeks(quarter_start: date) -> list:
    """Mondays of the 13 weeks starting at quarter_start (rounded back to a Monday)."""
    monday = quarter_start - timedelta(days=quarter_start.weekday())
    return [monday + timedelta(weeks=i) for i in range(13)]


def expected_incremental(history: pd.DataFrame, categories: list, weeks: list) -> np.ndarray:
    """(n_categories, n_weeks) matrix of expected incremental revenue for a promo in that week.

    Per cell: shrunk mean = (n * week_mean + K * category_mean) / (n + K).
    Categories never promoted get 0, so they are only picked if nothing positive is left.
    """
    exp = np.zeros((len(categories), len(weeks)))
    for ci, cat in enumerate(categories):
        hist_c = history[history["category"] == cat]
        if hist_c.empty:
            continue
        cat_mean = hist_c["incremental"].mean()
        for wi, wk in enumerate(weeks):
            iso = wk.isocalendar()[1]
            # neighbouring weeks count too: seasonality is smooth, a single week is too noisy
            obs = hist_c.loc[(hist_c["iso_week"] - iso).abs() <= 1, "incremental"].to_numpy()
            n = len(obs)
            week_mean = obs.mean() if n > 0 else 0.0
            exp[ci, wi] = (n * week_mean + SHRINK_K * cat_mean) / (n + SHRINK_K)
    return exp


def allowed(cat: str, ci: int, wi: int, iso: int, plan: np.ndarray) -> bool:
    if iso in GLOBAL_BLACKOUT_WEEKS:
        return False
    if iso in CATEGORY_BLACKOUTS.get(cat, set()):
        return False
    if plan[:, wi].sum() >= MAX_PROMOS_PER_WEEK:
        return False
    if plan[ci].sum() >= MAX_PROMOS_PER_CATEGORY:
        return False
    # cooldown: no other promo for this category within COOLDOWN_WEEKS either side (this also
    # rules out back-to-back weeks, i.e. overlapping promos)
    lo, hi = max(0, wi - COOLDOWN_WEEKS), min(plan.shape[1], wi + COOLDOWN_WEEKS + 1)
    if plan[ci, lo:hi].any():
        return False
    return True


def build_plan(exp: np.ndarray, categories: list, weeks: list) -> np.ndarray:
    plan = np.zeros_like(exp, dtype=bool)
    # visit cells from best to worst expected incremental revenue
    order = np.dstack(np.unravel_index(np.argsort(-exp, axis=None), exp.shape))[0]
    for ci, wi in order:
        if exp[ci, wi] <= MIN_EXPECTED_INCREMENTAL:
            break   # everything after this is worse, stop
        iso = weeks[wi].isocalendar()[1]
        if allowed(categories[ci], ci, wi, iso, plan):
            plan[ci, wi] = True
    return plan


def to_frame(plan: np.ndarray, exp: np.ndarray, history: pd.DataFrame, categories: list, weeks: list) -> pd.DataFrame:
    # suggested discount = median depth that category ran historically, rounded to 5%
    depth = history.groupby("category")["discount_pct"].median()
    rows = []
    for ci, cat in enumerate(categories):
        for wi, wk in enumerate(weeks):
            if not plan[ci, wi]:
                continue
            d = depth.get(cat, 0.15)
            rows.append({
                "category": cat,
                "week_start": wk,
                "iso_week": wk.isocalendar()[1],
                "discount_pct": round(d * 20) / 20,
                "expected_incremental_revenue": round(float(exp[ci, wi]), 2),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Build next quarter's promo calendar")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--quarter-start", type=date.fromisoformat, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(args.dsn)
    history = load_history(engine)
    categories = sorted(history["category"].unique())
    weeks = quarter_weeks(args.quarter_start)

    exp = expected_incremental(history, categories, weeks)
    plan = build_plan(exp, categories, weeks)
    cal = to_frame(plan, exp, history, categories, weeks)
    cal["quarter_start"] = args.quarter_start
    log.info("%d promo weeks planned across %d categories, expected incremental %.0f",
             len(cal), cal["category"].nunique(), cal["expected_incremental_revenue"].sum())

    if args.dry_run:
        print(cal.pivot_table(index="category", columns="iso_week", values="discount_pct").to_string())
        return
    schema, table = CALENDAR_TABLE.split(".")
    cal.to_sql(table, engine, schema=schema, if_exists="append", index=False)


if __name__ == "__main__":
    main()

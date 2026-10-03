"""End-of-season markdown planning.

For every SKU carrying more stock than it can sell at full price before the
season deadline, pick a markdown ladder (discount per week) that maximizes
revenue minus holding cost. Solved with dynamic programming over
(week, stock bucket, current discount level).

Reads:  pricing.elasticities, inventory.stock_levels
Writes: pricing.markdown_plan
"""
import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("markdown")

ELASTICITY_TABLE = "pricing.elasticities"
STOCK_TABLE = "inventory.stock_levels"
PLAN_TABLE = "pricing.markdown_plan"

# Allowed discount steps. Business rule: markdowns are permanent within a
# season, so the ladder may only stay or step down (never back up in price).
DISCOUNTS = np.array([0.0, 0.10, 0.20, 0.30, 0.40])
N_BUCKETS = 50                  # stock discretization
HOLDING_COST_PER_UNIT_WEEK = 0.004   # fraction of full price per unit per week
SALVAGE_FRACTION = 0.15         # liquidation value of unsold stock at deadline
UNSOLD_PENALTY = 0.25           # extra penalty per unsold unit (fraction of price);
                                # merch wants the shelf cleared for next season
DEFAULT_ELASTICITY = -1.5       # used when a category has no estimate


def load_inputs(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock = pd.read_sql(
        f"""SELECT sku, category, on_hand_units, full_price, baseline_weekly_units,
                   season_end_date
            FROM {STOCK_TABLE}
            WHERE season_end_date > CURRENT_DATE""",
        engine,
    )
    elast = pd.read_sql(f"SELECT category, elasticity FROM {ELASTICITY_TABLE}", engine)
    return stock, elast


def excess_stock(stock: pd.DataFrame, today: date) -> pd.DataFrame:
    weeks_left = ((pd.to_datetime(stock["season_end_date"]) - pd.Timestamp(today)).dt.days // 7)
    stock = stock.assign(weeks_left=weeks_left.clip(lower=0))
    # A SKU is "excess" if baseline (full price) sales cannot clear it in time.
    projected = stock["baseline_weekly_units"] * stock["weeks_left"]
    return stock[(stock["weeks_left"] > 0) & (stock["on_hand_units"] > projected)].copy()


def expected_demand(base: float, discount: float, elasticity: float) -> float:
    # Constant-elasticity demand: q = q0 * (p / p0) ^ e, with p / p0 = 1 - d.
    return base * (1.0 - discount) ** elasticity


def plan_sku(units: int, price: float, base: float, elasticity: float, weeks: int):
    """DP over weeks x stock buckets x discount level. Returns (ladder, value)."""
    bucket_size = max(units / N_BUCKETS, 1.0)
    n_b = int(np.ceil(units / bucket_size)) + 1
    n_d = len(DISCOUNTS)

    # V[t, b, d]: best value from week t onward holding bucket b with current discount index d.
    V = np.zeros((weeks + 1, n_b, n_d))
    choice = np.zeros((weeks, n_b, n_d), dtype=int)

    # Terminal: leftover stock is salvaged, minus penalty for not clearing.
    for b in range(n_b):
        left = b * bucket_size
        V[weeks, b, :] = left * price * (SALVAGE_FRACTION - UNSOLD_PENALTY)

    for t in range(weeks - 1, -1, -1):
        for b in range(n_b):
            stock_units = b * bucket_size
            for d in range(n_d):
                best, best_k = -np.inf, d
                # Only same or deeper discount is allowed.
                for k in range(d, n_d):
                    q = expected_demand(base, DISCOUNTS[k], elasticity)
                    sold = min(q, stock_units)
                    remaining = stock_units - sold
                    revenue = sold * price * (1.0 - DISCOUNTS[k])
                    # Holding cost charged on stock carried into next week.
                    holding = remaining * price * HOLDING_COST_PER_UNIT_WEEK
                    nb = min(int(round(remaining / bucket_size)), n_b - 1)
                    val = revenue - holding + V[t + 1, nb, k]
                    if val > best:
                        best, best_k = val, k
                V[t, b, d] = best
                choice[t, b, d] = best_k

    # Forward pass to recover the ladder from the starting state (full stock, no discount).
    ladder, rows = [], []
    stock_units, d = float(units), 0
    for t in range(weeks):
        b = min(int(round(stock_units / bucket_size)), n_b - 1)
        k = choice[t, b, d]
        q = expected_demand(base, DISCOUNTS[k], elasticity)
        sold = min(q, stock_units)
        rows.append((t, DISCOUNTS[k], sold, stock_units - sold))
        stock_units -= sold
        d = k
        ladder.append(DISCOUNTS[k])
    b0 = min(int(round(units / bucket_size)), n_b - 1)
    return ladder, rows, float(V[0, b0, 0])


def build_plan(cands: pd.DataFrame, elast: pd.DataFrame, today: date) -> pd.DataFrame:
    e_map = dict(zip(elast["category"], elast["elasticity"]))
    out = []
    for r in cands.itertuples(index=False):
        e = e_map.get(r.category, DEFAULT_ELASTICITY)
        # Guard against positive or near-zero estimates: markdowns would never help.
        if e > -0.2:
            log.warning("sku %s: implausible elasticity %.2f, using default", r.sku, e)
            e = DEFAULT_ELASTICITY
        ladder, rows, value = plan_sku(int(r.on_hand_units), float(r.full_price),
                                       float(r.baseline_weekly_units), e, int(r.weeks_left))
        for week, disc, sold, left in rows:
            out.append({
                "sku": r.sku,
                "week_start": today + timedelta(weeks=week),
                "discount_pct": round(disc * 100),
                "markdown_price": round(r.full_price * (1 - disc), 2),
                "expected_units": round(sold, 1),
                "expected_remaining": round(left, 1),
                "elasticity_used": e,
                "plan_value": round(value, 2),
            })
        log.info("sku %s: ladder %s, final stock %.0f", r.sku,
                 "/".join(f"{int(x * 100)}" for x in ladder), rows[-1][3] if rows else 0)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="End-of-season markdown planner")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    engine = create_engine(args.dsn)
    stock, elast = load_inputs(engine)
    cands = excess_stock(stock, args.as_of)
    log.info("%d of %d SKUs have excess stock", len(cands), len(stock))

    plan = build_plan(cands, elast, args.as_of)
    if args.dry_run or plan.empty:
        print(plan.head(50).to_string(index=False))
        return
    schema, table = PLAN_TABLE.split(".")
    plan.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    log.info("wrote %d plan rows to %s", len(plan), PLAN_TABLE)


if __name__ == "__main__":
    main()

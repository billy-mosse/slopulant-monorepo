"""Hourly trending products per category -> recs.trending.

score = decayed weighted activity in the last 48h vs. 28-day hourly baseline (z-score).
"""
import argparse
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import islice

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger("trending")

EVENT_WEIGHTS = {"view": 1.0, "add_to_cart": 4.0, "purchase": 10.0}
HALF_LIFE_H = 6.0
RECENT_H = 48
BASELINE_DAYS = 28
TOP_K = 50
MIN_BASELINE_EVENTS = 20
MIN_RECENT_EVENTS = 5
MIN_STOCK = 10

EVENTS_SQL = """
SELECT sku, category_id, event_type, event_ts
FROM sessions.events
WHERE event_type IN ('view', 'add_to_cart') AND event_ts >= :since
"""
PURCHASES_SQL = """
SELECT sku, category_id, 'purchase' AS event_type, order_ts AS event_ts
FROM orders.lines
WHERE order_ts >= :since AND status <> 'cancelled'
"""


def load(engine, now):
    since = now - timedelta(days=BASELINE_DAYS)
    with engine.connect() as c:
        ev = pd.read_sql(text(EVENTS_SQL), c, params={"since": since})
        po = pd.read_sql(text(PURCHASES_SQL), c, params={"since": since})
    df = pd.concat([ev, po], ignore_index=True)
    df["event_ts"] = pd.to_datetime(df.event_ts)
    df["w"] = df.event_type.map(EVENT_WEIGHTS)
    return df


def velocity(df, now):
    recent_cut = now - timedelta(hours=RECENT_H)
    age_h = (now - df.event_ts).dt.total_seconds() / 3600
    df = df.assign(decayed=df.w * (0.5 ** (age_h / HALF_LIFE_H)), recent=df.event_ts >= recent_cut)

    # baseline: weighted events per 48h window over the prior 26 days
    base = df[~df.recent]
    n_windows = (BASELINE_DAYS * 24 - RECENT_H) / RECENT_H
    win = ((recent_cut - base.event_ts).dt.total_seconds() // (RECENT_H * 3600)).astype(int)
    per_win = base.assign(win=win).groupby(["sku", "win"]).w.sum()
    stats = defaultdict(lambda: [0.0, 0.0, 0])
    for (sku, _), v in per_win.items():
        s = stats[sku]
        s[0] += v
        s[1] += v * v
        s[2] += 1
    rows = []
    rec = df[df.recent].groupby(["sku", "category_id"]).agg(decayed=("decayed", "sum"), n=("w", "size"))
    base_n = base.groupby("sku").size()
    # decayed sum over the window, normalised to an undecayed-equivalent
    norm = HALF_LIFE_H / math.log(2) / RECENT_H * (1 - 0.5 ** (RECENT_H / HALF_LIFE_H))
    for (sku, cat), r in rec.iterrows():
        tot, sq, _ = stats[sku]
        mu = tot / n_windows
        var = max(sq / n_windows - mu * mu, 0.0)
        sd = math.sqrt(var + mu + 1.0)  # Poisson floor keeps sparse items from exploding
        cur = r.decayed / norm
        rows.append((sku, cat, cur, mu, (cur - mu) / sd, r.n, int(base_n.get(sku, 0))))
    return pd.DataFrame(rows, columns=["sku", "category_id", "current", "baseline", "z", "recent_n", "baseline_n"])


def apply_filters(v, stock):
    keep = (v.recent_n >= MIN_RECENT_EVENTS) & (v.baseline_n >= MIN_BASELINE_EVENTS) & (v.z > 0)
    if stock is not None:
        keep &= v.sku.map(stock).fillna(0) >= MIN_STOCK
    log.info("filters keep %d / %d", keep.sum(), len(v))
    return v[keep]


def top_per_category(v, k=TOP_K):
    out = []
    for cat, g in v.sort_values("z", ascending=False).groupby("category_id", sort=False):
        for rank, row in enumerate(islice(g.itertuples(), k), 1):
            out.append({"category_id": cat, "sku": row.sku, "rank": rank,
                        "velocity_z": round(row.z, 3), "current": round(row.current, 2),
                        "baseline": round(row.baseline, 2)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--stock-snapshot", help="parquet of sku,units_on_hand from the WMS export")
    ap.add_argument("--now")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    now = datetime.fromisoformat(a.now) if a.now else datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    eng = create_engine(a.dsn)
    stock = pd.read_parquet(a.stock_snapshot).set_index("sku").units_on_hand if a.stock_snapshot else None
    out = top_per_category(apply_filters(velocity(load(eng, now), now), stock))
    out["computed_at"] = now
    out.to_sql("trending", eng, schema="recs", if_exists="append", index=False)
    log.info("recs.trending: %d rows across %d categories", len(out), out.category_id.nunique())


if __name__ == "__main__":
    main()

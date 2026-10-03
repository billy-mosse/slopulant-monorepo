"""
Best email send window per customer.

Hour-of-week bins h = 24 * weekday + hour (168 bins), in the customer's local time.
For customer u, s_uh sends and o_uh opens. Open rate shrunk to a prior:

    r_uh = (o_uh + a * p_h) / (s_uh + a)

p_h is the cohort open rate for bin h (itself shrunk to the global rate), a is the
prior strength in pseudo-sends. Neighbouring bins are then smoothed with a
circular kernel [0.25, 0.5, 0.25] since 9:00 and 10:00 are not independent.

Customers with fewer than MIN_SENDS sends get the cohort's best window.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger(__name__)

EVENTS_SQL = """
SELECT customer_id,
       local_tz,
       EXTRACT(ISODOW FROM event_ts AT TIME ZONE local_tz) - 1 AS dow,
       EXTRACT(HOUR   FROM event_ts AT TIME ZONE local_tz)     AS hour,
       SUM(CASE WHEN event_type = 'send' THEN 1 ELSE 0 END)    AS sends,
       SUM(CASE WHEN event_type = 'open' THEN 1 ELSE 0 END)    AS opens
FROM marketing.email_events
WHERE event_ts >= CURRENT_DATE - INTERVAL '180 days'
GROUP BY 1, 2, 3, 4
"""
TARGET = "marketing.send_windows"

H = 168
ALPHA = 20.0          # pseudo-sends for customer -> cohort prior
ALPHA_COHORT = 500.0  # pseudo-sends for cohort -> global prior
MIN_SENDS = 8
KERNEL = np.array([0.25, 0.5, 0.25])


def circ_smooth(r: np.ndarray) -> np.ndarray:
    """Smooth along the last axis, wrapping Sunday 23h into Monday 0h."""
    return (KERNEL[0] * np.roll(r, 1, -1) + KERNEL[1] * r + KERNEL[2] * np.roll(r, -1, -1))


def to_matrix(df: pd.DataFrame, key: str) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    df = df.assign(h=(df["dow"] * 24 + df["hour"]).astype(int))
    keys = pd.Index(df[key].unique())
    i = keys.get_indexer(df[key])
    S, O = np.zeros((len(keys), H)), np.zeros((len(keys), H))
    np.add.at(S, (i, df["h"].to_numpy()), df["sends"].to_numpy())
    np.add.at(O, (i, df["h"].to_numpy()), df["opens"].to_numpy())
    return keys, S, O


def cohort_priors(df: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
    # cohort = local timezone; global rate g is the base prior
    keys, S, O = to_matrix(df, "local_tz")
    g = O.sum() / max(S.sum(), 1)
    P = (O + ALPHA_COHORT * g) / (S + ALPHA_COHORT)
    log.info("global open rate %.4f over %d tz cohorts", g, len(keys))
    return keys, circ_smooth(P)


def customer_windows(df: pd.DataFrame, tz_keys: pd.Index, P: np.ndarray) -> pd.DataFrame:
    tz = df.groupby("customer_id")["local_tz"].agg(lambda s: s.mode().iat[0])
    cust, S, O = to_matrix(df, "customer_id")
    prior = P[tz_keys.get_indexer(tz.loc[cust])]
    R = circ_smooth((O + ALPHA * prior) / (S + ALPHA))

    total = S.sum(1)
    best = np.where(total >= MIN_SENDS, R.argmax(1), prior.argmax(1))
    rate = np.where(total >= MIN_SENDS, R.max(1), prior.max(1))
    lift = rate / prior.mean(1) - 1
    return pd.DataFrame({
        "customer_id": cust,
        "local_tz": tz.loc[cust].to_numpy(),
        "best_dow": best // 24,
        "best_hour": best % 24,
        "expected_open_rate": rate,
        "lift_vs_flat": lift,
        "source": np.where(total >= MIN_SENDS, "customer", "cohort_default"),
        "n_sends": total.astype(int),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    engine = create_engine(os.environ["WAREHOUSE_URL"])
    df = pd.read_sql(EVENTS_SQL, engine).dropna(subset=["local_tz"])
    tz_keys, P = cohort_priors(df)
    out = customer_windows(df, tz_keys, P)
    out["computed_at"] = pd.Timestamp.utcnow()
    log.info("%d customers, %.1f%% on cohort default",
             len(out), 100 * (out["source"] == "cohort_default").mean())

    if args.dry_run:
        print(out.groupby(["best_dow", "best_hour"]).size().nlargest(10))
        return
    schema, table = TARGET.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="replace", index=False)


if __name__ == "__main__":
    main()

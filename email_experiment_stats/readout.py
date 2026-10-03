"""Email A/B test readout: open/click rates, CUPED-adjusted lift, z-tests, ship call."""
import argparse
import logging
import os

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

PRE_PERIOD_DAYS = 28
ALPHA = 0.05
MIN_LIFT_TO_SHIP = 0.01  # relative lift on clicks we consider worth shipping
PRIMARY_METRIC = "clicked"

ASSIGNMENTS_SQL = """
SELECT experiment_id, unit_id AS customer_id, variant, assigned_at
FROM experiments.assignments
WHERE experiment_id = %(exp)s
"""
EVENTS_SQL = """
SELECT customer_id, campaign_id, event_type, event_ts
FROM marketing.email_events
WHERE event_ts >= %(start)s AND event_ts < %(end)s
"""
READOUT_TABLE = "marketing.experiment_readouts"


def two_prop_ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pooled if se_pooled > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(1 - ALPHA / 2)
    return z, p, (p2 - p1 - zc * se_unpooled, p2 - p1 + zc * se_unpooled)


def build_readout(exp_id, engine, end_ts=None):
    assignments = pd.read_sql(ASSIGNMENTS_SQL, engine, params={"exp": exp_id}, parse_dates=["assigned_at"])
    assignments = assignments.drop_duplicates("customer_id", keep="first")
    start = assignments["assigned_at"].min()
    end = pd.Timestamp(end_ts) if end_ts else pd.Timestamp.utcnow().tz_localize(None)
    pre_start = start - pd.Timedelta(days=PRE_PERIOD_DAYS)
    logger.info(f"experiment {exp_id}: {len(assignments):,} units, variants={sorted(assignments.variant.unique())}, "
                f"window {start:%Y-%m-%d} -> {end:%Y-%m-%d}")

    events = pd.read_sql(EVENTS_SQL, engine, params={"start": pre_start, "end": end}, parse_dates=["event_ts"])
    events = events[events["customer_id"].isin(assignments["customer_id"])]
    events = events.merge(assignments[["customer_id", "assigned_at"]], on="customer_id")
    events["period"] = np.where(events["event_ts"] < events["assigned_at"], "pre", "post")
    logger.info(f"{len(events):,} email events for assigned customers ({(events.period == 'pre').mean():.1%} pre-period)")

    # per-customer flags in post period, engagement counts in pre period
    flags = (
        events.assign(n=1)
        .pivot_table(index=["customer_id", "period"], columns="event_type", values="n", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    for col in ("delivered", "open", "click"):
        if col not in flags:
            flags[col] = 0
    post = flags[flags.period == "post"].set_index("customer_id")
    pre = flags[flags.period == "pre"].set_index("customer_id")

    units = assignments.set_index("customer_id")[["variant"]].copy()
    units["delivered"] = post["delivered"].reindex(units.index).fillna(0)
    units = units[units["delivered"] > 0]
    units["opened"] = (post["open"].reindex(units.index).fillna(0) > 0).astype(int)
    units["clicked"] = (post["click"].reindex(units.index).fillna(0) > 0).astype(int)
    units["pre_engagement"] = (pre["open"] + 2 * pre["click"]).reindex(units.index).fillna(0)
    logger.info(f"{len(units):,} units received at least one email in the test")

    # CUPED: theta = cov(Y, X) / var(X) pooled across arms; Y_adj = Y - theta * (X - mean X)
    x = units["pre_engagement"]
    x_bar = x.mean()
    for metric in ("opened", "clicked"):
        var_x = x.var()
        theta = units[metric].cov(x) / var_x if var_x > 0 else 0.0
        units[f"{metric}_cuped"] = units[metric] - theta * (x - x_bar)
        logger.info(f"CUPED theta for {metric}: {theta:.4f}, corr with pre-period={units[metric].corr(x):.3f}")

    variants = sorted(units["variant"].unique())
    control = "control" if "control" in variants else variants[0]
    summary = units.groupby("variant").agg(
        n=("delivered", "size"),
        opens=("opened", "sum"),
        clicks=("clicked", "sum"),
        open_rate=("opened", "mean"),
        click_rate=("clicked", "mean"),
        open_rate_cuped=("opened_cuped", "mean"),
        click_rate_cuped=("clicked_cuped", "mean"),
        click_var_cuped=("clicked_cuped", "var"),
    )
    logger.info(f"raw rates:\n{summary[['n', 'open_rate', 'click_rate']]}")

    c = summary.loc[control]
    rows = []
    for v in variants:
        if v == control:
            continue
        t = summary.loc[v]
        row = {"experiment_id": exp_id, "variant": v, "control": control, "n_control": int(c.n), "n_variant": int(t.n)}
        for metric, count_col in (("open", "opens"), ("click", "clicks")):
            z, p, (lo, hi) = two_prop_ztest(c[count_col], c.n, t[count_col], t.n)
            base = c[f"{metric}_rate"]
            row.update({
                f"{metric}_rate_control": base,
                f"{metric}_rate_variant": t[f"{metric}_rate"],
                f"{metric}_lift": (t[f"{metric}_rate"] - base) / base if base else np.nan,
                f"{metric}_lift_ci_low": lo / base if base else np.nan,
                f"{metric}_lift_ci_high": hi / base if base else np.nan,
                f"{metric}_z": z,
                f"{metric}_p": p,
            })
        # CUPED-adjusted click comparison via Welch-style normal approximation
        diff = t.click_rate_cuped - c.click_rate_cuped
        se = np.sqrt(t.click_var_cuped / t.n + c.click_var_cuped / c.n)
        z_adj = diff / se if se > 0 else 0.0
        row["click_lift_cuped"] = diff / c.click_rate if c.click_rate else np.nan
        row["click_p_cuped"] = 2 * (1 - stats.norm.cdf(abs(z_adj)))
        row["variance_reduction"] = 1 - (c.click_var_cuped / (c.click_rate * (1 - c.click_rate))) if 0 < c.click_rate < 1 else np.nan

        significant = row["click_p_cuped"] < ALPHA
        if significant and row["click_lift_cuped"] >= MIN_LIFT_TO_SHIP:
            decision = "ship"
        elif significant and row["click_lift_cuped"] < 0:
            decision = "don't ship (hurts clicks)"
        else:
            decision = "don't ship (inconclusive)"
        row["decision"] = decision
        logger.info(f"{v} vs {control}: click lift {row['click_lift_cuped']:+.2%} (CUPED, p={row['click_p_cuped']:.4f}), "
                    f"open lift {row['open_lift']:+.2%} (p={row['open_p']:.4f}) -> {decision}")
        rows.append(row)

    out = pd.DataFrame(rows)
    out["primary_metric"] = PRIMARY_METRIC
    out["read_at"] = pd.Timestamp.utcnow()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("--end", help="cut-off timestamp, defaults to now")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    engine = create_engine(os.environ["WAREHOUSE_URL"])
    out = build_readout(args.experiment_id, engine, args.end)
    if args.print_only or out.empty:
        print(out.T.to_string())
        return
    schema, table = READOUT_TABLE.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="append", index=False)
    logger.info(f"wrote {len(out)} readout rows to {READOUT_TABLE}")


if __name__ == "__main__":
    main()

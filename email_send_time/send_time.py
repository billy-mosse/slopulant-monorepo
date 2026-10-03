"""Pick the best email send hour per customer.

Beta-Binomial open rates per (customer, hour-of-week bucket), with an empirical-Bayes prior
fit to the global hour-of-week curve, then Thompson sampling so we keep exploring hours.
Customers with too few sends fall back to their segment's best hour.

Reads marketing.email_events, writes marketing.send_times.
"""
import argparse
import logging

import numpy as np
import pandas as pd
import sqlalchemy as sa

log = logging.getLogger("send_time")

EVENTS_TABLE = "marketing.email_events"
OUTPUT_TABLE = "marketing.send_times"
BUCKET_HOURS = 3                     # 7 days * 8 buckets = 56 hour-of-week buckets
N_BUCKETS = 7 * 24 // BUCKET_HOURS
MIN_SENDS_PERSONAL = 8
PRIOR_STRENGTH_CAP = 50.0            # cap on alpha+beta so personal data can move the estimate
N_THOMPSON_DRAWS = 1
LOOKBACK_DAYS = 365
DEFAULT_TZ = "America/New_York"


def run(engine, asof: pd.Timestamp, seed: int, dry_run: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sql = f"""
        SELECT customer_id, campaign_id, sent_ts_utc, opened_ts_utc, customer_tz, segment_name
        FROM {EVENTS_TABLE}
        WHERE event_type = 'send' AND sent_ts_utc >= %(start)s AND sent_ts_utc < %(asof)s
    """
    ev = pd.read_sql(sql, engine, params={"start": asof - pd.Timedelta(days=LOOKBACK_DAYS), "asof": asof},
                     parse_dates=["sent_ts_utc", "opened_ts_utc"])
    log.info(f"loaded {len(ev):,} sends for {ev.customer_id.nunique():,} customers from {EVENTS_TABLE}")

    # --- timezone: convert send time to the customer's local clock ------------------
    ev["customer_tz"] = ev["customer_tz"].fillna(DEFAULT_TZ)
    bad_tz = ~ev["customer_tz"].str.contains("/", regex=False)
    if bad_tz.any():
        log.warning(f"{bad_tz.sum():,} sends with malformed tz, using {DEFAULT_TZ}")
        ev.loc[bad_tz, "customer_tz"] = DEFAULT_TZ
    ev["sent_ts_utc"] = ev["sent_ts_utc"].dt.tz_localize("UTC") if ev["sent_ts_utc"].dt.tz is None else ev["sent_ts_utc"]
    local_parts = []
    for tz, grp in ev.groupby("customer_tz"):
        local = grp["sent_ts_utc"].dt.tz_convert(tz)
        local_parts.append(pd.DataFrame({"dow": local.dt.dayofweek, "hour": local.dt.hour}, index=grp.index))
    ev = ev.join(pd.concat(local_parts))
    ev["bucket"] = ev["dow"] * (24 // BUCKET_HOURS) + ev["hour"] // BUCKET_HOURS
    # an open counts if it happened within 48h of the send (later opens are inbox archaeology)
    ev["opened"] = ((ev["opened_ts_utc"].notna())
                    & ((ev["opened_ts_utc"].dt.tz_localize("UTC") - ev["sent_ts_utc"]) <= pd.Timedelta(hours=48))).astype(int)
    log.info(f"global open rate {ev.opened.mean():.4f} across {ev.bucket.nunique()} buckets used")

    # --- empirical-Bayes prior per bucket from the global curve -----------------------
    # Fit Beta(alpha_b, beta_b) by method of moments on per-customer open rates in each bucket,
    # mean anchored on the global bucket rate.
    cb = ev.groupby(["customer_id", "bucket"]).agg(sends=("opened", "size"), opens=("opened", "sum")).reset_index()
    glob = cb.groupby("bucket").agg(sends=("sends", "sum"), opens=("opens", "sum"))
    glob["mu"] = (glob["opens"] + 1) / (glob["sends"] + 2)
    cb_rate = cb[cb.sends >= 3].assign(rate=lambda d: d.opens / d.sends)
    var_by_bucket = cb_rate.groupby("bucket")["rate"].var()
    mean_sends = cb_rate.groupby("bucket")["sends"].mean()
    prior = glob.join(var_by_bucket.rename("var")).join(mean_sends.rename("n_bar"))
    # observed variance = mu(1-mu)/n_bar + between-customer variance; solve for kappa = alpha+beta
    binom_var = prior["mu"] * (1 - prior["mu"]) / prior["n_bar"].fillna(3)
    between = (prior["var"].fillna(0) - binom_var).clip(lower=1e-5)
    kappa = (prior["mu"] * (1 - prior["mu"]) / between - 1).clip(lower=2.0, upper=PRIOR_STRENGTH_CAP)
    prior["alpha"] = prior["mu"] * kappa
    prior["beta"] = (1 - prior["mu"]) * kappa
    prior = prior.reindex(range(N_BUCKETS))
    prior["alpha"] = prior["alpha"].fillna(prior["alpha"].median())
    prior["beta"] = prior["beta"].fillna(prior["beta"].median())
    log.info(f"prior kappa median {kappa.median():.1f}, best global bucket {glob.mu.idxmax()} (mu={glob.mu.max():.4f})")

    # --- posterior + Thompson sampling per customer -----------------------------------
    customers = ev.groupby("customer_id").agg(
        total_sends=("opened", "size"), total_opens=("opened", "sum"),
        customer_tz=("customer_tz", "last"), segment_name=("segment_name", "last"),
    )
    grid = pd.MultiIndex.from_product([customers.index, range(N_BUCKETS)], names=["customer_id", "bucket"])
    post = cb.set_index(["customer_id", "bucket"]).reindex(grid, fill_value=0).reset_index()
    post["a"] = prior["alpha"].to_numpy()[post["bucket"]] + post["opens"]
    post["b"] = prior["beta"].to_numpy()[post["bucket"]] + post["sends"] - post["opens"]
    post["mean"] = post["a"] / (post["a"] + post["b"])
    post["draw"] = rng.beta(post["a"].to_numpy(), post["b"].to_numpy(), size=(N_THOMPSON_DRAWS, len(post))).mean(axis=0)
    idx = post.groupby("customer_id")["draw"].idxmax()
    picks = post.loc[idx, ["customer_id", "bucket", "mean", "draw"]].set_index("customer_id")
    picks.columns = ["bucket", "posterior_mean", "thompson_draw"]
    customers = customers.join(picks)
    customers["method"] = "thompson"

    # --- segment fallback for thin histories ----------------------------------------
    seg = (post.merge(customers[["segment_name"]], left_on="customer_id", right_index=True)
           .groupby(["segment_name", "bucket"])[["a", "b"]].sum())
    seg["mean"] = seg["a"] / (seg["a"] + seg["b"])
    seg_best = seg.reset_index().loc[lambda d: d.groupby("segment_name")["mean"].idxmax()].set_index("segment_name")
    thin = customers["total_sends"] < MIN_SENDS_PERSONAL
    customers.loc[thin, "bucket"] = customers.loc[thin, "segment_name"].map(seg_best["bucket"]).fillna(glob.mu.idxmax())
    customers.loc[thin, "posterior_mean"] = customers.loc[thin, "segment_name"].map(seg_best["mean"])
    customers.loc[thin, "method"] = "segment_fallback"
    log.info(f"{thin.sum():,} of {len(customers):,} customers ({thin.mean():.1%}) use segment fallback")

    customers["bucket"] = customers["bucket"].astype(int)
    customers["send_dow"] = customers["bucket"] // (24 // BUCKET_HOURS)
    customers["send_hour_local"] = (customers["bucket"] % (24 // BUCKET_HOURS)) * BUCKET_HOURS + BUCKET_HOURS // 2
    out = customers.reset_index()[["customer_id", "customer_tz", "send_dow", "send_hour_local",
                                   "posterior_mean", "method", "total_sends"]]
    out["computed_at"] = asof
    log.info(f"send hour distribution: {out.send_hour_local.value_counts().sort_index().to_dict()}")

    if dry_run:
        log.info(f"dry run: {len(out):,} rows not written")
    else:
        schema, table = OUTPUT_TABLE.split(".")
        out.to_sql(table, engine, schema=schema, if_exists="replace", index=False, chunksize=50_000)
        log.info(f"wrote {len(out):,} rows to {OUTPUT_TABLE}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--asof", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(sa.create_engine(args.dsn), pd.Timestamp(args.asof), args.seed, args.dry_run)


if __name__ == "__main__":
    main()

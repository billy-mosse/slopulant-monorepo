"""Customer-level churn features built directly from order lines and session events.

Reads orders.lines and sessions.events, returns one row per customer_id.
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ORDERS_SQL = """
SELECT customer_id, order_id, order_ts, category_l1, quantity, net_revenue
FROM orders.lines
WHERE order_ts >= %(start)s AND order_ts < %(cutoff)s
  AND status NOT IN ('cancelled', 'fraud')
"""

SESSIONS_SQL = """
SELECT customer_id, session_id, event_ts, event_type
FROM sessions.events
WHERE event_ts >= %(start)s AND event_ts < %(cutoff)s
  AND customer_id IS NOT NULL
"""

CATEGORY_HALF_LIFE_DAYS = 90.0
TOP_CATEGORIES = ["bedding", "bath", "decor", "kitchen", "rugs", "lighting", "furniture"]
LOOKBACK_DAYS = 730


def build_features(conn, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Build the full churn feature matrix as of `cutoff` (exclusive)."""
    start = cutoff - pd.Timedelta(days=LOOKBACK_DAYS)
    params = {"start": start, "cutoff": cutoff}

    log.info(f"loading orders.lines from {start.date()} to {cutoff.date()}")
    lines = pd.read_sql(ORDERS_SQL, conn, params=params, parse_dates=["order_ts"])
    log.info(f"loaded {len(lines):,} order lines for {lines.customer_id.nunique():,} customers")

    # --- RFM ------------------------------------------------------------
    order_df = (
        lines.groupby(["customer_id", "order_id"], as_index=False)
        .agg(order_ts=("order_ts", "min"), order_value=("net_revenue", "sum"), n_items=("quantity", "sum"))
    )
    rfm = order_df.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        first_order_ts=("order_ts", "min"),
        frequency=("order_id", "nunique"),
        monetary_total=("order_value", "sum"),
        monetary_mean=("order_value", "mean"),
        items_mean=("n_items", "mean"),
    )
    rfm["recency_days"] = (cutoff - rfm["last_order_ts"]).dt.total_seconds() / 86400.0
    rfm["tenure_days"] = (cutoff - rfm["first_order_ts"]).dt.total_seconds() / 86400.0
    rfm["log_recency"] = np.log1p(rfm["recency_days"])
    rfm["log_frequency"] = np.log1p(rfm["frequency"])
    rfm["log_monetary_total"] = np.log1p(rfm["monetary_total"].clip(lower=0))
    rfm["log_monetary_mean"] = np.log1p(rfm["monetary_mean"].clip(lower=0))
    rfm["orders_per_month"] = rfm["frequency"] / (rfm["tenure_days"] / 30.0).clip(lower=1.0)

    # inter-purchase gaps: mean gap vs current recency is a strong signal
    order_df = order_df.sort_values(["customer_id", "order_ts"])
    order_df["gap_days"] = order_df.groupby("customer_id")["order_ts"].diff().dt.total_seconds() / 86400.0
    gaps = order_df.groupby("customer_id")["gap_days"].agg(gap_mean="mean", gap_std="std")
    rfm = rfm.join(gaps)
    rfm["recency_over_gap"] = rfm["recency_days"] / rfm["gap_mean"].fillna(rfm["recency_days"]).clip(lower=1.0)
    log.info(f"RFM done: median recency {rfm.recency_days.median():.1f}d, median freq {rfm.frequency.median():.0f}")

    # --- time-decayed category mix ---------------------------------------
    age_days = (cutoff - lines["order_ts"]).dt.total_seconds() / 86400.0
    lines["decay_w"] = np.power(0.5, age_days / CATEGORY_HALF_LIFE_DAYS) * lines["net_revenue"].clip(lower=0)
    lines["cat"] = lines["category_l1"].where(lines["category_l1"].isin(TOP_CATEGORIES), "other")
    cat = lines.groupby(["customer_id", "cat"])["decay_w"].sum().unstack(fill_value=0.0)
    cat_total = cat.sum(axis=1).replace(0, np.nan)
    cat_share = cat.div(cat_total, axis=0).fillna(0.0).add_prefix("cat_share_")
    p = cat_share.to_numpy().clip(1e-12, 1.0)
    cat_share["cat_entropy"] = -(p * np.log(p)).sum(axis=1)
    cat_share["decayed_spend"] = cat_total.fillna(0.0)
    log.info(f"category mix computed over {cat.shape[1]} buckets (half-life {CATEGORY_HALF_LIFE_DAYS:.0f}d)")

    # --- sessions ---------------------------------------------------------
    log.info("loading sessions.events")
    ev = pd.read_sql(SESSIONS_SQL, conn, params=params, parse_dates=["event_ts"])
    sess = ev.groupby(["customer_id", "session_id"], as_index=False).agg(
        session_start=("event_ts", "min"),
        n_events=("event_type", "size"),
        n_add_to_cart=("event_type", lambda s: (s == "add_to_cart").sum()),
        n_product_views=("event_type", lambda s: (s == "product_view").sum()),
    )
    sess["had_atc"] = (sess["n_add_to_cart"] > 0).astype(int)
    sess["days_ago"] = (cutoff - sess["session_start"]).dt.total_seconds() / 86400.0
    sess_feats = sess.groupby("customer_id").agg(
        session_recency_days=("days_ago", "min"),
        sessions_total=("session_id", "nunique"),
        events_per_session=("n_events", "mean"),
        atc_total=("n_add_to_cart", "sum"),
        atc_session_share=("had_atc", "mean"),
        product_views_total=("n_product_views", "sum"),
    )
    # browsing intent: how recently did they look at a product, not just visit
    pv = ev[ev["event_type"] == "product_view"]
    sess_feats["days_since_product_view"] = (
        (cutoff - pv.groupby("customer_id")["event_ts"].max()).dt.total_seconds() / 86400.0
    )
    for window in (7, 30, 90):
        sess_feats[f"sessions_{window}d"] = (
            sess[sess["days_ago"] <= window].groupby("customer_id")["session_id"].nunique()
        )
    sess_feats = sess_feats.fillna({f"sessions_{w}d": 0 for w in (7, 30, 90)})
    sess_feats["visit_freq_trend"] = (sess_feats["sessions_30d"] + 1) / (sess_feats["sessions_90d"] / 3 + 1)
    sess_feats["log_session_recency"] = np.log1p(sess_feats["session_recency_days"])
    sess_feats["product_views_30d"] = (
        sess[sess["days_ago"] <= 30].groupby("customer_id")["n_product_views"].sum()
    ).reindex(sess_feats.index, fill_value=0)
    sess_feats["views_per_session_30d"] = sess_feats["product_views_30d"] / sess_feats["sessions_30d"].clip(lower=1)
    log.info(f"session features for {len(sess_feats):,} customers from {len(ev):,} events")

    feats = rfm.join(cat_share, how="left").join(sess_feats, how="left")
    feats["session_recency_days"] = feats["session_recency_days"].fillna(LOOKBACK_DAYS)
    feats["log_session_recency"] = feats["log_session_recency"].fillna(np.log1p(LOOKBACK_DAYS))
    feats["days_since_product_view"] = feats["days_since_product_view"].fillna(LOOKBACK_DAYS)
    feats = feats.fillna(0.0).drop(columns=["last_order_ts", "first_order_ts"])
    log.info(f"feature matrix: {feats.shape[0]:,} rows x {feats.shape[1]} cols")
    return feats


def build_labels(conn, cutoff: pd.Timestamp, horizon_days: int = 90) -> pd.Index:
    """Customers with at least one order in [cutoff, cutoff + horizon); everyone else churned."""
    end = cutoff + pd.Timedelta(days=horizon_days)
    sql = "SELECT DISTINCT customer_id FROM orders.lines WHERE order_ts >= %(start)s AND order_ts < %(cutoff)s"
    buyers = pd.read_sql(sql, conn, params={"start": cutoff, "cutoff": end})
    log.info(f"{len(buyers):,} customers purchased in the {horizon_days}d label window after {cutoff.date()}")
    return pd.Index(buyers["customer_id"].unique(), name="customer_id")

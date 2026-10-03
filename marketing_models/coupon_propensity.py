"""Coupon redemption propensity.

For each active customer, estimate P(redeem | sent a coupon). Used by the CRM
team to skip customers who would buy anyway at full price and those who never
redeem, and to pick a discount depth. Logistic regression on promo history.

Input:  marketing.promo_history     (customer_id, promo_id, sent_at, discount_pct,
                                     redeemed, last_order_date, ...)
Output: marketing.coupon_propensity (customer_id, discount_pct, p_redeem, scored_at)
"""

import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("coupon_propensity")

WAREHOUSE_URI = "postgresql://analytics@warehouse/slopulent"
HISTORY_SQL = """
    SELECT customer_id, promo_id, sent_at, discount_pct, redeemed, last_order_date
    FROM marketing.promo_history
    WHERE sent_at >= CURRENT_DATE - INTERVAL '540 days'
"""
OUTPUT_TABLE = "coupon_propensity"   # in schema marketing
SCORING_DEPTHS = [10, 15, 20, 25]    # discount levels the CRM team actually uses
HOLDOUT_DAYS = 60

NUMERIC = ["prior_sends", "prior_redemptions", "prior_redeem_rate",
           "discount_pct", "days_since_last_order", "days_since_last_redeem", "avg_redeemed_depth"]


def load_history(engine) -> pd.DataFrame:
    df = pd.read_sql(HISTORY_SQL, engine, parse_dates=["sent_at", "last_order_date"])
    df = df.sort_values(["customer_id", "sent_at"]).reset_index(drop=True)
    log.info("loaded %d coupon sends for %d customers", len(df), df["customer_id"].nunique())
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features: everything uses only sends *before* the current one."""
    g = df.groupby("customer_id")
    df["prior_sends"] = g.cumcount()
    df["prior_redemptions"] = g["redeemed"].cumsum() - df["redeemed"]
    # smoothed rate so a customer with 1/1 doesn't look like a sure thing
    df["prior_redeem_rate"] = (df["prior_redemptions"] + 1) / (df["prior_sends"] + 4)
    df["days_since_last_order"] = (df["sent_at"] - df["last_order_date"]).dt.days.clip(lower=0)

    redeem_ts = df["sent_at"].where(df["redeemed"] == 1)
    df["last_redeem_ts"] = redeem_ts.groupby(df["customer_id"]).shift(1)
    df["last_redeem_ts"] = df.groupby("customer_id")["last_redeem_ts"].ffill()
    df["days_since_last_redeem"] = (df["sent_at"] - df["last_redeem_ts"]).dt.days

    depth = df["discount_pct"].where(df["redeemed"] == 1)
    df["avg_redeemed_depth"] = (depth.groupby(df["customer_id"])
                                .transform(lambda s: s.shift(1).expanding().mean()))
    return df


def build_model() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC),
    ])
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000)
    return Pipeline([("pre", pre), ("clf", clf)])


def time_split(df: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS):
    cutoff = df["sent_at"].max() - pd.Timedelta(days=holdout_days)
    return df[df["sent_at"] < cutoff], df[df["sent_at"] >= cutoff]


def evaluate(model: Pipeline, test: pd.DataFrame) -> None:
    p = model.predict_proba(test[NUMERIC])[:, 1]
    log.info("holdout AUC=%.3f  Brier=%.4f  base rate=%.3f",
             roc_auc_score(test["redeemed"], p), brier_score_loss(test["redeemed"], p), test["redeemed"].mean())
    deciles = pd.qcut(p, 10, labels=False, duplicates="drop")
    print(test.assign(decile=deciles).groupby("decile")["redeemed"].mean().rename("redeem_rate"))


def latest_state(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per customer describing their state 'as if' we sent a coupon today."""
    last = df.groupby("customer_id").tail(1).copy()
    last["prior_sends"] += 1
    last["prior_redemptions"] += last["redeemed"]
    last["prior_redeem_rate"] = (last["prior_redemptions"] + 1) / (last["prior_sends"] + 4)
    last["days_since_last_order"] = (as_of - last["last_order_date"]).dt.days.clip(lower=0)
    last["days_since_last_redeem"] = (as_of - last["last_redeem_ts"]).dt.days
    return last


def score_customers(model: Pipeline, state: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for depth in SCORING_DEPTHS:
        s = state.assign(discount_pct=depth)
        frames.append(pd.DataFrame({
            "customer_id": s["customer_id"].to_numpy(),
            "discount_pct": depth,
            "p_redeem": model.predict_proba(s[NUMERIC])[:, 1],
        }))
    out = pd.concat(frames, ignore_index=True)
    out["scored_at"] = datetime.utcnow()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score coupon redemption propensity")
    parser.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
    parser.add_argument("--skip-write", action="store_true")
    args = parser.parse_args()

    engine = create_engine(WAREHOUSE_URI)
    hist = add_history_features(load_history(engine))
    train, test = time_split(hist, args.holdout_days)

    model = build_model().fit(train[NUMERIC], train["redeemed"])
    evaluate(model, test)

    model.fit(hist[NUMERIC], hist["redeemed"])
    scores = score_customers(model, latest_state(hist, pd.Timestamp.utcnow().tz_localize(None)))
    log.info("scored %d customers x %d depths, mean p=%.3f",
             scores["customer_id"].nunique(), len(SCORING_DEPTHS), np.mean(scores["p_redeem"]))
    if not args.skip_write:
        scores.to_sql(OUTPUT_TABLE, engine, schema="marketing", if_exists="replace", index=False)
        log.info("wrote marketing.coupon_propensity")


if __name__ == "__main__":
    main()

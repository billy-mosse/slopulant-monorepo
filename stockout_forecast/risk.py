"""Probability that each SKU stocks out within the next 4 weeks.

Cumulative demand over the horizon is treated as Normal with mean = sum of the
weekly point forecasts and variance = H * sigma^2 (independent weekly errors).
Available supply = on hand - reserved + inbound POs that land inside the
horizon. P(stockout) = P(D_H > supply) = 1 - Phi((supply - mu_H) / sigma_H).

Business rules applied before scoring:
  * discontinued SKUs are excluded (no replenishment decision to make)
  * negative available stock (system lag) is floored at zero
  * inbound POs only count if their ETA is at least LEAD_BUFFER_DAYS before horizon end
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import create_engine

from forecast_units import WEEKLY_UNITS_SQL, forecast_all

log = logging.getLogger("stockout_risk")


@dataclass(frozen=True)
class RiskPolicy:
    horizon_weeks: int = 4
    lead_buffer_days: int = 3
    high_risk: float = 0.5
    medium_risk: float = 0.2
    min_sigma_units: float = 1.0   # floor so very stable SKUs don't get p=0/1 cliffs


STOCK_SQL = """
SELECT sku, on_hand_units, reserved_units, inbound_units, inbound_eta, is_discontinued
FROM inventory.stock_levels
WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM inventory.stock_levels)
"""
OUTPUT_TABLE = "supply.stockout_risk"


def available_supply(stock: pd.DataFrame, policy: RiskPolicy, today: pd.Timestamp) -> pd.Series:
    cutoff = today + pd.Timedelta(weeks=policy.horizon_weeks) - pd.Timedelta(days=policy.lead_buffer_days)
    inbound_ok = stock["inbound_eta"].notna() & (pd.to_datetime(stock["inbound_eta"]) <= cutoff)
    on_hand = (stock["on_hand_units"] - stock["reserved_units"].fillna(0)).clip(lower=0)
    return on_hand + np.where(inbound_ok, stock["inbound_units"].fillna(0), 0)


def score(forecasts, stock: pd.DataFrame, policy: RiskPolicy, today: pd.Timestamp) -> pd.DataFrame:
    H = policy.horizon_weeks
    fc = pd.DataFrame({
        "sku": [f.sku for f in forecasts],
        "demand_mean_h": [float(f.mean[:H].sum()) for f in forecasts],
        "demand_sd_h": [max(f.resid_sd, policy.min_sigma_units) * np.sqrt(H) for f in forecasts],
        "weekly_forecast": [f.mean[:H].round(2).tolist() for f in forecasts],
        "method": [f.method for f in forecasts],
    })
    stock = stock[~stock["is_discontinued"].fillna(False)].copy()
    stock["supply_units"] = available_supply(stock, policy, today)
    df = stock[["sku", "supply_units"]].merge(fc, on="sku", how="inner")

    z = (df["supply_units"] - df["demand_mean_h"]) / df["demand_sd_h"]
    df["p_stockout_4w"] = norm.sf(z)

    # expected week of stockout: first week where cumulative mean demand exceeds supply
    cum = np.cumsum(np.vstack(df["weekly_forecast"].to_numpy()), axis=1)
    hit = cum > df["supply_units"].to_numpy()[:, None]
    df["expected_stockout_week"] = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, np.nan)
    df["weeks_of_cover"] = df["supply_units"] / np.maximum(df["demand_mean_h"] / H, 1e-6)
    df["risk_band"] = pd.cut(df["p_stockout_4w"], [-0.01, policy.medium_risk, policy.high_risk, 1.0],
                             labels=["low", "medium", "high"]).astype(str)
    df["scored_at"] = today
    return df.drop(columns=["weekly_forecast"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Score 4-week stockout risk per SKU")
    ap.add_argument("--horizon-weeks", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy = RiskPolicy(horizon_weeks=args.horizon_weeks)
    eng = create_engine(os.environ["WAREHOUSE_URL"])
    today = pd.Timestamp.today().normalize()

    forecasts = forecast_all(pd.read_sql(WEEKLY_UNITS_SQL, eng), horizon=max(policy.horizon_weeks, 8))
    stock = pd.read_sql(STOCK_SQL, eng)
    risk = score(forecasts, stock, policy, today)
    log.info("scored %d SKUs: %s", len(risk), risk["risk_band"].value_counts().to_dict())

    if args.dry_run:
        print(risk.sort_values("p_stockout_4w", ascending=False).head(25).to_string(index=False))
        return
    schema, table = OUTPUT_TABLE.split(".")
    risk.to_sql(table, eng, schema=schema, if_exists="replace", index=False)


if __name__ == "__main__":
    main()

"""Feature engineering for the weekly SKU x warehouse demand model.

Builds a weekly panel from order lines and joins planned markdown depth.
All lag/rolling features are shifted so that row t only sees data up to t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 4, 52)
ROLL_WINDOWS = (4, 8, 13)
KEYS = ["sku", "warehouse_id"]

# US retail weeks we treat as promotional peaks (ISO week numbers).
HOLIDAY_WEEKS = {"memorial_day": {21, 22}, "july_4": {27}, "labor_day": {36},
                 "black_friday": {47, 48}, "holiday_season": {50, 51, 52}, "white_sale": {1, 2}}

ORDERS_SQL = """
SELECT sku, warehouse_id, category, DATE_TRUNC('week', order_ts) AS week_start, SUM(quantity) AS units
FROM orders.lines
WHERE order_ts >= %(start)s AND status <> 'CANCELLED'
GROUP BY 1, 2, 3, 4
"""

MARKDOWN_SQL = "SELECT sku, week_start, markdown_pct FROM pricing.markdown_plan WHERE week_start >= %(start)s"


def load_weekly_panel(conn, start: str) -> pd.DataFrame:
    sales = pd.read_sql(ORDERS_SQL, conn, params={"start": start}, parse_dates=["week_start"])
    md = pd.read_sql(MARKDOWN_SQL, conn, params={"start": start}, parse_dates=["week_start"])
    panel = densify(sales).merge(md, on=["sku", "week_start"], how="left")
    panel["markdown_pct"] = panel["markdown_pct"].fillna(0.0).clip(0, 0.8)
    return panel


def densify(sales: pd.DataFrame) -> pd.DataFrame:
    """Fill zero-sales weeks so lags are aligned on calendar weeks, not rows."""
    weeks = pd.date_range(sales["week_start"].min(), sales["week_start"].max(), freq="W-MON")
    combos = sales[KEYS + ["category"]].drop_duplicates()
    grid = combos.merge(pd.DataFrame({"week_start": weeks}), how="cross")
    out = grid.merge(sales, on=KEYS + ["category", "week_start"], how="left")
    out["units"] = out["units"].fillna(0.0)
    return out.sort_values(KEYS + ["week_start"]).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame, target: str = "units") -> pd.DataFrame:
    g = df.groupby(KEYS, sort=False)[target]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.groupby([df[k] for k in KEYS]).transform(lambda s: s.rolling(w, 1).mean())
    df["roll_std_8"] = shifted.groupby([df[k] for k in KEYS]).transform(lambda s: s.rolling(8, min_periods=2).std())
    df["yoy_ratio"] = (df["roll_mean_4"] / (df["lag_52"] + 1.0)).astype(np.float32)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    woy = df["week_start"].dt.isocalendar().week.astype(int)
    df["week_of_year"] = woy
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.0)
    for name, weeks in HOLIDAY_WEEKS.items():
        df[f"hol_{name}"] = woy.isin(weeks).astype(np.int8)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["markdown_depth"] = df["markdown_pct"]
    df["markdown_change"] = df.groupby(KEYS)["markdown_pct"].diff().fillna(0.0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_calendar_features(df)
    df = add_price_features(df)
    for col in ("category", "warehouse_id"):
        df[col] = df[col].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"units", "week_start", "sku", "markdown_pct"}
    return [c for c in df.columns if c not in drop]

"""Global LightGBM demand forecaster: 12-week horizon, P10/P50/P90.

Point model uses a Tweedie objective (lots of zero weeks); quantiles come from
separate quantile-objective models. Multi-step is recursive: each predicted P50
is fed back as `units` so the next week's lags see it.
"""
from __future__ import annotations

import argparse
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from features import KEYS, MARKDOWN_SQL, build_features, feature_columns, load_weekly_panel

log = logging.getLogger("demand_forecast")

HORIZON = 12
QUANTILES = (0.1, 0.5, 0.9)
OUTPUT_TABLE = "supply.demand_forecast"

BASE_PARAMS = dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=50,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)


def train_models(train: pd.DataFrame, num_rounds: int = 800) -> dict[str, lgb.Booster]:
    cols = feature_columns(train)
    train = train.dropna(subset=["lag_4"])
    ds = lgb.Dataset(train[cols], train["units"], categorical_feature=["category", "warehouse_id"])
    models = {"point": lgb.train({**BASE_PARAMS, "objective": "tweedie",
                                  "tweedie_variance_power": 1.3}, ds, num_rounds)}
    for q in QUANTILES:
        params = {**BASE_PARAMS, "objective": "quantile", "alpha": q}
        models[f"p{int(q * 100)}"] = lgb.train(params, ds, num_rounds // 2)
    return models


def recursive_forecast(models: dict[str, lgb.Booster], history: pd.DataFrame,
                       future_markdowns: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Roll forward one week at a time, appending predictions to history."""
    hist = history[KEYS + ["category", "week_start", "units", "markdown_pct"]].copy()
    last_week = hist["week_start"].max()
    outputs = []
    for h in range(1, horizon + 1):
        week = last_week + pd.Timedelta(weeks=h)
        step = hist.loc[hist["week_start"] == hist["week_start"].max(), KEYS + ["category"]].copy()
        step["week_start"] = week
        step["units"] = np.nan
        step = step.merge(future_markdowns, on=["sku", "week_start"], how="left")
        step["markdown_pct"] = step["markdown_pct"].fillna(0.0)
        frame = build_features(pd.concat([hist, step], ignore_index=True))
        X = frame.loc[frame["week_start"] == week, feature_columns(frame)]
        preds = {name: np.clip(m.predict(X), 0, None) for name, m in models.items()}
        step["units"] = preds["point"]
        res = step[KEYS + ["week_start"]].assign(horizon=h, p50=preds["point"])
        res["p10"] = np.minimum(preds["p10"], res["p50"])
        res["p90"] = np.maximum(preds["p90"], res["p50"])
        outputs.append(res)
        hist = pd.concat([hist, step], ignore_index=True)
    return pd.concat(outputs, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--history-start", default="2023-01-02")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(args.dsn)
    panel = load_weekly_panel(engine, args.history_start)
    feats = build_features(panel.copy())
    log.info("training on %d rows, %d features", len(feats), len(feature_columns(feats)))
    models = train_models(feats)

    future_md = pd.read_sql(MARKDOWN_SQL.replace(">=", ">"), engine,
                            params={"start": panel["week_start"].max()}, parse_dates=["week_start"])
    fc = recursive_forecast(models, panel, future_md)
    fc["run_date"] = pd.Timestamp.utcnow().normalize()
    log.info("forecast rows=%d, total p50 units=%.0f", len(fc), fc["p50"].sum())
    if not args.dry_run:
        schema, table = OUTPUT_TABLE.split(".")
        fc.to_sql(table, engine, schema=schema, if_exists="append", index=False)


if __name__ == "__main__":
    main()

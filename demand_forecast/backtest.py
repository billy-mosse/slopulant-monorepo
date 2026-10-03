"""Rolling-origin backtest. Reports WAPE and bias per category and horizon bucket."""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from features import KEYS, build_features, load_weekly_panel
from forecast import HORIZON, recursive_forecast, train_models

log = logging.getLogger("demand_forecast.backtest")


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    d = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / d) if d > 0 else np.nan


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    d = actual.sum()
    return float((pred.sum() - d) / d) if d > 0 else np.nan


def rolling_origins(weeks: pd.Series, n_folds: int, step: int) -> list[pd.Timestamp]:
    uniq = np.sort(weeks.unique())
    last_origin = len(uniq) - HORIZON - 1
    return [pd.Timestamp(uniq[last_origin - i * step]) for i in range(n_folds)][::-1]


def run_fold(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    hist = panel[panel["week_start"] <= origin]
    future = panel[panel["week_start"].between(origin + pd.Timedelta(weeks=1), origin + pd.Timedelta(weeks=HORIZON))]
    models = train_models(build_features(hist.copy()), num_rounds=400)
    fc = recursive_forecast(models, hist, future[["sku", "week_start", "markdown_pct"]].drop_duplicates())
    out = fc.merge(future[KEYS + ["category", "week_start", "units"]], on=KEYS + ["week_start"])
    out["origin"] = origin
    return out


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    results["h_bucket"] = pd.cut(results["horizon"], [0, 4, 8, 12], labels=["1-4", "5-8", "9-12"])
    rows = []
    for (cat, hb), g in results.groupby(["category", "h_bucket"], observed=True):
        a, p = g["units"].to_numpy(), g["p50"].to_numpy()
        cover = ((g["units"] >= g["p10"]) & (g["units"] <= g["p90"])).mean()
        rows.append(dict(category=cat, horizon=hb, wape=wape(a, p), bias=bias(a, p),
                         p10_p90_coverage=cover, n=len(g)))
    return pd.DataFrame(rows).sort_values(["category", "horizon"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--step-weeks", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    panel = load_weekly_panel(create_engine(args.dsn), "2022-01-03")
    results = []
    for origin in rolling_origins(panel["week_start"], args.folds, args.step_weeks):
        fold = run_fold(panel, origin)
        a, p = fold["units"].to_numpy(), fold["p50"].to_numpy()
        log.info("origin %s: WAPE=%.3f bias=%+.3f", origin.date(), wape(a, p), bias(a, p))
        results.append(fold)
    print(summarize(pd.concat(results, ignore_index=True)).to_string(index=False))


if __name__ == "__main__":
    main()

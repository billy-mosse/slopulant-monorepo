"""Weekly unit forecasts per SKU using additive Holt-Winters (season length 52).

Level / trend / season recursions (additive):

    l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
    b_t = beta  * (l_t - l_{t-1}) + (1 - beta)  * b_{t-1}
    s_t = gamma * (y_t - l_t)     + (1 - gamma) * s_{t-m}
    yhat_{t+h} = l_t + h * b_t + s_{t+h-m(k+1)}

Smoothing parameters are picked per SKU from a small grid by one-step-ahead
SSE. SKUs with less than two full seasons fall back to damped level-only
smoothing because seasonal indices cannot be initialised reliably.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SEASON = 52
HISTORY_WEEKS = 156
ALPHAS = (0.05, 0.1, 0.2, 0.35)
BETAS = (0.0, 0.02, 0.05)
GAMMAS = (0.05, 0.15, 0.3)
FALLBACK_ALPHA = 0.25

WEEKLY_UNITS_SQL = f"""
SELECT sku, DATE_TRUNC('week', order_ts)::date AS week, SUM(quantity) AS units
FROM orders.lines
WHERE order_ts >= CURRENT_DATE - INTERVAL '{HISTORY_WEEKS} weeks'
  AND order_status NOT IN ('cancelled', 'fraud')
GROUP BY 1, 2
"""


@dataclass(frozen=True)
class SkuForecast:
    sku: str
    mean: np.ndarray       # point forecast for weeks 1..H
    resid_sd: float        # in-sample one-step residual std dev
    method: str
    params: tuple[float, float, float]


def _hw_pass(y: np.ndarray, a: float, b: float, g: float) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    m = SEASON
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m
    season = y[:m] - level
    resid = np.empty(len(y) - m)
    for t in range(m, len(y)):
        s_prev = season[t % m]
        fc = level + trend + s_prev
        resid[t - m] = y[t] - fc
        new_level = a * (y[t] - s_prev) + (1 - a) * (level + trend)
        trend = b * (new_level - level) + (1 - b) * trend
        level = new_level
        season[t % m] = g * (y[t] - level) + (1 - g) * s_prev
    sse = float(resid @ resid)
    return sse, level, trend, season, resid


def holt_winters(sku: str, y: np.ndarray, horizon: int) -> SkuForecast:
    if len(y) < 2 * SEASON:
        return _level_only(sku, y, horizon)
    best = None
    for a, b, g in itertools.product(ALPHAS, BETAS, GAMMAS):
        out = _hw_pass(y, a, b, g)
        if best is None or out[0] < best[0][0]:
            best = (out, (a, b, g))
    (sse, level, trend, season, resid), params = best
    n = len(y)
    steps = np.arange(1, horizon + 1)
    mean = level + steps * trend + season[(n + steps - 1) % SEASON]
    return SkuForecast(sku, np.clip(mean, 0, None), float(resid[-SEASON:].std(ddof=1)), "holt_winters_add", params)


def _level_only(sku: str, y: np.ndarray, horizon: int) -> SkuForecast:
    level, resid = y[0], []
    for v in y[1:]:
        resid.append(v - level)
        level = FALLBACK_ALPHA * v + (1 - FALLBACK_ALPHA) * level
    sd = float(np.std(resid, ddof=1)) if len(resid) > 2 else float(max(level, 1.0))
    return SkuForecast(sku, np.full(horizon, max(level, 0.0)), sd, "ses_fallback", (FALLBACK_ALPHA, 0.0, 0.0))


def to_dense(weekly: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pivot to a complete week grid per SKU; missing weeks are zero sales."""
    weekly["week"] = pd.to_datetime(weekly["week"])
    grid = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq="W-MON")
    wide = weekly.pivot_table(index="week", columns="sku", values="units", aggfunc="sum").reindex(grid, fill_value=0).fillna(0)
    series = {}
    for sku in wide.columns:
        y = wide[sku].to_numpy(float)
        first = np.argmax(y > 0)  # drop leading zeros before launch
        series[sku] = y[first:]
    return series


def forecast_all(weekly: pd.DataFrame, horizon: int) -> list[SkuForecast]:
    series = to_dense(weekly)
    out = [holt_winters(sku, y, horizon) for sku, y in series.items() if y.sum() > 0]
    n_hw = sum(f.method == "holt_winters_add" for f in out)
    log.info("forecast %d SKUs (%d seasonal, %d fallback)", len(out), n_hw, len(out) - n_hw)
    return out

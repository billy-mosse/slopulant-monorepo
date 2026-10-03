"""CUPED variance reduction.

Y_adj = Y - theta * (X - mean(X)),  theta = cov(X, Y) / var(X)

X is the same metric measured over the pre-experiment period. theta is pooled
across arms so the adjustment does not bias the treatment effect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_PRE_COVERAGE = 0.5  # skip CUPED if fewer than half of units have a pre-period value


@dataclass(frozen=True)
class CupedResult:
    theta: float
    variance_reduction: float
    applied: bool


def estimate_theta(pre: np.ndarray, post: np.ndarray) -> float:
    var_x = np.var(pre, ddof=1)
    if var_x == 0:
        return 0.0
    return float(np.cov(pre, post, ddof=1)[0, 1] / var_x)


def adjust(df: pd.DataFrame, value_col: str = "value", pre_col: str = "pre_value") -> tuple[pd.Series, CupedResult]:
    """Return CUPED-adjusted metric and diagnostics. Missing pre values are imputed with the pooled mean."""
    y = df[value_col].to_numpy(dtype=float)
    coverage = df[pre_col].notna().mean() if len(df) else 0.0
    if coverage < MIN_PRE_COVERAGE:
        return df[value_col].astype(float), CupedResult(0.0, 0.0, applied=False)
    x = df[pre_col].fillna(df[pre_col].mean()).to_numpy(dtype=float)
    theta = estimate_theta(x, y)
    y_adj = y - theta * (x - x.mean())
    var_y = np.var(y, ddof=1)
    reduction = 1.0 - np.var(y_adj, ddof=1) / var_y if var_y > 0 else 0.0
    return pd.Series(y_adj, index=df.index, name=value_col), CupedResult(theta, float(reduction), applied=True)

"""Own-price elasticity per category.

Model (per category c, pooled over its SKUs i and weeks t):

    log(units_it) = beta_c * log(price_it) + sum_w gamma_w * 1[woy_t = w] + alpha_i + e_it

SKU fixed effects alpha_i are removed by within-SKU demeaning; week-of-year
dummies absorb seasonality. Standard errors are cluster-robust by SKU.
Noisy categories are shrunk toward the precision-weighted global elasticity
(empirical Bayes, normal-normal):

    beta_c_shrunk = w_c * beta_c + (1 - w_c) * beta_global,
    w_c = tau^2 / (tau^2 + se_c^2)

Input:  weekly panel built by sql/weekly_sales.sql (orders.lines x pricing.price_history)
Output: pricing.elasticities
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

log = logging.getLogger("elasticity")

SQL_PATH = Path(__file__).parent / "sql" / "weekly_sales.sql"
OUTPUT_TABLE = "pricing.elasticities"
MIN_SKUS_PER_CATEGORY = 3
MIN_OBS_PER_CATEGORY = 60


@dataclass
class CategoryFit:
    category: str
    beta: float
    se: float
    n_obs: int
    n_skus: int
    r2_within: float


def load_panel(engine, lookback_weeks: int) -> pd.DataFrame:
    query = SQL_PATH.read_text()
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"lookback_weeks": lookback_weeks})
    df = df[(df["units"] > 0) & (df["avg_price"] > 0)].copy()
    df["log_q"] = np.log(df["units"])
    df["log_p"] = np.log(df["avg_price"])
    return df


def _demean(df: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    return df[cols] - df.groupby(by)[cols].transform("mean")


def fit_category(df: pd.DataFrame, category: str) -> CategoryFit | None:
    if df["sku"].nunique() < MIN_SKUS_PER_CATEGORY or len(df) < MIN_OBS_PER_CATEGORY:
        return None
    # Week-of-year dummies (drop one level as reference).
    woy = pd.get_dummies(df["week_of_year"].astype(int), prefix="w", drop_first=True, dtype=float)
    X_raw = pd.concat([df[["log_p"]], woy], axis=1)
    X_raw["sku"] = df["sku"].values
    X_raw["log_q"] = df["log_q"].values
    cols = [c for c in X_raw.columns if c not in ("sku",)]
    within = _demean(X_raw, cols, by="sku")

    y = within["log_q"].to_numpy()
    X = within.drop(columns="log_q").to_numpy()
    # Drop dummy columns that are identically zero after demeaning.
    keep = np.abs(X).sum(axis=0) > 1e-12
    X = X[:, keep]
    if not keep[0]:
        log.warning("category %s: no within-SKU price variation", category)
        return None

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    # Cluster-robust (by SKU) sandwich: (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1
    meat = np.zeros((X.shape[1], X.shape[1]))
    skus = df["sku"].to_numpy()
    for g in np.unique(skus):
        idx = skus == g
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    G, n, k = len(np.unique(skus)), len(y), X.shape[1]
    # Small-sample correction; k includes absorbed FE count implicitly via G.
    c = (G / (G - 1)) * ((n - 1) / max(n - k - G, 1))
    vcov = c * XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(vcov[0, 0], 0.0)))

    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else 0.0
    return CategoryFit(category, float(beta[0]), se, n, G, float(r2))


def shrink(fits: list[CategoryFit]) -> pd.DataFrame:
    out = pd.DataFrame([f.__dict__ for f in fits])
    w_prec = 1.0 / out["se"].clip(lower=1e-6) ** 2
    beta_global = float(np.sum(w_prec * out["beta"]) / np.sum(w_prec))
    # Method-of-moments (DerSimonian-Laird) estimate of between-category variance.
    q = float(np.sum(w_prec * (out["beta"] - beta_global) ** 2))
    denom = float(w_prec.sum() - (w_prec**2).sum() / w_prec.sum())
    tau2 = max(0.0, (q - (len(out) - 1)) / denom) if denom > 0 else 0.0
    log.info("global elasticity %.3f, tau^2 %.4f, Q %.1f", beta_global, tau2, q)

    out["shrink_weight"] = tau2 / (tau2 + out["se"] ** 2) if tau2 > 0 else 0.0
    out["elasticity"] = out["shrink_weight"] * out["beta"] + (1 - out["shrink_weight"]) * beta_global
    out["t_stat"] = out["beta"] / out["se"]
    out["p_value"] = 2 * stats.t.sf(np.abs(out["t_stat"]), df=(out["n_skus"] - 1).clip(lower=1))
    out["ci_low"] = out["beta"] - 1.96 * out["se"]
    out["ci_high"] = out["beta"] + 1.96 * out["se"]
    out["global_elasticity"] = beta_global
    return out.rename(columns={"beta": "raw_elasticity"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--lookback-weeks", type=int, default=104)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(args.dsn)
    panel = load_panel(engine, args.lookback_weeks)
    log.info("panel: %d rows, %d skus, %d categories",
             len(panel), panel["sku"].nunique(), panel["category"].nunique())

    fits = []
    for cat, grp in panel.groupby("category"):
        fit = fit_category(grp.reset_index(drop=True), str(cat))
        if fit is None:
            log.info("skip %s (insufficient data)", cat)
            continue
        fits.append(fit)
    if not fits:
        raise SystemExit("no category could be estimated")

    result = shrink(fits)
    result["run_date"] = pd.Timestamp.utcnow().normalize()
    pos = (result["elasticity"] > 0).sum()
    if pos:
        log.warning("%d categories with positive elasticity after shrinkage", pos)

    if args.dry_run:
        print(result.sort_values("elasticity").to_string(index=False))
        return
    schema, table = OUTPUT_TABLE.split(".")
    result.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    log.info("wrote %d rows to %s", len(result), OUTPUT_TABLE)


if __name__ == "__main__":
    main()

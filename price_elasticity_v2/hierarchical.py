"""
Partially pooled price elasticities.

Model (per SKU i in category c, week t):

    log q_it = a_i + b_i * log p_it + e_it,     e_it ~ N(0, s2_i)
    b_i      ~ N(mu_c, tau2_c)                  (category prior on elasticity)
    a_i      flat

Sampling scheme (Gibbs):
    b_i | rest   ~ N(m_i, v_i)  with  v_i = 1 / (Sxx_i / s2_i + 1 / tau2_c)
                                     m_i = v_i * (Sxy_i / s2_i + mu_c / tau2_c)
    a_i | rest   ~ N(ybar_i - b_i * xbar_i, s2_i / n_i)
    s2_i | rest  ~ InvGamma(A0 + n_i/2, B0 + RSS_i/2)
    mu_c | rest  ~ N(mean(b_i in c), tau2_c / n_c)       (flat hyperprior)
    tau2_c | rest~ InvGamma(G0 + n_c/2, H0 + sum (b_i - mu_c)^2 / 2)

Sxx_i, Sxy_i are centred sums so a_i and b_i decouple cleanly.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger(__name__)

WEEKLY_SQL = """
WITH units AS (
    SELECT sku, category_id, DATE_TRUNC('week', order_ts) AS week, SUM(quantity) AS units
    FROM orders.lines
    WHERE order_ts >= CURRENT_DATE - INTERVAL '104 weeks'
    GROUP BY 1, 2, 3
), prices AS (
    SELECT sku, DATE_TRUNC('week', effective_from) AS week, AVG(unit_price) AS price
    FROM pricing.price_history
    GROUP BY 1, 2
)
SELECT u.sku, u.category_id, u.week, u.units, p.price
FROM units u JOIN prices p USING (sku, week)
WHERE u.units > 0 AND p.price > 0
"""
TARGET = "pricing.elasticity_posteriors"

A0, B0 = 2.0, 0.5           # InvGamma prior on residual variance
G0, H0 = 3.0, 0.4           # InvGamma prior on category spread tau2
MIN_WEEKS = 8
MIN_PRICE_POINTS = 3


def sufficient_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(x=np.log(df["price"]), y=np.log(df["units"]))
    g = df.groupby(["sku", "category_id"])
    st = g.agg(n=("x", "size"), xbar=("x", "mean"), ybar=("y", "mean"),
               n_px=("price", lambda s: s.round(2).nunique())).reset_index()
    df = df.merge(st[["sku", "xbar", "ybar"]], on="sku")
    df["dx"], df["dy"] = df["x"] - df["xbar"], df["y"] - df["ybar"]
    sums = df.assign(xx=df.dx**2, xy=df.dx * df.dy, yy=df.dy**2).groupby("sku")[["xx", "xy", "yy"]].sum()
    st = st.merge(sums, left_on="sku", right_index=True)
    keep = (st.n >= MIN_WEEKS) & (st.n_px >= MIN_PRICE_POINTS) & (st.xx > 1e-6)
    log.info("kept %d / %d SKUs with enough price variation", keep.sum(), len(st))
    return st[keep].reset_index(drop=True)


class PooledElasticityGibbs:
    def __init__(self, n_iter: int = 3000, burn: int = 1000, thin: int = 2, seed: int = 7):
        self.n_iter, self.burn, self.thin = n_iter, burn, thin
        self.rng = np.random.default_rng(seed)

    def _inv_gamma(self, shape, rate):
        return rate / self.rng.gamma(shape, 1.0, size=np.shape(shape))

    def fit(self, st: pd.DataFrame) -> dict[str, np.ndarray]:
        cat_codes, cats = pd.factorize(st["category_id"])
        n, Sxx, Sxy, Syy = (st[k].to_numpy(float) for k in ("n", "xx", "xy", "yy"))
        C, N = len(cats), len(st)
        n_c = np.bincount(cat_codes, minlength=C)

        # OLS start
        b = Sxy / Sxx
        s2 = np.maximum((Syy - b * Sxy) / np.maximum(n - 2, 1), 1e-3)
        mu = np.bincount(cat_codes, weights=b, minlength=C) / n_c
        tau2 = np.full(C, 0.5)

        keep = (self.n_iter - self.burn) // self.thin
        draws_b = np.empty((keep, N))
        draws_mu = np.empty((keep, C))
        k = 0
        for it in range(self.n_iter):
            m_c, t_c = mu[cat_codes], tau2[cat_codes]
            v = 1.0 / (Sxx / s2 + 1.0 / t_c)
            m = v * (Sxy / s2 + m_c / t_c)
            b = m + np.sqrt(v) * self.rng.standard_normal(N)

            # RSS with a_i integrated at its conditional mean: Syy - 2b Sxy + b^2 Sxx
            rss = np.maximum(Syy - 2 * b * Sxy + b * b * Sxx, 1e-9)
            s2 = self._inv_gamma(A0 + n / 2, B0 + rss / 2)

            bsum = np.bincount(cat_codes, weights=b, minlength=C)
            mu = bsum / n_c + np.sqrt(tau2 / n_c) * self.rng.standard_normal(C)
            dev = np.bincount(cat_codes, weights=(b - mu[cat_codes]) ** 2, minlength=C)
            tau2 = self._inv_gamma(G0 + n_c / 2, H0 + dev / 2)

            if it >= self.burn and (it - self.burn) % self.thin == 0:
                draws_b[k], draws_mu[k] = b, mu
                k += 1
            if it % 500 == 0:
                log.debug("iter %d  mean b=%.3f  mean tau2=%.3f", it, b.mean(), tau2.mean())
        return {"b": draws_b, "mu": draws_mu, "cats": np.asarray(cats)}


def summarize(st: pd.DataFrame, draws: dict[str, np.ndarray]) -> pd.DataFrame:
    b = draws["b"]
    q = np.quantile(b, [0.05, 0.5, 0.95], axis=0)
    ols = st["xy"] / st["xx"]
    mean = b.mean(0)
    out = pd.DataFrame({
        "sku": st["sku"], "category_id": st["category_id"], "n_weeks": st["n"],
        "elasticity_mean": mean, "elasticity_sd": b.std(0),
        "elasticity_p05": q[0], "elasticity_p50": q[1], "elasticity_p95": q[2],
        "prob_inelastic": (b > -1.0).mean(0),
        "ols_elasticity": ols,
        # how far the posterior moved from the raw OLS toward the category mean
        "shrinkage": 1 - (mean - draws["mu"].mean(0)[pd.factorize(st["category_id"])[0]])
                     / (ols - draws["mu"].mean(0)[pd.factorize(st["category_id"])[0]]).replace(0, np.nan),
    })
    out["fitted_at"] = pd.Timestamp.utcnow()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    eng = create_engine(os.environ["WAREHOUSE_URL"])
    weekly = pd.read_sql(WEEKLY_SQL, eng)
    st = sufficient_stats(weekly)
    draws = PooledElasticityGibbs(n_iter=a.iters, burn=a.burn).fit(st)
    res = summarize(st, draws)
    log.info("median posterior elasticity %.2f; %.1f%% SKUs likely inelastic",
             res.elasticity_p50.median(), 100 * (res.prob_inelastic > 0.5).mean())
    if a.no_write:
        print(res.describe().T)
        return
    sch, tbl = TARGET.split(".")
    res.to_sql(tbl, eng, schema=sch, if_exists="replace", index=False)


if __name__ == "__main__":
    main()

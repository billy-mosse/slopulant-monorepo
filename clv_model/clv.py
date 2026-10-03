"""12-month CLV: BG/NBD (Fader, Hardie & Lee 2005) x Gamma-Gamma spend model.

Input:  orders.lines
Output: scores.clv  (customer_id, frequency, recency, T, monetary, p_alive, exp_purchases_12m, exp_aov, clv_12m)
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

SRC = "orders.lines"
DST = "scores.clv"
HORIZON_WEEKS = 52.0
DISCOUNT_ANNUAL = 0.10


def rfm_summary(lines: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """x = repeat purchases, t_x = time of last purchase, T = age; all in weeks from first purchase."""
    d = (lines.assign(day=lines.order_ts.dt.normalize())
         .groupby(["customer_id", "day"], as_index=False).net_revenue.sum())
    g = d.groupby("customer_id")
    first, last = g.day.min(), g.day.max()
    s = pd.DataFrame({
        "frequency": g.day.count() - 1,
        "recency": (last - first).dt.days / 7.0,
        "T": (asof - first).dt.days / 7.0,
    })
    # monetary value over repeat transactions only (Gamma-Gamma convention)
    rep = d[d.day > d.customer_id.map(first)]
    s["monetary"] = rep.groupby("customer_id").net_revenue.mean().reindex(s.index).fillna(0.0)
    return s


# ---- BG/NBD ---------------------------------------------------------------
# L(r,a,a,b | x,t_x,T) = B(a,b+x)/B(a,b) * Γ(r+x)/Γ(r) * α^r *
#   [ (α+T)^-(r+x) + 1{x>0} * a/(b+x-1) * (α+t_x)^-(r+x) ]

def bgnbd_ll(p: np.ndarray, x, tx, T) -> float:
    r, alpha, a, b = np.exp(p)
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    A2 = betaln(a, b + x) - betaln(a, b)
    A3 = -(r + x) * np.log(alpha + T)
    A4 = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-12)) - (r + x) * np.log(alpha + tx), -np.inf)
    return float(np.sum(A1 + A2 + np.logaddexp(A3, A4)))


def fit_bgnbd(s: pd.DataFrame, penalizer: float = 1e-3) -> np.ndarray:
    x, tx, T = (s[c].to_numpy(float) for c in ("frequency", "recency", "T"))
    f = lambda p: -bgnbd_ll(p, x, tx, T) / len(x) + penalizer * np.sum(np.exp(p) ** 2)
    res = minimize(f, np.zeros(4), method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
    return np.exp(res.x)


def bgnbd_expected(params, t, x, tx, T) -> np.ndarray:
    """E[Y(t) | x, t_x, T]: expected purchases in (T, T+t]."""
    r, alpha, a, b = params
    z = t / (alpha + T + t)
    h = hyp2f1(r + x, b + x, a + b + x - 1, z)
    num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * h)
    den = 1 + (x > 0) * a / (b + x - 1) * ((alpha + T) / (alpha + tx)) ** (r + x)
    return num / den


def p_alive(params, x, tx, T) -> np.ndarray:
    r, alpha, a, b = params
    odds = (x > 0) * a / (b + x - 1) * ((alpha + T) / (alpha + tx)) ** (r + x)
    return 1.0 / (1.0 + odds)


# ---- Gamma-Gamma ------------------------------------------------------------
# z_bar | x ~ spend; v ~ Gamma(q, γ), z_i | v ~ Gamma(p, v)

def gg_ll(lp: np.ndarray, x, m) -> float:
    p, q, g = np.exp(lp)
    return float(np.sum(
        gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(g)
        + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(g + m * x)
    ))


def fit_gamma_gamma(s: pd.DataFrame) -> np.ndarray:
    r = s[(s.frequency > 0) & (s.monetary > 0)]
    x, m = r.frequency.to_numpy(float), r.monetary.to_numpy(float)
    res = minimize(lambda lp: -gg_ll(lp, x, m) / len(x), np.log([1.0, 1.0, m.mean()]), method="L-BFGS-B")
    return np.exp(res.x)


def gg_expected_aov(params, x, m) -> np.ndarray:
    p, q, g = params
    pop = p * g / (q - 1)
    w = (q - 1) / (p * x + q - 1)
    return w * pop + (1 - w) * np.where(x > 0, m, pop)


def clv(s: pd.DataFrame, bg, gg, weeks: float = HORIZON_WEEKS) -> pd.DataFrame:
    x, tx, T, m = (s[c].to_numpy(float) for c in ("frequency", "recency", "T", "monetary"))
    # discounted: integrate monthly increments of E[Y(t)]
    months = np.arange(1, int(round(weeks / 4.345)) + 1)
    cum = np.stack([bgnbd_expected(bg, k * 4.345, x, tx, T) for k in np.r_[0, months]])
    disc = (1 + DISCOUNT_ANNUAL) ** (-months / 12.0)
    aov = gg_expected_aov(gg, x, m)
    out = s.copy()
    out["p_alive"] = p_alive(bg, x, tx, T)
    out["exp_purchases_12m"] = cum[-1]
    out["exp_aov"] = aov
    out["clv_12m"] = aov * (np.diff(cum, axis=0) * disc[:, None]).sum(0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--asof", required=True)
    a = ap.parse_args()
    import sqlalchemy as sa
    eng, asof = sa.create_engine(a.dsn), pd.Timestamp(a.asof)
    lines = pd.read_sql(
        f"SELECT customer_id, order_ts, net_revenue FROM {SRC} WHERE order_ts < %(asof)s AND status = 'completed'",
        eng, params={"asof": asof}, parse_dates=["order_ts"])
    s = rfm_summary(lines, asof)
    bg = fit_bgnbd(s)
    gg = fit_gamma_gamma(s)
    print(f"BG/NBD r,α,a,b = {np.round(bg, 4)}; Gamma-Gamma p,q,γ = {np.round(gg, 4)}")
    out = clv(s, bg, gg).reset_index().assign(asof_date=asof.date())
    sch, tbl = DST.split(".")
    out.to_sql(tbl, eng, schema=sch, if_exists="append", index=False)
    print(f"{len(out)} rows -> {DST}; total 12m CLV {out.clv_12m.sum():,.0f}")


if __name__ == "__main__":
    main()

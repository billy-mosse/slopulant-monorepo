"""
Weekly input/score drift report for the churn model.

Compares the most recent week of churn-model inputs and predicted
probabilities to the window the model was trained on:

  * numeric features  -> Population Stability Index on reference deciles
  * categorical       -> Jensen-Shannon distance between level frequencies
  * p_churn scores    -> PSI on score deciles + KS statistic + mean shift

Results are appended to monitoring.churn_drift and rendered as an HTML
table that the churn team posts in the weekly model review.

Thresholds (team convention): PSI < 0.10 stable, 0.10-0.25 watch, > 0.25 alert.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sqlalchemy import create_engine

logger = logging.getLogger("churn_drift")

# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class DriftConfig:
    train_start: date = date(2026, 1, 1)
    train_end: date = date(2026, 3, 31)
    numeric: tuple[str, ...] = (
        "days_since_last_order", "orders_90d", "revenue_90d", "sessions_30d",
        "avg_basket_value", "returns_rate_180d", "email_open_rate_30d",
    )
    categorical: tuple[str, ...] = ("loyalty_tier", "acq_channel", "preferred_category")
    n_bins: int = 10
    psi_watch: float = 0.10
    psi_alert: float = 0.25
    js_alert: float = 0.10
    eps: float = 1e-4
    sample_rows: int = 400_000


FEATURE_SQL = Template("""
SELECT customer_id, snapshot_date, $cols
FROM features.customer_daily
WHERE snapshot_date BETWEEN '$start' AND '$end'
ORDER BY RANDOM() LIMIT $limit
""")
SCORE_SQL = Template("""
SELECT customer_id, score_date, p_churn
FROM scores.p_churn
WHERE score_date BETWEEN '$start' AND '$end'
""")
OUTPUT_TABLE = "monitoring.churn_drift"

# ---------------------------------------------------------------- metrics


def psi(ref: np.ndarray, cur: np.ndarray, n_bins: int, eps: float) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:  # near-constant feature, deciles collapse
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / ref.size
    q = np.histogram(cur, edges)[0] / cur.size
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def js_distance(ref: pd.Series, cur: pd.Series) -> float:
    levels = sorted(set(ref.dropna()) | set(cur.dropna()))
    p = ref.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    q = cur.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    return float(jensenshannon(p, q, base=2))


def status(value: float, watch: float, alert: float) -> str:
    if np.isnan(value):
        return "no_data"
    return "alert" if value > alert else "watch" if value > watch else "stable"


# ---------------------------------------------------------------- report


@dataclass
class DriftRow:
    item: str
    kind: str
    metric: str
    value: float
    status: str
    extra: dict = field(default_factory=dict)


def compute(ref_f: pd.DataFrame, cur_f: pd.DataFrame, ref_s: pd.Series, cur_s: pd.Series,
            cfg: DriftConfig) -> list[DriftRow]:
    rows: list[DriftRow] = []
    for col in cfg.numeric:
        v = psi(ref_f[col].to_numpy(float), cur_f[col].to_numpy(float), cfg.n_bins, cfg.eps)
        rows.append(DriftRow(col, "numeric", "psi", v, status(v, cfg.psi_watch, cfg.psi_alert),
                             {"ref_mean": ref_f[col].mean(), "cur_mean": cur_f[col].mean()}))
    for col in cfg.categorical:
        v = js_distance(ref_f[col].astype(str), cur_f[col].astype(str))
        rows.append(DriftRow(col, "categorical", "js_distance", v, status(v, cfg.js_alert / 2, cfg.js_alert)))

    s_psi = psi(ref_s.to_numpy(float), cur_s.to_numpy(float), cfg.n_bins, cfg.eps)
    ks = ks_2samp(ref_s, cur_s)
    rows.append(DriftRow("p_churn", "score", "psi", s_psi, status(s_psi, cfg.psi_watch, cfg.psi_alert),
                         {"ks_stat": ks.statistic, "ks_p": ks.pvalue,
                          "mean_shift": cur_s.mean() - ref_s.mean()}))
    return rows


HTML = Template("""<html><head><style>
td,th{padding:4px 10px;font-family:sans-serif;font-size:13px}
.alert{background:#f8d7da}.watch{background:#fff3cd}.stable{background:#d1e7dd}
</style></head><body><h3>Churn model drift - week of $week</h3>
<table><tr><th>item</th><th>kind</th><th>metric</th><th>value</th><th>status</th></tr>$rows</table>
</body></html>""")


def render_html(rows: list[DriftRow], week: date) -> str:
    body = "".join(
        f"<tr class='{r.status}'><td>{r.item}</td><td>{r.kind}</td><td>{r.metric}</td>"
        f"<td>{r.value:.4f}</td><td>{r.status}</td></tr>" for r in rows
    )
    return HTML.substitute(week=week.isoformat(), rows=body)


# ---------------------------------------------------------------- entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Churn model drift report")
    ap.add_argument("--week-end", type=date.fromisoformat, default=date.today())
    ap.add_argument("--html-out", type=Path, default=Path("churn_drift.html"))
    ap.add_argument("--skip-write", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = DriftConfig()
    eng = create_engine(os.environ["WAREHOUSE_URL"])
    cur_start = args.week_end - timedelta(days=6)
    cols = ", ".join(cfg.numeric + cfg.categorical)

    def features(s: date, e: date) -> pd.DataFrame:
        return pd.read_sql(FEATURE_SQL.substitute(cols=cols, start=s, end=e, limit=cfg.sample_rows), eng)

    def scores(s: date, e: date) -> pd.Series:
        return pd.read_sql(SCORE_SQL.substitute(start=s, end=e), eng)["p_churn"]

    rows = compute(features(cfg.train_start, cfg.train_end), features(cur_start, args.week_end),
                   scores(cfg.train_start, cfg.train_end), scores(cur_start, args.week_end), cfg)

    n_alert = sum(r.status == "alert" for r in rows)
    logger.info("%d items checked, %d alerts", len(rows), n_alert)
    args.html_out.write_text(render_html(rows, args.week_end))

    if not args.skip_write:
        out = pd.DataFrame([{**r.__dict__, **{"extra": str(r.extra)}} for r in rows])
        out["week_end"] = args.week_end
        schema, table = OUTPUT_TABLE.split(".")
        out.to_sql(table, eng, schema=schema, if_exists="append", index=False)


if __name__ == "__main__":
    main()

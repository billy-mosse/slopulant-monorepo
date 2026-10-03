"""Production model drift monitoring.

Compares a current window against a reference window for every monitored model:
  - PSI per feature, quantile bins fit on the reference window
  - two-sample Kolmogorov-Smirnov for continuous features
  - shift in the prediction score distribution
  - jumps in missing-value rate
Findings are passed to alerts.py, which thresholds, dedups and writes them.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from alerts import AlertSink, DriftFinding

log = logging.getLogger("model_monitoring.drift")

PREDICTIONS_TABLE = "monitoring.prediction_logs"
FEATURES_TABLE = "features.customer_daily"
N_BINS = 10
EPS = 1e-6
REFERENCE_DAYS = 28
CURRENT_DAYS = 1


@dataclass(frozen=True)
class Window:
    start: date
    end: date  # exclusive


def quantile_edges(reference: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = N_BINS) -> float:
    """PSI = sum_i (c_i - r_i) * ln(c_i / r_i) over reference-quantile bins."""
    edges = quantile_edges(reference, n_bins)
    r = np.histogram(reference, edges)[0] / max(len(reference), 1)
    c = np.histogram(current, edges)[0] / max(len(current), 1)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def is_continuous(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s) or (pd.api.types.is_integer_dtype(s) and s.nunique() > 20)


def feature_findings(model: str, ref: pd.DataFrame, cur: pd.DataFrame, features: list[str]) -> list[DriftFinding]:
    out: list[DriftFinding] = []
    for f in features:
        r, c = ref[f], cur[f]
        null_jump = float(c.isna().mean() - r.isna().mean())
        out.append(DriftFinding(model, f, "null_rate_delta", null_jump))
        r, c = r.dropna().to_numpy(dtype=float), c.dropna().to_numpy(dtype=float)
        if len(r) == 0 or len(c) == 0:
            continue
        out.append(DriftFinding(model, f, "psi", psi(r, c)))
        if is_continuous(ref[f]):
            ks = stats.ks_2samp(r, c)
            out.append(DriftFinding(model, f, "ks_pvalue", float(ks.pvalue), detail=f"D={ks.statistic:.3f}"))
    return out


def prediction_findings(model: str, ref: pd.Series, cur: pd.Series) -> list[DriftFinding]:
    r, c = ref.to_numpy(dtype=float), cur.to_numpy(dtype=float)
    shift = (c.mean() - r.mean()) / (r.std(ddof=1) + EPS)
    return [
        DriftFinding(model, "__prediction__", "psi", psi(r, c)),
        DriftFinding(model, "__prediction__", "mean_shift_sd", float(shift)),
        DriftFinding(model, "__prediction__", "ks_pvalue", float(stats.ks_2samp(r, c).pvalue)),
    ]


def load_window(wh, model: str, w: Window) -> pd.DataFrame:
    sql = f"""
        SELECT p.customer_id, p.score, f.*
        FROM {PREDICTIONS_TABLE} p
        LEFT JOIN {FEATURES_TABLE} f
          ON f.customer_id = p.customer_id AND f.as_of_date = CAST(p.predicted_at AS DATE)
        WHERE p.model_name = %(model)s AND p.predicted_at >= %(start)s AND p.predicted_at < %(end)s
    """
    return wh.query(sql, {"model": model, "start": w.start, "end": w.end})


def run(wh, models: list[str], run_date: date, sink: AlertSink) -> None:
    cur_w = Window(run_date - timedelta(days=CURRENT_DAYS), run_date)
    ref_w = Window(cur_w.start - timedelta(days=REFERENCE_DAYS), cur_w.start)
    for model in models:
        ref, cur = load_window(wh, model, ref_w), load_window(wh, model, cur_w)
        if cur.empty:
            log.warning("%s: no predictions logged in %s", model, cur_w)
            continue
        feats = [c for c in ref.columns if c not in {"customer_id", "score", "as_of_date", "expires_at"}]
        findings = feature_findings(model, ref, cur, feats) + prediction_findings(model, ref.score, cur.score)
        log.info("%s: %d findings over %d features", model, len(findings), len(feats))
        sink.submit(findings, run_date)
    sink.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily drift check")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--date", type=date.fromisoformat, default=date.today())
    ap.add_argument("--no-slack", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    from warehouse_client import connect
    wh = connect()
    run(wh, args.model, args.date, AlertSink(wh, post_to_slack=not args.no_slack))


if __name__ == "__main__":
    main()

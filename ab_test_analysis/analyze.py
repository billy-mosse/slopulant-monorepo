"""Standard A/B test analysis.

Input:  experiments.assignments (unit_id, experiment_id, variant, assigned_at)
        experiments.metrics     (unit_id, experiment_id, metric, metric_type, value, pre_value)
Output: experiments.results     one row per (experiment, metric, variant vs control)

Steps: SRM chi-square check -> CUPED -> Welch t-test (continuous) or
two-proportion z-test (binary) -> relative lift with delta-method CI ->
Benjamini-Hochberg across metrics.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from cuped import adjust

log = logging.getLogger("ab_test_analysis")

ASSIGNMENTS_TABLE = "experiments.assignments"
METRICS_TABLE = "experiments.metrics"
RESULTS_TABLE = "experiments.results"
CONTROL = "control"
SRM_ALPHA = 0.001
ALPHA = 0.05
MetricType = Literal["continuous", "binary"]


@dataclass
class MetricResult:
    experiment_id: str
    metric: str
    variant: str
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    lift: float
    lift_ci_low: float
    lift_ci_high: float
    p_value: float
    p_adjusted: float = float("nan")
    significant: bool = False
    cuped_variance_reduction: float = 0.0
    srm_flag: bool = False


def srm_check(counts: pd.Series, expected_split: dict[str, float] | None = None) -> float:
    """Chi-square goodness of fit of observed arm sizes vs the planned split (default equal)."""
    total = counts.sum()
    split = expected_split or {k: 1 / len(counts) for k in counts.index}
    expected = np.array([split[k] * total for k in counts.index])
    return float(stats.chisquare(counts.to_numpy(), expected).pvalue)


def welch_test(c: np.ndarray, t: np.ndarray) -> float:
    return float(stats.ttest_ind(t, c, equal_var=False).pvalue)


def two_proportion_z(c: np.ndarray, t: np.ndarray) -> float:
    p_c, p_t, n_c, n_t = c.mean(), t.mean(), len(c), len(t)
    pooled = (c.sum() + t.sum()) / (n_c + n_t)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    if se == 0:
        return 1.0
    return float(2 * stats.norm.sf(abs((p_t - p_c) / se)))


def relative_lift_ci(c: np.ndarray, t: np.ndarray, alpha: float = ALPHA) -> tuple[float, float, float]:
    """Lift = mean_t / mean_c - 1. Delta method on the ratio of independent means:
    Var(mt/mc) ~= var_t/(n_t mc^2) + mt^2 var_c/(n_c mc^4)."""
    mc, mt = c.mean(), t.mean()
    if mc == 0:
        return float("nan"), float("nan"), float("nan")
    var_ratio = t.var(ddof=1) / (len(t) * mc**2) + mt**2 * c.var(ddof=1) / (len(c) * mc**4)
    z = stats.norm.ppf(1 - alpha / 2)
    lift = mt / mc - 1
    half = z * np.sqrt(var_ratio)
    return float(lift), float(lift - half), float(lift + half)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def analyze_experiment(exp_id: str, assignments: pd.DataFrame, metrics: pd.DataFrame) -> list[MetricResult]:
    arms = assignments.groupby("variant")["unit_id"].nunique()
    srm_p = srm_check(arms)
    srm_flag = srm_p < SRM_ALPHA
    if srm_flag:
        log.warning("%s: sample ratio mismatch (p=%.2e) %s", exp_id, srm_p, arms.to_dict())
    data = metrics.merge(assignments[["unit_id", "variant"]], on="unit_id", how="inner")
    results: list[MetricResult] = []
    for (metric, mtype), grp in data.groupby(["metric", "metric_type"]):
        values, cuped = adjust(grp) if mtype == "continuous" else (grp["value"].astype(float), None)
        grp = grp.assign(value=values)
        c = grp.loc[grp.variant == CONTROL, "value"].to_numpy()
        for variant in (v for v in grp.variant.unique() if v != CONTROL):
            t = grp.loc[grp.variant == variant, "value"].to_numpy()
            p = welch_test(c, t) if mtype == "continuous" else two_proportion_z(c, t)
            lift, lo, hi = relative_lift_ci(c, t)
            results.append(MetricResult(
                exp_id, metric, variant, len(c), len(t), float(c.mean()), float(t.mean()),
                lift, lo, hi, p,
                cuped_variance_reduction=cuped.variance_reduction if cuped else 0.0,
                srm_flag=srm_flag,
            ))
    if results:
        adj = benjamini_hochberg(np.array([r.p_value for r in results]))
        for r, pa in zip(results, adj):
            r.p_adjusted, r.significant = float(pa), bool(pa < ALPHA and not r.srm_flag)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Run standard analysis for an experiment")
    ap.add_argument("experiment_id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    from warehouse_client import connect
    wh = connect()
    params = {"exp": args.experiment_id}
    assignments = wh.query(f"SELECT unit_id, variant FROM {ASSIGNMENTS_TABLE} WHERE experiment_id = %(exp)s", params)
    metrics = wh.query(
        f"SELECT unit_id, metric, metric_type, value, pre_value FROM {METRICS_TABLE} WHERE experiment_id = %(exp)s", params)
    results = pd.DataFrame([asdict(r) for r in analyze_experiment(args.experiment_id, assignments, metrics)])
    log.info("\n%s", results[["metric", "variant", "lift", "p_adjusted", "significant"]].to_string())
    if not args.dry_run:
        wh.write(RESULTS_TABLE, results, partition=args.experiment_id)


if __name__ == "__main__":
    main()

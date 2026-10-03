"""Offline ranking debugger for search engineers.

    python debug_ranking.py --model-version latest --sample 2000 --top-k 10

Loads the ranker artifact registered in search.ranker_model, replays a sample of logged
queries from search.click_logs, and writes one row per query to search.rank_debug_report:
NDCG@k against click/cart labels, the features that drive the top results (tree-path
contributions from the booster), and a disagreement score vs. what users actually clicked.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F

log = logging.getLogger("rank_debugger")

MODEL_REGISTRY = "search.ranker_model"
CLICK_LOGS = "search.click_logs"
REPORT_TABLE = "search.rank_debug_report"

GAIN = {"impression": 0.0, "click": 1.0, "add_to_cart": 3.0}


@dataclass(frozen=True, slots=True)
class RankerArtifact:
    version: str
    booster: lgb.Booster
    feature_names: tuple[str, ...]


@dataclass(slots=True)
class QueryReport:
    query_id: str
    query_text: str
    n_results: int
    ndcg_logged: float
    ndcg_model: float
    disagreement: float
    top_drivers: list[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        return {**{k: getattr(self, k) for k in self.__slots__ if k != "top_drivers"},
                "top_drivers": json.dumps(self.top_drivers)}


# ------------------------------------------------------------------ loading

def load_ranker(spark: SparkSession, version: str) -> RankerArtifact:
    reg = spark.table(MODEL_REGISTRY).where(F.col("status") == "active")
    if version != "latest":
        reg = reg.where(F.col("version") == version)
    row = reg.orderBy(F.col("trained_at").desc()).first()
    if row is None:
        raise SystemExit(f"no ranker version {version!r} in {MODEL_REGISTRY}")
    booster = lgb.Booster(model_str=Path(row["artifact_path"]).read_text())
    log.info("loaded ranker %s (%d trees)", row["version"], booster.num_trees())
    return RankerArtifact(row["version"], booster, tuple(booster.feature_name()))


def sample_queries(spark: SparkSession, sample: int, days: int, seed: int) -> pd.DataFrame:
    logs = spark.table(CLICK_LOGS).where(F.col("event_date") >= F.date_sub(F.current_date(), days))
    qids = logs.select("query_id").distinct().orderBy(F.rand(seed)).limit(sample)
    return logs.join(qids, "query_id").toPandas()


def iter_queries(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    for qid, grp in df.groupby("query_id", sort=False):
        if len(grp) >= 2:
            yield qid, grp.sort_values("position")


# ------------------------------------------------------------------ metrics

def dcg(gains: np.ndarray, k: int) -> float:
    g = gains[:k]
    return float(np.sum((2.0 ** g - 1) / np.log2(np.arange(2, g.size + 2))))


def ndcg(gains_in_order: np.ndarray, k: int) -> float:
    ideal = dcg(np.sort(gains_in_order)[::-1], k)
    return dcg(gains_in_order, k) / ideal if ideal > 0 else 0.0


def labels(grp: pd.DataFrame) -> np.ndarray:
    return np.where(grp["added_to_cart"], GAIN["add_to_cart"],
                    np.where(grp["clicked"], GAIN["click"], GAIN["impression"]))


def disagreement(model_scores: np.ndarray, gains: np.ndarray) -> float:
    """Fraction of (clicked, unclicked) pairs the model orders the wrong way."""
    pos, neg = model_scores[gains > 0], model_scores[gains == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    return float(np.mean(pos[:, None] < neg[None, :]))


def top_drivers(contrib: np.ndarray, names: tuple[str, ...], n: int) -> list[dict]:
    """contrib is (rows, n_features + 1) from pred_contrib; last column is the bias term."""
    feats = contrib[:, :-1].mean(axis=0)
    order = np.argsort(-np.abs(feats))[:n]
    return [{"feature": names[i], "contribution": round(float(feats[i]), 5)} for i in order]


# ------------------------------------------------------------------ replay

def replay(df: pd.DataFrame, art: RankerArtifact, k: int, n_drivers: int) -> Iterator[QueryReport]:
    feature_cols = list(art.feature_names)
    for qid, grp in iter_queries(df):
        X = pd.DataFrame(grp["features"].map(json.loads).tolist()).reindex(columns=feature_cols)
        gains = labels(grp)
        scores = art.booster.predict(X)
        order = np.argsort(-scores)
        contrib = art.booster.predict(X.iloc[order[:k]], pred_contrib=True)
        yield QueryReport(
            query_id=str(qid),
            query_text=grp["query_text"].iat[0],
            n_results=len(grp),
            ndcg_logged=ndcg(gains, k),
            ndcg_model=ndcg(gains[order], k),
            disagreement=disagreement(scores, gains),
            top_drivers=top_drivers(np.asarray(contrib), art.feature_names, n_drivers),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay logged queries through the ranker and explain it")
    ap.add_argument("--model-version", default="latest")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--drivers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--worst", type=int, default=25, help="log the N most-disagreeing queries")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    spark = SparkSession.builder.appName("rank_debugger").getOrCreate()
    art = load_ranker(spark, args.model_version)
    df = sample_queries(spark, args.sample, args.days, args.seed)

    report = pd.DataFrame(r.as_row() for r in replay(df, art, args.top_k, args.drivers))
    report["model_version"] = art.version
    log.info("mean NDCG@%d logged=%.4f model=%.4f over %d queries", args.top_k,
             report["ndcg_logged"].mean(), report["ndcg_model"].mean(), len(report))
    for row in report.nlargest(args.worst, "disagreement").itertuples():
        log.info("disagree %.2f  %r", row.disagreement, row.query_text)

    spark.createDataFrame(report).write.mode("overwrite").saveAsTable(REPORT_TABLE)


if __name__ == "__main__":
    main()

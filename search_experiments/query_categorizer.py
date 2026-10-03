"""Classify search queries into product categories.

Training labels come from clicks: a (query, clicked product) pair is labelled
with the product's category from the catalog. Features are character n-grams
(2..4 with word-boundary padding), which copes with typos and compounds
("bathsheet", "duvetcover") better than word tokens.

Model: multinomial naive Bayes with Laplace smoothing, written out:

    log P(c | q) ∝ log P(c) + Σ_g count(g, q) · log P(g | c)
    P(g | c) = (N_gc + α) / (N_c + α · |V|)

Each training query contributes with weight log(1 + clicks) so head queries do
not drown the tail. Queries whose top posterior is below ``min_confidence``
are written with category ``NULL`` and their best guess kept separately.

Inputs:  search.query_logs, catalog.products
Output:  search.query_categories (query_norm, category, confidence, runner_up, runner_up_conf, model_version, scored_at)
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

import pandas as pd

from shared_utils import (ExperimentConfig, char_ngrams, get_logger, load_config, normalize_query,
                          precision_at_k, read_sql, require_columns, summarize_metrics, timed,
                          write_table)

log = get_logger("search_exp.query_categorizer")

QUERY_LOGS = "search.query_logs"
PRODUCTS = "catalog.products"
OUTPUT = "search.query_categories"
MODEL_VERSION = "qcat-cnb-v4"

TRAIN_SQL = f"""
SELECT q.query_text, p.category, COUNT(*) AS clicks
FROM {QUERY_LOGS} q
JOIN {PRODUCTS} p ON p.product_id = q.clicked_product_id
WHERE q.event_ts >= CURRENT_DATE - (:lookback_days * INTERVAL '1 day')
  AND p.category IS NOT NULL
GROUP BY q.query_text, p.category
"""

SCORE_SQL = f"""
SELECT query_text, COUNT(*) AS n_searches
FROM {QUERY_LOGS}
WHERE event_ts >= CURRENT_DATE - (:score_days * INTERVAL '1 day')
GROUP BY query_text
HAVING COUNT(*) >= :min_query_count
"""


@dataclass
class CategorizerConfig(ExperimentConfig):
    lookback_days: int = 180
    score_days: int = 30
    min_query_count: int = 3
    alpha: float = 0.3
    n_min: int = 2
    n_max: int = 4
    min_label_share: float = 0.5
    min_confidence: float = 0.55
    holdout_frac: float = 0.1
    max_vocab: int = 300_000
    exclude_categories: list[str] = field(default_factory=lambda: ["gift cards", "unknown"])

    def validate(self) -> None:
        super().validate()
        if not 0 < self.alpha <= 10:
            raise ValueError("alpha out of range")
        if not 1 <= self.n_min <= self.n_max <= 6:
            raise ValueError("need 1 <= n_min <= n_max <= 6")
        if not 0 <= self.holdout_frac < 0.5:
            raise ValueError("holdout_frac must be in [0, 0.5)")


@dataclass(frozen=True)
class Example:
    query: str
    label: str
    weight: float


@dataclass(frozen=True)
class Prediction:
    query: str
    ranked: tuple[tuple[str, float], ...]

    @property
    def top(self) -> tuple[str, float]:
        return self.ranked[0]

    @property
    def runner_up(self) -> tuple[str | None, float]:
        return self.ranked[1] if len(self.ranked) > 1 else (None, 0.0)


class CharNgramNB:
    """Multinomial NB over char n-grams with weighted examples."""

    def __init__(self, alpha: float = 0.3, n_min: int = 2, n_max: int = 4, max_vocab: int = 300_000):
        self.alpha, self.n_min, self.n_max, self.max_vocab = alpha, n_min, n_max, max_vocab
        self.class_weight: Counter = Counter()
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.totals: Counter = Counter()
        self.vocab: set[str] = set()
        self._log_prior: dict[str, float] = {}
        self._log_unseen: dict[str, float] = {}

    def featurize(self, q: str) -> Counter:
        return Counter(char_ngrams(q, self.n_min, self.n_max))

    def fit(self, examples: Iterable[Example]) -> "CharNgramNB":
        df: Counter = Counter()
        for ex in examples:
            feats = self.featurize(ex.query)
            self.class_weight[ex.label] += ex.weight
            for g, c in feats.items():
                self.counts[ex.label][g] += c * ex.weight
            df.update(feats.keys())
        self.vocab = {g for g, _ in df.most_common(self.max_vocab)}
        for c, cnt in self.counts.items():
            for g in [g for g in cnt if g not in self.vocab]:
                del cnt[g]
            self.totals[c] = sum(cnt.values())
        z = sum(self.class_weight.values())
        v = len(self.vocab)
        self._log_prior = {c: math.log(w / z) for c, w in self.class_weight.items()}
        self._log_unseen = {c: -math.log(self.totals[c] + self.alpha * v) for c in self.counts}
        log.info("fitted NB: classes=%d vocab=%d", len(self.counts), v)
        return self

    def log_likelihoods(self, q: str) -> dict[str, float]:
        feats = {g: n for g, n in self.featurize(q).items() if g in self.vocab}
        out = {}
        for c, cnt in self.counts.items():
            denom = self._log_unseen[c]
            s = self._log_prior[c]
            for g, n in feats.items():
                s += n * (math.log(cnt.get(g, 0.0) + self.alpha) + denom)
            out[c] = s
        return out

    def predict(self, q: str, top_k: int = 3) -> Prediction:
        ll = self.log_likelihoods(q)
        m = max(ll.values())
        exp = {c: math.exp(v - m) for c, v in ll.items()}
        z = sum(exp.values())
        ranked = sorted(((c, p / z) for c, p in exp.items()), key=lambda x: -x[1])[:top_k]
        return Prediction(q, tuple(ranked))


def load_training(cfg: CategorizerConfig) -> list[Example]:
    df = read_sql(TRAIN_SQL, cfg.warehouse_uri, params={"lookback_days": cfg.lookback_days})
    require_columns(df, ["query_text", "category", "clicks"], "training pairs")
    df["query_norm"] = df["query_text"].map(normalize_query)
    df["category"] = df["category"].str.strip().str.lower()
    df = df[~df["category"].isin(cfg.exclude_categories) & (df["query_norm"] != "")]
    df = df.groupby(["query_norm", "category"], as_index=False)["clicks"].sum()
    share = df["clicks"] / df.groupby("query_norm")["clicks"].transform("sum")
    df = df[share >= cfg.min_label_share]  # ambiguous queries ("white") are left out of training
    log.info("training queries=%d categories=%d", df["query_norm"].nunique(), df["category"].nunique())
    return [Example(q, c, math.log1p(n)) for q, c, n in df[["query_norm", "category", "clicks"]].itertuples(index=False)]


def split(examples: Sequence[Example], frac: float, seed: int) -> tuple[list[Example], list[Example]]:
    """Split by query string so the same query never sits in both halves."""
    queries = sorted({e.query for e in examples})
    rng = random.Random(seed)
    held = set(rng.sample(queries, int(len(queries) * frac))) if frac else set()
    return [e for e in examples if e.query not in held], [e for e in examples if e.query in held]


def evaluate(model: CharNgramNB, test: Sequence[Example], min_conf: float) -> dict[str, float]:
    if not test:
        return {}
    preds = [(e, model.predict(e.query)) for e in test]
    w = sum(e.weight for e, _ in preds)
    acc = sum(e.weight for e, p in preds if p.top[0] == e.label) / w
    # one relevant label per query, so precision@3 * 3 is the weighted hit rate in the top 3
    hit_3 = sum(e.weight * 3 * precision_at_k([c for c, _ in p.ranked], {e.label}, 3) for e, p in preds) / w
    confident = [(e, p) for e, p in preds if p.top[1] >= min_conf]
    cov = len(confident) / len(preds)
    conf_acc = sum(p.top[0] == e.label for e, p in confident) / max(1, len(confident))
    return {"accuracy": acc, "hit_at_3": hit_3, "coverage_at_conf": cov, "precision_at_conf": conf_acc}


def per_category_report(model: CharNgramNB, test: Sequence[Example], top: int = 10) -> pd.DataFrame:
    rows = [(e.label, model.predict(e.query).top[0] == e.label) for e in test]
    df = pd.DataFrame(rows, columns=["category", "correct"])
    out = df.groupby("category")["correct"].agg(["mean", "size"]).rename(columns={"mean": "acc", "size": "n"})
    return out.sort_values("n", ascending=False).head(top)


def score_queries(model: CharNgramNB, cfg: CategorizerConfig) -> pd.DataFrame:
    params = {"score_days": cfg.score_days, "min_query_count": cfg.min_query_count}
    q = read_sql(SCORE_SQL, cfg.warehouse_uri, params=params)
    queries = sorted({normalize_query(t) for t in q["query_text"]} - {""})
    rows = []
    with timed(log, f"score {len(queries)} queries"):
        for query in queries:
            p = model.predict(query)
            cat, conf = p.top
            ru, ru_conf = p.runner_up
            rows.append({"query_norm": query, "category": cat if conf >= cfg.min_confidence else None,
                         "confidence": round(conf, 4), "runner_up": ru, "runner_up_conf": round(ru_conf, 4),
                         "best_guess": cat})
    df = pd.DataFrame(rows)
    df["model_version"] = MODEL_VERSION
    df["scored_at"] = datetime.now(timezone.utc)
    log.info("assigned a category to %.1f%% of queries", 100 * df["category"].notna().mean())
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train char n-gram NB and write {OUTPUT}")
    p.add_argument("--config")
    p.add_argument("--lookback-days", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--min-confidence", type=float)
    p.add_argument("--holdout-frac", type=float)
    p.add_argument("--dry-run", action="store_true", default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--predict", nargs="*", metavar="QUERY", help="print predictions and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "eval_only", "predict"}}
    cfg = load_config(CategorizerConfig, args.config, overrides)
    examples = load_training(cfg)
    train, test = split(examples, cfg.holdout_frac, cfg.seed)
    with timed(log, "fit"):
        model = CharNgramNB(cfg.alpha, cfg.n_min, cfg.n_max, cfg.max_vocab).fit(train)
    summarize_metrics(evaluate(model, test, cfg.min_confidence), log, "holdout")
    if test:
        print(per_category_report(model, test).to_string())
    if args.predict:
        for q in args.predict:
            print(q, "->", [(c, round(p, 3)) for c, p in model.predict(normalize_query(q)).ranked])
        return 0
    if args.eval_only:
        return 0
    model = CharNgramNB(cfg.alpha, cfg.n_min, cfg.n_max, cfg.max_vocab).fit(examples)
    write_table(score_queries(model, cfg), OUTPUT, cfg.warehouse_uri, dry_run=cfg.dry_run, logger=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

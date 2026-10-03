"""Email subject-line scorer.

Predicts the open rate of a candidate subject line from its text alone, so the
CRM team can rank variants before a send instead of burning an A/B split on
obviously weak ones. Ridge regression over hand-built features (length, emoji,
personalization token, urgency words, question mark) plus TF-IDF word n-grams,
trained on historical sends.

Input:  marketing.email_events   (send_id, subject_line, event_type, customer_id, event_ts)
Output: marketing.subject_line_scores (subject_line, predicted_open_rate, model_version, scored_at)
"""

import argparse
import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("subject_lines")

WAREHOUSE_URI = "postgresql://analytics@warehouse/slopulent"
SENDS_SQL = """
    SELECT send_id,
           MAX(subject_line)                                        AS subject_line,
           COUNT(*) FILTER (WHERE event_type = 'delivered')         AS delivered,
           COUNT(DISTINCT customer_id) FILTER (WHERE event_type = 'open') AS opens
    FROM marketing.email_events
    WHERE event_ts >= CURRENT_DATE - INTERVAL '365 days'
    GROUP BY send_id
    HAVING COUNT(*) FILTER (WHERE event_type = 'delivered') >= 500
"""
SCORES_TABLE = "marketing.subject_line_scores"
MODEL_VERSION = "subj-ridge-2026.09"
RIDGE_ALPHA = 3.0

URGENCY_WORDS = {"today", "now", "last", "ends", "hurry", "final", "hours", "tonight", "only", "limited"}
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
PERSONALIZATION_RE = re.compile(r"\{\{\s*(first_name|name|city)\s*\}\}|%FNAME%", re.I)
PERCENT_RE = re.compile(r"\d+\s*%")


class SubjectFeatures(BaseEstimator, TransformerMixin):
    """Hand-crafted subject line features. Stateless, so fit is a no-op."""

    names = ["n_chars", "n_words", "has_emoji", "has_personalization",
             "n_urgency", "has_question", "has_percent", "caps_ratio"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for s in pd.Series(X).fillna(""):
            words = re.findall(r"[a-zA-Z']+", s.lower())
            letters = [c for c in s if c.isalpha()]
            rows.append([
                len(s),
                len(words),
                int(bool(EMOJI_RE.search(s))),
                int(bool(PERSONALIZATION_RE.search(s))),
                sum(w in URGENCY_WORDS for w in words),
                int("?" in s),
                int(bool(PERCENT_RE.search(s))),
                sum(c.isupper() for c in letters) / max(len(letters), 1),
            ])
        return np.asarray(rows, dtype=float)


def load_sends(engine) -> pd.DataFrame:
    df = pd.read_sql(SENDS_SQL, engine)
    df["open_rate"] = df["opens"] / df["delivered"]
    # a handful of sends have open rate > 1 from bot pre-fetch (Apple MPP); clip them
    df["open_rate"] = df["open_rate"].clip(0, 0.95)
    log.info("loaded %d historical sends, mean open rate %.3f", len(df), df["open_rate"].mean())
    return df


def build_pipeline(alpha: float = RIDGE_ALPHA) -> Pipeline:
    features = ColumnTransformer([
        ("handmade", Pipeline([("feat", SubjectFeatures()), ("scale", StandardScaler())]), "subject_line"),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                                  token_pattern=r"(?u)\b\w+\b"), "subject_line"),
    ])
    return Pipeline([("features", features), ("ridge", Ridge(alpha=alpha))])


def train(sends: pd.DataFrame, alpha: float) -> Pipeline:
    X, y = sends[["subject_line"]], sends["open_rate"]
    w = np.sqrt(sends["delivered"])  # bigger sends -> less noisy open rate
    X_tr, X_te, y_tr, y_te, w_tr, _ = train_test_split(X, y, w, test_size=0.2, random_state=7)
    model = build_pipeline(alpha)
    model.fit(X_tr, y_tr, ridge__sample_weight=w_tr)
    pred = model.predict(X_te)
    log.info("holdout MAE=%.4f  R2=%.3f", mean_absolute_error(y_te, pred), r2_score(y_te, pred))
    model.fit(X, y, ridge__sample_weight=w)
    return model


def score_candidates(model: Pipeline, candidates: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"subject_line": candidates})
    out["predicted_open_rate"] = np.clip(model.predict(out[["subject_line"]]), 0, 1)
    out["model_version"] = MODEL_VERSION
    out["scored_at"] = datetime.utcnow()
    return out.sort_values("predicted_open_rate", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate email subject lines")
    parser.add_argument("--candidates", required=True, help="text file, one subject line per line")
    parser.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    engine = create_engine(WAREHOUSE_URI)
    model = train(load_sends(engine), args.alpha)

    with open(args.candidates, encoding="utf-8") as fh:
        candidates = [line.strip() for line in fh if line.strip()]
    scores = score_candidates(model, candidates)
    print(scores[["subject_line", "predicted_open_rate"]].to_string(index=False))

    if not args.no_write:
        schema, table = SCORES_TABLE.split(".")
        scores.to_sql(table, engine, schema=schema, if_exists="append", index=False)
        log.info("wrote %d scores to %s", len(scores), SCORES_TABLE)


if __name__ == "__main__":
    main()

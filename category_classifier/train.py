"""Train the hierarchical product category classifier.

Fits one TF-IDF + LogisticRegression pipeline per taxonomy level on human
labels from catalog.category_labels joined to catalog.products. Reports
macro-F1 per level on a stratified hold-out split and saves the models.

Usage:
    python train.py --dsn $CATALOG_DSN --out models/category_v3.joblib
"""
from __future__ import annotations

import argparse
import logging

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine

from taxonomy import is_valid, split_path

logger = logging.getLogger("train")

LEVELS = ("l1", "l2", "l3")
MIN_CLASS_COUNT = 5

TRAINING_QUERY = """
SELECT sku,
       title,
       description,
       category_path
FROM catalog.products
JOIN catalog.category_labels USING (sku)
WHERE label_source = 'human'
  AND is_current
"""


def build_text(df: pd.DataFrame) -> pd.Series:
    """Title is repeated so it carries more weight than the long description."""
    return (df["title"].fillna("") + " " + df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()


def make_pipeline(C: float = 4.0) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.9,
                                 sublinear_tf=True, max_features=200_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, max_features=300_000)),
    ])
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="saga")
    return Pipeline([("tfidf", features), ("clf", clf)])


def load_labels(dsn: str) -> pd.DataFrame:
    df = pd.read_sql(TRAINING_QUERY, create_engine(dsn))
    df[list(LEVELS)] = df["category_path"].apply(lambda p: pd.Series(split_path(p)))
    valid = df.apply(lambda r: is_valid(r.l1, r.l2, r.l3), axis=1)
    if (~valid).any():
        logger.warning("dropping %d rows with paths not in taxonomy", (~valid).sum())
    df = df[valid]
    counts = df["l3"].value_counts()
    rare = counts[counts < MIN_CLASS_COUNT].index
    return df[~df["l3"].isin(rare)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical category classifier")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--C", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load_labels(args.dsn)
    logger.info("loaded %d labelled products, %d leaf classes", len(df), df["l3"].nunique())

    X = build_text(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, df[list(LEVELS)], test_size=args.test_size, stratify=df["l3"], random_state=args.seed
    )

    models = {}
    for level in LEVELS:
        pipe = make_pipeline(args.C).fit(X_tr, y_tr[level])
        pred = pipe.predict(X_te)
        f1 = f1_score(y_te[level], pred, average="macro")
        logger.info("%s: macro-F1 = %.4f (%d classes)", level, f1, y_tr[level].nunique())
        models[level] = pipe

    joblib.dump({"models": models, "levels": LEVELS, "C": args.C}, args.out)
    logger.info("saved models to %s", args.out)


if __name__ == "__main__":
    main()

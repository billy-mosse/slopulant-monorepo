"""Feature builders for the search learning-to-rank model.

Every builder takes and returns a pandas DataFrame keyed on (query_id, sku).
"""
import json
import logging
import math
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "emb_cosine",
    "bm25_title",
    "query_category_match",
    "ctr_smoothed",
    "price_z_in_query",
    "review_rating",
    "review_count_log",
]


def add_embedding_cosine(df: pd.DataFrame, product_emb: pd.DataFrame, query_emb: pd.DataFrame) -> pd.DataFrame:
    """Cosine between query-tower and item-tower vectors."""
    logger.info("Computing embedding cosine for %d rows", len(df))
    p = product_emb.set_index("sku")["embedding"].map(np.asarray)
    q = query_emb.set_index("query_text")["embedding"].map(np.asarray)
    pv = np.stack(df["sku"].map(p).values)
    qv = np.stack(df["query_text"].map(q).values)
    num = (pv * qv).sum(axis=1)
    den = np.linalg.norm(pv, axis=1) * np.linalg.norm(qv, axis=1) + 1e-9
    df["emb_cosine"] = num / den
    return df


def add_bm25_title(df: pd.DataFrame, k1: float, b: float) -> pd.DataFrame:
    """Okapi BM25 of query terms against product title, IDF computed over candidate titles."""
    titles = df.drop_duplicates("sku").set_index("sku")["title"].fillna("").str.lower().str.split()
    n_docs = len(titles)
    avgdl = titles.map(len).mean() or 1.0
    doc_freq = Counter(t for toks in titles for t in set(toks))
    idf = {t: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in doc_freq.items()}
    logger.info("BM25 vocab size %d, avgdl %.1f", len(idf), avgdl)

    def score(query: str, sku: str) -> float:
        toks = titles.get(sku, [])
        tf = Counter(toks)
        norm = k1 * (1 - b + b * len(toks) / avgdl)
        return sum(idf.get(t, 0.0) * tf[t] * (k1 + 1) / (tf[t] + norm) for t in query.lower().split() if t in tf)

    df["bm25_title"] = [score(q, s) for q, s in zip(df["query_text"], df["sku"])]
    return df


def add_query_category_match(df: pd.DataFrame, intents: pd.DataFrame) -> pd.DataFrame:
    """Probability mass the query intent assigns to the product's category."""
    intent_map = {
        row.query_text: dict(json.loads(row.top_categories))
        for row in intents.itertuples()
    }
    df["query_category_match"] = [
        intent_map.get(q, {}).get(c, 0.0) for q, c in zip(df["query_text"], df["category_id"])
    ]
    logger.info("Intent coverage: %.1f%% of rows", 100 * (df["query_category_match"] > 0).mean())
    return df


def add_smoothed_ctr(df: pd.DataFrame, history: pd.DataFrame, prior_clicks: float, prior_impr: float) -> pd.DataFrame:
    """Beta-prior smoothed historical CTR per (query, sku): (clicks + a) / (impr + a + b)."""
    agg = history.groupby(["query_text", "sku"]).agg(clicks=("clicked", "sum"), impr=("clicked", "size")).reset_index()
    agg["ctr_smoothed"] = (agg["clicks"] + prior_clicks) / (agg["impr"] + prior_impr)
    df = df.merge(agg[["query_text", "sku", "ctr_smoothed"]], on=["query_text", "sku"], how="left")
    df["ctr_smoothed"] = df["ctr_smoothed"].fillna(prior_clicks / prior_impr)
    return df


def add_price_z(df: pd.DataFrame) -> pd.DataFrame:
    """Price z-score within each query's candidate set."""
    g = df.groupby("query_id")["price"]
    df["price_z_in_query"] = ((df["price"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    return df


def add_review_features(df: pd.DataFrame) -> pd.DataFrame:
    df["review_rating"] = df["avg_rating"].fillna(df["avg_rating"].median())
    df["review_count_log"] = np.log1p(df["review_count"].fillna(0))
    return df

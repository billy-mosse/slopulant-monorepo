"""Aspect-based sentiment over approved reviews.

Reads reviews.approved, writes per-SKU x aspect aggregates to reviews.aspect_sentiment.

For each aspect mention we score polarity words within +/- WINDOW tokens,
weighting by distance, applying negation and intensifiers that precede the
polarity word inside the same clause.
"""
from __future__ import annotations

import argparse
import logging
import re
import unicodedata

import pandas as pd
from sqlalchemy import create_engine, text

from lexicon import (ASPECTS, INTENSIFIERS, NEGATION_FACTOR, NEGATORS, POLARITY,
                     SCOPE_BREAKERS)

log = logging.getLogger("review_sentiment")

WINDOW = 5
LOOKBACK = 3   # tokens before a polarity word checked for negators/intensifiers
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,;!?]")
MAX_PHRASE = max(len(p.split()) for p in [*POLARITY, *INTENSIFIERS, *(t for ts in ASPECTS.values() for t in ts)])

ASPECT_TERMS = {term: aspect for aspect, terms in ASPECTS.items() for term in terms}


def normalise(raw: str | None) -> str:
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw).lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)   # "sooooo" -> "soo"
    return s


def tokenize(s: str) -> list[str]:
    """Greedy longest-match merge so 'fell apart' or 'a bit' become one token."""
    raw = TOKEN_RE.findall(s)
    out, i = [], 0
    while i < len(raw):
        for n in range(min(MAX_PHRASE, len(raw) - i), 1, -1):
            cand = " ".join(raw[i:i + n])
            if cand in POLARITY or cand in INTENSIFIERS or cand in ASPECT_TERMS:
                out.append(cand)
                i += n
                break
        else:
            out.append(raw[i])
            i += 1
    return out


def word_polarity(tokens: list[str], j: int) -> float:
    base = POLARITY[tokens[j]]
    mult = 1.0
    for k in range(j - 1, max(-1, j - 1 - LOOKBACK), -1):
        t = tokens[k]
        if t in SCOPE_BREAKERS:
            break
        if t in INTENSIFIERS:
            mult *= INTENSIFIERS[t]
        elif t in NEGATORS:
            mult *= NEGATION_FACTOR
            break  # double negation is rare in reviews and usually sarcasm
    return max(-1.0, min(1.0, base * mult))


def score_mentions(tokens: list[str]) -> list[tuple[str, float]]:
    results = []
    for i, tok in enumerate(tokens):
        aspect = ASPECT_TERMS.get(tok)
        if aspect is None:
            continue
        num = den = 0.0
        for j in range(max(0, i - WINDOW), min(len(tokens), i + WINDOW + 1)):
            if tokens[j] not in POLARITY:
                continue
            # an aspect word that is also a polarity word ("soft") scores itself
            w = 1.0 / (1 + abs(i - j))
            num += w * word_polarity(tokens, j)
            den += w
        if den > 0:
            results.append((aspect, num / den))
    return results


def score_reviews(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        text_ = normalise(f"{r.title or ''}. {r.body or ''}")
        for aspect, s in score_mentions(tokenize(text_)):
            rows.append((r.review_id, r.sku, aspect, s))
    return pd.DataFrame(rows, columns=["review_id", "sku", "aspect", "score"])


def aggregate(m: pd.DataFrame) -> pd.DataFrame:
    # one vote per review per aspect, so a rant repeating "scratchy" counts once
    per_review = m.groupby(["review_id", "sku", "aspect"], as_index=False)["score"].mean()
    per_review["pos"] = per_review["score"] > 0.15
    per_review["neg"] = per_review["score"] < -0.15
    agg = per_review.groupby(["sku", "aspect"]).agg(
        mentions=("review_id", "nunique"), mean_score=("score", "mean"),
        pos_share=("pos", "mean"), neg_share=("neg", "mean")).reset_index()
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--min-mentions", type=int, default=3)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    engine = create_engine(args.dsn)

    df = pd.read_sql(text("SELECT review_id, sku, title, body FROM reviews.approved"), engine)
    mentions = score_reviews(df)
    log.info("%d reviews -> %d aspect mentions", len(df), len(mentions))
    agg = aggregate(mentions)
    agg = agg[agg["mentions"] >= args.min_mentions]
    agg.to_sql("aspect_sentiment", engine, schema="reviews", if_exists="replace", index=False)


if __name__ == "__main__":
    main()

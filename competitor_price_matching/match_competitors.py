"""Match our SKUs to scraped competitor listings and compute price gaps.

Input:  catalog.products, pricing.competitor_listings
Output: pricing.competitor_matches (our_sku, competitor, competitor_price, price_gap, ...)

Scoring per candidate pair (same brand + category block):
    score = 0.70 * token_set_ratio(titles)
          + 0.15 * attribute_agreement(size, color)
          + 0.15 * price_plausibility
price_plausibility is the two-sided tail probability of log(p_comp / p_ours)
under N(0, sigma) -- listings priced wildly differently are rarely the same item.
"""
from __future__ import annotations

import argparse
import logging
import re
import string
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine

log = logging.getLogger("competitor_match")

PRODUCTS_SQL = """
SELECT sku, title, brand, category, size, color, list_price
FROM catalog.products
WHERE is_active = TRUE
"""
LISTINGS_SQL = """
SELECT listing_id, competitor, title, brand, category, price, scraped_at
FROM pricing.competitor_listings
WHERE scraped_at >= CURRENT_DATE - INTERVAL '7 days'
"""
OUTPUT_TABLE = "pricing.competitor_matches"

MATCH_THRESHOLD = 0.78
LOG_PRICE_SIGMA = 0.45          # ~ +/-57% at one sigma, fit on hand-labelled pairs
W_TITLE, W_ATTR, W_PRICE = 0.70, 0.15, 0.15

UNIT_PATTERNS = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:inches|inch|in\b|\")"), r"\1in"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:centimeters|centimetres|cm)\b"), r"\1cm"),
    (re.compile(r"(\d+)\s*(?:thread\s*count|tc)\b"), r"\1tc"),
    (re.compile(r"(\d+)\s*(?:pieces|piece|pcs|pc)\b"), r"\1pc"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:ounces|ounce|oz)\b"), r"\1oz"),
]
SIZE_WORDS = {"twin", "full", "queen", "king", "cal king", "california king", "standard", "euro"}
COLOR_WORDS = {"white", "ivory", "black", "grey", "gray", "navy", "blue", "green", "sage",
               "blush", "pink", "beige", "taupe", "charcoal", "natural", "linen"}
_PUNCT = re.compile(f"[{re.escape(string.punctuation.replace(chr(34), ''))}]")


def normalize_title(t: str) -> str:
    t = (t or "").lower().replace("grey", "gray").replace("california king", "cal king")
    for pat, rep in UNIT_PATTERNS:
        t = pat.sub(rep, t)
    t = _PUNCT.sub(" ", t).replace('"', " ")
    return re.sub(r"\s+", " ", t).strip()


def token_set_ratio(a: str, b: str) -> float:
    """fuzzywuzzy-style token set ratio in [0, 1]."""
    sa, sb = set(a.split()), set(b.split())
    inter = " ".join(sorted(sa & sb))
    diff_a = " ".join(sorted(sa - sb))
    diff_b = " ".join(sorted(sb - sa))
    t1, t2, t3 = inter, f"{inter} {diff_a}".strip(), f"{inter} {diff_b}".strip()
    return max(SequenceMatcher(None, x, y).ratio() for x, y in ((t1, t2), (t1, t3), (t2, t3)))


def extract_attr(title: str, vocab: set[str]) -> str | None:
    hits = [w for w in vocab if re.search(rf"\b{re.escape(w)}\b", title)]
    return max(hits, key=len) if hits else None


def attribute_agreement(our_size, our_color, title: str) -> float:
    """1 if both agree, 0 if either conflicts; unknowns count as half."""
    scores = []
    for ours, vocab in ((our_size, SIZE_WORDS), (our_color, COLOR_WORDS)):
        theirs = extract_attr(title, vocab)
        if not ours or theirs is None:
            scores.append(0.5)
        else:
            scores.append(1.0 if normalize_title(str(ours)) == theirs else 0.0)
    return 0.0 if 0.0 in scores else float(np.mean(scores))


def price_plausibility(ours: float, theirs: float) -> float:
    if not ours or not theirs or ours <= 0 or theirs <= 0:
        return 0.0
    z = np.log(theirs / ours) / LOG_PRICE_SIGMA
    return float(2 * stats.norm.sf(abs(z)))


def block_key(df: pd.DataFrame) -> pd.Series:
    return df["brand"].fillna("").str.lower().str.strip() + "|" + df["category"].fillna("").str.lower()


def match(products: pd.DataFrame, listings: pd.DataFrame,
          threshold: float = MATCH_THRESHOLD) -> pd.DataFrame:
    products = products.assign(norm=products["title"].map(normalize_title), block=block_key(products))
    listings = listings.assign(norm=listings["title"].map(normalize_title), block=block_key(listings))
    rows = []
    for block, comp in listings.groupby("block"):
        ours = products[products["block"] == block]
        if ours.empty:
            continue
        for c in comp.itertuples(index=False):
            best = None
            for p in ours.itertuples(index=False):
                s_title = token_set_ratio(p.norm, c.norm)
                if s_title < 0.5:
                    continue
                s_attr = attribute_agreement(p.size, p.color, c.norm)
                s_price = price_plausibility(p.list_price, c.price)
                score = W_TITLE * s_title + W_ATTR * s_attr + W_PRICE * s_price
                if best is None or score > best["score"]:
                    best = dict(our_sku=p.sku, competitor=c.competitor, listing_id=c.listing_id,
                                competitor_price=c.price, our_price=p.list_price,
                                title_score=s_title, attr_score=s_attr, price_score=s_price,
                                score=score)
            if best and best["score"] >= threshold:
                rows.append(best)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # One listing per (our_sku, competitor): keep the highest-scoring one.
    out = out.sort_values("score", ascending=False).drop_duplicates(["our_sku", "competitor"])
    out["price_gap"] = out["competitor_price"] - out["our_price"]
    out["price_gap_pct"] = out["price_gap"] / out["our_price"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Competitor listing matcher")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(args.dsn)
    products = pd.read_sql(PRODUCTS_SQL, engine)
    listings = pd.read_sql(LISTINGS_SQL, engine)
    log.info("%d products, %d competitor listings", len(products), len(listings))

    out = match(products, listings, args.threshold)
    if out.empty:
        log.warning("no matches above %.2f", args.threshold)
        return
    gaps = out["price_gap_pct"]
    log.info("%d matches; median gap %.1f%%, IQR [%.1f%%, %.1f%%]", len(out),
             100 * gaps.median(), 100 * gaps.quantile(0.25), 100 * gaps.quantile(0.75))
    if args.dry_run:
        print(out.head(30).to_string(index=False))
        return
    out["matched_at"] = pd.Timestamp.utcnow()
    schema, table = OUTPUT_TABLE.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    log.info("wrote %s", OUTPUT_TABLE)


if __name__ == "__main__":
    main()

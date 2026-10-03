"""Review ring detector.

Finds groups of reviewer accounts that post near-identical review text on the
same products within a short time span. Built by the trust team to feed the
manual investigation queue.

Approach:
    1. Pull recent reviews and normalize their text.
    2. Within each SKU, compare every pair of reviews using Jaccard similarity
       over character 5-gram shingles.
    3. Link reviewers whose reviews exceed the similarity threshold, and merge
       links into groups with a union-find structure.
    4. Keep groups whose members co-reviewed at least a few products within
       the time window, and write them out.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import re
from datetime import timedelta
from typing import Iterable, NamedTuple

import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger("review_rings")

SHINGLE_SIZE = 5
SIMILARITY_THRESHOLD = 0.62
WINDOW_DAYS = 10
MIN_RING_SIZE = 3
MIN_SHARED_SKUS = 2
MAX_REVIEWS_PER_SKU = 400  # caps the O(n^2) pairwise loop on very popular SKUs

REVIEWS_QUERY = """
    SELECT review_id, reviewer_id, sku, body, submitted_at
    FROM reviews.raw
    WHERE submitted_at >= CURRENT_DATE - INTERVAL '{days} days'
      AND body IS NOT NULL
"""
OUTPUT_TABLE = "trust.review_rings"

_WS = re.compile(r"\s+")
_NON_TEXT = re.compile(r"[^a-z0-9 ]")


class SuspiciousPair(NamedTuple):
    reviewer_a: str
    reviewer_b: str
    sku: str
    similarity: float


class DisjointSet:
    """Union-find with path compression and union by size."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for node in list(self.parent):
            out.setdefault(self.find(node), []).append(node)
        return out


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace."""
    text = _NON_TEXT.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def shingles(text: str, k: int = SHINGLE_SIZE) -> frozenset[str]:
    """Character k-grams of a normalized string.

    Short texts (fewer than k characters) return the whole string as a single
    shingle so they can still match exact duplicates.
    """
    if len(text) < k:
        return frozenset([text]) if text else frozenset()
    return frozenset(text[i : i + k] for i in range(len(text) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def similar_pairs_for_sku(sku: str, block: pd.DataFrame) -> Iterable[SuspiciousPair]:
    """All-pairs comparison of reviews posted on a single SKU.

    Args:
        sku: The product identifier.
        block: Reviews for that SKU with a precomputed ``shingles`` column.

    Yields:
        Pairs of distinct reviewers whose texts are above the threshold and
        were posted within ``WINDOW_DAYS`` of each other.
    """
    rows = block.sort_values("submitted_at").head(MAX_REVIEWS_PER_SKU).to_dict("records")
    window = timedelta(days=WINDOW_DAYS)
    for r1, r2 in itertools.combinations(rows, 2):
        if r1["reviewer_id"] == r2["reviewer_id"]:
            continue
        if abs(r2["submitted_at"] - r1["submitted_at"]) > window:
            continue
        sim = jaccard(r1["shingles"], r2["shingles"])
        if sim >= SIMILARITY_THRESHOLD:
            yield SuspiciousPair(r1["reviewer_id"], r2["reviewer_id"], sku, sim)


def build_rings(pairs: list[SuspiciousPair]) -> pd.DataFrame:
    """Merge suspicious pairs into rings and keep the ones worth investigating.

    Args:
        pairs: Output of ``similar_pairs_for_sku`` across all SKUs.

    Returns:
        One row per (ring, reviewer) with ring-level stats attached.
    """
    ds = DisjointSet()
    for p in pairs:
        ds.union(p.reviewer_a, p.reviewer_b)

    pair_df = pd.DataFrame(pairs, columns=SuspiciousPair._fields)
    pair_df["ring_root"] = pair_df["reviewer_a"].map(ds.find)

    records = []
    for root, members in ds.groups().items():
        if len(members) < MIN_RING_SIZE:
            continue
        ring_pairs = pair_df[pair_df["ring_root"] == root]
        shared_skus = ring_pairs["sku"].nunique()
        if shared_skus < MIN_SHARED_SKUS:
            continue
        ring_id = f"ring_{abs(hash(tuple(sorted(members)))) % 10**10:010d}"
        for reviewer in sorted(members):
            records.append(
                {
                    "ring_id": ring_id,
                    "reviewer_id": reviewer,
                    "ring_size": len(members),
                    "shared_skus": shared_skus,
                    "mean_similarity": round(ring_pairs["similarity"].mean(), 4),
                    "max_similarity": round(ring_pairs["similarity"].max(), 4),
                }
            )
    return pd.DataFrame.from_records(records)


def run(lookback_days: int, dry_run: bool) -> None:
    engine = create_engine(os.environ["WAREHOUSE_URL"])
    review_df = pd.read_sql(REVIEWS_QUERY.format(days=lookback_days), engine, parse_dates=["submitted_at"])
    logger.info("Loaded %d reviews across %d SKUs", len(review_df), review_df["sku"].nunique())

    review_df["shingles"] = review_df["body"].map(lambda b: shingles(normalize(b)))

    pairs: list[SuspiciousPair] = []
    for sku, block in review_df.groupby("sku"):
        if len(block) < 2:
            continue
        pairs.extend(similar_pairs_for_sku(sku, block))
    logger.info("Found %d suspicious reviewer pairs", len(pairs))

    rings = build_rings(pairs)
    if rings.empty:
        logger.info("No rings met the size/shared-SKU criteria")
        return
    rings["detected_at"] = pd.Timestamp.utcnow()
    logger.info("Flagged %d rings covering %d reviewers", rings["ring_id"].nunique(), len(rings))

    if dry_run:
        print(rings.head(30).to_string(index=False))
        return
    schema, table = OUTPUT_TABLE.split(".")
    rings.to_sql(table, engine, schema=schema, if_exists="append", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect coordinated review rings.")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run(args.lookback_days, args.dry_run)


if __name__ == "__main__":
    main()

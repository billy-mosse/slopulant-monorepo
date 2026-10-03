"""Coordinated fake-review ring detection.

Inputs:  reviews.raw, reviews.reviewer_devices
Output:  reviews.suspicious_rings (ring_id, members, evidence)

Edges between reviewers: shared device fingerprint, shared IP (excluding
high-fan-out IPs such as corporate NATs / mobile carriers), or near-duplicate
review text. A connected component is flagged when it shows a 5-star burst on
one SKU within 48h AND a large share of young accounts.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import pandas as pd
from sqlalchemy import create_engine, text

from minhash import near_duplicates

log = logging.getLogger("fake_review_detection")

BURST_WINDOW = pd.Timedelta(hours=48)
MIN_BURST_REVIEWS = 4
NEW_ACCOUNT_DAYS = 30
MIN_NEW_ACCOUNT_SHARE = 0.5
MAX_IP_FANOUT = 25        # shared IPs with more reviewers than this are noise
MIN_RING_SIZE = 3

RAW_SQL = """SELECT review_id, reviewer_id, sku, rating, body, created_at, account_created_at
FROM reviews.raw WHERE created_at >= :since"""
DEVICE_SQL = "SELECT reviewer_id, device_hash, ip_address FROM reviews.reviewer_devices"


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_edges(reviews: pd.DataFrame, devices: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    edges: dict[tuple[str, str], set[str]] = defaultdict(set)
    for col, label in (("device_hash", "shared_device"), ("ip_address", "shared_ip")):
        grp = devices.dropna(subset=[col]).groupby(col)["reviewer_id"].unique()
        for members in grp:
            if len(members) < 2 or (label == "shared_ip" and len(members) > MAX_IP_FANOUT):
                continue
            anchor = members[0]
            for m in members[1:]:
                edges[tuple(sorted((anchor, m)))].add(label)
    texts = dict(zip(reviews["review_id"], reviews["body"].fillna("")))
    owner = dict(zip(reviews["review_id"], reviews["reviewer_id"]))
    for a, b, _ in near_duplicates(texts):
        ra, rb = owner[a], owner[b]
        if ra != rb:  # same person reposting is a different problem
            edges[tuple(sorted((ra, rb)))].add("near_duplicate_text")
    return edges


def burst_evidence(ring_reviews: pd.DataFrame) -> list[dict]:
    hits = []
    five = ring_reviews[ring_reviews["rating"] == 5].sort_values("created_at")
    for sku, g in five.groupby("sku"):
        ts = g["created_at"].tolist()
        lo = 0
        for hi in range(len(ts)):
            while ts[hi] - ts[lo] > BURST_WINDOW:
                lo += 1
            if hi - lo + 1 >= MIN_BURST_REVIEWS:
                hits.append({"sku": sku, "five_star_in_48h": hi - lo + 1,
                             "start": str(ts[lo]), "end": str(ts[hi])})
                break
    return hits


def detect(reviews: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    edges = build_edges(reviews, devices)
    uf = UnionFind()
    for a, b in edges:
        uf.union(a, b)
    comps: dict[str, set[str]] = defaultdict(set)
    for node in list(uf.parent):
        comps[uf.find(node)].add(node)

    rows = []
    for members in comps.values():
        if len(members) < MIN_RING_SIZE:
            continue
        rr = reviews[reviews["reviewer_id"].isin(members)]
        bursts = burst_evidence(rr)
        age = (rr.groupby("reviewer_id")["created_at"].min()
               - rr.groupby("reviewer_id")["account_created_at"].first()).dt.days
        new_share = float((age <= NEW_ACCOUNT_DAYS).mean()) if len(age) else 0.0
        if not bursts or new_share < MIN_NEW_ACCOUNT_SHARE:
            continue
        reasons = defaultdict(int)
        for (a, b), labels in edges.items():
            if a in members:
                for l in labels:
                    reasons[l] += 1
        rows.append({"members": sorted(members), "evidence": {
            "edge_types": dict(reasons), "bursts": bursts, "new_account_share": round(new_share, 2)}})
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "ring_id", [f"ring_{i:05d}" for i in range(len(out))])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--since", default="2026-06-01")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    engine = create_engine(args.dsn)

    reviews = pd.read_sql(text(RAW_SQL), engine, params={"since": args.since},
                          parse_dates=["created_at", "account_created_at"])
    devices = pd.read_sql(text(DEVICE_SQL), engine)
    rings = detect(reviews, devices)
    log.info("flagged %d rings covering %d reviewers", len(rings),
             sum(len(m) for m in rings["members"]) if len(rings) else 0)
    if len(rings):
        rings["members"] = rings["members"].map(json.dumps)
        rings["evidence"] = rings["evidence"].map(json.dumps)
        rings.to_sql("suspicious_rings", engine, schema="reviews", if_exists="append", index=False)


if __name__ == "__main__":
    main()

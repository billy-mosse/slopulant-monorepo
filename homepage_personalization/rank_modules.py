"""Per-customer homepage module ordering and slot filling -> recs.homepage_slots."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("homepage")

CUSTOMER_SQL = "SELECT customer_id, category_mix, recent_skus FROM features.customer_embeddings"
TRENDING_SQL = "SELECT sku, category_id, velocity_z FROM recs.trending WHERE computed_at = (SELECT MAX(computed_at) FROM recs.trending)"
I2I_SQL = "SELECT source_sku, target_sku, score FROM recs.i2i"
OUTPUT = ("recs", "homepage_slots")

MODULES = ("trending", "because_you_viewed", "new_arrivals", "sale")
SLOTS_PER_MODULE = 12
MMR_LAMBDA = 0.7
W_AFFINITY, W_TRENDING = 0.65, 0.35
MIN_MIX_MASS = 0.05


@dataclass
class Candidate:
    sku: str
    category: int
    trend: float
    prior: float = 0.0


def squash(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z / 2.0))


def mmr(cands: list[Candidate], rel: np.ndarray, k: int, lam: float = MMR_LAMBDA) -> list[int]:
    picked: list[int] = []
    left = set(range(len(cands)))
    while left and len(picked) < k:
        def gain(i: int) -> float:
            sim = max((1.0 if cands[i].category == cands[j].category else 0.0 for j in picked), default=0.0)
            return lam * rel[i] - (1 - lam) * sim
        best = max(left, key=gain)
        picked.append(best)
        left.remove(best)
    return picked


class Ranker:
    def __init__(self, trending: pd.DataFrame, i2i: pd.DataFrame, merch: pd.DataFrame, cat_index: dict):
        self.cat_index = cat_index
        self.trend_z = trending.set_index("sku").velocity_z.to_dict()
        self.sku_cat = {**merch.set_index("sku").category_id.to_dict(), **trending.set_index("sku").category_id.to_dict()}
        self.pools = {
            "trending": [Candidate(r.sku, r.category_id, r.velocity_z) for r in trending.itertuples()],
            "new_arrivals": self._from_merch(merch, "new_arrivals"),
            "sale": self._from_merch(merch, "sale"),
        }
        self.i2i = i2i.groupby("source_sku")[["target_sku", "score"]].apply(lambda g: list(zip(g.target_sku, g.score))).to_dict()

    def _from_merch(self, merch: pd.DataFrame, module: str) -> list[Candidate]:
        m = merch[merch.module == module]
        return [Candidate(r.sku, r.category_id, self.trend_z.get(r.sku, 0.0)) for r in m.itertuples()]

    def viewed_pool(self, recent: list[str]) -> list[Candidate]:
        agg: dict[str, float] = {}
        for s in recent:
            for t, sc in self.i2i.get(s, []):
                if t not in recent:
                    agg[t] = agg.get(t, 0.0) + sc
        return [Candidate(t, self.sku_cat.get(t, -1), self.trend_z.get(t, 0.0), sc) for t, sc in agg.items()]

    def relevance(self, cands: list[Candidate], mix: np.ndarray | None) -> np.ndarray:
        trend = squash(np.array([c.trend for c in cands]))
        if mix is None:
            return trend
        aff = np.array([mix[self.cat_index[c.category]] if c.category in self.cat_index else 0.0 for c in cands])
        aff = aff / (aff.max() or 1.0)
        prior = np.array([c.prior for c in cands])
        prior = prior / (prior.max() or 1.0)
        return W_AFFINITY * np.maximum(aff, prior) + W_TRENDING * trend

    def fill(self, mix: np.ndarray | None, recent: list[str]) -> list[dict]:
        cold = mix is None or mix.sum() < MIN_MIX_MASS
        pools = {"trending": self.pools["trending"]} if cold else {**self.pools, "because_you_viewed": self.viewed_pool(recent)}
        mods = []
        for name in MODULES:
            cands = pools.get(name) or []
            if not cands:
                continue
            rel = self.relevance(cands, None if cold else mix)
            idx = mmr(cands, rel, SLOTS_PER_MODULE)
            mods.append({"module": name, "skus": [cands[i].sku for i in idx], "strength": float(rel[idx].mean())})
        return sorted(mods, key=lambda m: -m["strength"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--merch-feed", required=True, help="parquet: sku, category_id, module (new_arrivals|sale)")
    p.add_argument("--categories", required=True, help="json list of category ids in category_mix order")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    eng = create_engine(a.dsn)
    cat_index = {c: i for i, c in enumerate(json.load(open(a.categories)))}
    ranker = Ranker(pd.read_sql(TRENDING_SQL, eng), pd.read_sql(I2I_SQL, eng), pd.read_parquet(a.merch_feed), cat_index)
    profiles = pd.read_sql(CUSTOMER_SQL, eng)

    rows = []
    for c in profiles.itertuples():
        mix = np.asarray(c.category_mix, dtype=float) if c.category_mix is not None else None
        for pos, m in enumerate(ranker.fill(mix, list(c.recent_skus or [])), 1):
            rows.append({"customer_id": c.customer_id, "module": m["module"], "module_rank": pos, "skus": json.dumps(m["skus"])})
    out = pd.DataFrame(rows)
    out.to_sql(OUTPUT[1], eng, schema=OUTPUT[0], if_exists="replace", index=False, chunksize=50_000)
    log.info("%d customers, %d module rows", len(profiles), len(out))


if __name__ == "__main__":
    main()

"""Random-projection LSH over image vectors, per category. Top 12 lookalikes per SKU."""
import argparse
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("neighbors")

OUT_TABLE = "recs.visual_similar"
TOP_K = 12
N_TABLES = 8
N_BITS = 12
MIN_CANDIDATES = 40


class RPIndex:
    def __init__(self, vecs, n_tables=N_TABLES, n_bits=N_BITS, seed=0):
        rng = np.random.default_rng(seed)
        self.vecs = vecs
        self.planes = rng.standard_normal((n_tables, vecs.shape[1], n_bits)).astype(np.float32)
        self.pow2 = 1 << np.arange(n_bits)
        self.tables = []
        for t in range(n_tables):
            buckets = defaultdict(list)
            for i, h in enumerate(self._hash(vecs, t)):
                buckets[h].append(i)
            self.tables.append(buckets)

    def _hash(self, x, t):
        return ((x @ self.planes[t]) > 0) @ self.pow2

    def query(self, i, k):
        q = self.vecs[i : i + 1]
        cand = set()
        for t, buckets in enumerate(self.tables):
            cand.update(buckets.get(int(self._hash(q, t)[0]), ()))
        cand.discard(i)
        if len(cand) < MIN_CANDIDATES:  # sparse buckets -> just brute force, categories are small anyway
            cand = set(range(len(self.vecs))) - {i}
        if not cand:
            return [], []
        c = np.fromiter(cand, int)
        sims = self.vecs[c] @ q[0]
        top = np.argsort(-sims)[:k]
        return c[top], sims[top]


def neighbours_for_category(skus, vecs):
    if len(skus) < 2:
        return []
    idx = RPIndex(vecs)
    rows = []
    for i, sku in enumerate(skus):
        nb, sims = idx.query(i, TOP_K)
        rows += [(sku, skus[j], r + 1, float(s)) for r, (j, s) in enumerate(zip(nb, sims))]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="image_vectors.npz")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    z = np.load(a.vectors, allow_pickle=True)
    skus, cats, vecs = z["sku"], z["category"], z["vec"]
    by_cat = defaultdict(list)
    for i, c in enumerate(cats):
        by_cat[c].append(i)

    rows = []
    for c, ix in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        ix = np.array(ix)
        rows += neighbours_for_category(skus[ix], vecs[ix])
        log.info("category %s: %d skus", c, len(ix))

    out = pd.DataFrame(rows, columns=["sku", "similar_sku", "rank", "cosine"])
    out["built_at"] = pd.Timestamp.utcnow()
    log.info("%d pairs, mean cosine@1 %.3f", len(out), out.loc[out["rank"] == 1, "cosine"].mean())
    if a.dry_run:
        print(out.head(36))
        return
    schema, table = OUT_TABLE.split(".")
    out.to_sql(table, create_engine(os.environ["WAREHOUSE_URL"]), schema=schema, if_exists="replace", index=False, chunksize=20000)


if __name__ == "__main__":
    main()

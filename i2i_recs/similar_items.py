"""Item-to-item recommendations ("You may also like").

For each SKU, returns the top-k most similar SKUs to show on the product page.
We vectorize products ourselves from title + description so we don't block on
anyone else's pipeline.

Input:  catalog.products (sku, title, description)
Output: recs.i2i (sku, similar_sku, score)
"""
import hashlib

import numpy as np

VEC_SIZE = 128


def words(s):
    return [w for w in s.lower().replace(",", " ").split() if len(w) > 2]


def item_vector(product):
    v = np.zeros(VEC_SIZE)
    for w in words(product["title"] + " " + product["description"]):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % VEC_SIZE] += 1.0 if (h >> 8) % 2 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def top_k_similar(products, k=5):
    skus = [p["sku"] for p in products]
    mat = np.stack([item_vector(p) for p in products])
    sims = mat @ mat.T
    recs = {}
    for i, sku in enumerate(skus):
        order = np.argsort(-sims[i])
        recs[sku] = [(skus[j], float(sims[i, j])) for j in order if j != i][:k]
    return recs


if __name__ == "__main__":
    catalog = [
        {"sku": "BED-001", "title": "Queen linen duvet cover", "description": "Stonewashed linen, oat"},
        {"sku": "BED-003", "title": "Linen pillowcase set", "description": "Stonewashed linen, oat, set of 2"},
        {"sku": "BTH-010", "title": "Turkish cotton bath towel", "description": "600gsm, white"},
    ]
    print(top_k_similar(catalog, k=2))

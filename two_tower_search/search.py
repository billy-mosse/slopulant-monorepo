"""Two-tower search.

Query tower embeds the search string; item tower is the shared product
embedding table (features.product_embeddings). Ranks SKUs by dot product.

Input:  features.product_embeddings (sku, vector[128]), user query string
Output: ranked list of (sku, score)
"""
import hashlib

import numpy as np

DIM = 128


def query_tower(query):
    vec = np.zeros(DIM)
    for tok in (t for t in query.lower().split() if len(t) > 2):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0 if (h >> 8) % 2 else -1.0
    n = np.linalg.norm(vec)
    return vec / n if n else vec


def search(query, product_embeddings, k=10):
    q = query_tower(query)
    scored = [(sku, float(q @ v)) for sku, v in product_embeddings.items()]
    return sorted(scored, key=lambda x: -x[1])[:k]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake_table = {f"SKU-{i}": rng.normal(size=DIM) for i in range(5)}
    print(search("linen duvet", fake_table, k=3))

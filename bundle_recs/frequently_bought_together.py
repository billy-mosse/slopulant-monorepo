"""Frequently bought together.

Suggests a bundle of 2-3 items to add to cart. Combines co-purchase counts
with text similarity so brand-new items (no purchases yet) still get bundles.

Input:  orders.lines (order_id, sku), catalog.products (sku, title, description)
Output: recs.bundles (sku, bundle_sku, score)
"""
import hashlib
from collections import Counter
from itertools import permutations

import numpy as np

N_BUCKETS = 128


def text_vec(title, description):
    v = np.zeros(N_BUCKETS)
    for tok in (title + " " + description).lower().replace(",", " ").split():
        if len(tok) <= 2:
            continue
        digest = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[digest % N_BUCKETS] += 1.0 if (digest >> 8) % 2 else -1.0
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def co_purchase_counts(order_lines):
    baskets = {}
    for order_id, sku in order_lines:
        baskets.setdefault(order_id, set()).add(sku)
    counts = Counter()
    for items in baskets.values():
        counts.update(permutations(sorted(items), 2))
    return counts


def bundles(products, order_lines, k=3, alpha=0.5):
    vecs = {p["sku"]: text_vec(p["title"], p["description"]) for p in products}
    counts = co_purchase_counts(order_lines)
    max_count = max(counts.values(), default=1)
    out = {}
    for a in vecs:
        scored = []
        for b in vecs:
            if a == b:
                continue
            score = alpha * counts[(a, b)] / max_count + (1 - alpha) * float(vecs[a] @ vecs[b])
            scored.append((b, round(score, 3)))
        out[a] = sorted(scored, key=lambda x: -x[1])[:k]
    return out


if __name__ == "__main__":
    products = [
        {"sku": "BED-001", "title": "Queen linen duvet cover", "description": "Stonewashed linen, oat"},
        {"sku": "BED-003", "title": "Linen pillowcase set", "description": "Stonewashed linen, oat"},
        {"sku": "BTH-010", "title": "Turkish cotton bath towel", "description": "600gsm, white"},
    ]
    orders = [(1, "BED-001"), (1, "BED-003"), (2, "BED-001"), (2, "BTH-010")]
    print(bundles(products, orders, k=2))

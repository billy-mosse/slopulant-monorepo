"""Product embeddings.

Builds a dense vector per SKU from title + description + validated tags.
Output is the shared product embedding table other teams are meant to consume.

Input:  catalog.products (sku, title, description), catalog.validated_tags (sku, tag)
Output: features.product_embeddings (sku, vector[128])
"""
import hashlib

import numpy as np

DIM = 128


def tokenize(text):
    return [t for t in text.lower().replace(",", " ").split() if len(t) > 2]


def hash_embed(tokens, dim=DIM):
    vec = np.zeros(dim)
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def build(products, tags_by_sku):
    out = {}
    for p in products:
        tokens = tokenize(p["title"]) + tokenize(p["description"])
        tokens += [f"tag:{t}" for t in tags_by_sku.get(p["sku"], [])]
        out[p["sku"]] = hash_embed(tokens)
    return out


if __name__ == "__main__":
    products = [
        {"sku": "BED-001", "title": "Queen linen duvet cover", "description": "Stonewashed linen, oat"},
        {"sku": "BED-002", "title": "Queen linen duvet", "description": "Stone washed linen in oat color"},
    ]
    embs = build(products, {"BED-001": ["bedding", "linen"], "BED-002": ["bedding", "linen"]})
    print({k: v[:4].round(3).tolist() for k, v in embs.items()})

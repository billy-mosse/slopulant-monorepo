"""Customer intent — p(conversion).

Predicts the probability a session converts, from session behavior plus the
product embeddings of the items viewed (mean-pooled).

Input:  sessions.events (session_id, sku, event, dwell_s),
        features.product_embeddings (sku, vector[128])
Output: scores.p_cvr (session_id, p)
"""
import numpy as np

DIM = 128


def session_features(events, product_embeddings):
    views = [e for e in events if e["event"] == "view"]
    add_to_cart = sum(e["event"] == "add_to_cart" for e in events)
    dwell = sum(e.get("dwell_s", 0) for e in views)
    vecs = [product_embeddings[e["sku"]] for e in views if e["sku"] in product_embeddings]
    pooled = np.mean(vecs, axis=0) if vecs else np.zeros(DIM)
    return np.concatenate([[len(views), add_to_cart, np.log1p(dwell)], pooled])


def p_cvr(features, weights, bias=-3.0):
    return float(1 / (1 + np.exp(-(features @ weights + bias))))


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    embs = {"BED-001": rng.normal(size=DIM), "BED-003": rng.normal(size=DIM)}
    events = [
        {"sku": "BED-001", "event": "view", "dwell_s": 40},
        {"sku": "BED-003", "event": "view", "dwell_s": 15},
        {"sku": "BED-001", "event": "add_to_cart"},
    ]
    w = np.zeros(3 + DIM)
    w[:3] = [0.3, 1.5, 0.2]
    print(round(p_cvr(session_features(events, embs), w), 3))

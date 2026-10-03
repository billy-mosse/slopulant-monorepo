"""Return risk — p(return).

At checkout, estimate the probability an order line gets returned so we can
show fit/material guidance or route to stricter QA.

Features: who is buying (customer vector) x what they buy (product vector),
plus material/category tags, which drive most returns (linen wrinkles, sizes run small).

Input:  features.customer_embeddings (customer_id, vector),
        features.product_embeddings (sku, vector[128]),
        catalog.validated_tags (sku, tag)
Output: scores.p_return (order_line_id, p)
"""
import numpy as np

RISKY_TAGS = {"linen": 0.8, "duvet": 0.4, "pillow": 0.2}


def features(customer_vec, product_vec, tags):
    tag_risk = sum(RISKY_TAGS.get(t, 0.0) for t in tags)
    affinity = float(np.dot(customer_vec[:3], product_vec[:3]))
    return np.array([1.0, tag_risk, affinity, customer_vec[0]])


def p_return(x, w):
    return float(1 / (1 + np.exp(-(x @ w))))


def score_lines(lines, customer_embeddings, product_embeddings, tags_by_sku, w):
    out = {}
    for line in lines:
        x = features(
            customer_embeddings[line["customer_id"]],
            product_embeddings[line["sku"]],
            tags_by_sku.get(line["sku"], []),
        )
        out[line["order_line_id"]] = round(p_return(x, w), 3)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    custs = {"C1": rng.normal(size=8)}
    prods = {"BED-001": rng.normal(size=128)}
    lines = [{"order_line_id": 1, "customer_id": "C1", "sku": "BED-001"}]
    w = np.array([-2.0, 1.2, 0.3, -0.5])
    print(score_lines(lines, custs, prods, {"BED-001": ["bedding", "linen"]}, w))

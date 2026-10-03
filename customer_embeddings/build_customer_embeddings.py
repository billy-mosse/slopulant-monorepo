"""Customer embeddings.

One vector per customer summarizing taste and value: recency/frequency/
monetary stats plus a decayed average of the categories they've bought.

Input:  orders.lines (customer_id, sku, category, amount, days_ago)
Output: features.customer_embeddings (customer_id, vector)
"""
import numpy as np

CATEGORIES = ["bedding", "bath", "decor", "kitchen", "furniture"]
HALF_LIFE_DAYS = 90


def customer_vector(lines):
    cat = np.zeros(len(CATEGORIES))
    for l in lines:
        if l["category"] in CATEGORIES:
            cat[CATEGORIES.index(l["category"])] += 0.5 ** (l["days_ago"] / HALF_LIFE_DAYS)
    if cat.sum():
        cat /= cat.sum()
    recency = min(l["days_ago"] for l in lines)
    frequency = len(lines)
    monetary = sum(l["amount"] for l in lines)
    rfm = np.array([np.exp(-recency / 30), np.log1p(frequency), np.log1p(monetary)])
    return np.concatenate([rfm, cat])


def build(order_lines):
    by_customer = {}
    for l in order_lines:
        by_customer.setdefault(l["customer_id"], []).append(l)
    return {c: customer_vector(ls) for c, ls in by_customer.items()}


if __name__ == "__main__":
    lines = [
        {"customer_id": "C1", "sku": "BED-001", "category": "bedding", "amount": 189.0, "days_ago": 10},
        {"customer_id": "C1", "sku": "BTH-010", "category": "bath", "amount": 34.0, "days_ago": 200},
    ]
    print({k: v.round(3).tolist() for k, v in build(lines).items()})

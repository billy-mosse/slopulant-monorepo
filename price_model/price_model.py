"""Price model.

Suggests a list price for new products from category tags and material,
fit with least squares on historical sell-through prices.

Input:  catalog.products (sku, price), catalog.raw_tags (sku, tag)
Output: pricing.suggested_price (sku, price)
"""
import numpy as np

FEATURES = ["bedding", "bath", "linen", "cotton", "towel", "duvet", "pillow"]


def clean_tag(tag):
    t = tag.strip().lower()
    return {"linens": "linen", "towels": "towel", "bed": "bedding", "pillows": "pillow"}.get(t, t)


def featurize(tags):
    tags = {clean_tag(t) for t in tags}
    return np.array([1.0] + [1.0 if f in tags else 0.0 for f in FEATURES])


def fit(rows):
    X = np.stack([featurize(r["tags"]) for r in rows])
    y = np.array([r["price"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict(coef, tags):
    return float(featurize(tags) @ coef)


if __name__ == "__main__":
    history = [
        {"tags": ["Bed", "Linens", "duvet"], "price": 189.0},
        {"tags": ["bed", "cotton", "duvet"], "price": 129.0},
        {"tags": ["bath", "Towels", "cotton"], "price": 34.0},
        {"tags": ["bed", "linen", "Pillows"], "price": 59.0},
    ]
    coef = fit(history)
    print(round(predict(coef, ["bed", "linen", "duvet"]), 2))

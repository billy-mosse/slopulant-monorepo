"""Review summaries.

Produces a short "what customers say" blurb per product from its reviews:
average rating plus the most frequently mentioned aspects.

Input:  reviews.approved (sku, rating, text)
Output: reviews.summaries (sku, avg_rating, summary)
"""
import re
from collections import Counter

ASPECTS = {"soft", "scratchy", "color", "size", "shrink", "warm", "quality", "price", "pilling"}


def clean_text(text):
    text = re.sub(r"http\S+", "", text.lower())
    return re.sub(r"[^a-z\s]", " ", text)


def summarize(reviews):
    by_sku = {}
    for r in reviews:
        by_sku.setdefault(r["sku"], []).append(r)
    out = {}
    for sku, rs in by_sku.items():
        avg = sum(r["rating"] for r in rs) / len(rs)
        counts = Counter(w for r in rs for w in clean_text(r["text"]).split() if w in ASPECTS)
        top = ", ".join(a for a, _ in counts.most_common(3)) or "n/a"
        out[sku] = {"avg_rating": round(avg, 2), "summary": f"Customers mention: {top}"}
    return out


if __name__ == "__main__":
    reviews = [
        {"sku": "BED-001", "rating": 5, "text": "So soft, great color!"},
        {"sku": "BED-001", "rating": 3, "text": "Soft but did shrink after washing"},
    ]
    print(summarize(reviews))

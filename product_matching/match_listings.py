"""Product matching (catalog cleanup).

Finds listings in the catalog that are the same physical product uploaded
twice by merchants, so they can be merged into one SKU.

Input:  catalog.products (sku, title, brand, price)
Output: catalog.merge_candidates (sku_a, sku_b, score)
"""
from difflib import SequenceMatcher
from itertools import combinations

THRESHOLD = 0.85


def clean_title(title):
    return " ".join(title.lower().replace("-", " ").split())


def listing_similarity(a, b):
    if a["brand"] != b["brand"]:
        return 0.0
    title_sim = SequenceMatcher(None, clean_title(a["title"]), clean_title(b["title"])).ratio()
    price_ratio = min(a["price"], b["price"]) / max(a["price"], b["price"])
    return 0.8 * title_sim + 0.2 * price_ratio


def find_merge_candidates(products):
    out = []
    for a, b in combinations(products, 2):
        score = listing_similarity(a, b)
        if score >= THRESHOLD:
            out.append((a["sku"], b["sku"], round(score, 3)))
    return out


if __name__ == "__main__":
    products = [
        {"sku": "BED-001", "title": "Queen Linen Duvet Cover - Oat", "brand": "Hearth", "price": 189.0},
        {"sku": "BED-002", "title": "Queen linen duvet cover oat", "brand": "Hearth", "price": 179.0},
        {"sku": "BTH-010", "title": "Turkish Cotton Bath Towel", "brand": "Loom", "price": 34.0},
    ]
    print(find_merge_candidates(products))

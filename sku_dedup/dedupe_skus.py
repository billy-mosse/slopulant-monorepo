"""SKU dedup for the search index.

Search results show the same item twice when a merchant re-uploads it.
Collapse near-identical SKUs before indexing.

Input:  catalog.products (sku, title, brand, price)
Output: search.sku_groups (canonical_sku, sku)
"""
import re


def norm(title):
    return set(re.sub(r"[^a-z0-9 ]", " ", title.lower()).split())


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def group_skus(products, min_sim=0.8):
    groups = {}
    for p in products:
        canonical = p["sku"]
        for q in products:
            if q["sku"] in groups and q["brand"] == p["brand"]:
                if jaccard(norm(p["title"]), norm(q["title"])) >= min_sim:
                    canonical = groups[q["sku"]]
                    break
        groups[p["sku"]] = canonical
    return groups


if __name__ == "__main__":
    products = [
        {"sku": "BED-001", "title": "Queen Linen Duvet Cover - Oat", "brand": "Hearth", "price": 189.0},
        {"sku": "BED-002", "title": "Queen linen duvet cover oat", "brand": "Hearth", "price": 179.0},
        {"sku": "BTH-010", "title": "Turkish Cotton Bath Towel", "brand": "Loom", "price": 34.0},
    ]
    print(group_skus(products))

"""Listing quality score.

Gives merchants a 0-100 score for each listing so we can nudge them to fix
weak ones: short titles, missing descriptions, no images, messy tags.

Input:  catalog.products (sku, title, description, n_images), catalog.raw_tags (sku, tag)
Output: merchant.listing_quality (sku, score, issues)
"""
ALLOWED_TAGS = {"bedding", "bath", "linen", "cotton", "towel", "duvet", "pillow", "decor", "kitchen"}
TAG_FIXES = {"linens": "linen", "towels": "towel", "pillows": "pillow", "bed": "bedding", "home-decor": "decor"}


def canonical_tag(tag):
    t = tag.strip().lower().replace(" ", "-")
    return TAG_FIXES.get(t, t)


def score_listing(product, tags):
    issues = []
    if len(product["title"].split()) < 4:
        issues.append("short_title")
    if not product.get("description"):
        issues.append("no_description")
    if product.get("n_images", 0) < 3:
        issues.append("few_images")
    good_tags = {canonical_tag(t) for t in tags} & ALLOWED_TAGS
    if len(good_tags) < 2:
        issues.append("weak_tags")
    return max(0, 100 - 25 * len(issues)), issues


def run(products, raw_tags):
    tags_by_sku = {}
    for sku, tag in raw_tags:
        tags_by_sku.setdefault(sku, []).append(tag)
    return {p["sku"]: score_listing(p, tags_by_sku.get(p["sku"], [])) for p in products}


if __name__ == "__main__":
    products = [
        {"sku": "BED-001", "title": "Queen Linen Duvet Cover - Oat", "description": "Stonewashed", "n_images": 5},
        {"sku": "BTH-010", "title": "Towel", "description": "", "n_images": 1},
    ]
    print(run(products, [("BED-001", "Linens"), ("BED-001", "Bed"), ("BTH-010", "Towels")]))

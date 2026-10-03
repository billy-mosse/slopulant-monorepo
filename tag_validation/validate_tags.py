"""Tag validation.

Merchants tag products by hand and the tags are a mess ("Linen", "linens",
"LINEN-blend"). This normalizes tags onto the controlled vocabulary and drops
anything we can't map.

Input:  catalog.raw_tags (sku, tag)
Output: catalog.validated_tags (sku, tag)
"""
VOCAB = {"bedding", "bath", "linen", "cotton", "towel", "duvet", "pillow", "decor", "kitchen"}

ALIASES = {
    "linens": "linen",
    "linen-blend": "linen",
    "towels": "towel",
    "pillows": "pillow",
    "bed": "bedding",
    "home-decor": "decor",
}


def normalize(tag):
    t = tag.strip().lower().replace(" ", "-")
    return ALIASES.get(t, t)


def validate(raw_tags):
    out = {}
    for sku, tag in raw_tags:
        t = normalize(tag)
        if t in VOCAB:
            out.setdefault(sku, set()).add(t)
    return {sku: sorted(tags) for sku, tags in out.items()}


if __name__ == "__main__":
    raw = [("BED-001", "Linen"), ("BED-001", "Bed"), ("BED-001", "sale!!"), ("BTH-010", "Towels")]
    print(validate(raw))

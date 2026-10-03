"""
make_titles.py -- generate <title> strings for product detail pages.

Content team tool. Google shows roughly 60-65 characters of a page title, so we
build titles from the parts shoppers actually search on (product type, material,
size, colour, brand) in a fixed priority order and stop before we run out of room.
Promotional junk from merchant-entered names ("SALE!!", "Best Seller", "NEW") is
removed first because it wastes characters and search engines ignore it.

Sources:  catalog.products, catalog.product_attributes
Target:   content.pdp_titles
"""
import logging
import re
import sys
from datetime import datetime, timezone

import click
import pandas as pd
import sqlalchemy as sa

MAX_LEN = 65
SITE_SUFFIX = " | Slopulent Living"

# Order in which we try to fit attribute fragments into the title.
# Earlier = more search value. "product_type" is always kept.
KEYWORD_PRIORITY = [
    "product_type",
    "material",
    "size",
    "thread_count",
    "color",
    "pattern",
    "brand",
    "collection",
]

PROMO_PATTERNS = [
    r"\b(?:on\s+)?sale\b[!]*",
    r"\bbest[\s-]?seller\b",
    r"\bnew(?:\s+arrival)?\b[!]*",
    r"\blimited(?:\s+time)?(?:\s+offer)?\b",
    r"\b\d{1,2}\s?%\s?off\b",
    r"\bfree\s+shipping\b",
    r"\bhot\s+deal\b",
    r"\bclearance\b",
    r"[!*]{2,}",
    r"[☀-➿\U0001F300-\U0001FAFF]",  # emoji / dingbats
]
PROMO_RE = re.compile("|".join(PROMO_PATTERNS), re.IGNORECASE)

SIZE_ALIASES = {
    "cal king": "California King", "california king": "California King",
    "king": "King", "queen": "Queen", "full": "Full", "twin xl": "Twin XL", "twin": "Twin",
}

PRODUCTS_SQL = "SELECT sku, name, product_type, brand, is_active FROM catalog.products WHERE is_active"
ATTRS_SQL = "SELECT sku, attr_name, attr_value FROM catalog.product_attributes"
OUT_TABLE = "content.pdp_titles"

log = logging.getLogger("seo_titles")


def strip_promo(text):
    if not isinstance(text, str):
        return ""
    cleaned = PROMO_RE.sub(" ", text)
    cleaned = re.sub(r"\s*[-|:,]\s*(?=[-|:,]|$)", "", cleaned)   # dangling separators
    return re.sub(r"\s{2,}", " ", cleaned).strip(" -|:,")


def normalize_size(value):
    v = value.strip().lower()
    for alias in sorted(SIZE_ALIASES, key=len, reverse=True):  # longest first: "twin xl" before "twin"
        if re.search(rf"\b{re.escape(alias)}\b", v):
            return SIZE_ALIASES[alias]
    return value.strip()


def normalize_fragment(attr, value):
    value = strip_promo(str(value))
    if not value or value.lower() in {"n/a", "none", "other", "-"}:
        return None
    if attr == "size":
        return normalize_size(value)
    if attr == "thread_count":
        digits = re.sub(r"\D", "", value)
        return f"{digits} Thread Count" if digits else None
    if attr == "brand":
        return value  # keep merchant casing for brands
    return value.title()


def pivot_attributes(attrs):
    attrs = attrs[attrs["attr_name"].isin(KEYWORD_PRIORITY)]
    wide = (attrs.drop_duplicates(["sku", "attr_name"], keep="last")
                 .pivot(index="sku", columns="attr_name", values="attr_value"))
    return wide.reset_index()


def compose_title(row, budget=MAX_LEN):
    """Greedily add fragments in priority order while the result fits the budget."""
    parts = {}
    for attr in KEYWORD_PRIORITY:
        raw = row.get(attr)
        if pd.isna(raw):
            continue
        frag = normalize_fragment(attr, raw)
        if frag:
            parts[attr] = frag

    if "product_type" not in parts:
        parts["product_type"] = strip_promo(row.get("name", "")) or "Home Goods"

    # Reading order differs from priority order: "Brand Material Color Type - Size"
    head_order = ["brand", "collection", "material", "thread_count", "color", "pattern", "product_type"]
    chosen = set()
    title = ""
    for attr in KEYWORD_PRIORITY:
        if attr not in parts:
            continue
        trial = chosen | {attr}
        candidate = render(parts, trial, head_order)
        if len(candidate) <= budget:
            chosen, title = trial, candidate
    if not title:  # even product_type alone is too long
        title = parts["product_type"][: budget - 1].rsplit(" ", 1)[0]
    if len(title) + len(SITE_SUFFIX) <= budget:
        title += SITE_SUFFIX
    return title


def render(parts, chosen, head_order):
    head = " ".join(parts[a] for a in head_order if a in chosen)
    if "size" in chosen:
        head = f"{head} - {parts['size']}"
    # collapse repeated words, e.g. brand "Cotton Co" + material "Cotton"
    words, seen = [], set()
    for w in head.split():
        key = w.lower()
        if key in seen and key.isalpha():
            continue
        seen.add(key)
        words.append(w)
    return " ".join(words)


@click.command()
@click.option("--dsn", envvar="WAREHOUSE_URL", required=True)
@click.option("--sku", multiple=True, help="Only regenerate these SKUs.")
@click.option("--preview", is_flag=True, help="Print instead of writing.")
def cli(dsn, sku, preview):
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    engine = sa.create_engine(dsn)
    products = pd.read_sql(PRODUCTS_SQL, engine)
    attrs = pd.read_sql(ATTRS_SQL, engine)
    if sku:
        products = products[products["sku"].isin(sku)]

    df = products.merge(pivot_attributes(attrs), on="sku", how="left", suffixes=("", "_attr"))
    df["product_type"] = df["product_type_attr"].fillna(df["product_type"]) if "product_type_attr" in df else df["product_type"]
    df["brand"] = df["brand_attr"].fillna(df["brand"]) if "brand_attr" in df else df["brand"]

    df["seo_title"] = df.apply(lambda r: compose_title(r.to_dict()), axis=1)
    df["title_length"] = df["seo_title"].str.len()
    df["generated_at"] = datetime.now(timezone.utc)
    out = df[["sku", "seo_title", "title_length", "generated_at"]]

    log.info("built %d titles, mean length %.1f, %d over limit",
             len(out), out.title_length.mean(), (out.title_length > MAX_LEN).sum())
    if preview:
        click.echo(out.head(40).to_string(index=False))
        return
    schema, table = OUT_TABLE.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="replace", index=False)


if __name__ == "__main__":
    cli()

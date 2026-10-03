"""Generate SEO product titles from structured attributes and colors.

Input tables:
    catalog.products            sku, title, brand, product_type, category_l2
    catalog.product_attributes  sku, material, size, dimensions, thread_count, pattern
                                plus <attr>_confidence columns
    catalog.product_colors      sku, color_1, share_1

Output table:
    catalog.seo_titles          sku, original_title, seo_title, template, kept_original, reason

Titles are rendered from per-category templates (see ``templates.py``),
cleaned of merchant noise, de-duplicated, title-cased and trimmed to 70
characters by dropping optional slots in priority order.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine

from templates import MAX_TITLE_CHARS, TitleTemplate, template_for

log = logging.getLogger(__name__)

PRODUCTS = "catalog.products"
ATTRIBUTES = "catalog.product_attributes"
COLORS = "catalog.product_colors"
OUTPUT = "catalog.seo_titles"

MIN_CONFIDENCE = 0.75
MIN_COLOR_SHARE = 0.30

NOISE_RE = re.compile(
    r"(!{2,}|\bfree\s+shipping\b|\bsale\b|\bclearance\b|\bbest\s+seller\b|\bhot\s+deal\b|\blimited\s+time\b|\bnew!?\b)",
    re.IGNORECASE,
)
SMALL_WORDS = {"a", "an", "and", "the", "of", "in", "with", "for", "x", "by"}
UNITS_RE = re.compile(r"^\d+(\.\d+)?(cm|in|tc)?$", re.IGNORECASE)


@dataclass
class RenderResult:
    """Final title, whether we fell back to the merchant title, and why ('ok' if not)."""

    title: str
    kept_original: bool
    reason: str


def strip_noise(text: str) -> str:
    """Remove merchant noise like '!!!', 'SALE', 'free shipping'; collapse whitespace."""
    text = NOISE_RE.sub(" ", text or "")
    text = re.sub(r"\s*([,|-])\s*(?=[,|-]|$)", "", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,-|")


def dedupe_words(text: str) -> str:
    """Drop repeated words (case-insensitive), keeping the first occurrence and punctuation."""
    seen: set[str] = set()
    out = []
    for tok in text.split():
        key = re.sub(r"\W", "", tok).lower()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(tok)
    return " ".join(out)


def smart_title_case(text: str) -> str:
    """Title-case while keeping small words lower and units like '400TC' or '90x92in' intact."""
    words = text.split()
    for i, w in enumerate(words):
        core = w.strip(",")
        if UNITS_RE.match(core) or re.search(r"\d", core):
            continue
        if i > 0 and core.lower() in SMALL_WORDS:
            words[i] = w.lower()
        else:
            words[i] = w[:1].upper() + w[1:].lower()
    return " ".join(words)


def format_slot(name: str, value: object) -> str | None:
    """Turn a raw attribute value into display text; None means 'slot empty'."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    if name == "thread_count":
        return f"{int(value)} Thread Count"
    if name == "material" and isinstance(value, str) and value.startswith("{"):
        blend = json.loads(value)
        top = sorted(blend.items(), key=lambda kv: -(kv[1] or 0))
        return "-".join(k for k, _ in top[:2]) + (" Blend" if len(top) > 1 else "")
    if name == "dimensions" and isinstance(value, str) and value.startswith("{"):
        d = json.loads(value)
        return f"{round(d['width_cm'])}x{round(d['length_cm'])}cm"
    return str(value)


def render(tpl: TitleTemplate, slots: dict[str, str | None]) -> str:
    """Fill the template, then drop optional slots in priority order until it fits."""
    active = {k: v for k, v in slots.items() if v}
    dropped = list(tpl.drop_order)
    while True:
        text = tpl.pattern
        for s in tpl.slots:
            text = text.replace(f"{{{s}}}", active.get(s, ""))
        text = smart_title_case(dedupe_words(strip_noise(re.sub(r"\s+", " ", text))))
        if len(text) <= MAX_TITLE_CHARS or not dropped:
            return text[:MAX_TITLE_CHARS].rstrip(" ,")
        active.pop(dropped.pop(0), None)


def rewrite_one(row: pd.Series) -> RenderResult:
    """Build slot values for one product and render, or keep the original title."""
    original = strip_noise(row["title"])
    tpl = template_for(row.get("category_l2"))
    slots: dict[str, str | None] = {}
    for name in tpl.slots:
        conf = row.get(f"{name}_confidence")
        if conf is not None and not pd.isna(conf) and conf < MIN_CONFIDENCE:
            slots[name] = None
            continue
        slots[name] = format_slot(name, row.get(name))
    if row.get("share_1", 0) and row["share_1"] >= MIN_COLOR_SHARE:
        slots["color"] = format_slot("color", row.get("color_1"))
    else:
        slots["color"] = None

    missing = [r for r in tpl.required if not slots.get(r)]
    if missing:
        return RenderResult(original, True, "missing:" + ",".join(missing))
    return RenderResult(render(tpl, slots), False, "ok")


def load(engine) -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM {PRODUCTS}
        LEFT JOIN {ATTRIBUTES} USING (sku)
        LEFT JOIN {COLORS} USING (sku)
        WHERE is_active
    """
    return pd.read_sql(sql, engine)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite product titles for SEO.")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    engine = create_engine(args.dsn)
    df = load(engine)
    log.info("loaded %d products", len(df))

    results = [rewrite_one(r) for _, r in df.iterrows()]
    out = pd.DataFrame({
        "sku": df["sku"],
        "original_title": df["title"],
        "seo_title": [r.title for r in results],
        "template": df["category_l2"].fillna("default"),
        "kept_original": [r.kept_original for r in results],
        "reason": [r.reason for r in results],
    })
    log.info("rewrote %.1f%% of titles", 100 * (~out["kept_original"]).mean())

    if args.dry_run:
        print(out.sample(min(25, len(out))).to_string(index=False))
        return
    schema, table = OUTPUT.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    log.info("wrote %d rows to %s", len(out), OUTPUT)


if __name__ == "__main__":
    main()

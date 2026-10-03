"""Extract structured attributes from product titles and descriptions.

Reads the input table named in ``config.yaml`` (catalog.products), runs a set
of regex extractors and gazetteer lookups over ``title`` and ``description``,
and writes one row per SKU to catalog.product_attributes.
Values below the per-attribute confidence threshold are written as NULL.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy import create_engine

from gazetteers import CARE_GAZ, MATERIAL_GAZ, PATTERN_GAZ, SIZE_GAZ

log = logging.getLogger(__name__)

IN_TO_CM = 2.54

BLEND_RE = re.compile(
    r"(?P<pct>\d{1,3})\s*%\s*(?P<mat>[a-z][a-z \-]{1,24}?)(?=\s*(?:\d{1,3}\s*%|[,/;&]|\band\b|$))",
    re.IGNORECASE,
)
DIM_RE = re.compile(
    r"(?P<w>\d{1,3}(?:\.\d+)?)\s*(?:\"|in\.?|inches)?\s*[x×]\s*"
    r"(?P<l>\d{1,3}(?:\.\d+)?)\s*(?P<unit>\"|in\.?|inches|cm)\b",
    re.IGNORECASE,
)
THREAD_RE = re.compile(r"\b(?P<tc>[1-9]\d{1,3})\s*(?:-|\s)?(?:thread[\s\-]?count|tc)\b", re.IGNORECASE)


@dataclass
class Extracted:
    """One attribute value with confidence in [0, 1] and its source field."""
    value: object
    confidence: float
    source: str


def _clean(text: str | None) -> str:
    """Collapse whitespace and strip HTML tags merchants paste into descriptions."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def extract_material(title: str, desc: str) -> Extracted | None:
    """'55% linen 45% cotton' -> {'linen': 55, 'cotton': 45}; blends beat bare mentions."""
    for src, text in (("title", title), ("description", desc)):
        blend: dict[str, int] = {}
        for m in BLEND_RE.finditer(text):
            canon = MATERIAL_GAZ.canonical(m.group("mat"))
            if canon:
                blend[canon] = blend.get(canon, 0) + int(m.group("pct"))
        if blend:
            conf = 0.95 if sum(blend.values()) == 100 else 0.7
            return Extracted(blend, conf, src)
    for src, text, conf in (("title", title, 0.85), ("description", desc, 0.65)):
        hits = MATERIAL_GAZ.find_all(text)
        if hits:
            canon = {h[0] for h in hits}
            return Extracted({c: None for c in sorted(canon)}, conf if len(canon) == 1 else conf - 0.15, src)
    return None


def _first_gazetteer_hit(gaz, title: str, desc: str, confs: tuple[float, float]) -> Extracted | None:
    """First canonical hit, title before description; conflicting values halve confidence."""
    for src, text, conf in (("title", title, confs[0]), ("description", desc, confs[1])):
        hits = gaz.find_all(text)
        if hits:
            return Extracted(hits[0][0], conf if len({h[0] for h in hits}) == 1 else conf / 2, src)
    return None


def extract_size(title: str, desc: str) -> Extracted | None:
    return _first_gazetteer_hit(SIZE_GAZ, title, desc, (0.9, 0.7))


def extract_dimensions(title: str, desc: str) -> Extracted | None:
    """'90 x 92 in' -> {'width_cm': 228.6, 'length_cm': 233.7}."""
    for src, text in (("title", title), ("description", desc)):
        m = DIM_RE.search(text)
        if m:
            w, l = float(m.group("w")), float(m.group("l"))
            factor = 1.0 if m.group("unit").lower() == "cm" else IN_TO_CM
            return Extracted({"width_cm": round(w * factor, 1), "length_cm": round(l * factor, 1)}, 0.9, src)
    return None


def extract_thread_count(title: str, desc: str) -> Extracted | None:
    """Thread count; values outside 150-1500 are almost always marketing ply math."""
    for src, text in (("title", title), ("description", desc)):
        m = THREAD_RE.search(text)
        if m:
            tc = int(m.group("tc"))
            return Extracted(tc, 0.95 if 150 <= tc <= 1500 else 0.4, src)
    return None


def extract_care(_: str, desc: str) -> Extracted | None:
    """Care is only reliably stated in descriptions."""
    labels = sorted({h[0] for h in CARE_GAZ.find_all(desc)})
    if not labels:
        return None
    conflict = {"dry_clean_only", "machine_wash_cold"} <= set(labels)
    return Extracted(labels, 0.4 if conflict else 0.8, "description")


def extract_pattern(title: str, desc: str) -> Extracted | None:
    return _first_gazetteer_hit(PATTERN_GAZ, title, desc, (0.85, 0.6))


EXTRACTORS = {"material": extract_material, "size": extract_size, "dimensions": extract_dimensions,
              "thread_count": extract_thread_count, "care": extract_care, "pattern": extract_pattern}


def extract_row(sku: str, title: str, desc: str, thresholds: dict[str, float]) -> dict:
    """Run every extractor and flatten into an output row, nulling low-confidence values."""
    title, desc = _clean(title), _clean(desc)
    row: dict = {"sku": sku}
    for name, fn in EXTRACTORS.items():
        ex = fn(title, desc)
        keep = ex is not None and ex.confidence >= thresholds.get(name, 0.5)
        value = ex.value if keep else None  # never persist guesses
        row[name] = json.dumps(value) if isinstance(value, (dict, list)) else value
        row[f"{name}_confidence"] = round(ex.confidence, 2) if ex else None
        row[f"{name}_source"] = ex.source if ex else None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dsn", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    sql = f"SELECT {', '.join(cfg['input']['columns'])} FROM {cfg['input']['table']} WHERE {cfg['input']['where']}"
    engine = create_engine(args.dsn)

    out_frames = []
    for chunk in pd.read_sql(sql, engine, chunksize=cfg["batch_size"]):
        out_frames.append(pd.DataFrame([extract_row(r.sku, r.title, r.description, cfg["thresholds"])
                                        for r in chunk.itertuples()]))
        log.info("processed %d products", sum(len(f) for f in out_frames))

    out = pd.concat(out_frames, ignore_index=True)
    out["run_date"] = pd.Timestamp.utcnow().date()
    log.info("attribute coverage: %s", {k: f"{out[k].notna().mean():.1%}" for k in EXTRACTORS})

    schema, table = cfg["output"]["table"].split(".")
    out.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    log.info("wrote %d rows to %s", len(out), cfg["output"]["table"])


if __name__ == "__main__":
    main()

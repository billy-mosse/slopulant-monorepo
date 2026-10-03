"""Category and attribute intent for corrected queries."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

CLICK_CATEGORY_SQL = """
SELECT ql.query_text, pc.category_id, COUNT(*) AS clicks
FROM search.query_logs ql
JOIN catalog.predicted_category pc ON pc.sku = ql.clicked_sku
WHERE ql.clicked_sku IS NOT NULL
  AND ql.event_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY ql.query_text, pc.category_id
"""

PRICE_RES = [
    (re.compile(r"\b(?:under|below|less than)\s*\$?(\d+)"), "max_price"),
    (re.compile(r"\b(?:over|above|more than)\s*\$?(\d+)"), "min_price"),
    (re.compile(r"\$(\d+)\s*-\s*\$?(\d+)"), "range"),
]
ATTRIBUTES = {
    "size": ["twin xl", "twin", "full", "queen", "california king", "king"],
    "material": ["linen", "cotton", "percale", "sateen", "bamboo", "silk", "wool", "velvet", "jute"],
}
TOP_K = 3
MIN_SHARE = 0.05


@dataclass
class QueryIntent:
    query: str
    corrected: str
    categories: list[tuple[str, float]] = field(default_factory=list)
    filters: dict[str, object] = field(default_factory=dict)


def category_distribution(clicks: pd.DataFrame, prior: float = 1.0) -> dict[str, list[tuple[str, float]]]:
    dist: dict[str, list[tuple[str, float]]] = {}
    for query, grp in clicks.groupby("query_text"):
        w = grp.set_index("category_id")["clicks"].astype(float) + prior
        w = (w / w.sum()).sort_values(ascending=False)
        dist[query] = [(c, round(p, 4)) for c, p in w.items() if p >= MIN_SHARE][:TOP_K]
    return dist


def price_filters(q: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for rx, kind in PRICE_RES:
        m = rx.search(q)
        if not m:
            continue
        if kind == "range":
            out["min_price"], out["max_price"] = float(m.group(1)), float(m.group(2))
        else:
            out[kind] = float(m.group(1))
    return out


def attribute_filters(q: str) -> dict[str, str]:
    out = {}
    for attr, values in ATTRIBUTES.items():
        for v in values:
            if re.search(rf"\b{re.escape(v)}\b", q):
                out[attr] = v
                break
    return out


def strip_filters(q: str) -> str:
    for rx, _ in PRICE_RES:
        q = rx.sub("", q)
    return re.sub(r"\s+", " ", q).strip()


class IntentResolver:
    def __init__(self, dist: dict[str, list[tuple[str, float]]]):
        self.dist = dist
        self.backoff = defaultdict(list)
        for q, cats in dist.items():
            for tok in q.split():
                self.backoff[tok].extend(cats)

    def categories(self, q: str) -> list[tuple[str, float]]:
        if q in self.dist:
            return self.dist[q]
        agg: dict[str, float] = defaultdict(float)
        for tok in q.split():
            for c, p in self.backoff.get(tok, []):
                agg[c] += p
        total = sum(agg.values()) or 1.0
        return sorted(((c, round(v / total, 4)) for c, v in agg.items()), key=lambda x: -x[1])[:TOP_K]

    def resolve(self, raw: str, corrected: str) -> QueryIntent:
        filters = {**price_filters(corrected), **attribute_filters(corrected)}
        return QueryIntent(raw, corrected, self.categories(strip_filters(corrected)), filters)

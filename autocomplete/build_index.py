"""Build the search-box autocomplete index from search.query_logs into search.autocomplete_index."""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("autocomplete")

SOURCE_TABLE = "search.query_logs"
TARGET_TABLE = "search.autocomplete_index"
HALF_LIFE_DAYS = 14.0
TOP_N = 8
MIN_DECAYED_COUNT = 3.0
SUCCESS_WEIGHT = 0.6
MAX_PREFIX_LEN = 20

BLOCKLIST = {"fuck", "shit", "bitch", "cunt", "porn", "nsfw", "dick", "pussy", "nazi", "slut"}
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "$": "s", "@": "a"})

QUERY_SQL = f"""
SELECT LOWER(TRIM(query_text)) AS query_text,
       event_date,
       COUNT(*) AS searches,
       SUM(CASE WHEN clicked_sku IS NOT NULL THEN 1 ELSE 0 END) AS clicks
FROM {SOURCE_TABLE}
WHERE event_date >= CURRENT_DATE - INTERVAL '120 days'
GROUP BY 1, 2
"""


@dataclass(frozen=True)
class Suggestion:
    text: str
    score: float


def normalize(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 '&\-]", " ", q.lower())).strip()


def is_profane(q: str) -> bool:
    toks = set(q.translate(LEET).split())
    squashed = q.translate(LEET).replace(" ", "")
    return bool(toks & BLOCKLIST) or any(b in squashed for b in BLOCKLIST if len(b) > 4)


def singular(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("ches", "shes", "sses", "xes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us")) and len(tok) > 3:
        return tok[:-1]
    return tok


def dedup_key(q: str) -> str:
    return " ".join(singular(t) for t in q.replace("-", " ").split())


def decay(age_days: float, half_life: float = HALF_LIFE_DAYS) -> float:
    return math.exp(-math.log(2) * age_days / half_life)


def score_queries(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    df = df.assign(query_text=df.query_text.map(normalize))
    df = df[df.query_text.str.len().between(2, 60) & ~df.query_text.map(is_profane)]
    age = (pd.Timestamp(as_of) - pd.to_datetime(df.event_date)).dt.days.clip(lower=0)
    w = age.map(decay)
    df = df.assign(w_searches=df.searches * w, w_clicks=df.clicks * w)
    agg = df.groupby("query_text", as_index=False)[["w_searches", "w_clicks"]].sum()
    agg = agg[agg.w_searches >= MIN_DECAYED_COUNT]
    success = (agg.w_clicks + 1.0) / (agg.w_searches + 2.0)
    agg["score"] = agg.w_searches.map(math.log1p) * ((1 - SUCCESS_WEIGHT) + SUCCESS_WEIGHT * success)
    return collapse_variants(agg)


def collapse_variants(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.assign(key=agg.query_text.map(dedup_key))
    best = agg.sort_values("score", ascending=False).drop_duplicates("key")
    pooled = agg.groupby("key").score.sum()
    return best.assign(score=best.key.map(pooled))[["query_text", "score"]]


class PrefixTrie:
    def __init__(self, top_n: int = TOP_N):
        self.root: dict = {}
        self.top_n = top_n

    def insert(self, s: Suggestion) -> None:
        node = self.root
        for ch in s.text[:MAX_PREFIX_LEN]:
            node = node.setdefault(ch, {})
            top = node.setdefault("$top", [])
            top.append(s)
            if len(top) > self.top_n * 2:
                node["$top"] = sorted(top, key=lambda x: -x.score)[: self.top_n]

    def finalize(self, node: dict | None = None) -> None:
        node = self.root if node is None else node
        if "$top" in node:
            node["$top"] = sorted(node["$top"], key=lambda x: -x.score)[: self.top_n]
        for k, child in node.items():
            if k != "$top":
                self.finalize(child)

    def lookup(self, prefix: str) -> list[Suggestion]:
        node = self.root
        for ch in normalize(prefix):
            if ch not in node:
                return []
            node = node[ch]
        return node.get("$top", [])

    def rows(self, node: dict | None = None, prefix: str = ""):
        node = self.root if node is None else node
        for k, child in node.items():
            if k == "$top":
                continue
            p = prefix + k
            yield p, [(s.text, round(s.score, 4)) for s in child.get("$top", [])]
            yield from self.rows(child, p)


def build(df: pd.DataFrame, as_of: date) -> PrefixTrie:
    trie = PrefixTrie()
    for row in score_queries(df, as_of).itertuples():
        trie.insert(Suggestion(row.query_text, row.score))
    trie.finalize()
    return trie


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--as-of", default=date.today().isoformat())
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    engine = create_engine(a.dsn)
    as_of = datetime.fromisoformat(a.as_of).date()
    trie = build(pd.read_sql(QUERY_SQL, engine), as_of)
    out = pd.DataFrame(
        [{"prefix": p, "suggestions": json.dumps(s), "built_at": as_of} for p, s in trie.rows()]
    )
    log.info("%d prefixes; sample 'duv' -> %s", len(out), [s.text for s in trie.lookup("duv")])
    schema, table = TARGET_TABLE.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="replace", index=False, chunksize=10_000)


if __name__ == "__main__":
    main()

"""Nightly query understanding job: spell-correct, resolve intents, write search.query_intents."""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd
from sqlalchemy import create_engine

from intent import CLICK_CATEGORY_SQL, IntentResolver, category_distribution
from spell import SpellCorrector, Vocabulary

log = logging.getLogger("query_understanding")

QUERY_FREQ_SQL = """
SELECT LOWER(TRIM(query_text)) AS query_text, COUNT(*) AS n
FROM search.query_logs
WHERE event_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY 1
HAVING COUNT(*) >= :min_count
"""
OUTPUT_TABLE = "search.query_intents"


def run(engine, min_count: int, run_date: str) -> pd.DataFrame:
    freq = pd.read_sql(QUERY_FREQ_SQL, engine, params={"min_count": min_count})
    clicks = pd.read_sql(CLICK_CATEGORY_SQL, engine)
    corrector = SpellCorrector(Vocabulary.from_queries(list(zip(freq.query_text, freq.n))))
    clicks["query_text"] = clicks.query_text.str.lower().map(corrector.correct)
    resolver = IntentResolver(category_distribution(clicks.groupby(["query_text", "category_id"], as_index=False).clicks.sum()))

    rows = []
    for q in freq.query_text:
        it = resolver.resolve(q, corrector.correct(q))
        rows.append({
            "query_text": it.query,
            "corrected_query": it.corrected,
            "top_categories": json.dumps(it.categories),
            "attribute_filters": json.dumps(it.filters),
            "run_date": run_date,
        })
    out = pd.DataFrame(rows)
    log.info("resolved %d queries, %d corrected", len(out), (out.query_text != out.corrected_query).sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--run-date", required=True)
    p.add_argument("--min-count", type=int, default=5)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    engine = create_engine(a.dsn)
    out = run(engine, a.min_count, a.run_date)
    schema, table = OUTPUT_TABLE.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="append", index=False)


if __name__ == "__main__":
    main()

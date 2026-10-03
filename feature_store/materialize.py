"""Daily feature materialization.

Parses definitions.yaml, computes point-in-time-correct windowed aggregations
for each feature view as of a given date, validates the output, and appends a
daily snapshot to the view's sink table.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
import yaml

log = logging.getLogger("feature_store.materialize")

DEFINITIONS_PATH = Path(__file__).with_name("definitions.yaml")
SUPPORTED_FUNCTIONS = {"count", "count_distinct", "sum", "avg"}
MAX_NULL_RATE = 0.05


class Warehouse(Protocol):
    def query(self, sql: str, params: dict | None = None) -> pd.DataFrame: ...
    def write(self, table: str, df: pd.DataFrame, partition: str) -> None: ...


@dataclass(frozen=True)
class Aggregation:
    column: str
    function: str
    alias: str


@dataclass(frozen=True)
class Source:
    table: str
    timestamp_column: str
    aggregations: list[Aggregation]
    filter: str | None = None


@dataclass(frozen=True)
class FeatureView:
    name: str
    entity: str
    sink: str
    windows: list[int]
    ttl_days: int
    sources: list[Source]
    static_table: str | None = None
    static_columns: list[str] = field(default_factory=list)


class DefinitionError(ValueError):
    pass


def load_definitions(path: Path) -> list[FeatureView]:
    raw = yaml.safe_load(path.read_text())
    defaults = raw.get("defaults", {})
    views: list[FeatureView] = []
    for v in raw["feature_views"]:
        sources = []
        for s in v["sources"]:
            aggs = [Aggregation(**a) for a in s["aggregations"]]
            bad = [a.function for a in aggs if a.function not in SUPPORTED_FUNCTIONS]
            if bad:
                raise DefinitionError(f"{v['name']}: unsupported functions {bad}")
            sources.append(Source(s["table"], s["timestamp_column"], aggs, s.get("filter")))
        static = v.get("static") or {}
        views.append(FeatureView(
            name=v["name"], entity=v["entity"], sink=v["sink"],
            windows=v.get("windows", defaults["windows"]),
            ttl_days=v.get("ttl_days", defaults["ttl_days"]),
            sources=sources,
            static_table=static.get("table"),
            static_columns=static.get("columns", []),
        ))
    log.info("loaded %d feature views from %s", len(views), path)
    return views


def fetch_source(wh: Warehouse, src: Source, entity: str, as_of: date, lookback: int) -> pd.DataFrame:
    """Pull rows strictly before as_of (exclusive) so no same-day leakage."""
    sql = (
        f"SELECT * FROM {src.table} "
        f"WHERE {src.timestamp_column} >= %(start)s AND {src.timestamp_column} < %(end)s"
    )
    df = wh.query(sql, {"start": as_of - timedelta(days=lookback), "end": as_of})
    if src.filter:
        df = df.query(src.filter)
    return df


def aggregate_window(df: pd.DataFrame, src: Source, entity: str, as_of: date, window: int) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of - timedelta(days=window))
    ts = pd.to_datetime(df[src.timestamp_column])
    scoped = df.loc[(ts >= cutoff) & (ts < pd.Timestamp(as_of))]
    grouped = scoped.groupby(entity)
    out = {}
    for agg in src.aggregations:
        name = f"{agg.alias}_{window}d"
        col = grouped[agg.column]
        out[name] = {
            "count": col.count, "count_distinct": col.nunique,
            "sum": col.sum, "avg": col.mean,
        }[agg.function]()
    return pd.DataFrame(out)


def materialize_view(wh: Warehouse, view: FeatureView, as_of: date) -> pd.DataFrame:
    lookback = max(view.windows)
    frames = []
    for src in view.sources:
        df = fetch_source(wh, src, view.entity, as_of, lookback)
        log.info("%s: %d rows from %s", view.name, len(df), src.table)
        frames.extend(aggregate_window(df, src, view.entity, as_of, w) for w in view.windows)
    feat = pd.concat(frames, axis=1).fillna(0)
    if view.static_table:
        static = wh.query(f"SELECT {view.entity}, {', '.join(view.static_columns)} FROM {view.static_table}")
        feat = feat.join(static.set_index(view.entity), how="left")
    feat = feat.reset_index().rename(columns={"index": view.entity})
    feat["as_of_date"] = as_of
    feat["expires_at"] = as_of + timedelta(days=view.ttl_days)
    return feat


def validate(view: FeatureView, df: pd.DataFrame) -> None:
    if df[view.entity].isna().any():
        raise ValueError(f"{view.name}: null entity keys")
    if df[view.entity].duplicated().any():
        raise ValueError(f"{view.name}: duplicate entity keys")
    expected = {f"{a.alias}_{w}d" for s in view.sources for a in s.aggregations for w in view.windows}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{view.name}: missing columns {sorted(missing)}")
    null_rates = df.isna().mean()
    offenders = null_rates[null_rates > MAX_NULL_RATE]
    if not offenders.empty:
        raise ValueError(f"{view.name}: null rate above {MAX_NULL_RATE}: {offenders.to_dict()}")


def run(wh: Warehouse, as_of: date, only: list[str] | None = None, dry_run: bool = False) -> None:
    for view in load_definitions(DEFINITIONS_PATH):
        if only and view.name not in only:
            continue
        df = materialize_view(wh, view, as_of)
        validate(view, df)
        log.info("%s: %d entities, %d columns", view.name, len(df), df.shape[1])
        if not dry_run:
            wh.write(view.sink, df, partition=as_of.isoformat())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    p.add_argument("--view", action="append", help="limit to these feature views")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from warehouse_client import connect  # internal platform client
    run(connect(), args.as_of, args.view, args.dry_run)


if __name__ == "__main__":
    main()

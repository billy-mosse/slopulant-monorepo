"""Mine query synonyms from in-session reformulations.

A reformulation pair (A -> B) is two consecutive queries in one session, B
issued within ``max_gap_s`` of A, where A got no click and B did. Across all
sessions we score each term-level pair with pointwise mutual information

    PMI(a, b) = log( P(a, b) / (P(a) * P(b)) )

using the counts of rewritten terms, then keep pairs with enough support, a
positive (normalized) PMI and an edit distance large enough that the pair is
not just a typo correction (typos are handled by the spell checker). A
manual blocklist removes pairs the merchandising team rejected.

Input:  search.query_logs
Output: search.synonyms (term, synonym, pmi, npmi, support, direction, mined_at)
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from typing import Iterable, Iterator

import pandas as pd

from shared_utils import (ExperimentConfig, get_logger, levenshtein, load_config, read_sql,
                          require_columns, simple_stem, timed, tokenize, write_table)

log = get_logger("search_exp.synonyms")

SOURCE_TABLE = "search.query_logs"
TARGET_TABLE = "search.synonyms"

LOG_SQL = f"""
SELECT session_id, query_text, event_ts, clicked_product_id
FROM {SOURCE_TABLE}
WHERE event_ts >= CURRENT_DATE - (:lookback_days * INTERVAL '1 day')
  AND query_text IS NOT NULL
ORDER BY session_id, event_ts
"""

# pairs reviewed and rejected by merchandising; symmetric
BLOCKLIST: frozenset[frozenset[str]] = frozenset(map(frozenset, [
    ("queen", "king"), ("twin", "full"), ("cotton", "linen"), ("bath", "beach"),
    ("duvet", "comforter"), ("white", "ivory"), ("rug", "runner"), ("towel", "robe"),
    ("sheet", "blanket"), ("pillow", "sham"), ("grey", "black"), ("curtain", "blind"),
]))
BLOCKED_TERMS = frozenset({"sale", "clearance", "cheap", "new", "best", "gift"})


@dataclass
class SynonymConfig(ExperimentConfig):
    lookback_days: int = 90
    max_gap_s: int = 120
    min_support: int = 8
    min_npmi: float = 0.15
    min_edit_distance: int = 3
    max_len_ratio: float = 3.0
    max_terms_changed: int = 2
    top_n_per_term: int = 5
    blocklist_extra: list[str] = field(default_factory=list)

    def validate(self) -> None:
        super().validate()
        if not -1.0 <= self.min_npmi <= 1.0:
            raise ValueError("min_npmi must be in [-1, 1]")
        if self.max_gap_s <= 0 or self.min_support < 1:
            raise ValueError("max_gap_s and min_support must be positive")


@dataclass(frozen=True)
class Reformulation:
    session_id: str
    before: tuple[str, ...]
    after: tuple[str, ...]

    @property
    def removed(self) -> tuple[str, ...]:
        return tuple(t for t in self.before if t not in self.after)

    @property
    def added(self) -> tuple[str, ...]:
        return tuple(t for t in self.after if t not in self.before)


@dataclass
class PairStats:
    joint: Counter = field(default_factory=Counter)
    left: Counter = field(default_factory=Counter)
    right: Counter = field(default_factory=Counter)
    total: int = 0

    def add(self, a: str, b: str) -> None:
        self.joint[(a, b)] += 1
        self.left[a] += 1
        self.right[b] += 1
        self.total += 1

    def pmi(self, a: str, b: str) -> float:
        n = self.total
        return math.log(self.joint[(a, b)] * n / (self.left[a] * self.right[b]))

    def npmi(self, a: str, b: str) -> float:
        p_ab = self.joint[(a, b)] / self.total
        return self.pmi(a, b) / -math.log(p_ab) if p_ab < 1 else 1.0


def load_logs(cfg: SynonymConfig) -> pd.DataFrame:
    df = read_sql(LOG_SQL, cfg.warehouse_uri, params={"lookback_days": cfg.lookback_days},
                  parse_dates=["event_ts"])
    require_columns(df, ["session_id", "query_text", "event_ts", "clicked_product_id"], SOURCE_TABLE)
    df["clicked"] = df["clicked_product_id"].notna()
    # collapse repeated rows for the same query (pagination, facet clicks) into one step
    df = (df.groupby(["session_id", "query_text"], sort=False)
            .agg(event_ts=("event_ts", "min"), clicked=("clicked", "max"))
            .reset_index()
            .sort_values(["session_id", "event_ts"]))
    log.info("loaded %d query steps over %d sessions", len(df), df["session_id"].nunique())
    return df


def _norm_tokens(q: str) -> tuple[str, ...]:
    return tuple(simple_stem(t) for t in tokenize(q))


def iter_reformulations(df: pd.DataFrame, cfg: SynonymConfig) -> Iterator[Reformulation]:
    """Yield (A -> B) where A had no click, B had a click, and B followed A quickly."""
    for sid, g in df.groupby("session_id", sort=False):
        rows = list(g[["query_text", "event_ts", "clicked"]].itertuples(index=False))
        for a, b in pairwise(rows):
            if a.clicked or not b.clicked:
                continue
            if (b.event_ts - a.event_ts).total_seconds() > cfg.max_gap_s:
                continue
            ta, tb = _norm_tokens(a.query_text), _norm_tokens(b.query_text)
            if not ta or not tb or ta == tb:
                continue
            yield Reformulation(str(sid), ta, tb)


def is_term_swap(r: Reformulation, max_changed: int) -> bool:
    """A real substitution: some terms kept, a few swapped. Pure additions are refinements."""
    rem, add = r.removed, r.added
    if not rem or not add:
        return False
    if len(rem) > max_changed or len(add) > max_changed:
        return False
    return len(set(r.before) & set(r.after)) > 0 or (len(r.before) == 1 and len(r.after) == 1)


def term_pairs(r: Reformulation) -> Iterable[tuple[str, str]]:
    if len(r.removed) == len(r.added):
        return zip(r.removed, r.added)
    return ((" ".join(r.removed), " ".join(r.added)),)


def accumulate(refs: Iterable[Reformulation], cfg: SynonymConfig) -> PairStats:
    stats = PairStats()
    n_seen = n_swaps = 0
    for r in refs:
        n_seen += 1
        if not is_term_swap(r, cfg.max_terms_changed):
            continue
        n_swaps += 1
        for a, b in term_pairs(r):
            stats.add(a, b)
    log.info("reformulations=%d term_swaps=%d distinct_pairs=%d", n_seen, n_swaps, len(stats.joint))
    return stats


def looks_like_typo(a: str, b: str, min_dist: int) -> bool:
    """Edit distance below threshold, scaled down for short words ('rug'/'rugs')."""
    threshold = min(min_dist, max(1, min(len(a), len(b)) // 3))
    return levenshtein(a, b, max_dist=threshold) < threshold or a.replace(" ", "") == b.replace(" ", "")


def is_blocked(a: str, b: str, extra: frozenset[frozenset[str]]) -> bool:
    if frozenset((a, b)) in BLOCKLIST or frozenset((a, b)) in extra:
        return True
    return bool(BLOCKED_TERMS & (set(a.split()) | set(b.split())))


def parse_extra_blocklist(items: list[str]) -> frozenset[frozenset[str]]:
    out = set()
    for item in items:
        left, sep, right = item.partition("|")
        if not sep or not left.strip() or not right.strip():
            raise ValueError(f"bad blocklist entry {item!r}; expected 'term|term'")
        out.add(frozenset((left.strip().lower(), right.strip().lower())))
    return frozenset(out)


def score_pairs(stats: PairStats, cfg: SynonymConfig) -> pd.DataFrame:
    extra = parse_extra_blocklist(cfg.blocklist_extra)
    dropped = Counter()
    rows = []
    for (a, b), support in stats.joint.items():
        if support < cfg.min_support:
            dropped["support"] += 1
            continue
        if max(len(a), len(b)) / max(1, min(len(a), len(b))) > cfg.max_len_ratio:
            dropped["len_ratio"] += 1
            continue
        if looks_like_typo(a, b, cfg.min_edit_distance):
            dropped["typo"] += 1
            continue
        if is_blocked(a, b, extra):
            dropped["blocklist"] += 1
            continue
        npmi = stats.npmi(a, b)
        if npmi < cfg.min_npmi:
            dropped["npmi"] += 1
            continue
        rows.append((a, b, stats.pmi(a, b), npmi, support))
    log.info("kept %d pairs; dropped %s", len(rows), dict(dropped))
    return pd.DataFrame(rows, columns=["term", "synonym", "pmi", "npmi", "support"])


def add_direction(pairs: pd.DataFrame) -> pd.DataFrame:
    """Mark pairs seen both ways as bidirectional; keep the stronger row for each."""
    if pairs.empty:
        return pairs.assign(direction=pd.Series(dtype=str))
    key = pairs.apply(lambda r: tuple(sorted((r.term, r.synonym))), axis=1)
    both = key.map(key.value_counts()) > 1
    pairs = pairs.assign(direction=both.map({True: "both", False: "one_way"}), _key=key)
    pairs = pairs.sort_values("npmi", ascending=False)
    two_way = pairs[pairs["direction"] == "both"].drop_duplicates("_key")
    return pd.concat([two_way, pairs[pairs["direction"] == "one_way"]]).drop(columns="_key")


def top_per_term(pairs: pd.DataFrame, n: int) -> pd.DataFrame:
    return (pairs.sort_values(["term", "npmi"], ascending=[True, False])
                 .groupby("term", sort=False).head(n).reset_index(drop=True))


def mine(cfg: SynonymConfig) -> pd.DataFrame:
    with timed(log, "load"):
        logs = load_logs(cfg)
    with timed(log, "accumulate"):
        stats = accumulate(iter_reformulations(logs, cfg), cfg)
    if stats.total == 0:
        log.warning("no reformulation pairs found; check lookback and click join")
        return pd.DataFrame(columns=["term", "synonym", "pmi", "npmi", "support", "direction"])
    pairs = top_per_term(add_direction(score_pairs(stats, cfg)), cfg.top_n_per_term)
    pairs["mined_at"] = datetime.now(timezone.utc)
    return pairs


def preview(pairs: pd.DataFrame, n: int = 25) -> None:
    if pairs.empty:
        return
    cols = ["term", "synonym", "npmi", "support", "direction"]
    print(pairs.nlargest(n, "support")[cols].to_string(index=False, float_format="%.3f"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Mine synonyms from {SOURCE_TABLE} into {TARGET_TABLE}")
    p.add_argument("--config", help="yaml/json config file")
    p.add_argument("--lookback-days", type=int)
    p.add_argument("--min-support", type=int)
    p.add_argument("--min-npmi", type=float)
    p.add_argument("--min-edit-distance", type=int)
    p.add_argument("--block", action="append", dest="blocklist_extra", metavar="A|B",
                   help="extra blocked pair, repeatable")
    p.add_argument("--dry-run", action="store_true", default=None)
    p.add_argument("--preview", type=int, default=0, help="print top N pairs by support")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "preview"}}
    cfg = load_config(SynonymConfig, args.config, overrides)
    log.info("config: %s", cfg)
    pairs = mine(cfg)
    if args.preview:
        preview(pairs, args.preview)
    write_table(pairs, TARGET_TABLE, cfg.warehouse_uri, mode="replace", dry_run=cfg.dry_run, logger=log)
    return 0 if len(pairs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

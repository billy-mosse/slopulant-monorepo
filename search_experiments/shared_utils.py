"""Shared helpers for the search experiment modules.

Text normalization and tokenization, IDF tables, a tiny config loader,
logging setup, warehouse I/O wrappers and offline evaluation metrics
(precision@k, coverage, intra-list diversity).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from functools import lru_cache
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeVar

import pandas as pd

T = TypeVar("T")

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
DEFAULT_WAREHOUSE_URI = "postgresql://search@warehouse/slopulent"

STOPWORDS = frozenset("a an and are as at be by for from in into is it of on or the to with "
                      "my our your me i you we us this that these those".split())
UNIT_ALIASES = {"inch": "in", "inches": "in", '"': "in", "in.": "in", "centimeter": "cm", "cms": "cm",
                "foot": "ft", "feet": "ft", "'": "ft", "pc": "piece", "pcs": "piece", "pieces": "piece"}
SIZE_ALIASES = {"california king": "cal king", "calking": "cal king", "cal-king": "cal king",
                "twin xl": "twinxl", "twin-xl": "twinxl", "full/queen": "full queen"}

_PUNCT_RE = re.compile(r"[^\w\s\"'.-]+")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*|\"")
_NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(inches|inch|in\.|in|cm|cms|ft|feet|\"|')\b?")
_THREAD_RE = re.compile(r"(\d{3,4})\s*(?:tc|thread\s*count)\b")


# --------------------------------------------------------------------------- logging

def get_logger(name: str, level: str | int | None = None) -> logging.Logger:
    """Return a module logger with a single stderr handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(h)
        logger.propagate = False
    logger.setLevel(level or os.environ.get("SEARCH_EXP_LOG_LEVEL", "INFO"))
    return logger


@contextmanager
def timed(logger: logging.Logger, label: str) -> Iterator[None]:
    """Log wall time of a block: ``with timed(log, "load"): ...``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s took %.2fs", label, time.perf_counter() - t0)


# --------------------------------------------------------------------------- config

@dataclass
class ExperimentConfig:
    """Base config. Subclasses add fields; ``load_config`` fills them from yaml/json/env."""

    warehouse_uri: str = DEFAULT_WAREHOUSE_URI
    lookback_days: int = 30
    min_query_count: int = 5
    dry_run: bool = False
    seed: int = 13
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if self.min_query_count < 1:
            raise ValueError("min_query_count must be >= 1")


C = TypeVar("C", bound=ExperimentConfig)


def _coerce(value: Any, current: Any) -> Any:
    if isinstance(current, bool) and isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(current, (int, float)) and not isinstance(current, bool) and isinstance(value, str):
        return type(current)(value)
    return value


def load_config(cls: type[C], path: str | os.PathLike | None = None,
                overrides: Mapping[str, Any] | None = None, env_prefix: str = "SEARCH_EXP_") -> C:
    """Build ``cls`` from defaults <- file (yaml/json) <- env vars <- explicit overrides.

    Unknown keys from the file land in ``extra`` rather than failing, so old
    configs keep working after a field is removed.
    """
    raw: dict[str, Any] = {}
    if path:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix in {".yaml", ".yml"}:
            import yaml
            raw = yaml.safe_load(text) or {}
        else:
            raw = json.loads(text)
    cfg = cls()
    names = {f.name for f in fields(cls)}
    for k, v in raw.items():
        if k in names:
            setattr(cfg, k, _coerce(v, getattr(cfg, k)))
        else:
            cfg.extra[k] = v
    for name in names:
        env = os.environ.get(env_prefix + name.upper())
        if env is not None:
            setattr(cfg, name, _coerce(env, getattr(cfg, name)))
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k not in names:
            raise KeyError(f"unknown config key {k!r} for {cls.__name__}")
        setattr(cfg, k, _coerce(v, getattr(cfg, k)))
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- text

@lru_cache(maxsize=200_000)
def normalize_query(q: str) -> str:
    """Lowercase, strip accents, unify sizes/units, collapse whitespace.

    >>> normalize_query("  Égyptian Cotton  Sheets 1000 Thread Count, Cal-King ")
    'egyptian cotton sheets 1000tc cal king'
    """
    if not q:
        return ""
    s = unicodedata.normalize("NFKD", q)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = _THREAD_RE.sub(lambda m: f"{m.group(1)}tc", s)
    for k, v in SIZE_ALIASES.items():
        s = s.replace(k, v)
    s = _NUM_UNIT_RE.sub(lambda m: f"{m.group(1)}{UNIT_ALIASES.get(m.group(2), m.group(2))} ", s)
    s = _PUNCT_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()


def tokenize(q: str, drop_stopwords: bool = True, min_len: int = 1) -> list[str]:
    toks = [t for t in _TOKEN_RE.findall(normalize_query(q)) if t != '"']
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [t for t in toks if len(t) >= min_len]


def simple_stem(tok: str) -> str:
    """Very light plural stripping; good enough for 'towels'->'towel', 'throws'->'throw'."""
    if len(tok) <= 3 or tok.isdigit():
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("ches", "shes", "sses", "xes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def char_ngrams(text: str, n_min: int = 2, n_max: int = 4, boundary: str = " ") -> list[str]:
    """Character n-grams with word boundary padding: 'rug' -> ' r', 'ru', 'ug', 'g ', ..."""
    s = f"{boundary}{normalize_query(text)}{boundary}"
    return [s[i:i + n] for n in range(n_min, n_max + 1) for i in range(len(s) - n + 1)]


def levenshtein(a: str, b: str, max_dist: int | None = None) -> int:
    """Edit distance with an optional early exit once every cell exceeds ``max_dist``."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- IDF

class IdfTable:
    """Smoothed IDF over a document collection (queries or product titles).

    idf(t) = log((N + 1) / (df(t) + 1)) + 1, the same form sklearn uses, so
    unseen terms get the maximum weight log(N + 1) + 1.
    """

    def __init__(self, df_counts: Mapping[str, int], n_docs: int):
        self.df = dict(df_counts)
        self.n_docs = n_docs
        self._max = math.log(n_docs + 1) + 1

    @classmethod
    def fit(cls, docs: Iterable[Sequence[str]], weights: Iterable[float] | None = None) -> "IdfTable":
        counts: Counter[str] = Counter()
        n = 0.0
        for doc, w in zip(docs, weights if weights is not None else repeat(1.0)):
            counts.update({t: w for t in set(doc)})
            n += w
        return cls({t: int(round(c)) for t, c in counts.items()}, int(round(n)))

    def idf(self, term: str) -> float:
        d = self.df.get(term)
        return self._max if d is None else math.log((self.n_docs + 1) / (d + 1)) + 1

    def weights(self, tokens: Sequence[str]) -> dict[str, float]:
        return {t: self.idf(t) for t in tokens}

    def least_informative(self, tokens: Sequence[str]) -> str | None:
        return min(tokens, key=self.idf) if tokens else None



# --------------------------------------------------------------------------- warehouse I/O

def get_engine(uri: str = DEFAULT_WAREHOUSE_URI):
    from sqlalchemy import create_engine
    return create_engine(uri, pool_pre_ping=True)

def read_sql(sql: str, uri: str = DEFAULT_WAREHOUSE_URI, params: Mapping[str, Any] | None = None,
             **kw: Any) -> pd.DataFrame:
    from sqlalchemy import text
    with get_engine(uri).connect() as conn:
        return pd.read_sql(text(sql), conn, params=dict(params or {}), **kw)


def write_table(df: pd.DataFrame, qualified_name: str, uri: str = DEFAULT_WAREHOUSE_URI,
                mode: str = "replace", dry_run: bool = False,
                logger: logging.Logger | None = None) -> int:
    """Write ``df`` to ``schema.table``; returns rows written (0 on dry run)."""
    log = logger or get_logger("search_exp.io")
    schema, _, table = qualified_name.partition(".")
    if not table:
        raise ValueError(f"expected schema.table, got {qualified_name!r}")
    if df.empty:
        log.warning("refusing to %s %s with an empty frame", mode, qualified_name)
        return 0
    if dry_run:
        log.info("[dry-run] would write %d rows to %s", len(df), qualified_name)
        return 0
    df.to_sql(table, get_engine(uri), schema=schema, if_exists=mode, index=False,
              chunksize=10_000, method="multi")
    log.info("wrote %d rows to %s (%s)", len(df), qualified_name, mode)
    return len(df)


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str = "frame") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label} missing columns: {missing}")


# --------------------------------------------------------------------------- metrics

def precision_at_k(ranked: Sequence[T], relevant: set[T], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for x in ranked[:k] if x in relevant) / k


def coverage(results: Mapping[str, Sequence[Any]], universe: Iterable[str] | None = None) -> float:
    """Share of queries (in ``universe`` if given) that got at least one result."""
    keys = list(universe) if universe is not None else list(results)
    return sum(1 for q in keys if results.get(q)) / len(keys) if keys else 0.0


def intra_list_diversity(items: Sequence[T], dist: Callable[[T, T], float]) -> float:
    n = len(items)
    if n < 2:
        return 0.0
    total = sum(dist(items[i], items[j]) for i in range(n) for j in range(i + 1, n))
    return total / (n * (n - 1) / 2)


def summarize_metrics(metrics: Mapping[str, float], logger: logging.Logger, label: str) -> None:
    logger.info("%s | %s", label, "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())))

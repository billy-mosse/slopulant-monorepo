"""Dense item vectors for the Slopulent Living catalog.

Produces one fixed-width, L2-normalised vector per SKU from three signal blocks:

* text   -- TF-IDF over cleaned product title + description (word 1-2 grams)
* tags   -- multi-hot over validated merchandising tags, IDF-weighted
* category -- one-hot over the leaf category

The blocks are weighted, horizontally stacked and projected with TruncatedSVD
(latent semantic analysis) down to ``n_components`` columns. The fitted
vectorizer, tag vocabulary, category vocabulary and SVD are persisted next to a
model version string so that vectors can be reproduced and audited later.

Author: Priyanka Nair (ML Platform)
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

try:  # faiss is optional; brute-force numpy search is the fallback.
    import faiss  # type: ignore[import-not-found]

    _HAS_FAISS = True
except ImportError:  # pragma: no cover - depends on the runtime image
    faiss = None
    _HAS_FAISS = False

log = logging.getLogger("catalog_vectors")

PRODUCTS_SQL = """
SELECT p.sku, p.title, p.description, p.category_path, p.updated_at
FROM catalog.products AS p
WHERE p.is_active = TRUE
"""

TAGS_SQL = """
SELECT t.sku, LOWER(TRIM(t.tag)) AS tag
FROM catalog.validated_tags AS t
WHERE t.tag IS NOT NULL
"""

OUTPUT_TABLE = "item_vectors"
OUTPUT_SCHEMA = "features"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\"|in\b|inch(?:es)?\b)")
_CM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*cm\b")
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")

# Boilerplate copy that appears on most product pages and carries no signal.
_BOILERPLATE = ("free shipping on orders over", "see our return policy", "colors may vary slightly")


@dataclass(frozen=True)
class VectorConfig:
    """All knobs that influence the produced vectors.

    Anything in here participates in the model version hash, so changing a
    weight or vocabulary threshold yields a new ``model_version``.
    """

    n_components: int = 128
    min_df: int = 3
    max_df: float = 0.6
    max_features: int = 200_000
    ngram_range: tuple[int, int] = (1, 2)
    title_repeat: int = 2
    min_tag_count: int = 5
    text_weight: float = 1.0
    tag_weight: float = 0.6
    category_weight: float = 0.4
    category_depth: int = 2
    svd_iter: int = 7
    random_state: int = 20261003
    version_prefix: str = "lsa"


@dataclass
class QualityThresholds:
    """Acceptance criteria evaluated by ``validate`` and at the end of a run."""

    min_coverage: float = 0.98
    max_norm_deviation: float = 1e-3
    duplicate_cosine: float = 0.999
    max_duplicate_share: float = 0.02
    nn_sample_size: int = 500
    nn_k: int = 5
    min_nn_category_agreement: float = 0.6


@dataclass
class QualityReport:
    n_products: int
    n_vectors: int
    coverage: float
    norm_mean: float
    norm_std: float
    norm_min: float
    norm_max: float
    duplicate_pairs: int
    duplicate_share: float
    nn_category_agreement: float
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def get_engine(dsn: str) -> Engine:
    return create_engine(dsn, pool_pre_ping=True, future=True)


def load_products(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql(PRODUCTS_SQL, engine)
    before = len(df)
    df = df.dropna(subset=["sku"]).drop_duplicates(subset=["sku"], keep="last")
    if len(df) != before:
        log.warning("Dropped %d product rows with null or duplicate SKU", before - len(df))
    df["sku"] = df["sku"].astype(str)
    log.info("Loaded %d active products", len(df))
    return df.reset_index(drop=True)


def load_tags(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql(TAGS_SQL, engine)
    df["sku"] = df["sku"].astype(str)
    df = df[df["tag"].str.len() > 0].drop_duplicates()
    log.info("Loaded %d validated tag assignments across %d SKUs", len(df), df["sku"].nunique())
    return df


def write_vectors(engine: Engine, skus: Sequence[str], vectors: np.ndarray, model_version: str) -> int:
    """Replace this model version's rows in features.item_vectors (idempotent re-runs)."""
    out = pd.DataFrame({
        "sku": list(skus),
        "vector": [v.astype(np.float32).tolist() for v in vectors],
        "model_version": model_version,
        "built_at": datetime.now(timezone.utc),
    })
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "DELETE FROM features.item_vectors WHERE model_version = %(mv)s", {"mv": model_version}
        )
        out.to_sql(OUTPUT_TABLE, conn, schema=OUTPUT_SCHEMA, if_exists="append",
                   index=False, chunksize=5_000, method="multi")
    log.info("Wrote %d vectors to features.item_vectors (%s)", len(out), model_version)
    return len(out)


def read_vectors(engine: Engine, model_version: str | None) -> tuple[list[str], np.ndarray, str]:
    if model_version is None:
        latest = pd.read_sql(
            "SELECT model_version FROM features.item_vectors ORDER BY built_at DESC LIMIT 1", engine
        )
        if latest.empty:
            raise RuntimeError("features.item_vectors is empty")
        model_version = str(latest.iloc[0, 0])
    df = pd.read_sql("SELECT sku, vector FROM features.item_vectors WHERE model_version = %(mv)s",
                     engine, params={"mv": model_version})
    mat = np.vstack([np.asarray(v, dtype=np.float32) for v in df["vector"]])
    return df["sku"].astype(str).tolist(), mat, model_version


def clean_text(raw: str | float | None) -> str:
    """Normalise marketing copy into lowercase, ASCII, whitespace-separated words."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    text = html.unescape(str(raw))
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    for phrase in _BOILERPLATE:
        text = text.replace(phrase, " ")
    # Collapse measurements into coarse buckets so "84in" and "84 inches" agree.
    text = _SIZE_RE.sub(lambda m: f" size_{int(float(m.group(1)) // 12)}ft ", text)
    text = _CM_RE.sub(lambda m: f" size_{int(float(m.group(1)) // 30)}ft ", text)
    text = _NON_WORD_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def product_documents(products: pd.DataFrame, title_repeat: int) -> list[str]:
    """Title is repeated to up-weight it against long descriptions."""
    titles = products["title"].map(clean_text)
    descs = products["description"].map(clean_text)
    return [((t + " ") * title_repeat + d).strip() for t, d in zip(titles, descs)]


def category_key(path: str | None, depth: int) -> str:
    if not path:
        return "__unknown__"
    parts = [p.strip().lower() for p in re.split(r"[>/|]", str(path)) if p.strip()]
    return " > ".join(parts[:depth]) if parts else "__unknown__"


@dataclass
class TagEncoder:
    """Multi-hot tag encoder with smoothed IDF weights."""

    vocabulary: dict[str, int] = field(default_factory=dict)
    idf: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, tags: pd.DataFrame, n_docs: int, min_count: int) -> "TagEncoder":
        counts = tags.groupby("tag")["sku"].nunique()
        kept = counts[counts >= min_count].sort_index()
        self.vocabulary = {t: i for i, t in enumerate(kept.index)}
        self.idf = np.log((1.0 + n_docs) / (1.0 + kept.to_numpy(dtype=np.float64))) + 1.0
        log.info("Tag vocabulary: %d tags (dropped %d rare)", len(kept), len(counts) - len(kept))
        return self

    def transform(self, skus: Sequence[str], tags: pd.DataFrame) -> sp.csr_matrix:
        row_of = {s: i for i, s in enumerate(skus)}
        hits = tags[tags["tag"].isin(self.vocabulary) & tags["sku"].isin(row_of)]
        rows = hits["sku"].map(row_of).to_numpy()
        cols = hits["tag"].map(self.vocabulary).to_numpy()
        data = self.idf[cols] if len(cols) else np.zeros(0)
        mat = sp.csr_matrix((data, (rows, cols)), shape=(len(skus), len(self.vocabulary)))
        return normalize(mat, norm="l2", copy=False)


@dataclass
class CategoryEncoder:
    depth: int = 2
    vocabulary: dict[str, int] = field(default_factory=dict)

    def fit(self, paths: Iterable[str | None]) -> "CategoryEncoder":
        keys = sorted({category_key(p, self.depth) for p in paths})
        self.vocabulary = {k: i for i, k in enumerate(keys)}
        return self

    def transform(self, paths: Sequence[str | None]) -> sp.csr_matrix:
        """Unseen categories produce an all-zero row rather than an error."""
        rows, cols = [], []
        for i, p in enumerate(paths):
            j = self.vocabulary.get(category_key(p, self.depth))
            if j is not None:
                rows.append(i)
                cols.append(j)
        data = np.ones(len(rows), dtype=np.float64)
        return sp.csr_matrix((data, (rows, cols)), shape=(len(paths), len(self.vocabulary)))


@dataclass
class FittedModel:
    config: VectorConfig
    vectorizer: TfidfVectorizer
    tags: TagEncoder
    categories: CategoryEncoder
    svd: TruncatedSVD
    model_version: str

    def save(self, artifact_dir: Path) -> Path:
        target = artifact_dir / self.model_version
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, target / "tfidf.joblib")
        joblib.dump(self.svd, target / "svd.joblib")
        joblib.dump(self.tags, target / "tags.joblib")
        joblib.dump(self.categories, target / "categories.joblib")
        meta = {
            "model_version": self.model_version, "config": asdict(self.config),
            "n_text_terms": len(self.vectorizer.vocabulary_), "n_tags": len(self.tags.vocabulary),
            "n_categories": len(self.categories.vocabulary),
            "explained_variance": float(self.svd.explained_variance_ratio_.sum()),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (target / "meta.json").write_text(json.dumps(meta, indent=2))
        log.info("Saved artifacts to %s", target)
        return target


def model_version_for(cfg: VectorConfig, n_products: int) -> str:
    digest = hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{cfg.version_prefix}{cfg.n_components}-{stamp}-{digest}-n{n_products}"


def assemble_blocks(cfg: VectorConfig, products: pd.DataFrame, tags: pd.DataFrame,
                    vectorizer: TfidfVectorizer, tag_enc: TagEncoder,
                    cat_enc: CategoryEncoder) -> sp.csr_matrix:
    """Weighted horizontal stack of the text, tag and category blocks."""
    docs = product_documents(products, cfg.title_repeat)
    x_text = vectorizer.transform(docs)
    x_tags = tag_enc.transform(products["sku"].tolist(), tags)
    x_cat = cat_enc.transform(products["category_path"].tolist())
    log.info("Block shapes: text=%s tags=%s category=%s", x_text.shape, x_tags.shape, x_cat.shape)
    blocks = [cfg.text_weight * x_text, cfg.tag_weight * x_tags, cfg.category_weight * x_cat]
    return sp.hstack(blocks, format="csr")


def fit_model(cfg: VectorConfig, products: pd.DataFrame, tags: pd.DataFrame) -> tuple[FittedModel, np.ndarray]:
    docs = product_documents(products, cfg.title_repeat)
    vectorizer = TfidfVectorizer(
        ngram_range=cfg.ngram_range, min_df=cfg.min_df, max_df=cfg.max_df,
        max_features=cfg.max_features, sublinear_tf=True, strip_accents="unicode",
        stop_words="english", dtype=np.float32,
    )
    vectorizer.fit(docs)
    log.info("TF-IDF vocabulary: %d terms", len(vectorizer.vocabulary_))

    tag_enc = TagEncoder().fit(tags[tags["sku"].isin(products["sku"])], len(products), cfg.min_tag_count)
    cat_enc = CategoryEncoder(depth=cfg.category_depth).fit(products["category_path"])
    x = assemble_blocks(cfg, products, tags, vectorizer, tag_enc, cat_enc)

    n_comp = min(cfg.n_components, x.shape[1] - 1)
    if n_comp < cfg.n_components:
        log.warning("Feature space too small; reducing SVD components to %d", n_comp)
    svd = TruncatedSVD(n_components=n_comp, n_iter=cfg.svd_iter, random_state=cfg.random_state)
    z = svd.fit_transform(x)
    log.info("SVD explained variance: %.3f", svd.explained_variance_ratio_.sum())

    z = normalize(z, norm="l2").astype(np.float32)
    version = model_version_for(cfg, len(products))
    return FittedModel(cfg, vectorizer, tag_enc, cat_enc, svd, version), z


def topk_cosine(vectors: np.ndarray, rows: np.ndarray, k: int, exclude_self: bool = True) -> np.ndarray:
    """Brute-force top-k neighbours of ``vectors[rows]``; unit vectors, so IP == cosine."""
    sims = vectors[rows] @ vectors.T
    if exclude_self:
        sims[np.arange(len(rows)), rows] = -np.inf
    k = min(k, sims.shape[1] - 1)
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    order = np.take_along_axis(sims, idx, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(idx, order, axis=1)


def count_near_duplicates(vectors: np.ndarray, threshold: float, chunk: int = 2048) -> int:
    """Count unordered pairs with cosine >= threshold, chunked to bound memory."""
    pairs = 0
    for start in range(0, len(vectors), chunk):
        block = vectors[start : start + chunk] @ vectors.T
        for i in range(block.shape[0]):
            block[i, : start + i + 1] = -1.0  # upper triangle only
        pairs += int((block >= threshold).sum())
    return pairs


def run_quality_checks(skus: Sequence[str], vectors: np.ndarray, products: pd.DataFrame,
                       thr: QualityThresholds, category_depth: int, seed: int = 7) -> QualityReport:
    """Coverage, norm, near-duplicate and neighbour-category checks against ``thr``."""
    n_products = len(products)
    have = set(skus)
    coverage = sum(s in have for s in products["sku"]) / max(n_products, 1)

    norms = np.linalg.norm(vectors, axis=1)
    dup_pairs = count_near_duplicates(vectors, thr.duplicate_cosine)
    dup_share = 2 * dup_pairs / max(len(vectors), 1)

    cat_keys = products["category_path"].map(lambda p: category_key(p, category_depth))
    cat_by_sku = dict(zip(products["sku"], cat_keys))
    cats = np.array([cat_by_sku.get(s, "__unknown__") for s in skus])
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(vectors), size=min(thr.nn_sample_size, len(vectors)), replace=False)
    nbrs = topk_cosine(vectors, sample, thr.nn_k)
    agreement = float((cats[nbrs] == cats[sample][:, None]).mean()) if len(sample) else 0.0

    report = QualityReport(
        n_products=n_products, n_vectors=len(vectors), coverage=coverage,
        norm_mean=float(norms.mean()), norm_std=float(norms.std()),
        norm_min=float(norms.min()), norm_max=float(norms.max()),
        duplicate_pairs=dup_pairs, duplicate_share=dup_share, nn_category_agreement=agreement,
    )
    if coverage < thr.min_coverage:
        report.failures.append(f"coverage {coverage:.3f} < {thr.min_coverage}")
    if abs(report.norm_mean - 1.0) > thr.max_norm_deviation or report.norm_min < 1 - 10 * thr.max_norm_deviation:
        report.failures.append(f"norms off unit length (mean={report.norm_mean:.4f}, min={report.norm_min:.4f})")
    if dup_share > thr.max_duplicate_share:
        report.failures.append(f"near-duplicate share {dup_share:.3f} > {thr.max_duplicate_share}")
    if agreement < thr.min_nn_category_agreement:
        report.failures.append(f"top-{thr.nn_k} category agreement {agreement:.3f} "
                               f"< {thr.min_nn_category_agreement}")
    return report


def log_report(report: QualityReport) -> None:
    log.info(
        "coverage=%.4f norms=%.4f±%.4f [%.4f, %.4f] dup_pairs=%d (%.3f) nn_agree=%.3f",
        report.coverage, report.norm_mean, report.norm_std, report.norm_min,
        report.norm_max, report.duplicate_pairs, report.duplicate_share,
        report.nn_category_agreement,
    )
    for msg in report.failures:
        log.error("Quality check failed: %s", msg)


def export_index(skus: Sequence[str], vectors: np.ndarray, out_dir: Path, model_version: str, use_faiss: bool) -> Path:
    """Write an ANN index (faiss inner-product) or a raw numpy matrix for brute force."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "skus.json").write_text(json.dumps(list(skus)))
    vecs = np.ascontiguousarray(vectors, dtype=np.float32)
    if use_faiss and _HAS_FAISS:
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        path = out_dir / "items.faiss"
        faiss.write_index(index, str(path))
        backend = "faiss-flat-ip"
    else:
        if use_faiss:
            log.warning("faiss not installed; falling back to numpy brute-force index")
        path = out_dir / "items.npy"
        np.save(path, vecs)
        backend = "numpy-bruteforce"
    (out_dir / "index_meta.json").write_text(
        json.dumps({"model_version": model_version, "backend": backend, "n": len(skus), "width": int(vecs.shape[1])})
    )
    log.info("Exported %s index with %d items to %s", backend, len(skus), path)
    return path


def cmd_fit(args: argparse.Namespace) -> int:
    cfg = VectorConfig(n_components=args.n_components, min_df=args.min_df,
                       tag_weight=args.tag_weight, category_weight=args.category_weight)
    engine = get_engine(args.dsn)
    products = load_products(engine)
    tags = load_tags(engine)
    if products.empty:
        log.error("No active products found; aborting")
        return 2

    model, vectors = fit_model(cfg, products, tags)
    model.save(Path(args.artifact_dir))
    report = run_quality_checks(products["sku"].tolist(), vectors, products, QualityThresholds(), cfg.category_depth)
    log_report(report)
    if not report.passed and not args.force:
        log.error("Refusing to write vectors for %s; pass --force to override", model.model_version)
        return 1
    if args.dry_run:
        log.info("Dry run: skipping write of %d vectors", len(vectors))
        return 0
    write_vectors(engine, products["sku"].tolist(), vectors, model.model_version)
    print(model.model_version)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    engine = get_engine(args.dsn)
    products = load_products(engine)
    skus, vectors, version = read_vectors(engine, args.model_version)
    log.info("Validating %d vectors for %s", len(skus), version)
    thr = QualityThresholds(nn_sample_size=args.sample_size, nn_k=args.k)
    report = run_quality_checks(skus, vectors, products, thr, args.category_depth)
    log_report(report)
    if args.json:
        print(json.dumps({**asdict(report), "model_version": version, "passed": report.passed}, indent=2))
    return 0 if report.passed else 1


def cmd_export(args: argparse.Namespace) -> int:
    engine = get_engine(args.dsn)
    skus, vectors, version = read_vectors(engine, args.model_version)
    export_index(skus, vectors, Path(args.out), version, use_faiss=not args.no_faiss)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Item vectors for catalog search and recommendations")
    parser.add_argument("--dsn", required=True, help="SQLAlchemy URL for the warehouse")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("build", help="fit TF-IDF/LSA model and write vectors")
    p_fit.add_argument("--artifact-dir", default="./artifacts")
    p_fit.add_argument("--n-components", type=int, default=VectorConfig.n_components)
    p_fit.add_argument("--min-df", type=int, default=VectorConfig.min_df)
    p_fit.add_argument("--tag-weight", type=float, default=VectorConfig.tag_weight)
    p_fit.add_argument("--category-weight", type=float, default=VectorConfig.category_weight)
    p_fit.add_argument("--dry-run", action="store_true", help="fit and check, do not write")
    p_fit.add_argument("--force", action="store_true", help="write even if quality checks fail")
    p_fit.set_defaults(func=cmd_fit)

    p_val = sub.add_parser("validate", help="run quality checks on stored vectors")
    p_val.add_argument("--model-version", default=None)
    p_val.add_argument("--sample-size", type=int, default=QualityThresholds.nn_sample_size)
    p_val.add_argument("-k", type=int, default=QualityThresholds.nn_k)
    p_val.add_argument("--category-depth", type=int, default=VectorConfig.category_depth)
    p_val.add_argument("--json", action="store_true", help="print the report as JSON")
    p_val.set_defaults(func=cmd_validate)

    p_exp = sub.add_parser("export", help="export a nearest-neighbour index")
    p_exp.add_argument("--model-version", default=None)
    p_exp.add_argument("--out", required=True)
    p_exp.add_argument("--no-faiss", action="store_true", help="always write the numpy matrix")
    p_exp.set_defaults(func=cmd_export)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

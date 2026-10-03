"""Lookalike audiences for paid social and email prospecting.

Started life as a notebook (lookalikes_v3.ipynb). Given a seed list of
high-value customers, we build a seed centroid in the shared customer vector
space and rank every other customer by cosine similarity to it. The top
customers (above a similarity floor, up to a max audience size) become the
lookalike audience for that campaign.

Input:  features.customer_embeddings  (customer_id, embedding, updated_at)
Output: marketing.lookalike_audiences (campaign_id, customer_id, similarity, rank, run_date)
"""

import argparse
import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lookalikes")

# --- notebook "settings" cell -------------------------------------------------
WAREHOUSE_URI = "postgresql://analytics@warehouse/slopulent"
EMBEDDINGS_SQL = """
    SELECT customer_id, embedding
    FROM features.customer_embeddings
    WHERE updated_at >= CURRENT_DATE - INTERVAL '14 days'
"""
OUTPUT_TABLE = "lookalike_audiences"
OUTPUT_SCHEMA = "marketing"   # -> marketing.lookalike_audiences

MAX_AUDIENCE = 50_000         # hard cap requested by paid social team
MIN_SIMILARITY = 0.55         # below this the audience gets noisy (see Q2 test)
MIN_SEED_SIZE = 25            # fewer seeds -> centroid is basically one customer
TRIM_SEED_OUTLIERS = 0.1      # drop the 10% of seeds furthest from the raw centroid


def load_embeddings(engine) -> pd.DataFrame:
    """Pull the latest customer vectors and stack them into a matrix-friendly frame."""
    df = pd.read_sql(EMBEDDINGS_SQL, engine)
    df["embedding"] = df["embedding"].apply(np.asarray, dtype=np.float32)
    log.info("loaded %d customer vectors (dim=%d)", len(df), len(df["embedding"].iloc[0]))
    return df.drop_duplicates("customer_id").reset_index(drop=True)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def seed_centroid(seed_vecs: np.ndarray, trim: float = TRIM_SEED_OUTLIERS) -> np.ndarray:
    """Mean of normalized seed vectors, after trimming the least typical seeds.

    Seed lists from CRM often contain a few odd accounts (B2B buyers, staff).
    We compute a raw centroid, drop the `trim` fraction with the lowest cosine
    to it, and recompute.
    """
    seeds = l2_normalize(seed_vecs)
    raw = seeds.mean(axis=0)
    raw /= np.linalg.norm(raw)
    if trim > 0 and len(seeds) >= MIN_SEED_SIZE * 2:
        sims = seeds @ raw
        keep = sims >= np.quantile(sims, trim)
        seeds = seeds[keep]
        log.info("trimmed %d outlier seeds", int((~keep).sum()))
    centroid = seeds.mean(axis=0)
    return centroid / np.linalg.norm(centroid)


def build_audience(emb: pd.DataFrame, seed_ids: set, campaign_id: str,
                   max_size: int = MAX_AUDIENCE, min_sim: float = MIN_SIMILARITY) -> pd.DataFrame:
    """Rank non-seed customers by cosine similarity to the seed centroid."""
    is_seed = emb["customer_id"].isin(seed_ids).to_numpy()
    if is_seed.sum() < MIN_SEED_SIZE:
        raise ValueError(f"only {is_seed.sum()} seeds found in embeddings, need {MIN_SEED_SIZE}")

    mat = np.vstack(emb["embedding"].to_numpy())
    centroid = seed_centroid(mat[is_seed])
    sims = l2_normalize(mat) @ centroid

    cand = emb.loc[~is_seed, ["customer_id"]].copy()
    cand["similarity"] = sims[~is_seed]
    cand = cand[cand["similarity"] >= min_sim]
    cand = cand.nlargest(max_size, "similarity")
    cand["rank"] = np.arange(1, len(cand) + 1)
    cand["campaign_id"] = campaign_id
    cand["run_date"] = date.today()
    log.info("campaign %s: %d seeds -> audience of %d (min sim %.3f)",
             campaign_id, int(is_seed.sum()), len(cand), cand["similarity"].min() if len(cand) else float("nan"))
    return cand[["campaign_id", "customer_id", "similarity", "rank", "run_date"]]


def audience_summary(aud: pd.DataFrame) -> pd.DataFrame:
    """Quick look at the similarity distribution, the same table we paste into the campaign brief."""
    buckets = pd.cut(aud["similarity"], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    return aud.groupby(buckets, observed=True)["customer_id"].count().rename("customers").to_frame()


def read_seed_file(path: str) -> set:
    seeds = pd.read_csv(path)
    col = "customer_id" if "customer_id" in seeds.columns else seeds.columns[0]
    return set(seeds[col].dropna().astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lookalike audience from a seed list")
    parser.add_argument("--seed-file", required=True, help="CSV with a customer_id column")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--max-audience", type=int, default=MAX_AUDIENCE)
    parser.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    engine = create_engine(WAREHOUSE_URI)
    emb = load_embeddings(engine)
    emb["customer_id"] = emb["customer_id"].astype(str)
    seeds = read_seed_file(args.seed_file)

    audience = build_audience(emb, seeds, args.campaign_id, args.max_audience, args.min_similarity)
    print(audience_summary(audience))

    if args.dry_run:
        log.info("dry run, not writing %s.%s", OUTPUT_SCHEMA, OUTPUT_TABLE)
        return
    audience.to_sql(OUTPUT_TABLE, engine, schema=OUTPUT_SCHEMA, if_exists="append", index=False)
    log.info("wrote %d rows to marketing.lookalike_audiences", len(audience))


if __name__ == "__main__":
    main()

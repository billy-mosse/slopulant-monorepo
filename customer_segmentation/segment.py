"""Customer segmentation for marketing.

Started life as the `segments_exploration.ipynb` notebook; cells were folded into functions.
Reads the shared customer vectors, clusters them, names each cluster from its centroid profile
and writes the result to marketing.segments.
"""
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

INPUT_TABLE = "features.customer_embeddings"
OUTPUT_TABLE = "marketing.segments"
N_COMPONENTS = 5
K_RANGE = range(4, 11)
SILHOUETTE_SAMPLE = 20_000
RANDOM_STATE = 42

# Profile columns carried alongside the embedding, used only for naming clusters.
PROFILE_COLS = ["recency_days", "orders_12m", "spend_12m", "share_bedding", "share_bath", "share_decor"]

# Rule table: first matching rule wins. Each condition compares a centroid's z-score
# (relative to the population) against a threshold.
NAMING_RULES = [
    ("Lapsed high spenders", {"spend_12m": (">", 0.5), "recency_days": (">", 0.75)}),
    ("VIP regulars",         {"spend_12m": (">", 1.0), "orders_12m": (">", 1.0)}),
    ("Bedding loyalists",    {"share_bedding": (">", 0.75)}),
    ("Bath enthusiasts",     {"share_bath": (">", 0.75)}),
    ("Decor browsers",       {"share_decor": (">", 0.75), "orders_12m": ("<", 0.0)}),
    ("New & curious",        {"recency_days": ("<", -0.5), "orders_12m": ("<", -0.25)}),
    ("Dormant one-timers",   {"recency_days": (">", 0.5), "orders_12m": ("<", -0.25)}),
]
DEFAULT_NAME = "Steady mainstream"


def load_vectors(engine) -> pd.DataFrame:
    sql = f"""
        SELECT e.*
        FROM {INPUT_TABLE} e
        WHERE e.snapshot_date = (SELECT MAX(snapshot_date) FROM {INPUT_TABLE})
    """
    df = pd.read_sql(sql, engine).set_index("customer_id")
    logger.info("Loaded %d customer vectors from %s", len(df), INPUT_TABLE)
    return df


def embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    return df[emb_cols].to_numpy(dtype=np.float32)


def choose_k(Z: np.ndarray) -> tuple[int, dict]:
    """Fit k-means for each k in K_RANGE on the PCA space and keep the best silhouette."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Z), size=min(SILHOUETTE_SAMPLE, len(Z)), replace=False)
    sil_scores = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        sil_scores[k] = silhouette_score(Z[idx], km.labels_[idx])
        logger.info("k=%d silhouette=%.4f inertia=%.1f", k, sil_scores[k], km.inertia_)
    best_k = max(sil_scores, key=sil_scores.get)
    return best_k, sil_scores


def name_clusters(profile: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Map cluster id -> human name using centroid z-scores and NAMING_RULES."""
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    centroids = z.groupby(labels).mean()
    names, used = {}, set()
    for cid, row in centroids.iterrows():
        name = DEFAULT_NAME
        for candidate, conds in NAMING_RULES:
            ok = all((row[c] > t) if op == ">" else (row[c] < t) for c, (op, t) in conds.items())
            if ok and candidate not in used:
                name = candidate
                break
        if name in used:  # two clusters fell to the default
            name = f"{name} {cid}"
        used.add(name)
        names[cid] = name
    logger.info("Segment names: %s", names)
    return names


def run(engine, write: bool = True) -> pd.DataFrame:
    df = load_vectors(engine)
    X = embedding_matrix(df)

    reducer = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE))])
    Z = reducer.fit_transform(X)
    logger.info("PCA explained variance: %s", np.round(reducer["pca"].explained_variance_ratio_, 3))

    best_k, sil_scores = choose_k(Z)
    logger.info("Chose k=%d (silhouette %.4f)", best_k, sil_scores[best_k])
    km = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE).fit(Z)

    names = name_clusters(df[PROFILE_COLS], km.labels_)
    dist = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    out = pd.DataFrame({
        "customer_id": df.index,
        "segment_id": km.labels_,
        "segment_name": [names[c] for c in km.labels_],
        "distance_to_centroid": dist.round(4),
        "k": best_k,
        "snapshot_date": pd.Timestamp.today().normalize(),
    })
    print(out.groupby("segment_name").size().sort_values(ascending=False).to_string())

    if write:
        schema, table = OUTPUT_TABLE.split(".")
        out.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
        logger.info("Wrote %d rows to %s", len(out), OUTPUT_TABLE)
    return out


if __name__ == "__main__":
    import sqlalchemy as sa

    parser = argparse.ArgumentParser(description="Build marketing segments")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(sa.create_engine(args.dsn), write=not args.no_write)

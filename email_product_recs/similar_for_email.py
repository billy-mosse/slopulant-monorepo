"""Post-purchase email: "you might also like" products.

For each customer's most recent purchased item, find the most similar catalog items by text
(title + description, TF-IDF + cosine) and keep the top-k they haven't already bought.

Reads catalog.products and orders.lines, writes marketing.email_recs.
"""
import argparse
import logging

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

log = logging.getLogger("email_recs")

PRODUCTS_SQL = """
SELECT product_id, title, description, category_l1, is_active, in_stock
FROM catalog.products
"""
LAST_PURCHASE_SQL = """
SELECT customer_id, product_id, order_ts
FROM orders.lines
WHERE order_ts >= %(start)s AND status = 'completed'
"""
OUTPUT_TABLE = "marketing.email_recs"
TOP_K = 6
LOOKBACK_DAYS = 30          # only customers who bought recently get the post-purchase email
HISTORY_DAYS = 730          # "already bought" exclusion window
MIN_SIMILARITY = 0.08
BATCH = 2048


def clean_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower()
    s = s.str.replace(r"<[^>]+>", " ", regex=True)              # html from the CMS
    s = s.str.replace(r"\b\d+\s*(?:tc|thread count)\b", " threadcount ", regex=True)
    return s.str.replace(r"[^a-z0-9 ]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def run(engine, asof: pd.Timestamp, dry_run: bool) -> pd.DataFrame:
    products = pd.read_sql(PRODUCTS_SQL, engine)
    log.info(f"loaded {len(products):,} products, {products.is_active.sum():,} active")
    products["text"] = clean_text(products["title"]) + " " + clean_text(products["title"]) + " " + clean_text(products["description"])
    products = products.reset_index(drop=True)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True, stop_words="english")
    V = normalize(vec.fit_transform(products["text"]))
    log.info(f"tfidf matrix {V.shape[0]:,} x {V.shape[1]:,}, nnz {V.nnz:,}")
    pid_to_row = pd.Series(products.index, index=products.product_id)
    eligible = (products["is_active"] & products["in_stock"]).to_numpy()

    lines = pd.read_sql(LAST_PURCHASE_SQL, engine, params={"start": asof - pd.Timedelta(days=HISTORY_DAYS)},
                        parse_dates=["order_ts"])
    bought = lines.groupby("customer_id")["product_id"].agg(set)
    recent = lines[lines.order_ts >= asof - pd.Timedelta(days=LOOKBACK_DAYS)]
    last_item = (recent.sort_values("order_ts").groupby("customer_id").tail(1)
                 .set_index("customer_id")[["product_id", "order_ts"]])
    last_item = last_item[last_item.product_id.isin(pid_to_row.index)]
    log.info(f"{len(last_item):,} customers with a purchase in the last {LOOKBACK_DAYS}d, "
             f"{last_item.product_id.nunique():,} distinct anchor items")

    # similarity is per anchor item, so compute once per distinct item then fan out to customers
    anchors = last_item["product_id"].unique()
    neighbours = {}
    for start in range(0, len(anchors), BATCH):
        chunk = anchors[start:start + BATCH]
        rows = pid_to_row.loc[chunk].to_numpy()
        sims = (V[rows] @ V.T).toarray()
        sims[:, ~eligible] = -1.0
        sims[np.arange(len(rows)), rows] = -1.0                  # never recommend the item itself
        top = np.argpartition(-sims, kth=min(TOP_K * 4, sims.shape[1] - 1), axis=1)[:, :TOP_K * 4]
        for i, pid in enumerate(chunk):
            cand = top[i][np.argsort(-sims[i, top[i]])]
            neighbours[pid] = [(products.product_id.iat[j], float(sims[i, j])) for j in cand if sims[i, j] >= MIN_SIMILARITY]
        log.info(f"scored anchors {start:,}-{start + len(chunk):,}")

    records = []
    for cust, row in last_item.iterrows():
        owned = bought.get(cust, set())
        recs = [(p, s) for p, s in neighbours.get(row.product_id, []) if p not in owned][:TOP_K]
        for rank, (p, s) in enumerate(recs, 1):
            records.append((cust, row.product_id, p, rank, round(s, 4)))
    out = pd.DataFrame(records, columns=["customer_id", "anchor_product_id", "rec_product_id", "rank", "similarity"])
    out["generated_at"] = asof

    per_cust = out.groupby("customer_id").size()
    log.info(f"{len(out):,} recs for {out.customer_id.nunique():,} customers; "
             f"{(per_cust < TOP_K).sum():,} customers got fewer than {TOP_K}; mean sim {out.similarity.mean():.3f}")
    cat_mix = out.merge(products[["product_id", "category_l1"]], left_on="rec_product_id", right_on="product_id") \
        .groupby("category_l1").size().sort_values(ascending=False).head(8)
    log.info(f"rec category mix: {cat_mix.to_dict()}")

    if dry_run:
        log.info("dry run, skipping write")
    else:
        schema, table = OUTPUT_TABLE.split(".")
        out.to_sql(table, engine, schema=schema, if_exists="replace", index=False, chunksize=50_000)
        log.info(f"wrote {len(out):,} rows to {OUTPUT_TABLE}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--asof", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(sa.create_engine(args.dsn), pd.Timestamp(args.asof), args.dry_run)


if __name__ == "__main__":
    main()

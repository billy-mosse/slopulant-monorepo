"""
Promo targeting by predicted incremental conversion (S-learner).

One model, treatment as a feature:

    mu(x, t) = P(convert | x, t),   t in {0, 1}
    tau(x)   = mu(x, 1) - mu(x, 0)          (CATE / uplift)

x is the customer embedding vector plus a few recency features from past promos.
Customers are ranked by tau(x); we target the top fraction that fits the budget.

Evaluation on a held-out split, Qini curve over the uplift ranking:

    Q(k) = Y_T(k) - Y_C(k) * N_T(k) / N_C(k)

where Y_T(k), N_T(k) are converted / total treated among the top-k, same for C.
Qini coefficient = area between Q and the random-targeting line, normalised by
the number of customers.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

log = logging.getLogger(__name__)

HISTORY_SQL = """
SELECT customer_id, promo_id, sent_at, treated, converted,
       days_since_last_order, orders_last_180d
FROM marketing.promo_history
WHERE sent_at >= CURRENT_DATE - INTERVAL '18 months'
"""
EMB_SQL = "SELECT customer_id, embedding FROM features.customer_embeddings"
TARGET = "marketing.promo_targets"

TABULAR = ["days_since_last_order", "orders_last_180d"]
GB_PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                 min_samples_leaf=100, l2_regularization=1.0, random_state=0)


def load(engine) -> tuple[pd.DataFrame, np.ndarray]:
    hist = pd.read_sql(HISTORY_SQL, engine)
    emb = pd.read_sql(EMB_SQL, engine)
    E = np.vstack(emb["embedding"].map(np.asarray).to_numpy()).astype(np.float32)
    idx = pd.Series(np.arange(len(emb)), index=emb["customer_id"])
    hist = hist[hist["customer_id"].isin(idx.index)].reset_index(drop=True)
    log.info("%d promo exposures, %d customers with vectors, dim %d",
             len(hist), len(emb), E.shape[1])
    return hist, E[idx.loc[hist["customer_id"]].to_numpy()]


def design(df: pd.DataFrame, E: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.hstack([E, df[TABULAR].fillna(-1).to_numpy(np.float32), t[:, None]])


def fit(df: pd.DataFrame, E: np.ndarray) -> HistGradientBoostingClassifier:
    X = design(df, E, df["treated"].to_numpy(np.float32))
    return HistGradientBoostingClassifier(**GB_PARAMS).fit(X, df["converted"].to_numpy())


def uplift(model, df: pd.DataFrame, E: np.ndarray) -> np.ndarray:
    n = len(df)
    p1 = model.predict_proba(design(df, E, np.ones(n, np.float32)))[:, 1]
    p0 = model.predict_proba(design(df, E, np.zeros(n, np.float32)))[:, 1]
    return p1 - p0


def qini(tau: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    order = np.argsort(-tau)
    t, y = t[order], y[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    q = y_t - y_c * np.divide(n_t, n_c, out=np.zeros_like(n_t, float), where=n_c > 0)
    rand = np.linspace(0, q[-1], len(q))
    return q, float((q - rand).sum() / len(q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-frac", type=float, default=0.2, help="share of customers to target")
    ap.add_argument("--promo-id", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    engine = create_engine(os.environ["WAREHOUSE_URL"])
    df, E = load(engine)

    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=0,
                              stratify=df["treated"])
    m = fit(df.iloc[tr], E[tr])
    tau_te = uplift(m, df.iloc[te], E[te])
    _, q = qini(tau_te, df["treated"].to_numpy()[te], df["converted"].to_numpy()[te])
    log.info("holdout qini coefficient %.5f, mean tau %.4f", q, tau_te.mean())

    # refit on everything, score one row per customer (latest exposure)
    m = fit(df, E)
    last = df.sort_values("sent_at").groupby("customer_id").tail(1)
    tau = uplift(m, last, E[last.index.to_numpy()])
    out = pd.DataFrame({"customer_id": last["customer_id"].to_numpy(), "uplift": tau})
    out = out.sort_values("uplift", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    k = int(np.ceil(args.budget_frac * len(out)))
    out["target"] = (out["rank"] <= k) & (out["uplift"] > 0)
    out["promo_id"] = args.promo_id
    out["qini_holdout"] = q
    log.info("targeting %d of %d customers", int(out["target"].sum()), len(out))

    if args.dry_run:
        print(out.head(20).to_string(index=False))
        return
    schema, table = TARGET.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="append", index=False)


if __name__ == "__main__":
    main()

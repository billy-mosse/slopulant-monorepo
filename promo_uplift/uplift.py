"""Promo uplift model: who should get a promo code?

T-learner: one classifier trained on customers who received a promo in past
randomized campaigns (treated), one on the holdout group (control).
    uplift(x) = P(convert | x, treated) - P(convert | x, control)

Data
----
- marketing.promo_history: customer_id, campaign_id, treated (0/1), converted (0/1), sent_at
- features.customer_embeddings: customer_id, emb_0 ... emb_63, plus RFM columns

Output: marketing.promo_uplift (customer_id, p_treat, p_control, uplift, scored_at)
"""
import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---- config (was the first cell of the notebook) ----
HISTORY_QUERY = """
SELECT h.customer_id, h.campaign_id, h.treated, h.converted, h.sent_at
FROM marketing.promo_history h
WHERE h.is_randomized = TRUE
"""
FEATURES_QUERY = "SELECT * FROM features.customer_embeddings"
OUTPUT_TABLE = "marketing.promo_uplift"
RANDOM_STATE = 17
HOLDOUT_FRAC = 0.25
GBM_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
                  validation_fraction=0.1, random_state=RANDOM_STATE)


def load_training_data(engine):
    history = pd.read_sql(HISTORY_QUERY, engine, parse_dates=["sent_at"])
    feats = pd.read_sql(FEATURES_QUERY, engine)
    # A customer can appear in several campaigns -- keep the most recent exposure
    # so each customer is one row and treatment is not mixed within a person.
    history = (history.sort_values("sent_at")
                      .drop_duplicates("customer_id", keep="last"))
    df = history.merge(feats, on="customer_id", how="inner")
    logger.info("training rows: %d (treated share %.2f, base conversion %.3f)",
                len(df), df["treated"].mean(), df["converted"].mean())
    return df, feats


def feature_columns(df):
    skip = {"customer_id", "campaign_id", "treated", "converted", "sent_at", "updated_at"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


class TLearner:
    """Two independent classifiers, one per arm."""

    def __init__(self, params=None):
        params = params or GBM_PARAMS
        self.model_t = HistGradientBoostingClassifier(**params)
        self.model_c = HistGradientBoostingClassifier(**params)
        self.features = None

    def fit(self, X, treated, y):
        self.features = list(X.columns)
        t = treated.astype(bool).to_numpy()
        self.model_t.fit(X[t], y[t])
        self.model_c.fit(X[~t], y[~t])
        logger.info("fitted treat model on %d rows, control model on %d rows", t.sum(), (~t).sum())
        return self

    def predict(self, X):
        X = X[self.features]
        p_t = self.model_t.predict_proba(X)[:, 1]
        p_c = self.model_c.predict_proba(X)[:, 1]
        return pd.DataFrame({"p_treat": p_t, "p_control": p_c, "uplift": p_t - p_c}, index=X.index)


def split(df, holdout_frac=HOLDOUT_FRAC):
    # stratify on treatment x outcome so both arms keep their conversion rate
    strata = df["treated"].astype(str) + "_" + df["converted"].astype(str)
    return train_test_split(df, test_size=holdout_frac, stratify=strata, random_state=RANDOM_STATE)


def main():
    parser = argparse.ArgumentParser(description="Train the promo uplift T-learner and score customers")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--model-path", default="uplift_tlearner.joblib")
    parser.add_argument("--holdout-path", default="holdout_scored.parquet")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    engine = create_engine(args.dsn)
    df, feats = load_training_data(engine)
    cols = feature_columns(df)
    train, holdout = split(df)

    model = TLearner().fit(train[cols], train["treated"], train["converted"])
    joblib.dump(model, args.model_path)

    # save scored holdout for evaluate.py (Qini / AUUC)
    scored = holdout[["customer_id", "treated", "converted"]].join(model.predict(holdout[cols]))
    scored.to_parquet(args.holdout_path, index=False)
    logger.info("holdout mean predicted uplift %.4f, observed ATE %.4f",
                scored["uplift"].mean(),
                scored.loc[scored.treated == 1, "converted"].mean()
                - scored.loc[scored.treated == 0, "converted"].mean())

    # score the whole customer base
    all_scores = feats[["customer_id"]].join(model.predict(feats[cols].fillna(0)))
    all_scores["scored_at"] = pd.Timestamp.utcnow()
    logger.info("scored %d customers; %.1f%% with positive uplift",
                len(all_scores), 100 * (all_scores["uplift"] > 0).mean())
    if not args.no_write:
        schema, table = OUTPUT_TABLE.split(".")
        all_scores.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
        logger.info("wrote %s", OUTPUT_TABLE)


if __name__ == "__main__":
    main()

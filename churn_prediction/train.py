"""Train the 90-day churn model and write calibrated p_churn to scores.p_churn."""
import argparse
import logging

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from features import build_features, build_labels

log = logging.getLogger("churn")

OUTPUT_TABLE = "scores.p_churn"
HORIZON_DAYS = 90
MODEL_VERSION = "churn_gbm_v4"  # v4: browsing features (product views, ATC share)
BROWSING_FEATURES = ["atc_session_share", "product_views_total", "days_since_product_view",
                     "product_views_30d", "views_per_session_30d"]


def labelled_snapshot(conn, cutoff: pd.Timestamp) -> pd.DataFrame:
    feats = build_features(conn, cutoff)
    buyers = build_labels(conn, cutoff, HORIZON_DAYS)
    feats["churned"] = (~feats.index.isin(buyers)).astype(int)
    log.info(f"snapshot {cutoff.date()}: {len(feats):,} customers, churn rate {feats.churned.mean():.3f}")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--score-date", required=True, help="date to score as of (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = sa.create_engine(args.dsn)
    score_date = pd.Timestamp(args.score_date)
    # time-based split: train on older snapshot, calibrate on a later one, both fully labelled
    train_cutoff = score_date - pd.Timedelta(days=2 * HORIZON_DAYS + 30)
    calib_cutoff = score_date - pd.Timedelta(days=HORIZON_DAYS)

    with engine.connect() as conn:
        train = labelled_snapshot(conn, train_cutoff)
        calib = labelled_snapshot(conn, calib_cutoff)
        score = build_features(conn, score_date)

    feature_cols = [c for c in train.columns if c != "churned"]
    missing = [c for c in BROWSING_FEATURES if c not in feature_cols]
    if missing:
        raise ValueError(f"browsing features missing from snapshot: {missing}")
    log.info(f"{len(feature_cols)} features ({len(BROWSING_FEATURES)} browsing)")
    calib = calib.reindex(columns=train.columns, fill_value=0.0)
    score = score.reindex(columns=feature_cols, fill_value=0.0)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=7,
    )
    model.fit(train[feature_cols], train["churned"])
    log.info(f"fitted GBM with {model.n_iter_} iterations on {len(train):,} rows")
    if args.dry_run:
        imp = pd.Series(np.abs(model.predict_proba(calib[feature_cols])[:, 1] - calib["churned"]).mean(), index=["calib_mae"])
        log.info(f"calibration-set MAE before isotonic: {imp.iloc[0]:.4f}")

    raw_calib = model.predict_proba(calib[feature_cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_calib, calib["churned"])
    cal = iso.predict(raw_calib)
    log.info(
        f"calibration set: AUC {roc_auc_score(calib.churned, raw_calib):.4f}, "
        f"brier raw {brier_score_loss(calib.churned, raw_calib):.4f} -> isotonic {brier_score_loss(calib.churned, cal):.4f}"
    )

    p = iso.predict(model.predict_proba(score[feature_cols])[:, 1])
    out = pd.DataFrame({
        "customer_id": score.index,
        "p_churn": np.round(p, 5),
        "horizon_days": HORIZON_DAYS,
        "score_date": score_date.date(),
        "model_version": MODEL_VERSION,
    })
    deciles = out.groupby(pd.qcut(out.p_churn.rank(method="first"), 10, labels=False))["p_churn"].mean()
    log.info(f"scored {len(out):,} customers, mean p_churn {out.p_churn.mean():.3f}, decile means {deciles.round(3).tolist()}")

    if args.dry_run:
        log.info(f"dry run, not writing {OUTPUT_TABLE}")
        return
    schema, table = OUTPUT_TABLE.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="append", index=False, chunksize=50_000)
    log.info(f"wrote {len(out):,} rows to {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()

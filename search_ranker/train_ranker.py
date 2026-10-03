"""Train the LambdaMART search ranker with position-bias correction."""
import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine

import features as F

logger = logging.getLogger("search_ranker")


def load_frames(engine, cfg: dict) -> dict:
    t = cfg["tables"]
    days = cfg["data"]["lookback_days"]
    logger.info("Loading %s (last %d days)", t["click_logs"], days)
    clicks = pd.read_sql(
        f"SELECT query_id, query_text, sku, position, clicked, title, category_id, price, avg_rating, review_count, "
        f"query_embedding, event_date FROM {t['click_logs']} WHERE event_date >= CURRENT_DATE - {days} "
        f"AND position <= {cfg['data']['max_position']}",
        engine,
    )
    emb = pd.read_sql(f"SELECT sku, embedding FROM {t['product_embeddings']}", engine)
    intents = pd.read_sql(f"SELECT query_text, top_categories FROM {t['query_intents']}", engine)
    logger.info("Loaded %d impressions, %d embeddings, %d intents", len(clicks), len(emb), len(intents))
    return {"clicks": clicks, "emb": emb, "intents": intents}


def position_propensity(clicks: pd.DataFrame) -> pd.Series:
    """Relative CTR by position, normalised so position 1 == 1.0."""
    curve = clicks.groupby("position")["clicked"].mean()
    curve = curve.rolling(3, min_periods=1, center=True).mean()
    prop = (curve / curve.iloc[0]).clip(lower=0.02)
    logger.info("Propensity curve: %s", prop.head(10).round(3).to_dict())
    return prop


def build_features(frames: dict, cfg: dict) -> pd.DataFrame:
    fc = cfg["features"]
    df = frames["clicks"].copy()
    qemb = df.drop_duplicates("query_text")[["query_text", "query_embedding"]].rename(columns={"query_embedding": "embedding"})
    df = F.add_embedding_cosine(df, frames["emb"], qemb)
    df = F.add_bm25_title(df, fc["bm25_k1"], fc["bm25_b"])
    df = F.add_query_category_match(df, frames["intents"])
    df = F.add_smoothed_ctr(df, frames["clicks"], fc["ctr_prior_clicks"], fc["ctr_prior_impressions"])
    df = F.add_price_z(df)
    return F.add_review_features(df)


def to_dataset(df: pd.DataFrame, prop: pd.Series) -> lgb.Dataset:
    df = df.sort_values(["query_id", "position"])
    groups = df.groupby("query_id", sort=False).size().values
    # IPW: a click at a low-visibility position counts for more; non-clicks keep weight 1
    ipw = np.where(df["clicked"] == 1, 1.0 / df["position"].map(prop).fillna(prop.min()).values, 1.0)
    ipw = np.clip(ipw, 1.0, 20.0)
    logger.info("IPW mean %.3f, max %.3f over %d groups", ipw.mean(), ipw.max(), len(groups))
    return lgb.Dataset(df[F.FEATURE_COLUMNS], label=df["clicked"].astype(int), group=groups, weight=ipw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    engine = create_engine(args.dsn)
    frames = load_frames(engine, cfg)
    df = build_features(frames, cfg)

    counts = df.groupby("query_id").size()
    df = df[df["query_id"].isin(counts[counts >= cfg["data"]["min_impressions_per_query"]].index)]
    cutoff = df["event_date"].max() - pd.Timedelta(days=cfg["validation"]["holdout_days"])
    train, valid = df[df["event_date"] < cutoff], df[df["event_date"] >= cutoff]
    logger.info("Train rows %d, valid rows %d", len(train), len(valid))

    prop = position_propensity(train)
    params = {k: v for k, v in cfg["lightgbm"].items() if k not in ("num_boost_round", "early_stopping_rounds")}
    dtrain, dvalid = to_dataset(train, prop), to_dataset(valid, prop)
    model = lgb.train(
        params, dtrain, num_boost_round=cfg["lightgbm"]["num_boost_round"], valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(cfg["lightgbm"]["early_stopping_rounds"]), lgb.log_evaluation(50)],
    )
    logger.info("Best iteration %d, valid NDCG@10 %.4f", model.best_iteration, model.best_score["valid_0"]["ndcg@10"])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / "ranker.txt"))
    imp = dict(zip(F.FEATURE_COLUMNS, model.feature_importance("gain").round(2).tolist()))
    (out / "feature_importance.json").write_text(json.dumps(imp, indent=2))
    pd.DataFrame([{
        "model_path": str(out / "ranker.txt"), "best_iteration": model.best_iteration,
        "ndcg_at_10": model.best_score["valid_0"]["ndcg@10"], "feature_importance": json.dumps(imp),
        "trained_at": pd.Timestamp.utcnow(),
    }]).to_sql(cfg["tables"]["output"].split(".")[1], engine, schema=cfg["tables"]["output"].split(".")[0],
               if_exists="append", index=False)
    logger.info("Registered model in %s", cfg["tables"]["output"])


if __name__ == "__main__":
    main()

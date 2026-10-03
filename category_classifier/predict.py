"""Predict taxonomy paths for active products.

Scores catalog.products with the per-level models from train.py and writes
to catalog.predicted_category. A child prediction must belong to the chosen
parent; if the best valid child has probability below --child-threshold we
stop at the parent level.
"""
from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from taxonomy import PARENT_OF_L2, PARENT_OF_L3
from train import build_text

logger = logging.getLogger("predict")

INPUT_TABLE = "catalog.products"
OUTPUT_TABLE = "catalog.predicted_category"


def constrained_argmax(proba: np.ndarray, classes: np.ndarray, parent_of: dict, parent: str) -> tuple[str | None, float]:
    """Best class whose parent equals `parent`; renormalised over valid children."""
    mask = np.array([parent_of.get(c) == parent for c in classes])
    if not mask.any():
        return None, 0.0
    p = proba * mask
    total = p.sum()
    if total == 0:
        return None, 0.0
    i = int(p.argmax())
    return classes[i], float(p[i] / total)


def predict_hierarchical(models: dict, text: pd.Series, child_threshold: float) -> pd.DataFrame:
    p1 = models["l1"].predict_proba(text)
    p2 = models["l2"].predict_proba(text)
    p3 = models["l3"].predict_proba(text)
    c1, c2, c3 = (models[k].classes_ for k in ("l1", "l2", "l3"))

    rows = []
    for i in range(len(text)):
        j = int(p1[i].argmax())
        l1, conf1 = c1[j], float(p1[i][j])
        l2, conf2 = constrained_argmax(p2[i], c2, PARENT_OF_L2, l1)
        if l2 is None or conf2 < child_threshold:
            rows.append((l1, None, None, conf1, conf2, None, 1))
            continue
        l3, conf3 = constrained_argmax(p3[i], c3, PARENT_OF_L3, l2)
        if l3 is None or conf3 < child_threshold:
            rows.append((l1, l2, None, conf1, conf2, conf3, 2))
            continue
        rows.append((l1, l2, l3, conf1, conf2, conf3, 3))
    return pd.DataFrame(rows, columns=["l1", "l2", "l3", "conf_l1", "conf_l2", "conf_l3", "depth"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict product categories")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--child-threshold", type=float, default=0.55)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(args.dsn)
    bundle = joblib.load(args.model)
    df = pd.read_sql(f"SELECT sku, title, description FROM {INPUT_TABLE} WHERE is_active", engine)
    logger.info("scoring %d products", len(df))

    pred = predict_hierarchical(bundle["models"], build_text(df), args.child_threshold)
    pred.insert(0, "sku", df["sku"].values)
    pred["category_path"] = pred[["l1", "l2", "l3"]].apply(lambda r: " > ".join(x for x in r if x), axis=1)
    pred["scored_at"] = pd.Timestamp.utcnow()
    logger.info("depth distribution: %s", pred["depth"].value_counts(normalize=True).round(3).to_dict())

    schema, table = OUTPUT_TABLE.split(".")
    pred.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    logger.info("wrote %d predictions to %s", len(pred), OUTPUT_TABLE)


if __name__ == "__main__":
    main()

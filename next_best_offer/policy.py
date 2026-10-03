"""Next-best-offer policy: LinUCB scores + business guardrails, written to marketing.offers.

Guardrails (agreed with merchandising, Sept 2026):
  * a customer can be shown at most MAX_DISCOUNT_OFFERS_PER_WEEK discount offers in 7 days
  * customers with negative predicted uplift for an offer never get that offer
  * if nothing survives the guardrails, show "none"
Every decision is logged with its propensity so we can do IPS / doubly-robust evaluation offline.
"""
import argparse
import logging

import numpy as np
import pandas as pd
import sqlalchemy as sa

from linucb import ARMS, LinUCB, build_context

logger = logging.getLogger(__name__)

OUTPUT_TABLE = "marketing.offers"
MODEL_PATH = "linucb_state.npz"
MAX_DISCOUNT_OFFERS_PER_WEEK = 2
DISCOUNT_ARMS = {"pct10_bedding", "bundle_discount"}
EPSILON = 0.05      # small uniform floor so every allowed arm has propensity > 0
UPLIFT_COL = {"free_shipping": "uplift_free_shipping", "pct10_bedding": "uplift_pct10_bedding",
              "bundle_discount": "uplift_bundle_discount"}


def allowed_mask(ctx_raw: pd.DataFrame, recent: pd.Series) -> np.ndarray:
    """Boolean (n, K) mask of arms each customer is allowed to see."""
    n = len(ctx_raw)
    mask = np.ones((n, len(ARMS)), dtype=bool)
    for k, arm in enumerate(ARMS):
        col = UPLIFT_COL.get(arm)
        if col and col in ctx_raw:
            mask[:, k] &= ctx_raw[col].to_numpy() >= 0
        if arm in DISCOUNT_ARMS:
            mask[:, k] &= recent.reindex(ctx_raw.index).fillna(0).to_numpy() < MAX_DISCOUNT_OFFERS_PER_WEEK
    mask[:, ARMS.index("none")] = True
    return mask


def decide(model: LinUCB, X: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Epsilon-greedy over the masked UCB argmax; returns chosen arm idx and its propensity."""
    _, ucb = model.scores(X)
    ucb = np.where(mask, ucb, -np.inf)
    greedy = ucb.argmax(axis=1)
    n_allowed = mask.sum(axis=1)
    probs = mask * (EPSILON / n_allowed[:, None])
    probs[np.arange(len(X)), greedy] += 1 - EPSILON
    u = rng.random(len(X))[:, None]
    chosen = (probs.cumsum(axis=1) < u).sum(axis=1)
    return chosen, probs[np.arange(len(X)), chosen], ucb[np.arange(len(X)), chosen]


def main():
    parser = argparse.ArgumentParser(description="Assign next-best offers")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    engine = sa.create_engine(args.dsn)
    rng = np.random.default_rng(args.seed)

    ctx = build_context(engine)
    X = ctx.to_numpy()
    try:
        model = LinUCB.load(MODEL_PATH)
        model.alpha = args.alpha
    except FileNotFoundError:
        logger.warning("No saved state, starting a fresh LinUCB with d=%d", X.shape[1])
        model = LinUCB(X.shape[1], alpha=args.alpha)

    # learn from last week's logged offers + outcomes before deciding
    hist = pd.read_sql(
        f"SELECT customer_id, arm, reward, decided_at FROM {OUTPUT_TABLE} "
        "WHERE decided_at >= CURRENT_DATE - INTERVAL '7 days'", engine)
    seen = hist[hist.reward.notna() & hist.customer_id.isin(ctx.index)]
    if len(seen):
        model.update(ctx.loc[seen.customer_id].to_numpy(), seen.arm.map(ARMS.index).to_numpy(), seen.reward.to_numpy())
        logger.info("Updated LinUCB with %d rewarded decisions", len(seen))
    recent_discounts = hist[hist.arm.isin(DISCOUNT_ARMS)].groupby("customer_id").size()

    # uplift columns before row-normalisation are needed for the sign check
    raw_uplift = pd.read_sql("SELECT customer_id, promo_type, predicted_uplift FROM marketing.promo_uplift", engine) \
        .pivot_table(index="customer_id", columns="promo_type", values="predicted_uplift").add_prefix("uplift_") \
        .reindex(ctx.index).fillna(0.0)
    mask = allowed_mask(raw_uplift, recent_discounts)
    chosen, prop, score = decide(model, X, mask, rng)

    out = pd.DataFrame({
        "customer_id": ctx.index,
        "arm": [ARMS[i] for i in chosen],
        "propensity": prop.round(5),
        "ucb_score": score.round(5),
        "n_allowed_arms": mask.sum(axis=1),
        "reward": np.nan,
        "decided_at": pd.Timestamp.utcnow(),
        "policy_version": f"linucb_a{args.alpha}",
    })
    logger.info("Offer mix:\n%s", out.arm.value_counts(normalize=True).round(3).to_string())

    if not args.dry_run:
        out.to_sql("offers", engine, schema="marketing", if_exists="append", index=False)
        model.save(MODEL_PATH)
        logger.info("Wrote %d decisions to %s", len(out), OUTPUT_TABLE)


if __name__ == "__main__":
    main()

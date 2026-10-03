"""Evaluate the uplift model on the holdout: Qini curve, AUUC, targeting cutoff.

Qini at the top-k customers ranked by predicted uplift:
    Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)
where Y_* are conversions and N_* are counts in treated/control within the top k.
AUUC here = area between the model Qini curve and the random-targeting line,
normalized by the number of customers.

The cutoff is the fraction of customers to target that maximizes incremental
profit = incremental conversions * margin - promo cost * customers targeted.
"""
import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MARGIN_PER_CONVERSION = 22.0   # avg order margin, $ (finance, Q2 2026)
COST_PER_CODE = 1.80           # avg discount redeemed + send cost, $


def qini_curve(df, score_col="uplift"):
    d = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    t = d["treated"].to_numpy() == 1
    y = d["converted"].to_numpy()
    n_t = np.cumsum(t)
    n_c = np.cumsum(~t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * ~t)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = y_t - np.where(n_c > 0, y_c * n_t / n_c, 0.0)
    frac = np.arange(1, len(d) + 1) / len(d)
    return pd.DataFrame({"frac": frac, "qini": q, "n_t": n_t, "n_c": n_c,
                         "uplift_score": d[score_col].to_numpy()})


def auuc(curve):
    # random line goes from 0 to the total incremental conversions at 100%
    total = curve["qini"].iloc[-1]
    random_line = curve["frac"] * total
    area = np.trapz(curve["qini"] - random_line, curve["frac"])
    # per-customer scale so holdouts of different size are comparable
    return float(area / max(len(curve), 1))


def qini_coefficient(curve):
    """Model area over random, divided by perfect area over random is the usual
    normalization; without a perfect-model curve we report raw area / N."""
    total = curve["qini"].iloc[-1]
    return float(np.trapz(curve["qini"] - curve["frac"] * total, curve["frac"]))


def pick_cutoff(curve, n_population, margin=MARGIN_PER_CONVERSION, cost=COST_PER_CODE):
    # scale holdout incremental conversions to the full population
    scale = n_population / len(curve)
    incr = curve["qini"] * scale
    targeted = curve["frac"] * n_population
    profit = incr * margin - targeted * cost
    best = int(np.argmax(profit.to_numpy()))
    return {
        "target_frac": float(curve["frac"].iloc[best]),
        "uplift_threshold": float(curve["uplift_score"].iloc[best]),
        "expected_incremental_conversions": float(incr.iloc[best]),
        "expected_profit": float(profit.iloc[best]),
        "customers_targeted": int(targeted.iloc[best]),
    }


def decile_table(df):
    d = df.copy()
    d["decile"] = pd.qcut(d["uplift"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    g = d.groupby(["decile", "treated"], observed=True)["converted"].mean().unstack()
    g.columns = ["cr_control", "cr_treat"]
    g["observed_uplift"] = g["cr_treat"] - g["cr_control"]
    g["predicted_uplift"] = d.groupby("decile", observed=True)["uplift"].mean()
    return g


def main():
    parser = argparse.ArgumentParser(description="Qini / AUUC evaluation for promo uplift")
    parser.add_argument("--holdout-path", default="holdout_scored.parquet")
    parser.add_argument("--population", type=int, default=1_200_000,
                        help="number of customers eligible for the next campaign")
    args = parser.parse_args()

    df = pd.read_parquet(args.holdout_path)
    curve = qini_curve(df)
    logger.info("AUUC (normalized): %.4f", auuc(curve))
    logger.info("Qini area over random: %.2f", qini_coefficient(curve))

    print(decile_table(df).round(4).to_string())

    cutoff = pick_cutoff(curve, args.population)
    logger.info("target top %.1f%% (uplift >= %.4f): %d customers, +%.0f conversions, profit $%.0f",
                100 * cutoff["target_frac"], cutoff["uplift_threshold"],
                cutoff["customers_targeted"], cutoff["expected_incremental_conversions"],
                cutoff["expected_profit"])


if __name__ == "__main__":
    main()

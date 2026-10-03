"""Markdown announcement campaign builder.

Started life as the `markdown_email_draft.ipynb` notebook; cells are kept roughly in order.

Each week: take the SKUs that start a markdown this week (from the pricing markdown plan),
match them to marketing segments whose category affinity is strongest, and emit one row per
(segment, sku list, subject line variant) for the email team to load into the ESP.

Reads:  pricing.markdown_plan, marketing.segments
Writes: marketing.markdown_campaign
"""
import argparse
import datetime as dt
import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# %% Parameters
MAX_SKUS_PER_EMAIL = 8
MIN_AFFINITY = 0.15          # cosine between segment affinity vector and markdown category mix
MIN_DISCOUNT = 0.20          # don't announce shallow markdowns, they underperform in email
SUBJECT_VARIANTS = {
    "A": "Up to {max_pct}% off {top_category} - this week only",
    "B": "Your favourite {top_category} just got cheaper",
    "C": "{n_items} picks we think you'll love, now marked down",
}


# %% Load
def load_markdowns(con, week_start: dt.date) -> pd.DataFrame:
    sql = """
        SELECT sku, category, regular_price, markdown_price, markdown_start, inventory_units
        FROM pricing.markdown_plan
        WHERE markdown_start >= %(start)s AND markdown_start < %(end)s
    """
    df = pd.read_sql(sql, con, params={"start": week_start, "end": week_start + dt.timedelta(days=7)})
    df["discount_pct"] = 1 - df["markdown_price"] / df["regular_price"]
    df = df[(df["discount_pct"] >= MIN_DISCOUNT) & (df["inventory_units"] > 0)]
    log.info("%d SKUs enter markdown this week (>= %.0f%% off)", len(df), MIN_DISCOUNT * 100)
    return df


def load_segments(con) -> pd.DataFrame:
    # category_affinity is stored long: one row per (segment_id, category, affinity)
    seg = pd.read_sql(
        "SELECT segment_id, segment_name, category, affinity, audience_size FROM marketing.segments "
        "WHERE is_emailable = TRUE",
        con,
    )
    log.info("%d emailable segments", seg["segment_id"].nunique())
    return seg


# %% Affinity matching
def segment_category_matrix(segments: pd.DataFrame, categories) -> pd.DataFrame:
    wide = segments.pivot_table(index="segment_id", columns="category", values="affinity", aggfunc="mean")
    return wide.reindex(columns=categories, fill_value=0.0).fillna(0.0)


def markdown_category_mix(markdowns: pd.DataFrame, categories) -> np.ndarray:
    # weight categories by units on markdown so a big clearance dominates the mix
    mix = markdowns.groupby("category")["inventory_units"].sum().reindex(categories, fill_value=0)
    return normalize(mix.to_numpy().reshape(1, -1), norm="l2")


def match_segments(markdowns: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    categories = sorted(set(markdowns["category"]) | set(segments["category"]))
    seg_mat = segment_category_matrix(segments, categories)
    sims = cosine_similarity(normalize(seg_mat.to_numpy(), norm="l2"), markdown_category_mix(markdowns, categories))
    out = pd.DataFrame({"segment_id": seg_mat.index, "affinity": sims.ravel()})
    out = out[out["affinity"] >= MIN_AFFINITY].sort_values("affinity", ascending=False)
    log.info("%d segments pass the affinity threshold", len(out))
    return out.merge(segments[["segment_id", "segment_name", "audience_size"]].drop_duplicates(), on="segment_id")


# %% Pick SKUs per segment
def pick_skus(markdowns: pd.DataFrame, segments: pd.DataFrame, segment_id) -> pd.DataFrame:
    aff = segments.loc[segments["segment_id"] == segment_id, ["category", "affinity"]]
    df = markdowns.merge(aff, on="category", how="left").fillna({"affinity": 0.0})
    # rank by affinity first, then by depth of discount
    df["rank_score"] = df["affinity"] * 0.7 + df["discount_pct"] * 0.3
    return df.nlargest(MAX_SKUS_PER_EMAIL, "rank_score")


def subject_lines(skus: pd.DataFrame) -> dict:
    top_category = skus["category"].mode().iat[0].replace("_", " ")
    fields = {
        "max_pct": int(round(skus["discount_pct"].max() * 100)),
        "top_category": top_category,
        "n_items": len(skus),
    }
    return {k: tpl.format(**fields) for k, tpl in SUBJECT_VARIANTS.items()}


# %% Assemble
def build_campaign(markdowns: pd.DataFrame, segments: pd.DataFrame, week_start: dt.date) -> pd.DataFrame:
    matched = match_segments(markdowns, segments)
    rows = []
    for seg in matched.itertuples(index=False):
        skus = pick_skus(markdowns, segments, seg.segment_id)
        if skus.empty:
            continue
        for variant, subject in subject_lines(skus).items():
            rows.append({
                "campaign_week": week_start,
                "segment_id": seg.segment_id,
                "segment_name": seg.segment_name,
                "audience_size": seg.audience_size,
                "segment_affinity": round(float(seg.affinity), 4),
                "sku_list": ",".join(skus["sku"].astype(str)),
                "subject_variant": variant,
                "subject_line": subject,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build the weekly markdown email campaign")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--week-start", type=dt.date.fromisoformat, default=dt.date.today())
    parser.add_argument("--preview", action="store_true", help="print instead of writing")
    args = parser.parse_args()

    from sqlalchemy import create_engine
    con = create_engine(args.dsn)

    markdowns = load_markdowns(con, args.week_start)
    if markdowns.empty:
        log.warning("no markdowns starting week of %s, nothing to send", args.week_start)
        return
    campaign = build_campaign(markdowns, load_segments(con), args.week_start)
    log.info("campaign has %d rows across %d segments", len(campaign), campaign["segment_id"].nunique())

    if args.preview:
        print(campaign.head(30).to_string())
    else:
        campaign.to_sql("markdown_campaign", con, schema="marketing", if_exists="append", index=False)
        log.info("appended to marketing.markdown_campaign")


if __name__ == "__main__":
    main()

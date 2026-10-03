"""
optimize_subjects.py -- pick the subject line most likely to get opened.

CRM asked for something quicker than A/B testing every send: the copywriter drops
3-8 candidate subject lines per campaign into a CSV, we score each one and write a
ranked list back so the ESP sync can pick rank 1 (or split-test the top two).

How it works:
  1. Pull a year of sends/opens and compute the unique open rate per subject line
     we've actually shipped (subjects with too few sends are dropped, too noisy).
  2. Turn each subject into a handful of text features -- length, word count,
     emoji, personalisation token, digits/percentages, "urgency" and "curiosity"
     vocab, punctuation, ALL CAPS shouting.
  3. Fit a gradient-boosted regressor on open rate (weighted by sends).
  4. Score the new candidates, rank within campaign.

Sources:  marketing.email_events
Target:   marketing.subject_line_recommendations

Candidate CSV columns: campaign_id, subject_line
"""
import logging
import re
import sys
from datetime import datetime, timezone

import click
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

log = logging.getLogger("subject_opt")

OUT_TABLE = "marketing.subject_line_recommendations"
MIN_SENDS = 2000          # below this the open rate is mostly noise
LOOKBACK_DAYS = 365

HISTORY_SQL = f"""
SELECT
    subject_line,
    COUNT(DISTINCT CASE WHEN event_type = 'send' THEN message_id END) AS sends,
    COUNT(DISTINCT CASE WHEN event_type = 'open' THEN message_id END) AS opens,
    MIN(event_ts) AS first_sent
FROM marketing.email_events
WHERE event_ts >= CURRENT_DATE - INTERVAL '{LOOKBACK_DAYS} days'
  AND subject_line IS NOT NULL
GROUP BY subject_line
HAVING COUNT(DISTINCT CASE WHEN event_type = 'send' THEN message_id END) >= {MIN_SENDS}
"""

URGENCY_WORDS = {
    "today", "tonight", "now", "hurry", "last", "ends", "ending", "final",
    "hours", "left", "limited", "only", "quick", "deadline", "midnight",
}
CURIOSITY_WORDS = {
    "secret", "why", "how", "guess", "surprise", "revealed", "inside",
    "you", "your", "new", "meet", "discover", "introducing", "psst",
}
# merge tags our ESP uses for first name
FIRST_NAME_RE = re.compile(r"\{\{\s*first_?name\s*\}\}|\*\|FNAME\|\*|%%first_name%%", re.I)
EMOJI_RE = re.compile(r"[☀-➿\U0001F300-\U0001FAFF]")
NUMBER_RE = re.compile(r"\d+")
PERCENT_RE = re.compile(r"\d+\s?%")
WORD_RE = re.compile(r"[a-z']+")

FEATURES = [
    "n_chars", "n_words", "has_emoji", "has_first_name", "has_number",
    "has_percent", "n_urgency", "n_curiosity", "has_question", "has_exclaim",
    "caps_ratio", "starts_with_number",
]


def subject_features(subject: str) -> dict:
    # replace merge tag with a typical-length name so char counts are realistic
    rendered = FIRST_NAME_RE.sub("Alex", subject)
    words = WORD_RE.findall(rendered.lower())
    letters = [c for c in rendered if c.isalpha()]
    return {
        "n_chars": len(rendered),
        "n_words": len(rendered.split()),
        "has_emoji": int(bool(EMOJI_RE.search(subject))),
        "has_first_name": int(bool(FIRST_NAME_RE.search(subject))),
        "has_number": int(bool(NUMBER_RE.search(subject))),
        "has_percent": int(bool(PERCENT_RE.search(subject))),
        "n_urgency": sum(w in URGENCY_WORDS for w in words),
        "n_curiosity": sum(w in CURIOSITY_WORDS for w in words),
        "has_question": int("?" in subject),
        "has_exclaim": int("!" in subject),
        "caps_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "starts_with_number": int(bool(re.match(r"\s*\d", rendered))),
    }


def featurize(subjects: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([subject_features(s) for s in subjects], index=subjects.index)[FEATURES]


def fit_model(history: pd.DataFrame, seed: int = 7) -> GradientBoostingRegressor:
    X = featurize(history["subject_line"])
    y = history["open_rate"].to_numpy()
    w = np.sqrt(history["sends"].to_numpy())  # sqrt so a few mega-sends don't dominate

    params = dict(n_estimators=300, learning_rate=0.03, max_depth=3,
                  subsample=0.8, min_samples_leaf=20, random_state=seed)

    # quick sanity check: grouped CV by month so near-identical resends don't leak
    groups = pd.to_datetime(history["first_sent"]).dt.to_period("M").astype(str)
    maes = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = GradientBoostingRegressor(**params).fit(X.iloc[tr], y[tr], sample_weight=w[tr])
        maes.append(np.average(np.abs(m.predict(X.iloc[te]) - y[te]), weights=w[te]))
    log.info("cv weighted MAE on open rate: %.4f (baseline %.4f)",
             np.mean(maes), np.average(np.abs(y - np.average(y, weights=w)), weights=w))

    model = GradientBoostingRegressor(**params).fit(X, y, sample_weight=w)
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
    log.info("top features: %s", ", ".join(f"{f}={v:.2f}" for f, v in imp[:5]))
    return model


def rank_candidates(model, candidates: pd.DataFrame) -> pd.DataFrame:
    df = candidates.dropna(subset=["subject_line"]).copy()
    df["subject_line"] = df["subject_line"].str.strip()
    df = df[df["subject_line"].str.len() > 0].drop_duplicates(["campaign_id", "subject_line"])
    df["predicted_open_rate"] = np.clip(model.predict(featurize(df["subject_line"])), 0, 1)
    df["rank"] = (df.groupby("campaign_id")["predicted_open_rate"]
                    .rank(ascending=False, method="first").astype(int))
    best = df.groupby("campaign_id")["predicted_open_rate"].transform("max")
    df["lift_vs_best"] = df["predicted_open_rate"] / best - 1.0
    return df.sort_values(["campaign_id", "rank"])


@click.command()
@click.option("--dsn", envvar="WAREHOUSE_URL", required=True)
@click.option("--candidates", "cand_path", type=click.Path(exists=True), required=True,
              help="CSV with campaign_id, subject_line.")
@click.option("--preview", is_flag=True, help="Print instead of writing.")
def cli(dsn, cand_path, preview):
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    engine = sa.create_engine(dsn)

    history = pd.read_sql(HISTORY_SQL, engine)
    history["open_rate"] = history["opens"] / history["sends"]
    log.info("training on %d historical subject lines", len(history))
    if len(history) < 200:
        raise click.ClickException("not enough history to train on")

    model = fit_model(history)
    out = rank_candidates(model, pd.read_csv(cand_path))
    out["model_version"] = datetime.now(timezone.utc).strftime("gbr-%Y%m%d")
    out["scored_at"] = datetime.now(timezone.utc)
    out = out[["campaign_id", "subject_line", "predicted_open_rate", "rank",
               "lift_vs_best", "model_version", "scored_at"]]

    if preview:
        click.echo(out.to_string(index=False))
        return
    schema, table = OUT_TABLE.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="append", index=False)
    log.info("wrote %d ranked subject lines for %d campaigns",
             len(out), out.campaign_id.nunique())


if __name__ == "__main__":
    cli()

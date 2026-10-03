"""Return-reason classifier.

Input:  returns.return_requests  (return_id, comment, reason_code_customer, created_at)
Output: returns.reason_labels    (return_id, reason_code, confidence, method)

Two stages:
  1. high-precision keyword rules (DAMAGED, LATE, TOO_SMALL are usually explicit)
  2. multinomial Naive Bayes with Laplace smoothing, trained on agent-verified labels
Predictions below MIN_CONFIDENCE are labelled UNKNOWN and go to manual review.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger("returns_reason_classifier")

REASONS = ("TOO_SMALL", "COLOR_DIFFERENT", "DAMAGED", "QUALITY", "CHANGED_MIND", "LATE")
MIN_CONFIDENCE = 0.55
LAPLACE_ALPHA = 1.0

# Rules only fire on unambiguous phrases; recall comes from the NB model.
KEYWORD_RULES: dict[str, list[str]] = {
    "DAMAGED": [r"\b(arrived|came) (broken|torn|ripped|cracked|damaged)\b", r"\bshattered\b",
                r"\bstain(ed|s)? (on|out of) the box\b"],
    "LATE": [r"\b(arrived|came|delivered) (too )?late\b", r"\bmissed (the|my) (event|date)\b",
             r"\bnever arrived on time\b"],
    "TOO_SMALL": [r"\btoo (small|short|narrow)\b", r"\bdoesn'?t fit (my|the) (bed|mattress)\b"],
    "COLOR_DIFFERENT": [r"\b(colou?r|shade) (is |was )?(different|off|not the same)\b"],
}
STOPWORDS = {"the", "a", "an", "and", "it", "is", "was", "i", "to", "of", "for", "my", "this",
             "that", "in", "on", "with", "but", "so", "me", "they", "be", "at", "as"}


def clean(comment: str | None) -> str:
    if not comment:
        return ""
    s = comment.lower()
    s = re.sub(r"https?://\S+|\S+@\S+", " ", s)
    s = re.sub(r"\border\s*#?\s*\d+\b", " ", s)
    s = re.sub(r"[^a-z' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s: str) -> list[str]:
    toks = [t.strip("'") for t in s.split() if t not in STOPWORDS and len(t) > 1]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def rule_label(s: str) -> str | None:
    for reason, patterns in KEYWORD_RULES.items():
        if any(re.search(p, s) for p in patterns):
            return reason
    return None


@dataclass
class MultinomialNB:
    alpha: float = LAPLACE_ALPHA
    log_prior: dict[str, float] = field(default_factory=dict)
    log_lik: dict[str, dict[str, float]] = field(default_factory=dict)
    log_unseen: dict[str, float] = field(default_factory=dict)
    vocab: set[str] = field(default_factory=set)

    def fit(self, docs: list[list[str]], labels: list[str]) -> "MultinomialNB":
        class_docs = Counter(labels)
        counts: dict[str, Counter] = defaultdict(Counter)
        for toks, y in zip(docs, labels):
            counts[y].update(toks)
            self.vocab.update(toks)
        V, n = len(self.vocab), len(labels)
        for c in REASONS:
            # P(c) = N_c / N ; P(w|c) = (count(w,c) + alpha) / (sum_w count(w,c) + alpha*|V|)
            self.log_prior[c] = math.log((class_docs[c] + 1) / (n + len(REASONS)))
            total = sum(counts[c].values()) + self.alpha * V
            self.log_lik[c] = {w: math.log((k + self.alpha) / total) for w, k in counts[c].items()}
            self.log_unseen[c] = math.log(self.alpha / total)
        return self

    def predict_proba(self, toks: list[str]) -> dict[str, float]:
        toks = [t for t in toks if t in self.vocab]
        scores = {c: self.log_prior[c] + sum(self.log_lik[c].get(t, self.log_unseen[c]) for t in toks)
                  for c in REASONS}
        m = max(scores.values())  # log-sum-exp for numerical stability
        z = sum(math.exp(v - m) for v in scores.values())
        return {c: math.exp(v - m) / z for c, v in scores.items()}


def classify(df: pd.DataFrame, nb: MultinomialNB) -> pd.DataFrame:
    out = []
    for row in df.itertuples():
        s = clean(row.comment)
        if not s:
            out.append((row.return_id, "UNKNOWN", 0.0, "empty"))
            continue
        rl = rule_label(s)
        if rl:
            out.append((row.return_id, rl, 1.0, "rule"))
            continue
        probs = nb.predict_proba(tokenize(s))
        best = max(probs, key=probs.get)
        if probs[best] < MIN_CONFIDENCE:
            out.append((row.return_id, "UNKNOWN", probs[best], "nb_low_conf"))
        else:
            out.append((row.return_id, best, probs[best], "nb"))
    return pd.DataFrame(out, columns=["return_id", "reason_code", "confidence", "method"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--since", default="2026-01-01")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    engine = create_engine(args.dsn)

    q = text("SELECT return_id, comment, verified_reason, created_at FROM returns.return_requests "
             "WHERE created_at >= :since")
    df = pd.read_sql(q, engine, params={"since": args.since})
    train = df[df["verified_reason"].isin(REASONS)]
    nb = MultinomialNB().fit([tokenize(clean(c)) for c in train["comment"]],
                             train["verified_reason"].tolist())
    log.info("NB trained on %d labelled returns, vocab=%d", len(train), len(nb.vocab))

    labels = classify(df[df["verified_reason"].isna()], nb)
    log.info("label mix:\n%s", labels["reason_code"].value_counts().to_string())
    if not args.dry_run:
        labels.to_sql("reason_labels", engine, schema="returns", if_exists="append", index=False)


if __name__ == "__main__":
    main()

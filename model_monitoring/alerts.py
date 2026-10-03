"""Turn drift findings into alerts: thresholds, 24h dedup, Slack formatting."""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone

import pandas as pd

log = logging.getLogger("model_monitoring.alerts")

ALERTS_TABLE = "monitoring.drift_alerts"
DEDUP_WINDOW = timedelta(hours=24)
SLACK_WEBHOOK_ENV = "DRIFT_SLACK_WEBHOOK"

# (metric, comparator, threshold, severity)
RULES: list[tuple[str, str, float, str]] = [
    ("psi", ">", 0.2, "alert"),
    ("psi", ">", 0.1, "warn"),
    ("ks_pvalue", "<", 0.001, "alert"),
    ("null_rate_delta", ">", 0.05, "alert"),
    ("mean_shift_sd", "abs>", 0.5, "warn"),
]


@dataclass
class DriftFinding:
    model: str
    feature: str
    metric: str
    value: float
    detail: str = ""
    severity: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.model, self.feature, self.metric


def classify(f: DriftFinding) -> str | None:
    for metric, op, thr, sev in RULES:
        if metric != f.metric:
            continue
        hit = {">": f.value > thr, "<": f.value < thr, "abs>": abs(f.value) > thr}[op]
        if hit:
            return sev  # rules ordered most severe first
    return None


def format_slack(alerts: list[DriftFinding], run_date: date) -> dict:
    lines = [f"*Model drift report {run_date.isoformat()}* ({len(alerts)} alerts)"]
    for a in sorted(alerts, key=lambda x: (x.severity != "alert", x.model, x.feature)):
        icon = ":red_circle:" if a.severity == "alert" else ":large_yellow_circle:"
        lines.append(f"{icon} `{a.model}` / `{a.feature}` {a.metric}={a.value:.4g} {a.detail}".rstrip())
    return {"text": "\n".join(lines)}


class AlertSink:
    def __init__(self, wh, post_to_slack: bool = True) -> None:
        self.wh = wh
        self.post_to_slack = post_to_slack
        self.pending: list[DriftFinding] = []
        self.run_date: date | None = None

    def submit(self, findings: list[DriftFinding], run_date: date) -> None:
        self.run_date = run_date
        for f in findings:
            f.severity = classify(f)
            if f.severity:
                self.pending.append(f)

    def _recent_keys(self) -> set[tuple[str, str, str]]:
        since = datetime.now(timezone.utc) - DEDUP_WINDOW
        df = self.wh.query(
            f"SELECT model, feature, metric FROM {ALERTS_TABLE} WHERE created_at >= %(since)s", {"since": since})
        return set(map(tuple, df[["model", "feature", "metric"]].itertuples(index=False)))

    def flush(self) -> list[DriftFinding]:
        seen = self._recent_keys()
        fresh: dict[tuple[str, str, str], DriftFinding] = {}
        for f in self.pending:
            if f.key not in seen and f.key not in fresh:
                fresh[f.key] = f
        alerts = list(fresh.values())
        log.info("%d alerts after dedup (%d suppressed)", len(alerts), len(self.pending) - len(alerts))
        self.pending.clear()
        if not alerts:
            return []
        self.wh.write(ALERTS_TABLE, pd.DataFrame([asdict(a) for a in alerts]), partition=self.run_date.isoformat())
        if self.post_to_slack and (url := os.environ.get(SLACK_WEBHOOK_ENV)):
            body = json.dumps(format_slack(alerts, self.run_date)).encode()
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        return alerts

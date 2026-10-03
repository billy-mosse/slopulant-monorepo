"""Sync vendor terms from portal CSV exports.

Vendors upload a CSV of their terms to the portal; the raw files land (one row per file) in
supply.vendor_portal_exports. Formats drift constantly: lead time given as "21", "3 weeks",
"3w", "15-20 days"; MOQ with thousands separators; prices in EUR/GBP/USD.

This job parses every export received since the last run, normalises units and currency,
compares against the previous snapshot of supply.vendor_terms, and writes a new snapshot.
Rows whose lead time jumped by more than 50% vs the previous snapshot are written with
needs_review = True rather than dropped, so buyers can confirm.
"""
from __future__ import annotations

import argparse
import io
import logging
import re
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger("vendor_terms_sync")

SOURCE_TABLE = "supply.vendor_portal_exports"
TARGET_TABLE = "supply.vendor_terms"

LEAD_TIME_JUMP = 0.50
FX_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "CAD": 0.73, "CNY": 0.14, "INR": 0.012}
COLUMN_ALIASES = {
    "sku": "vendor_sku", "item": "vendor_sku", "item_no": "vendor_sku", "vendor_sku": "vendor_sku",
    "lead_time": "lead_time_raw", "leadtime": "lead_time_raw", "lt": "lead_time_raw",
    "moq": "moq", "min_order_qty": "moq", "minimum_order": "moq",
    "case_pack": "case_pack", "pack_size": "case_pack", "inner_pack": "case_pack",
    "unit_cost": "unit_cost", "price": "unit_cost", "cost": "unit_cost",
    "currency": "currency", "ccy": "currency",
}
_LT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*(d|day|days|w|wk|wks|week|weeks)?\s*$", re.I)


# ------------------------------------------------------------- parsing helpers

def parse_lead_time_days(raw) -> float:
    """'21' -> 21, '3 weeks' -> 21, '15-20 days' -> 20 (we plan on the upper bound)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan
    m = _LT_RE.match(str(raw))
    if not m:
        return np.nan
    lo, hi, unit = m.groups()
    value = float(hi or lo)
    return value * 7 if unit and unit.lower().startswith("w") else value


def parse_int(raw) -> float:
    if raw is None:
        return np.nan
    cleaned = re.sub(r"[,\s]", "", str(raw))
    return float(cleaned) if cleaned.isdigit() else np.nan


def parse_money(raw) -> float:
    cleaned = re.sub(r"[^\d.\-]", "", str(raw).replace(",", "."))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def normalise_export(vendor_id: str, payload: str, default_ccy: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(payload), dtype=str, sep=None, engine="python")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES})
    missing = {"vendor_sku", "lead_time_raw"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")

    out = pd.DataFrame({"vendor_id": vendor_id, "vendor_sku": df["vendor_sku"].str.strip()})
    out["lead_time_days"] = df["lead_time_raw"].map(parse_lead_time_days)
    out["moq"] = df.get("moq", pd.Series(dtype=str)).map(parse_int)
    out["case_pack"] = df.get("case_pack", pd.Series(dtype=str)).map(parse_int).fillna(1)
    ccy = df.get("currency", pd.Series(default_ccy, index=df.index)).fillna(default_ccy).str.upper().str.strip()
    cost = df.get("unit_cost", pd.Series(dtype=str)).map(parse_money)
    out["currency_original"] = ccy
    out["unit_cost_usd"] = cost * ccy.map(FX_TO_USD)
    # MOQ must be a multiple of case pack; vendors round up in practice
    out["moq"] = np.ceil(out["moq"] / out["case_pack"]) * out["case_pack"]
    return out


# ------------------------------------------------------------- validation

def validate(new: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    key = ["vendor_id", "vendor_sku"]
    merged = new.merge(prev[key + ["lead_time_days"]].rename(columns={"lead_time_days": "prev_lead_time_days"}),
                       on=key, how="left")
    change = (merged["lead_time_days"] - merged["prev_lead_time_days"]) / merged["prev_lead_time_days"]
    merged["lead_time_change_pct"] = change.round(3)
    merged["needs_review"] = (change.abs() > LEAD_TIME_JUMP) | merged["lead_time_days"].isna() \
        | merged["unit_cost_usd"].isna()
    merged["review_reason"] = np.select(
        [change.abs() > LEAD_TIME_JUMP, merged["lead_time_days"].isna(), merged["unit_cost_usd"].isna()],
        ["lead_time_jump", "unparseable_lead_time", "unknown_currency_or_cost"], default=None)
    return merged


def run(dsn: str, since: date | None) -> pd.DataFrame:
    eng = create_engine(dsn)
    with eng.connect() as con:
        since = since or con.execute(text(f"SELECT COALESCE(MAX(snapshot_date), DATE '2000-01-01') FROM {TARGET_TABLE}")).scalar()
        exports = pd.read_sql(text(f"SELECT vendor_id, uploaded_at, default_currency, file_body "
                                   f"FROM {SOURCE_TABLE} WHERE uploaded_at > :since ORDER BY uploaded_at"),
                              con, params={"since": since})
        prev = pd.read_sql(text(f"SELECT vendor_id, vendor_sku, lead_time_days FROM {TARGET_TABLE} "
                                f"WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM {TARGET_TABLE})"), con)
    log.info("%d exports since %s; previous snapshot has %d rows", len(exports), since, len(prev))

    frames, failed = [], 0
    for exp in exports.itertuples(index=False):
        try:
            frames.append(normalise_export(exp.vendor_id, exp.file_body, exp.default_currency or "USD"))
        except (ValueError, pd.errors.ParserError) as err:
            failed += 1
            log.warning("vendor %s export %s rejected: %s", exp.vendor_id, exp.uploaded_at, err)
    if not frames:
        return pd.DataFrame()
    # latest upload per vendor wins
    new = pd.concat(frames).drop_duplicates(["vendor_id", "vendor_sku"], keep="last")
    checked = validate(new, prev).assign(snapshot_date=date.today())
    log.info("%d terms parsed, %d flagged for review, %d exports rejected",
             len(checked), int(checked["needs_review"].sum()), failed)
    return checked


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalise vendor portal exports into vendor terms")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--since", type=date.fromisoformat)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    terms = run(args.dsn, args.since)
    if terms.empty or args.dry_run:
        log.info("nothing written (empty=%s, dry_run=%s)", terms.empty, args.dry_run)
        return
    terms.to_sql("vendor_terms", create_engine(args.dsn), schema="supply", if_exists="append", index=False)


if __name__ == "__main__":
    main()

"""AP three-way match: vendor invoice lines vs. warehouse receipts.

An invoice line is approved for payment only if the goods were received in the quantity billed
and the unit price is within tolerance. Everything else becomes an exception for AP clerks.

    finance.invoices   -> invoice_id, vendor_name, invoice_date, sku, qty_billed, unit_price, po_number
    finance.receipts   -> receipt_id, vendor_name, received_at, sku, qty_received, po_unit_price, po_number
    finance.invoice_exceptions <- one row per problem line
"""
from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from enum import StrEnum
from typing import TypedDict

import pandas as pd
from rapidfuzz import fuzz, process
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

PRICE_TOLERANCE = 0.02
VENDOR_MATCH_MIN = 88          # rapidfuzz token_set_ratio, 0-100
LOOKBACK_DAYS = 90
LEGAL_SUFFIXES = r"\b(inc|incorporated|llc|ltd|limited|gmbh|co|corp|corporation|sa|srl|bv|plc)\b\.?"

INVOICE_SQL = """
SELECT invoice_id, vendor_name, invoice_date, sku, qty_billed, unit_price, po_number
FROM finance.invoices
WHERE status = 'pending_match' AND invoice_date >= CURRENT_DATE - INTERVAL '{days} days'
"""
RECEIPT_SQL = """
SELECT receipt_id, vendor_name, received_at, sku, qty_received, po_unit_price, po_number
FROM finance.receipts
WHERE received_at >= CURRENT_DATE - INTERVAL '{days} days'
"""


class ExceptionType(StrEnum):
    UNKNOWN_VENDOR = "unknown_vendor"
    NOT_RECEIVED = "not_received"
    QTY_OVER_BILLED = "qty_over_billed"
    QTY_UNDER_BILLED = "qty_under_billed"
    PRICE_VARIANCE = "price_variance"


class ExceptionRow(TypedDict):
    invoice_id: str
    po_number: str
    sku: str | None
    exception_type: str
    invoice_vendor: str
    matched_vendor: str | None
    vendor_score: float | None
    qty_billed: float | None
    qty_received: float | None
    unit_price: float | None
    po_unit_price: float | None
    amount_at_risk: float


def normalize_vendor(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(LEGAL_SUFFIXES, " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class VendorResolver:
    """Maps the free-text vendor name on an invoice to the name used in receiving."""

    def __init__(self, receipt_vendors: pd.Series):
        self._canon = {normalize_vendor(v): v for v in receipt_vendors.dropna().unique()}
        self._keys = list(self._canon)
        self._cache: dict[str, tuple[str | None, float]] = {}

    def resolve(self, invoice_vendor: str) -> tuple[str | None, float]:
        if invoice_vendor in self._cache:
            return self._cache[invoice_vendor]
        key = normalize_vendor(invoice_vendor)
        if key in self._canon:
            hit = (self._canon[key], 100.0)
        else:
            best = process.extractOne(key, self._keys, scorer=fuzz.token_set_ratio)
            hit = (self._canon[best[0]], float(best[1])) if best and best[1] >= VENDOR_MATCH_MIN else (None, best[1] if best else 0.0)
        self._cache[invoice_vendor] = hit
        return hit


def aggregate_receipts(receipts: pd.DataFrame) -> pd.DataFrame:
    # multiple partial deliveries against one PO line are summed
    return (receipts.groupby(["vendor_name", "po_number", "sku"], as_index=False)
            .agg(qty_received=("qty_received", "sum"), po_unit_price=("po_unit_price", "last")))


def classify(line: pd.Series) -> ExceptionType | None:
    if pd.isna(line["qty_received"]):
        return ExceptionType.NOT_RECEIVED
    if line["qty_billed"] > line["qty_received"]:
        return ExceptionType.QTY_OVER_BILLED
    if line["qty_billed"] < line["qty_received"]:
        return ExceptionType.QTY_UNDER_BILLED
    if abs(line["unit_price"] - line["po_unit_price"]) > PRICE_TOLERANCE * line["po_unit_price"]:
        return ExceptionType.PRICE_VARIANCE
    return None


def amount_at_risk(line: pd.Series, kind: ExceptionType) -> float:
    billed = line["qty_billed"] * line["unit_price"]
    match kind:
        case ExceptionType.NOT_RECEIVED | ExceptionType.UNKNOWN_VENDOR:
            return float(billed)
        case ExceptionType.QTY_OVER_BILLED:
            return float((line["qty_billed"] - line["qty_received"]) * line["unit_price"])
        case ExceptionType.PRICE_VARIANCE:
            return float(line["qty_billed"] * (line["unit_price"] - line["po_unit_price"]))
        case _:
            return 0.0


def match(invoices: pd.DataFrame, receipts: pd.DataFrame) -> list[ExceptionRow]:
    resolver = VendorResolver(receipts["vendor_name"])
    resolved = invoices["vendor_name"].map(resolver.resolve)
    invoices = invoices.assign(matched_vendor=resolved.str[0], vendor_score=resolved.str[1])

    out: list[ExceptionRow] = []
    unknown = invoices["matched_vendor"].isna()
    lines = invoices[~unknown].merge(
        aggregate_receipts(receipts), how="left",
        left_on=["matched_vendor", "po_number", "sku"], right_on=["vendor_name", "po_number", "sku"],
        suffixes=("", "_rcv"),
    )
    for _, line in pd.concat([invoices[unknown], lines]).iterrows():
        kind = ExceptionType.UNKNOWN_VENDOR if pd.isna(line["matched_vendor"]) else classify(line)
        if kind is None:
            continue
        out.append(ExceptionRow(
            invoice_id=line["invoice_id"], po_number=line["po_number"], sku=line.get("sku"),
            exception_type=kind.value, invoice_vendor=line["vendor_name"],
            matched_vendor=line.get("matched_vendor"), vendor_score=line.get("vendor_score"),
            qty_billed=line.get("qty_billed"), qty_received=line.get("qty_received"),
            unit_price=line.get("unit_price"), po_unit_price=line.get("po_unit_price"),
            amount_at_risk=round(amount_at_risk(line, kind), 2),
        ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way match invoices against receipts")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    engine = create_engine(args.dsn)
    invoices = pd.read_sql(INVOICE_SQL.format(days=args.lookback_days), engine)
    receipts = pd.read_sql(RECEIPT_SQL.format(days=args.lookback_days), engine)
    logger.info("matching %d invoice lines against %d receipt lines", len(invoices), len(receipts))

    exceptions = pd.DataFrame(match(invoices, receipts))
    if exceptions.empty:
        logger.info("all invoice lines matched cleanly")
        return
    exceptions["run_date"] = pd.Timestamp.today().normalize()
    exceptions.to_sql("invoice_exceptions", engine, schema="finance", if_exists="append", index=False)
    logger.info("%d exceptions written, %.2f total at risk; by type: %s", len(exceptions),
                exceptions["amount_at_risk"].sum(), exceptions["exception_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()

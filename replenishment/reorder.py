"""Replenishment: (s, S) reorder policy per SKU x warehouse.

Inputs:  supply.demand_forecast   (weekly P10/P50/P90 units, 12-week horizon)
         inventory.stock_levels   (on hand, on order, allocated)
         supply.vendor_terms      (lead time, case pack, MOQ, unit cost)
Output:  supply.purchase_orders   (one row per PO line)

Business constraints:
  * 95% cycle service level -> z = 1.645
  * order-up-to S covers lead time + review period demand + safety stock
  * quantities rounded UP to case pack, then raised to vendor MOQ (per vendor-SKU)
  * weekly open-to-buy budget; if exceeded, drop lowest-margin lines first
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger("replenishment")

SERVICE_LEVEL_Z = 1.645          # 95% service level
REVIEW_PERIOD_WEEKS = 1           # we review weekly
P90_P10_SPREAD_TO_SIGMA = 2.563   # P90 - P10 = 2 * 1.2816 sigma under normality


@dataclass(frozen=True)
class VendorTerms:
    vendor_id: str
    lead_time_weeks: float
    case_pack: int
    moq_units: int
    unit_cost: float


@dataclass
class PolicyLine:
    sku: str
    warehouse_id: str
    vendor_id: str
    reorder_point: float
    order_up_to: float
    inventory_position: float
    order_qty: int
    unit_cost: float
    margin_pct: float

    @property
    def extended_cost(self) -> float:
        return self.order_qty * self.unit_cost


def load_inputs(engine) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fc = pd.read_sql(text("""
        SELECT sku, warehouse_id, horizon, p10, p50, p90
        FROM supply.demand_forecast
        WHERE run_date = (SELECT MAX(run_date) FROM supply.demand_forecast)"""), engine)
    stock = pd.read_sql(text("""
        SELECT sku, warehouse_id, on_hand, on_order, allocated, margin_pct
        FROM inventory.stock_levels"""), engine)
    terms = pd.read_sql(text("""
        SELECT sku, vendor_id, lead_time_days, case_pack, moq_units, unit_cost
        FROM supply.vendor_terms WHERE is_primary"""), engine)
    return fc, stock, terms


def demand_over(fc_sku: pd.DataFrame, weeks: float) -> tuple[float, float]:
    """Mean demand and sigma over a (possibly fractional) number of weeks.

    Sigma per week is derived from the quantile spread; weekly errors are
    assumed independent, so variances add across weeks.
    """
    fc_sku = fc_sku.sort_values("horizon")
    full = int(math.floor(weeks))
    frac = weeks - full
    mu, var = 0.0, 0.0
    for i, row in enumerate(fc_sku.itertuples()):
        w = 1.0 if i < full else (frac if i == full else 0.0)
        if w == 0.0:
            break
        sigma_week = max(row.p90 - row.p10, 0.0) / P90_P10_SPREAD_TO_SIGMA
        mu += w * row.p50
        var += w * sigma_week ** 2
    return mu, math.sqrt(var)


def round_to_pack(qty: float, terms: VendorTerms) -> int:
    if qty <= 0:
        return 0
    packs = math.ceil(qty / terms.case_pack)
    units = packs * terms.case_pack
    if units < terms.moq_units:
        units = math.ceil(terms.moq_units / terms.case_pack) * terms.case_pack
    return int(units)


def compute_policy(fc: pd.DataFrame, stock: pd.DataFrame, terms: pd.DataFrame) -> list[PolicyLine]:
    terms_by_sku = {r.sku: VendorTerms(r.vendor_id, r.lead_time_days / 7.0, int(r.case_pack),
                                       int(r.moq_units), float(r.unit_cost))
                    for r in terms.itertuples()}
    lines: list[PolicyLine] = []
    for (sku, wh), grp in fc.groupby(["sku", "warehouse_id"]):
        vt = terms_by_sku.get(sku)
        if vt is None:
            log.warning("no primary vendor terms for %s, skipping", sku)
            continue
        st = stock[(stock.sku == sku) & (stock.warehouse_id == wh)]
        if st.empty:
            continue
        st = st.iloc[0]
        mu_lt, sd_lt = demand_over(grp, vt.lead_time_weeks)
        mu_ltr, sd_ltr = demand_over(grp, vt.lead_time_weeks + REVIEW_PERIOD_WEEKS)
        s = mu_lt + SERVICE_LEVEL_Z * sd_lt
        S = mu_ltr + SERVICE_LEVEL_Z * sd_ltr
        ip = st.on_hand + st.on_order - st.allocated
        qty = round_to_pack(S - ip, vt) if ip <= s else 0
        if qty > 0:
            lines.append(PolicyLine(sku, wh, vt.vendor_id, s, S, ip, qty, vt.unit_cost, st.margin_pct))
    return lines


def apply_budget(lines: list[PolicyLine], budget: float) -> list[PolicyLine]:
    """Trim lowest-margin SKUs first until total spend fits the open-to-buy."""
    total = sum(l.extended_cost for l in lines)
    if total <= budget:
        return lines
    kept = sorted(lines, key=lambda l: l.margin_pct, reverse=True)
    while kept and total > budget:
        dropped = kept.pop()
        total -= dropped.extended_cost
        log.info("budget cut: dropped %s @ %s (margin %.1f%%, $%.0f)", dropped.sku,
                 dropped.warehouse_id, 100 * dropped.margin_pct, dropped.extended_cost)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--budget", type=float, default=750_000.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    engine = create_engine(args.dsn)
    lines = apply_budget(compute_policy(*load_inputs(engine)), args.budget)
    po = pd.DataFrame([{**l.__dict__, "extended_cost": l.extended_cost} for l in lines])
    po["created_at"] = pd.Timestamp.utcnow()
    log.info("%d PO lines, total $%.0f", len(po), po["extended_cost"].sum() if len(po) else 0)
    if not args.dry_run and len(po):
        po.to_sql("purchase_orders", engine, schema="supply", if_exists="append", index=False)


if __name__ == "__main__":
    main()

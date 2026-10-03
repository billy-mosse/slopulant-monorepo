"""Shipping box recommender.

For each open order pick the cheapest box (by billable weight) that physically fits all items.

Packing: first-fit decreasing by volume into a list of free sub-spaces (guillotine split), trying
all 6 axis-aligned rotations per item. Cost: carriers bill max(actual weight, L*W*H / DIM_DIVISOR),
plus the box's own unit cost.

Reads  catalog.product_dimensions, fulfillment.box_catalog
Writes fulfillment.box_assignments
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from itertools import permutations

import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("packaging_recommender")

# --- business constraints -------------------------------------------------------
DIM_DIVISOR_IN3_PER_LB = 139      # carrier dimensional-weight divisor (inches, lb)
PADDING_IN = 0.5                  # dunnage on each side; fragile items get double
MAX_BOX_WEIGHT_LB = 50.0          # warehouse handling limit for a single carton
RATE_PER_BILLABLE_LB = 0.62       # blended zone-5 ground rate, good enough to rank boxes
FRAGILE_CATEGORIES = {"glassware", "ceramics", "mirrors", "vases"}


@dataclass(frozen=True)
class Item:
    sku: str
    l: float
    w: float
    h: float
    weight_lb: float
    fragile: bool

    @property
    def volume(self) -> float:
        return self.l * self.w * self.h

    def padded(self) -> tuple[float, float, float]:
        p = PADDING_IN * (4 if self.fragile else 2)
        return self.l + p, self.w + p, self.h + p


@dataclass(frozen=True)
class Box:
    box_id: str
    l: float
    w: float
    h: float
    tare_lb: float
    unit_cost: float

    @property
    def volume(self) -> float:
        return self.l * self.w * self.h

    @property
    def dim_weight_lb(self) -> float:
        return self.volume / DIM_DIVISOR_IN3_PER_LB


@dataclass
class Space:
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float

    def fits(self, d: tuple[float, float, float]) -> bool:
        return d[0] <= self.l and d[1] <= self.w and d[2] <= self.h


@dataclass
class PackResult:
    box: Box
    fits: bool
    billable_lb: float = 0.0
    cost: float = float("inf")
    fill_ratio: float = 0.0
    placements: list[tuple[str, tuple[float, float, float]]] = field(default_factory=list)


def try_pack(items: list[Item], box: Box) -> PackResult:
    """First-fit decreasing with rotations into guillotine-split free spaces."""
    spaces = [Space(0, 0, 0, box.l, box.w, box.h)]
    placements = []
    for item in sorted(items, key=lambda i: i.volume, reverse=True):
        placed = False
        for si, sp in enumerate(spaces):
            for dims in set(permutations(item.padded())):
                if not sp.fits(dims):
                    continue
                dl, dw, dh = dims
                # split the remaining space into three disjoint boxes: right, front, top
                new = [
                    Space(sp.x + dl, sp.y, sp.z, sp.l - dl, sp.w, sp.h),
                    Space(sp.x, sp.y + dw, sp.z, dl, sp.w - dw, sp.h),
                    Space(sp.x, sp.y, sp.z + dh, dl, dw, sp.h - dh),
                ]
                spaces[si:si + 1] = [s for s in new if s.l > 0 and s.w > 0 and s.h > 0]
                spaces.sort(key=lambda s: (s.z, s.y, s.x))
                placements.append((item.sku, dims))
                placed = True
                break
            if placed:
                break
        if not placed:
            return PackResult(box=box, fits=False)

    actual = sum(i.weight_lb for i in items) + box.tare_lb
    if actual > MAX_BOX_WEIGHT_LB:
        return PackResult(box=box, fits=False)
    billable = max(actual, box.dim_weight_lb)
    return PackResult(
        box=box, fits=True, billable_lb=round(billable, 2),
        cost=round(billable * RATE_PER_BILLABLE_LB + box.unit_cost, 2),
        fill_ratio=round(sum(i.volume for i in items) / box.volume, 3),
        placements=placements,
    )


def choose_box(items: list[Item], boxes: list[Box]) -> PackResult | None:
    # cheap volume pre-filter before running the packer
    need = sum(i.volume for i in items)
    candidates = [b for b in sorted(boxes, key=lambda b: b.volume) if b.volume >= need]
    results = [r for r in (try_pack(items, b) for b in candidates) if r.fits]
    return min(results, key=lambda r: (r.cost, r.box.volume)) if results else None


def load(engine, wave_file: str) -> tuple[pd.DataFrame, list[Box]]:
    # the WMS drops each pick wave as a parquet of (order_id, sku, qty)
    wave = pd.read_parquet(wave_file, columns=["order_id", "sku", "qty"])
    dims = pd.read_sql(
        "SELECT sku, length_in, width_in, height_in, weight_lb, category "
        "FROM catalog.product_dimensions WHERE sku = ANY(%(skus)s)",
        engine, params={"skus": wave["sku"].unique().tolist()},
    )
    lines = wave.merge(dims, on="sku", how="left")
    missing = lines["length_in"].isna()
    if missing.any():
        # no dimensions -> cannot pack safely; drop the whole order to manual
        bad = set(lines.loc[missing, "order_id"])
        log.warning("%d orders have SKUs without dimensions; sending to manual", len(bad))
        lines = lines[~lines["order_id"].isin(bad)]
    box_df = pd.read_sql("SELECT box_id, inner_l_in, inner_w_in, inner_h_in, tare_lb, unit_cost "
                         "FROM fulfillment.box_catalog WHERE active", engine)
    boxes = [Box(r.box_id, r.inner_l_in, r.inner_w_in, r.inner_h_in, r.tare_lb, r.unit_cost)
             for r in box_df.itertuples(index=False)]
    return lines, boxes


def main() -> None:
    ap = argparse.ArgumentParser(description="Assign the cheapest fitting box to each order")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--wave-file", required=True, help="parquet of order lines for one pick wave")
    ap.add_argument("--wave-id", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(args.dsn)
    lines, boxes = load(engine, args.wave_file)
    rows, unpackable = [], 0
    for order_id, grp in lines.groupby("order_id"):
        items = [Item(r.sku, r.length_in, r.width_in, r.height_in, r.weight_lb, r.category in FRAGILE_CATEGORIES)
                 for r in grp.itertuples(index=False) for _ in range(int(r.qty))]
        best = choose_box(items, boxes)
        if best is None:
            unpackable += 1   # goes to manual multi-carton handling on the floor
            rows.append({"order_id": order_id, "box_id": None, "status": "manual"})
            continue
        rows.append({"order_id": order_id, "box_id": best.box.box_id, "status": "auto",
                     "billable_lb": best.billable_lb, "est_cost": best.cost, "fill_ratio": best.fill_ratio})
    out = pd.DataFrame(rows).assign(wave_id=args.wave_id)
    out.to_sql("box_assignments", engine, schema="fulfillment", if_exists="append", index=False)
    log.info("assigned %d orders, %d need manual packing", len(out) - unpackable, unpackable)


if __name__ == "__main__":
    main()

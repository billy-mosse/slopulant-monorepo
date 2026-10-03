"""Shipping ETA.

Predicts delivery days shown on the product page, per warehouse and
destination zone, from recent carrier transit times.

Input:  logistics.shipments (warehouse, zone, carrier, ship_date, delivered_date)
Output: logistics.eta_days (warehouse, zone, p50_days, p90_days)
"""
from datetime import date
from statistics import quantiles


def transit_days(shipment):
    return (shipment["delivered_date"] - shipment["ship_date"]).days


def eta_table(shipments, lookback_days=60, today=None):
    today = today or date.today()
    by_lane = {}
    for s in shipments:
        if (today - s["ship_date"]).days <= lookback_days and s.get("delivered_date"):
            by_lane.setdefault((s["warehouse"], s["zone"]), []).append(transit_days(s))
    out = {}
    for lane, days in by_lane.items():
        if len(days) < 2:
            out[lane] = (days[0], days[0])
            continue
        q = quantiles(days, n=10)
        out[lane] = (q[4], q[8])
    return out


if __name__ == "__main__":
    d = date
    shipments = [
        {"warehouse": "NJ", "zone": 4, "carrier": "UPS", "ship_date": d(2026, 9, 1), "delivered_date": d(2026, 9, 4)},
        {"warehouse": "NJ", "zone": 4, "carrier": "UPS", "ship_date": d(2026, 9, 2), "delivered_date": d(2026, 9, 6)},
        {"warehouse": "NJ", "zone": 4, "carrier": "FedEx", "ship_date": d(2026, 9, 3), "delivered_date": d(2026, 9, 5)},
    ]
    print(eta_table(shipments, today=d(2026, 9, 20)))


def carrier_mix(shipments):
    """Share of shipments per carrier, to spot lanes that depend on one carrier."""
    counts = {}
    for s in shipments:
        counts[s["carrier"]] = counts.get(s["carrier"], 0) + 1
    total = sum(counts.values())
    return {carrier: n / total for carrier, n in counts.items()}

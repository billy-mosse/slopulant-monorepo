"""Showroom staff shift scheduling.

Builds next week's shift plan for every showroom as a mixed-integer program (PuLP / CBC).

Input tables:
    stores.foot_traffic_forecast  (store_id, ts_hour, expected_visitors)
    stores.staff_availability     (store_id, employee_id, day, available_from, available_to,
                                   hourly_rate, max_hours_week)
Output table:
    stores.shift_plan             (store_id, employee_id, day, shift_start, shift_end, hours, cost)
"""
import argparse
import logging
from datetime import date, timedelta

import pandas as pd
import pulp
from sqlalchemy import create_engine

logger = logging.getLogger("staff_scheduling")

# =============================================================================
# Constants
# =============================================================================
FORECAST_TABLE = "stores.foot_traffic_forecast"
AVAILABILITY_TABLE = "stores.staff_availability"
SHIFT_PLAN_TABLE = "stores.shift_plan"

VISITORS_PER_STAFF = 18        # one associate comfortably handles ~18 visitors / hour
MIN_STAFF_OPEN = 2             # never fewer than two people on the floor (safety policy)
SHIFT_LENGTHS = (4, 6, 8)      # allowed shift lengths in hours
MIN_REST_HOURS = 11            # rest between end of one shift and start of the next
OPEN_HOUR, CLOSE_HOUR = 10, 20


# =============================================================================
# Data loading
# =============================================================================
def load_inputs(engine, week_start: date):
    """Load forecast and availability for the 7 days starting at ``week_start``."""
    week_end = week_start + timedelta(days=7)
    forecast = pd.read_sql(
        f"SELECT store_id, ts_hour, expected_visitors FROM {FORECAST_TABLE} "
        "WHERE ts_hour >= %(s)s AND ts_hour < %(e)s",
        engine, params={"s": week_start, "e": week_end},
    )
    avail = pd.read_sql(
        f"SELECT store_id, employee_id, day, available_from, available_to, hourly_rate, max_hours_week "
        f"FROM {AVAILABILITY_TABLE} WHERE day >= %(s)s AND day < %(e)s",
        engine, params={"s": week_start, "e": week_end},
    )
    forecast["day"] = pd.to_datetime(forecast["ts_hour"]).dt.date
    forecast["hour"] = pd.to_datetime(forecast["ts_hour"]).dt.hour
    return forecast, avail


def required_staff(forecast: pd.DataFrame) -> dict:
    """Map (day, hour) -> required headcount, from expected visitors."""
    req = {}
    for row in forecast.itertuples(index=False):
        if OPEN_HOUR <= row.hour < CLOSE_HOUR:
            need = -(-int(row.expected_visitors) // VISITORS_PER_STAFF)  # ceil division
            req[(row.day, row.hour)] = max(MIN_STAFF_OPEN, need)
    return req


def candidate_shifts(avail: pd.DataFrame) -> list:
    """Enumerate every (employee, day, start, length) shift that fits inside availability."""
    shifts = []
    for row in avail.itertuples(index=False):
        lo = max(OPEN_HOUR, int(row.available_from))
        hi = min(CLOSE_HOUR, int(row.available_to))
        for length in SHIFT_LENGTHS:
            for start in range(lo, hi - length + 1):
                shifts.append((row.employee_id, row.day, start, length, float(row.hourly_rate)))
    return shifts


# =============================================================================
# Model
# =============================================================================
def build_model(store_id, req: dict, shifts: list, max_hours: dict):
    """Create the PuLP problem for one store.

    Decision variables:
        x[s]      binary, shift s is worked
        short[t]  continuous, uncovered headcount at hour t (penalised heavily, keeps model feasible)
    """
    prob = pulp.LpProblem(f"shifts_{store_id}", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(shifts))}
    short = {t: pulp.LpVariable(f"short_{t[0]}_{t[1]}", lowBound=0) for t in req}

    # Objective: labour cost + a penalty well above any wage for every uncovered staff-hour
    prob += (pulp.lpSum(shifts[i][3] * shifts[i][4] * x[i] for i in x)
             + 500 * pulp.lpSum(short.values()))

    # Coverage per hour
    for (day, hour), need in req.items():
        covering = [i for i, (_, d, st, ln, _) in enumerate(shifts) if d == day and st <= hour < st + ln]
        prob += pulp.lpSum(x[i] for i in covering) + short[(day, hour)] >= need, f"cover_{day}_{hour}"

    by_emp = {}
    for i, (emp, day, st, ln, _) in enumerate(shifts):
        by_emp.setdefault(emp, []).append(i)

    for emp, idx in by_emp.items():
        # Weekly hours cap (contract)
        prob += pulp.lpSum(shifts[i][3] * x[i] for i in idx) <= max_hours[emp], f"maxh_{emp}"
        # At most one shift per day
        for day in {shifts[i][1] for i in idx}:
            prob += pulp.lpSum(x[i] for i in idx if shifts[i][1] == day) <= 1, f"oneday_{emp}_{day}"
        # Minimum rest: shift on day d ending at e, next-day shift starting at s needs (24 - e + s) >= MIN_REST
        for i in idx:
            _, d1, s1, l1, _ = shifts[i]
            for j in idx:
                _, d2, s2, _, _ = shifts[j]
                if d2 == d1 + timedelta(days=1) and (24 - (s1 + l1) + s2) < MIN_REST_HOURS:
                    prob += x[i] + x[j] <= 1, f"rest_{i}_{j}"
    return prob, x, short


def solve_store(store_id, forecast, avail, time_limit):
    req = required_staff(forecast)
    shifts = candidate_shifts(avail)
    max_hours = avail.groupby("employee_id")["max_hours_week"].first().to_dict()
    prob, x, short = build_model(store_id, req, shifts, max_hours)
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    logger.info("store %s: status=%s cost=%.0f uncovered_hours=%.1f", store_id,
                pulp.LpStatus[prob.status], pulp.value(prob.objective) or 0,
                sum(v.value() or 0 for v in short.values()))
    rows = []
    for i, var in x.items():
        if var.value() and var.value() > 0.5:
            emp, day, start, length, rate = shifts[i]
            rows.append({"store_id": store_id, "employee_id": emp, "day": day,
                         "shift_start": start, "shift_end": start + length,
                         "hours": length, "cost": length * rate})
    return rows


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Plan showroom shifts for one week")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--week-start", type=date.fromisoformat, required=True)
    parser.add_argument("--time-limit", type=int, default=120, help="CBC seconds per store")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    engine = create_engine(args.dsn)
    forecast, avail = load_inputs(engine, args.week_start)
    plan = []
    for store_id, store_avail in avail.groupby("store_id"):
        store_fc = forecast[forecast["store_id"] == store_id]
        plan.extend(solve_store(store_id, store_fc, store_avail, args.time_limit))

    out = pd.DataFrame(plan)
    out["week_start"] = args.week_start
    schema, table = SHIFT_PLAN_TABLE.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="append", index=False)
    logger.info("wrote %d shifts to %s", len(out), SHIFT_PLAN_TABLE)


if __name__ == "__main__":
    main()

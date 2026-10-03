"""Build search click logs from raw session events.

    spark-submit sessionize.py --date 2026-10-01

Steps
  1. read one day of sessions.events (plus a 30 min lookback so sessions crossing midnight stay whole)
  2. drop bot-like visitors by event rate
  3. split each visitor's stream into search sessions on a 30-minute inactivity gap
  4. explode every search_results_shown event into one row per (query, result position)
  5. attach clicks, add-to-carts and dwell time for each shown product
  6. write to search.click_logs, partitioned by event_date
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession, Window, functions as F

log = logging.getLogger("click_sessionizer")

EVENTS = "sessions.events"
CLICK_LOGS = "search.click_logs"


@dataclass(frozen=True, slots=True)
class SessionizerConfig:
    gap_minutes: int = 30
    max_events_per_minute: float = 40.0   # sustained rate above this is not a human
    max_session_events: int = 5_000
    dwell_cap_seconds: int = 600          # tab left open overnight shouldn't count as engagement
    bot_ua_pattern: str = r"(?i)(bot|crawler|spider|headless|python-requests|curl)"


def read_events(spark: SparkSession, day: dt.date, cfg: SessionizerConfig) -> DataFrame:
    start = dt.datetime.combine(day, dt.time.min) - dt.timedelta(minutes=cfg.gap_minutes)
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min)
    return (
        spark.table(EVENTS)
        .where((F.col("event_ts") >= F.lit(start)) & (F.col("event_ts") < F.lit(end)))
        .select("visitor_id", "event_ts", "event_type", "query_text", "product_id",
                "result_product_ids", "user_agent", "page_url")
    )


def drop_bots(events: DataFrame, cfg: SessionizerConfig) -> DataFrame:
    rates = (
        events.groupBy("visitor_id")
        .agg(F.count("*").alias("n"),
             ((F.max("event_ts").cast("long") - F.min("event_ts").cast("long")) / 60.0).alias("minutes"),
             F.max(F.col("user_agent").rlike(cfg.bot_ua_pattern).cast("int")).alias("ua_bot"))
        .withColumn("rate", F.col("n") / F.greatest(F.col("minutes"), F.lit(1.0)))
    )
    bots = rates.where((F.col("rate") > cfg.max_events_per_minute) | (F.col("ua_bot") == 1)
                       | (F.col("n") > cfg.max_session_events))
    return events.join(bots.select("visitor_id"), "visitor_id", "left_anti")


def assign_sessions(events: DataFrame, cfg: SessionizerConfig) -> DataFrame:
    w = Window.partitionBy("visitor_id").orderBy("event_ts")
    gap = F.col("event_ts").cast("long") - F.lag("event_ts").over(w).cast("long")
    new_session = F.when(gap.isNull() | (gap > cfg.gap_minutes * 60), 1).otherwise(0)
    return (
        events.withColumn("is_new", new_session)
        .withColumn("session_seq", F.sum("is_new").over(w.rowsBetween(Window.unboundedPreceding, 0)))
        .withColumn("session_id", F.concat_ws("-", "visitor_id", "session_seq"))
        .drop("is_new")
    )


def impressions(sess: DataFrame) -> DataFrame:
    """One row per shown result; query_id is unique per search within a session."""
    searches = sess.where(F.col("event_type") == "search_results_shown")
    w = Window.partitionBy("session_id").orderBy("event_ts")
    return (
        searches.withColumn("search_seq", F.row_number().over(w))
        .withColumn("query_id", F.concat_ws(":", "session_id", "search_seq"))
        .withColumn("next_search_ts", F.lead("event_ts").over(w))
        .select("query_id", "session_id", "visitor_id", F.col("event_ts").alias("search_ts"),
                "next_search_ts", F.lower(F.trim("query_text")).alias("query_text"),
                F.posexplode("result_product_ids").alias("pos0", "product_id"))
        .withColumn("position", F.col("pos0") + 1).drop("pos0")
    )


def interactions(sess: DataFrame, cfg: SessionizerConfig) -> DataFrame:
    w = Window.partitionBy("session_id").orderBy("event_ts")
    acts = sess.withColumn("next_ts", F.lead("event_ts").over(w))
    acts = acts.where(F.col("event_type").isin("product_click", "add_to_cart", "product_view"))
    dwell = F.least(F.col("next_ts").cast("long") - F.col("event_ts").cast("long"), F.lit(cfg.dwell_cap_seconds))
    return acts.select("session_id", "product_id", "event_type", "event_ts",
                       F.when(F.col("event_type") == "product_view", dwell).alias("dwell_s"))


def attach(imp: DataFrame, acts: DataFrame) -> DataFrame:
    # an interaction belongs to the most recent search before it that showed the product
    j = imp.alias("i").join(
        acts.alias("a"),
        (F.col("i.session_id") == F.col("a.session_id"))
        & (F.col("i.product_id") == F.col("a.product_id"))
        & (F.col("a.event_ts") >= F.col("i.search_ts"))
        & (F.col("i.next_search_ts").isNull() | (F.col("a.event_ts") < F.col("i.next_search_ts"))),
        "left",
    )
    return (
        j.groupBy("i.query_id", "i.session_id", "i.visitor_id", "i.search_ts", "i.query_text",
                  "i.product_id", "i.position")
        .agg(F.max((F.col("a.event_type") == "product_click").cast("boolean")).alias("clicked"),
             F.max((F.col("a.event_type") == "add_to_cart").cast("boolean")).alias("added_to_cart"),
             F.coalesce(F.sum("a.dwell_s"), F.lit(0)).alias("dwell_seconds"))
        .fillna({"clicked": False, "added_to_cart": False})
        .withColumn("event_date", F.to_date("search_ts"))
    )


def run(spark: SparkSession, day: dt.date, cfg: SessionizerConfig) -> DataFrame:
    events = drop_bots(read_events(spark, day, cfg), cfg)
    sess = assign_sessions(events, cfg)
    logs = attach(impressions(sess), interactions(sess, cfg))
    return logs.where(F.col("event_date") == F.lit(day))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--gap-minutes", type=int, default=SessionizerConfig.gap_minutes)
    ap.add_argument("--max-rate", type=float, default=SessionizerConfig.max_events_per_minute)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = SessionizerConfig(gap_minutes=args.gap_minutes, max_events_per_minute=args.max_rate)
    spark = SparkSession.builder.appName("click_sessionizer").getOrCreate()
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    logs = run(spark, args.date, cfg)
    logs.write.mode("overwrite").partitionBy("event_date").saveAsTable(CLICK_LOGS)
    log.info("wrote %s for %s", CLICK_LOGS, args.date)


if __name__ == "__main__":
    main()

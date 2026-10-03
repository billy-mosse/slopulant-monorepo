-- Weekly units and average realized price per SKU.
-- Price comes from the price history (list price in effect), units from order lines.
-- Weeks with zero sales are dropped: log(units) is undefined there.
WITH weekly_units AS (
    SELECT
        l.sku,
        DATE_TRUNC('week', l.order_ts)          AS week_start,
        SUM(l.quantity)                         AS units,
        SUM(l.quantity * l.unit_price)          AS revenue
    FROM orders.lines l
    WHERE l.order_ts >= DATEADD('week', -:lookback_weeks, CURRENT_DATE)
      AND l.is_return = FALSE
    GROUP BY 1, 2
),
weekly_price AS (
    SELECT
        ph.sku,
        DATE_TRUNC('week', ph.effective_date)   AS week_start,
        AVG(ph.price)                           AS list_price,
        MAX(ph.category)                        AS category
    FROM pricing.price_history ph
    GROUP BY 1, 2
)
SELECT
    u.sku,
    p.category,
    u.week_start,
    EXTRACT(WEEK FROM u.week_start)             AS week_of_year,
    u.units,
    COALESCE(u.revenue / NULLIF(u.units, 0), p.list_price) AS avg_price
FROM weekly_units u
JOIN weekly_price p
  ON p.sku = u.sku
 AND p.week_start = u.week_start
WHERE u.units > 0;

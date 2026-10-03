# Customer segments

Weekly refresh of marketing segments, written to `marketing.segments` (one row per customer).

## How it's built
1. Take the latest shared customer vectors from `features.customer_embeddings`.
2. Standardize, reduce to 5 PCA components.
3. Run k-means for k = 4..10 and keep the k with the best silhouette score.
4. Name each cluster from its centroid profile (recency, orders, spend, category shares)
   using the rule table in `segment.py`, e.g. "Bedding loyalists", "Lapsed high spenders".

## How marketing should use it
- **Targeting:** filter campaign audiences on `segment_name`, not `segment_id` — ids can
  change between refreshes when k changes.
- **Lapsed high spenders:** win-back emails; avoid deep discounts, they respond to newness.
- **Bedding loyalists / Bath enthusiasts:** category launches and cross-sell into the other.
- **New & curious:** onboarding series, no discounts in the first 30 days.
- `distance_to_centroid` is a rough "how typical" score; use the closest 50% for creative tests.

Segments are descriptive, not predictive: they say who a customer looks like today, not what they will do next.

Owner: Grace Liu (marketing analytics). Run: `python segment.py --dsn $WAREHOUSE_DSN`

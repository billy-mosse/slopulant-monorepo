# Homepage personalization

Decides which homepage modules a customer sees, in what order, and which products fill each one.

**Modules:** trending, because you viewed, new arrivals, sale (12 slots each).

**Inputs**
- `features.customer_embeddings` — per-customer category-mix vector and recently viewed SKUs
- `recs.trending` — latest per-category trending velocity
- `recs.i2i` — item-to-item similarity, seeds "because you viewed"
- merchandising feed (parquet) — new-arrival and sale SKUs

**Output:** `recs.homepage_slots` (customer_id, module, module_rank, skus)

**Scoring:** `0.65 * category affinity + 0.35 * sigmoid(trending z)`, then MMR (lambda = 0.7)
so one category cannot take over a module. Modules are ordered by mean relevance of their picks.

**Cold start:** customers with no meaningful category mix get the trending module only, ranked by velocity.

Run nightly:

    python rank_modules.py --dsn $WAREHOUSE_DSN --merch-feed merch.parquet --categories categories.json

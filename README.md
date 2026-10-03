# Slopulent Living — ML monorepo

Each top-level folder is an independently owned system.

| Folder | Area | What it does |
|---|---|---|
| `product_matching/` | Catalog cleanup | Finds duplicate listings to merge |
| `tag_validation/` | Catalog cleanup | Normalizes merchant tags to the controlled vocabulary |
| `product_embeddings/` | Catalog cleanup | Shared product embedding table |
| `review_summaries/` | Catalog cleanup | "What customers say" blurbs |
| `review_moderation/` | Catalog cleanup | Flags spam/abusive reviews |
| `two_tower_search/` | Search & recs | Query → product retrieval |
| `i2i_recs/` | Search & recs | "You may also like" |
| `price_model/` | Pricing | Suggested list price |
| `customer_intent/` | Customer | Session p(conversion) |
| `customer_embeddings/` | Customer | Customer taste/value vectors |

# Slopulent Living — ML monorepo

Each top-level folder is an independently owned system. Owners are the main authors in git history.

| Folder | Team | What it does |
|---|---|---|
| `attribute_extraction/` | Catalog & content | Material/size/etc. from text |
| `category_classifier/` | Catalog & content | 3-level taxonomy classifier |
| `color_extraction/` | Catalog & content | Dominant named colors per product |
| `image_quality/` | Catalog & content | Scores product images |
| `product_embeddings/` | Catalog & content | Shared product embedding table |
| `product_matching/` | Catalog & content | Finds duplicate listings to merge |
| `tag_validation/` | Catalog & content | Normalizes merchant tags |
| `title_rewriter/` | Catalog & content | SEO titles from attributes |
| `autocomplete/` | Search & recommendations | Search-box suggestions |
| `cart_recs/` | Search & recommendations | Cart add-ons from association rules |
| `homepage_personalization/` | Search & recommendations | Per-customer homepage modules |
| `i2i_recs/` | Search & recommendations | "You may also like" |
| `query_understanding/` | Search & recommendations | Spell correction + query intent |
| `search_ranker/` | Search & recommendations | LambdaMART second-stage ranker |
| `trending_products/` | Search & recommendations | Trending per category |
| `two_tower_search/` | Search & recommendations | Query → product retrieval |
| `competitor_price_matching/` | Pricing | Our SKUs ↔ competitor listings |
| `elasticity_estimation/` | Pricing | Price elasticity per category |
| `markdown_optimizer/` | Pricing | End-of-season markdown plan |
| `price_model/` | Pricing | Suggested list price |
| `churn_prediction/` | Customer analytics | 90-day churn probability |
| `clv_model/` | Customer analytics | Customer lifetime value |
| `customer_embeddings/` | Customer analytics | Customer taste/value vectors |
| `customer_intent/` | Customer analytics | Session p(conversion) |
| `customer_segmentation/` | Marketing | Marketing segments |
| `email_product_recs/` | Marketing | Recs for post-purchase email |
| `email_send_time/` | Marketing | Best email hour per customer |
| `next_best_offer/` | Marketing | Offer selection (contextual bandit) |
| `promo_uplift/` | Marketing | Who should get a promo (uplift) |
| `demand_forecast/` | Supply chain | Weekly SKU demand forecast |
| `replenishment/` | Supply chain | Reorder points and POs |
| `returns_reason_classifier/` | Supply chain | Return comment → reason code |
| `fake_review_detection/` | Reviews & trust | Fake review rings |
| `review_moderation/` | Reviews & trust | Flags spam/abusive reviews |
| `review_sentiment/` | Reviews & trust | Aspect-based review sentiment |
| `review_summaries/` | Reviews & trust | "What customers say" blurbs |
| `ab_test_analysis/` | ML platform | Standard A/B test analysis |
| `feature_store/` | ML platform | Daily feature materialization |
| `model_monitoring/` | ML platform | Model/feature drift alerts |

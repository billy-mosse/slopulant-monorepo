# Marketing models

Small models owned by the marketing data science team. Each file is standalone
(`python <file>.py --help`) and runs on the analytics warehouse.

| File | What it answers | Reads | Writes |
|------|-----------------|-------|--------|
| `lookalikes.py` | "Who looks like our best customers?" Seed-centroid cosine similarity, capped audience size | `features.customer_embeddings` | `marketing.lookalike_audiences` |
| `subject_lines.py` | "Which subject line will get more opens?" Ridge regression on text features + TF-IDF | `marketing.email_events` | `marketing.subject_line_scores` |
| `coupon_propensity.py` | "Will this customer redeem a coupon, and at what depth?" Logistic regression on promo history | `marketing.promo_history` | `marketing.coupon_propensity` |

## For campaign managers

- **Lookalikes**: send us a CSV of seed customer IDs (at least 25). Default audience
  cap is 50k; ask if you need more. Audiences below 0.55 similarity are cut.
- **Subject lines**: drop candidate lines in a text file; scores are relative, so
  use them to rank variants, not to forecast exact open rates.
- **Coupons**: scores refresh weekly for 10/15/20/25% off. Low scores at every
  depth usually mean "don't send"; high at 10% means "a small nudge is enough".

Questions: #marketing-data (Grace Liu).

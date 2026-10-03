# Image quality scoring

Nightly job that scores every active product image 0-100 so merchandisers can find and replace weak photos.

**Input:** `catalog.product_images` (decoded pixels, height, width)
**Output:** `catalog.image_quality_scores` (one row per image_id)

## Components

| Component  | Weight | Signal |
|------------|--------|--------|
| blur       | 0.35   | variance of Laplacian response on luminance |
| exposure   | 0.25   | mean luminance + share of clipped pixels |
| background | 0.25   | share of near-white pixels in the border band |
| resolution | 0.15   | min side vs 1000px, aspect ratio in 0.75-1.34 |

## Score bands

- **80-100** good, no action
- **60-79** acceptable, review when convenient
- **40-59** poor, queue for reshoot
- **0-39** unusable, hide from PDP hero slot

## Reason codes

`BLURRY`, `DARK`, `BUSY_BACKGROUND`, `LOW_RES` (comma-separated in `reasons`).

## Run

```
python score_images.py --dsn $CATALOG_DSN [--limit 500 --dry-run]
```

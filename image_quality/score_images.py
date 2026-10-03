"""Score product images for merchandiser review.

Reads decoded images from catalog.product_images, writes one row per image
to catalog.image_quality_scores with a 0-100 score and reason codes.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("score_images")

SOURCE_TABLE = "catalog.product_images"
TARGET_TABLE = "catalog.image_quality_scores"

LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

BLUR_VAR_MIN = 120.0        # Laplacian variance below this -> BLURRY
DARK_MEAN_MIN = 0.35        # mean luminance in [0, 1]
CLIP_SHARE_MAX = 0.08       # share of pixels at 0 or 1
WHITE_THRESH = 0.92         # luminance counted as "near white"
BORDER_FRAC = 0.06          # border band width as fraction of min side
WHITE_SHARE_MIN = 0.85
MIN_SIDE_PX = 1000
ASPECT_RANGE = (0.75, 1.34)

WEIGHTS = {"blur": 0.35, "exposure": 0.25, "background": 0.25, "resolution": 0.15}


@dataclass
class ImageScore:
    image_id: str
    sku: str
    blur_var: float
    luma_mean: float
    clip_share: float
    white_share: float
    width: int
    height: int
    score: float
    reasons: str


def luminance(img: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> HxW float in [0, 1]."""
    rgb = img[..., :3].astype(np.float32) / 255.0
    return rgb @ LUMA


def convolve2d(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Valid-mode 2D convolution via strided windows."""
    kh, kw = k.shape
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw))
    return np.einsum("ijkl,kl->ij", win, k[::-1, ::-1])


def blur_variance(y: np.ndarray) -> float:
    """Variance of the Laplacian response, on 0-255 scale."""
    return float(convolve2d(y * 255.0, LAPLACIAN).var())


def exposure_stats(y: np.ndarray) -> tuple[float, float]:
    """Mean luminance and share of clipped pixels (crushed or blown)."""
    clipped = (y <= 0.01) | (y >= 0.99)
    return float(y.mean()), float(clipped.mean())


def border_white_share(y: np.ndarray) -> float:
    """Share of near-white pixels in a band around the image edge."""
    h, w = y.shape
    b = max(1, int(min(h, w) * BORDER_FRAC))
    mask = np.zeros_like(y, dtype=bool)
    mask[:b, :] = mask[-b:, :] = True
    mask[:, :b] = mask[:, -b:] = True
    return float((y[mask] >= WHITE_THRESH).mean())


def sub_scores(bv: float, mean: float, clip: float, white: float, h: int, w: int) -> dict[str, float]:
    """Each component mapped to [0, 1]."""
    blur = np.clip(np.log1p(bv) / np.log1p(4 * BLUR_VAR_MIN), 0, 1)
    expo = np.clip(1 - abs(mean - 0.6) / 0.6, 0, 1) * np.clip(1 - clip / (2 * CLIP_SHARE_MAX), 0, 1)
    bg = np.clip(white / WHITE_SHARE_MIN, 0, 1)
    ar = w / h
    res = np.clip(min(h, w) / MIN_SIDE_PX, 0, 1) * (1.0 if ASPECT_RANGE[0] <= ar <= ASPECT_RANGE[1] else 0.6)
    return {"blur": float(blur), "exposure": float(expo), "background": float(bg), "resolution": float(res)}


def reason_codes(bv: float, mean: float, clip: float, white: float, h: int, w: int) -> list[str]:
    out = []
    if bv < BLUR_VAR_MIN:
        out.append("BLURRY")
    if mean < DARK_MEAN_MIN or clip > CLIP_SHARE_MAX:
        out.append("DARK")
    if white < WHITE_SHARE_MIN:
        out.append("BUSY_BACKGROUND")
    ar = w / h
    if min(h, w) < MIN_SIDE_PX or not (ASPECT_RANGE[0] <= ar <= ASPECT_RANGE[1]):
        out.append("LOW_RES")
    return out


def score_image(image_id: str, sku: str, img: np.ndarray) -> ImageScore:
    """Full scoring for one decoded image."""
    h, w = img.shape[:2]
    y = luminance(img)
    bv = blur_variance(y)
    mean, clip = exposure_stats(y)
    white = border_white_share(y)
    subs = sub_scores(bv, mean, clip, white, h, w)
    score = 100.0 * sum(WEIGHTS[k] * v for k, v in subs.items())
    reasons = reason_codes(bv, mean, clip, white, h, w)
    return ImageScore(image_id, sku, bv, mean, clip, white, w, h, round(score, 1), ",".join(reasons))


def load_images(engine, limit: int | None) -> pd.DataFrame:
    sql = f"SELECT image_id, sku, pixels, height, width FROM {SOURCE_TABLE} WHERE is_active"
    if limit:
        sql += f" LIMIT {limit}"
    return pd.read_sql(sql, engine)


def decode_row(row: pd.Series) -> np.ndarray:
    """Pixels are stored as raw uint8 bytes, HxWx3."""
    return np.frombuffer(row.pixels, dtype=np.uint8).reshape(int(row.height), int(row.width), 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score product images.")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    engine = create_engine(args.dsn)
    df = load_images(engine, args.limit)
    log.info("loaded %d images from %s", len(df), SOURCE_TABLE)

    rows = [score_image(r.image_id, r.sku, decode_row(r)).__dict__ for r in df.itertuples()]
    out = pd.DataFrame(rows)
    log.info("mean score %.1f, flagged %d", out.score.mean(), (out.reasons != "").sum())

    if args.dry_run:
        print(out.head(20).to_string(index=False))
        return
    schema, table = TARGET_TABLE.split(".")
    out.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    log.info("wrote %d rows to %s", len(out), TARGET_TABLE)


if __name__ == "__main__":
    main()

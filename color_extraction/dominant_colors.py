"""Dominant product colors per SKU.

Input:  catalog.product_images
Output: catalog.product_colors  (sku, color_1, share_1, color_2, share_2)
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from palette import nearest

log = logging.getLogger("dominant_colors")

SRC = "catalog.product_images"
DST = "catalog.product_colors"

K = 4
N_ITER = 8
BG_MIN_RGB = 235            # all channels >= this -> background
MAX_PIXELS = 20_000         # subsample per image
SEED = 7

# sRGB D65 -> XYZ
M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 sRGB -> (N, 3) CIE Lab."""
    xyz = srgb_to_linear(rgb.astype(np.float64) / 255.0) @ M_RGB2XYZ.T
    t = xyz / WHITE_D65
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def foreground_pixels(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Drop near-white background, subsample."""
    px = img[..., :3].reshape(-1, 3)
    px = px[~(px >= BG_MIN_RGB).all(1)]
    if len(px) > MAX_PIXELS:
        px = px[rng.choice(len(px), MAX_PIXELS, replace=False)]
    return px


def kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++: next center sampled with p ~ D(x)^2."""
    c = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = ((x[:, None, :] - np.array(c)[None]) ** 2).sum(-1).min(1)
        c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    return np.array(c)


def kmeans(x: np.ndarray, k: int, n_iter: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd iterations. Returns centroids (k, 3) and labels (N,)."""
    c = kmeans_pp_init(x, k, rng)
    for _ in range(n_iter):
        lab = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c, atol=1e-3):
            break
        c = new
    return c, lab


def image_colors(img: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Named color -> pixel share for one image."""
    px = foreground_pixels(img, rng)
    if len(px) < K * 10:
        return {}
    lab = rgb_to_lab(px)
    cents, labels = kmeans(lab, K, N_ITER, rng)
    shares = np.bincount(labels, minlength=K) / len(labels)
    out: dict[str, float] = defaultdict(float)
    for c, s in zip(cents, shares):
        out[nearest(c)[0]] += float(s)
    return out


def sku_top2(images: list[np.ndarray], rng: np.random.Generator) -> list[tuple[str, float]]:
    """Average shares across a SKU's images, keep top 2."""
    acc: dict[str, float] = defaultdict(float)
    for img in images:
        for name, s in image_colors(img, rng).items():
            acc[name] += s / len(images)
    return sorted(acc.items(), key=lambda kv: -kv[1])[:2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eng = create_engine(args.dsn)
    q = f"SELECT sku, pixels, height, width FROM {SRC} WHERE is_active ORDER BY sku"
    df = pd.read_sql(q + (f" LIMIT {args.limit}" if args.limit else ""), eng)
    log.info("loaded %d images", len(df))

    rng = np.random.default_rng(SEED)
    rows = []
    for sku, g in df.groupby("sku"):
        imgs = [np.frombuffer(r.pixels, np.uint8).reshape(int(r.height), int(r.width), 3) for r in g.itertuples()]
        top = sku_top2(imgs, rng) + [(None, 0.0)] * 2
        rows.append({"sku": sku, "color_1": top[0][0], "share_1": round(top[0][1], 3),
                     "color_2": top[1][0], "share_2": round(top[1][1], 3)})

    out = pd.DataFrame(rows)
    schema, table = DST.split(".")
    out.to_sql(table, eng, schema=schema, if_exists="replace", index=False)
    log.info("wrote %d skus to %s", len(out), DST)


if __name__ == "__main__":
    main()

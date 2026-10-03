"""Embed primary product images with ResNet-50 (penultimate layer, L2-normalized)."""
import argparse
import io
import logging
import os
from itertools import islice

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from sqlalchemy import create_engine
from torchvision import models, transforms

log = logging.getLogger("embed")

IMAGES_SQL = """
SELECT sku, category_id, image_url
FROM catalog.product_images
WHERE is_primary AND image_url IS NOT NULL
"""
BATCH = 64
DIM = 2048

prep = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(device):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()  # drop classifier -> 2048-d pooled features
    return m.eval().to(device)


def fetch(url, session):
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:  # broken CDN links happen, skip them
        log.warning("skip %s: %s", url, e)
        return None


def chunks(it, n):
    it = iter(it)
    while batch := list(islice(it, n)):
        yield batch


@torch.inference_mode()
def embed(rows, model, device):
    s = requests.Session()
    skus, cats, vecs = [], [], []
    for i, batch in enumerate(chunks(rows, BATCH)):
        imgs = [(sku, cat, fetch(url, s)) for sku, cat, url in batch]
        imgs = [x for x in imgs if x[2] is not None]
        if not imgs:
            continue
        x = torch.stack([prep(im) for _, _, im in imgs]).to(device)
        v = model(x).cpu().numpy().astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        skus += [k for k, _, _ in imgs]
        cats += [c for _, c, _ in imgs]
        vecs.append(v)
        if i % 20 == 0:
            log.info("batch %d, %d embedded", i, len(skus))
    return np.array(skus), np.array(cats), np.vstack(vecs) if vecs else np.zeros((0, DIM), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="image_vectors.npz")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    df = pd.read_sql(IMAGES_SQL, create_engine(os.environ["WAREHOUSE_URL"]))
    df = df.drop_duplicates("sku")
    if a.limit:
        df = df.head(a.limit)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    skus, cats, vecs = embed(df[["sku", "category_id", "image_url"]].itertuples(index=False), load_model(dev), dev)
    np.savez_compressed(a.out, sku=skus, category=cats, vec=vecs)
    log.info("wrote %s: %d x %d", a.out, *vecs.shape)


if __name__ == "__main__":
    main()

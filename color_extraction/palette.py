"""Merchandising color palette in CIE Lab (D65)."""
from __future__ import annotations

import numpy as np

PALETTE: dict[str, tuple[float, float, float]] = {
    "white":      (97.0, 0.0, 1.5),
    "ivory":      (95.0, -1.0, 9.0),
    "oat":        (82.0, 2.0, 14.0),
    "sand":       (74.0, 4.0, 20.0),
    "camel":      (60.0, 12.0, 32.0),
    "rust":       (45.0, 35.0, 40.0),
    "terracotta": (52.0, 30.0, 30.0),
    "blush":      (82.0, 14.0, 8.0),
    "rose":       (62.0, 32.0, 10.0),
    "burgundy":   (28.0, 35.0, 12.0),
    "mustard":    (68.0, 5.0, 58.0),
    "sage":       (68.0, -12.0, 14.0),
    "olive":      (48.0, -8.0, 30.0),
    "forest":     (32.0, -22.0, 10.0),
    "eucalyptus": (60.0, -18.0, 2.0),
    "sky":        (78.0, -6.0, -16.0),
    "denim":      (45.0, 0.0, -25.0),
    "navy":       (22.0, 6.0, -30.0),
    "lavender":   (72.0, 12.0, -18.0),
    "plum":       (32.0, 25.0, -12.0),
    "dove":       (76.0, 0.0, 1.0),
    "stone":      (62.0, 1.0, 5.0),
    "slate":      (48.0, -3.0, -8.0),
    "charcoal":   (30.0, 0.0, -1.0),
    "black":      (12.0, 0.0, 0.0),
    "walnut":     (35.0, 12.0, 18.0),
}

NAMES: list[str] = list(PALETTE)
LAB: np.ndarray = np.array([PALETTE[n] for n in NAMES], dtype=np.float64)


def delta_e76(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 Delta-E between (..., 3) Lab arrays."""
    return np.sqrt(((a - b) ** 2).sum(-1))


def nearest(lab: np.ndarray) -> tuple[str, float]:
    """Closest palette name and its Delta-E for one Lab triple."""
    d = delta_e76(LAB, lab[None, :])
    i = int(d.argmin())
    return NAMES[i], float(d[i])

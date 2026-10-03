"""MinHash signatures + LSH banding for near-duplicate review text.

Jaccard is computed over word shingles (k=3). With b bands of r rows the
probability two docs with Jaccard s collide in some band is 1 - (1 - s^r)^b;
the defaults (b=20, r=5) put the threshold near (1/b)^(1/r) ~= 0.55.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from itertools import combinations

import numpy as np

MERSENNE = (1 << 61) - 1
SHINGLE_K = 3
NUM_PERM = 100
BANDS = 20
ROWS = NUM_PERM // BANDS
MAX_BUCKET = 200  # template spam ("great product!") makes huge buckets; skip them


def shingles(text: str, k: int = SHINGLE_K) -> set[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    if not words:
        return set()
    if len(words) < k:
        return {" ".join(words)}  # very short reviews: one shingle, still comparable
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


class MinHasher:
    def __init__(self, num_perm: int = NUM_PERM, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MERSENNE, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, MERSENNE, num_perm, dtype=np.uint64)

    def signature(self, sh: set[str]) -> np.ndarray | None:
        if not sh:
            return None
        hv = np.array([_hash64(s) for s in sh], dtype=np.uint64) % MERSENNE
        # h_i(x) = (a_i * x + b_i) mod p ; uses object dtype to avoid uint64 overflow
        prod = (np.outer(self.a.astype(object), hv.astype(object)) + self.b.astype(object)[:, None]) % MERSENNE
        return prod.min(axis=1).astype(np.uint64)


def lsh_candidates(sigs: dict[str, np.ndarray], bands: int = BANDS) -> set[tuple[str, str]]:
    rows = NUM_PERM // bands
    pairs: set[tuple[str, str]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[str]] = defaultdict(list)
        for key, sig in sigs.items():
            buckets[sig[band * rows:(band + 1) * rows].tobytes()].append(key)
        for members in buckets.values():
            if 1 < len(members) <= MAX_BUCKET:
                pairs.update(tuple(sorted(p)) for p in combinations(members, 2))
    return pairs


def near_duplicates(texts: dict[str, str], threshold: float = 0.6) -> list[tuple[str, str, float]]:
    """Return (id_a, id_b, jaccard) pairs; LSH candidates are verified with exact Jaccard."""
    sh = {k: shingles(v) for k, v in texts.items()}
    mh = MinHasher()
    sigs = {k: s for k, s in ((k, mh.signature(v)) for k, v in sh.items()) if s is not None}
    out = []
    for a, b in lsh_candidates(sigs):
        j = jaccard(sh[a], sh[b])
        if j >= threshold:
            out.append((a, b, j))
    return out

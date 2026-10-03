"""Home-goods product taxonomy (3 levels: department > category > subcategory).

Paths are written as 'L1 > L2 > L3'. Helper functions expose parent/child
lookups used to enforce hierarchical consistency at prediction time.
"""
from __future__ import annotations

TAXONOMY: dict[str, dict[str, list[str]]] = {
    "Bedding": {
        "Sheets": ["Fitted Sheets", "Flat Sheets", "Sheet Sets", "Pillowcases"],
        "Duvet Covers": ["Duvet Cover Sets", "Duvet Covers Only"],
        "Comforters & Quilts": ["Comforters", "Quilts", "Coverlets"],
        "Pillows & Inserts": ["Bed Pillows", "Duvet Inserts", "Mattress Toppers"],
        "Blankets & Throws": ["Throws", "Bed Blankets"],
    },
    "Bath": {
        "Towels": ["Bath Towels", "Hand Towels", "Washcloths", "Towel Sets"],
        "Bath Mats & Rugs": ["Bath Mats", "Bath Rugs"],
        "Shower": ["Shower Curtains", "Shower Liners"],
    },
    "Decor": {
        "Pillows": ["Throw Pillows", "Pillow Covers", "Lumbar Pillows"],
        "Rugs": ["Area Rugs", "Runners", "Rug Pads"],
        "Wall Decor": ["Mirrors", "Framed Art", "Wall Hangings"],
        "Lighting": ["Table Lamps", "Floor Lamps", "Pendants"],
        "Vases & Objects": ["Vases", "Candles", "Decorative Bowls"],
    },
}

def level2(parent: str | None = None) -> list[str]:
    if parent is not None:
        return list(TAXONOMY[parent])
    return [l2 for l1 in TAXONOMY for l2 in TAXONOMY[l1]]


def level3(parent: str | None = None) -> list[str]:
    if parent is not None:
        return list(TAXONOMY[PARENT_OF_L2[parent]][parent])
    return [l3 for l1 in TAXONOMY for l2 in TAXONOMY[l1] for l3 in TAXONOMY[l1][l2]]


PARENT_OF_L2: dict[str, str] = {l2: l1 for l1, d in TAXONOMY.items() for l2 in d}
PARENT_OF_L3: dict[str, str] = {l3: l2 for d in TAXONOMY.values() for l2, l3s in d.items() for l3 in l3s}


def split_path(path: str) -> tuple[str, str, str]:
    """'Bedding > Sheets > Fitted Sheets' -> ('Bedding', 'Sheets', 'Fitted Sheets')."""
    parts = [p.strip() for p in path.split(">")]
    if len(parts) != 3:
        raise ValueError(f"expected 3-level path, got {path!r}")
    return parts[0], parts[1], parts[2]


def is_valid(l1: str, l2: str | None, l3: str | None) -> bool:
    if l1 not in TAXONOMY:
        return False
    if l2 is not None and PARENT_OF_L2.get(l2) != l1:
        return False
    return l3 is None or PARENT_OF_L3.get(l3) == l2

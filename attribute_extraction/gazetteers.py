"""Gazetteers: canonical value -> lowercase surface forms merchants actually type."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MATERIALS: dict[str, list[str]] = {
    "linen": ["linen", "flax", "french linen", "belgian linen", "stonewashed linen"],
    "cotton": ["cotton", "organic cotton", "supima", "pima", "egyptian cotton", "percale", "sateen"],
    "velvet": ["velvet", "velour", "crushed velvet"],
    "wool": ["wool", "merino", "lambswool", "boiled wool"],
}

BED_SIZES: dict[str, list[str]] = {
    "Twin": ["twin", "single"],
    "Twin XL": ["twin xl", "twin extra long", "txl"],
    "Full": ["full", "double"],
    "Queen": ["queen", "qn"],
    "King": ["king", "eastern king", "ek"],
    "Cal King": ["cal king", "california king", "ck", "western king"],
}

PATTERNS: dict[str, list[str]] = {
    "solid": ["solid", "plain", "classic"],
    "striped": ["stripe", "striped", "pinstripe", "ticking"],
    "gingham": ["gingham", "check", "checked", "plaid", "tartan"],
    "floral": ["floral", "botanical", "flower", "blossom"],
    "geometric": ["geometric", "geo", "diamond", "chevron", "herringbone"],
}

CARE: dict[str, list[str]] = {
    "machine_wash_cold": ["machine wash cold", "wash cold", "cold wash"],
    "tumble_dry_low": ["tumble dry low", "dry low", "low heat"],
    "dry_clean_only": ["dry clean only", "professional dry clean"],
}


@dataclass
class Gazetteer:
    """Compiled lookup over a canonical -> synonyms map.

    Surface forms are tried longest first, with word boundaries so 'king'
    does not match inside 'kingston'. ``reverse`` maps surface -> canonical.
    """

    name: str
    entries: dict[str, list[str]]
    pattern: re.Pattern = field(init=False)
    reverse: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.reverse = {s: canon for canon, forms in self.entries.items() for s in forms}
        alts = sorted(self.reverse, key=len, reverse=True)
        body = "|".join(re.escape(a).replace(r"\ ", r"[\s\-]+") for a in alts)
        self.pattern = re.compile(rf"\b(?:{body})\b", re.IGNORECASE)

    def find_all(self, text: str) -> list[tuple[str, str, int]]:
        """Return (canonical, surface, start) for each non-overlapping match."""
        out = []
        for m in self.pattern.finditer(text):
            surface = re.sub(r"[\s\-]+", " ", m.group(0).lower())
            out.append((self.reverse.get(surface, surface), m.group(0), m.start()))
        return out

    def canonical(self, surface: str) -> str | None:
        """Normalize a single surface form, or None if unknown."""
        return self.reverse.get(re.sub(r"[\s\-]+", " ", surface.strip().lower()))


MATERIAL_GAZ = Gazetteer("material", MATERIALS)
SIZE_GAZ = Gazetteer("size", BED_SIZES)
PATTERN_GAZ = Gazetteer("pattern", PATTERNS)
CARE_GAZ = Gazetteer("care", CARE)

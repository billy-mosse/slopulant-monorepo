"""Per-category SEO title templates.

Each template is an ordered list of slots. ``drop_order`` lists the slots to
remove, first to last, when a rendered title exceeds the character limit.
Slots in ``required`` are never dropped; if any is missing the original
title is kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_TITLE_CHARS = 70


@dataclass(frozen=True)
class TitleTemplate:
    """Slot-based template: ``{slot}`` placeholders plus literal separators."""

    pattern: str
    required: tuple[str, ...]
    drop_order: tuple[str, ...]

    @property
    def slots(self) -> list[str]:
        return re.findall(r"\{(\w+)\}", self.pattern)


TEMPLATES: dict[str, TitleTemplate] = {
    "Sheets": TitleTemplate(
        "{brand} {thread_count} {material} {product_type}, {size}, {color}",
        required=("material", "product_type"),
        drop_order=("thread_count", "brand", "color", "size"),
    ),
    "Duvet Covers": TitleTemplate(
        "{brand} {material} {pattern} {product_type}, {size}, {color}",
        required=("material", "product_type", "size"),
        drop_order=("pattern", "brand", "color"),
    ),
    "Towels": TitleTemplate(
        "{brand} {material} {pattern} {product_type}, {color}",
        required=("product_type",),
        drop_order=("pattern", "brand", "material"),
    ),
    "Pillows": TitleTemplate(
        "{brand} {material} {pattern} {product_type}, {dimensions}, {color}",
        required=("product_type",),
        drop_order=("dimensions", "pattern", "brand", "material"),
    ),
    "Rugs": TitleTemplate(
        "{brand} {material} {pattern} {product_type}, {dimensions}, {color}",
        required=("product_type", "dimensions"),
        drop_order=("pattern", "brand", "color", "material"),
    ),
}

DEFAULT_TEMPLATE = TitleTemplate(
    "{brand} {material} {product_type}, {color}",
    required=("product_type",),
    drop_order=("brand", "material", "color"),
)


def template_for(category_l2: str | None) -> TitleTemplate:
    """Template for a level-2 category, falling back to the generic one."""
    return TEMPLATES.get(category_l2 or "", DEFAULT_TEMPLATE)

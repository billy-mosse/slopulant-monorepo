"""Lexicons for aspect-based review sentiment.

Polarity values are in [-1, 1]. Multi-word entries are matched before single
tokens. Keep entries lowercase; apostrophes are normalised to ASCII upstream.
"""

ASPECTS: dict[str, tuple[str, ...]] = {
    "softness": ("soft", "softness", "plush", "silky", "scratchy", "rough", "itchy", "smooth",
                 "hand feel", "feel"),
    "color": ("color", "colour", "shade", "hue", "faded", "fade", "dye", "bleed"),
    "size_fit": ("size", "fit", "fits", "sizing", "too small", "too big", "pocket depth",
                 "deep pocket", "shrink", "shrank", "shrunk"),
    "durability": ("durable", "durability", "pilling", "pills", "pilled", "tear", "torn", "rip",
                   "seams", "holes", "lasted", "wore out", "fell apart"),
    "smell": ("smell", "smells", "odor", "odour", "scent", "chemical smell", "musty"),
    "shipping": ("shipping", "delivery", "arrived", "package", "packaging", "box", "courier"),
}

POLARITY: dict[str, float] = {
    # generic
    "love": 0.9, "loved": 0.9, "great": 0.7, "good": 0.5, "nice": 0.4, "perfect": 0.9,
    "amazing": 0.9, "excellent": 0.9, "fine": 0.2, "ok": 0.1, "okay": 0.1,
    "bad": -0.6, "terrible": -0.9, "awful": -0.9, "horrible": -0.9, "poor": -0.6,
    "disappointed": -0.7, "disappointing": -0.7, "cheap": -0.4, "worst": -1.0,
    # tactile
    "soft": 0.6, "plush": 0.6, "silky": 0.6, "smooth": 0.5, "cozy": 0.6, "cosy": 0.6,
    "scratchy": -0.7, "rough": -0.6, "itchy": -0.7, "stiff": -0.5, "thin": -0.4,
    # appearance
    "vibrant": 0.6, "beautiful": 0.8, "accurate": 0.4, "faded": -0.6, "dull": -0.5, "bled": -0.7,
    # fit / durability
    "fits": 0.4, "snug": 0.3, "shrank": -0.7, "shrunk": -0.7, "pilling": -0.7, "pilled": -0.7,
    "durable": 0.6, "sturdy": 0.6, "flimsy": -0.6, "torn": -0.8, "ripped": -0.8,
    "fell apart": -0.9, "wore out": -0.7, "holds up": 0.6,
    # smell
    "fresh": 0.5, "musty": -0.7, "chemical smell": -0.7, "stinks": -0.8, "odor": -0.4,
    # shipping
    "fast": 0.5, "quick": 0.5, "on time": 0.4, "late": -0.6, "delayed": -0.6, "damaged": -0.8,
    "crushed": -0.7, "lost": -0.7,
}

NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't", "didn't",
    "hardly", "barely", "nothing", "neither", "nor", "without",
})

# Multiplicative weights. Downtoners (<1) are intensifiers too, just weaker.
INTENSIFIERS: dict[str, float] = {
    "very": 1.5, "really": 1.4, "so": 1.3, "super": 1.6, "extremely": 1.8, "incredibly": 1.8,
    "too": 1.3, "absolutely": 1.6, "quite": 1.2, "somewhat": 0.7, "slightly": 0.6,
    "a bit": 0.7, "kind of": 0.7, "kinda": 0.7, "a little": 0.7,
}

# A negator flips and dampens; "not great" is milder than "terrible".
NEGATION_FACTOR = -0.7

# Clause boundaries stop negation / intensifier scope.
SCOPE_BREAKERS: frozenset[str] = frozenset({"but", "however", "although", "though", "yet", ".", ",", ";", "!", "?"})

"""Rank cart add-ons: sum lift over cart items, drop anything already in (or substituting for) the cart."""
from collections import defaultdict

import pandas as pd

RULES_TABLE = "recs.cart_addons"
MAX_ADDONS = 4


class CartAddons:
    def __init__(self, rules: pd.DataFrame, sku_cat: dict):
        self.sku_cat = sku_cat
        self.by_item = defaultdict(list)
        for r in rules.itertuples():
            self.by_item[r.antecedent].append((r.consequent, r.lift, r.level))

    @classmethod
    def from_db(cls, engine, sku_cat):
        return cls(pd.read_sql(f"SELECT antecedent, consequent, lift, level FROM {RULES_TABLE}", engine), sku_cat)

    def recommend(self, cart, k=MAX_ADDONS):
        cart = list(dict.fromkeys(cart))
        cart_cats = {self.sku_cat.get(s) for s in cart}
        score, why = defaultdict(float), defaultdict(list)
        for item in cart:
            for cons, lift, level in self.by_item.get(item, ()):
                if cons in cart or self.sku_cat.get(cons) in cart_cats:
                    continue
                score[cons] += lift if level == "item" else 0.5 * lift
                why[cons].append(item)
        ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
        out, seen_cats = [], set()
        for sku, s in ranked:
            c = self.sku_cat.get(sku)
            if c in seen_cats:   # one add-on per category
                continue
            seen_cats.add(c)
            out.append({"sku": sku, "score": round(s, 3), "because": why[sku]})
            if len(out) == k:
                break
        return out

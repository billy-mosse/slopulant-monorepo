"""Mine 'complete your order' pair rules from orders.lines -> recs.cart_addons."""
import argparse
import logging
from collections import Counter
from itertools import combinations

import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("cart_rules")

MIN_SUPPORT = 15          # baskets containing both items
MIN_CONFIDENCE = 0.02
MIN_LIFT = 1.5
SPARSE_ITEM_ORDERS = 40   # below this, back off to category-level rules
MAX_BASKET = 30
TOP_PER_ITEM = 20

BASKETS_SQL = """
SELECT order_id, sku, category_id
FROM orders.lines
WHERE order_date >= CURRENT_DATE - INTERVAL '365 days'
  AND status NOT IN ('cancelled', 'returned')
"""


def baskets(df, key):
    b = df.groupby("order_id")[key].apply(lambda s: sorted(set(s)))
    return [x for x in b if 1 < len(x) <= MAX_BASKET]


def mine(bs):
    n = len(bs)
    single = Counter(i for b in bs for i in b)
    pair = Counter(p for b in bs for p in combinations(b, 2))
    rows = []
    for (a, b), c in pair.items():
        if c < MIN_SUPPORT:
            continue
        for x, y in ((a, b), (b, a)):
            conf = c / single[x]
            lift = conf / (single[y] / n)
            if conf >= MIN_CONFIDENCE and lift >= MIN_LIFT:
                rows.append((x, y, c / n, conf, lift))
    log.info("%d baskets, %d pairs >= support, %d rules", n, sum(v >= MIN_SUPPORT for v in pair.values()), len(rows))
    return pd.DataFrame(rows, columns=["antecedent", "consequent", "support", "confidence", "lift"])


def category_backoff(df, item_rules, cat_rules, sku_cat):
    """Sparse antecedents inherit their category's rules, filled with that target category's best sellers."""
    counts = df.groupby("sku").order_id.nunique()
    sparse = counts[counts < SPARSE_ITEM_ORDERS].index
    best = (df.groupby(["category_id", "sku"]).order_id.nunique().reset_index()
              .sort_values("order_id", ascending=False).groupby("category_id").head(3))
    best_by_cat = best.groupby("category_id").sku.apply(list).to_dict()
    cat_by = cat_rules.groupby("antecedent")
    rows = []
    for sku in sparse:
        cat = sku_cat.get(sku)
        if cat not in cat_by.groups:
            continue
        for r in cat_by.get_group(cat).itertuples():
            for target in best_by_cat.get(r.consequent, []):
                rows.append((sku, target, r.support, r.confidence, r.lift * 0.8))
    log.info("backoff added %d rules for %d sparse items", len(rows), len(sparse))
    back = pd.DataFrame(rows, columns=item_rules.columns).assign(level="category")
    return pd.concat([item_rules.assign(level="item"), back], ignore_index=True)


def drop_substitutes(rules, sku_cat):
    same = rules.antecedent.map(sku_cat) == rules.consequent.map(sku_cat)
    log.info("dropping %d same-category substitute rules", same.sum())
    return rules[~same]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    eng = create_engine(a.dsn)

    df = pd.read_sql(BASKETS_SQL, eng)
    sku_cat = df.drop_duplicates("sku").set_index("sku").category_id.to_dict()
    item_rules = mine(baskets(df, "sku"))
    cat_rules = mine(baskets(df, "category_id"))
    cat_rules = cat_rules[cat_rules.antecedent != cat_rules.consequent]

    rules = drop_substitutes(category_backoff(df, item_rules, cat_rules, sku_cat), sku_cat)
    rules = (rules.sort_values("lift", ascending=False)
                  .drop_duplicates(["antecedent", "consequent"])
                  .groupby("antecedent").head(TOP_PER_ITEM))
    rules["built_at"] = pd.Timestamp.utcnow()
    rules.to_sql("cart_addons", eng, schema="recs", if_exists="replace", index=False)
    log.info("wrote %d rules to recs.cart_addons", len(rules))


if __name__ == "__main__":
    main()

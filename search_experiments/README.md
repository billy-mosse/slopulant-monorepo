# search experiments

Experiments that graduated to scheduled jobs. Each module is a standalone CLI
(`python <module>.py --help`); all of them take `--config <yaml>` and
`--dry-run`, and env vars `SEARCH_EXP_<FIELD>` override config fields.

| Module | What | Reads | Writes |
|---|---|---|---|
| `synonyms.py` | Synonym mining from in-session reformulations (no-click A -> clicked B), PMI/NPMI scoring, edit-distance typo filter, manual blocklist | `search.query_logs` | `search.synonyms` |
| `zero_results.py` | Zero-result fallbacks: IDF-based query relaxation, then category trending, then global trending | `search.query_logs`, `catalog.products`, `recs.trending` | `search.zero_result_fallbacks` |
| `query_categorizer.py` | Query -> category with char n-gram naive Bayes trained on click labels | `search.query_logs`, `catalog.products` | `search.query_categories` |
| `diversity_rerank.py` | MMR re-ranking over brand / colour / price band | `search.query_logs`, `catalog.products` | `search.diverse_results` |
| `shared_utils.py` | Normalization, tokenization, IDF tables, config loader, logging, warehouse I/O, metrics | - | - |

Schedules: synonyms weekly, query categories nightly, zero-result fallbacks and
diverse results daily. Synonyms are reviewed by merchandising before the
search config picks them up; add rejections to `BLOCKLIST` or pass
`--block "a|b"`.

Useful one-offs:

    python diversity_rerank.py --sweep                 # lambda trade-off table
    python diversity_rerank.py --show "throw pillows"  # before/after for one query
    python query_categorizer.py --predict "cal king sheets" "bathsheet"
    python zero_results.py --explain "linen duvet cover sage 108x98"

Owner: Yuki Tanaka (#search-relevance)

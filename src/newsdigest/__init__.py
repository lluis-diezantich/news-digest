"""Personal multilingual news aggregation pipeline.

Two pipelines, separated by cadence -- which is the central architectural idea:

    DAILY   sources -> fetch -> normalize -> detect language -> dedupe -> SQLite
    WEEKLY  SQLite  -> embed -> cluster -> LLM -> rank -> digest -> static site

Collection is cheap and calls no model, so it can run every day. The expensive
semantic work runs once a week over a whole finished week.

Deterministic work (fetching, normalizing, deduping, ranking, rendering) is
Python. The LLM does language understanding only: summarizing, classifying,
extracting entities, rating importance and relevance, and adjudicating a
borderline cluster. Embeddings do the cross-language matching that token overlap
provably cannot.
"""

__version__ = "0.2.0"

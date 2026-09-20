"""Personal weekly news digest, built from newsletters in a dedicated inbox.

    EMAIL -> INGEST -> PARSE -> NORMALIZE -> FILTER -> DEDUPLICATE
          -> CLUSTER -> RANK -> SUMMARIZE -> DIGEST -> MARKDOWN

One cadence: newsletters arrive weekly, so the whole thing runs once a week. The
goal is not to reproduce ten newsletters but to synthesize them -- coverage of the
same event grouped across English and Spanish, ranked by how many independent
outlets carried it and where their editors placed it, and written up as five to
ten minutes of reading.

Deterministic work (parsing, normalizing, deduping, ranking, rendering) is
Python. The LLM does language understanding only: triaging what is news,
summarizing, extracting entities, and adjudicating a borderline cluster.
Embeddings do the cross-language matching that token overlap provably cannot --
measured on one headline in three languages, overlap scores 0.00 and 0.06 against
a 0.60 merge threshold.
"""

__version__ = "0.3.0"

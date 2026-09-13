"""Personal news aggregation pipeline.

Stages, each independently runnable from the CLI:

    fetch -> normalize -> dedupe -> store -> enrich (LLM) -> cluster -> build

Deterministic work (fetching, normalizing, deduping, scoring, rendering) lives
in Python. The LLM is used only for language understanding: summarizing,
classifying topics, extracting entities, estimating importance, and naming the
underlying event so articles can be clustered.
"""

__version__ = "0.1.0"

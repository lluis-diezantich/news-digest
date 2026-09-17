"""Build the clustering eval fixture from `data/news.db`.

Run once to regenerate `tests/fixtures/cluster_eval.json`; the fixture is what
the tests and `python -m newsdigest.eval` read, so the database is not needed
afterwards. That matters because `embedding_retention_days` prunes vectors and
articles scroll out of the window -- a fixture that pointed at the live database
would rot, and the labels are too expensive to re-derive.

The ground truth comes from two stories the owner dissected by hand, both of
which the pre-fix clustering welded out of several unrelated events:

  s9b661259ac146ae  "Sanchez targets end-of-year resolution for Ceuta crisis"
                    -- Ceuta/Morocco, the Podemos primaries and a Junts piece.
                    doc: a 0.726 edge pulled the Junts article in and a 0.705
                    edge pulled in the three Podemos articles.
  scf09754259aef38  "Electoral Board Divided on Supreme Court's Vote Suspension"
                    -- four unrelated court matters: the JEC's response to the
                    'ley de nietos' order, the Alacant antifascists trial, the
                    Ceuta government-delegate case and the Caso Lezo.

Labels are stored as *events*, not as pairs: two articles in the same event are
a positive, two in different events are a negative. That is far harder to get
subtly wrong than enumerating ~100 pairs by hand, and it makes the judgement
calls explicit -- an event pair listed in UNSURE is excluded from scoring
entirely rather than guessed at.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "news.db"
OUT = ROOT / "tests" / "fixtures" / "cluster_eval.json"

#: Only vectors from this provider are used. `cache_key` includes the provider,
#: so a database that has switched embedder holds two sets; mixing them would
#: compare vectors from different spaces and quietly produce nonsense.
CACHE_KEY = "local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2:0"

#: One entry per distinct real-world event. Within an event -> same; across two
#: events -> different, unless the pair of events appears in UNSURE.
EVENTS: dict[str, list[str]] = {
    # --- from s9b661259ac146ae ---------------------------------------------
    # Sanchez putting a year-end date on the single command in Ceuta: the same
    # quote, two outlets.
    "ceuta_timeline": ["6a5dce44bd", "20e6056333"],
    # The "chantaje" row: partners' suspicions, and Sanchez rejecting them.
    "morocco_blackmail": ["d51f1890ee", "213e79ae68"],
    # Clavijo demanding Sanchez lead the response -- a reaction piece.
    "clavijo_reaction": ["ebe4d22299"],
    # The trio the LLM wrongly split (doc: "a genuinely-single story").
    "podemos_primaries": ["2ad4c45ff1", "25583c6dc9", "9389c2d7c4"],
    # The documented 0.726 false positive that welded two clusters.
    "junts_puigdemont": ["f61a4935ab"],

    # --- from scf09754259aef38 --------------------------------------------
    # The JEC session complying with the Supreme Court order, nine outlets
    # including one Catalan headline that is a near-translation of eldiario's.
    "jec_session": [
        "71ce56a31e", "057bf842e4", "beadff1688", "1854147a28", "047a356925",
        "db566b466b", "db1017caf7", "df59aac3a1", "3ad98903e9",
    ],
    # Oscar Lopez's "salvajada" / "jueces salvapatrias" remarks.
    "lopez_reaction": ["ba5edb102c", "d2c18db028"],
    # The judge who froze the vote, reported on its own.
    "judge_froze_vote": ["471225301b"],
    # The government's response to the Supreme Court president's speech.
    "govt_pushback": ["5b4d16467d"],
    # An opinion column on the conflict.
    "opinion_manel_perez": ["8ce7c4585e"],
    # Three court matters unrelated to the 'ley de nietos' affair.
    "alacant_antifascists": ["94c5e3b70a"],
    "ceuta_delegate_case": ["bc25f27437", "d971895459"],
    "caso_lezo": ["c10375c011"],
}

#: Event pairs the owner has NOT adjudicated. Excluded from scoring in both
#: directions -- neither a positive nor a negative. Everything here is a
#: "same broader affair, but is it the same event?" call, which is a judgement
#: about editorial intent rather than something the data settles.
UNSURE: list[tuple[str, str]] = [
    # Possibly one press appearance, possibly two stories.
    ("ceuta_timeline", "morocco_blackmail"),
    ("clavijo_reaction", "ceuta_timeline"),
    ("clavijo_reaction", "morocco_blackmail"),
    # The 'ley de nietos' affair's internal structure: the suspension itself,
    # the JEC's response, the political reaction and a column on all of it.
    ("judge_froze_vote", "jec_session"),
    ("judge_froze_vote", "lopez_reaction"),
    ("judge_froze_vote", "govt_pushback"),
    ("judge_froze_vote", "opinion_manel_perez"),
    ("govt_pushback", "jec_session"),
    ("govt_pushback", "lopez_reaction"),
    ("govt_pushback", "opinion_manel_perez"),
    ("opinion_manel_perez", "jec_session"),
    ("opinion_manel_perez", "lopez_reaction"),
]

#: Where each event's articles came from, for provenance.
ORIGIN = {
    "s9b661259ac146ae": [
        "ceuta_timeline", "morocco_blackmail", "clavijo_reaction",
        "podemos_primaries", "junts_puigdemont",
    ],
    "scf09754259aef38": [
        "jec_session", "lopez_reaction", "judge_froze_vote", "govt_pushback",
        "opinion_manel_perez", "alacant_antifascists", "ceuta_delegate_case",
        "caso_lezo",
    ],
}

# Ids above are the first 10 characters of the real article id, which is what
# the dissection worked from; expand them against the database so a prefix
# collision cannot silently pick the wrong article.
def _resolve(conn: sqlite3.Connection, prefix: str) -> str:
    rows = conn.execute(
        "SELECT id FROM articles WHERE id LIKE ?", (prefix + "%",)
    ).fetchall()
    if len(rows) != 1:
        raise SystemExit(f"{prefix!r} matched {len(rows)} articles, expected 1")
    return rows[0][0]


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    events = {k: [_resolve(conn, p) for p in v] for k, v in EVENTS.items()}

    seen: dict[str, str] = {}
    for event, ids in events.items():
        for aid in ids:
            if aid in seen:
                raise SystemExit(f"{aid} is in both {seen[aid]} and {event}")
            seen[aid] = event

    known = set(events)
    for left, right in UNSURE:
        missing = {left, right} - known
        if missing:
            raise SystemExit(f"UNSURE names unknown event(s): {sorted(missing)}")

    articles = []
    for aid in seen:
        row = conn.execute(
            """
            SELECT a.id, a.title, a.description, a.summary, a.language,
                   a.publisher, a.source, a.url, a.published_at, a.collected_at,
                   a.source_weight, a.content_type, e.vector
              FROM articles a
              JOIN embedding_cache e ON e.content_hash = a.content_hash
             WHERE a.id = ? AND e.cache_key = ?
            """,
            (aid, CACHE_KEY),
        ).fetchone()
        if row is None:
            raise SystemExit(f"no {CACHE_KEY} vector cached for {aid}")

        vector = np.frombuffer(row["vector"], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if abs(norm - 1.0) > 1e-3:
            raise SystemExit(f"{aid} vector is not unit length ({norm:.4f})")

        articles.append({
            "id": row["id"],
            "title": row["title"],
            # Truncated: the fixture only needs enough text to be readable and
            # to exercise the text-similarity fallback. Every article here has
            # a vector, so that fallback does not actually fire.
            "description": (row["description"] or "")[:600],
            "summary": (row["summary"] or None),
            "language": row["language"],
            "publisher": row["publisher"],
            "source": row["source"],
            "url": row["url"],
            "published_at": row["published_at"],
            "collected_at": row["collected_at"],
            "source_weight": row["source_weight"],
            "content_type": row["content_type"],
            "vector": base64.b64encode(row["vector"]).decode("ascii"),
        })

    articles.sort(key=lambda a: a["id"])
    payload = {
        "notes": __doc__.strip(),
        "embedding_cache_key": CACHE_KEY,
        "dimensions": len(np.frombuffer(
            base64.b64decode(articles[0]["vector"]), dtype=np.float32)),
        "origin": ORIGIN,
        "events": events,
        "unsure_event_pairs": [list(p) for p in UNSURE],
        "articles": articles,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"{OUT.relative_to(ROOT)}: {len(articles)} articles, "
          f"{len(events)} events, {len(UNSURE)} unsure event pairs")


if __name__ == "__main__":
    main()

"""What the extractor would get out of the stored mail RIGHT NOW.

For the hour after changing an extraction rule, and no longer. `parse` never
revisits a message it has already parsed, so the database cannot tell you
whether a change worked -- this re-runs the current extractor over the stored
HTML and prints what it finds. The day it was written the two disagreed by six
headlines, including two sources that had stored nothing at all.

`news-digest headlines` is the everyday command and reads the articles table.
Prefer it. This stays a script because it cannot be part of the pipeline:

  * it only works inside `email_body_retention_days` (30), after which `prune`
    empties `html_body` and there is nothing left to re-extract from;
  * it only works where a `fetch` has run, so never on a fresh clone or a CI
    runner, the database no longer being committed.

Nothing here writes. It reads the mailbox cache and prints.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from newsdigest.config import DEFAULT_DB, DEFAULT_SOURCES, load_sources  # noqa: E402
from newsdigest.extract import newsletter as extract  # noqa: E402
from newsdigest.models import Email  # noqa: E402


def _emails(conn: sqlite3.Connection, *, days: int | None, source: str | None):
    """Stored messages, newest first, optionally narrowed."""
    clauses, params = [], []
    if days:
        clauses.append("received_at >= datetime('now', ?)")
        params.append(f"-{int(days)} days")
    if source:
        clauses.append("source LIKE ?")
        params.append(f"%{source}%")
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM emails{where} ORDER BY source, received_at DESC", params
    ).fetchall()
    return [
        Email(
            message_id=r["message_id"], source=r["source"], subject=r["subject"],
            sender=r["sender"], sender_name=r["sender_name"] or "",
            newsletter=r["newsletter"] or "", html_body=r["html_body"] or "",
            text_body=r["text_body"] or "",
        )
        for r in rows
    ]


def _from_extraction(conn, *, days, source, sources_path):
    """Re-run the extractor over the stored HTML."""
    settings = {s.name: s for s in load_sources(sources_path)}
    out = defaultdict(list)
    for email in _emails(conn, days=days, source=source):
        conf = settings.get(email.source)
        items = extract.extract(
            email, excerpt_chars=conf.excerpt_chars if conf else 1200
        )
        out[email.source].extend(
            {"title": i.title, "url": i.url, "subject": email.subject}
            for i in items
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--days", type=int, default=None,
                        help="only messages from the last N days (default: all)")
    parser.add_argument("--source", default=None,
                        help="substring match on the source name")
    parser.add_argument("--urls", action="store_true", help="print the URL too")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    if not args.db.exists():
        parser.error(f"no database at {args.db}")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        grouped = _from_extraction(
            conn, days=args.days, source=args.source, sources_path=args.sources
        )
    finally:
        conn.close()

    if args.as_json:
        print(json.dumps(grouped, ensure_ascii=False, indent=2))
        return 0

    total = sum(len(v) for v in grouped.values())
    if not total:
        print("no headlines; is the window too narrow, or the mailbox unfetched?")
        return 1

    for source in sorted(grouped):
        items = grouped[source]
        print(f"--- {source} ({len(items)})")
        for item in items:
            # No `x` marks here: classification runs after parse, so a fresh
            # extraction has no newsworthiness verdict to report. `news-digest
            # headlines` shows those.
            print(f"   {item['title']}")
            if args.urls:
                print(f"      {item['url']}")
        print()

    plural = "" if len(grouped) == 1 else "s"
    print(f"{total} headlines across {len(grouped)} source{plural} (re-extracted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

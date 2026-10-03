"""Stories -> narratives.

The layer the digest was missing, and the reason its output read as a list where
a human would have written a story.

Clustering's unit is an EVENT: several outlets covering one thing, merged. That
is the right unit for deduplication and it is the wrong unit for a weekly digest,
because a week is not a set of unrelated events. In the real 2026-W40 run, an
eviction, the protest it set off, the emergency decrees it produced, an
investigation into the building's owner, and an election won on rent in Berlin
came out as five separate entries scoring independently -- while a person
reading the same five newsletters would have written one piece and called it the
story of the week.

No clustering threshold fixes that. Those five are not the same event, and
loosening the threshold until they merged would fuse genuinely unrelated
coverage first: the band between 0.65 and 0.80 is already where false merges
live, which is what `clustering`'s adjudication pass exists to police.

So themes are a pass ABOVE clusters, and they are asked for explicitly rather
than inferred from distance. Embedding centroids would connect "rents in Madrid"
to "rents in Berlin" only by lexical luck; connecting "an 87-year-old's eviction"
to "Die Linke wins Berlin" needs to know what the housing argument is about.

Nothing here is required. A provider that cannot group returns no themes and the
digest publishes its stories flat, exactly as before.
"""

from __future__ import annotations

import logging

from .llm.base import (
    Context,
    LLMProvider,
    ThemeGroupInput,
    ThemeInput,
)
from .models import Article, Story, Theme

log = logging.getLogger(__name__)

#: Fewest stories a theme may hold. Two is the point of the exercise; one is a
#: story with a label on it.
MIN_THEME_STORIES = 2


def _excerpt(articles: list[Article], limit: int) -> str:
    """The best text available for one story, for the theme writer to work from.

    Prefers a summary the enrichment pass already produced, because it is
    shorter and cleaner than raw newsletter copy -- and falls back to the
    article's own excerpt, which is what exists when enrichment ran out of
    quota. That fallback is the normal case on a free tier, not an edge one.
    """
    for article in articles:
        for candidate in (article.summary, article.description, article.content):
            text = (candidate or "").strip()
            if text:
                return text[:limit]
    return ""


def group(
    ranked: list[tuple[Story, list[Article]]],
    provider: LLMProvider,
    context: Context,
    *,
    max_themes: int = 6,
    excerpt_chars: int = 600,
    enabled: bool = True,
    editorial: list[str] | None = None,
) -> list[Theme]:
    """Group `ranked` stories into themes and write each one.

    `ranked` is (story, articles) in score order. The return value holds only the
    stories that landed in a theme; everything else stays where it was. Order
    follows the grouping pass, which is asked to put the most consequential
    first -- so the first theme is the digest's lead.

    `editorial` carries the newsletters' own descriptions of their week, where
    they gave one. A hint, never an instruction: only one of seven sources
    publishes one, so it cannot be relied on, and a week where it is absent must
    group exactly as well.

    Costs at most 1 + len(themes) requests, and REPLACES nothing: the stories
    keep their own briefs, so a theme that fails to write still has publishable
    members underneath it.
    """
    if not enabled or len(ranked) < MIN_THEME_STORIES:
        return []

    provider.max_themes = max_themes
    by_id = {story.id: (story, articles) for story, articles in ranked}
    items = [
        ThemeGroupInput(
            story_id=story.id,
            headline=story.headline,
            topics=list(story.topics or []),
            publishers=sorted({a.publisher for a in articles if a.publisher}),
        )
        for story, articles in ranked
    ]

    try:
        groups = provider.group_themes(items, context, editorial=editorial or [])
    except Exception as exc:  # a grouping failure must not lose the digest
        log.warning("theme grouping failed (%s); publishing stories flat", exc)
        return []
    if not groups:
        log.info("no themes returned; publishing %d stories flat", len(ranked))
        return []

    themes: list[Theme] = []
    #: Enforced HERE, not in a provider. A story in two themes is published
    #: twice, and the check belongs on the path every provider flows through --
    #: it was in the Gemini client alone until a test with a deliberately greedy
    #: stub published four stories as six.
    claimed: set[str] = set()
    for group_result in groups:
        members = [
            by_id[sid] for sid in group_result.story_ids
            if sid in by_id and sid not in claimed
        ]
        if len(members) < MIN_THEME_STORIES:
            continue
        claimed.update(story.id for story, _ in members)
        coverage = [
            (
                story.headline,
                ", ".join(sorted({a.publisher for a in articles if a.publisher})),
                _excerpt(articles, excerpt_chars),
            )
            for story, articles in members
        ]
        theme = Theme(
            label=group_result.label,
            story_ids=[story.id for story, _ in members],
        )
        try:
            brief = provider.write_theme(
                ThemeInput(label=group_result.label, coverage=coverage), context
            )
        except Exception as exc:
            # The group is still worth keeping: rendering falls back to the
            # label plus the stories' own briefs, which is no worse than the
            # flat output and keeps the grouping that was already paid for.
            log.warning("theme %r failed to write (%s)", group_result.label, exc)
            brief = None
        if brief is not None:
            theme.headline = brief.headline
            theme.narrative = brief.narrative
            theme.why_it_matters = brief.why_it_matters
            theme.open_questions = list(brief.open_questions)
        themes.append(theme)

    grouped = sum(len(t.story_ids) for t in themes)
    log.info(
        "%d themes over %d of %d stories (%d published on their own)",
        len(themes), grouped, len(ranked), len(ranked) - grouped,
    )
    return themes

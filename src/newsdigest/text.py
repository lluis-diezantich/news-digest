"""Small text helpers shared by dedupe, clustering and the offline enricher."""

from __future__ import annotations

import html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

# Deliberately short: we want "us china trade" and "china us trade" to collide,
# but we do not want to strip words that carry the story ("no", "not").
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    from by with without about into over after before as is are was were be
    been being it its it's he she they them his her their we you i our your
    has have had do does did will would can could should may might must
    say says said report reports reported new news more most first last
    amid says' via how why what when who
    """.split()
)


def strip_html(value: str | None) -> str:
    """Unescape entities, drop tags, collapse whitespace."""
    if not value:
        return ""
    text = _TAG_RE.sub(" ", value)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def truncate(value: str, limit: int) -> str:
    """Cut to `limit` characters on a word boundary, appending an ellipsis."""
    if limit <= 0 or len(value) <= limit:
        return value
    cut = value[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:-") + "…"


def normalize(value: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", value or "")
    ascii_ish = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _WS_RE.sub(" ", ascii_ish.lower()).strip()


_SUFFIXES = ("ing", "ied", "ed", "es", "s")


def stem(word: str) -> str:
    """Crude suffix stripper, only ever used for comparison.

    Correctness is not the goal -- agreement is. "raises", "raised" and "raise"
    must all land on the same string so two outlets wording one event
    differently still look alike. Longest suffix first, then a trailing "e",
    which is what makes "raises" -> "rais" meet "raised" -> "rais".
    """
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(value: str, *, keep_stopwords: bool = False, stemmed: bool = True) -> list[str]:
    """Content words, in order, suitable for set/shingle comparison."""
    words = _WORD_RE.findall(normalize(value))
    if keep_stopwords:
        return words
    kept = [w for w in words if w not in STOPWORDS and (len(w) > 2 or w.isdigit())]
    return [stem(w) for w in kept] if stemmed else kept


def shingles(tokens: list[str], size: int = 2) -> set[tuple[str, ...]]:
    """Overlapping n-grams. Falls back to unigrams for very short inputs."""
    if len(tokens) < size:
        return {(t,) for t in tokens}
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(left: str, right: str) -> float:
    """Blend of unigram and bigram Jaccard over two pieces of text.

    Unigrams catch reworded headlines about the same event; bigrams keep
    unrelated stories that merely share vocabulary apart.
    """
    lt, rt = tokenize(left), tokenize(right)
    if not lt or not rt:
        return 0.0
    uni = jaccard(set(lt), set(rt))
    bi = jaccard(shingles(lt, 2), shingles(rt, 2))
    return 0.6 * uni + 0.4 * bi


def title_key(title: str) -> str:
    """Order-insensitive fingerprint used to catch republished headlines."""
    return " ".join(sorted(set(tokenize(title))))


def sentences(value: str) -> list[str]:
    text = strip_html(value)
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


# Single capitalized words that are almost always a capitalized sentence start
# rather than a name. English capitalizes sentence openings, so "Six dead, 130
# missing..." yields "Six" and "Ships and helicopters are searching" yields
# "Ships" unless they are filtered.
NON_ENTITY_WORDS = frozenset(
    """
    one two three four five six seven eight nine ten eleven twelve dozens
    hundreds thousands millions billions several many some most both few
    ships planes police officials sources people residents families workers
    scientists researchers experts analysts investors customers users
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october
    november december today tomorrow yesterday here there this that these
    those what when where which while after before during despite although
    """.split()
)


def capitalized_phrases(value: str, limit: int = 8) -> list[str]:
    """Cheap entity guess for the offline enricher: runs of Capitalized words.

    A heuristic, not a named-entity recogniser -- it feeds the offline provider
    and clustering's entity-overlap check, never anything a reader relies on.
    """
    text = strip_html(value)
    found: list[str] = []
    # Lowercase connectors that genuinely sit inside one name ("Bank of England",
    # "Ursula von der Leyen"). "and" is excluded on purpose -- it joins two
    # separate entities, and gluing them produced "Anthropic and Google".
    for match in re.finditer(
        r"\b([A-Z][\w'-]+(?:(?:\s+(?:of|de|von|van|der|la|el))*\s+[A-Z][\w'-]+)*)",
        text,
    ):
        phrase = _WS_RE.sub(" ", match.group(1)).strip()
        # A leading article is never part of the name we want to match on.
        if phrase.startswith(("The ", "A ", "An ")):
            phrase = phrase.split(" ", 1)[1]
        if len(phrase) < 3:
            continue
        lowered = normalize(phrase)
        if lowered in STOPWORDS:
            continue
        if " " not in phrase and lowered in NON_ENTITY_WORDS:
            continue
        if phrase not in found:
            found.append(phrase)
        if len(found) >= limit:
            break
    return found

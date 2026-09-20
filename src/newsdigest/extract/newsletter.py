"""Newsletter body -> individual news items.

This is the stage with no equivalent in a feed reader, and the only one whose
input has no schema at all. A feed hands over titled entries; a newsletter hands
over a table layout from 2006 and expects a human to read it.

The approach is structural rather than per-publisher. Every newsletter item, in
every one of the ten configured sources, is the same shape:

    a link to an article  +  a headline  +  (usually) a sentence or two

So the extractor finds the links that could be articles, and for each one takes
the SMALLEST enclosing block that contains only that link. That block is the
item: its heading is the headline and the rest of its text is the blurb. Nothing
depends on a class name, a table depth or a publisher's template, all of which
change without notice.

Per-source overrides are deliberately absent for now. Section 30 of the
specification asks for the simple version first, and the structural rule is what
the fixtures in tests/fixtures/emails are for -- add a source-specific strategy
when a real newsletter defeats this one, not in advance of it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from ..models import Article, Email, parse_date
from ..text import normalize, strip_boilerplate, truncate
from ..urls import canonical_url
from . import links
from .boilerplate import (
    is_boilerplate_link,
    is_boilerplate_text,
    is_housekeeping,
    is_sponsored,
    strip_chrome,
)

log = logging.getLogger(__name__)

#: A headline is at least this many characters and this many words. Tuned to
#: reject the things that are shaped like headlines but are not: section labels
#: ("WORLD", "Middle East"), bylines, and the date line.
MIN_TITLE_CHARS = 22
MIN_TITLE_WORDS = 3
#: Past this, we have caught a whole paragraph rather than a headline.
MAX_TITLE_CHARS = 300

#: Alt text that names the image rather than the story. A masthead's alt text is
#: the right length and word count to pass as a headline -- "Logo de Hoy en
#: Público" was extracted as item 1 of a real issue -- and there is no other way
#: to tell it apart, because structurally it IS a linked image with text.
_IMAGE_LABEL_RE = re.compile(
    r"^(?:logo|logotipo|imagen|image|foto|photo|banner|cabecera|header|icon)\b"
    r"|\blogo (?:de|of)\b|_cab$|\bcampana\b"
)

#: Blocks bigger than this are the whole newsletter, not one item -- which
#: happens when a publisher wraps everything in a single table cell.
MAX_BLOCK_CHARS = 6_000

#: Tags that can plausibly enclose exactly one item.
_BLOCK_TAGS = ("p", "li", "td", "div", "tr", "table", "section", "article", "blockquote")
_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_EMPHASIS_TAGS = ("strong", "b", "span")

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


@dataclass
class ExtractedItem:
    """One candidate news item, before it becomes an `Article`."""

    title: str
    url: str
    blurb: str = ""
    #: Document order within the newsletter. The editor's own ranking.
    position: int = -1
    published_at: object | None = None
    #: Why this item was dropped, when it was. Carried for `--debug`.
    dropped: str = ""
    extras: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# titles
# --------------------------------------------------------------------------- #

def plausible_title(text: str) -> bool:
    """Could this string be a headline?

    Deliberately a shape test, not a content one. A newsletter's section labels
    and its headlines are typographically identical and differ only in length,
    so length and word count are what separate them.
    """
    cleaned = (text or "").strip()
    if not (MIN_TITLE_CHARS <= len(cleaned) <= MAX_TITLE_CHARS):
        return False
    if len(cleaned.split()) < MIN_TITLE_WORDS:
        return False
    # An all-caps run this long is a standing banner ("THE WEEK IN NUMBERS").
    letters = [c for c in cleaned if c.isalpha()]
    return not (letters and all(c.isupper() for c in letters))


def _title_from(block, anchor) -> str:
    """The best headline for one item, trying four sources in order.

    The anchor's own text comes first and is usually right. It is empty when the
    link is wrapped around an image, which every newsletter does for its lead
    story -- so the fallbacks matter for exactly the item that matters most.
    """
    candidates: list[str] = [anchor.get_text(" ", strip=True)]

    for tag in _HEADING_TAGS:
        for heading in block.find_all(tag):
            candidates.append(heading.get_text(" ", strip=True))
    for tag in _EMPHASIS_TAGS:
        for emphasis in block.find_all(tag):
            candidates.append(emphasis.get_text(" ", strip=True))

    # An image's alt text is a real headline surprisingly often, because the
    # template feeds it the same field -- but it is just as often the image's own
    # name, so anything that describes the picture is dropped rather than used.
    for image in block.find_all("img", alt=True):
        alt = str(image.get("alt") or "").strip()
        if alt and not _IMAGE_LABEL_RE.search(normalize(alt)):
            candidates.append(alt)

    for candidate in candidates:
        if plausible_title(candidate):
            return re.sub(r"\s+", " ", candidate).strip()

    # Last resort: the longest line of the block's own text.
    lines = [line.strip() for line in block.get_text("\n", strip=True).split("\n")]
    longest = max((line for line in lines if plausible_title(line)), key=len, default="")
    return re.sub(r"\s+", " ", longest).strip()


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #

def _candidate_anchors(soup, sender_domain: str) -> list:
    """Every link that could be an article, in document order."""
    found = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        text = anchor.get_text(" ", strip=True)
        if not is_boilerplate_link(href, text, sender_domain=sender_domain):
            found.append(anchor)
    return found


def _item_block(anchor, targets: dict[int, str]):
    """The LARGEST ancestor of `anchor` that still holds only this one article.

    Largest rather than smallest, because the blurb is a sibling of the headline
    rather than a child of it -- stopping at the smallest block gets a headline
    with no summary. Growing until ANOTHER article appears is what keeps two
    adjacent items from merging into one.

    "Another article" means another URL, not another link. A newsletter links its
    lead story three times from one block -- the image, the headline, and a "read
    more" -- and counting anchors made that block look like three items, so the
    block never grew past the anchor and the lead story was published with no
    summary. It is the item most worth getting right, and the failure was
    invisible in the counts: five items extracted, one of them hollow.
    """
    best = anchor
    own = targets.get(id(anchor), "")
    node = anchor.parent
    while node is not None and getattr(node, "name", None) not in (None, "[document]"):
        if node.name in _BLOCK_TAGS:
            others = {
                targets[id(other)]
                for other in node.find_all("a", href=True)
                if id(other) in targets and targets[id(other)] != own
            }
            if others:
                break
            if len(node.get_text(" ", strip=True)) > MAX_BLOCK_CHARS:
                break
            best = node
        node = node.parent
    return best


#: A byline left behind once the headline is removed: "Por Cristina Fallarás",
#: or just the bare "Por" when the name itself was a link.
_BYLINE_RE = re.compile(r"^(?:por|by|de)(?:\s+[^.]{2,40})?$", re.IGNORECASE)

# A byline sitting in FRONT of the summary was stripped here until it was
# measured against real issues, and the attempt is recorded rather than repeated.
# Templates do put the author first -- "Pilar Araque Conde Leer artículo completo"
# -- but a byline and the subject of a Spanish sentence are the SAME SHAPE: one to
# three capitalised words followed by another capital. Stripping it turned "El
# Movimiento Regularización Ya, junto a otras entidades..." into "Ya, junto a..."
# and "Emmett Doyle es carpintero..." into "Doyle es carpintero...". Both are
# invisible in the output -- the blurb still reads as a sentence -- which is worse
# than the byline it removed. A bare byline with nothing after it is handled by
# `_BYLINE_RE` and the length floor instead.

#: Call-to-action text that survives in a block after its own link was rejected.
#: The link is gone, the words are not, and they are not a summary.
#:
#: Accents spelled as character classes, because this runs on the RAW text rather
#: than `normalize`d text -- it has to, since the result is kept and shown. The
#: first version matched only "articulo" and so did nothing at all to the
#: "Leer artículo completo »" it was written for.
_CTA_RE = re.compile(
    r"(?:leer (?:el )?art[ií]culo(?: completo)?|read (?:more|the full [a-z]+)|"
    r"seguir leyendo|sigue leyendo|leer m[áa]s|ver (?:la )?noticia|"
    r"m[áa]s informaci[óo]n|continue reading)\s*[»\u2192>]*\s*",
    re.IGNORECASE,
)

#: A call-to-action cut off mid-phrase, which is what is left when the link text
#: sat in its own element: El Salto's blocks end "...aseguradora alemana. Leer".
#: Anchored to the END of the string on purpose -- a bare "leer" or "más" mid-blurb
#: is ordinary prose ("leer es importante"), and only its position makes it junk.
_CTA_TAIL_RE = re.compile(
    r"[\s.,;:]+(?:leer|read|ver|escuchar|listen|watch|m[áa]s|more)"
    r"\s*[»\u2192>]*\s*$",
    re.IGNORECASE,
)

#: Below this a blurb is a fragment rather than a summary, and an empty blurb is
#: more honest than "Por" or a stray byline. The digest falls back to the headline,
#: which is what a newsletter like Público gives us and all it gives us.
_MIN_BLURB_CHARS = 25


def _blurb(block, title: str, limit: int) -> str:
    """The block's text with the headline taken out of it.

    EVERY copy of the headline, not the first. A newsletter commonly prints it
    twice in one block -- as the link text and again as the heading -- so removing
    one occurrence left the other, and 14 of 27 blurbs in a real Público issue
    were just their own headline repeated back.
    """
    text = block.get_text(" ", strip=True)
    if title:
        text = re.sub(re.escape(title), " ", text)
    text = _clean_blurb(text)
    return truncate(strip_boilerplate(text), limit) if text else ""


def _clean_blurb(text: str) -> str:
    """Whatever is left after the headline, or "" if it is not a summary.

    Público's issues are the case that drove this: each item is a headline, an
    author and a "Leer artículo completo »", with no standfirst at all. Stripping
    only the headline left `"Pilar Araque Conde Leer artículo completo »"` as the
    summary of every story, which is worse than no summary -- the digest falls back
    to the headline, which is genuinely all that newsletter provides.
    """
    text = _CTA_RE.sub(" ", text)
    text = _CTA_TAIL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" |-–—·•\t")
    if _BYLINE_RE.match(text):
        return ""
    return text if len(text) >= _MIN_BLURB_CHARS else ""


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def extract_html(
    html: str, *, sender_domain: str = "", excerpt_chars: int = 1200
) -> list[ExtractedItem]:
    """Items from an HTML newsletter, in the order the newsletter put them."""
    soup = BeautifulSoup(html or "", "html.parser")
    strip_chrome(soup)

    anchors = _candidate_anchors(soup, sender_domain)
    if not anchors:
        return []
    # id() -> canonical target, so a block containing three links to one article
    # reads as one item. Keyed on id() because bs4 tags compare by markup, which
    # would make two identical links the same key.
    targets = {
        id(a): (canonical_url(str(a.get("href") or "")) or str(a.get("href") or ""))
        for a in anchors
    }

    items: list[ExtractedItem] = []
    for anchor in anchors:
        block = _item_block(anchor, targets)
        block_text = block.get_text(" ", strip=True)

        # A paid slot and a housekeeping block both have an item's shape, so they
        # are rejected here rather than being left to topic classification --
        # which cannot see the "Sponsored" label, only the copy beneath it.
        if is_sponsored(block_text):
            log.debug("dropped sponsored block: %s", block_text[:60])
            continue
        if is_housekeeping(block_text):
            continue

        title = _title_from(block, anchor)
        if not plausible_title(title):
            continue
        # Checked here as well as on the link text, because a link wrapped around
        # an image has no text to reject and the title then comes from the alt
        # attribute instead -- which is how a membership appeal became item 0.
        if is_boilerplate_text(title):
            log.debug("dropped boilerplate title: %s", title[:60])
            continue

        items.append(
            ExtractedItem(
                title=title,
                url=str(anchor.get("href") or "").strip(),
                blurb=_blurb(block, title, excerpt_chars),
            )
        )

    return _dedupe_items(items)


def extract_text(text: str, *, excerpt_chars: int = 1200) -> list[ExtractedItem]:
    """Items from a plain-text newsletter.

    The plain-text alternative is a fallback for a message with no HTML part at
    all, and it is much weaker: the convention is a headline line followed by its
    URL, with no markup to say which is which.
    """
    lines = [line.strip() for line in (text or "").splitlines()]
    items: list[ExtractedItem] = []

    for index, line in enumerate(lines):
        match = _URL_RE.search(line)
        if not match:
            continue
        url = match.group(0).rstrip(".,);]")
        # The headline is the nearest preceding line that looks like one; the
        # blurb is whatever follows the URL.
        title = ""
        for previous in reversed(lines[max(0, index - 4):index]):
            if plausible_title(previous):
                title = previous
                break
        if not title:
            stripped = line.replace(url, " ").strip(" |-–—·•")
            if plausible_title(stripped):
                title = stripped
        if not title:
            continue
        blurb = " ".join(
            l for l in lines[index + 1:index + 4] if l and not _URL_RE.search(l)
        )
        items.append(
            ExtractedItem(
                title=re.sub(r"\s+", " ", title),
                url=url,
                blurb=truncate(strip_boilerplate(blurb), excerpt_chars),
            )
        )

    return _dedupe_items(items)


def _dedupe_items(items: list[ExtractedItem]) -> list[ExtractedItem]:
    """Collapse the several links every newsletter gives one article.

    Two passes, because there are two ways one article appears twice.

    BY URL: a single item is commonly linked three times -- from its image, from
    its headline, and from a "read more" -- and the three carry different text.
    The image link has none at all, which is why they are MERGED rather than the
    first simply being kept: the best title and the longest blurb can come from
    different copies of the same link.

    BY TITLE: some trackers mint a fresh token per link, so the same article has a
    DIFFERENT url each time it appears and the URL pass cannot see it. A real
    Público issue carried four articles twice for exactly this reason. Within one
    newsletter an identical headline is the same article, so the title is a safe
    second key -- and it is only safe within one newsletter, which is why this runs
    per email rather than against the database.

    Position is the FIRST appearance, because that is where the editor put the
    item; a "read more" at the end of a block is not a new position.
    """
    def merge(into: ExtractedItem, other: ExtractedItem) -> None:
        if len(other.title) > len(into.title) and plausible_title(other.title):
            into.title = other.title
        if len(other.blurb) > len(into.blurb):
            into.blurb = other.blurb

    def clean(item: ExtractedItem) -> ExtractedItem:
        """Re-strip the headline from the blurb, after merging.

        `merge` takes the longest blurb among the copies of one link, and the
        copy with the longest blurb is often a bare "Leer artículo completo"
        whose block text is the headline -- so the winning blurb can be the very
        thing `_blurb` removed from the others. Stripping again here is what makes
        the merge order stop mattering.
        """
        if item.title and item.blurb:
            item.blurb = re.sub(
                r"\s+", " ", re.sub(re.escape(item.title), " ", item.blurb)
            ).strip(" |-–—·•\t")
            if _BYLINE_RE.match(item.blurb):
                item.blurb = ""
        return item

    by_url: dict[str, ExtractedItem] = {}
    for item in items:
        key = canonical_url(item.url) or item.url
        if key in by_url:
            merge(by_url[key], item)
        else:
            by_url[key] = item

    by_title: dict[str, ExtractedItem] = {}
    for item in by_url.values():
        key = normalize(item.title)
        if key in by_title:
            merge(by_title[key], item)
        else:
            by_title[key] = item

    ordered = [clean(item) for item in by_title.values()]
    for position, item in enumerate(ordered):
        item.position = position
    return ordered


def extract(email: Email, *, excerpt_chars: int = 1200) -> list[ExtractedItem]:
    """Items from whichever body this message has."""
    sender_domain = email.sender.rpartition("@")[2]
    if email.html_body:
        items = extract_html(
            email.html_body, sender_domain=sender_domain, excerpt_chars=excerpt_chars
        )
        if items:
            return items
        log.info("%s: HTML body yielded no items, trying plain text", email.source)
    return extract_text(email.text_body, excerpt_chars=excerpt_chars)


# --------------------------------------------------------------------------- #
# items -> articles
# --------------------------------------------------------------------------- #

def to_articles(
    email: Email,
    items: list[ExtractedItem],
    *,
    publisher: str = "",
    weight: float = 1.0,
    topics: list[str] | None = None,
    resolver=None,
) -> tuple[list[Article], int]:
    """Turn extracted items into articles, resolving tracking links.

    Returns (articles, unresolved), where `unresolved` counts items whose real
    URL we never established. Those are KEPT: a story we can only link through a
    tracker is still a story, and dropping it would silently shrink the digest
    whenever a publisher changed mailers. They are counted so the run can say so.

    `collected_at` is the email's own timestamp rather than the clock. It is the
    honest answer -- the item reached us when the newsletter did -- and it makes
    every downstream window and recency calculation work on newsletter data
    without any of them pretending to know a publication time nobody sent.
    """
    articles: list[Article] = []
    unresolved = 0

    for item in items:
        url, real = links.resolve(item.url, resolver)
        if not real:
            unresolved += 1
        articles.append(
            Article(
                title=item.title,
                source=email.source,
                url=url,
                published_at=parse_date(item.published_at),
                description=item.blurb,
                publisher=publisher or email.source,
                newsletter=email.newsletter,
                email_id=email.id,
                received_at=email.received_at,
                collected_at=email.received_at,
                item_position=item.position,
                item_count=len(items),
                source_topics=sorted(set(topics or [])),
                source_weight=weight,
            )
        )
    return articles, unresolved

"""What is not news in a newsletter.

Every newsletter carries the same furniture: a "view in browser" line, an
unsubscribe footer, social buttons, an app promo, a sponsor slot, and a block of
housekeeping about the newsletter itself. None of it is a story, and all of it
looks exactly like one to a link-based extractor -- a sponsor slot in particular
is a headline, a blurb and a link, which is the shape we are looking for.

Three defences, applied in order:

  1. `strip_chrome` deletes structural furniture from the tree outright.
  2. `is_boilerplate_link` rejects individual links (social, unsubscribe, the
     publisher's own front page).
  3. `is_sponsored` rejects a whole block whose text announces it is paid.

Written for English and Spanish, which are the configured languages. Adding a
language means adding its wording here, not changing the logic.
"""

from __future__ import annotations

import re

from ..text import normalize
from ..urls import domain

#: Link text that is furniture whatever it points at. Matched against
#: accent-stripped lowercase text, anchored loosely because publishers pad these
#: with arrows, bullets and pipes.
_SKIP_TEXT = (
    r"unsubscribe", r"darse de baja", r"baja de la newsletter", r"cancelar",
    r"gestionar (?:tus )?(?:preferencias|suscripciones)", r"manage (?:your )?preferences",
    r"view (?:this )?(?:email )?in (?:your )?browser", r"ver en el navegador",
    r"ver (?:esta )?newsletter en", r"view online", r"web version",
    # El Salto opens with this, and it was extracted as story number one.
    r"si no (?:ves|visualizas|puedes ver)", r"pulsa aqui", r"haz clic aqui",
    r"clica aqui", r"if you cannot (?:see|read)",
    r"privacy (?:policy|notice)", r"politica de privacidad", r"aviso legal",
    r"terms (?:of (?:use|service)|and conditions)", r"condiciones de uso",
    r"cookie", r"contact us", r"contacta", r"help cent(?:er|re)", r"ayuda",
    r"follow us", r"siguenos", r"siga?nos", r"share this", r"comparte",
    r"forward (?:this )?to a friend", r"reenvia", r"add to your address book",
    r"download the app", r"descarga(?:r)? la app", r"get the app",
    r"subscribe", r"suscribete", r"suscribirse", r"suscripcion", r"hazte socio",
    r"become a member",
    # Membership appeals, which are the shape of a story and are not one. All
    # three of these were extracted from real issues.
    r"apoya (?:el|nuestro) periodismo", r"unete a[l]? ", r"leernos es",
    r"apoyarnos", r"support (?:our|independent) journalism", r"donate",
    # "Fund independent journalism" sat in the Guardian's lead item block and
    # the `support` spelling above did not reach it.
    r"fund (?:our|independent) journalism", r"contribute",
    r"colabora con", r"haz(?:te)? una donacion",
    r"read more", r"leer mas", r"seguir leyendo", r"continue reading",
    r"leer (?:el )?articulo", r"ver (?:la )?noticia", r"full (?:story|article)",
    r"sigue leyendo", r"mas informacion",
    # A podcast promo's link text, which replaced the episode's real headline.
    r"escucha (?:el|la|nuestro|aqui)", r"listen (?:to|here)", r"ver el video",
    r"watch (?:the|now)",
    r"see all", r"ver todo", r"ver mas", r"more newsletters", r"mas boletines",
    r"advertisement", r"publicidad",
    # A bare date or section word is a navigation crumb, not a headline.
    r"^(?:home|inicio|portada|menu|top|index)$",
    # Added 2026-10-03, all five extracted as stories from one week's mail.
    # The existing `if you cannot (see|read)` missed Al Jazeera's wording, and
    # `ver en el navegador` missed elDiario's, which puts the object in between.
    r"if you are not able to (?:see|read|view)", r"para ver este e-?mail",
    r"sign up for", r"join our", r"all-?access", r"digital subscription",
    # A donation appeal that names neither donation nor support.
    r"one-off payment", r"monthly amount",
    r"if you have any questions", r"te ha gustado est",
    r"hasta la semana que viene", r"conoces nuestros boletines",
    # --- Catalan ---------------------------------------------------------
    # Added 2026-10-03 with the elDiario Catalunya source. The module was
    # written for en/es and `ca` joined `supported_languages` on the same day,
    # so none of this vocabulary existed: every Catalan appeal and pointer came
    # through as a story. These newsletters are a columnist's letter, so the
    # links are mostly the writer pointing at their own work -- "aqui podeu
    # llegir-ne la critica" -- which opens its paragraph and so survives the
    # structural test in `newsletter.py`. Only wording rejects it.
    r"fes-te soci", r"et fessis soci", r"fer-te soci",
    r"aqui (?:podeu|pots|teniu|en teniu|trobareu)",
    r"us (?:recomano|deixo|convido)", r"he escrit aquesta",
    # elDiario's standing Spanish appeal, which names neither support nor member.
    r"periodismo valiente",
)
_SKIP_TEXT_RE = re.compile("|".join(_SKIP_TEXT))

#: Hosts that are never a news article, however the link is dressed up.
SOCIAL_HOSTS = frozenset(
    """
    facebook.com fb.me twitter.com x.com t.co instagram.com linkedin.com
    youtube.com youtu.be tiktok.com whatsapp.com wa.me telegram.me t.me
    bsky.app threads.net mastodon.social reddit.com pinterest.com
    apps.apple.com play.google.com itunes.apple.com open.spotify.com
    podcasts.apple.com soundcloud.com flipboard.com
    """.split()
)
# The four ESP hosts that used to end this list -- list-manage.com,
# mailchimp.com, sendgrid.net, constantcontact.com -- were removed 2026-10-03.
# They are mail vendors, not social networks, and a vendor's host is what EVERY
# link in a newsletter wears: elDiario sends through `eldiario.us6.list-manage.com`,
# so 23 of its 24 links were discarded here and both elDiario sources had never
# yielded an item. The one survivor was the "ver este e-mail" line.
#
# Rejecting a tracker is the wrong response to it anyway. `links._MAILER_HOSTS`
# already names these, where the answer is to RESOLVE the link to the article
# behind it -- which is the whole point of that module.

#: Subdomain labels a publisher uses for everything that is NOT editorial.
#: `support.theguardian.com` was stored as the URL for "How Europe's housing
#: crisis is fuelling a new movement" -- the week's lead story, attributed to a
#: donate page. The host is the only reliable signal, because the link sits
#: inside the item's own block and wears the article's headline.
_NON_EDITORIAL_LABELS = frozenset(
    """
    support donate donations donaciones give giving contribute
    shop store checkout payment pay secure billing
    account myaccount accounts profile login signin
    subscribe suscripcion suscribete membership
    """.split()
)


def is_never_article(url: str) -> bool:
    """True when a URL's HOST cannot be an article, whatever it is labelled.

    Host-only, deliberately: this runs both before resolution (on whatever the
    newsletter wrote) and again after, where the link text is long gone and the
    only thing left to judge is where the link actually went.
    """
    host = domain(url)
    if not host:
        return False
    if host in SOCIAL_HOSTS or any(host.endswith("." + h) for h in SOCIAL_HOSTS):
        return True
    return host.split(".")[0] in _NON_EDITORIAL_LABELS


#: Wording that marks a block as paid placement. A sponsor slot is deliberately
#: built to look like editorial, so the label is the only reliable signal -- and
#: publishers are required to print one.
_SPONSOR = (
    r"\bsponsored\b", r"\bsponsor(?:ed by| content|:)", r"\bpaid (?:post|content|for by)\b",
    r"\bpresented by\b", r"\bin partnership with\b", r"\bbrought to you by\b",
    r"\badvertis(?:ement|ing)\b", r"\bpromoted\b",
    r"\bpublicidad\b", r"\bpatrocinad[oa]", r"\bcontenido patrocinado\b",
    r"\ben colaboracion con\b", r"\bespacio de marca\b", r"\bbranded content\b",
)
_SPONSOR_RE = re.compile("|".join(_SPONSOR))

#: Wording that marks a block as the newsletter talking about itself.
_HOUSEKEEPING = (
    r"you (?:are|re) receiving this", r"recibes est[ae] (?:correo|boletin|newsletter)",
    # El Salto's footer. Without this the whole block became a story, complete
    # with the outlet's phone number as its summary.
    r"si no (?:quieres|deseas) (?:seguir )?recib",
    r"para dejar de recibir", r"darte de baja",
    r"this email was sent to", r"este correo se envio a",
    r"was forwarded to you", r"te lo han reenviado",
    r"copyright (?:\(c\)|©)", r"all rights reserved", r"todos los derechos reservados",
    r"^\s*(?:\(c\)|©)\s*\d{4}",
)
_HOUSEKEEPING_RE = re.compile("|".join(_HOUSEKEEPING))

#: Tags that never contain a story and confuse the text extraction if kept.
_DROP_TAGS = ("script", "style", "head", "meta", "link", "noscript", "svg", "title")


def strip_chrome(soup) -> None:
    """Delete furniture from the tree, in place.

    Deliberately conservative about deleting whole blocks: it removes the tags
    that cannot contain a story, and the smallest ancestor of a housekeeping
    phrase. Over-eager block removal is the failure mode that quietly halves a
    newsletter, and unlike over-extraction it leaves no trace.
    """
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    # An unsubscribe or copyright line marks the footer. Remove from its own
    # block downward, not from the line to the end of the document -- some
    # newsletters put the legal boilerplate in a header.
    for element in soup.find_all(string=_HOUSEKEEPING_RE):
        block = _enclosing_block(element)
        if block is not None:
            block.decompose()


def _enclosing_block(element):
    """The nearest ancestor that looks like a self-contained block."""
    node = getattr(element, "parent", None)
    while node is not None:
        if getattr(node, "name", None) in ("td", "tr", "table", "div", "p", "li", "section"):
            return node
        node = getattr(node, "parent", None)
    return None


def is_boilerplate_link(href: str, text: str, *, sender_domain: str = "") -> bool:
    """True when this link cannot be a news article.

    `sender_domain` lets us reject the publisher's own front page -- every
    newsletter links its masthead, and a masthead link has the shape of a story
    link with the outlet's name as its headline.
    """
    url = (href or "").strip()
    if not url or url.startswith(("mailto:", "tel:", "#", "javascript:")):
        return True

    if is_never_article(url):
        return True

    host = domain(url)
    label = normalize(text)
    if label and _SKIP_TEXT_RE.search(label):
        return True

    # A link with no path is a front page, not an article. Checked after the
    # text rules so a mislabelled article link still gets its chance.
    path = url.partition("//")[2].partition("/")[2].split("?")[0].strip("/")
    if not path:
        return True
    if sender_domain and host and host == sender_domain and path in ("", "index.html"):
        return True
    return False


def is_boilerplate_text(text: str) -> bool:
    """True when a string is furniture rather than a headline.

    The same vocabulary `is_boilerplate_link` applies to link text, exposed so it
    can also be applied to a title AFTER it has been chosen. A link wrapped
    around an image has no text to reject, so the appeal reaches the title through
    the image's alt attribute -- which is how "Únete al periodismo valiente"
    became item 0 of a real issue.
    """
    label = normalize(text)
    return bool(label and _SKIP_TEXT_RE.search(label))


def is_sponsored(text: str) -> bool:
    """True when a block announces itself as paid placement."""
    return bool(_SPONSOR_RE.search(normalize(text)))


def is_housekeeping(text: str) -> bool:
    """True when a block is the newsletter talking about itself."""
    return bool(_HOUSEKEEPING_RE.search(normalize(text)))

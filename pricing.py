"""Read the Mirasim model price list.

The prices live on a rendered page rather than in the status payload, so
this module pulls the page and reads its tables. A price is a pair: the
list price the site strikes through, and the price actually charged while a
promotion runs. Both are kept so the plugin can show which one applies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PRICE_URL = "https://mirasim.ai/zh/models"
# The site changes its prices rarely, so a stale list costs nothing.
PRICE_CACHE_TTL = 6 * 3600

_ARTICLE = re.compile(
    r'<article class="mp-card"[^>]*aria-labelledby="mp-v-(?P<vendor>[^"]+)"'
    r"(?P<body>.*?)</article>",
    re.S,
)
_VENDOR = re.compile(r'<h3 class="mp-vendor"[^>]*>(?P<name>.*?)</h3>', re.S)
_BADGE = re.compile(r'<span class="mp-badge">(?P<text>.*?)</span>', re.S)
_TAG = re.compile(r'<span class="mp-tag">(?P<text>.*?)</span>', re.S)
_ROW = re.compile(r"<tr>(?P<row>.*?)</tr>", re.S)
_ROW_HEAD = re.compile(r'<th scope="row">(?P<name>.*?)</th>(?P<cells>.*?)$', re.S)
_CELL = re.compile(r"<td>(?P<cell>.*?)</td>", re.S)
_STRIKE = re.compile(
    r'<s class="mp-was"[^>]*>(?P<list>.*?)</s>(?P<current>[^<]*)', re.S
)
_UPDATED = re.compile(r"更新于\s*<?[^>]*>?\s*(\d{4}-\d{2}-\d{2})")
_SVG = re.compile(r"<svg.*?</svg>", re.S)
_TAG_STRIP = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    """Return the visible text of an HTML fragment."""
    return _TAG_STRIP.sub("", _SVG.sub("", fragment)).strip()


def _model_name(raw: str) -> str:
    """Return a model's name, without the promotion tag that follows it.

    Args:
        raw: The row heading's HTML.

    Returns:
        The model's name.
    """
    return _text(_TAG.sub("", raw))


def _slug(text: str) -> str:
    """Return a lowercase key for a name.

    Args:
        text: The name to normalise.

    Returns:
        The name with spaces and dots replaced by dashes.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")


@dataclass(frozen=True)
class Price:
    """One model's prices, in credits per million tokens.

    Attributes:
        name: The model name as the price page writes it.
        vendor: The vendor heading the model sits under.
        badge: The promotion the whole vendor block carries, if any.
        tag: The promotion this single model carries, if any.
        input_price: ``(list price, charged price)`` for input tokens.
        output_price: ``(list price, charged price)`` for output tokens.
        cache_read_price: ``(list price, charged price)`` for cache reads.
    """

    name: str
    vendor: str
    badge: str
    tag: str
    input_price: tuple[float | None, float | None]
    output_price: tuple[float | None, float | None]
    cache_read_price: tuple[float | None, float | None]

    @property
    def discounted(self) -> bool:
        """Whether a promotion is running for this model."""
        return bool(self.badge or self.tag)


@dataclass(frozen=True)
class PriceList:
    """The whole price page, indexed by the slugs of its model names.

    Attributes:
        entries: The prices, keyed by slug.
        updated: The date the page says it was updated on.
    """

    entries: dict[str, Price]
    updated: str

    def find(self, vendor: str, model_name: str) -> Price | None:
        """Find the price of one model.

        The page names a model after its vendor (``Claude Opus 5.5``) while the
        status payload names it after the model alone (``opus 5.5``), and some
        vendors repeat their own name in the model (``GLM-5.3 Flash``). Both
        spellings are tried, so either way finds the entry.

        A dated snapshot such as ``v4.1-flash-1017`` is priced like the model
        it is a snapshot of, so its suffix is dropped and the base model is
        looked up. Only a trailing ``-<digits>`` counts, which keeps a
        version like ``5.5`` from being mistaken for such a suffix.

        Args:
            vendor: The vendor the status payload reports.
            model_name: The model name the status payload reports.

        Returns:
            The price, or None when the page does not list the model.
        """
        candidates = [
            _slug(f"{vendor} {model_name}"),
            _slug(model_name),
        ]
        base = re.sub(r"-\d+$", "", model_name.strip())
        if base != model_name.strip():
            candidates.append(_slug(f"{vendor} {base}"))
            candidates.append(_slug(base))
        for candidate in candidates:
            price = self.entries.get(candidate)
            if price is not None:
                return price
        return None


def _parse_cell(raw: str) -> tuple[float | None, float | None]:
    """Read one price cell into its list price and charged price.

    Args:
        raw: The cell's HTML.

    Returns:
        ``(list price, charged price)``; the list price is None when the
        price is not struck through, and both are None when the cell holds
        no price at all.
    """
    text = _text(raw)
    if not text or text == "—":
        return None, None
    strike = _STRIKE.search(raw)
    if strike:
        return _number(strike.group("list")), _number(strike.group("current"))
    return None, _number(text)


def _number(text: str) -> float | None:
    """Read a price out of a cell's text.

    Args:
        text: The text of a price.

    Returns:
        The number, or None when the text is not a number.
    """
    text = _text(text).replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_prices(html: str) -> PriceList:
    """Read the price tables out of the price page.

    Args:
        html: The rendered page.

    Returns:
        The prices, indexed by slug, with the page's update date.
    """
    entries: dict[str, Price] = {}
    for article in _ARTICLE.finditer(html):
        body = article.group("body")
        vendor_match = _VENDOR.search(body)
        vendor = _text(vendor_match.group("name")) if vendor_match else ""
        header = body[: body.find("<table")] if "<table" in body else body
        badge_match = _BADGE.search(header)
        badge = _text(badge_match.group("text")) if badge_match else ""
        for row in _ROW.finditer(body):
            head = _ROW_HEAD.search(row.group("row"))
            if not head:
                continue
            raw_name = head.group("name")
            tag_match = _TAG.search(raw_name)
            tag = _text(tag_match.group("text")) if tag_match else ""
            name = _model_name(raw_name)
            cells = [
                _parse_cell(c.group("cell"))
                for c in _CELL.finditer(head.group("cells"))
            ]
            if not name:
                continue
            entry = Price(
                name=name,
                vendor=vendor,
                badge=badge,
                tag=tag,
                input_price=cells[0] if len(cells) > 0 else (None, None),
                output_price=cells[1] if len(cells) > 1 else (None, None),
                cache_read_price=cells[2] if len(cells) > 2 else (None, None),
            )
            entries[_slug(name)] = entry
    updated = _UPDATED.search(html)
    return PriceList(entries=entries, updated=updated.group(1) if updated else "")

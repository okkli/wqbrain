"""WorldQuant BRAIN support-site access: forum search, post/article reading and the glossary.

The support site (https://support.worldquantbrain.com) is a Zendesk Help Center. Pages are
rendered in one shared headless Chromium (Playwright, imported lazily so the server starts
without it) and parsed with BeautifulSoup in a worker thread. BRAIN session cookies come from
an injected ``cookie_provider``; this module never calls the BRAIN API itself. The pure
parsing helpers are module-level so they can be tested against saved HTML.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, List, Optional
from urllib.parse import quote, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag

__all__ = [
    "CookieProvider",
    "ForumClient",
    "ForumError",
    "build_search_url",
    "element_text",
    "has_next_page",
    "is_login_page",
    "is_signed_out",
    "merge_comments",
    "next_page_href",
    "parse_comments",
    "parse_glossary",
    "parse_post",
    "parse_search_results",
    "resolve_post_urls",
    "to_playwright_cookies",
    "validate_locale",
]

log = logging.getLogger("wqmcp.forum")

CookieProvider = Callable[[], Awaitable[List[Dict[str, Any]]]]
"""Async callable returning the current BRAIN session cookies as [{name, value, domain, path}, ...]."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        log.warning("ignoring invalid %s=%r", name, os.environ.get(name))
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        log.warning("ignoring invalid %s=%r", name, os.environ.get(name))
        return default


DEFAULT_BASE_URL = "https://support.worldquantbrain.com"
BRAIN_COOKIE_DOMAIN = "worldquantbrain.com"
GLOSSARY_PATH = "/hc/en-us/articles/4902349883927-Click-here-for-a-list-of-terms-and-their-definitions"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)

MAX_SEARCH_RESULTS = 50
MAX_COMMENTS = 500
POST_BODY_MAX_CHARS = 20_000
COMMENT_MAX_CHARS = 4_000
SNIPPET_MAX_CHARS = 500
MAX_SEARCH_PAGES = 10
MAX_COMMENT_PAGES = 25

NAV_TIMEOUT_S = 30.0      # one page.goto
READY_TIMEOUT_S = 3.0     # wait for server-rendered content to be attached
SSO_TIMEOUT_S = 20.0      # /access/sso redirect chain
SSO_RETRY_AFTER_S = 300.0  # don't retry a failed SSO on every call
SUPPORT_SESSION_TTL_S = 3600.0
PAGE_RESERVE_S = 5.0      # stop paginating when less than this is left of the budget

_LOCALE_RE = re.compile(r"^[a-z]{2}(-[a-z]{2})?$")
_POST_PATH_RE = re.compile(
    r"^/hc/(?:[A-Za-z]{2}(?:-[A-Za-z]{2})?/)?(community/posts|articles)/(\d{1,20})(?:-[^/]*)?/?$"
)
_BARE_ID_RE = re.compile(r"(\d{1,20})(?:-[^/\s]*)?")
_SHORT_REF_RE = re.compile(r"/?(?:community/)?(posts|articles)/(\d{1,20})(?:-[^/\s]*)?/?")
# Sign-in / access endpoints (Zendesk and platform), matched against the lower-cased path.
_LOGIN_PATH_RE = re.compile(
    r"^/(?:hc/(?:[a-z]{2}(?:-[a-z]{2})?/)?signin\b|access/|auth/|sign-?in\b|login\b|users/sign_in\b)"
)
_SUPPORT_LOGIN_PATH_RE = re.compile(r"^/(?:hc/(?:[a-z]{2}(?:-[a-z]{2})?/)?signin\b|access/unauthenticated\b)")


class ForumError(Exception):
    """A forum operation failed. ``code`` is a short machine-readable reason."""

    def __init__(self, message: str, *, code: str = "forum_error") -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- #
# URL helpers
# --------------------------------------------------------------------------- #


def validate_locale(locale: str) -> str:
    """Return the normalised Help Center locale (``zh-cn``, ``en-us``, ``ja``...)."""
    value = (locale or "").strip().lower()
    if not _LOCALE_RE.match(value):
        raise ForumError(f"invalid locale {locale!r}; expected e.g. 'zh-cn' or 'en-us'",
                         code="invalid_argument")
    return value


def _split_base(base_url: str) -> tuple[str, str, str]:
    """(scheme, netloc, root) of a validated base URL."""
    parts = urlsplit(base_url.strip().rstrip("/"))
    if (parts.scheme not in ("http", "https") or not parts.hostname or parts.username
            or parts.password or parts.path or parts.query or parts.fragment):
        raise ValueError(f"invalid forum base_url {base_url!r}")
    netloc = parts.netloc.lower()
    return parts.scheme, netloc, f"{parts.scheme}://{netloc}"


def build_search_url(base_url: str, query: str, page: int = 1, locale: str = "zh-cn") -> str:
    """Help Center search URL with the query properly percent-encoded."""
    _, _, root = _split_base(base_url)
    params: list[tuple[str, str]] = [("query", query)]
    if page > 1:
        params.append(("page", str(page)))
    return f"{root}/hc/{validate_locale(locale)}/search?{urlencode(params, quote_via=quote)}"


def canonical_url(url: str) -> str:
    """URL without query string and fragment."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def post_ref(url: str) -> tuple[Optional[str], Optional[str]]:
    """("post" | "article", id) for a Help Center post/article URL, else (None, None)."""
    m = _POST_PATH_RE.match(urlsplit(url).path)
    if not m:
        return None, None
    return ("post" if m.group(1) == "community/posts" else "article"), m.group(2)


def resolve_post_urls(post: str, base_url: str = DEFAULT_BASE_URL, locale: str = "zh-cn") -> list[str]:
    """Candidate URLs for a post reference, in the order they should be tried.

    Accepts a numeric id (optionally ``123-slug``), ``posts/<id>``, ``community/posts/<id>``,
    ``articles/<id>``, a ``/hc/...`` path, or a full URL on the support host itself. A bare id
    is tried as a community post first, then as an article. Anything else raises ForumError.
    """
    scheme, netloc, root = _split_base(base_url)
    locale = validate_locale(locale)
    ref = (post or "").strip()
    if not ref or any(ch.isspace() or ord(ch) < 32 for ch in ref):
        raise ForumError("post must be a post/article id or a support-site URL", code="invalid_argument")
    if m := _BARE_ID_RE.fullmatch(ref):
        pid = m.group(1)
        return [f"{root}/hc/{locale}/community/posts/{pid}", f"{root}/hc/{locale}/articles/{pid}"]
    if m := _SHORT_REF_RE.fullmatch(ref):
        kind = "community/posts" if m.group(1) == "posts" else "articles"
        return [f"{root}/hc/{locale}/{kind}/{m.group(2)}"]
    if ref.startswith("/hc/"):
        target = root + ref
    elif "://" in ref:
        target = ref
    else:
        raise ForumError(f"unsupported post reference {ref[:100]!r}; pass an id, 'posts/<id>', "
                         f"'articles/<id>' or a {root} URL", code="invalid_argument")
    parts = urlsplit(target)
    if (parts.scheme.lower() != scheme or parts.netloc.lower() != netloc
            or parts.username is not None or parts.password is not None):
        raise ForumError(f"only {root} URLs are allowed", code="invalid_argument")
    if not _POST_PATH_RE.match(parts.path):
        raise ForumError("URL must point to /hc/<locale>/community/posts/<id> or /hc/<locale>/articles/<id>",
                         code="invalid_argument")
    return [urlunsplit((scheme, netloc, parts.path, "", ""))]


def to_playwright_cookies(cookies: Iterable[Dict[str, Any]],
                          domain: str = BRAIN_COOKIE_DOMAIN) -> list[dict[str, Any]]:
    """Playwright cookie dicts for the BRAIN cookies scoped to ``domain`` or its subdomains."""
    out: list[dict[str, Any]] = []
    for c in cookies:
        try:
            name, value = str(c["name"]), str(c["value"])
        except (KeyError, TypeError):
            continue
        cookie_domain = str(c.get("domain") or "").strip().lower()
        host = cookie_domain.lstrip(".")
        if not name or not (host == domain or host.endswith("." + domain)):
            continue
        item: dict[str, Any] = {
            "name": name,
            "value": value,
            "domain": cookie_domain,
            "path": str(c.get("path") or "/"),
            "secure": True,
            "httpOnly": bool(c.get("httpOnly", True)),
            "sameSite": "Lax",
        }
        expires = c.get("expires")
        if isinstance(expires, (int, float)) and not isinstance(expires, bool) and expires > 0:
            item["expires"] = float(expires)
        out.append(item)
    return out


def _redact(url: str) -> str:
    """URL without query/fragment, for logs and errors (SSO URLs can carry tokens)."""
    return canonical_url(url) if url else ""


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #

_SKIPPED_STRINGS = (Comment, Declaration, Doctype, ProcessingInstruction)
_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "template", "head", "title", "meta", "link", "button",
    "svg", "iframe", "object", "input", "select", "textarea",
})
_BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "caption", "dd", "details", "dialog", "div",
    "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
    "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "summary",
    "table", "tbody", "tfoot", "thead", "tr", "ul",
})
_WS_RE = re.compile(r"[ \t\n\r\f\v]+")
_SPACES_RE = re.compile(r"[ \t]+")
_BREAKS_RE = re.compile(r"[ \t]*[\x01\n][\x01\n \t]*")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_PLACEHOLDER_RE = re.compile("\x00(\\d+)\x00")
_SOFT = "\x01"  # block boundary: at least one line break, merged with neighbours


def _clean_chars(text: str) -> str:
    return (text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
            .replace("​", "").replace("﻿", "").replace("\x00", "").replace(_SOFT, ""))


def _raw_text(node: Tag) -> str:
    """Verbatim text of a <pre>/<code> element; <br> and nested blocks become newlines."""
    out: list[str] = []

    def newline() -> None:
        if out and not out[-1].endswith("\n"):
            out.append("\n")

    stack: list[tuple[Any, bool]] = [(c, False) for c in reversed(node.contents)]
    while stack:
        cur, closing = stack.pop()
        if closing:
            newline()
            continue
        if isinstance(cur, NavigableString):
            if not isinstance(cur, _SKIPPED_STRINGS) and str(cur):
                out.append(_clean_chars(str(cur)))
            continue
        if not isinstance(cur, Tag) or cur.name in _SKIP_TAGS:
            continue
        if cur.name == "br":
            out.append("\n")
            continue
        if cur.name in _BLOCK_TAGS:
            newline()
            stack.append((cur, True))
        stack.extend((c, False) for c in reversed(cur.contents))
    return "".join(out).strip("\n")


def _list_marker(li: Tag) -> str:
    parent = li.parent
    if isinstance(parent, Tag) and parent.name == "ol":
        try:
            start = int(parent.get("start", 1))
        except (TypeError, ValueError):
            start = 1
        return f"{start + len(li.find_previous_siblings('li'))}. "
    return "- "


def _nodes_text(nodes: Iterable[Any]) -> str:
    parts: list[str] = []
    verbatim: list[str] = []

    def keep(text: str, block: bool) -> None:
        token = f"\x00{len(verbatim)}\x00"
        verbatim.append(text)
        parts.append(f"{_SOFT}{token}{_SOFT}" if block else token)

    stack: list[tuple[Any, bool]] = [(n, False) for n in reversed(list(nodes))]
    while stack:
        node, closing = stack.pop()
        if closing:
            parts.append(_SOFT)
            continue
        if isinstance(node, NavigableString):
            if not isinstance(node, _SKIPPED_STRINGS):
                parts.append(_WS_RE.sub(" ", _clean_chars(str(node))))
            continue
        if not isinstance(node, Tag) or node.name in _SKIP_TAGS:
            continue
        name = node.name
        if name == "br":
            parts.append("\n")
            continue
        if name == "pre" or name == "code":
            code = _raw_text(node)
            if name == "pre" or "\n" in code:
                if code.strip():
                    keep(f"```\n{code}\n```", block=True)
            elif code.strip():
                keep(f"`{code}`", block=False)
            continue
        if name in ("td", "th") and node.find_previous_sibling(["td", "th"]) is not None:
            parts.append(" | ")
        if name in _BLOCK_TAGS:
            parts.append(_SOFT)
            if name == "li":
                parts.append(_list_marker(node))
            stack.append((node, True))
        stack.extend((c, False) for c in reversed(node.contents))

    text = "".join(parts)
    # A run of block boundaries and <br>s becomes max(1, number of <br>) line breaks.
    text = _BREAKS_RE.sub(lambda m: "\n" * max(1, m.group(0).count("\n")), text)
    text = "\n".join(_SPACES_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = _BLANK_LINES_RE.sub("\n\n", text).strip()
    if verbatim:
        text = _PLACEHOLDER_RE.sub(lambda m: verbatim[int(m.group(1))], text)
    return text


def element_text(el: Any) -> str:
    """Readable text of an element: block elements and <br> become line breaks, <pre>/<code>
    content is kept verbatim (fenced with ``` / wrapped in backticks), runs of blank lines are
    collapsed to one."""
    if el is None:
        return ""
    return _nodes_text(el.contents if isinstance(el, Tag) else [el])


def _one_line(el: Any) -> str:
    return " ".join(element_text(el).split()) if el is not None else ""


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    newline = cut.rfind("\n", int(limit * 0.8))
    if newline > 0:
        cut = cut[:newline]
    return cut.rstrip() + "\n…[truncated]", True


def _int_in(text: str) -> Optional[int]:
    m = re.search(r"-?\d+", (text or "").replace(",", ""))
    return int(m.group()) if m else None


# --------------------------------------------------------------------------- #
# Page classification
# --------------------------------------------------------------------------- #

_CONTENT_SELECTOR = (
    ".post-body, .article-body, li.search-result-list-item, .search-results-list, "
    ".search-results, .search-result-title"
)
_LOGIN_FORM_SELECTOR = (
    'input[type="password"], form#login-form, form[action*="signin"], form[action*="/access/login"]'
)
_DENIED_MARKERS = (
    "您未被授权", "未被授权", "无权访问", "unauthorized", "not authorized", "not authorised",
    "access denied", "sign in", "sign-in", "log in", "登录", "登入",
)
_SIGNIN_LINK_SELECTOR = 'a[data-auth-action="signin"], a.sign-in, header a[href*="/signin"]'
_USER_MENU_SELECTOR = '.user-info, #user-menu, .user-avatar, [data-auth-action="signout"], a[href*="/access/logout"]'
_NEXT_SELECTOR = (
    'nav.pagination li.pagination-next a[href], li.pagination-next a[href], '
    'a.pagination-next-link[href], a[rel~="next"][href]'
)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html or "", "html.parser")


def _login_from_soup(url: str, soup: BeautifulSoup) -> bool:
    if url and _LOGIN_PATH_RE.match(urlsplit(url).path.lower()):
        return True
    if soup.select_one(_CONTENT_SELECTOR) is not None:
        return False  # real content; words like "Unauthorized" in a post are not a signal
    if soup.select_one(_LOGIN_FORM_SELECTOR) is not None:
        return True
    texts = [soup.title.get_text(" ") if soup.title else ""]
    texts += [el.get_text(" ") for el in soup.select(".error-page")]
    blob = " ".join(texts).lower()
    return any(marker in blob for marker in _DENIED_MARKERS)


def is_login_page(url: str, html: str) -> bool:
    """True for a sign-in / not-authorized page (by URL, or by page markers when the page
    carries no forum content)."""
    return _login_from_soup(url, _soup(html))


def _signed_out_from_soup(soup: BeautifulSoup) -> bool:
    return (soup.select_one(_SIGNIN_LINK_SELECTOR) is not None
            and soup.select_one(_USER_MENU_SELECTOR) is None)


def is_signed_out(html: str) -> bool:
    """True when the Help Center header shows a "Sign in" link and no user menu."""
    return _signed_out_from_soup(_soup(html))


def _next_href_from_soup(soup: BeautifulSoup) -> Optional[str]:
    el = soup.select_one(_NEXT_SELECTOR)
    href = (el.get("href") or "").strip() if el is not None else ""
    return href or None


def next_page_href(html: str) -> Optional[str]:
    """href of the pagination "next" link, if any."""
    return _next_href_from_soup(_soup(html))


def has_next_page(html: str) -> bool:
    """Whether the page has a pagination "next" link (decides "last page" without waiting)."""
    return next_page_href(html) is not None


# --------------------------------------------------------------------------- #
# Search, post and comment parsing
# --------------------------------------------------------------------------- #


def _search_results_from_soup(soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
    _, _, root = _split_base(base_url)
    results: list[dict[str, Any]] = []
    for item in soup.select("li.search-result-list-item"):
        link = item.select_one("h2.search-result-title a[href], .search-result-title a[href]")
        if link is None:
            continue
        url = canonical_url(urljoin(root + "/", link["href"].strip()))
        kind, pid = post_ref(url)

        votes_el = (item.select_one('.search-result-votes span[aria-hidden="true"]')
                    or item.select_one(".search-result-votes"))
        count_el = (item.select_one('.search-result-meta-count span[aria-hidden="true"]')
                    or item.select_one(".search-result-meta-count"))
        author = None
        for meta in item.select(".meta-group .meta-data"):
            if meta.select_one("time") is None and (text := _one_line(meta)):
                author = text
                break
        time_el = item.select_one(".meta-group time[datetime], time[datetime]")
        date = (time_el.get("datetime") or _one_line(time_el)) if time_el is not None else None
        snippet, _ = _truncate(_one_line(item.select_one(".search-results-description")), SNIPPET_MAX_CHARS)

        results.append({
            "title": _one_line(link),
            "url": url,
            "id": pid,
            "type": kind,
            "snippet": snippet,
            "votes": (_int_in(votes_el.get_text()) if votes_el is not None else None) or 0,
            "comments": (_int_in(count_el.get_text()) if count_el is not None else None) or 0,
            "author": author,
            "date": date,
            "breadcrumbs": [t for li in item.select("ol.search-result-breadcrumbs li") if (t := _one_line(li))],
        })
    return results


def parse_search_results(html: str, base_url: str = DEFAULT_BASE_URL) -> list[dict[str, Any]]:
    """Results of one Help Center search page: {title, url, id, type, snippet, votes, comments,
    author, date, breadcrumbs}."""
    return _search_results_from_soup(_soup(html), base_url)


def _outside_comments(soup: BeautifulSoup, selector: str) -> Optional[Tag]:
    for el in soup.select(selector):
        if el.find_parent(class_="comment") is None:
            return el
    return None


def _post_from_soup(soup: BeautifulSoup, max_chars: int = POST_BODY_MAX_CHARS) -> Optional[dict[str, Any]]:
    body_el, kind = soup.select_one(".post-body"), "post"
    if body_el is None:
        body_el, kind = soup.select_one(".article-body"), "article"
    if body_el is None:
        return None

    title_el = soup.select_one(".post-title h1[title], h1.post-title[title], h1.article-title[title]")
    title = (title_el.get("title") or "").strip() if title_el is not None else ""
    if not title:
        title = _one_line(soup.select_one(
            ".post-title h1, h1.post-title, .post-title, h1.article-title, .article-title, .article__title"))

    author_el = soup.select_one(".post-author span[title], .article-author span[title]")
    author = (author_el.get("title") or "").strip() if author_el is not None else ""
    if not author:
        author = _one_line(soup.select_one(
            '.post-author a[href*="/profiles/"], .article-author a[href*="/profiles/"], '
            '.article-meta a[href*="/profiles/"]'))

    time_el = soup.select_one(".post-meta time[datetime], .article-meta time[datetime]")
    if time_el is not None:
        date = time_el.get("datetime") or _one_line(time_el)
    else:
        date = _one_line(soup.select_one(".post-meta .meta-data, .article-meta .meta-data"))

    votes_el = _outside_comments(soup, ".vote-sum")
    body, truncated = _truncate(element_text(body_el), max_chars)
    return {
        "type": kind,
        "title": title or None,
        "author": author or None,
        "date": date or None,
        "votes": _int_in(votes_el.get_text()) if votes_el is not None else None,
        "body": body,
        "truncated": truncated,
    }


def parse_post(html: str) -> Optional[dict[str, Any]]:
    """Post or article on a page: {type, title, author, date, votes, body, truncated}; None when
    the page has no post/article body."""
    return _post_from_soup(_soup(html))


_COMMENT_ID_RE = re.compile(r"^(?:community_)?comment_\d+$")


def _comments_from_soup(soup: BeautifulSoup, max_chars: int = COMMENT_MAX_CHARS) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    for el in soup.select(".comment"):
        if el.find_parent(class_="comment") is not None:
            continue  # nested; counted with its outer comment
        body_el = el.select_one(".comment-body")
        if body_el is None:
            continue
        cid = (el.get("id") or "").strip() or None
        if cid is None:
            inner = el.find(id=_COMMENT_ID_RE)
            cid = inner.get("id") if inner is not None else None

        author_el = el.select_one(".comment-author span[title], .comment-meta span[title]")
        author = (author_el.get("title") or "").strip() if author_el is not None else ""
        if not author:
            author = _one_line(el.select_one('.comment-author a[href*="/profiles/"], .comment-meta a'))
        time_el = el.select_one(".comment-meta time[datetime], time[datetime]")
        if time_el is not None:
            date = time_el.get("datetime") or _one_line(time_el)
        else:
            date = _one_line(el.select_one(".comment-meta .meta-data"))
        body, truncated = _truncate(element_text(body_el), max_chars)
        comments.append({"id": cid, "author": author or None, "date": date or None,
                         "body": body, "truncated": truncated})
    return comments


def parse_comments(html: str) -> list[dict[str, Any]]:
    """Comments on one page: [{id, author, date, body, truncated}] (id is the element id)."""
    return _comments_from_soup(_soup(html))


def merge_comments(collected: list[dict[str, Any]], page: list[dict[str, Any]], seen: set) -> int:
    """Append the comments of ``page`` not seen on earlier pages; return how many were added.

    Comments are keyed by element id. Only id-less comments fall back to (author, date, body),
    and that key is compared with earlier pages only, so identical short replies on the same
    page are all kept.
    """
    added = 0
    page_keys = set()
    for c in page:
        key = ("id", c["id"]) if c.get("id") else ("text", c.get("author"), c.get("date"), c.get("body"))
        if key in seen:
            continue
        page_keys.add(key)
        collected.append(c)
        added += 1
    seen |= page_keys
    return added


# --------------------------------------------------------------------------- #
# Glossary parsing
# --------------------------------------------------------------------------- #

_RELATIVE_TIME_RE = re.compile(
    r"\b(?:about\s+|over\s+|almost\s+)?(?:\d+|an?|one|a\s+few)\s+"
    r"(?:second|minute|hour|day|week|month|year)s?\s+ago\b", re.I)
_META_LINE_RES = (
    re.compile(r"~?\s*\d+\s+minutes?\s+read", re.I),
    re.compile(r"(?:follow|following|unfollow|not yet followed|updated|edited)", re.I),
    re.compile(r"AS\d+"),                                # avatar initials
    re.compile(r"[A-Z]"),                                # letter section heading
    re.compile(r"[A-Z](?:\s*[-|·•,/]\s*[A-Z])+"),        # "A - B - C" letter navigation
)
_DEFINITION_STARTERS = frozenset({
    "the", "a", "an", "this", "that", "it", "is", "are", "was", "were", "for", "to", "in", "on",
    "at", "by", "with", "if", "when", "and", "or", "of", "as",
})
_TERM_BLOCKS = frozenset({"h2", "h3", "h4", "h5", "h6", "dt"})
_CONTAINER_BLOCKS = frozenset({
    "div", "section", "article", "ul", "ol", "dl", "blockquote", "main", "header", "footer",
    "figure", "details", "tbody", "thead", "tfoot",
})
_TABLE_HEADER_WORDS = frozenset({"term", "terms", "name", "术语", "名称", "definition"})
_INLINE_DEF_RE = re.compile(r"^([^:：]{2,60}?)\s*[:：]\s+(.+)$")
_MIN_STRUCTURED_TERMS = 3


def _is_navigation_or_metadata(line: str) -> bool:
    text = line.strip()
    if not text:
        return True
    # Relative timestamps ("Updated 3 days ago") are matched on word boundaries, so words that
    # merely contain "ago" (Chicago, diagonal) and real sentences stay.
    if _RELATIVE_TIME_RE.search(text):
        rest = _RELATIVE_TIME_RE.sub("", text).strip(" .·-").lower()
        if rest in ("", "updated", "posted", "edited", "created"):
            return True
    return any(p.fullmatch(text) for p in _META_LINE_RES)


def _looks_like_term(line: str) -> bool:
    text = line.strip().rstrip(":：").strip()
    if not 2 <= len(text) <= 80 or _is_navigation_or_metadata(text):
        return False
    if text[-1] in ".!?;,。；，！？":
        return False
    words = text.split()
    if len(words) > 8 or words[0].lower() in _DEFINITION_STARTERS:
        return False
    return bool(re.match(r"[A-Z0-9]", text))


def _finalize_terms(entries: list[tuple[str, list[str]]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_term, parts in entries:
        term = " ".join(raw_term.split()).rstrip(":：").strip()
        definition = "\n".join(p.strip() for p in parts if p.strip() and not _is_navigation_or_metadata(p))
        if not term or not definition or len(term) > 100 or _is_navigation_or_metadata(term):
            continue
        key = (term.lower(), definition)
        if key not in seen:
            seen.add(key)
            out.append({"term": term, "definition": definition})
    return out


def _glossary_from_tables(body: Tag) -> list[dict[str, str]]:
    entries: list[tuple[str, list[str]]] = []
    for tr in body.select("tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
        if len(cells) < 2 or all(c.name == "th" for c in cells):
            continue
        term = _one_line(cells[0])
        if term.lower() in _TABLE_HEADER_WORDS:
            continue
        entries.append((term, [element_text(c) for c in cells[1:]]))
    return _finalize_terms(entries)


def _has_block_child(tag: Tag) -> bool:
    return any(isinstance(c, Tag) and (c.name in _BLOCK_TAGS) for c in tag.children)


def _iter_blocks(root: Tag) -> Iterable[Any]:
    """Leaf blocks of ``root`` in document order; runs of loose inline nodes are yielded as lists."""
    run: list[Any] = []
    for child in root.children:
        if isinstance(child, Tag) and child.name in _SKIP_TAGS:
            continue
        if not (isinstance(child, Tag) and child.name in _BLOCK_TAGS):
            run.append(child)
            continue
        if run:
            yield run
            run = []
        if child.name in _CONTAINER_BLOCKS and _has_block_child(child):
            yield from _iter_blocks(child)
        else:
            yield child


def _leading_bold(nodes: Iterable[Any]) -> Optional[str]:
    """Text of a <strong>/<b> that starts a line (possibly inside span/a/em wrappers)."""
    for n in nodes:
        if isinstance(n, NavigableString):
            if isinstance(n, _SKIPPED_STRINGS) or not n.strip():
                continue
            return None
        if not isinstance(n, Tag):
            continue
        if n.name in ("strong", "b"):
            return _one_line(n) or None
        if n.name in ("span", "a", "em", "i", "u", "font"):
            return _leading_bold(n.contents)
        return None
    return None


def _segments(block: Any) -> list[tuple[str, Optional[str]]]:
    """(text, leading bold text) for each <br>-separated line of a block."""
    nodes = block.contents if isinstance(block, Tag) else block
    groups: list[list[Any]] = [[]]
    for n in nodes:
        if isinstance(n, Tag) and n.name == "br":
            groups.append([])
        else:
            groups[-1].append(n)
    return [(_nodes_text(g), _leading_bold(g)) for g in groups if g]


def _split_bold_term(text: str, bold: Optional[str]) -> Optional[tuple[str, str]]:
    if not bold:
        return None
    term = bold.strip().rstrip(":：").strip()
    if not term or len(term) > 100 or term[-1] in ".!?。" or len(term.split()) > 12:
        return None
    flat = " ".join(text.split())
    if not flat.startswith(term):
        return None
    return term, re.sub(r"^\s*[:：\-–—]?\s*", "", flat[len(term):])


def _glossary_from_markers(body: Tag) -> list[dict[str, str]]:
    """Terms marked by headings, <dt>, or a leading <strong>/<b>; definitions are what follows."""
    entries: list[tuple[str, list[str]]] = []
    current: Optional[tuple[str, list[str]]] = None
    for block in _iter_blocks(body):
        if isinstance(block, Tag) and block.name == "table":
            continue
        if isinstance(block, Tag) and block.name in _TERM_BLOCKS:
            text = _one_line(block)
            if not text or _is_navigation_or_metadata(text):
                current = None  # letter heading or noise: close the previous term
                continue
            current = (text, [])
            entries.append(current)
            continue
        for text, bold in _segments(block):
            if not text or _is_navigation_or_metadata(text):
                continue
            split = _split_bold_term(text, bold)
            if split is not None:
                current = (split[0], [split[1]] if split[1] else [])
                entries.append(current)
            elif current is not None:
                current[1].append(text)
    return _finalize_terms(entries)


def _glossary_from_lines(lines: Iterable[str]) -> list[dict[str, str]]:
    """Fallback for unstructured pages: short capitalised lines start a term."""
    entries: list[tuple[str, list[str]]] = []
    current: Optional[tuple[str, list[str]]] = None
    for raw in lines:
        line = raw.strip()
        if not line or _is_navigation_or_metadata(line):
            continue
        m = _INLINE_DEF_RE.match(line)
        if m and _looks_like_term(m.group(1)):
            current = (m.group(1), [m.group(2)])
            entries.append(current)
        elif _looks_like_term(line):
            current = (line, [])
            entries.append(current)
        elif current is not None:
            current[1].append(line)
    return _finalize_terms(entries)


def _glossary_from_body(body: Tag) -> list[dict[str, str]]:
    candidates = []
    for strategy in (_glossary_from_tables, _glossary_from_markers):
        terms = strategy(body)
        if len(terms) >= _MIN_STRUCTURED_TERMS:
            return terms
        candidates.append(terms)
    candidates.append(_glossary_from_lines(element_text(body).split("\n")))
    return max(candidates, key=len)


def parse_glossary(html: str) -> list[dict[str, str]]:
    """[{term, definition}] from the glossary article. Uses table rows, then heading/<dt>/bold
    term markers, then a line heuristic. Raises ForumError when the article body is missing."""
    soup = _soup(html)
    body = soup.select_one(".article-body") or soup.select_one(".post-body")
    if body is None:
        raise ForumError("glossary article body (.article-body) not found", code="parse_error")
    return _glossary_from_body(body)


# --------------------------------------------------------------------------- #
# Page analysis (runs in a worker thread: one parse per page)
# --------------------------------------------------------------------------- #


@dataclass
class _Parsed:
    login_page: bool = False
    signed_out: bool = False
    next_href: Optional[str] = None
    results: list[dict[str, Any]] = field(default_factory=list)
    post: Optional[dict[str, Any]] = None
    comments: list[dict[str, Any]] = field(default_factory=list)
    terms: Optional[list[dict[str, str]]] = None
    error: Optional[str] = None


def _analyze_base(html: str, url: str) -> tuple[BeautifulSoup, _Parsed]:
    soup = _soup(html)
    return soup, _Parsed(login_page=_login_from_soup(url, soup), signed_out=_signed_out_from_soup(soup),
                         next_href=_next_href_from_soup(soup))


def _analyze_search(html: str, url: str, *, base_url: str) -> _Parsed:
    soup, parsed = _analyze_base(html, url)
    parsed.results = _search_results_from_soup(soup, base_url)
    return parsed


def _analyze_post(html: str, url: str) -> _Parsed:
    soup, parsed = _analyze_base(html, url)
    parsed.post = _post_from_soup(soup)
    parsed.comments = _comments_from_soup(soup)
    return parsed


def _analyze_glossary(html: str, url: str) -> _Parsed:
    soup, parsed = _analyze_base(html, url)
    body = soup.select_one(".article-body") or soup.select_one(".post-body")
    if body is None:
        parsed.error = "glossary article body (.article-body) not found"
    else:
        parsed.terms = _glossary_from_body(body)
    return parsed


_SEARCH_READY = "li.search-result-list-item, .search-results, .search-results-list, .error-page, input[type=password]"
_POST_READY = ".post-body, .article-body, .error-page, input[type=password]"
_BLOCKED_RESOURCES = frozenset({"image", "media", "font"})


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0][:300] if text else type(exc).__name__


class _Deadline:
    def __init__(self, seconds: float) -> None:
        self._end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def ms(self, cap_s: float) -> float:
        """Playwright timeout in ms, capped by the remaining budget (never 0 = unlimited)."""
        return max(500.0, min(cap_s, self.remaining()) * 1000.0)


async def _route_filter(route: Any) -> None:
    with contextlib.suppress(Exception):  # page may close mid-flight
        if route.request.resource_type in _BLOCKED_RESOURCES:
            await route.abort()
        else:
            await route.continue_()


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class _Visit:
    """One operation's browser context: navigation with sign-in handling."""

    def __init__(self, client: ForumClient, page: Any, deadline: _Deadline, want_login: bool) -> None:
        self._client = client
        self._page = page
        self._deadline = deadline
        self._want_login = want_login
        self.sso_attempted = False
        self.sso_ok = False
        self.signed_in: Optional[bool] = None

    async def sign_in(self) -> bool:
        self.sso_attempted = True
        self.sso_ok = await self._client._sso(self._page, self._deadline)
        return self.sso_ok

    async def fetch(self, url: str, *, ready: str,
                    analyze: Callable[[str, str], _Parsed]) -> tuple[Optional[int], str, _Parsed]:
        """Navigate to ``url`` and parse it; sign in once via SSO when the site asks for it."""
        for attempt in range(2):
            status, final_url, html = await self._client._navigate(self._page, url, ready, self._deadline)
            parsed = await asyncio.to_thread(analyze, html, final_url)
            denied = parsed.login_page or status in (401, 403)
            wants_sso = denied or (self._want_login and parsed.signed_out)
            if wants_sso and attempt == 0 and not self.sso_attempted and await self.sign_in():
                continue
            if denied:
                raise self._client._auth_error(url, status, self)
            self._client._check_same_site(final_url)
            self.signed_in = not parsed.signed_out
            return status, final_url, parsed
        raise AssertionError("unreachable")


class ForumClient:
    """Support-site client sharing one headless browser across calls.

    Each call gets a fresh browser context seeded with the BRAIN cookies (and a cached Zendesk
    session when available); at most ``max_concurrency`` calls use the browser at once and each
    call is bounded by ``timeout`` seconds.
    """

    def __init__(
        self,
        cookie_provider: CookieProvider,
        *,
        base_url: str = DEFAULT_BASE_URL,
        max_concurrency: int = _env_int("WQMCP_FORUM_CONCURRENCY", 2),
        browser_channel: Optional[str] = os.environ.get("WQMCP_FORUM_BROWSER_CHANNEL", "chrome"),
        timeout: float = _env_float("WQMCP_FORUM_TIMEOUT", 90.0),
        glossary_ttl: float = _env_float("WQMCP_GLOSSARY_TTL", 86400.0),
        chromium_sandbox: bool = os.environ.get("WQMCP_FORUM_CHROMIUM_SANDBOX", "") in ("1", "true", "yes"),
    ) -> None:
        self._scheme, self._netloc, self.base_url = _split_base(base_url)
        self._hostname = urlsplit(self.base_url).hostname or ""
        self._cookie_provider = cookie_provider
        self._channel = (browser_channel or "").strip() or None
        self._channel_failed = False
        # Playwright passes --no-sandbox unless chromium_sandbox=True; the sandbox cannot run as
        # root in containers, so it is opt-in. Navigation is limited to the support host anyway.
        self._chromium_sandbox = chromium_sandbox
        self._timeout = max(5.0, float(timeout))
        self._glossary_ttl = max(0.0, float(glossary_ttl))
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._browser_lock = asyncio.Lock()
        self._glossary_lock = asyncio.Lock()
        self._playwright: Any = None
        self._browser: Any = None
        self._support_cookies: list[dict[str, Any]] = []
        self._support_cookies_at = 0.0
        self._sso_failed_at: Optional[float] = None
        self._cookie_error: Optional[str] = None
        self._glossary: Optional[tuple[float, dict[str, Any]]] = None

    async def __aenter__(self) -> ForumClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ public API

    async def search_posts(self, query: str, max_results: int = 20, locale: str = "zh-cn") -> Dict[str, Any]:
        """Search the Help Center (community posts and articles).

        Returns {query, locale, results: [{title, url, id, type, snippet, votes, comments, author,
        date, breadcrumbs}], count, has_more, complete, signed_in, warning?}. ``complete`` is False
        when a later page failed and only the results collected so far are returned.
        """
        query = (query or "").strip()
        if not query:
            raise ForumError("query must not be empty", code="invalid_argument")
        if len(query) > 500:
            raise ForumError("query is too long (max 500 characters)", code="invalid_argument")
        locale = validate_locale(locale)
        limit = min(max(int(max_results), 1), MAX_SEARCH_RESULTS)
        deadline = _Deadline(self._timeout)
        return await self._run(self._search(query, limit, locale, deadline))

    async def read_post(self, post: str, include_comments: bool = True, max_comments: int = 100,
                        locale: str = "zh-cn") -> Dict[str, Any]:
        """Read a community post or article and (optionally) its comments.

        ``post`` is an id (tried as a post, then as an article), ``posts/<id>``, ``articles/<id>``
        or a URL on the support host. Returns {post: {id, type, title, author, date, votes, body,
        url, truncated}, comments: [{id, author, date, body, truncated}], total_comments,
        has_more_comments, complete, signed_in, warning?}.
        """
        candidates = resolve_post_urls(post, self.base_url, locale)
        limit = min(max(int(max_comments), 0), MAX_COMMENTS)
        deadline = _Deadline(self._timeout)
        return await self._run(self._read_post(candidates, bool(include_comments) and limit > 0, limit, deadline))

    async def get_glossary_terms(self) -> Dict[str, Any]:
        """Glossary terms: {terms: [{term, definition}], count, source_url, fetched_at, cached}."""
        hit = self._glossary_hit()
        if hit is not None:
            return hit
        async with self._glossary_lock:
            hit = self._glossary_hit()
            if hit is not None:
                return hit
            result = await self._run(self._fetch_glossary(_Deadline(self._timeout)))
            self._glossary = (time.monotonic(), result)
            return {**result, "terms": list(result["terms"]), "cached": False}

    async def aclose(self) -> None:
        """Close the shared browser and the Playwright driver."""
        async with self._browser_lock:
            browser, playwright = self._browser, self._playwright
            self._browser = self._playwright = None
        if browser is not None:
            with contextlib.suppress(Exception):
                await browser.close()
        if playwright is not None:
            with contextlib.suppress(Exception):
                await playwright.stop()

    # ------------------------------------------------------------------ operations

    async def _run(self, coro: Awaitable[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return await asyncio.wait_for(coro, timeout=self._timeout + 10.0)
        except ForumError:
            raise
        except asyncio.TimeoutError as exc:
            raise ForumError(f"forum operation timed out after {self._timeout:.0f}s", code="timeout") from exc
        except Exception as exc:
            log.exception("forum operation failed")
            raise ForumError(f"forum operation failed: {type(exc).__name__}: {_first_line(exc)}",
                             code="internal") from exc

    async def _search(self, query: str, limit: int, locale: str, deadline: _Deadline) -> Dict[str, Any]:
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        warnings: list[str] = []
        complete, has_more = True, False
        analyze = functools.partial(_analyze_search, base_url=self.base_url)
        async with self._visit(deadline, want_login=True) as visit:
            page_no = 1
            while True:
                url = build_search_url(self.base_url, query, page=page_no, locale=locale)
                try:
                    status, _, parsed = await visit.fetch(url, ready=_SEARCH_READY, analyze=analyze)
                    if status == 404 and page_no > 1:
                        break  # past the last page
                    if status is not None and status >= 400:
                        raise ForumError(f"search page {page_no} returned HTTP {status}", code="http_error")
                except ForumError as exc:
                    if not results:
                        raise
                    complete = False
                    warnings.append(f"stopped at search page {page_no}: {exc}")
                    break
                added = 0
                for item in parsed.results:
                    if item["url"] in seen:
                        continue
                    if len(results) >= limit:
                        has_more = True  # more on this page than requested
                        break
                    seen.add(item["url"])
                    results.append(item)
                    added += 1
                if len(results) >= limit:
                    has_more = has_more or parsed.next_href is not None
                    break
                if parsed.next_href is None or added == 0:
                    break
                if page_no >= MAX_SEARCH_PAGES:
                    has_more = True
                    break
                if deadline.remaining() < PAGE_RESERVE_S:
                    complete = False
                    warnings.append("time budget exhausted; later result pages were not read")
                    break
                page_no += 1
            signed_in = visit.signed_in
        if signed_in is False:
            warnings.append("not signed in to the support site; restricted (consultant-only) posts may be missing")
        out: Dict[str, Any] = {"query": query, "locale": locale, "results": results, "count": len(results),
                               "has_more": has_more, "complete": complete, "signed_in": signed_in}
        if warnings:
            out["warning"] = "; ".join(warnings)
        return out

    async def _read_post(self, candidates: list[str], include_comments: bool, max_comments: int,
                         deadline: _Deadline) -> Dict[str, Any]:
        async with self._visit(deadline, want_login=True) as visit:
            first: Optional[_Parsed] = None
            final_url = ""
            tried: list[str] = []
            for url in candidates:
                status, final_url, parsed = await visit.fetch(url, ready=_POST_READY, analyze=_analyze_post)
                if parsed.post is not None and (status is None or status < 400):
                    first = parsed
                    break
                if status is not None and status >= 400 and status != 404:
                    raise ForumError(f"{_redact(url)} returned HTTP {status}", code="http_error")
                tried.append(f"{_redact(url)} ({'HTTP 404' if status == 404 else 'no post body'})")
            if first is None:
                hint = ("; not signed in to the support site, so it may be restricted"
                        if visit.signed_in is False else "")
                raise ForumError("post not found; tried " + ", ".join(tried) + hint, code="not_found")

            canonical = canonical_url(final_url)
            kind, pid = post_ref(canonical)
            post = {"id": pid, **first.post, "url": canonical}
            if kind:
                post["type"] = kind

            comments: list[dict[str, Any]] = []
            warnings: list[str] = []
            complete, has_more = True, False
            if include_comments:
                seen: set = set()
                page_comments, next_href, page_no = first.comments, first.next_href, 1
                while True:
                    added = merge_comments(comments, page_comments, seen)
                    if len(comments) >= max_comments:
                        has_more = len(comments) > max_comments or next_href is not None
                        del comments[max_comments:]
                        break
                    if next_href is None or (page_no > 1 and added == 0):
                        break  # last page (or the site clamped to a page we already have)
                    if page_no >= MAX_COMMENT_PAGES:
                        has_more = True
                        break
                    if deadline.remaining() < PAGE_RESERVE_S:
                        complete = False
                        warnings.append("time budget exhausted; later comment pages were not read")
                        break
                    page_no += 1
                    url = self._comment_page_url(canonical, next_href, page_no)
                    try:
                        status, _, parsed = await visit.fetch(url, ready=_POST_READY, analyze=_analyze_post)
                        if status == 404:
                            break
                        if status is not None and status >= 400:
                            raise ForumError(f"HTTP {status}", code="http_error")
                    except ForumError as exc:
                        complete = False
                        warnings.append(f"comment page {page_no} could not be read: {exc}")
                        break
                    page_comments, next_href = parsed.comments, parsed.next_href
            signed_in = visit.signed_in
        out: Dict[str, Any] = {"post": post, "comments": comments, "total_comments": len(comments),
                               "has_more_comments": has_more, "complete": complete, "signed_in": signed_in}
        if warnings:
            out["warning"] = "; ".join(warnings)
        return out

    async def _fetch_glossary(self, deadline: _Deadline) -> Dict[str, Any]:
        url = self.base_url + GLOSSARY_PATH
        async with self._visit(deadline, want_login=False) as visit:
            status, final_url, parsed = await visit.fetch(url, ready=_POST_READY, analyze=_analyze_glossary)
        if status is not None and status >= 400:
            raise ForumError(f"glossary page returned HTTP {status}", code="http_error")
        if parsed.error:
            raise ForumError(parsed.error, code="parse_error")
        if not parsed.terms:
            raise ForumError("no glossary terms found; the article layout may have changed", code="parse_error")
        log.info("glossary: %d terms", len(parsed.terms))
        return {"terms": parsed.terms, "count": len(parsed.terms), "source_url": canonical_url(final_url),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def _glossary_hit(self) -> Optional[Dict[str, Any]]:
        if self._glossary is None or time.monotonic() - self._glossary[0] >= self._glossary_ttl:
            return None
        result = self._glossary[1]
        return {**result, "terms": list(result["terms"]), "cached": True}

    def _comment_page_url(self, canonical: str, next_href: Optional[str], page_no: int) -> str:
        """Follow the pagination link when it stays on this post; else build ``?page=N``."""
        if next_href:
            target = urljoin(canonical, next_href)
            parts = urlsplit(target)
            if ((parts.scheme, parts.netloc.lower()) == (self._scheme, self._netloc)
                    and post_ref(target) == post_ref(canonical)):
                return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
        return f"{canonical}?page={page_no}"

    # ------------------------------------------------------------------ browser

    @contextlib.asynccontextmanager
    async def _visit(self, deadline: _Deadline, *, want_login: bool) -> AsyncIterator[_Visit]:
        cookies = await self._brain_cookies()
        async with self._sem:
            context = await self._new_context()
            try:
                await context.route("**/*", _route_filter)
                support_fresh = self._support_session_fresh()
                if support_fresh:
                    await context.add_cookies(self._support_cookies)
                if cookies:
                    await context.add_cookies(cookies)
                visit = _Visit(self, await context.new_page(), deadline, want_login)
                if want_login and not support_fresh:
                    await visit.sign_in()
                yield visit
            finally:
                with contextlib.suppress(Exception):
                    await context.close()

    async def _new_context(self) -> Any:
        for attempt in range(2):
            browser = await self._get_browser()
            try:
                return await browser.new_context(user_agent=USER_AGENT)
            except Exception as exc:
                if attempt or browser.is_connected():
                    raise ForumError(f"could not open a browser context: {_first_line(exc)}",
                                     code="unavailable") from exc
                log.warning("browser disconnected; relaunching")
        raise AssertionError("unreachable")

    async def _get_browser(self) -> Any:
        async with self._browser_lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            if self._browser is not None:
                with contextlib.suppress(Exception):
                    await self._browser.close()
                self._browser = None
            if self._playwright is None:
                try:
                    from playwright.async_api import async_playwright
                except ImportError as exc:
                    raise ForumError("forum tools need Playwright: pip install playwright && "
                                     "playwright install chromium", code="unavailable") from exc
                self._playwright = await async_playwright().start()
            try:
                self._browser = await self._launch(self._playwright.chromium)
            except ForumError:
                with contextlib.suppress(Exception):
                    await self._playwright.stop()
                self._playwright = None
                raise
            return self._browser

    async def _launch(self, chromium: Any) -> Any:
        options: dict[str, Any] = {"headless": True, "chromium_sandbox": self._chromium_sandbox}
        if self._channel and not self._channel_failed:
            try:
                browser = await chromium.launch(channel=self._channel, **options)
                log.info("forum browser: channel %s", self._channel)
                return browser
            except Exception as exc:
                self._channel_failed = True
                log.warning("browser channel %r unavailable (%s); using Playwright's Chromium",
                            self._channel, _first_line(exc))
        try:
            browser = await chromium.launch(**options)
        except Exception as exc:
            raise ForumError(f"could not launch Chromium ({_first_line(exc)}); run `playwright install chromium`",
                             code="unavailable") from exc
        log.info("forum browser: bundled Chromium %s", browser.version)
        return browser

    async def _navigate(self, page: Any, url: str, ready: str, deadline: _Deadline) -> tuple[Optional[int], str, str]:
        if deadline.remaining() < 1.0:
            raise ForumError("time budget exhausted", code="timeout")
        log.debug("GET %s", url)
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=deadline.ms(NAV_TIMEOUT_S))
        except Exception as exc:
            raise ForumError(f"could not load {_redact(url)}: {_first_line(exc)}", code="navigation") from exc
        status = response.status if response is not None else None
        with contextlib.suppress(Exception):
            # Help Center pages are server-rendered; this only covers late client-side rendering,
            # and a miss is fine because the parser decides what the page is.
            await page.wait_for_selector(ready, state="attached", timeout=deadline.ms(READY_TIMEOUT_S))
        return status, page.url, await self._content(page, deadline)

    @staticmethod
    async def _content(page: Any, deadline: _Deadline) -> str:
        for attempt in range(3):
            try:
                return await page.content()
            except Exception as exc:  # the page is still navigating (client-side redirect)
                if attempt == 2:
                    raise ForumError(f"could not read page content: {_first_line(exc)}", code="navigation") from exc
                with contextlib.suppress(Exception):
                    await page.wait_for_load_state("domcontentloaded", timeout=deadline.ms(5.0))
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------ session

    async def _brain_cookies(self) -> list[dict[str, Any]]:
        try:
            raw = await self._cookie_provider()
        except Exception as exc:
            self._cookie_error = _first_line(exc)
            log.warning("cookie provider failed: %s", self._cookie_error)
            return []
        self._cookie_error = None
        cookies = to_playwright_cookies(raw or [])
        if not cookies:
            log.warning("no BRAIN cookies for %s; forum sign-in will likely fail", BRAIN_COOKIE_DOMAIN)
        return cookies

    def _support_session_fresh(self) -> bool:
        return bool(self._support_cookies) and time.monotonic() - self._support_cookies_at < SUPPORT_SESSION_TTL_S

    def _is_help_center_url(self, url: str) -> bool:
        parts = urlsplit(url)
        return (parts.scheme == self._scheme and parts.netloc.lower() == self._netloc
                and parts.path.startswith("/hc") and not _LOGIN_PATH_RE.match(parts.path.lower()))

    def _sso_settled(self, url: str) -> bool:
        """SSO is over once we are back on the Help Center or on the support site's own sign-in page."""
        parts = urlsplit(url)
        on_support = parts.scheme == self._scheme and parts.netloc.lower() == self._netloc
        return self._is_help_center_url(url) or (on_support and bool(_SUPPORT_LOGIN_PATH_RE.match(parts.path.lower())))

    async def _sso(self, page: Any, deadline: _Deadline) -> bool:
        """Zendesk SSO: /access/sso bounces through the BRAIN login (authenticated by the injected
        cookies) back to /hc. Caches the resulting support-site cookies for later contexts."""
        if self._sso_failed_at is not None and time.monotonic() - self._sso_failed_at < SSO_RETRY_AFTER_S:
            log.debug("skipping forum SSO: failed %.0fs ago", time.monotonic() - self._sso_failed_at)
            return False
        log.info("forum SSO via %s/access/sso", self.base_url)
        budget = _Deadline(min(SSO_TIMEOUT_S, deadline.remaining()))
        with contextlib.suppress(Exception):  # redirect chains may abort the first navigation
            await page.goto(f"{self.base_url}/access/sso", wait_until="domcontentloaded",
                            timeout=budget.ms(SSO_TIMEOUT_S))
        ok = False
        try:
            await page.wait_for_url(self._sso_settled, wait_until="domcontentloaded",
                                    timeout=budget.ms(SSO_TIMEOUT_S))
            html = await self._content(page, budget)
            ok = self._is_help_center_url(page.url) and not await asyncio.to_thread(is_login_page, page.url, html)
        except Exception as exc:
            log.debug("forum SSO did not settle: %s", _first_line(exc))
        if not ok:
            self._sso_failed_at = time.monotonic()
            log.warning("forum SSO failed (ended at %s)", _redact(page.url))
            return False
        self._sso_failed_at = None
        with contextlib.suppress(Exception):
            cookies = await page.context.cookies([self.base_url])
            self._support_cookies = [c for c in cookies if str(c.get("domain", "")).lstrip(".") == self._hostname]
            self._support_cookies_at = time.monotonic()
        log.info("forum SSO succeeded")
        return True

    def _check_same_site(self, url: str) -> None:
        parts = urlsplit(url)
        if (parts.scheme, parts.netloc.lower()) != (self._scheme, self._netloc):
            raise ForumError(f"navigation left the support site (ended at {_redact(url)})",
                             code="unexpected_redirect")

    def _auth_error(self, url: str, status: Optional[int], visit: _Visit) -> ForumError:
        where = _redact(url) + (f" (HTTP {status})" if status else "")
        if visit.sso_ok:
            reason = "signed in via SSO, but this account is not authorized to view it"
        elif visit.sso_attempted or self._sso_failed_at is not None:
            reason = "forum sign-in (SSO with the BRAIN session) failed"
        else:
            reason = "not signed in to the support site"
        msg = f"support site returned a sign-in / not-authorized page for {where}: {reason}"
        if self._cookie_error:
            msg += f"; BRAIN cookie provider error: {self._cookie_error}"
        return ForumError(msg, code="auth_required")

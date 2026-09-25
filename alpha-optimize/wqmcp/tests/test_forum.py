"""Tests for forum_functions: pure parsers against hand-written Zendesk-like HTML fixtures, plus
one end-to-end run of ForumClient against a local HTTP server (skipped without Chromium)."""

from __future__ import annotations

import asyncio
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import forum_functions as ff  # noqa: E402
from forum_functions import ForumClient, ForumError  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "forum"
BASE = "https://support.worldquantbrain.com"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def frag(html: str):
    return BeautifulSoup(html, "html.parser").find()


# --------------------------------------------------------------------------- URLs


def test_validate_locale():
    assert ff.validate_locale("zh-cn") == "zh-cn"
    assert ff.validate_locale(" EN-US ") == "en-us"
    assert ff.validate_locale("ja") == "ja"
    for bad in ("", "zh_cn", "zh-cn/../x", "english", "zh-cn?x=1", "z"):
        with pytest.raises(ForumError):
            ff.validate_locale(bad)


def test_build_search_url_encodes_query():
    query = "rank & ts_mean #1 +x 100% 中文/?"
    url = ff.build_search_url(BASE, query, page=3, locale="en-us")
    parts = urlsplit(url)
    assert (parts.scheme, parts.netloc, parts.path) == ("https", "support.worldquantbrain.com", "/hc/en-us/search")
    assert parts.fragment == "" and "#" not in url
    assert parse_qs(parts.query) == {"query": [query], "page": ["3"]}
    assert "page=" not in ff.build_search_url(BASE, "rank")
    with pytest.raises(ForumError):
        ff.build_search_url(BASE, "rank", locale="zh-cn/../../x")


def test_resolve_post_urls_bare_id_tries_post_then_article():
    expected = [f"{BASE}/hc/zh-cn/community/posts/123", f"{BASE}/hc/zh-cn/articles/123"]
    assert ff.resolve_post_urls("123") == expected
    assert ff.resolve_post_urls(" 123-some-slug ") == expected
    assert ff.resolve_post_urls("123", locale="en-us")[1] == f"{BASE}/hc/en-us/articles/123"


def test_resolve_post_urls_explicit_forms():
    assert ff.resolve_post_urls("posts/42") == [f"{BASE}/hc/zh-cn/community/posts/42"]
    assert ff.resolve_post_urls("community/posts/42-x") == [f"{BASE}/hc/zh-cn/community/posts/42"]
    assert ff.resolve_post_urls("articles/7") == [f"{BASE}/hc/zh-cn/articles/7"]
    assert ff.resolve_post_urls(f"{BASE}/hc/en-us/articles/4902349883927-Terms?page=2#comments") == [
        f"{BASE}/hc/en-us/articles/4902349883927-Terms"]
    assert ff.resolve_post_urls("/hc/zh-cn/community/posts/99-title") == [
        f"{BASE}/hc/zh-cn/community/posts/99-title"]


@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "abc",
    "javascript:alert(1)",
    "file:///etc/passwd",
    "http://support.worldquantbrain.com/hc/zh-cn/articles/1",          # wrong scheme
    "https://evil.example.com/hc/zh-cn/articles/1",
    "https://support.worldquantbrain.com.evil.com/hc/zh-cn/articles/1",
    "https://user@support.worldquantbrain.com/hc/zh-cn/articles/1",
    "https://support.worldquantbrain.com:8443/hc/zh-cn/articles/1",
    "//evil.example.com/hc/zh-cn/articles/1",
    "https://127.0.0.1/hc/zh-cn/articles/1",
    "https://support.worldquantbrain.com/access/sso",
    "https://support.worldquantbrain.com/hc/zh-cn/articles/1/../../admin",
    "https://support.worldquantbrain.com/hc/zh-cn/articles/1%2F..%2F",
    "posts/../../admin",
    "123 456",
])
def test_resolve_post_urls_rejects(bad):
    with pytest.raises(ForumError) as err:
        ff.resolve_post_urls(bad)
    assert err.value.code == "invalid_argument"


def test_resolve_post_urls_host_follows_base_url():
    base = "http://127.0.0.1:8123"
    assert ff.resolve_post_urls(f"{base}/hc/zh-cn/community/posts/5", base_url=base) == [
        f"{base}/hc/zh-cn/community/posts/5"]
    with pytest.raises(ForumError):
        ff.resolve_post_urls(f"{BASE}/hc/zh-cn/community/posts/5", base_url=base)


def test_to_playwright_cookies_filters_and_hardens():
    cookies = ff.to_playwright_cookies([
        {"name": "t", "value": "abc", "domain": ".worldquantbrain.com", "path": "/", "secure": False},
        {"name": "u", "value": "1", "domain": "api.worldquantbrain.com", "path": "/", "httpOnly": False,
         "expires": 2_000_000_000},
        {"name": "x", "value": "leak", "domain": ".evilworldquantbrain.com", "path": "/"},
        {"name": "y", "value": "leak", "domain": "example.com", "path": "/"},
        {"value": "no-name", "domain": ".worldquantbrain.com"},
    ])
    assert [c["name"] for c in cookies] == ["t", "u"]
    assert all(c["secure"] is True and c["sameSite"] == "Lax" for c in cookies)
    assert cookies[0]["httpOnly"] is True and cookies[1]["httpOnly"] is False
    assert cookies[1]["expires"] == 2_000_000_000 and "expires" not in cookies[0]


# --------------------------------------------------------------------------- text


def test_element_text_keeps_lines_and_code():
    el = frag(
        "<div><p>Hello <b>world</b></p><p>line1<br>line2</p>"
        "<pre>\ngroup_rank(\n    ts_mean(returns, 20),\n\n    industry\n)</pre>"
        "<p>inline <code>rank(close)</code> end&nbsp;here</p>"
        "<ul><li>a</li><li>b</li></ul><ol start='3'><li>c</li></ol>"
        "<table><tr><th>k</th><th>v</th></tr><tr><td>a</td><td>1</td></tr></table>"
        "<script>bad()</script><!-- note --><p>last</p></div>")
    assert ff.element_text(el) == (
        "Hello world\nline1\nline2\n"
        "```\ngroup_rank(\n    ts_mean(returns, 20),\n\n    industry\n)\n```\n"
        "inline `rank(close)` end here\n- a\n- b\n3. c\nk | v\na | 1\nlast")


def test_element_text_collapses_blank_lines_and_br_in_pre():
    el = frag("<div><p>a</p><br><br><br><br><p>b</p><pre>x = 1<br>y = 2</pre>"
              "<code>multi\nline</code></div>")
    assert ff.element_text(el) == "a\n\nb\n```\nx = 1\ny = 2\n```\n```\nmulti\nline\n```"
    assert ff.element_text(None) == ""


# --------------------------------------------------------------------------- search


def test_parse_search_results():
    results = ff.parse_search_results(fixture("search_page1.html"), BASE)
    assert len(results) == 3
    first = results[0]
    assert first == {
        "title": "How to combine rank & ts_mean",
        "url": f"{BASE}/hc/zh-cn/community/posts/123-rank-ts-mean-template",
        "id": "123",
        "type": "post",
        "snippet": "Use rank(ts_mean(close, 20)) to smooth the signal …",
        "votes": 12,
        "comments": 4,
        "author": "XY98765",
        "date": "2025-02-10T12:34:56Z",
        "breadcrumbs": ["社区", "顾问专属中文论坛"],
    }
    article = results[1]
    assert article["url"] == f"{BASE}/hc/zh-cn/articles/777-Getting-started"
    assert (article["type"], article["id"], article["votes"], article["comments"]) == ("article", "777", 0, 0)
    assert article["author"] == "BRAIN Team" and article["breadcrumbs"] == ["文档"]
    assert results[2]["votes"] == -2


def test_has_next_page():
    assert ff.has_next_page(fixture("search_page1.html"))
    assert not ff.has_next_page(fixture("search_page2.html"))
    assert ff.next_page_href(fixture("post_page1.html")) == (
        "/hc/zh-cn/community/posts/123-rank-ts-mean-template?page=2#comments")
    assert not ff.has_next_page(fixture("post_page2.html"))
    assert ff.has_next_page('<div><a rel="next" href="?page=3">more</a></div>')
    assert not ff.has_next_page('<link rel="next" href="?page=3"><a rel="prev" href="?page=1">x</a>')


# --------------------------------------------------------------------------- post / comments


def test_parse_post_community_post():
    post = ff.parse_post(fixture("post_page1.html"))
    assert post["type"] == "post"
    assert post["title"] == "Rank & ts_mean template"  # not "... 精选" from the status label
    assert post["author"] == "XY98765"
    assert post["date"] == "2025-02-10T12:34:56Z"
    assert post["votes"] == 7  # the post's own vote, not a comment's (99)
    assert post["truncated"] is False
    assert post["body"] == (
        "大家好，分享一个模板：\n"
        "```\ngroup_rank(\n    ts_mean(returns, 20),\n    industry\n)\n```\n"
        "第一行\n第二行 with `rank(close)` inline\n"
        "- step one\n- step two\n"
        "Unauthorized access errors are unrelated.")


def test_parse_post_article_and_missing_body():
    post = ff.parse_post(fixture("article.html"))
    assert (post["type"], post["title"], post["author"], post["date"]) == (
        "article", "Getting started", "BRAIN Team", "2024-06-01T00:00:00Z")
    assert post["votes"] is None
    assert post["body"] == "Operators\nStart with `rank(x)`.\nThen try:\n```\nts_mean(close, 20)\n```"
    assert ff.parse_post(fixture("not_found.html")) is None
    assert ff.parse_post(fixture("signin.html")) is None


def test_parse_post_truncates_long_body():
    html = "<div class='post-body'>" + "".join(f"<p>{'x' * 99}</p>" for _ in range(400)) + "</div>"
    post = ff.parse_post(html)
    assert post["truncated"] is True
    assert len(post["body"]) <= ff.POST_BODY_MAX_CHARS + 20
    assert post["body"].endswith("…[truncated]")


def test_parse_comments_and_dedupe_by_id():
    page1 = ff.parse_comments(fixture("post_page1.html"))
    assert [c["id"] for c in page1] == ["community_comment_1001", "community_comment_1002", "community_comment_1003"]
    assert page1[0] == {"id": "community_comment_1001", "author": "AA11111", "date": "2025-02-11T08:00:00Z",
                        "body": "Thanks!\nWorks for me.", "truncated": False}

    collected: list = []
    seen: set = set()
    # Two identical "+1" replies with different ids are both kept.
    assert ff.merge_comments(collected, page1, seen) == 3
    page2 = ff.parse_comments(fixture("post_page2.html"))
    # 1003 overlaps with page 1 (clamped/shifted page); only 1004 is new.
    assert ff.merge_comments(collected, page2, seen) == 1
    assert [c["id"] for c in collected][-1] == "community_comment_1004"
    assert collected[-1]["body"] == "My variant:\n```\nrank(ts_mean(returns, 5))\n  - rank(ts_mean(returns, 60))\n```"
    # Re-reading the same page adds nothing.
    assert ff.merge_comments(collected, page2, seen) == 0


def test_merge_comments_without_ids_only_dedupes_across_pages():
    same = {"id": None, "author": "A", "date": "d", "body": "+1", "truncated": False}
    collected: list = []
    seen: set = set()
    assert ff.merge_comments(collected, [dict(same), dict(same)], seen) == 2
    assert ff.merge_comments(collected, [dict(same)], seen) == 0


def test_parse_comments_truncates():
    html = f"<ul><li class='comment' id='c1'><div class='comment-body'><p>{'y' * 9000}</p></div></li></ul>"
    (comment,) = ff.parse_comments(html)
    assert comment["truncated"] is True and len(comment["body"]) <= ff.COMMENT_MAX_CHARS + 20


# --------------------------------------------------------------------------- sign-in detection


def test_is_login_page():
    assert ff.is_login_page(f"{BASE}/hc/zh-cn/signin?return_to=x", "<html></html>")
    assert ff.is_login_page(f"{BASE}/access/unauthenticated", "<html></html>")
    assert ff.is_login_page("https://platform.worldquantbrain.com/sign-in?next=/", "<html></html>")
    assert ff.is_login_page(f"{BASE}/hc/zh-cn/community/posts/1", fixture("signin.html"))
    assert ff.is_login_page(f"{BASE}/hc/zh-cn/community/posts/1", fixture("unauthorized.html"))
    # Real content that happens to mention "Unauthorized" is not a login page.
    assert not ff.is_login_page(f"{BASE}/hc/zh-cn/community/posts/1", fixture("post_page1.html"))
    assert not ff.is_login_page(f"{BASE}/hc/zh-cn/search?query=x", fixture("search_page1.html"))
    assert not ff.is_login_page(f"{BASE}/hc/zh-cn/community/posts/1-how-to-login", fixture("not_found.html"))


def test_is_signed_out():
    assert ff.is_signed_out(fixture("glossary.html"))
    assert not ff.is_signed_out(fixture("post_page1.html"))
    assert not ff.is_signed_out(fixture("search_page1.html"))


# --------------------------------------------------------------------------- glossary


def test_parse_glossary_structured():
    terms = {t["term"]: t["definition"] for t in ff.parse_glossary(fixture("glossary.html"))}
    assert list(terms) == ["Alpha", "Decay", "IR", "IS", "Sharpe ratio", "Turnover"]
    # An inline link stays inside the definition instead of starting a new term.
    assert terms["Alpha"] == ("A mathematical model that seeks to predict the future price movement of financial "
                              "instruments. It is written in the Fast Expression language.")
    assert terms["Decay"] == "Linear decay applied to the alpha vector, e.g. `decay_linear(x, 5)`."
    assert terms["IR"] == "Information ratio: mean daily PnL divided by its standard deviation."
    assert terms["IS"] == "In-sample period used for simulation."
    # A capitalised short definition line does not become a term under structured parsing.
    assert terms["Sharpe ratio"].startswith("Sharpe ratio divided by volatility")
    # Word-boundary filter: "ago" inside words or a sentence is kept.
    assert "Data from 5 days ago still counts." in terms["Turnover"]
    assert "Chicago" in terms["Turnover"] and "diagonal" in terms["Turnover"]
    assert "Updated 3 days ago" not in terms["Alpha"]


def test_parse_glossary_line_fallback():
    terms = {t["term"]: t["definition"] for t in ff.parse_glossary(fixture("glossary_lines.html"))}
    assert list(terms) == ["Alpha", "Neutralization", "Pasteurization", "Universe"]
    assert terms["Alpha"] == "A model that predicts price movements, written in the Fast Expression language."
    assert terms["Neutralization"].endswith("such as Industry or Market.")
    assert terms["Pasteurization"] == "Replaces input values outside the universe with NaN."
    assert terms["Universe"] == "The set of instruments an alpha trades, for example TOP3000."


def test_parse_glossary_table():
    html = ("<div class='article-body'><table><tr><th>Term</th><th>Definition</th></tr>"
            "<tr><td>Alpha</td><td>A model.</td></tr><tr><td>Decay</td><td>Smoothing.</td></tr>"
            "<tr><td>Delay</td><td>Data lag<br>in days.</td></tr></table></div>")
    assert ff.parse_glossary(html) == [
        {"term": "Alpha", "definition": "A model."},
        {"term": "Decay", "definition": "Smoothing."},
        {"term": "Delay", "definition": "Data lag\nin days."},
    ]


def test_parse_glossary_without_body_raises():
    with pytest.raises(ForumError):
        ff.parse_glossary(fixture("signin.html"))


@pytest.mark.parametrize("line,meta", [
    ("3 days ago", True),
    ("Updated 2 hours ago", True),
    ("about a month ago", True),
    ("~5 minute read", True),
    ("A - B - C", True),
    ("B", True),
    ("Follow", True),
    ("diagonal", False),
    ("Chicago", False),
    ("hexagon agonist", False),
    ("Data from 5 days ago is used", False),
    ("IR", False),
])
def test_metadata_filter_uses_word_boundaries(line, meta):
    assert ff._is_navigation_or_metadata(line) is meta


# --------------------------------------------------------------------------- client without a browser


async def _no_cookies():
    return []


def test_client_validates_before_launching_browser():
    client = ForumClient(_no_cookies)

    async def scenario():
        for call in (client.read_post("https://evil.example.com/hc/zh-cn/articles/1"),
                     client.read_post("javascript:alert(1)"),
                     client.search_posts("   "),
                     client.search_posts("rank", locale="../x")):
            with pytest.raises(ForumError) as err:
                await call
            assert err.value.code == "invalid_argument"

    asyncio.run(scenario())
    assert client._browser is None and client._playwright is None


def test_client_reports_missing_playwright(monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright.async_api", None)
    client = ForumClient(_no_cookies)
    with pytest.raises(ForumError) as err:
        asyncio.run(client.get_glossary_terms())
    assert err.value.code == "unavailable" and "playwright" in str(err.value).lower()


def test_client_rejects_bad_base_url():
    for bad in ("ftp://x", "https://", "https://host/path", "https://u:p@host"):
        with pytest.raises(ValueError):
            ForumClient(_no_cookies, base_url=bad)


# --------------------------------------------------------------------------- end-to-end (local server)

_HEADER_RE = re.compile(r"<!--HEADER-->.*?<!--/HEADER-->", re.S)
_SIGNED_OUT = ('<a class="sign-in" rel="nofollow" data-auth-action="signin" role="button" '
               'href="/hc/zh-cn/signin?return_to=%2Fhc%2Fzh-cn">登录</a>')


class _FakeSupportSite(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _send(self, status: int, name: str, *, signed_in: bool = True, headers=()):
        html = fixture(name)
        if not signed_in:
            html = _HEADER_RE.sub(_SIGNED_OUT, html)
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str, headers=()):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()

    def do_GET(self):  # noqa: N802
        srv = self.server
        url = urlsplit(self.path)
        path, query = url.path, parse_qs(url.query)
        signed_in = "hc_session=ok" in (self.headers.get("Cookie") or "")
        srv.paths.append(path)
        if path == "/access/sso":
            if srv.sso_ok:
                self._redirect("/hc/zh-cn", headers=[("Set-Cookie", "hc_session=ok; Path=/; HttpOnly")])
            else:
                self._redirect("/hc/zh-cn/signin?return_to=%2Fhc%2Fzh-cn")
        elif path in ("/hc/zh-cn", "/hc/zh-cn/"):
            self._send(200, "hc_home.html", signed_in=signed_in)
        elif path == "/hc/zh-cn/signin":
            self._send(200, "signin.html")
        elif path == "/hc/zh-cn/search":
            srv.queries.append(query.get("query", [""])[0])
            page = query.get("page", ["1"])[0]
            if page in ("1", "2"):
                self._send(200, f"search_page{page}.html", signed_in=signed_in)
            else:
                self._send(404, "not_found.html", signed_in=signed_in)
        elif m := re.match(r"^/hc/zh-cn/community/posts/(\d+)", path):
            if m.group(1) == "123":
                if not signed_in:
                    self._redirect("/hc/zh-cn/signin?return_to=" + quote(self.path, safe=""))
                else:
                    page = query.get("page", ["1"])[0]
                    self._send(200, "post_page2.html" if page == "2" else "post_page1.html")
            elif m.group(1) == "403":
                self._send(403, "unauthorized.html")
            else:
                self._send(404, "not_found.html", signed_in=signed_in)
        elif path.startswith("/hc/zh-cn/articles/777"):
            self._send(200, "article.html", signed_in=signed_in)
        elif path.startswith("/hc/en-us/articles/4902349883927"):
            self._send(200, "glossary.html")
        else:
            self._send(404, "not_found.html", signed_in=signed_in)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _FakeSupportSite)
        self.paths: list[str] = []
        self.queries: list[str] = []
        self.sso_ok = True


@pytest.fixture
def support_site():
    srv = _Server()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


async def _brain_cookies():
    return [{"name": "t", "value": "session", "domain": ".worldquantbrain.com", "path": "/"}]


def _playwright_available() -> bool:
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _playwright_available(), reason="playwright not installed")
def test_forum_client_end_to_end(support_site):
    base = f"http://127.0.0.1:{support_site.server_address[1]}"

    async def scenario() -> str | None:
        # channel="chrome" exercises the fallback to Playwright's bundled Chromium.
        client = ForumClient(_brain_cookies, base_url=base, max_concurrency=2, browser_channel="chrome", timeout=60)
        try:
            try:
                await client._get_browser()
            except ForumError as exc:
                return str(exc)

            query = "rank & ts_mean #1 +x 100%"
            found = await client.search_posts(query, max_results=50)
            assert support_site.queries == [query, query]  # encoded and decoded intact, 2 pages
            assert support_site.paths.count("/access/sso") == 1
            assert found["complete"] is True and found["has_more"] is False and found["signed_in"] is True
            assert found["count"] == 4 and "warning" not in found
            assert [r["id"] for r in found["results"]] == ["123", "777", "456", "789"]
            assert found["results"][0]["url"] == f"{base}/hc/zh-cn/community/posts/123-rank-ts-mean-template"

            capped = await client.search_posts("rank", max_results=2)
            assert capped["count"] == 2 and capped["has_more"] is True
            assert support_site.paths.count("/access/sso") == 1  # support session reused

            post = await client.read_post("123")
            assert post["post"]["title"] == "Rank & ts_mean template" and post["post"]["id"] == "123"
            assert "```\ngroup_rank(\n    ts_mean(returns, 20),\n    industry\n)\n```" in post["post"]["body"]
            assert [c["id"][-4:] for c in post["comments"]] == ["1001", "1002", "1003", "1004"]
            assert post["total_comments"] == 4 and post["complete"] is True
            assert post["has_more_comments"] is False

            short = await client.read_post(f"{base}/hc/zh-cn/community/posts/123-x?page=9", max_comments=2)
            assert short["total_comments"] == 2 and short["has_more_comments"] is True

            support_site.paths.clear()
            article = await client.read_post("777", include_comments=False)
            assert article["post"]["type"] == "article" and article["post"]["title"] == "Getting started"
            assert support_site.paths[-2:] == ["/hc/zh-cn/community/posts/777", "/hc/zh-cn/articles/777"]

            with pytest.raises(ForumError) as err:
                await client.read_post("403")
            assert err.value.code == "auth_required"

            with pytest.raises(ForumError) as err:
                await client.read_post("555")
            assert err.value.code == "not_found"

            first, second = await asyncio.gather(client.get_glossary_terms(), client.get_glossary_terms())
            assert first["count"] == 6 and [t["term"] for t in first["terms"]][:2] == ["Alpha", "Decay"]
            assert {first["cached"], second["cached"]} == {False, True}
            assert support_site.paths.count("/hc/en-us/articles/4902349883927-"
                                            "Click-here-for-a-list-of-terms-and-their-definitions") == 1

            results = await asyncio.gather(*(client.search_posts("rank", max_results=3) for _ in range(3)))
            assert all(r["count"] == 3 for r in results)
        finally:
            await client.aclose()

        # SSO that ends on the sign-in page: search degrades with a warning, a restricted post errors.
        support_site.sso_ok = False
        client = ForumClient(_brain_cookies, base_url=base, browser_channel=None, timeout=60)
        try:
            found = await client.search_posts("rank", max_results=5)
            assert found["signed_in"] is False and "not signed in" in found["warning"]
            assert found["count"] == 4
            with pytest.raises(ForumError) as err:
                await client.read_post("123")
            assert err.value.code == "auth_required" and "sign-in" in str(err.value)
        finally:
            await client.aclose()
        return None

    skip_reason = asyncio.run(scenario())
    if skip_reason:
        pytest.skip(f"Chromium unavailable: {skip_reason}")

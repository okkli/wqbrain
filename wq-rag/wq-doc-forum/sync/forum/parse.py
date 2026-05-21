"""论坛 HTML → 结构化数据。

`parse_post_list`：从列表页提取帖子元信息（不含正文）。
`parse_post_detail`：从详情页提取正文 + 当前页评论。
`parse_comments`：从评论翻页 HTML 中提取增量评论。
`next_page_url`：定位列表分页"下一页"链接。
"""

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup

from ..common.config import settings

_POST_ID_RE = re.compile(r"/posts/(\d+)")


def parse_post_list(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[dict] = []
    for section in soup.select('section[role="region"]'):
        item = section.select_one(".striped-list-item")
        if not item:
            continue
        title_el = item.select_one("a.striped-list-title")
        if not title_el:
            continue

        link = title_el.get("href", "")
        if link and not link.startswith("http"):
            link = settings.support_base + link

        post_id = ""
        m = _POST_ID_RE.search(link)
        if m:
            post_id = m.group(1)

        author = ""
        post_date = ""
        meta_items = item.select("ul.meta-group li.meta-data")
        if meta_items:
            author = meta_items[0].get_text(strip=True)
            if len(meta_items) > 1:
                time_el = meta_items[1].select_one("time")
                if time_el:
                    post_date = time_el.get("datetime", "")

        votes, comments_count = 0, 0
        count_items = item.select(".striped-list-count-item")
        if count_items:
            vm = re.search(r"\d+", count_items[0].get_text())
            if vm:
                votes = int(vm.group())
            if len(count_items) > 1:
                cm = re.search(r"\d+", count_items[1].get_text())
                if cm:
                    comments_count = int(cm.group())

        cat_el = item.select_one(".striped-list-status")
        category = cat_el.get_text(strip=True) if cat_el else ""

        posts.append({
            "post_id": post_id,
            "title": title_el.get_text(strip=True),
            "link": link,
            "author": author,
            "date": post_date,
            "votes": votes,
            "comments_count": comments_count,
            "is_pinned": bool(item.select_one(".status-label-pinned")),
            "category": category,
            "_fetched_at": None,
            "_comments_fetched": 0,
        })
    return posts


def next_page_url(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    nxt = soup.select_one("a.pagination-next-link")
    if not nxt:
        return None
    href = nxt.get("href", "")
    if not href:
        return None
    return href if href.startswith("http") else settings.support_base + href


def _comment_records(soup: BeautifulSoup) -> list[dict]:
    out: list[dict] = []
    for ce in soup.select(".comment"):
        ca = ce.select_one('.comment-author span[title]')
        cb = ce.select_one(".comment-body")
        cd = ce.select_one(".comment-meta time")
        out.append({
            "author": ca["title"] if ca else "?",
            "body": cb.get_text(strip=True) if cb else "",
            "date": cd.get("datetime", "") if cd else "",
        })
    return out


def parse_post_detail(html: str, post_url: str, *, with_comments: bool = False) -> dict:
    """解析正文。默认不带评论；with_comments=True 时附带第一页评论。"""
    soup = BeautifulSoup(html, "html.parser")
    # 标题：优先取 <h1 title="..."> 属性，避免 .status-label 同级 span 被拼进来
    h1 = soup.select_one(".post-title h1, h1.article-title")
    title = ""
    if h1:
        title = h1.get("title") or h1.get_text(strip=True)
    author_el = soup.select_one('.post-author span[title]')
    body_el = soup.select_one(".post-body, .article-body")
    votes_el = soup.select_one(".vote-sum")
    date_el = soup.select_one(".post-meta time")
    result = {
        "title": title,
        "author": author_el["title"] if author_el else "",
        "body": body_el.get_text(strip=True) if body_el else "",
        "votes": votes_el.get_text(strip=True) if votes_el else "0",
        "date": date_el.get("datetime", "") if date_el else "",
        "url": post_url,
    }
    if with_comments:
        result["comments"] = _comment_records(soup)
    return result


def parse_comments(html: str) -> list[dict]:
    return _comment_records(BeautifulSoup(html, "html.parser"))

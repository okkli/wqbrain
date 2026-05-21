"""读 data/forum/posts/{post_id}.json，规范化成 ForumPost + ForumComment。

实际 JSON 字段（来自 wq-doc-forum 爬虫）:
    post: {title, author, body, votes, date, comments[], total_comments}
    comment: {author, body, date}

爬虫产物不包含 post_id（用文件名）、url、author_badges、replies_to 等字段，
本 loader 用文件名当 post_id，comment_id 由 {post_id}_c{idx} 生成。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import config


@dataclass
class ForumComment:
    post_id: str
    comment_id: str
    author: str | None
    body: str
    date: str | None
    index: int                                  # 在该帖下的序号
    source_path: str                            # 用于增量
    extra: dict = field(default_factory=dict)   # parser 阶段填


@dataclass
class ForumPost:
    post_id: str
    title: str
    author: str | None
    body: str
    votes: int
    date: str | None
    total_comments: int
    source_path: str
    extra: dict = field(default_factory=dict)
    comments: list[ForumComment] = field(default_factory=list)


def _load_one(path: Path) -> ForumPost:
    raw = json.loads(path.read_text(encoding="utf-8"))
    post_id = path.stem
    src = str(path)
    comments = [
        ForumComment(
            post_id=post_id,
            comment_id=f"{post_id}_c{i}",
            author=c.get("author"),
            body=c.get("body", "") or "",
            date=c.get("date"),
            index=i,
            source_path=src,
        )
        for i, c in enumerate(raw.get("comments", []) or [])
    ]
    return ForumPost(
        post_id=post_id,
        title=raw.get("title", "") or "",
        author=raw.get("author"),
        body=raw.get("body", "") or "",
        votes=int(raw.get("votes", 0) or 0),
        date=raw.get("date"),
        total_comments=int(raw.get("total_comments", len(comments)) or 0),
        source_path=src,
        comments=comments,
    )


def load_forum(root: Path | None = None) -> Iterator[ForumPost]:
    """流式 yield 论坛帖子。"""
    d = Path(root or config.FORUM_POSTS_DIR)
    if not d.exists():
        raise FileNotFoundError(f"forum posts dir not found: {d}")
    for p in sorted(d.glob("*.json")):
        yield _load_one(p)

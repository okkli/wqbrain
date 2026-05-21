"""统一 chunker。

设计原则（与原指南一致）:
1. 代码块绝不切碎。长 chunk 切前先用占位符隔离 ``` 代码，切完再还原。
2. 短内容整条索引。短评论 (<SHORT_COMMENT) / 短帖 (<SHORT_POST) 一个 chunk。
3. 长内容按结构标记切（"类别:" / "Idea:" / "表达式:" / "Selection Expression:" 等），段加[上下文]前缀。
4. 含 SuperAlpha 表达式的代码块（≥2 个已知操作符）额外独立索引为 code_block chunk，便于精确召回。
5. 回复型短评论 (<300字 且有 replies_to) 拼接被回复者正文做上下文。
6. tutorial section 等同 post 处理（长则切、短则整）；含 SIMULATION_EXAMPLE 自动产出代码 chunk。

不依赖 BGE-M3 sparse —— sparse 是检索时的能力，与切分独立。
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterable

import config
from sources import ForumPost, ForumComment, TutorialSection


# ─── 切分基础 ──────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text: str                              # 实际入嵌入的文本（含上下文头/代码围栏）
    chunk_type: str                        # post | post_segment | comment | comment_segment | code_block | tutorial_section
    parent_id: str                         # post_id / comment_id / tutorial_id
    chunk_index: int                       # 在 parent 内的序号
    metadata: dict = field(default_factory=dict)


STRUCTURE_MARKERS = [
    r"类别\s*[:：]", r"[Ii]dea\s*[:：]", r"表达式\s*[:：]",
    r"[Ss]election\s*[Ee]xpression\s*[:：]", r"[Cc]ombo\s*[Ee]xpression\s*[:：]",
    r"[Ss]etting[s]?\s*[:：]", r"结果\s*[:：]", r"解释\s*[:：]", r"结论\s*[:：]",
    r"[Hh]ypothesis\s*[:：]?", r"[Ii]mplementation\s*[:：]?",
    r"第[一二三四五六七八九十\d]+步", r"^\s*\d+[\.、)]\s",
    r"^#{1,6}\s",   # markdown heading
]
STRUCTURE_PATTERN = re.compile("|".join(STRUCTURE_MARKERS), re.MULTILINE)


def is_selection_expression(code: str) -> bool:
    """≥2 个已知 SA 算子/字段 → 判定为表达式型代码。"""
    low = code.lower()
    return sum(1 for op in config.SA_OPERATORS if re.search(rf"\b{re.escape(op.lower())}\b", low)) >= 2


def split_long_code(code: str, max_chars: int) -> list[str]:
    """超长代码按行切。保证每段 <= max_chars 且换行边界。"""
    if len(code) <= max_chars:
        return [code]
    lines = code.split("\n")
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for ln in lines:
        # +1 for newline
        if cur and cur_len + len(ln) + 1 > max_chars:
            out.append("\n".join(cur))
            cur = [ln]
            cur_len = len(ln)
        else:
            cur.append(ln)
            cur_len += len(ln) + 1
        # 单行就超长，硬切
        if len(ln) > max_chars:
            big = cur.pop()
            if cur:
                out.append("\n".join(cur))
                cur, cur_len = [], 0
            for k in range(0, len(big), max_chars):
                out.append(big[k:k + max_chars])
    if cur:
        out.append("\n".join(cur))
    return out


_FENCE_RE = re.compile(r"```[ \t]*[\w-]*\n?(.*?)```", re.DOTALL)


def extract_and_isolate_code(md_text: str) -> tuple[str, list[str]]:
    """提取 ``` 代码块，正文用 [CODE_BLOCK_N] 占位符替换。返回 (cleaned, code_list)。"""
    blocks: list[str] = []

    def repl(m: re.Match) -> str:
        code = m.group(1).strip("\n").strip()
        if len(code) > 20:
            blocks.append(code)
            return f"\n[CODE_BLOCK_{len(blocks) - 1}]\n"
        return m.group(0)

    cleaned = _FENCE_RE.sub(repl, md_text)
    return cleaned, blocks


def restore_code(text: str, blocks: list[str], max_inline_chars: int | None = None) -> str:
    """还原代码占位符。超长代码块（> max_inline_chars）只内联前 N 字符 + 截断标记。

    完整代码在独立 code_block chunk 里被分片索引，所以正文段内联截断不会丢信息。
    """
    cap = max_inline_chars if max_inline_chars is not None else config.MAX_CODE_CHUNK_CHARS
    for i, c in enumerate(blocks):
        if len(c) > cap:
            preview = c[: cap // 2].rstrip()
            replacement = (
                f"```\n{preview}\n... [代码过长，已截断；完整内容见 code_block chunk]\n```"
            )
        else:
            replacement = f"```\n{c}\n```"
        text = text.replace(f"[CODE_BLOCK_{i}]", replacement)
    return text


def _split_long(text: str, max_chars: int) -> list[str]:
    """超长段：先按双换行，再按中文/英文句末，最后硬切。"""
    parts = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) < max_chars:
            cur = (cur + "\n\n" + p).strip()
            continue
        if cur:
            chunks.append(cur)
            cur = ""
        if len(p) > max_chars:
            sentences = re.split(r"(?<=[。！？!?\.])\s*", p)
            for s in sentences:
                if not s:
                    continue
                if len(cur) + len(s) < max_chars:
                    cur = (cur + s).strip()
                else:
                    if cur:
                        chunks.append(cur)
                    if len(s) > max_chars:
                        # 硬切（极端情况）
                        for k in range(0, len(s), max_chars):
                            chunks.append(s[k:k + max_chars])
                        cur = ""
                    else:
                        cur = s
        else:
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_by_structure(text: str, max_chars: int) -> list[str]:
    """按结构标记切；<200 的段并入前一段；>max_chars 的段递归切。"""
    if not text:
        return []
    positions = [m.start() for m in STRUCTURE_PATTERN.finditer(text)]
    if not positions or positions[0] > 0:
        positions = [0] + positions
    positions.append(len(text))
    segs = [
        text[positions[i]:positions[i + 1]].strip()
        for i in range(len(positions) - 1)
        if text[positions[i]:positions[i + 1]].strip()
    ]
    final: list[str] = []
    buf = ""
    for seg in segs:
        if len(buf) + len(seg) < 200:
            buf = (buf + "\n\n" + seg).strip()
        else:
            if buf:
                final.append(buf)
            if len(seg) > max_chars:
                final.extend(_split_long(seg, max_chars))
                buf = ""
            else:
                buf = seg
    if buf:
        final.append(buf)
    return final


# ─── chunk_post / chunk_comment / chunk_tutorial ───────────────────────────
def _post_meta_base(p: ForumPost) -> dict:
    ex = p.extra or {}
    return {
        "source": "forum_post",
        "post_id": p.post_id,
        "comment_id": None,
        "tutorial_id": None,
        "author_id": p.author,
        "title": p.title,
        "vote_count": p.votes,
        "total_comments": p.total_comments,
        "created_at": p.date,
        "created_at_ts": ex.get("created_at_ts"),
        "recency_tag": ex.get("recency_tag"),
        "year": ex.get("year"),
        "age_days": ex.get("age_days"),
        "sa_type": ex.get("sa_type"),
        "has_code": ex.get("has_code", False),
        "operators_mentioned": ex.get("operators_mentioned", []),
        "regions": ex.get("regions", []),
        "reported_sharpe": ex.get("reported_sharpe"),
        "tags": ex.get("tags", []),
        "source_path": p.source_path,
    }


def _comment_meta_base(c: ForumComment, thread_title: str | None, thread_tags: list[str] | None) -> dict:
    ex = c.extra or {}
    # 评论继承所在帖子的内容标签 + 自身衍生 tag
    inherited = list(thread_tags or [])
    own = list(ex.get("tags") or [])
    tags = list(dict.fromkeys(inherited + own))
    return {
        "source": "forum_comment",
        "post_id": c.post_id,
        "comment_id": c.comment_id,
        "tutorial_id": None,
        "author_id": c.author,
        "title": thread_title,
        "vote_count": 0,
        "created_at": c.date,
        "created_at_ts": ex.get("created_at_ts"),
        "recency_tag": ex.get("recency_tag"),
        "year": ex.get("year"),
        "age_days": ex.get("age_days"),
        "sa_type": ex.get("sa_type"),
        "has_code": ex.get("has_code", False),
        "operators_mentioned": ex.get("operators_mentioned", []),
        "regions": ex.get("regions", []),
        "reported_sharpe": ex.get("reported_sharpe"),
        "replies_to": ex.get("replies_to"),
        "tags": tags,
        "source_path": c.source_path,
    }


def _tutorial_meta_base(s: TutorialSection) -> dict:
    ex = s.extra or {}
    return {
        "source": "tutorial",
        "post_id": None,
        "comment_id": None,
        "tutorial_id": s.tutorial_id,
        "author_id": None,
        "title": s.title,
        "section_heading": s.section_heading,
        "category": s.category,
        "has_code": ex.get("has_code", False),
        "has_simulation": s.has_simulation,
        "operators_mentioned": ex.get("operators_mentioned", []),
        "tags": ex.get("tags", []),
        "source_path": s.source_path,
    }


def chunk_post(p: ForumPost) -> list[Chunk]:
    body = p.body or ""
    meta = _post_meta_base(p)
    chunks: list[Chunk] = []

    if len(body) < config.SHORT_POST_CHARS:
        if body:
            chunks.append(Chunk(body, "post", p.post_id, 0, {**meta}))
    else:
        no_code, blocks = extract_and_isolate_code(body)
        segs = chunk_by_structure(no_code, config.LONG_POST_SEG_MAX_CHARS)
        for i, s in enumerate(segs):
            restored = restore_code(s, blocks)
            # 不加 [来自原帖: ...] 前缀 —— title/author 由 indexer 的 _build_embedding_text 统一注入元数据头
            chunks.append(Chunk(restored, "post_segment", p.post_id, i, {**meta, "segment_index": i}))

    # 独立 code chunk（含 SA 表达式）
    cb_idx = 0
    for code in p.extra.get("code_blocks", []):
        if not is_selection_expression(code):
            continue
        pieces = split_long_code(code, config.MAX_CODE_CHUNK_CHARS)
        n = len(pieces)
        for k, piece in enumerate(pieces):
            tag = f" (part {k+1}/{n})" if n > 1 else ""
            # 仅保留代码本体 + 上下文片段；元数据头由 indexer 统一注入
            t = f"```\n{piece}\n```\n\n来源上下文{tag}:\n{body[:300]}"
            chunks.append(Chunk(
                t, "code_block", p.post_id, 1000 + cb_idx,
                {**meta, "is_code_only": True, "code_part": k},
            ))
            cb_idx += 1
    return chunks


def chunk_comment(c: ForumComment, thread_title: str | None, thread_tags: list[str] | None = None) -> list[Chunk]:
    body = c.body or ""
    meta = _comment_meta_base(c, thread_title, thread_tags)
    chunks: list[Chunk] = []

    if not body:
        return chunks
    if len(body) < config.SHORT_COMMENT_CHARS:
        chunks.append(Chunk(body, "comment", c.comment_id, 0, {**meta}))
    else:
        no_code, blocks = extract_and_isolate_code(body)
        segs = chunk_by_structure(no_code, config.LONG_SEG_MAX_CHARS)
        for i, s in enumerate(segs):
            restored = restore_code(s, blocks)
            chunks.append(Chunk(restored, "comment_segment", c.comment_id, i,
                                {**meta, "segment_index": i}))

    cb_idx = 0
    for code in c.extra.get("code_blocks", []):
        if not is_selection_expression(code):
            continue
        pieces = split_long_code(code, config.MAX_CODE_CHUNK_CHARS)
        n = len(pieces)
        for k, piece in enumerate(pieces):
            tag = f" (part {k+1}/{n})" if n > 1 else ""
            # 元数据头由 indexer 统一注入
            t = f"```\n{piece}\n```\n\n来源上下文{tag}:\n{body[:300]}"
            chunks.append(Chunk(
                t, "code_block", c.comment_id, 1000 + cb_idx,
                {**meta, "is_code_only": True, "code_part": k},
            ))
            cb_idx += 1
    return chunks


def chunk_tutorial_section(s: TutorialSection) -> list[Chunk]:
    body = s.body_md or ""
    meta = _tutorial_meta_base(s)
    chunks: list[Chunk] = []

    if not body:
        return chunks
    if len(body) < config.SHORT_POST_CHARS:
        chunks.append(Chunk(body, "tutorial_section", f"{s.tutorial_id}#{s.section_index}", 0, {**meta}))
    else:
        no_code, blocks = extract_and_isolate_code(body)
        segs = chunk_by_structure(no_code, config.LONG_POST_SEG_MAX_CHARS)
        for i, seg in enumerate(segs):
            restored = restore_code(seg, blocks)
            # title/小节 由 indexer 注入；body 内 ## heading 已是结构信号
            chunks.append(Chunk(restored, "tutorial_section", f"{s.tutorial_id}#{s.section_index}",
                                i, {**meta, "segment_index": i}))

    # tutorial 内的 SIMULATION_EXAMPLE 已经在 body 里以代码块出现；
    # 提取后独立索引一份纯代码 chunk（便于代码精确召回）
    cb_idx = 0
    for code in s.extra.get("code_blocks", []):
        if not is_selection_expression(code):
            continue
        pieces = split_long_code(code, config.MAX_CODE_CHUNK_CHARS)
        n = len(pieces)
        for k, piece in enumerate(pieces):
            tag = f" (part {k+1}/{n})" if n > 1 else ""
            t = f"```\n{piece}\n```" + (f"\n\n{tag.strip()}" if tag.strip() else "")
            chunks.append(Chunk(
                t, "code_block", f"{s.tutorial_id}#{s.section_index}", 1000 + cb_idx,
                {**meta, "is_code_only": True, "code_part": k},
            ))
            cb_idx += 1
    return chunks


# ─── 回复链拼接 ────────────────────────────────────────────────────────────
def merge_reply_chains(comments: list[ForumComment]) -> list[ForumComment]:
    """短回复（<REPLY_CTX_SHORT_CHARS 且有 replies_to）拼接被回复者正文作上下文。"""
    by_author = {}
    for c in comments:
        by_author.setdefault(c.author, c)  # 取该作者的第一条作为代表
    out: list[ForumComment] = []
    for c in comments:
        rt = (c.extra or {}).get("replies_to")
        if rt and len(c.body or "") < config.REPLY_CTX_SHORT_CHARS:
            target = by_author.get(rt)
            if target and target is not c:
                c.body = (
                    f"[上下文 - 回复 {target.author}]\n"
                    f"> {target.body[:200]}...\n\n"
                    f"[{c.author} 的回复]\n{c.body}"
                )
        out.append(c)
    return out


# ─── 顶层 collect ──────────────────────────────────────────────────────────
def collect_all(posts: Iterable[ForumPost], tutorials: Iterable[TutorialSection]) -> list[Chunk]:
    """全量切分。调用方负责传入已 enrich 过的 posts/tutorials。"""
    chunks: list[Chunk] = []
    for p in posts:
        chunks.extend(chunk_post(p))
        thread_tags = (p.extra or {}).get("tags", [])
        for c in merge_reply_chains(p.comments):
            chunks.extend(chunk_comment(c, p.title, thread_tags))
    for s in tutorials:
        chunks.extend(chunk_tutorial_section(s))
    return chunks

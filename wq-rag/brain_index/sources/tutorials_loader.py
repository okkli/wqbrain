"""读 data/kb/tutorials/{page_id}.json，把结构化 content[] 块还原成可切分单元。

实际 JSON 字段（爬虫产物）:
    {id, title, category, content[], sequence, lastModified}
    content block: {type, value}

block.type 分布 (实测全量统计):
    TEXT(568): value 是 HTML 字符串 → 用 stdlib 抽纯文本
    HEADING(323): value = {level: "2", content: "..."}
    IMAGE(134): value = {title, url, ...} → 渲染 ![title](url)
    TABLE(28): value = {data: [[...rows]], firstRow/Col header} → markdown 表
    SIMULATION_EXAMPLE(23): value = {settings, type, regular} → 代码块 + settings 上下文
    EQUATION(16): value = LaTeX string → 用 $$...$$ 包

切分策略：按 HEADING 起新 section，每个 section 一段 markdown body_md。
SIMULATION_EXAMPLE 在 chunker 里会被识别为"含表达式代码块"独立索引。
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator

import config


# ─── HTML → 纯文本（stdlib，避免引入 bs4）─────────────────────────────────
class _HTMLToText(HTMLParser):
    _BLOCK_TAGS = {"p", "div", "li", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._in_pre = False
        self._link_href: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "pre" or tag == "code":
            self._in_pre = True
            if tag == "pre":
                self._chunks.append("\n```\n")
        elif tag == "a":
            self._link_href = dict(attrs).get("href")
        elif tag == "li":
            self._chunks.append("\n- ")
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in ("pre", "code"):
            self._in_pre = False
            if tag == "pre":
                self._chunks.append("\n```\n")
        elif tag == "a" and self._link_href:
            self._chunks.append(f" ({self._link_href})")
            self._link_href = None
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data):
        self._chunks.append(data)

    def text(self) -> str:
        s = "".join(self._chunks)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n[ \t]+", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()


def _html_to_text(html: str) -> str:
    p = _HTMLToText()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return html
    return p.text()


# ─── 块渲染 ────────────────────────────────────────────────────────────────
def _render_heading(v) -> tuple[str, str]:
    """返回 (md, plain_heading_text) 用于 section 切分。"""
    if isinstance(v, dict):
        txt = (v.get("content") or v.get("text") or v.get("value") or "").strip()
        lvl_raw = v.get("level", 2)
        try:
            lvl = int(lvl_raw)
        except (TypeError, ValueError):
            lvl = 2
    else:
        txt, lvl = str(v or "").strip(), 2
    lvl = max(1, min(6, lvl))
    return ("#" * lvl + " " + txt) if txt else "", txt


def _render_text(v) -> str:
    if isinstance(v, str):
        return _html_to_text(v)
    if isinstance(v, dict):
        return _html_to_text(v.get("text") or v.get("content") or v.get("value") or "")
    return str(v or "")


def _render_image(v) -> str:
    if isinstance(v, dict):
        alt = v.get("title") or v.get("alt") or "image"
        url = v.get("url") or v.get("src") or ""
        return f"![{alt}]({url})" if url else f"[image: {alt}]"
    return "[image]"


def _render_table(v) -> str:
    if not isinstance(v, dict):
        return ""
    rows = v.get("data") or []
    if not rows:
        return ""
    out = []
    header = rows[0]
    out.append("| " + " | ".join(str(c) for c in header) + " |")
    out.append("|" + "|".join("---" for _ in header) + "|")
    for r in rows[1:]:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _render_simulation(v) -> str:
    """SIMULATION_EXAMPLE 是核心资产。settings + regular 表达式都要保留。"""
    if not isinstance(v, dict):
        return f"```\n{v}\n```"
    settings = v.get("settings") or {}
    sa_type = v.get("type", "REGULAR")
    expr = v.get("regular") or v.get("combo") or v.get("selection") or ""
    parts = [f"**SIMULATION_EXAMPLE** ({sa_type})"]
    if settings:
        s_lines = "\n".join(f"- {k}: {v_}" for k, v_ in settings.items())
        parts.append("Settings:\n" + s_lines)
    if expr:
        parts.append(f"Expression:\n```\n{expr}\n```")
    return "\n\n".join(parts)


def _render_equation(v) -> str:
    s = v if isinstance(v, str) else (v.get("text", "") if isinstance(v, dict) else "")
    return f"$$\n{s}\n$$"


def _render_block(block: dict) -> str:
    t = (block.get("type") or "").upper()
    v = block.get("value")
    if t == "HEADING":
        return _render_heading(v)[0]
    if t in ("TEXT", "PARAGRAPH"):
        return _render_text(v)
    if t in ("IMAGE", "FIGURE"):
        return _render_image(v)
    if t == "TABLE":
        return _render_table(v)
    if t == "SIMULATION_EXAMPLE":
        return _render_simulation(v)
    if t == "EQUATION":
        return _render_equation(v)
    # 未知类型：保底尽力文本化
    if isinstance(v, str):
        return _html_to_text(v) if "<" in v else v
    if isinstance(v, dict):
        return _html_to_text(v.get("text") or v.get("content") or json.dumps(v, ensure_ascii=False))
    return str(v or "")


# ─── 切分到 section ────────────────────────────────────────────────────────
@dataclass
class TutorialSection:
    tutorial_id: str
    title: str
    category: str
    section_index: int          # 0 = 简介段（无 HEADING）
    section_heading: str
    body_md: str
    source_path: str
    has_simulation: bool = False  # 含 SIMULATION_EXAMPLE 块
    extra: dict = field(default_factory=dict)


def _load_one(path: Path) -> list[TutorialSection]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tid = raw.get("id") or path.stem
    title = raw.get("title", "") or ""
    category = raw.get("category", "") or ""
    src = str(path)
    blocks = raw.get("content") or []

    sections: list[TutorialSection] = []
    cur_heading = ""
    cur_blocks: list[dict] = []

    def flush(idx: int):
        rendered_parts = [m for m in (_render_block(b) for b in cur_blocks) if m]
        body = "\n\n".join(rendered_parts).strip()
        body = re.sub(r"\n{3,}", "\n\n", body)
        if not body and not cur_heading:
            return
        head_md = f"## {cur_heading}\n\n" if cur_heading else ""
        has_sim = any((b.get("type") or "").upper() == "SIMULATION_EXAMPLE" for b in cur_blocks)
        sections.append(TutorialSection(
            tutorial_id=tid, title=title, category=category,
            section_index=idx, section_heading=cur_heading,
            body_md=(head_md + body).strip(),
            source_path=src, has_simulation=has_sim,
        ))

    for b in blocks:
        if (b.get("type") or "").upper() == "HEADING":
            flush(len(sections))
            cur_heading = _render_heading(b.get("value"))[1]
            cur_blocks = []
        else:
            cur_blocks.append(b)
    flush(len(sections))

    return sections


def load_tutorials(root: Path | None = None) -> Iterator[TutorialSection]:
    d = Path(root or config.KB_TUTORIALS_DIR)
    if not d.exists():
        raise FileNotFoundError(f"tutorials dir not found: {d}")
    for p in sorted(d.glob("*.json")):
        for s in _load_one(p):
            yield s

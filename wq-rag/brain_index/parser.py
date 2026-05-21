"""把 loader 产出的原始文本规整 + 抽元数据。

针对 wq-doc-forum 爬虫产物的真实状况：
1. body 已经是文本 + Markdown 围栏，**不需要 HTML→MD**。
2. 但围栏与中文常紧贴在一行 (e.g. "...影响效率。```@mcp.tool()...")，要补换行。
3. 论坛 JSON 没有 author_badges / url / replies_to，元数据只能从正文抽。
4. 时效性 + 内容标签：根据日期算 recency_tag/year/created_at_ts；按标题关键词推内容 tags。
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Any

import config


# ─── 围栏规整 ──────────────────────────────────────────────────────────────
_LANG_TAG_RE = re.compile(r"[\w-]{1,20}")


def normalize_fences(text: str) -> str:
    """让所有 ``` 围栏前后都有换行，保留首个开 fence 的语言标签。

    用 toggle（in/out fence）状态机：每次遇到 ```，根据当前状态决定加换行格式。
    - 开 fence: 前补 \\n，后吃掉一个可选语言标签（[\\w-]{1,20}），再补 \\n
    - 关 fence: 前后都补 \\n
    fence 内部内容不动。
    """
    if not text or "```" not in text:
        return text
    out: list[str] = []
    i, n, in_fence = 0, len(text), False
    while i < n:
        idx = text.find("```", i)
        if idx < 0:
            out.append(text[i:])
            break
        # 输出 fence 之前的内容
        out.append(text[i:idx])
        if not in_fence:
            # 开 fence
            if out and out[-1] and not out[-1].endswith("\n"):
                out.append("\n")
            out.append("```")
            j = idx + 3
            # 抓语言标签（紧跟在 ``` 后、字母数字短横）
            m = _LANG_TAG_RE.match(text, j)
            if m:
                out.append(m.group(0))
                j = m.end()
            out.append("\n")
            in_fence = True
            i = j
        else:
            # 关 fence
            if out and out[-1] and not out[-1].endswith("\n"):
                out.append("\n")
            out.append("```\n")
            in_fence = False
            i = idx + 3
    s = "".join(out)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


_FENCE_RE = re.compile(r"```[ \t]*[\w-]*\n?(.*?)```", re.DOTALL)


def extract_code_blocks(text: str, min_len: int = 20) -> list[str]:
    """抽出所有 ``` 围栏代码块，去除过短的（噪声）。"""
    out = []
    for m in _FENCE_RE.finditer(text):
        code = m.group(1).strip("\n").strip()
        if len(code) >= min_len:
            out.append(code)
    return out


# ─── SuperAlpha 元数据抽取 ─────────────────────────────────────────────────
def detect_sa_type(text: str) -> str | None:
    """从正文判断 Selection / Combo / Regular Alpha。"""
    if not text:
        return None
    m = re.search(r"类别\s*[:：]\s*([A-Za-z]+)", text)
    if m:
        t = m.group(1).lower()
        if "select" in t:
            return "selection"
        if "combo" in t:
            return "combo"
        if "regular" in t or "alpha" in t:
            return "regular"
    low = text.lower()
    if re.search(r"\bselection[\s_]?expression\s*[:：]", low):
        return "selection"
    if re.search(r"\bcombo[\s_]?expression\s*[:：]", low):
        return "combo"
    return None


def extract_operators(text: str) -> list[str]:
    """命中已知 SuperAlpha 算子/字段（大小写不敏感）。"""
    if not text:
        return []
    low = text.lower()
    seen, out = set(), []
    for op in config.SA_OPERATORS:
        # 用词边界匹配，避免 'ts' 误命中 'tests'
        if re.search(rf"\b{re.escape(op.lower())}\b", low):
            if op not in seen:
                seen.add(op)
                out.append(op)
    return out


def extract_regions(text: str) -> list[str]:
    if not text:
        return []
    return [r for r in config.SA_REGIONS if re.search(rf"\b{r}\b", text)]


_SHARPE_RE = re.compile(r"[Ss]harpe[^\d\-]{0,8}([\-\d.]+)")


def extract_sharpe(text: str) -> float | None:
    if not text:
        return None
    m = _SHARPE_RE.search(text)
    if not m:
        return None
    try:
        v = float(m.group(1).rstrip("."))
    except ValueError:
        return None
    if -10 < v < 50:  # 合理范围保护
        return v
    return None


_REPLY_RE = re.compile(r"(?:^|\s)(?:回复\s*)?@?([A-Z]{2}\d{4,6})\b")


def detect_replies_to(text: str) -> str | None:
    """从评论正文头部推断被回复的作者 ID。"""
    if not text:
        return None
    head = text.strip()[:60]
    m = _REPLY_RE.search(head)
    return m.group(1) if m else None


# ─── 时效性 + 标签 ─────────────────────────────────────────────────────────
_NOW_TS = int(datetime.now(timezone.utc).timestamp())


def parse_date_ts(date_str: str | None) -> int | None:
    """ISO 8601 字符串 → epoch seconds。失败返 None。"""
    if not date_str:
        return None
    s = date_str.strip()
    # 把 'Z' 替换成 +00:00 让 fromisoformat 接受
    s = re.sub(r"Z$", "+00:00", s)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        # 兜底：截取前 19 字符按通用格式
        try:
            dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            return None


def derive_recency(ts: int | None) -> tuple[str | None, str | None, int | None]:
    """返回 (recency_tag, year, age_days)。"""
    if ts is None:
        return None, None, None
    age = max(0, _NOW_TS - ts)
    bucket = next((name for name, lim in config.RECENCY_BUCKETS if age <= lim), "archived")
    year = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y")
    return bucket, year, age // 86400


_TAG_REGEXES = [(tag, re.compile(pat, re.IGNORECASE)) for tag, pat in config.TAG_RULES]


def derive_content_tags(title: str) -> list[str]:
    """**仅在 title 上**按 config.TAG_RULES 规则匹配。

    论坛标题大多带【XXX】明确分类，body 内容里很多关键词（"dataset"/"如何"/"工具"）会泛命中，
    所以 tag 只看 title 才有信号意义。body 关键词应通过 operators_mentioned/regions/sa_type 等
    具体字段反映，而不是 tag。
    """
    if not title:
        return []
    return [tag for tag, rx in _TAG_REGEXES if rx.search(title)]


def derive_engagement_tag(total_comments: int) -> str | None:
    if total_comments >= config.TOP_ENGAGEMENT_COMMENTS:
        return "top_engagement"
    if total_comments >= config.HIGH_ENGAGEMENT_COMMENTS:
        return "high_engagement"
    return None


# ─── 顶层 enrich 函数（loader 产物 in-place 改）────────────────────────────
def enrich_post(post: Any) -> None:
    """规整 body 围栏，抽元数据塞进 post.extra。"""
    post.body = normalize_fences(post.body)
    ts = parse_date_ts(post.date)
    recency_tag, year, age_days = derive_recency(ts)
    content_tags = derive_content_tags(post.title)
    eng = derive_engagement_tag(post.total_comments)
    if eng:
        content_tags.append(eng)
    if "```" in post.body:
        content_tags.append("has_code")
    content_tags = list(dict.fromkeys(content_tags))  # 去重保序

    post.extra.update({
        "code_blocks": extract_code_blocks(post.body),
        "operators_mentioned": extract_operators(post.body + " " + post.title),
        "regions": extract_regions(post.body),
        "sa_type": detect_sa_type(post.body + "\n" + post.title),
        "reported_sharpe": extract_sharpe(post.body),
        "has_code": "```" in post.body,
        "created_at_ts": ts,
        "recency_tag": recency_tag,
        "year": year,
        "age_days": age_days,
        "tags": content_tags,
    })


def enrich_comment(comment: Any) -> None:
    comment.body = normalize_fences(comment.body)
    ts = parse_date_ts(comment.date)
    recency_tag, year, age_days = derive_recency(ts)
    # comment 不做 title-based 标签；继承父帖在 chunker 里处理
    tags: list[str] = []
    if "```" in comment.body:
        tags.append("has_code")
    comment.extra.update({
        "code_blocks": extract_code_blocks(comment.body),
        "operators_mentioned": extract_operators(comment.body),
        "regions": extract_regions(comment.body),
        "sa_type": detect_sa_type(comment.body),
        "reported_sharpe": extract_sharpe(comment.body),
        "replies_to": detect_replies_to(comment.body),
        "has_code": "```" in comment.body,
        "created_at_ts": ts,
        "recency_tag": recency_tag,
        "year": year,
        "age_days": age_days,
        "tags": list(dict.fromkeys(tags)),
    })


def enrich_tutorial(section: Any) -> None:
    section.body_md = normalize_fences(section.body_md)
    tags = ["tutorial"]
    if getattr(section, "has_simulation", False):
        tags.append("has_simulation_example")
    if "```" in section.body_md:
        tags.append("has_code")
    section.extra.update({
        "code_blocks": extract_code_blocks(section.body_md),
        "operators_mentioned": extract_operators(section.body_md),
        "has_code": "```" in section.body_md,
        "tags": tags,
    })

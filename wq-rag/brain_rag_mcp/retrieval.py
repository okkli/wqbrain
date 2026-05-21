"""核心检索 + 聚合 + LLM 格式化。

负责：
1. 调 BGE-M3 拿 dense+sparse，走 Qdrant Fusion RRF
2. 同 post_id / tutorial_id 多 segment 聚合为单 hit
3. 字段裁剪（去掉 source_path / source_hash / segment_index / chunk_index 等 LLM 不需要的）
4. 时间人类可读化
5. content 截断到 800 字符 + truncated 标记 + fetch hint
6. ref 解析 + 按 ref 反查全文
"""
from __future__ import annotations
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 复用 brain_index 的 config / embedder
_BRAIN_INDEX = (Path(__file__).resolve().parent.parent / "brain_index").resolve()
if str(_BRAIN_INDEX) not in sys.path:
    sys.path.insert(0, str(_BRAIN_INDEX))

import config  # noqa: E402  brain_index/config.py
from embedder import get_embedder  # noqa: E402

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ─── 常量 ──────────────────────────────────────────────────────────────────
DEFAULT_CONTENT_TRUNCATE = 800
DEFAULT_TOP_K = 8
MAX_TOP_K = 50
PREFETCH_MULTIPLIER = 2   # 内部召回 top_k * 2 再融合
MAX_FETCH_CHUNKS = 200    # rag_fetch 单次 scroll 上限（防超长 post 内存爆）
STATS_OP_SAMPLE = 20      # rag_stats 返回前 N 个高频算子

# Qdrant client 单例
_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            timeout=30,
            check_compatibility=False,
        )
    return _client


# ─── 时间格式化 ────────────────────────────────────────────────────────────
def _format_age(ts: int | None) -> str | None:
    if not ts:
        return None
    now = int(time.time())
    age_sec = max(0, now - ts)
    days = age_sec // 86400
    if days < 1:
        return "今天"
    if days < 7:
        return f"{days} 天前"
    if days < 30:
        return f"{days // 7} 周前"
    if days < 365:
        return f"约 {days // 30} 个月前"
    years = days // 365
    months = (days % 365) // 30
    if months == 0:
        return f"约 {years} 年前"
    return f"约 {years} 年 {months} 个月前"


def _format_date(iso_str: str | None) -> str | None:
    if not iso_str:
        return None
    # 截前 10 字符（YYYY-MM-DD）
    return iso_str[:10] if len(iso_str) >= 10 else iso_str


# ─── ref 编解码 ────────────────────────────────────────────────────────────
def _make_ref(payload: dict) -> str:
    """从 payload 生成可被 rag_fetch 用的 ref。"""
    src = payload.get("source")
    if src in ("forum_post", "forum_comment"):
        pid = payload.get("post_id")
        cid = payload.get("comment_id")
        if cid:
            return f"comment:{cid}"
        if pid:
            return f"post:{pid}"
    if src == "tutorial":
        tid = payload.get("tutorial_id")
        if tid:
            return f"tutorial:{tid}"
    return f"chunk:{payload.get('chunk_type','?')}:{payload.get('post_id') or payload.get('tutorial_id') or '?'}"


_REF_RE = re.compile(r"^(post|comment|tutorial|chunk):(.+)$")


def parse_ref(ref: str) -> tuple[str, str]:
    """'post:18632798681623' → ('post', '18632798681623')"""
    m = _REF_RE.match(ref.strip())
    if not m:
        raise ValueError(
            f"Invalid ref: {ref!r}. "
            "Format: 'post:<post_id>' / 'comment:<comment_id>' / 'tutorial:<tutorial_id>' / 'chunk:<uuid>'"
        )
    return m.group(1), m.group(2)


# ─── hit 格式化（单 chunk） ────────────────────────────────────────────────
_FIELDS_TO_DROP = {
    "source_path", "source_hash", "chunk_index", "segment_index",
    "created_at_ts", "age_days", "code_part", "is_code_only",
}


def _format_chunk(payload: dict, score: float | None, truncate: int) -> dict:
    """单个 chunk 的 LLM 友好表示。"""
    body = payload.get("body_md") or ""
    is_truncated = len(body) > truncate
    content = body[:truncate] + ("..." if is_truncated else "")

    out: dict[str, Any] = {
        "ref": _make_ref(payload),
        "source": payload.get("source"),
        "chunk_type": payload.get("chunk_type"),
        "title": payload.get("title"),
        "author": payload.get("author_id"),
        "date": _format_date(payload.get("created_at")),
        "age": _format_age(payload.get("created_at_ts")),
        "year": payload.get("year"),
        "tags": payload.get("tags") or [],
        "operators": payload.get("operators_mentioned") or [],
        "regions": payload.get("regions") or [],
        "sa_type": payload.get("sa_type"),
        "has_code": bool(payload.get("has_code")),
        "vote_count": payload.get("vote_count"),
        "total_comments": payload.get("total_comments"),
        "reported_sharpe": payload.get("reported_sharpe"),
        "content": content,
    }
    if score is not None:
        out["score"] = round(float(score), 4)
    if is_truncated:
        out["truncated"] = True
        out["fetch_full_with"] = f"rag_fetch(ref='{out['ref']}')"
    # 去掉 None 值，减少 token
    return {k: v for k, v in out.items() if v not in (None, [], "")}


# ─── 同 parent 聚合 ────────────────────────────────────────────────────────
def _parent_key(payload: dict) -> tuple[str, str]:
    """聚合 key：(source, parent_id)。"""
    src = payload.get("source") or "?"
    if src == "tutorial":
        return (src, str(payload.get("tutorial_id") or ""))
    # forum_post / forum_comment 都按 post_id 聚合，因为同一帖子下的评论
    # 单独 chunk_type=comment 时不该被 post 吞掉 → 按 (source, comment_id 或 post_id)
    cid = payload.get("comment_id")
    if src == "forum_comment" and cid:
        return (src, str(cid))
    return (src, str(payload.get("post_id") or ""))


def _aggregate_hits(points: list, truncate: int) -> list[dict]:
    """按 parent_key 聚合多 segment 为单 hit，取最高分。"""
    by_key: dict[tuple, list] = {}
    for p in points:
        payload = p.payload or {}
        k = _parent_key(payload)
        by_key.setdefault(k, []).append(p)

    aggregated = []
    for key, group in by_key.items():
        # 取该 parent 内最高分 chunk 作为代表
        group.sort(key=lambda p: -(getattr(p, "score", 0) or 0))
        top = group[0]
        formatted = _format_chunk(top.payload or {}, getattr(top, "score", None), truncate)
        # 标记该 parent 下还有多少 chunk 命中
        if len(group) > 1:
            formatted["segments_matched"] = len(group)
        aggregated.append(formatted)

    # 按 score 降序
    aggregated.sort(key=lambda h: -(h.get("score") or 0))
    return aggregated


# ─── 过滤器构造 ────────────────────────────────────────────────────────────
def build_filter(
    *,
    source: str | None = None,
    tags: list[str] | None = None,
    operators: list[str] | None = None,
    recency: str | None = None,
    since_days: int | None = None,
    has_code: bool | None = None,
    sa_type: str | None = None,
) -> qm.Filter | None:
    """构造 Qdrant Filter。多条件 AND。tags / operators 是 OR (MatchAny)。"""
    must: list[qm.Condition] = []

    if source:
        must.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=source)))
    if tags:
        must.append(qm.FieldCondition(key="tags", match=qm.MatchAny(any=list(tags))))
    if operators:
        must.append(qm.FieldCondition(key="operators_mentioned", match=qm.MatchAny(any=list(operators))))
    if recency:
        must.append(qm.FieldCondition(key="recency_tag", match=qm.MatchValue(value=recency)))
    if since_days is not None:
        cutoff = int(time.time()) - since_days * 86400
        must.append(qm.FieldCondition(key="created_at_ts", range=qm.Range(gte=cutoff)))
    if has_code is not None:
        must.append(qm.FieldCondition(key="has_code", match=qm.MatchValue(value=has_code)))
    if sa_type:
        must.append(qm.FieldCondition(key="sa_type", match=qm.MatchValue(value=sa_type)))

    return qm.Filter(must=must) if must else None


# ─── 主入口：检索 ──────────────────────────────────────────────────────────
def search_hybrid(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    truncate: int = DEFAULT_CONTENT_TRUNCATE,
    filt: qm.Filter | None = None,
) -> dict:
    if not (query or "").strip():
        raise ValueError("query 不能为空。请传入自然语言或算子名。如果想纯过滤浏览，给一个泛 query 如 'alpha' 配合 filters。")
    top_k = max(1, min(top_k, MAX_TOP_K))
    client = get_client()
    r = get_embedder().embed([query], input_type="query")
    dense_q = r.dense[0]
    sparse_q = r.sparse[0] if config.SUPPORTS_SPARSE else None

    # prefetch 多召回再融合（聚合后可能掉一些 candidate）
    prefetch_limit = top_k * PREFETCH_MULTIPLIER
    prefetch = [qm.Prefetch(query=dense_q, using="dense", limit=prefetch_limit, filter=filt)]
    if sparse_q:
        prefetch.append(qm.Prefetch(
            query=qm.SparseVector(indices=sparse_q["indices"], values=sparse_q["values"]),
            using="sparse",
            limit=prefetch_limit,
            filter=filt,
        ))

    res = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=prefetch,
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=prefetch_limit,
        with_payload=True,
    )
    hits = _aggregate_hits(res.points, truncate)[:top_k]

    # 汇总
    sources = {}
    years = {}
    score_min = score_max = None
    for h in hits:
        sources[h.get("source")] = sources.get(h.get("source"), 0) + 1
        y = h.get("year")
        if y:
            years[y] = years.get(y, 0) + 1
        s = h.get("score")
        if s is not None:
            score_min = s if score_min is None else min(score_min, s)
            score_max = s if score_max is None else max(score_max, s)

    return {
        "query": query,
        "summary": {
            "total_hits": len(hits),
            "score_range": [score_min, score_max] if score_min is not None else None,
            "by_source": sources,
            "by_year": years,
        },
        "hits": hits,
    }


# ─── 主入口：fetch（按 ref 拉完整内容） ────────────────────────────────────
def fetch_by_ref(ref: str, *, full_content: bool = True) -> dict:
    """按 ref 拉完整内容。post/tutorial 会聚合该来源下所有 chunks 按 segment_index 排序拼接。"""
    kind, ident = parse_ref(ref)
    client = get_client()

    if kind == "chunk":
        # 直接按 point id 取 — 但 chunk: 形式我们实际不暴露，留作扩展
        raise NotImplementedError("chunk: ref 暂不支持直接按 point id 取；用 post:/comment:/tutorial: 形式")

    # 按 key 字段过滤所有相关 chunks
    if kind == "post":
        filt = qm.Filter(must=[qm.FieldCondition(key="post_id", match=qm.MatchValue(value=ident))])
    elif kind == "comment":
        filt = qm.Filter(must=[qm.FieldCondition(key="comment_id", match=qm.MatchValue(value=ident))])
    elif kind == "tutorial":
        filt = qm.Filter(must=[qm.FieldCondition(key="tutorial_id", match=qm.MatchValue(value=ident))])
    else:
        raise ValueError(f"Unknown ref kind: {kind}")

    points, next_offset = client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=filt,
        limit=MAX_FETCH_CHUNKS,
        with_payload=True,
        with_vectors=False,
    )
    truncated_fetch = next_offset is not None  # 还有更多 chunk 没拉完
    if not points:
        return {"ref": ref, "found": False, "message": f"No chunks found for {ref}"}

    # 排序：先按 chunk_type 优先级，再按 segment_index
    type_order = {"post": 0, "post_segment": 1, "comment": 2, "comment_segment": 3, "tutorial_section": 4, "code_block": 5}
    points.sort(key=lambda p: (
        type_order.get((p.payload or {}).get("chunk_type"), 99),
        (p.payload or {}).get("segment_index") or 0,
        (p.payload or {}).get("chunk_index") or 0,
    ))

    # 取第一个 chunk 的 meta 作为整体元信息
    head_payload = points[0].payload or {}
    meta = _format_chunk(head_payload, score=None, truncate=10**9)  # 不截断
    meta.pop("content", None)
    meta.pop("truncated", None)
    meta.pop("fetch_full_with", None)

    if not full_content:
        # 只返回 chunk 列表概要
        chunks_info = []
        for p in points:
            md = p.payload or {}
            chunks_info.append({
                "chunk_type": md.get("chunk_type"),
                "segment_index": md.get("segment_index") or md.get("chunk_index"),
                "char_count": len(md.get("body_md") or ""),
                "has_code": bool(md.get("has_code")),
            })
        return {"ref": ref, "found": True, "meta": meta, "chunks": chunks_info, "full_content": None}

    # 拼接全文（同 chunk_type 内按 segment_index 顺序）
    parts: list[str] = []
    seen_main = False
    code_blocks = []
    for p in points:
        md = p.payload or {}
        body = (md.get("body_md") or "").strip()
        if not body:
            continue
        ct = md.get("chunk_type")
        if ct == "code_block":
            # code_block 单独收集，避免和主体重复（主体内部 ``` 已含代码）
            code_blocks.append(body)
            continue
        # post / comment 完整版只取一次；segment 多个按顺序拼
        if ct in ("post", "comment", "tutorial_section") and not seen_main and (md.get("segment_index") is None or md.get("segment_index") == 0):
            parts.append(body)
            seen_main = True
        elif ct in ("post_segment", "comment_segment", "tutorial_section"):
            parts.append(body)

    full = "\n\n".join(parts).strip()
    result = {
        "ref": ref,
        "found": True,
        "meta": meta,
        "chunk_count": len(points),
        "full_content": full,
        "code_blocks_count": len(code_blocks),
    }
    if truncated_fetch:
        result["truncated_fetch"] = True
        result["warning"] = (
            f"该来源 chunks 数 > {MAX_FETCH_CHUNKS}，仅拉取了前 {MAX_FETCH_CHUNKS} 个。"
            "如需完整内容，请加大 MAX_FETCH_CHUNKS 或调小 rag_search 的过滤范围。"
        )
    return result


# ─── 主入口：stats ─────────────────────────────────────────────────────────
_KNOWN_TAGS = [
    "alpha_inspiration", "template_share", "code_share", "experience",
    "consultant_lead", "tool_mcp", "question", "paper_research",
    "competition", "dataset_intro", "tutorial",
    "has_code", "has_simulation_example",
    "high_engagement", "top_engagement",
]
_KNOWN_RECENCY = ["fresh", "recent", "older", "archived"]
_KNOWN_YEARS = ["2023", "2024", "2025", "2026"]


def compute_stats(*, sample_operators: int = STATS_OP_SAMPLE) -> dict:
    """返回知识库元信息。每个分布字段用 server 端 count + filter 拿到精确值。"""
    client = get_client()
    info = client.get_collection(config.QDRANT_COLLECTION)
    total = info.points_count

    def cnt(field, value):
        return client.count(
            config.QDRANT_COLLECTION,
            count_filter=qm.Filter(must=[qm.FieldCondition(
                key=field, match=qm.MatchValue(value=value))]),
            exact=True,
        ).count

    sources = {v: cnt("source", v) for v in ("forum_post", "forum_comment", "tutorial")}
    chunk_types = {v: cnt("chunk_type", v) for v in (
        "post", "post_segment", "comment", "comment_segment", "code_block", "tutorial_section")}
    recency = {v: cnt("recency_tag", v) for v in _KNOWN_RECENCY}
    years = {v: cnt("year", v) for v in _KNOWN_YEARS}
    tags = {v: cnt("tags", v) for v in _KNOWN_TAGS}
    tags = {k: v for k, v in tags.items() if v > 0}

    # 算子样本（按 config.SA_OPERATORS 前 N 个 + 命中数）
    operators = {}
    for op in config.SA_OPERATORS[:sample_operators]:
        n = cnt("operators_mentioned", op)
        if n > 0:
            operators[op] = n

    return {
        "collection": config.QDRANT_COLLECTION,
        "total_chunks": total,
        "status": str(info.status),
        "distribution": {
            "by_source": sources,
            "by_chunk_type": chunk_types,
            "by_recency": recency,
            "by_year": years,
        },
        "tags": tags,
        "operators_sample": operators,
        "filter_hint": {
            "recency_values": _KNOWN_RECENCY,
            "source_values": list(sources.keys()),
            "tag_values": list(tags.keys()),
            "sa_type_values": ["selection", "combo", "regular"],
            "examples": {
                "近期Alpha灵感": {"tags": ["alpha_inspiration"], "recency": "fresh"},
                "含ts_corr的代码": {"operators": ["ts_corr"], "has_code": True},
                "教程": {"source": "tutorial"},
            },
        },
    }

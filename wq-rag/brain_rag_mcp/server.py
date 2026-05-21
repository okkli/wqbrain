"""brain-rag-mcp — WorldQuant BRAIN 知识库的 MCP 工具。

默认走 **streamable-http** 模式（无状态），多 CLI 窗口可共享同一进程不阻塞。

启动方式:
    # 默认 HTTP，监听 127.0.0.1:8765
    python server.py

    # 自定义端口 / 公网监听
    RAG_MCP_PORT=9000 python server.py
    RAG_MCP_HOST=0.0.0.0 RAG_MCP_PORT=8765 python server.py

    # 切回 stdio（单进程，命令式挂载到 Claude Code）
    RAG_MCP_TRANSPORT=stdio python server.py

环境变量:
    RAG_MCP_TRANSPORT  stdio | streamable-http (default)
    RAG_MCP_HOST       默认 127.0.0.1
    RAG_MCP_PORT       默认 8765
    RAG_MCP_PATH       默认 /mcp（streamable-http 路径）
"""
from __future__ import annotations
import logging
import os
import sys
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

# retrieval.py 自带 sys.path 注入 brain_index/
from retrieval import (  # noqa: E402
    DEFAULT_CONTENT_TRUNCATE,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    build_filter,
    compute_stats,
    fetch_by_ref,
    search_hybrid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("brain-rag-mcp")


# ─── MCP server ────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="brain-rag-mcp",
    instructions=(
        "WorldQuant BRAIN 知识库 RAG 工具集。"
        "提供混合检索（BGE-M3 dense+sparse + Qdrant Fusion RRF）+ 按 ref 拉全文 + 库元信息查询。"
        "数据源: 论坛帖/评论(中文+SA表达式代码) + 官方教程。"
        "工具调用顺序建议: 不熟悉数据时先 rag_stats() 看可过滤维度; 然后 rag_search(query, ...filters); "
        "命中后用 ref 字段调 rag_fetch(ref=...) 取完整原文。"
    ),
    host=os.getenv("RAG_MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("RAG_MCP_PORT", "8765")),
    streamable_http_path=os.getenv("RAG_MCP_PATH", "/mcp"),
    stateless_http=True,    # 无状态：多 CLI 各自独立请求，不共享 session
    json_response=True,     # JSON 响应（非 SSE 流），LLM 处理简单
)


# ─── Tool 1: rag_search ────────────────────────────────────────────────────
@mcp.tool()
async def rag_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    source: Optional[str] = None,
    tags: Optional[list[str]] = None,
    operators: Optional[list[str]] = None,
    recency: Optional[str] = None,
    since_days: Optional[int] = None,
    has_code: Optional[bool] = None,
    sa_type: Optional[str] = None,
    truncate_content: int = DEFAULT_CONTENT_TRUNCATE,
) -> dict[str, Any]:
    """🔍 在 BRAIN 知识库做混合检索（dense + sparse Fusion RRF）。

    返回相关 chunks，同一 post/tutorial 的多个 segment 会自动聚合为一条 hit。
    每条 hit 含 ref 字段，可用 rag_fetch 拉完整原文。

    Args:
        query: 自然语言查询。中文/英文/算子名都可，sparse 在精确算子名（ts_corr/trade_when）召回上更强。
        top_k: 返回 hit 数量，默认 8，最大 50。
        source: 限定来源。可选 "forum_post" / "forum_comment" / "tutorial"。
        tags: 标签过滤（OR）。可用值见 rag_stats().tags。常用：alpha_inspiration / template_share /
              code_share / experience / consultant_lead / tool_mcp / question / paper_research /
              competition / dataset_intro / tutorial / has_code / has_simulation_example /
              high_engagement / top_engagement。
        operators: SA 算子过滤（OR）。要求 chunk 命中至少一个，如 ["ts_corr", "trade_when"]。
        recency: 时效过滤。可选 "fresh"(≤6m) / "recent"(≤2y) / "older"(≤5y) / "archived"(>5y)。
        since_days: 替代 recency 的精确时间窗，如 since_days=90 表示最近 90 天。
        has_code: 仅查含代码块的 chunk。
        sa_type: SA 类型过滤。可选 "selection" / "combo" / "regular"。
        truncate_content: 单 hit content 字符数上限（默认 800），超过会截断并标 truncated=true。

    Returns:
        {
          "query": str,                          # 原 query 回显
          "applied_filters": dict,               # 实际生效的过滤条件
          "summary": {                           # 召回汇总
            "total_hits": int,
            "score_range": [low, high],
            "by_source": dict,
            "by_year": dict,
          },
          "hits": [                              # 已按 score 降序
            {
              "rank": int, "score": float,
              "ref": str,                        # 用于 rag_fetch
              "source": str, "chunk_type": str, "title": str, "author": str,
              "date": "YYYY-MM-DD", "age": "人类可读", "year": str,
              "tags": [...], "operators": [...], "regions": [...],
              "sa_type": str | None, "has_code": bool,
              "vote_count": int | None, "total_comments": int | None,
              "content": str,                    # 已截断
              "truncated": bool,                 # 仅截断时出现
              "fetch_full_with": str,            # 仅截断时出现
              "segments_matched": int,           # 同 parent 命中 >1 segment 时出现
            }, ...
          ]
        }

    Examples:
        rag_search("如何降低 turnover")
        rag_search("ts_corr 用法", operators=["ts_corr"], has_code=True)
        rag_search("alpha 灵感", tags=["alpha_inspiration", "high_engagement"], recency="fresh")
        rag_search("最近的顾问帖", source="forum_post", tags=["consultant_lead"], since_days=30)
    """
    try:
        filt = build_filter(
            source=source, tags=tags, operators=operators,
            recency=recency, since_days=since_days,
            has_code=has_code, sa_type=sa_type,
        )
        result = search_hybrid(query, top_k=top_k, truncate=truncate_content, filt=filt)
        # 给 hits 加 rank
        for i, h in enumerate(result.get("hits", [])):
            h["rank"] = i + 1
        # 回显过滤条件
        result["applied_filters"] = {
            k: v for k, v in {
                "source": source, "tags": tags, "operators": operators,
                "recency": recency, "since_days": since_days,
                "has_code": has_code, "sa_type": sa_type,
            }.items() if v is not None
        }
        return result
    except Exception as e:
        log.exception("rag_search failed")
        return {"error": str(e), "type": type(e).__name__}


# ─── Tool 2: rag_fetch ─────────────────────────────────────────────────────
@mcp.tool()
async def rag_fetch(ref: str, full_content: bool = True) -> dict[str, Any]:
    """📄 按 ref 拉完整原文。配合 rag_search 命中后取细节。

    Args:
        ref: rag_search hit 里的 ref 字段。格式:
             "post:<post_id>"           — 拉论坛帖正文（多 segment 自动按顺序拼接）
             "comment:<comment_id>"     — 拉单条评论
             "tutorial:<tutorial_id>"   — 拉教程全文
        full_content: True 返回拼接后的完整原文；False 仅返回 chunk 概览（数量/长度），不传内容。

    Returns:
        {
          "ref": str, "found": bool,
          "meta": { title, author, date, age, tags, operators, ... },  # 元信息（不含 content）
          "chunk_count": int,
          "full_content": str | None,    # 拼接后的完整 markdown
          "code_blocks_count": int,      # 该来源下命中的独立 code_block chunk 数
        }

    Examples:
        rag_fetch(ref="post:18632798681623")
        rag_fetch(ref="tutorial:19-alpha-examples")
        rag_fetch(ref="comment:18632798681623_c0")
    """
    try:
        return fetch_by_ref(ref, full_content=full_content)
    except Exception as e:
        log.exception("rag_fetch failed")
        return {"error": str(e), "type": type(e).__name__}


# ─── Tool 3: rag_stats ─────────────────────────────────────────────────────
@mcp.tool()
async def rag_stats() -> dict[str, Any]:
    """📊 知识库元信息：总数、分布、可用过滤维度。

    **不确定怎么过滤时优先调这个**——告诉你都有哪些 tags / sources / operators / years
    可以塞进 rag_search 的参数。无参数。

    Returns:
        {
          "collection": str, "total_chunks": int, "status": "green" | ...,
          "distribution": {
            "by_source": {forum_post: N, forum_comment: N, tutorial: N},
            "by_chunk_type": {post: N, post_segment: N, ...},
            "by_recency": {fresh: N, recent: N, older: N, archived: N},
            "by_year": {"2023": N, "2024": N, ...},
          },
          "tags": {alpha_inspiration: N, template_share: N, ...},   # 仅含 N>0
          "operators_sample": {ts_corr: N, ts_rank: N, ...},        # 前 20 个高频算子
          "filter_hint": {
            "recency_values": [...], "source_values": [...],
            "tag_values": [...], "sa_type_values": [...],
            "examples": {...},
          }
        }

    LLM 决策流程示例:
        # 拿到 rag_stats 返回后
        stats = rag_stats()
        # 1) 想查特定算子：检查它在不在 operators_sample
        if "ts_corr" in stats["operators_sample"]:
            rag_search(query=user_q, operators=["ts_corr"])
        # 2) 想查近期内容：用 recency 取代手动估时间
        rag_search(query=user_q, recency="fresh")    # ≤6 个月
        # 3) 想限定主题：从 tags 里选
        rag_search(query=user_q, tags=["alpha_inspiration", "consultant_lead"])
        # 4) 看 filter_hint.examples 直接照抄常见组合
    """
    try:
        return compute_stats()
    except Exception as e:
        log.exception("rag_stats failed")
        return {"error": str(e), "type": type(e).__name__}


# ─── 入口 ──────────────────────────────────────────────────────────────────
def _print_startup_banner(transport: str):
    if transport == "stdio":
        log.info("brain-rag-mcp running on stdio")
    else:
        host = mcp.settings.host
        port = mcp.settings.port
        path = mcp.settings.streamable_http_path
        log.info(f"brain-rag-mcp listening on http://{host}:{port}{path}")
        log.info("Claude Code mcp config: type=http url=above")


def main():
    transport = os.getenv("RAG_MCP_TRANSPORT", "streamable-http")
    if transport not in ("stdio", "streamable-http", "sse"):
        print(f"unknown transport: {transport}", file=sys.stderr)
        sys.exit(2)
    _print_startup_banner(transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()

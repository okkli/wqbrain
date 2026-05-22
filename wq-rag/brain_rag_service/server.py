"""brain-rag-service — 一个进程同时提供 MCP + REST。

入口架构:
    FastAPI app (root)
      ├── /docs           Swagger UI（FastAPI 自动）
      ├── /openapi.json   OpenAPI schema
      ├── /healthz        健康检查
      ├── /v1/search      REST: 混合检索
      ├── /v1/fetch       REST: 按 ref 拉全文
      ├── /v1/stats       REST: 库元信息
      ├── /v1/rag/answer  REST: 便利接口（检索 + 拼 context）
      └── /mcp            FastMCP streamable-http（给 Claude Code 等 LLM CLI 用）

启动方式（推荐）:
    python server.py                                # 默认 127.0.0.1:8765
    RAG_SVC_HOST=0.0.0.0 RAG_SVC_PORT=9000 python server.py
    # 或直接走 uvicorn:
    uvicorn server:app --host 127.0.0.1 --port 8765 --reload

环境变量:
    RAG_SVC_HOST          默认 127.0.0.1（公网开放前请加反代+防火墙）
    RAG_SVC_PORT          默认 8765
    RAG_SVC_RELOAD        `1` 启用 dev 模式热重载（仅 dev）
    RAG_MCP_PATH          默认 /mcp（MCP streamable-http 路径）
    RAG_DISABLE_DOCS      `1` 关闭 /docs /redoc /openapi.json（生产推荐）
    RAG_SYSTEM_PROMPT     覆盖 /v1/rag/answer 默认的中文 system prompt（业务自定义/英文场景）
"""
from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

# retrieval.py 自带 sys.path 注入 brain_index/
from retrieval import (  # noqa: E402
    DEFAULT_CONTENT_TRUNCATE,
    DEFAULT_TOP_K,
    build_filter,
    compute_stats,
    fetch_by_ref,
    search_hybrid,
)
from api import router as api_router  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("brain-rag-service")


# ─── MCP server（路径 /mcp）────────────────────────────────────────────────
mcp = FastMCP(
    name="brain-rag-service",
    instructions=(
        "WorldQuant BRAIN 知识库 RAG 工具集。"
        "提供混合检索（BGE-M3 dense+sparse + Qdrant Fusion RRF）+ 按 ref 拉全文 + 库元信息查询。"
        "数据源: 论坛帖/评论（中文+SA表达式代码）+ 官方教程。"
        "工具调用顺序建议: 不熟悉数据时先 rag_stats() 看可过滤维度; 然后 rag_search(query, ...filters); "
        "命中后用 ref 字段调 rag_fetch(ref=...) 取完整原文。"
    ),
    # 子应用挂载在 /mcp，所以内部 path 用 "/"，最终对外路径仍是 /mcp
    streamable_http_path="/",
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def rag_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    source: str | None = None,
    tags: list[str] | None = None,
    operators: list[str] | None = None,
    recency: str | None = None,
    since_days: int | None = None,
    has_code: bool | None = None,
    sa_type: str | None = None,
    truncate_content: int = DEFAULT_CONTENT_TRUNCATE,
) -> dict:
    """🔍 BRAIN 知识库混合检索（dense + sparse Fusion RRF）。

    Args:
        query: 自然语言或算子名查询
        top_k: 返回数量，1-50，默认 8
        source: 来源过滤 forum_post / forum_comment / tutorial
        tags: 标签过滤 (OR)，可用值见 rag_stats().tags
        operators: SA 算子过滤 (OR)，如 ['ts_corr','trade_when']
        recency: 时效 fresh(≤6m)/recent(≤2y)/older(≤5y)/archived(>5y)
        since_days: 精确时间窗（天）
        has_code: 仅查含代码块
        sa_type: SA 类型 selection/combo/regular
        truncate_content: content 字符数上限，默认 800

    Returns:
        dict 含 query, applied_filters, summary, hits[]
        每个 hit 含 ref（用于 rag_fetch）, score, title, author, content, tags, ...

    Examples:
        rag_search("如何降低 turnover")
        rag_search("ts_corr 用法", operators=["ts_corr"], has_code=True)
        rag_search("alpha 灵感", tags=["alpha_inspiration"], recency="fresh")
    """
    try:
        filt = build_filter(source=source, tags=tags, operators=operators,
                            recency=recency, since_days=since_days,
                            has_code=has_code, sa_type=sa_type)
        result = search_hybrid(query, top_k=top_k, truncate=truncate_content, filt=filt)
        for i, h in enumerate(result.get("hits", [])):
            h["rank"] = i + 1
        result["applied_filters"] = {
            k: v for k, v in {"source": source, "tags": tags, "operators": operators,
                              "recency": recency, "since_days": since_days,
                              "has_code": has_code, "sa_type": sa_type}.items()
            if v is not None
        }
        return result
    except Exception as e:
        log.exception("rag_search failed")
        return {"error": str(e), "type": type(e).__name__}


@mcp.tool()
async def rag_fetch(ref: str, full_content: bool = True) -> dict:
    """📄 按 ref 拉完整原文。

    Args:
        ref: 'post:<id>' / 'comment:<id>' / 'tutorial:<id>' （从 rag_search hit 拿）
        full_content: True 返回拼接全文；False 仅返回 chunk 概览

    Examples:
        rag_fetch(ref="post:18632798681623")
        rag_fetch(ref="tutorial:19-alpha-examples")
    """
    try:
        return fetch_by_ref(ref, full_content=full_content)
    except Exception as e:
        log.exception("rag_fetch failed")
        return {"error": str(e), "type": type(e).__name__}


@mcp.tool()
async def rag_stats() -> dict:
    """📊 知识库元信息：总数、分布、可用过滤维度。

    **不确定怎么过滤时优先调这个**。返回 tags / sources / operators / years 列表
    可塞进 rag_search 的参数。

    LLM 决策流程:
        1) stats = rag_stats()
        2) 检查 stats["operators_sample"] 是否含目标算子
        3) 从 stats["tags"] 挑相关 tag
        4) 用 stats["filter_hint"]["examples"] 参考常见组合
        5) 调 rag_search(query=..., **filters)
    """
    try:
        return compute_stats()
    except Exception as e:
        log.exception("rag_stats failed")
        return {"error": str(e), "type": type(e).__name__}


# ─── FastAPI app（路由 /v1/* + /healthz + /docs）───────────────────────────
# MCP 的 ASGI 子应用要在 FastAPI lifespan 里启动其内部 task group
mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """组合 MCP 子应用的 lifespan，让其 SessionManager / task group 正确启动。"""
    async with mcp_app.router.lifespan_context(app):
        log.info("brain-rag-service: MCP + REST 同时启动")
        yield


# 生产环境可关 docs 减少攻击面：RAG_DISABLE_DOCS=1
_DOCS_DISABLED = os.getenv("RAG_DISABLE_DOCS") == "1"

app = FastAPI(
    title="BRAIN RAG Service",
    description=(
        "WorldQuant BRAIN 论坛 + 教程 RAG 知识库的统一对外服务。\n\n"
        "**双入口**:\n"
        "- **REST API** (本文档) 给业务系统 / agent 框架用 → `/v1/*` + `/healthz` + `/readyz` + `/livez`\n"
        "- **MCP** 给 Claude Code 等 LLM CLI 用 → `/mcp` (streamable-http)\n\n"
        "数据源: BRAIN 论坛 + 官方教程。"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _DOCS_DISABLED else "/docs",
    redoc_url=None if _DOCS_DISABLED else "/redoc",
    openapi_url=None if _DOCS_DISABLED else "/openapi.json",
)

# REST 路由
app.include_router(api_router, tags=["RAG"])

# 挂载 MCP 子应用到 /mcp（注意：必须在 include_router 之后，避免 catch-all 吞 FastAPI 路由）
# FastMCP 的 streamable_http_path="/" 配合 mount("/mcp") = 对外 URL /mcp
app.mount(os.getenv("RAG_MCP_PATH", "/mcp"), mcp_app)


# ─── 启动入口 ──────────────────────────────────────────────────────────────
def main():
    import uvicorn
    host = os.getenv("RAG_SVC_HOST", "127.0.0.1")
    port = int(os.getenv("RAG_SVC_PORT", "8765"))
    reload = os.getenv("RAG_SVC_RELOAD") == "1"
    log.info(f"brain-rag-service starting on http://{host}:{port}")
    log.info(f"  REST API   : http://{host}:{port}/docs" + (" [disabled]" if _DOCS_DISABLED else ""))
    log.info(f"  MCP        : http://{host}:{port}/mcp")
    log.info(f"  K8s probes : /readyz (严格) /livez (宽松) /healthz (信息)")
    uvicorn.run(
        "server:app" if reload else app,
        host=host, port=port, reload=reload, log_level="info",
    )


if __name__ == "__main__":
    main()

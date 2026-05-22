"""FastAPI 路由 — REST 入口给业务系统 / agent 框架 / 后端服务调。

设计:
- /v1/search       主力混合检索（dense+sparse Fusion RRF）
- /v1/fetch        按 ref 拉全文
- /v1/stats        库元信息
- /v1/rag/answer   LLM 便利接口（pre-拼 context + citations，不调 LLM）
- /healthz         总健康（degraded 仍返 200，仅 down 返 503）
- /readyz          严格 readiness（全部 ok 才 200，给 K8s readiness probe）
- /livez           宽松 liveness（进程在跑就 200，给 K8s liveness probe）

错误约定:
- 422 Pydantic schema 校验（空 query、缺字段、越界、非法 Literal 都是这个）
- 400 业务参数错（ref 格式错等运行时校验）
- 404 资源不存在（ref 合法但库里没有）
- 502 后端依赖不通（BGE-M3 / Qdrant 超时或异常）
- 500 内部未知错误（兜底）
- 503 healthz 报告依赖全部 down

返回 body 始终是 application/json，业务直接 .json()。
"""
from __future__ import annotations
import logging
import os
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

import config  # from brain_index via retrieval.py path injection
from models import (
    AnswerRequest,
    AnswerResponse,
    Citation,
    ComponentHealth,
    FetchRequest,
    Filters,
    HealthResponse,
    SearchRequest,
)
from retrieval import (
    build_filter,
    compute_stats,
    fetch_by_ref,
    get_client,
    search_hybrid,
)

log = logging.getLogger("brain-rag-api")
router = APIRouter()


# ─── 内部辅助 ──────────────────────────────────────────────────────────────
def _filters_to_kwargs(f: Filters | None) -> dict[str, Any]:
    if f is None:
        return {}
    return {k: v for k, v in f.model_dump().items() if v is not None}


def _to_qm_filter(f: Filters | None) -> Optional[Any]:
    """构造 qdrant Filter。返回 qm.Filter | None。"""
    return build_filter(**_filters_to_kwargs(f))


def _wrap_backend_error(e: Exception) -> HTTPException:
    """把后端依赖异常映射成 502；其它未知异常映射 500。"""
    name = type(e).__name__
    if isinstance(e, (httpx.HTTPError, httpx.ConnectError, httpx.ReadTimeout)):
        return HTTPException(status_code=502, detail=f"backend unreachable: {name}: {e}")
    return HTTPException(status_code=500, detail=f"{name}: {e}")


# ─── /v1/search ────────────────────────────────────────────────────────────
@router.post(
    "/v1/search",
    summary="混合检索（dense + sparse Fusion RRF）",
    response_description="hits + summary + applied_filters",
)
async def search(req: SearchRequest) -> dict[str, Any]:
    """主力检索接口。

    返回结构与 MCP rag_search 一致。同 post/tutorial 的多 segment 自动聚合为单 hit，
    content 按 `truncate_content` 截断；超出会带 `truncated=true` 和 `fetch_full_with` 提示。

    示例 body:
    ```json
    {
      "query": "ts_corr 算子的常见用法",
      "top_k": 5,
      "filters": {"operators": ["ts_corr"], "has_code": true},
      "truncate_content": 800
    }
    ```
    """
    try:
        result = search_hybrid(
            req.query,
            top_k=req.top_k,
            truncate=req.truncate_content,
            filt=_to_qm_filter(req.filters),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("search failed")
        raise _wrap_backend_error(e) from e

    for i, h in enumerate(result.get("hits", [])):
        h["rank"] = i + 1
    result["applied_filters"] = _filters_to_kwargs(req.filters)
    return result


# ─── /v1/fetch ─────────────────────────────────────────────────────────────
@router.post(
    "/v1/fetch",
    summary="按 ref 拉完整原文",
    response_description="完整 markdown + 元信息 + chunk_count",
)
async def fetch(req: FetchRequest) -> dict[str, Any]:
    """配合 /v1/search 命中后取细节。同 parent 的多 chunk 按 segment_index 自动拼接。

    ref 格式:
    - `post:<post_id>` — 论坛帖
    - `comment:<comment_id>` — 单条评论
    - `tutorial:<tutorial_id>` — 官方教程
    """
    try:
        r = fetch_by_ref(req.ref, full_content=req.full_content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("fetch failed")
        raise _wrap_backend_error(e) from e

    if not r.get("found"):
        # 用 404 让业务能区分"参数对但库里没有"
        raise HTTPException(status_code=404, detail=r.get("message") or "not found")
    return r


# ─── /v1/stats ─────────────────────────────────────────────────────────────
@router.get(
    "/v1/stats",
    summary="知识库元信息：总数 / 分布 / 可用过滤维度",
    response_description="distribution + tags + operators_sample + filter_hint",
)
async def stats() -> dict[str, Any]:
    """不熟悉数据时先调这个 —— 告诉调用方有哪些 tags / sources / operators / years 可塞 filter。"""
    try:
        return compute_stats()
    except Exception as e:
        log.exception("stats failed")
        raise _wrap_backend_error(e) from e


# ─── /v1/rag/answer ────────────────────────────────────────────────────────
_DEFAULT_SYSTEM_PROMPT = (
    "你是 WorldQuant BRAIN 平台的专家助手。基于下面提供的【检索结果】回答用户问题。\n"
    "要求:\n"
    "1. 答案必须有 [n] 形式的引用，对应 citations 数组里的 id\n"
    "2. 检索结果不足以回答时明确说『基于现有资料无法确定』，不要编造\n"
    "3. 涉及代码/表达式时尽量给出具体片段\n"
    "4. 简洁、给出可操作的建议"
)


def _get_system_prompt() -> str:
    """支持 RAG_SYSTEM_PROMPT env 覆盖默认中文 prompt（业务方可换英文等）。"""
    return os.getenv("RAG_SYSTEM_PROMPT") or _DEFAULT_SYSTEM_PROMPT


def _format_context(hits: list[dict[str, Any]], fmt: str) -> str:
    if fmt == "numbered":
        parts = []
        for i, h in enumerate(hits, 1):
            head = f"[{i}] " + " | ".join(filter(None, [
                h.get("source"),
                h.get("title"),
                h.get("author"),
                h.get("date"),
            ]))
            parts.append(f"{head}\n{h.get('content','')}")
        return "\n\n".join(parts)
    if fmt == "markdown":
        parts = []
        for i, h in enumerate(hits, 1):
            parts.append(
                f"## 来源 {i}: {h.get('title') or h.get('source')}\n"
                f"_作者 {h.get('author','?')} / {h.get('date','?')} / score={h.get('score')}_\n\n"
                f"{h.get('content','')}"
            )
        return "\n\n---\n\n".join(parts)
    if fmt == "json":
        import json
        return json.dumps(
            [{"id": i, **{k: h.get(k) for k in ("ref", "source", "title", "author", "date", "score", "content")}}
             for i, h in enumerate(hits, 1)],
            ensure_ascii=False, indent=2,
        )
    return ""


@router.post(
    "/v1/rag/answer",
    summary="LLM 便利接口：检索 + 预拼接 context + citations（不调 LLM）",
    response_model=AnswerResponse,
    response_description="可直接塞 LLM 的 context + citations + 可选 system prompt",
)
async def rag_answer(req: AnswerRequest) -> AnswerResponse:
    """**最常用的业务接入端点**。

    流程: 用 question 走混合检索 → 把 hits 拼成 context 字符串 → 返回 context + citations。
    业务方拿到 context 后**自己**调 LLM（保持服务无外部 LLM 依赖）。

    示例业务调用流程:
    ```python
    r = httpx.post("http://localhost:8765/v1/rag/answer", json={
        "question": "如何降低 turnover?",
        "filters": {"tags": ["consultant_lead"], "recency": "recent"},
    }).json()
    # 然后业务自己调 LLM:
    llm.chat(system=r['system_prompt'], user=f"问题: {r['question']}\\n\\n检索结果:\\n{r['context']}")
    ```
    """
    try:
        result = search_hybrid(
            req.question,
            top_k=req.top_k,
            truncate=req.truncate_content,
            filt=_to_qm_filter(req.filters),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("answer search failed")
        raise _wrap_backend_error(e) from e

    hits = result.get("hits", [])
    context = _format_context(hits, req.context_format)
    citations = [
        Citation(
            id=i,
            ref=h.get("ref", ""),
            score=h.get("score"),
            title=h.get("title"),
            source=h.get("source", ""),
            author=h.get("author"),
            date=h.get("date"),
        )
        for i, h in enumerate(hits, 1)
    ]

    return AnswerResponse(
        question=req.question,
        context=context,
        citations=citations,
        applied_filters=_filters_to_kwargs(req.filters),
        summary=result.get("summary", {}),
        system_prompt=_get_system_prompt() if req.system_prompt_hint else None,
    )


# ─── /healthz + /readyz + /livez ───────────────────────────────────────────
async def _probe_health() -> tuple[ComponentHealth, ComponentHealth, int | None]:
    """探测 Qdrant + BGE-M3 状态。被 /healthz 和 /readyz 共用。"""
    qd = ComponentHealth(status="down", detail="not checked")
    bm = ComponentHealth(status="down", detail="not checked")
    points: int | None = None

    try:
        info = get_client().get_collection(config.QDRANT_COLLECTION)
        points = info.points_count
        qd = ComponentHealth(status="ok", extra={"status": str(info.status), "points": points})
    except Exception as e:
        qd = ComponentHealth(status="down", detail=f"{type(e).__name__}: {e}")

    try:
        async with httpx.AsyncClient(timeout=5.0) as cli:
            r = await cli.get(f"{config.BGEM3_BASE_URL}/v1/models")
            if r.status_code == 200:
                bm = ComponentHealth(status="ok", extra={"models": r.json().get("data", [])})
            else:
                bm = ComponentHealth(status="down", detail=f"HTTP {r.status_code}")
    except Exception as e:
        bm = ComponentHealth(status="down", detail=f"{type(e).__name__}: {e}")

    return qd, bm, points


@router.get(
    "/healthz",
    summary="健康检查（信息型）",
    response_model=HealthResponse,
    responses={503: {"description": "全部依赖 down"}},
)
async def healthz() -> JSONResponse:
    """报告式健康检查：返回 Qdrant + BGE-M3 各自状态。

    - **全部 ok** → 200, status=ok
    - **降级（一端 ok 一端 down）** → 200, status=degraded（信息型，不让 K8s 误杀）
    - **全部 down** → 503, status=down

    K8s 探针请用 `/readyz`（严格）和 `/livez`（宽松）。
    """
    qd, bm, points = await _probe_health()
    if qd.status == "ok" and bm.status == "ok":
        overall, code = "ok", 200
    elif qd.status == "ok" or bm.status == "ok":
        overall, code = "degraded", 200   # 降级不返 503，避免 K8s 驱逐
    else:
        overall, code = "down", 503

    body = HealthResponse(
        status=overall, qdrant=qd, bgem3=bm,
        collection=config.QDRANT_COLLECTION, points=points,
    ).model_dump()
    return JSONResponse(status_code=code, content=body)


@router.get(
    "/readyz",
    summary="K8s readiness probe（严格）",
    responses={
        200: {"description": "所有依赖 ok，可接流量"},
        503: {"description": "任一依赖不 ok，readiness 失败"},
    },
)
async def readyz() -> PlainTextResponse:
    """**严格**：Qdrant 和 BGE-M3 都必须 ok 才返 200。给 K8s readinessProbe 用。"""
    qd, bm, _ = await _probe_health()
    if qd.status == "ok" and bm.status == "ok":
        return PlainTextResponse("ok", status_code=200)
    return PlainTextResponse(
        f"not ready: qdrant={qd.status} bgem3={bm.status}",
        status_code=503,
    )


@router.get(
    "/livez",
    summary="K8s liveness probe（宽松）",
    responses={200: {"description": "进程存活"}},
)
async def livez() -> PlainTextResponse:
    """**宽松**：进程在跑就返 200，不探外部依赖。给 K8s livenessProbe 用，避免外部抖动导致 pod 被杀。"""
    return PlainTextResponse("ok", status_code=200)

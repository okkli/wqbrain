"""Pydantic 请求 / 响应模型 — 让 FastAPI Swagger 自动生成交互式 API 文档。

设计原则:
- 字段命名与 MCP tools 一致（business 用 REST、LLM CLI 用 MCP，输入语义相同）
- filters 嵌套成单独对象，便于业务方组合传参
- 响应严格反映 retrieval.py 的真实输出结构（避免文档与实现漂移）
"""
from __future__ import annotations
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# ─── 共用过滤器 ───────────────────────────────────────────────────────────
RecencyTag = Literal["fresh", "recent", "older", "archived"]
SourceType = Literal["forum_post", "forum_comment", "tutorial"]
SaType = Literal["selection", "combo", "regular"]


class Filters(BaseModel):
    """检索 / 浏览的过滤条件。所有字段可选，组合 AND；tags/operators 内部为 OR。"""
    source: Optional[SourceType] = Field(None, description="限定来源")
    tags: Optional[list[str]] = Field(None, description="任一命中即可，可用值见 /v1/stats.tags")
    operators: Optional[list[str]] = Field(None, description="任一命中即可，如 ['ts_corr', 'trade_when']")
    recency: Optional[RecencyTag] = Field(None, description="fresh ≤6m / recent ≤2y / older ≤5y / archived >5y")
    since_days: Optional[int] = Field(None, ge=1, le=365 * 20, description="精确时间窗（天）。与 recency 互斥")
    has_code: Optional[bool] = Field(None, description="仅查含代码块")
    sa_type: Optional[SaType] = Field(None, description="SA 类型")


# ─── /v1/search ────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="自然语言或算子名。中英文均可")
    top_k: int = Field(8, ge=1, le=50, description="返回 hit 数，默认 8 上限 50")
    filters: Optional[Filters] = None
    truncate_content: int = Field(800, ge=0, le=20000, description="content 截断字符数；0 表示不截断")


# ─── /v1/fetch ─────────────────────────────────────────────────────────────
class FetchRequest(BaseModel):
    ref: str = Field(
        ...,
        description="rag_search hit 里的 ref。格式 'post:<id>' / 'comment:<id>' / 'tutorial:<id>'",
        examples=["post:18632798681623", "tutorial:19-alpha-examples"],
    )
    full_content: bool = Field(True, description="False 仅返回 chunk 概览不传内容")


# ─── /v1/rag/answer ────────────────────────────────────────────────────────
ContextFormat = Literal["numbered", "json", "markdown"]


class AnswerRequest(BaseModel):
    """LLM 便利接口：检索 + 拼好 context，**不调 LLM**。业务方拿到 context 自己塞给 LLM。"""
    question: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1, le=50)
    filters: Optional[Filters] = None
    truncate_content: int = Field(1500, ge=100, le=20000, description="answer 模式下默认给更长上下文")
    context_format: ContextFormat = Field(
        "numbered",
        description=(
            "拼接格式: numbered=[1]xxx [2]yyy 适合 LLM 引用; "
            "markdown=## 来源 1\\nxxx 适合人读; "
            "json=结构化数组适合机器再处理"
        ),
    )
    system_prompt_hint: bool = Field(
        True, description="返回里是否带一段建议的 system prompt 给业务直接拼"
    )


class Citation(BaseModel):
    id: int
    ref: str
    score: float | None = None
    title: str | None = None
    source: str
    author: str | None = None
    date: str | None = None
    url: str | None = None   # 未来如果加论坛 url 可用


class AnswerResponse(BaseModel):
    question: str
    context: str = Field(..., description="预拼接好的检索结果，业务直接塞 LLM prompt")
    citations: list[Citation]
    applied_filters: dict[str, Any]
    summary: dict[str, Any]
    system_prompt: str | None = Field(
        None, description="建议的 system prompt 模板。当 system_prompt_hint=true 时返回"
    )


# ─── /healthz ──────────────────────────────────────────────────────────────
ComponentStatus = Literal["ok", "down", "degraded"]


class ComponentHealth(BaseModel):
    status: ComponentStatus
    detail: str | None = None
    extra: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: ComponentStatus
    qdrant: ComponentHealth
    bgem3: ComponentHealth
    collection: str
    points: int | None = None

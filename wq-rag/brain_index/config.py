"""配置中心。所有 secrets / 端点 / 维度都来自 .env，代码不硬编码。"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")

# ─── Embedding ─────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bgem3").lower()

# BGE-M3 (主力，原生 dense + sparse)
# 真实端点写在 .env 的 BGEM3_BASE_URL；下面 fallback 仅作为本地起服务时的默认值
BGEM3_BASE_URL = os.getenv("BGEM3_BASE_URL", "http://localhost:30130").rstrip("/")
BGEM3_POOLING_PATH = os.getenv("BGEM3_POOLING_PATH", "/pooling")
BGEM3_DIM = 1024  # BGE-M3 dense 固定 1024

# Doubao (dense-only 备选)
DOUBAO_BASE_URL = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3").rstrip("/")
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-embedding-vision")
DOUBAO_DIM = int(os.getenv("DOUBAO_DIM", "1024"))

# Voyage / OpenAI (dense-only 备选)
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL = "voyage-3-large"
VOYAGE_DIM = 1024

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "text-embedding-3-large"
OPENAI_DIM = 3072

# Provider → dense 维度映射（用于建 Qdrant 集合）
DENSE_DIM_BY_PROVIDER = {
    "bgem3": BGEM3_DIM,
    "doubao": DOUBAO_DIM,
    "voyage": VOYAGE_DIM,
    "openai": OPENAI_DIM,
}
DENSE_DIM = DENSE_DIM_BY_PROVIDER[EMBEDDING_PROVIDER]

# 只有 BGE-M3 原生出 sparse
SUPPORTS_SPARSE = EMBEDDING_PROVIDER == "bgem3"

# ─── Qdrant ────────────────────────────────────────────────────────────────
# 真实端点写在 .env 的 QDRANT_HOST / QDRANT_PORT；下面 fallback 仅作为本地 docker 默认值
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "brain_kb_bgem3")

# ─── 数据源 ────────────────────────────────────────────────────────────────
DATA_ROOT = (_HERE / os.getenv("DATA_ROOT", "../wq-doc-forum/data")).resolve()
FORUM_POSTS_DIR = DATA_ROOT / "forum" / "posts"
KB_TUTORIALS_DIR = DATA_ROOT / "kb" / "tutorials"

# ─── 索引逻辑版本 ──────────────────────────────────────────────────────────
# 改了切分阈值、TAG_RULES、parser/chunker 逻辑后，**手动 bump 这个值**。
# source_hash 混入此版本号，重跑 indexer 会触发未变更文件的 chunk 重新嵌入，
# 保证新规则覆盖到全库。形式建议 "vMAJOR.MINOR"，主版本变 = 切分变 / payload 变。
INDEX_VERSION = os.getenv("INDEX_VERSION", "v1.0")

# ─── SuperAlpha 领域词表（chunker / parser 元数据抽取用）─────────────────
SA_OPERATORS = [
    # 已知高频算子/字段（小写匹配；后续可按需扩充）
    "ts_corr", "ts_rank", "ts_zscore", "ts_mean", "ts_std", "ts_delta",
    "ts_decay_linear", "ts_arg_max", "ts_arg_min", "ts_min", "ts_max",
    "ts_sum", "ts_product", "ts_skewness", "ts_kurtosis", "ts_av_diff",
    "ts_returns", "ts_log_diff", "ts_co_skewness", "ts_regression",
    "rank", "zscore", "group_neutralize", "group_rank", "group_mean",
    "group_zscore", "winsorize", "quantile", "scale", "scale_down",
    "trade_when", "decay_linear", "sign", "abs", "log", "power",
    "if_else", "and_or", "max", "min", "sigmoid", "tanh", "subindustry",
    "industry", "sector", "neutralize", "vec_choose", "vec_avg",
    # 配置属性
    "turnover", "prod_correlation", "self_correlation", "operator_count",
    "dataset_count", "datafield_count", "datacategory_count", "long_count",
    "short_count", "decay", "neutralization", "universe", "own", "favorite",
    "classifications", "datacategories", "datasets", "author_sharpe",
    "author_turnover", "author_tenure", "color", "category", "os_start_date",
    "truncation", "universe_size",
]
SA_REGIONS = ["USA", "EUR", "ASI", "GLB", "CHN", "JPN", "KOR", "TWN", "HKG", "AMR"]

# ─── 切分阈值 ──────────────────────────────────────────────────────────────
# ─── 时效性 ────────────────────────────────────────────────────────────────
# 论坛实测日期范围 2023-2026，绝大部分 2025+。边界单位：秒
RECENCY_BUCKETS = [
    ("fresh", 180 * 86400),       # ≤ 6 个月
    ("recent", 730 * 86400),      # ≤ 2 年
    ("older", 1825 * 86400),      # ≤ 5 年
    ("archived", float("inf")),   # 其余
]

# ─── 内容标签规则（基于实测标题分布校准）─────────────────────────────────
# 每条 rule: (tag, regex)。所有匹配的 tag 都加进 chunk 的 tags 列表。
TAG_RULES = [
    ("alpha_inspiration", r"alpha\s*灵感|superalpha\s*灵感|灵感分享"),
    ("template_share",    r"模板|template"),
    ("code_share",        r"代码分享|代码优化|脚本分享"),
    ("experience",        r"经验|心得|学习笔记|日记|日常生活贴|记录"),
    ("consultant_lead",   r"community\s*leader|顾问"),
    ("tool_mcp",          r"\bmcp\b|工具|automation|自动化"),
    ("question",          r"请问|求助|如何|怎么|为什么|为何|\?$|？$"),
    ("paper_research",    r"论文|paper|研报|whitepaper|abstract|hypothesis"),
    ("competition",       r"比赛|iqc|竞赛|competition"),
    ("dataset_intro",     r"数据集|dataset|datafield"),
]

# 互动度阈值（实测 comments_count median=0, p95=23）
HIGH_ENGAGEMENT_COMMENTS = 5
TOP_ENGAGEMENT_COMMENTS = 20

SHORT_COMMENT_CHARS = 800       # < 该值，整条评论一个 chunk
SHORT_POST_CHARS = 1500         # < 该值，整篇 post 一个 chunk
LONG_SEG_MAX_CHARS = 1000       # 长评论结构切分时的段最大字符
LONG_POST_SEG_MAX_CHARS = 1200  # 长帖结构切分时的段最大字符
REPLY_CTX_SHORT_CHARS = 300     # < 该值且有 replies_to 触发上下文拼接
MIN_FRAGMENT_CHARS = 50         # < 该值的非代码碎片在 dry_run 中标记为低质量
MAX_CODE_CHUNK_CHARS = 6000     # code_block chunk 超出按行二次切（BGE-M3 max_model_len=8192 token ≈ 20k 字符，留余量）


def summary() -> str:
    return (
        f"provider={EMBEDDING_PROVIDER} dense_dim={DENSE_DIM} "
        f"sparse={SUPPORTS_SPARSE} qdrant={QDRANT_HOST}:{QDRANT_PORT}/{QDRANT_COLLECTION} "
        f"data_root={DATA_ROOT}"
    )

# WorldQuant BRAIN 论坛 + 知识库 — RAG 索引指南

> 实测后修订版。基于 `wq-rag/wq-doc-forum/` 抓好的 JSON 数据，
> 切分 → BGE-M3 嵌入（dense + sparse）→ 索引到远程 Qdrant，
> 用 Fusion (RRF) 做原生混合检索。

---

## 0. 范围与技术选型（已实测锁定，勿改）

**做什么**：把 `wq-doc-forum` 爬好的本地 JSON（论坛 + 教程）切分成高质量 chunk，
用 **自托管 BGE-M3 服务** 同时拿 dense + sparse 向量，索引进 **远程 Qdrant**，
用 Qdrant Fusion API（RRF）做混合检索。

**不做什么**：不爬数据、不做 LLM 生成式问答 Web 服务、不索引 datasets/operators（本期范围之外）。

**技术选型（已实测）**:

| 组件 | 选择 | 备注 |
|------|------|------|
| 向量库 | Qdrant @ `<qdrant-host>:6333` | 命名双向量（dense + sparse） |
| 嵌入主力 | **BGE-M3** @ `<bge-m3-host>:<port>/pooling` | vLLM 自托管，单请求拿 dense (1024) + sparse |
| 嵌入备选 | doubao / voyage / openai | 仅 dense；`.env` 切换；本次不启用 |
| 结构化存储 | **跳过 Postgres** | 数据已是 JSON，直读避免 ETL |
| 数据源 | `wq-doc-forum/data/forum/posts/*.json` + `data/kb/tutorials/*.json` | 爬虫产物 |

**关键设计决策（实测后）**:

1. **BGE-M3 单次返回 dense + sparse**，Qdrant 直接配命名双向量 → 真正的原生混合检索。
   实测 sparse 在算子/字段精确名（`ts_corr`/`trade_when`/`prod_correlation`）召回上显著优于纯 dense。
2. **跳过 Postgres**：爬虫产物已是 JSON，无需多一层 ETL；增量靠 source_path 的 mtime+size hash。
3. **batch_size = 64**（BGE-M3 实测 ≥128 安全，64 是稳定吞吐甜区，~50 条/s）。
4. **代码块超 6000 字符截断内联**（BGE-M3 max_model_len=8192 token），完整代码保留在独立 `code_block` chunk。

**论坛数据实际结构（实测，比原指南假设少）**:

| 字段 | 论坛 post | 论坛 comment | tutorial section |
|---|---|---|---|
| ID | **文件名** 当 post_id | `{post_id}_c{idx}` 生成 | tutorial id + section_index |
| body | 纯文本 + Markdown 围栏 | 同 | 含结构化块（HEADING/TEXT/CODE/SIMULATION_EXAMPLE/TABLE/EQUATION/IMAGE） |
| 元数据 | `title, author, votes, date, comments[]` | `author, body, date` | `id, title, category, content[]` |
| **不存在** | url / author_badges / post_id | comment_id / replies_to / vote_count | — |

→ parser **不做 HTML→MD**（论坛 body 已是文本+围栏），只做"围栏规整 + 元数据抽取"；
  `replies_to` 从评论正文 `@XX12345` 推断；tutorial 的 TEXT 块嵌的是 HTML，用 stdlib html.parser 抽纯文本。

---

## 1. 目录结构

```
wq-rag/
├── BRAIN_RAG_INDEX_GUIDE.md         # 本文件
├── wq-doc-forum/                    # 爬虫（已存在，不在本范围内）
│   └── data/
│       ├── forum/posts/{id}.json    # 1416 帖
│       └── kb/tutorials/{id}.json   # 77 教程
└── brain_index/                     # 本期交付
    ├── .env / .env.example          # 端点 + 凭据
    ├── requirements.txt
    ├── config.py                    # 配置中心（provider 切换、维度映射、SA 词表、切分阈值）
    ├── parser.py                    # 围栏规整 + 元数据抽取（不含 HTML→MD）
    ├── chunker.py                   # 切分核心
    ├── embedder.py                  # 四 provider 统一封装（BGE-M3 主、其他备）
    ├── setup_qdrant.py              # 建集合（命名双向量）
    ├── indexer.py                   # 切分→嵌入→upsert，支持增量
    ├── sources/
    │   ├── forum_loader.py          # data/forum/posts/*.json → ForumPost/Comment
    │   └── tutorials_loader.py      # data/kb/tutorials/*.json → TutorialSection
    └── scripts/
        ├── dry_run_chunks.py        # 切分质量验收（无网络）
        ├── run_index.py             # 一键：setup + index
        └── smoke_search.py          # Fusion RRF 混合检索
```

---

## 2. Phase 1 — 基础设施

### 任务 1.0 — conda 环境

后续所有命令均在 conda `brain` 环境（Python 3.11.15）下执行。

```bash
# 已存在则跳过创建
conda env list | grep -q '^brain' || conda create -n brain python=3.11 -y

# 推荐显式激活，便于在 shell 里直接 python xxx
conda activate brain
cd wq-rag/brain_index

# 安装依赖
pip install -r requirements.txt

# 一次性配置 .env
[ ! -f .env ] && cp .env.example .env

# 验证配置加载
python -c "import config; print(config.summary())"
```

替代：不激活也行，用 `conda run -n brain python scripts/dry_run_chunks.py`。

### 任务 1.1 — `requirements.txt`

```
qdrant-client>=1.12.0,<1.20.0
httpx>=0.27.0
python-dotenv>=1.0.0
tenacity>=9.0.0

# 仅在切到对应 provider 时安装
# voyageai>=0.3.0
# openai>=1.40.0
```

> 注意：BGE-M3 走 HTTP `/pooling`，**不需要** voyageai/openai/FlagEmbedding/torch。

### 任务 1.2 — `.env.example`

见仓库 `.env.example`。关键变量：

```bash
EMBEDDING_PROVIDER=bgem3
BGEM3_BASE_URL=http://<bge-m3-host>:<port>
QDRANT_HOST=<qdrant-host>
QDRANT_PORT=6333
QDRANT_COLLECTION=brain_kb_bgem3
DATA_ROOT=../wq-doc-forum/data
```

### 任务 1.3 — `config.py`

集中所有配置。要点：
- `EMBEDDING_PROVIDER` 切换 → 自动选 `DENSE_DIM`（bgem3:1024 / doubao:1024 / voyage:1024 / openai:3072）
- `SUPPORTS_SPARSE = (provider == "bgem3")`
- 切分阈值：`SHORT_COMMENT_CHARS=800 / SHORT_POST_CHARS=1500 / MAX_CODE_CHUNK_CHARS=6000`
- `SA_OPERATORS` 词表（30+ 已知算子/字段，可按需扩充）

### ✅ 验收检查 Phase 1

```bash
python -c "import config; print(config.summary())"
# 期望输出：provider=bgem3 dense_dim=1024 sparse=True qdrant=<host>:6333/brain_kb_bgem3 ...

# 端点连通性（不发嵌入，只 health check）
python -c "
import httpx, config
r = httpx.get(f'{config.BGEM3_BASE_URL}/health', timeout=5)
print('BGE-M3 health:', r.status_code)
r = httpx.get(f'http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/readyz', timeout=5)
print('Qdrant readyz:', r.status_code)
"
```

---

## 3. Phase 2 — 数据加载（无清洗）

**判断分支**：
- 如果 `wq-doc-forum/data/forum/posts/` 有 JSON 文件 → 直接跑 loaders。
- 如果还没爬数据 → 先到 `wq-rag/wq-doc-forum/` 跑 `python forum_sync.py` 和 `python kb_sync.py`。

### 任务 2.1 — `sources/forum_loader.py`

读 `data/forum/posts/*.json`，规范化成 dataclass：
- `ForumPost(post_id, title, author, body, votes, date, total_comments, source_path, extra, comments)`
- `ForumComment(post_id, comment_id, author, body, date, index, source_path, extra)`

`post_id` 用文件名（stem）；`comment_id` 用 `{post_id}_c{idx}`。

### 任务 2.2 — `sources/tutorials_loader.py`

读 `data/kb/tutorials/*.json`，按 `HEADING` 块切 `TutorialSection`：

各类 content block 渲染规则（实测发现的真实类型分布）：
- `HEADING`: `{level, content}` → `## title`
- `TEXT`: HTML 字符串 → 用 stdlib `html.parser` 抽纯文本（**不引入 bs4**）
- `IMAGE`: `{title, url, ...}` → `![title](url)`
- `TABLE`: `{data: [[rows]]}` → markdown 表
- `SIMULATION_EXAMPLE`: `{settings, type, regular}` → settings + 表达式代码块（核心资产）
- `EQUATION`: LaTeX 字符串 → `$$...$$`

### ✅ 验收检查 Phase 2

```bash
python3 -c "
from sources import load_forum, load_tutorials
posts = list(load_forum())
print(f'forum: {len(posts)} posts, {sum(len(p.comments) for p in posts)} comments')
secs = list(load_tutorials())
print(f'tutorials: {len(secs)} sections across {len(set(s.tutorial_id for s in secs))} tutorials')
"
# 期望：forum: 1416 posts, 7171 comments
# 期望：tutorials: 360 sections across 77 tutorials
```

---

## 4. Phase 3 — 切分（Chunking）【质量核心】

### 切分决策树

```
代码块（含 ≥2 SA 算子） → 独立 code_block chunk + 周围 ≤300 字"来源上下文"
                          超 6000 字符 → 按行二次切，标 (part k/N)
短评论 (<800 字)         → 整条 comment chunk
长评论 (≥800 字)         → extract_and_isolate_code → chunk_by_structure → restore_code
短帖 (<1500 字)          → 整篇 post chunk
长帖 (≥1500 字)          → 同上（不加 title 前缀；indexer 统一注入元数据头）
短回复(<300字)+replies_to → 拼接被回复者正文做上下文
tutorial section         → 长则切短则整；含 SIMULATION_EXAMPLE 额外独立 code_block
```

> **重要：chunker 不在 body 文本里塞 title/author 前缀** —— 这些来自 metadata，由
> `indexer._build_embedding_text()` 统一拼接为 `[主题: ...]\n[作者: ...]\n\n{body}` 形式。
> 这样保证不重复，且后续切 provider 时元数据头一致可控。

### 任务 3.1 — `parser.py`

1. `normalize_fences(text)` —— toggle 状态机，让所有 ``` 围栏前后都有换行；保留语言标签（如 ```python）紧贴开 fence。
2. `extract_code_blocks(text)` —— 抽出所有 ≥20 字符的代码块。
3. `extract_operators / regions / sharpe / sa_type / replies_to` —— 正则抽 SA 元数据。
4. `enrich_post / enrich_comment / enrich_tutorial` —— 顶层 in-place enrich。

**关键 case**：论坛实际 body 形如 `...影响效率。\`\`\`@mcp.tool()...` —— 围栏紧贴中文无换行；`normalize_fences` 修正后变成 `...影响效率。\n\`\`\`\n@mcp.tool()...`。

### 任务 3.2 — `chunker.py`

核心组件：
- `extract_and_isolate_code(md)`：用 `[CODE_BLOCK_N]` 占位符替换代码
- `chunk_by_structure(text, max_chars)`：按 `类别:` / `Selection Expression:` / `## heading` / `第N步` 等结构标记切
- `restore_code(text, blocks, max_inline)`：还原占位符；**超 max_inline 的代码内联截断**（完整内容已在独立 code_block chunk）
- `split_long_code(code, max_chars)`：按行二次切超长代码
- `is_selection_expression(code)`：≥2 SA 算子 → 判为表达式
- `merge_reply_chains(comments)`：短回复（<300 字 + replies_to）拼接被回复者正文

### ✅ 验收检查 Phase 3

```bash
python3 scripts/dry_run_chunks.py
```

期望输出（基于当前数据）:
```
[chunk] total ~17.6k chunks
by chunk_type: post(700) + comment(6903) + post_segment(8139)
               + comment_segment(1040) + code_block(278) + tutorial_section(559)
length min=1 mean=629 median=575 p95=1200 max=7536  ✅ BGE-M3 安全
BROKEN ``` 不成对: 0
operators_mentioned 命中率: ~40%
total chars ≈ 11M ≈ 4.4M tokens
BGE-M3 估算: ~6 min @ 50 req/s (batch=64)
```

**排错**：
- 出现 `max > 8000`：超长代码内联截断没生效，检查 `restore_code` 的 `max_inline_chars` 参数
- `BROKEN ≠ 0`：parser 的 `normalize_fences` 没把围栏前后补换行
- 碎片占比 >20%：调整 `MIN_FRAGMENT_CHARS` 或在 `chunk_by_structure` 里更激进合并

---

## 5. Phase 4 — 嵌入（BGE-M3 主力）

### 任务 4.1 — `embedder.py`

统一抽象 `EmbedResult { dense, sparse, usage_tokens, provider }`。
- `_BGEM3.embed(texts)` → 调 `POST /pooling`，payload 形如：
  ```json
  {"task": "plugin", "data": {"input": ["..."], "return_tokens": false}}
  ```
  返回 `data.data[].dense_embedding` 和 `sparse_embedding[{token_id, weight}]`，
  转成 Qdrant 期望的 `{"indices": [...], "values": [...]}`。
- `_Doubao.embed` → OpenAI 兼容 `/embeddings`，max batch = 10，`dimensions=1024` 降维。
- `_Voyage / _OpenAI` → 用各自官方 SDK，dense-only，sparse 全 None。

`recommended_batch_size()` 返回各 provider 的实测安全值：bgem3=64 / doubao=10 / voyage=128 / openai=256。

### ✅ 验收检查 Phase 4

```bash
python3 -c "
from embedder import get_embedder
e = get_embedder()
r = e.embed(['测试中文 turnover ts_corr 表达式', 'second text'])
print('provider:', r.provider)
print('dense dim:', len(r.dense[0]))
print('sparse terms (first):', len(r.sparse[0]['indices']) if r.sparse[0] else None)
"
```

期望：`dense dim: 1024 / sparse terms: ~7-15`

---

## 6. Phase 5 — 建集合 + 索引

### 任务 5.1 — `setup_qdrant.py`

BGE-M3 主力时建命名双向量：

```python
client.create_collection(
    collection_name="brain_kb_bgem3",
    vectors_config={"dense": VectorParams(size=1024, distance=COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
)
```

dense-only provider（doubao/voyage/openai）则只建 `{"dense": ...}`，不配 sparse。

payload 索引（覆盖检索、过滤、聚合三类用途）：

| 类型 | 字段 |
|---|---|
| 来源/身份 | `source / post_id / comment_id / tutorial_id / author_id / chunk_type / category` |
| SA 元数据 | `sa_type / operators_mentioned / regions / reported_sharpe / has_code` |
| 时效性 | `recency_tag / year / created_at_ts / age_days` |
| 标签 | `tags`（keyword 数组） |
| 互动度 | `vote_count / total_comments` |
| 增量 | `source_hash` |

### 任务 5.2 — `indexer.py`

增量：
1. `_source_hash(path) = sha1(path|mtime|size)[:16]`
2. 启动时 `_existing_hashes(client)` scroll 全库 → `{point_id: source_hash}`
3. 当前 chunk 的 `point_id = uuid5(chunk_type|parent_id|chunk_index)`
4. 已存在且 source_hash 一致 → 跳过

upsert 时同时带 dense 和 sparse：

```python
vector = {"dense": dense_vec}
if SUPPORTS_SPARSE and sparse_vec:
    vector["sparse"] = SparseVector(indices=..., values=...)
```

### ✅ 验收检查 Phase 5

```bash
python3 scripts/run_index.py             # 增量
# 或
python3 scripts/run_index.py --recreate  # 谨慎：删集合重建
```

完成后：
```bash
python3 -c "
from qdrant_client import QdrantClient
import config
c = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
info = c.get_collection(config.QDRANT_COLLECTION)
print(f'points: {info.points_count}')
print(f'vectors_config: {info.config.params.vectors}')
print(f'sparse_vectors_config: {info.config.params.sparse_vectors}')
"
```

期望：`points: ~17600`，命名双向量结构正确。

---

## 7. Phase 6 — 混合检索冒烟测试

### 任务 6.1 — `scripts/smoke_search.py`

走 Qdrant Fusion API (RRF) 原生混合：

```python
prefetch = [
    Prefetch(query=dense_q, using="dense", limit=top_k * 2),
    Prefetch(query=SparseVector(indices=..., values=...), using="sparse", limit=top_k * 2),
]
client.query_points(
    collection_name="brain_kb_bgem3",
    prefetch=prefetch,
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k,
)
```

### 典型查询（人工核验前 5）

```
- "如何选取低相关的因子做SuperAlpha组合"   → 应召回中文同主题帖 + 含相关性算子的 code chunk
- "ts_corr 算子的常见用法"                  → sparse 应精准命中 ts_corr 代码 chunk
- "turnover 怎么分段降低换手率"             → 中英混合，dense + sparse 协同
- "Selection 表达式 trade_when 用法"        → 含 trade_when 的代码 chunk 应优先
- "alpha submission 的流程是什么"           → 应召回 tutorial chunk
```

---

## 7.5. 时效性 + 内容标签

论坛帖有时效性，搜"最近一年的内容"很常见；同时不同主题（灵感 / 模板 / 经验 / 工具）应能精确过滤。
为此 chunk payload 包含两组字段。**这些是元数据，不进嵌入文本，零额外 token 成本。**

### 时效性字段（每个 chunk 都有，tutorial 除外）

| 字段 | 类型 | 例值 |
|---|---|---|
| `created_at` | string | `"2025-10-28T15:38:04Z"` |
| `created_at_ts` | integer | `1761662284` (epoch seconds) |
| `year` | keyword | `"2025"` |
| `recency_tag` | keyword | `fresh` (≤180d) / `recent` (≤2y) / `older` (≤5y) / `archived` (>5y) |
| `age_days` | integer | `212` |

实测分布：fresh 38% / recent 55% / older 4% / archived 0% / 无日期 3%。

**检索用法**：

```python
# 只看 1 年内
from qdrant_client.http import models as qm
filt = qm.Filter(must=[qm.FieldCondition(
    key="recency_tag", match=qm.MatchAny(any=["fresh", "recent"])
)])

# 或精确时间范围
filt = qm.Filter(must=[qm.FieldCondition(
    key="created_at_ts", range=qm.Range(gte=int(time.time()) - 365*86400)
)])
```

### 内容标签（仅基于 post.title 推断，避免 body 关键词泛命中）

| tag | 规则（不区分大小写） | 命中数 / 占比 |
|---|---|---|
| `alpha_inspiration` | `Alpha\s*灵感` / `SuperAlpha\s*灵感` / `灵感分享` | 683 / 3.9% |
| `template_share` | `模板` / `template` | 1650 / 9.4% |
| `code_share` | `代码分享` / `代码优化` / `脚本分享` | 462 / 2.6% |
| `experience` | `经验` / `心得` / `学习笔记` / `日记` / `日常生活贴` / `记录` | 2071 / 11.8% |
| `consultant_lead` | `Community\s*Leader` / `顾问` | 1800 / 10.2% |
| `tool_mcp` | `\bMCP\b` / `工具` / `自动化` | 1684 / 9.6% |
| `question` | `请问` / `求助` / `如何` / `怎么` / `?$` / `？$` | 961 / 5.5% |
| `paper_research` | `论文` / `paper` / `研报` / `whitepaper` | 102 / 0.6% |
| `competition` | `比赛` / `IQC` / `竞赛` | 882 / 5.0% |
| `dataset_intro` | `数据集` / `dataset` / `datafield` | 317 / 1.8% |
| `has_code` | body 含 ``` 围栏 | 1688 / 9.6% |
| `high_engagement` | `total_comments ≥ 5` | 4634 / 26.3% |
| `top_engagement` | `total_comments ≥ 20` | 4654 / 26.4% |
| `tutorial` | tutorial 来源 | 574 / 3.3% |
| `has_simulation_example` | tutorial 含 SIMULATION_EXAMPLE 块 | 53 / 0.3% |

**关键设计**：
- **comment 继承父帖的内容 tag**（不重新匹配自身 body），便于"找该主题下的讨论"。
- comment 自身仅独立判定 `has_code`。
- 规则只看 title，body 的关键词通过 `operators_mentioned` / `regions` / `sa_type` 等具体字段反映。

**检索用法**：

```python
# 只看 Alpha 灵感分享类的高互动帖
filt = qm.Filter(must=[
    qm.FieldCondition(key="tags", match=qm.MatchValue(value="alpha_inspiration")),
    qm.FieldCondition(key="tags", match=qm.MatchValue(value="high_engagement")),
])
```

### 调标签规则 / 切分阈值后的全库回填

`config.TAG_RULES` / 切分阈值 / parser 逻辑等改动后，**必须 bump `config.INDEX_VERSION`**
（如 v1.0 → v1.1）。机制：`source_hash = sha1(path|mtime|size|INDEX_VERSION)`，
版本号变 → 所有 chunk 的 source_hash 都变 → 增量识别为"已过期"重嵌一遍。

```bash
# 改了规则 / 切分阈值 / parser
vim config.py    # 例如把 INDEX_VERSION 从 v1.0 改成 v1.1

# 普通增量重跑就能全库回填
python scripts/run_index.py
```

若只想刷新 payload tag 不重嵌向量（省时省钱）：单独写 `scripts/refresh_tags.py` 用
`client.set_payload(points=..., payload={"tags": [...]})`，不动 vector。本期未实现，
按需补。

---

## 8. 调优清单（索引跑通后再做）

1. **碎片合并**：若 dry_run 报告 fragments >15%，把短评论（<50 字）合并到上一条同帖评论。
2. **algorithm tuning**：Qdrant RRF 的默认参数已够好；如果某类查询召回不准，可改用 `Fusion.DBSF`（distribution-based）或加 query-level filter。
3. **dense / sparse 权重**：RRF 是 rank-based 不需要调权；如果想 weighted hybrid，改用 `query=NearestQuery` + 自实现 score 融合。
4. **Reranker 二阶段**：BGE-M3 服务暴露 `/rerank` 端点（vLLM 内置），召回 top-50 后用 reranker 精排到 top-5，对复杂 query 提升明显。
5. **operators 词表扩充**：从已索引 chunk 的 `operators_mentioned` 统计高频未命中 token，回填到 `config.SA_OPERATORS`。
6. **datasets 接入**：本期未做。需先到 `wq-doc-forum/` 跑 `kb_sync.py --datasets-only`，再加 `sources/datasets_loader.py`。

---

## 9. 给后续执行者的纪律

1. **严格按 Phase 顺序**，每个「✅ 验收检查」必须实际运行通过才能进入下一个。
2. **Phase 3 跑 `dry_run_chunks.py`** 前不要碰嵌入和 Qdrant。
3. **每改一次 chunker 都重跑 dry_run**，确认断言通过且 max chunk ≤ 7600。
4. **集合是双向量**：BGE-M3 主力时 dense + sparse；切到 dense-only provider 时**必须换集合名**（避免维度/结构冲突）。
5. **不要把 `code_block` chunk 当成"普通文本"切**——这类 chunk 应保持代码完整或按行二次切。
6. **API 调用区分 input_type**：BGE-M3 对称（参数被忽略）；voyage 必须区分 `document/query`。
7. 遇验收失败先查对应 Phase 的「排错」，不要绕过验收硬往下。

---

## 附：常见错误速查表

| 现象 | 原因 | 解决 |
|------|------|------|
| 维度不符 | provider 与集合 dim 不一致 | 删集合重建（`run_index.py --recreate`），或对齐 `EMBEDDING_PROVIDER` |
| upsert vector 格式错 | 命名向量 dict 字段名错 | BGE-M3 主力必须用 `{"dense": [...], "sparse": SparseVector(...)}` |
| BGE-M3 `model does not exist` | vLLM 期望完整路径 | 用 `/v1/models` 看真实 id，或省略 `model` 字段（单模型部署时） |
| Doubao `Embeddings API input limit exceeded: max 10` | 批量超 10 | `recommended_batch_size("doubao") == 10` |
| BGE-M3 sparse 缺失 | 用了 `/v1/embeddings` 而非 `/pooling` | 必须调 `/pooling` 才能拿 sparse |
| chunk max > 8000 | 超长代码 inline 没截断 | 检查 `restore_code(max_inline_chars=MAX_CODE_CHUNK_CHARS)` |
| 中文召回差 | 用了纯英文模型 | 必须用 BGE-M3 或 doubao/voyage 多语言模型 |
| 重跑全量重嵌 | 没用增量 | 默认是增量；不要加 `--full` |
| `query_points` 报错 | Qdrant 版本旧 | 升级到 ≥1.12（已在 requirements） |
| `ModuleNotFoundError: tenacity/qdrant_client` | 没在 brain env 装依赖 | `conda activate brain && pip install -r requirements.txt` |
| `httpx.ConnectError` 连 BGE-M3 | 端点不通或没 VPN | curl 直接验证 `$BGEM3_BASE_URL/v1/models`；必要时设 `http_proxy` |

完成所有 Phase 后，论坛 + tutorials 将以双向量形式索引在 `brain_kb_bgem3` 集合中，
可用 `smoke_search.py` 验证混合检索效果，下游接 LLM 即可做检索增强问答。

---

## 附 B：首次导入实测验收记录（2026-05-21）

实际跑通的基线数据，后续改动可对标。

### 数据规模

| 维度 | 实测 |
|---|---|
| 总 chunks | **17,619** |
| 总字符 | 11,087,181 |
| 全量索引耗时 | **447.1 s（7.5 分钟）** |
| 平均吞吐 | **39.4 chunks/s**（BGE-M3 batch=64） |

### chunk 分布

```
by source     : forum_post=9087, forum_comment=7958, tutorial=574
by chunk_type : post=700, post_segment=8139, comment=6903,
                comment_segment=1040, code_block=278, tutorial_section=559
length        : min=1, mean=629, median=575, p95=1200, max=7536  (≤BGE-M3 8K token 安全)
broken code   : 0  ✅
```

### 时效性 / 标签覆盖（与 dry_run 完全对齐）

```
recency: fresh=6664, recent=9664, older=717, archived=0, 无日期=574(tutorial)
year:    2023=213, 2024=2258, 2025=10765, 2026=3809
tags:    top_engagement=4654, high_engagement=4634, experience=2071,
         consultant_lead=1800, has_code=1688, tool_mcp=1684, template_share=1650,
         question=961, competition=882, alpha_inspiration=683, tutorial=574,
         code_share=462, dataset_intro=317, paper_research=102,
         has_simulation_example=53
```

### 检索质量抽样（Fusion RRF 混合检索）

5 类典型 query 跑下来 top-5 全部高度相关：

| Query | top-1 score | 命中说明 |
|---|---|---|
| 如何选取低相关的因子做SuperAlpha组合 | 0.50 | 顾问帖《SuperAlpha 理论与实操指南》 |
| turnover 怎么分段降低换手率 | 0.83 | 直接讨论 turn / sharpe / fit / margin |
| ts_corr 算子的常见用法 | 0.64 | `ts_corr(illiquidity,amihud,42)` 代码片段 |
| Selection 表达式 trade_when 用法 | 0.64 | 完整 trade_when 模板 |
| alpha submission 的流程是什么 | 0.58 | submit_alpha / submission status 代码 |

### 网络拓扑

当前默认 **直连可达**：`.env` 用真实 `QDRANT_HOST` / `PORT=6333`。

**应急路径**：观察到云防火墙偶发拦截（TCP 通但 HTTP 应用层丢包），如再次出现：

```bash
# 起 SSH 隧道（替换 <user>@<qdrant-host>）
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -fN \
    -L 16333:127.0.0.1:6333 -L 16334:127.0.0.1:6334 <user>@<qdrant-host>

# 临时切换（env 变量覆盖 .env，不动文件）
export QDRANT_HOST=127.0.0.1
export QDRANT_PORT=16333
```

恢复直连：`pkill -f "ssh.*16333" && unset QDRANT_HOST QDRANT_PORT`。


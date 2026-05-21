# WorldQuant BRAIN — RAG 知识库

把 BRAIN 论坛 + 官方教程 + 算子文档转成可检索的向量知识库（dense+sparse 混合检索），供下游 LLM 做 RAG 问答。

> **三份文档分工**：
> - 本文件 — 操作手册（部署 / 使用 / 维护）
> - [`BRAIN_RAG_INDEX_GUIDE.md`](./BRAIN_RAG_INDEX_GUIDE.md) — 实现指南（架构 / 设计决策 / 分阶段细节）
> - [`wq-doc-forum/README.md`](./wq-doc-forum/README.md) — 爬虫使用文档

---

## 一、架构

```
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ wq-doc-forum/    │ ────▶ │ brain_index/     │ ────▶ │ Qdrant 集合      │
│ 爬虫 (Playwright)│ JSON  │ 切分→嵌入→索引  │ vec   │ brain_kb_bgem3   │
│ 论坛/教程/算子   │       │                  │       │ dense+sparse     │
└──────────────────┘       └──────────────────┘       └──────────────────┘
        ↑                          │                          │
   定时器/手动                     ↓                          ↓
                          BGE-M3 (vLLM /pooling)        下游 LLM 检索
                          单次返回 dense+sparse         Fusion RRF 混合
```

### 组件依赖

| 组件 | 类型 | 配置项（.env） |
|---|---|---|
| BGE-M3 服务 | 自托管 vLLM，暴露 `/pooling`、`/v1/models`、`/v1/embeddings` | `BGEM3_BASE_URL` |
| Qdrant | 向量库，需开 HTTP 6333（可选 gRPC 6334） | `QDRANT_HOST` / `QDRANT_PORT` |
| 数据源 | `wq-doc-forum` 爬虫的 JSON 产物 | `DATA_ROOT`（相对路径）|
| 嵌入备选 | Doubao / Voyage / OpenAI（仅 dense） | 各自 `*_API_KEY` |

### 关键数字（默认配置）

| 项 | 值 |
|---|---|
| Dense 向量维度 | 1024（BGE-M3） |
| 嵌入吞吐参考 | ~39 chunks/s（BGE-M3 batch=64） |
| 单次最长 chunk | 7,600 字符（≤ BGE-M3 8K token 安全余量） |
| 集合命名 | `brain_kb_<provider>`（切 provider 必须换集合，维度可能不同） |

---

## 二、首次部署

### 0. 前置

- conda 环境 `brain`（Python 3.11+）
- 能访问 BGE-M3 服务（HTTP `/pooling`）
- 能访问 Qdrant 服务（HTTP 6333）
- 爬虫已跑过，`wq-doc-forum/data/` 下有 JSON

### 1. 装依赖 + 配 `.env`

```bash
conda activate brain
cd wq-rag/brain_index

pip install -r requirements.txt

cp .env.example .env
$EDITOR .env    # 填入真实 BGEM3_BASE_URL / QDRANT_HOST / 可选 API keys
```

`.env` 已在 `.gitignore`，不会进仓库。

### 2. 连通性自检

```bash
python -c "
import httpx, config
print('BGE-M3:', httpx.get(f'{config.BGEM3_BASE_URL}/v1/models', timeout=5).status_code)
print('Qdrant:', httpx.get(f'http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/readyz', timeout=5).status_code)
"
# 期望两个都 200。不通看 §六 排错。
```

### 3. 切分质量自检（离线、不嵌入）

```bash
python scripts/dry_run_chunks.py
```

期望：`✅ dry_run done — all assertions passed`。看到 `BROKEN ``` 不成对: 0`、`max ≤ 7600`、tag 分布合理。

### 4. 建集合 + 全量索引

```bash
python scripts/run_index.py --recreate
```

参考耗时：1.7 万 chunks 约 7-8 分钟。完成后 `[index] ✅ done`。

### 5. 冒烟检索

```bash
python scripts/smoke_search.py
```

人工核验 top-5 是否与 query 相关。

---

## 三、网络访问

### 默认：直连 Qdrant

`.env` 直接填 Qdrant 主机和端口，HTTP 客户端走 6333（REST），gRPC 走 6334（可选）。


---

## 四、检索 API（给 LLM / 应用用）

### 1. 最简模板（dense + sparse Fusion RRF）

```python
# 在 brain_index/ 下运行，brain conda env
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import config
from embedder import get_embedder

client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=30)
embedder = get_embedder()

def search(query: str, top_k: int = 10, filt: qm.Filter | None = None):
    r = embedder.embed([query], input_type="query")
    prefetch = [qm.Prefetch(query=r.dense[0], using="dense", limit=top_k * 2, filter=filt)]
    sp = r.sparse[0]
    if sp:
        prefetch.append(qm.Prefetch(
            query=qm.SparseVector(indices=sp["indices"], values=sp["values"]),
            using="sparse", limit=top_k * 2, filter=filt,
        ))
    return client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=prefetch,
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points
```

### 2. Payload 过滤示例

```python
import time

# 只查 Alpha 灵感 + 近 2 年 + 高互动
filt = qm.Filter(must=[
    qm.FieldCondition(key="tags", match=qm.MatchValue(value="alpha_inspiration")),
    qm.FieldCondition(key="recency_tag", match=qm.MatchAny(any=["fresh", "recent"])),
    qm.FieldCondition(key="tags", match=qm.MatchValue(value="high_engagement")),
])

# 只查含 ts_corr 算子的代码块
filt = qm.Filter(must=[
    qm.FieldCondition(key="operators_mentioned", match=qm.MatchValue(value="ts_corr")),
    qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value="code_block")),
])

# 只查官方教程
filt = qm.Filter(must=[
    qm.FieldCondition(key="source", match=qm.MatchValue(value="tutorial")),
])

# 时间窗口（最近 90 天）
filt = qm.Filter(must=[
    qm.FieldCondition(key="created_at_ts",
                      range=qm.Range(gte=int(time.time()) - 90 * 86400)),
])
```

### 3. Payload 字段速查

| 字段 | 类型 | 说明 |
|---|---|---|
| `body_md` | str | chunk 原文 |
| `source` | str | `forum_post` / `forum_comment` / `tutorial` |
| `chunk_type` | str | `post / post_segment / comment / comment_segment / code_block / tutorial_section` |
| `title` | str | 帖子标题 / tutorial 标题 |
| `author_id` | str | 论坛 ID |
| `created_at` | str | ISO 时间 |
| `created_at_ts` | int | epoch seconds（用于范围过滤） |
| `recency_tag` | keyword | `fresh` (≤6m) / `recent` (≤2y) / `older` (≤5y) / `archived` |
| `year` | keyword | `"2024" / "2025"` |
| `tags` | str[] | 内容标签数组（见下） |
| `operators_mentioned` | str[] | 命中的 SA 算子 |
| `regions` | str[] | `USA / CHN / EUR` 等 |
| `sa_type` | str | `selection / combo / regular` |
| `reported_sharpe` | float | 正文里抽到的 Sharpe |
| `has_code` | bool | 是否含代码块 |
| `vote_count` | int | 帖子点赞数 |
| `total_comments` | int | 帖子评论数（仅 post chunk） |

**可用 tags**（基于标题规则推断，可在 `config.TAG_RULES` 扩展）：

```
内容类型 : alpha_inspiration / template_share / code_share / experience
         / consultant_lead / tool_mcp / question / paper_research
         / competition / dataset_intro / tutorial
特征类型 : has_code / has_simulation_example
互动级别 : high_engagement (≥5 评论) / top_engagement (≥20 评论)
```

### 4. RAG 端到端模板

```python
def rag_answer(question: str, llm_call) -> str:
    hits = search(question, top_k=8)
    context = "\n\n---\n\n".join(
        f"[{p.payload['source']} | {p.payload.get('title','')}]\n{p.payload['body_md']}"
        for p in hits
    )
    return llm_call(
        system="你基于 WorldQuant BRAIN 论坛和教程回答。引用时标注 [post_id] 或 [tutorial_id]。",
        user=f"问题：{question}\n\n相关资料：\n{context}",
    )
```

---

## 五、日常维护

### 1. 爬新数据 + 增量入库

```bash
# 拉新论坛 / 教程
cd wq-rag/wq-doc-forum
python forum_sync.py     # 论坛增量（日级）
python kb_sync.py        # 教程 / 算子（周级）

# 入库增量（只处理新增 / 改动的源文件）
cd ../brain_index
python scripts/run_index.py
```

增量原理：`source_hash = sha1(path|mtime|size|INDEX_VERSION)`，hash 一致即跳过。

### 2. 改规则后全库回填

修改 `config.py` 的 `TAG_RULES` / `SA_OPERATORS` / 切分阈值后：

```bash
$EDITOR config.py     # 在文件里 bump INDEX_VERSION，例如 "v1.0" → "v1.1"
python scripts/run_index.py
```

`INDEX_VERSION` 一改，所有 source_hash 都变，增量识别为"已过期"自动重嵌全库。

### 3. 完全重建（慎用）

```bash
python scripts/run_index.py --recreate   # 删集合 + 全量重嵌
```

### 4. 切换嵌入 provider

```bash
$EDITOR .env
# EMBEDDING_PROVIDER=doubao
# QDRANT_COLLECTION=brain_kb_doubao   # 维度不同，必须换集合
python scripts/run_index.py --recreate
```

### 5. 查集合状态

```bash
python -c "
from qdrant_client import QdrantClient
import config
c = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=30, check_compatibility=False)
info = c.get_collection(config.QDRANT_COLLECTION)
print(f'points: {info.points_count}, status: {info.status}')
"
```

---

## 六、常见问题

### Q1：`curl <qdrant>:6333` 0 字节超时但 TCP 通

→ 云防火墙拦截应用层。走 §三 SSH 隧道应急。

### Q2：`httpx.ConnectError` 连 BGE-M3

→ 直接 `curl $BGEM3_BASE_URL/v1/models` 验端点。挂了找服务运维。

### Q3：`ModuleNotFoundError: tenacity / qdrant_client`

→ 不在 brain env 或没装依赖：`conda activate brain && pip install -r requirements.txt`。

### Q4：`Embeddings API input limit exceeded: max 10`

→ 切到 doubao provider 时 batch 太大。doubao 强制 `batch ≤ 10`，indexer 已自动用 `recommended_batch_size()` 适配，确保没硬编码。

### Q5：改了 `TAG_RULES` 重跑 tag 没更新

→ 没 bump `INDEX_VERSION`，增量跳过了未变文件。改 `config.INDEX_VERSION` 后重跑。

### Q6：召回结果漏算子

- 加进 `config.SA_OPERATORS` → bump INDEX_VERSION → 重跑
- 或检索时显式过滤：`operators_mentioned: ts_corr`

### Q7：要强制重新跑某一帖

```bash
touch ../wq-doc-forum/data/forum/posts/<post_id>.json
python scripts/run_index.py
```

### Q8：检索很慢

- 检查 SSH 隧道是不是僵死了（curl 超时）
- 用 gRPC 减少 HTTP 开销：`QdrantClient(host=..., grpc_port=6334, prefer_grpc=True)`
- 减小 prefetch limit（默认 top_k×2）

---

## 七、目录结构

```
wq-rag/
├── README.md                       ← 本文件
├── BRAIN_RAG_INDEX_GUIDE.md        ← 实现指南
├── wq-doc-forum/                   ← 爬虫
│   ├── README.md
│   ├── forum_sync.py / kb_sync.py
│   └── data/
│       ├── forum/posts/{id}.json
│       └── kb/tutorials/{id}.json
└── brain_index/                    ← RAG 索引
    ├── .env.example                ← 模板（无真实凭据，入 git）
    ├── .env                        ← 真实凭据（gitignored）
    ├── .gitignore
    ├── requirements.txt
    ├── config.py                   ← 配置中心 + 阈值 + 词表
    ├── parser.py                   ← 围栏规整 + 元数据抽取
    ├── chunker.py                  ← 切分逻辑
    ├── embedder.py                 ← 四 provider 统一封装
    ├── setup_qdrant.py             ← 建集合（命名双向量）
    ├── indexer.py                  ← 切分→嵌入→upsert，增量
    ├── sources/                    ← 数据加载器
    │   ├── forum_loader.py
    │   └── tutorials_loader.py
    └── scripts/
        ├── dry_run_chunks.py       ← 切分质量自检
        ├── run_index.py            ← 一键索引（setup + index）
        └── smoke_search.py         ← 冒烟检索
```

---

## 八、Cheat Sheet

```bash
conda activate brain
cd wq-rag/brain_index

# 连通自检
python -c "import httpx,config; print(httpx.get(f'{config.BGEM3_BASE_URL}/v1/models',timeout=5).status_code, httpx.get(f'http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/readyz',timeout=5).status_code)"

# 日常增量
cd ../wq-doc-forum && python forum_sync.py && cd ../brain_index && python scripts/run_index.py

# 改规则回填
$EDITOR config.py    # bump INDEX_VERSION
python scripts/run_index.py

# 冒烟检索
python scripts/smoke_search.py

# 集合状态
python -c "from qdrant_client import QdrantClient; import config; c=QdrantClient(host=config.QDRANT_HOST,port=config.QDRANT_PORT,timeout=30,check_compatibility=False); i=c.get_collection(config.QDRANT_COLLECTION); print(f'{i.points_count} points, status={i.status}')"

# 重建（慎用）
python scripts/run_index.py --recreate
```

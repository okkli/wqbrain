# brain-rag-service（MCP + REST 双入口）

WorldQuant BRAIN 知识库的统一对外服务。**同一个进程同时提供**：

| 入口 | 协议 | 用途 |
|---|---|---|
| **REST API** | HTTP/JSON | 给业务系统、agent 框架、其它语言的后端用 |
| **MCP** | streamable-http | 给 Claude Code / Cursor 等 LLM CLI 用 |

两者共享 `retrieval.py` 检索核心 + Qdrant 集合 `brain_kb_bgem3`，逻辑零重复。

---

## 一、端点一览

### REST API（业务接入）

| Method | Path | 用途 |
|---|---|---|
| `POST` | `/v1/search` | 混合检索（dense+sparse Fusion RRF） |
| `POST` | `/v1/fetch` | 按 ref 拉完整原文 |
| `GET`  | `/v1/stats` | 库元信息 + 可用过滤维度 |
| `POST` | `/v1/rag/answer` | **LLM 便利接口**：检索 + 预拼 context + citations（不调 LLM） |
| `GET`  | `/healthz` | 信息型健康（degraded 也返 200）|
| `GET`  | `/readyz` | **K8s readiness 严格**（全 ok 才 200，否则 503） |
| `GET`  | `/livez` | **K8s liveness 宽松**（进程在跑就 200） |
| `GET`  | `/docs` | Swagger UI（可用 `RAG_DISABLE_DOCS=1` 关闭） |
| `GET`  | `/openapi.json` | OpenAPI schema（同上） |

### MCP（LLM CLI 接入）

| Tool | 等价 REST |
|---|---|
| `rag_search` | `POST /v1/search` |
| `rag_fetch` | `POST /v1/fetch` |
| `rag_stats` | `GET /v1/stats` |

**路径**：`http://<host>:<port>/mcp` (streamable-http, 无状态)

---

## 二、启动服务

### 前置

- conda `brain` 环境装 `requirements.txt`
- `../brain_index/.env` 已配置好 BGE-M3 + Qdrant
- Qdrant 集合 `brain_kb_bgem3` 已索引完成

### 启动

```bash
conda activate brain
cd wq-rag/brain_rag_service
pip install -r requirements.txt

# 前台
python server.py

# 后台 + 日志
nohup python server.py > /tmp/brain-rag-service.log 2>&1 &
echo $! > /tmp/brain-rag-service.pid

# 停服务（注意杀 python 而非 conda 包装进程）
pkill -f "python.*server.py"
```

启动后会显示：

```
brain-rag-service starting on http://127.0.0.1:8765
  REST API   : http://127.0.0.1:8765/docs
  MCP        : http://127.0.0.1:8765/mcp
```

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_SVC_HOST` | `127.0.0.1` | 监听地址。改 `0.0.0.0` 公网开放**不带任何鉴权**，必须前置反代 + 防火墙 |
| `RAG_SVC_PORT` | `8765` | 端口 |
| `RAG_SVC_RELOAD` | — | `=1` 启用 dev 热重载（仅开发用） |
| `RAG_MCP_PATH` | `/mcp` | MCP streamable-http 子路径 |
| `RAG_DISABLE_DOCS` | — | `=1` 关闭 `/docs` `/redoc` `/openapi.json`（生产推荐） |
| `RAG_SYSTEM_PROMPT` | — | 覆盖 `/v1/rag/answer` 默认中文 system prompt（业务自定义/英文场景）|

---

## 三、REST API 调用示例

### Python (httpx / requests)

```python
import httpx

BASE = "http://127.0.0.1:8765"

# 1) 检索
r = httpx.post(f"{BASE}/v1/search", json={
    "query": "如何降低 turnover 提高 margin",
    "top_k": 5,
    "filters": {"tags": ["consultant_lead"], "recency": "recent"},
}).json()
for h in r["hits"]:
    print(h["rank"], h["score"], h["ref"], h["title"])

# 2) 拉全文
r = httpx.post(f"{BASE}/v1/fetch", json={
    "ref": "post:18632798681623",
}).json()
print(r["full_content"][:500])

# 3) 库元信息
stats = httpx.get(f"{BASE}/v1/stats").json()
print("可用 tags:", list(stats["tags"].keys()))

# 4) RAG 便利接口（推荐）
ans = httpx.post(f"{BASE}/v1/rag/answer", json={
    "question": "ts_corr 算子怎么用？",
    "filters": {"operators": ["ts_corr"], "has_code": True},
    "top_k": 5,
}).json()
# 直接喂 LLM：
# llm.chat(system=ans["system_prompt"], user=f"问题：{ans['question']}\n\n参考资料：\n{ans['context']}")
print(ans["context"][:500])
print("citations:", [(c["id"], c["ref"]) for c in ans["citations"]])
```

### curl

```bash
# 健康检查
curl -s http://127.0.0.1:8765/healthz | jq

# 检索
curl -s -X POST http://127.0.0.1:8765/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"ts_corr 用法","top_k":3,"filters":{"operators":["ts_corr"]}}' | jq

# RAG 便利接口
curl -s -X POST http://127.0.0.1:8765/v1/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"如何降低 turnover","top_k":3}' | jq '.citations, .context[0:300]'
```

### Node.js / TS

```typescript
const r = await fetch("http://127.0.0.1:8765/v1/rag/answer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "ts_corr 算子用法",
    top_k: 5,
    filters: { operators: ["ts_corr"] },
  }),
});
const data = await r.json();
// data.context, data.citations, data.system_prompt
```

### 其它语言

`GET /openapi.json` 拿到完整 OpenAPI 3 schema，丢给 [openapi-generator](https://github.com/OpenAPITools/openapi-generator) 自动生成任意语言的 client。

---

## 四、`/v1/rag/answer` 推荐流程

这是**最适合业务接入的端点**——一次调用拿到喂 LLM 的全部素材：

```
业务接口
   ↓ (用户问题)
POST /v1/rag/answer
   ↓
{
  "question": "...",
  "context": "[1] xxx\n\n[2] yyy\n\n[3] zzz",   # 预拼好，直接塞 LLM
  "citations": [                                  # 引用对照表
    {"id": 1, "ref": "post:123", "title": "...", "score": 0.83},
    ...
  ],
  "applied_filters": {...},
  "summary": {...},
  "system_prompt": "你是 BRAIN 平台的专家助手..."  # 默认 system prompt
}
   ↓
业务方调 LLM (Doubao / Claude / GPT / 本地 vLLM)
   ↓ 返回带 [n] 引用的答案
LLM 答案 + citations 一起返回给前端
```

`context_format` 可选 `numbered` (默认，适合 LLM 引用) / `markdown` (适合人读) / `json` (适合机器再处理)。

---

## 五、Payload 字段速查（hits / citations）

| 字段 | 类型 | 说明 |
|---|---|---|
| `ref` | str | `post:<id>` / `comment:<id>` / `tutorial:<id>` |
| `source` | str | `forum_post` / `forum_comment` / `tutorial` |
| `chunk_type` | str | `post / post_segment / comment / comment_segment / code_block / tutorial_section` |
| `score` | float | RRF 融合分（0-1） |
| `title` | str | 帖子标题 / tutorial 标题 |
| `author` | str | 论坛 ID |
| `date` | str | YYYY-MM-DD |
| `age` | str | 人类可读 (`约 5 个月前`) |
| `recency_tag` | str | fresh / recent / older / archived |
| `year` | str | `"2025"` 等 |
| `tags` | str[] | 内容标签数组 |
| `operators` | str[] | 命中的 SA 算子 |
| `has_code` | bool | 是否含代码块 |
| `content` | str | chunk 正文（按 `truncate_content` 截断） |
| `truncated` | bool | 仅截断时出现 |
| `fetch_full_with` | str | 仅截断时出现：调用提示 |

**可用 filter 值**：调一次 `GET /v1/stats` 拿全量列表（含 tags / operators / sources / sa_types）。

---

## 六、接入 Claude Code（MCP）

服务起来后：

```bash
claude mcp add --transport http brain-rag http://127.0.0.1:8765/mcp
claude mcp list | grep brain-rag    # 期望 ✓ Connected
```

之后在 Claude Code 里直接调用 `mcp__brain-rag__rag_search` 等工具，**多个 Claude Code 窗口可共享同一进程不阻塞**（`stateless_http=True`）。

---

## 七、调试

### 直接调函数（无网络）

```bash
conda run -n brain --cwd wq-rag/brain_rag_service python -c "
import retrieval, json
r = retrieval.search_hybrid('ts_corr', top_k=3,
    filt=retrieval.build_filter(operators=['ts_corr']))
print(json.dumps(r, ensure_ascii=False, indent=2))
"
```

### Swagger UI（浏览器交互测试）

打开 http://127.0.0.1:8765/docs，所有端点带交互表单。

### MCP Inspector

```bash
npx @modelcontextprotocol/inspector
# 填 http://127.0.0.1:8765/mcp，可视化调 3 个工具
```

---

## 八、常见问题

### Q1：HTTP 起来但 `/healthz` 等返 404

→ 旧进程未杀干净。`pkill -f "python.*server.py"` 后重启。注意 `nohup conda run` 启动的进程要直接 pkill python，杀 conda 包装 PID 没用。

### Q2：MCP 工作但 REST 404

→ `app.mount("/", mcp_app)` 写错挂载路径吞了所有 FastAPI 路由。正确写法 `app.mount("/mcp", mcp_app)` + FastMCP 内部 `streamable_http_path="/"`。

### Q3：错误返回 `{"detail": "..."}`

→ FastAPI 标准。状态码：
- **422** schema 不匹配（空 query / 缺字段 / 越界 / 非法 Literal —— Pydantic 拦截）
- **400** 业务参数错（ref 格式不对等运行时校验）
- **404** 资源不存在（ref 合法但库里没有）
- **502** 后端依赖（BGE-M3 / Qdrant）不通
- **500** 内部未知错（兜底）
- **503** 全部依赖 down（仅 `/readyz` 和 `/healthz down` 状态会用）

### Q4：高并发下慢

- 检查 BGE-M3 服务负载（`curl $BGEM3_BASE_URL/metrics`）
- 启 gunicorn 多 worker：`gunicorn server:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8765`（注意 worker 间不共享 `_client` 单例，每个 worker 自己起 Qdrant client）
- 减小 prefetch_multiplier（retrieval.py 顶部常量）

### Q5：要鉴权

本服务默认 127.0.0.1 监听，无鉴权。要鉴权：
- 前置 nginx + Bearer Token / API Key
- 或在 `api.py` 加 `Depends(verify_token)` 装饰器
- 或在 `server.py` mount 之前加 `@app.middleware("http")` 做 token 校验

生产环境建议组合：
```bash
RAG_SVC_HOST=127.0.0.1            # 仅本地监听
RAG_DISABLE_DOCS=1                # 关闭 /docs /openapi.json
# 前置 nginx 做 Bearer Token + TLS + rate limit
```

### Q7：K8s 部署用哪个健康端点？

- **livenessProbe** → `/livez`（宽松，进程死了才返非 200，避免被外部抖动误杀）
- **readinessProbe** → `/readyz`（严格，依赖未 ready 时返 503 摘流量）
- 监控告警 → `/healthz`（信息型，返结构化 JSON 给 Prometheus exporter 解析）

### Q6：`/v1/fetch` 单 post 太长

`MAX_FETCH_CHUNKS=200` 是 scroll 上限。返回会带 `truncated_fetch: true` 标记。要拉更多：调大 `retrieval.py` 顶部 `MAX_FETCH_CHUNKS`。

---

## 九、目录结构

```
brain_rag_service/                      # MCP+REST 服务（目录名沿用历史）
├── README.md                       ← 本文件
├── server.py                       ← 入口：FastAPI + mount MCP + uvicorn
├── api.py                          ← FastAPI 路由定义
├── models.py                       ← Pydantic 请求/响应 schema
├── retrieval.py                    ← 检索 + 聚合 + LLM 友好格式化（MCP 和 REST 共用）
├── requirements.txt                ← mcp + fastapi + uvicorn + qdrant-client + ...
├── .env.example                    ← 通常复用 ../brain_index/.env，本目录可空
├── .gitignore
└── __init__.py
```

依赖关系：

```
retrieval.py  ← sys.path 注入 ../brain_index/ 后 import config / embedder
   ↑
   ├── api.py     （FastAPI routes 调）
   └── server.py  （MCP @mcp.tool 调）
```

---

## 十、Cheat Sheet

```bash
# 启动
conda activate brain
cd wq-rag/brain_rag_service
nohup python server.py > /tmp/brain-rag-service.log 2>&1 &

# 健康检查
curl -s http://127.0.0.1:8765/healthz | jq

# REST 调用
curl -s -X POST http://127.0.0.1:8765/v1/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"...","top_k":5}' | jq

# 浏览器: http://127.0.0.1:8765/docs    # Swagger UI 交互测试

# Claude Code 接入
claude mcp add --transport http brain-rag http://127.0.0.1:8765/mcp

# 停服务（注意：直接杀 python 不是 conda）
pkill -f "python.*server.py"
```

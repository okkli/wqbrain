# brain-rag-mcp

WorldQuant BRAIN 知识库的 MCP 工具服务器。把 `brain_index/` 已索引的 Qdrant 向量库
（BGE-M3 dense+sparse）封装成 3 个 MCP 工具，给 Claude Code 等 LLM CLI 用。

**默认走 streamable-http 模式**，无状态多客户端共享一个进程，不阻塞其它 CLI 窗口。

---

## 一、提供的工具

| 工具 | 用途 |
|---|---|
| `rag_search(query, ...filters)` | 混合检索 (Fusion RRF)。支持 tags / operators / recency / since_days / has_code / sa_type / source 过滤。返回 LLM 友好的 hits（聚合同 parent + content 截断 + 时间人类化） |
| `rag_fetch(ref)` | 按 `post:<id>` / `comment:<id>` / `tutorial:<id>` 拉完整原文 |
| `rag_stats()` | 知识库元信息：可用 tags / sources / operators / years 列表。**不知道怎么过滤时先调它** |

工具的详尽 docstring 直接在 `server.py`，会被 MCP 暴露给 LLM 用于工具选择。

---

## 二、启动 MCP 服务

### 前置：brain_index 已部署

本服务复用 `../brain_index/` 的 `config.py` 和 `embedder.py`。先确保：
- `brain_index/.env` 已配置（BGEM3 + Qdrant 端点 + 凭据）
- Qdrant 集合 `brain_kb_bgem3` 已索引完成
- conda `brain` 环境已装 `brain_index/requirements.txt`

详见 [`../README.md`](../README.md) §二。

### 装 MCP 依赖

```bash
conda activate brain
cd wq-rag/brain_rag_mcp
pip install -r requirements.txt
```

### 启动（streamable-http，推荐）

```bash
python server.py
# 输出: brain-rag-mcp listening on http://127.0.0.1:8765/mcp
```

后台启动 + 写日志：

```bash
nohup python server.py > /tmp/brain-rag-mcp.log 2>&1 &
echo $! > /tmp/brain-rag-mcp.pid

# 看日志
tail -f /tmp/brain-rag-mcp.log

# 停服务
kill $(cat /tmp/brain-rag-mcp.pid)
```

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_MCP_TRANSPORT` | `streamable-http` | `stdio` / `streamable-http` / `sse` |
| `RAG_MCP_HOST` | `127.0.0.1` | 监听地址。改 `0.0.0.0` 公网开放**不带任何鉴权**，务必前置反向代理（nginx）+ 防火墙白名单 |
| `RAG_MCP_PORT` | `8765` | 端口 |
| `RAG_MCP_PATH` | `/mcp` | HTTP 路径 |

---

## 三、接入 Claude Code（HTTP 模式）

启动服务后，在 `~/.claude.json` 或项目级 `.claude.json` 加：

```json
{
  "mcpServers": {
    "brain-rag": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

也可用 `claude mcp add` 命令：

```bash
claude mcp add --transport http brain-rag http://127.0.0.1:8765/mcp
```

之后在 Claude Code 里直接调用 `mcp__brain-rag__rag_search` / `rag_fetch` / `rag_stats`，
**多个 Claude Code 窗口可以共享同一个服务进程**，互不阻塞。

### stdio 备选（单进程独占）

如果不想常驻 HTTP 服务，可以让 Claude Code 按需启动 stdio：

```json
{
  "mcpServers": {
    "brain-rag": {
      "command": "/Users/<you>/data/miniconda3/envs/brain/bin/python",
      "args": ["/path/to/wq-rag/brain_rag_mcp/server.py"],
      "env": {"RAG_MCP_TRANSPORT": "stdio"}
    }
  }
}
```

缺点：每个 CLI 窗口都会 fork 一个进程。

---

## 四、给 LLM 看的调用样例

LLM 在你问"BRAIN 论坛里有人怎么处理高 turnover？"时的预期调用链：

```python
# 1) (可选) 不熟悉数据先看维度
rag_stats()
# → 知道有 alpha_inspiration / consultant_lead / has_code 这些 tag

# 2) 混合检索 + 过滤
rag_search(query="如何降低 turnover 提高 margin", top_k=8, tags=["consultant_lead"], recency="recent")
# → 返回 hits, 每条有 ref

# 3) 命中后看完整原文
rag_fetch(ref="post:37117908343831")
# → 拿完整 markdown，含所有 segments + 代码块
```

### 召回信息结构（LLM 友好）

每条 hit 已做了：
- **同 post/tutorial 的多 segment 合并**，避免重复
- **去掉 LLM 不需要的字段**（source_path / source_hash / chunk_index 等）
- **时间双格式**：`date: "2024-10-28"` + `age: "约 5 个月前"`
- **content 截断到 800 字符** + `truncated: true` + `fetch_full_with: "rag_fetch(ref='...')"` 提示
- **applied_filters 回显**，帮 LLM 理解结果来源
- **summary**：score 范围 / by_source / by_year

详见 `server.py` 里 `rag_search` 的 docstring。

---

## 五、调试 / 验证

### 直接调函数（不经 HTTP）

```bash
conda run -n brain --cwd wq-rag/brain_rag_mcp python -c "
import retrieval, json
r = retrieval.search_hybrid('ts_corr 用法', top_k=3,
    filt=retrieval.build_filter(operators=['ts_corr'], has_code=True))
print(json.dumps(r, ensure_ascii=False, indent=2))
"
```

### HTTP 服务自检

```bash
# 注意：路径不带尾斜杠（带斜杠会 307 重定向）
curl -sSL -X POST http://127.0.0.1:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  | python3 -m json.tool | head -30
```

### MCP Inspector（可视化）

官方推荐：

```bash
npx @modelcontextprotocol/inspector
# 在 UI 里填 http://127.0.0.1:8765/mcp，可视化调用 3 个工具
```

---

## 六、常见问题

### Q1：HTTP 启动后 Claude Code 连不上

- 看日志：`tail -f /tmp/brain-rag-mcp.log`
- 验 HTTP 是否能 curl 到（§五）
- 确认 Claude Code 配置里 `"type": "http"` 不是 `"sse"`
- 端口冲突：换 `RAG_MCP_PORT`

### Q2：`Module not found: brain_index.config`

`retrieval.py` 自动注入 `../brain_index/` 到 `sys.path`。如果你把 `brain_rag_mcp/` 移到了别处，
要么改 `retrieval.py` 顶部的 `_BRAIN_INDEX` 常量，要么 `pip install -e ../brain_index/`。

### Q3：返回 `{"error": "...", "type": "..."}`

工具内部异常被捕获返回。看 server 日志拿完整 traceback。

### Q4：召回不准 / 漏 hit

- 先调 `rag_stats()` 看可用过滤维度
- 检查 query 关键词在 `operators_sample` 里是否存在（不存在的算子要先加进 `config.SA_OPERATORS`）
- 试不带 filter 跑一遍，确认是过滤太严还是 query 不匹配

### Q5：rag_fetch 返回的全文非常长

正常 —— 论坛长帖加评论加代码可能 50K 字符。LLM 上下文够就喂；不够就 `full_content=False`
只拿元信息和 chunk 概览，再决定具体取哪几个 segment。

---

## 七、与其它 MCP 工具的关系

| MCP server | 路径 | 作用 |
|---|---|---|
| brain-rag-mcp (本服务) | `wq-rag/brain_rag_mcp/` | **查 RAG 知识库** |
| brain-platform-mcp | `alpha-optimize/wqmcp/` | 调 WQ Brain 平台 API（创建 simulation / 提交 alpha / 查 datafields） |

两个 MCP 互相独立，可同时挂载。Claude Code 会按工具名分发，不冲突。

---

## 八、Cheat Sheet

```bash
# 启动服务（后台）
cd wq-rag/brain_rag_mcp
nohup python server.py > /tmp/brain-rag-mcp.log 2>&1 &
echo $! > /tmp/brain-rag-mcp.pid

# 看是不是活的
curl -sSL -X POST http://127.0.0.1:8765/mcp \
  -H "Content-Type: application/json" -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | head -c 500

# Claude Code 一键加
claude mcp add --transport http brain-rag http://127.0.0.1:8765/mcp

# 停服务
kill $(cat /tmp/brain-rag-mcp.pid)
```

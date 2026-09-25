# wqmcp：WorldQuant BRAIN MCP 服务

给 Claude Code、Cursor 这类 LLM 客户端用的 BRAIN 平台工具集，共 25 个工具。v2 按
`API_AUDIT.md` 的审计结果重写，接口用法以《WQ API Catalog 1.15.3》为准。这份目录是从
前端逆向整理的，不是官方契约。

| 文件 | 作用 |
|---|---|
| `brain_client.py` | BRAIN HTTP 客户端：credd cookie、按接口设置 Accept 版本、Retry-After 轮询、429 冷却、参数校验、结果裁剪 |
| `server.py` | MCP 工具定义（FastMCP），同时是启动入口 |
| `platform_functions.py` | 旧入口，`python platform_functions.py` 仍然可用 |
| `forum_functions.py` | 论坛和术语表抓取（Playwright，support.worldquantbrain.com） |
| `tests/` | fake BRAIN + credd，以及 pytest 用例（客户端、MCP 端到端、论坛解析） |

## 启动

```bash
pip install -r requirements.txt
python server.py                      # 默认 streamable-http，地址 http://127.0.0.1:8761/mcp
WQMCP_TRANSPORT=stdio python server.py   # stdio：由 MCP 客户端直接拉起进程
```

登录由 credd（creds-daemon）负责，本服务不接触 BRAIN 密码。credd 必须先运行，本服务通过
`CREDD_URL`/`CREDD_TOKEN` 从它那里取 cookie。

接入 Claude Code：

```bash
claude mcp add --transport http brain-platform http://127.0.0.1:8761/mcp
```

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WQMCP_HOST` | `127.0.0.1` | 监听地址。要对外提供服务就设成 `0.0.0.0`，并在前面加反代做鉴权，因为服务本身不鉴权 |
| `WQMCP_PORT` | `8761` | 端口 |
| `WQMCP_TRANSPORT` | `streamable-http` | 可选 `stdio`、`sse` |
| `WQMCP_ALLOWED_HOSTS` | 空 | 反代转发的 Host 值，逗号分隔（例如 `mcp.example.com`）。监听 127.0.0.1 时 FastMCP 会开启 DNS rebinding 保护，其他 Host 一律拒绝 |
| `CREDD_URL` / `CREDD_TOKEN` | `http://127.0.0.1:8762` / 空 | credd 地址和令牌 |
| `WQMCP_ACCEPT_VERSIONS` | `1` | 按目录发送版本化的 Accept 头（如 listAlphas 用 4.0）。设为 `0` 则只发 `application/json` |
| `WQMCP_USE_PLATFORM_DEFAULTS` | `1` | `create_simulation` 没填的设置，用你在 BRAIN 网页上保存的默认设置补齐 |
| `WQMCP_READ_ONLY` | `0` | 设为 `1` 时禁用所有写工具：建模拟、取消模拟、改 alpha、真正提交 |
| `WQMCP_ALLOW_SUBMIT` | `1` | 设为 `0` 时禁止真正提交，`submit_alpha` 只能做 dry-run 检查 |
| `WQMCP_MAX_INFLIGHT` | `16` | 同时发往 BRAIN 的请求数上限（整个账户共享） |
| `WQMCP_HTTP_WORKERS` / `WQMCP_POOL_MAXSIZE` | `32` / `32` | HTTP 线程池大小和连接池大小 |
| `WQMCP_CONNECT_TIMEOUT` / `WQMCP_READ_TIMEOUT` | `10` / `60` | 单次请求的连接超时和读超时（秒） |
| `WQMCP_FORUM_CONCURRENCY` / `WQMCP_FORUM_TIMEOUT` | `2` / `90` | 论坛工具的并发页数上限和单次操作超时（秒） |
| `WQMCP_FORUM_BROWSER_CHANNEL` | `chrome` | 先尝试本机 Chrome，失败时退回 Playwright 自带的 Chromium |
| `WQMCP_FORUM_CHROMIUM_SANDBOX` | `0` | 设为 `1` 时启用 Chromium 沙箱。Playwright 默认关闭沙箱，而无头浏览器只会访问 support 站点；在非 root、非容器环境里建议开启 |
| `WQMCP_GLOSSARY_TTL` | `86400` | 术语表缓存时间（秒） |
| `WQMCP_LOG_LEVEL` | `INFO` | 日志只输出到 stderr |

## 工具一览

| 分组 | 工具 | 说明 |
|---|---|---|
| 账户 | `brain_status` | 登录状态。`overview=True` 时附带 alpha 数量、顾问摘要和未读消息 |
| 模拟 | `create_simulation` | 1 条表达式发单模拟，2–10 条发一个 multi 模拟；SUPER 类型用 `combo` + `selection`。立即返回，结果里的 `settings_used` 是实际发出的设置 |
| | `get_simulation` | 查询一个或多个模拟，`wait_seconds` 可以在服务端等待。不传 id 时列出本进程最近创建的模拟，即使当初那次 MCP 调用被取消了也能找回 |
| | `cancel_simulation` | 取消模拟，释放并发槽 |
| | `get_platform_setting_options` | 合法的 region/delay/universe/neutralization 组合，以及你保存的默认设置 |
| | `preview_super_selection` | 预览某个 SuperAlpha selection 表达式会选中哪些 alpha |
| Alpha | `list_alphas` | 支持过滤和分页（返回 `next_offset`），默认返回紧凑摘要 |
| | `get_alpha` | 单个 alpha 的摘要，`full=True` 返回原始对象 |
| | `get_alpha_recordset` | pnl、sharpe、turnover、daily-pnl、yearly-stats，以列 + 行的形式返回，按 Retry-After 轮询 |
| | `check_alpha` | 平台权威的提交前检查（`/check`），可以同时拉取 self/prod/power-pool 相关性 |
| | `submit_alpha` | **默认是 dry-run**，只做检查；`confirm=True` 才真正提交，并会一直跟踪到最终结果 |
| | `update_alpha` | 批量设置 favorite/hidden/color；单个 alpha 可改 name/category/tags/描述 |
| | `get_alpha_performance` | 这个 alpha 加入组合前后的表现对比（也支持比赛版本） |
| 数据 | `get_datasets` | 数据集检索。`include_fields=True` 时按关键词同时搜数据集和字段 |
| | `get_datafields` | 字段检索和分页；传 `field_id` 查单个字段详情 |
| | `get_operators` | 算子列表，可按 category/scope/name 过滤，缓存 1 小时 |
| 活动 | `get_activity` | diversity、pyramid-*、payment、simulations/submissions、value-factor（官方）、diversity-score（本地估算） |
| | `get_leaderboard` | 四种顾问榜单，支持分页、排序和聚合 |
| 社区 | `get_competitions` | 进行中/已结束/全部/已参加的比赛列表，或单个比赛详情（可附带协议） |
| | `get_events` | 线上和线下活动 |
| | `get_messages` | 公告和通知，内嵌图片会被去掉 |
| | `get_documentation` | 官方文档：教程目录，或某一页的内容块 |
| 论坛 | `search_forum_posts` / `read_forum_post` / `get_glossary_terms` | 用 Playwright 抓取 support 站点。已入库的内容用 brain-rag 的 `rag_search` 查更快 |

所有工具出错时都会抛异常，MCP 客户端收到的是 `isError: true` 和一条可读的错误信息。耗时任务
返回 `RUNNING` 或 `PENDING` 以及 `retry_after_seconds`，稍后再调同一个工具即可。读请求遇到
429/502/503/504 或网络错误时，会按 Retry-After 自动重试两次；写请求从不自动重试。

论坛工具需要浏览器：`pip install playwright` 之后运行 `playwright install chromium`，或者本机
装有 Chrome。未安装 Playwright 时服务仍能启动，只是论坛工具会报错并提示安装方法。support 站点的
登录走 `/access/sso`，和 `wq-rag/wq-doc-forum` 的做法一致，复用 credd 提供的 BRAIN cookie。

## 旧工具 → 新工具（v1 有 40 个工具，v2 有 25 个）

| v1 | v2 |
|---|---|
| `authenticate`、`manage_config` | `brain_status` |
| `create_simulation`、`create_multi_simulation` | `create_simulation(expressions=[...])` |
| `check_simulation_progress`、`lookINTO_SimError_message` | `get_simulation` |
| `get_alpha_details` | `get_alpha` |
| `get_user_alphas` | `list_alphas` |
| `get_alpha_pnl`、`get_alpha_yearly_stats`、`get_record_sets`、`get_record_set_data` | `get_alpha_recordset` |
| `check_correlation`、`get_submission_check` | `check_alpha` |
| `submit_alpha` | `submit_alpha(confirm=True)` |
| `set_alpha_properties` | `update_alpha` |
| `performance_comparison` | `get_alpha_performance` |
| `run_selection` | `preview_super_selection` |
| `get_user_activities`、`get_pyramid_multipliers`、`get_pyramid_alphas`、`get_daily_and_quarterly_payment`、`value_factor_trendScore` | `get_activity(kind=...)` |
| `get_user_profile` | `brain_status`（本人）；他人的公开档案目前没有对应工具 |
| `get_user_competitions`、`get_competition_details`、`get_competition_agreement` | `get_competitions` |
| `get_documentations`、`get_documentation_page` | `get_documentation` |
| `expand_nested_data` | 删除（纯本地转换，不涉及 BRAIN） |
| `get_events`、`get_leaderboard`、`get_messages`、`get_operators`、`get_datasets`、`get_datafields`、`get_platform_setting_options`、论坛 3 个工具 | 名称不变，参数有增加 |

## 行为变化（升级时注意）

- **提交**：`submit_alpha` 默认只做检查，不提交；要真正提交必须传 `confirm=True`。提交会按
  Retry-After 跟踪到最终的检查结果，返回 SUBMITTED、REJECTED 或 PENDING。返回 PENDING 后再次
  调用只会继续轮询，不会重复提交。
- **模拟默认值**：没填的设置取你在网页上保存的默认值（例如 decay 4、SUBINDUSTRY、truncation
  0.08），不再固定用 `NONE/0/0`。实际使用的设置见返回的 `settings_used`。`decay` 现在必须是整数。
- **监听地址**：默认改为 `127.0.0.1`。原来跨机器访问的部署，需要设置 `WQMCP_HOST=0.0.0.0`，
  并在前面加反代做鉴权。
- **返回结构**：列表类工具返回 `count/limit/offset/has_more/next_offset`，alpha 默认只返回紧凑
  摘要，recordset 默认只保留最近 300 行。
- **安全**：id 参数会校验并做 URL 编码，拦截 `../` 这类路径穿越；`get_simulation` 只接受 BRAIN
  API 域名下的 `/simulations/<id>` 地址；email/password 参数全部删除；`manage_config` 删除。

## 需要实测确认的地方

目录是逆向整理的，下面几处没有用真实账号验证过：

- `preview_super_selection` 按目录发送 `settings.region/settings.delay/settings.instrumentType`；
  v1 发的是不带前缀的参数名。
- `OPTIONS /simulations` 在 version=3.0 下的响应结构：目录里的示例是节选的。如果解析不出组合，
  会自动改用不带版本的 Accept 再请求一次。
- `list_alphas` 的日期过滤（`dateCreated>` 等）和 `hidden` 不在目录里，沿用 v1 的写法。
- `update_alpha` 的 `osmosis_points` 和 tags 字符串数组写法、比赛的 `/agreement` 端点，目录里都
  没有记载。
- 论坛页面的选择器和术语表的解析规则是按旧代码推断的，没有拿真实页面核对过。

如果某个接口在新的 Accept 版本下行为异常，可以先设置 `WQMCP_ACCEPT_VERSIONS=0` 退回旧行为。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest            # fake BRAIN + credd 在进程内运行，不访问真实平台
```

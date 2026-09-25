# wqmcp 接口使用与冗余审计（对照 WQ API Catalog 1.15.3）

- 审计对象（v1）：`alpha-optimize/wqmcp/platform_functions.py`（40 个 `@mcp.tool`）、`alpha-optimize/wqmcp/forum_functions.py`
- 参照：`wqapi-visible-catalog-1.15.3.md`，93 个前端可见接口
- 当前代码的状态：见第八节（main 与 v2 的合并版，30 个工具）。
- 方法：7 个按接口域划分的审计 agent 和 1 个冗余分析 agent 独立审计，每组结果再由一个复核 agent 逐条对照代码行和目录行尝试推翻，最后由完整性 critic 补漏。138 条已确认或部分确认的发现中，高危问题我另外逐一对照代码和目录核对过。
- 结果：共提出 127 条，推翻 1 条（SIM-16，见文末）；复核阶段补充 5 条，critic 补充 7 条。最终保留 **138 条：高 11、中 59、低 68**。
- 局限：
  - 目录是逆向整理的结果，不是官方契约。
  - 目录不收录 `visibility: hidden` 的接口，所以"不在目录中"只代表无法验证，不代表接口不存在。
  - 本次是静态分析，没有用真实账号实测。标注"需实测"的条目上线前应先验证。

---

## 一、结论速览

**最需要先修的 10 件事**（结果错误或会误导 Agent）：

| # | 问题 | 涉及工具 | 发现编号 |
|---|---|---|---|
| 1 | **提交不可靠**：`submit_alpha` 只发一次 `POST /alphas/{id}/submit` 就报成功，没有按 Retry-After 轮询 `GET /alphas/{id}/submit` 取最终 `is.checks`。成功时返回 `requests.Response.__dict__`（类型不可序列化，可能带出响应头和 cookie），失败时吞掉异常只返回 `False` | submit_alpha | ALPHA-2/3/4, RED-15 |
| 2 | **提交前检查口径错误**：`get_submission_check` / `check_correlation` 没有使用平台权威的 `GET /alphas/{id}/check`，只用 0.7 阈值自行判断 prod/self 相关性，其他检查项（sharpe、fitness、turnover、sub-universe 等）全部没看。`correlation_type='prod'` 或 `'power-pool'` 会被静默跳过，并返回 `all_passed=True` | check_correlation, get_submission_check | ALPHA-1/7, ANLY-1/5, RED-3 |
| 3 | **轮询不遵守 Retry-After**：recordset 和相关性轮询固定 sleep（最长 20s × 5 次）。重试耗尽后把"仍在计算"当成空数据 `{}` 返回。401/404/429 也盲目重试，单次 check_correlation 最多空耗约 160s | get_alpha_pnl, get_alpha_yearly_stats, check_correlation, get_record_set_data | ANLY-2/3/7, ALPHA-5, ALPHA-M1 |
| 4 | **调错端点**：`get_user_activities` 的 docstring 写的是"活动多样性"，实际请求 `GET /users/{id}/activities`，这个端点只返回 5 个分类名。`grouping` 属于 `GET /users/self/activities/diversity` | get_user_activities | AUTH-1, RED-16 |
| 5 | **调错端点**：`get_user_profile` 调用的是 `getUser`。查他人一律 403；查自己会把 email、电话、地址等 PII 塞进 LLM 上下文。公开档案应改用 `GET /users/{id}/profile` | get_user_profile | AUTH-2 |
| 6 | **分页被静默截断**：`get_datafields` 写死 `limit=50`；`get_datasets` 不传 limit，默认只有 20 条；`run_selection` 10 条；`get_messages` 和 `get_events` 10 条；`value_factor_trendScore` 只取一页 500 条。这些工具都没有暴露 offset，无法翻页 | get_datafields, get_datasets, run_selection, get_messages, get_events, value_factor_trendScore | DATA-1/2, SIM-8, MISC-2/3/9, AUTH-4 |
| 7 | **multi 模拟状态误判**：某个 child 返回 429/5xx 时被当作已结束，整个 multi 报 COMPLETE，这个 child 的结果丢失。单模拟轮询遇到 429 直接返回 ERROR。创建失败时 BRAIN 返回的错误原因被丢掉 | check_simulation_progress, create_*_simulation | SIM-3/4/5/18 |
| 8 | **value_factor_trendScore 结果偏低**：对每个 alpha 串行调一次 `GET /alphas/{id}`（N+1），失败时 `continue`，但分母 N 仍计入这个 alpha，分数被系统性压低。而 listAlphas 结果里本来就有 classifications/pyramids；目录里的 `getConsultantPerformance` 直接给官方 `valueFactor` | value_factor_trendScore | AUTH-3/6/7, RED-1 |
| 9 | **super-selection 参数名与目录不一致**：代码发送扁平的 `instrumentType/region/delay`，目录写的是 `settings.instrumentType/settings.region/settings.delay`（需实测确认哪种生效） | run_selection | SIM-1 |
| 10 | **未收录端点**：`performance_comparison` 调用 `/alphas/{id}/performance-comparison`。目录里有 `GET /users/self/alphas/{id}/before-and-after-performance`，以及比赛版 `/competitions/{cid}/alphas/{id}/before-and-after-performance` | performance_comparison | ANLY-6 |

**安全问题**（服务监听 `0.0.0.0`，本身没有鉴权，下面几条的风险因此被放大，见 RED-M1）：

- **SSRF**：`lookINTO_SimError_message` 会对调用方给的任意 URL 发 GET，并把响应体原样返回。`check_simulation_progress` 只检查 URL 里是否包含 `worldquantbrain.com` 子串，`http://127.0.0.1:8762/cookies?x=worldquantbrain.com` 也能通过。如果 credd 没配 `CREDD_TOKEN`，就可能被读出 cookie（SIM-2、RED-5）。`read_forum_post` 同样接受任意 URL，并交给以 `--no-sandbox` 运行的 Chrome 打开（FORUM-M1）。
- **路径穿越**：`alpha_id`、`competition_id`、`page_id` 直接拼进 URL，urllib3 会规范化 `../`，结果 `set_alpha_properties`（PATCH）和 `submit_alpha`（POST）可以打到任意 BRAIN 路径（CRIT-1、MISC-7）。
- **有副作用的工具没有任何防护**：`submit_alpha`、`set_alpha_properties`、`create_*_simulation`、`manage_config(set)` 都没有确认、dry-run，也没有设置 MCP `destructiveHint` 注解（CRIT-2）。
- **明文凭据**：`manage_config(get)` 会把配置文件原样回显，其中可能有明文 credentials。5 个工具共 10 个 email/password 参数已全部失效，却仍会诱导 LLM 传明文密码（SIM-19、RED-8/9、AUTH-16、FORUM-9）。

**横切问题**：

- **缺少 Accept 版本头**：所有请求都没发目录要求的 Accept 头。多数接口要 `application/json;version=2.0`，`listAlphas`、`listTagAlphas`、`getSuggestedFields` 要 4.0，`base-payment` 和 OPTIONS `/simulations` 要 3.0。版本不同，返回的 schema 可能不同，建议在 `_request` 里按接口统一设置（AUTH-9、DATA-6、SIM-14、ALPHA-10、ANLY-10、MISC-6）。
- **错误都被当成正常结果返回**：所有工具都 try/except 后返回 `{"error": ...}` 这类普通结果，MCP 的 `isError` 永远是 false，错误结构有五六种。内部的 credd 地址和 biometric_url 也会回传给远程客户端（CRIT-5）。
- **MCP 调用被取消后请求仍在执行**：`_request` 在线程池里执行，取消信号传不到线程，`POST /simulations` 会照常发出。201 响应里的 Location 一旦丢失，这个模拟就成了孤儿，重试还会再建一个重复模拟（CRIT-3）。
- **没有账户级限流**：32 个工作线程加上 gather 扇出，多个 MCP 客户端会同时对同一个 BRAIN 账户发突发请求；一处收到 429，其他调用不会跟着退避（CRIT-6）。
- **论坛模块把服务器文件又执行了一遍**：`forum_functions.py:131` 的 `from platform_functions import brain_client` 在以脚本方式启动时，会用新模块名重新执行整个 `platform_functions.py`，生成第二个 `BrainApiClient`（外加一个 32 线程的线程池）和第二个 FastMCP 实例（RED-10）。

---

## 二、冗余分析

### 2.1 功能重叠的工具

| 冗余组 | 当前工具 | 重叠方式 | 建议 |
|---|---|---|---|
| Recordset | `get_alpha_pnl`、`get_alpha_yearly_stats`、`get_record_sets`、`get_record_set_data` | 前两个是 `get_record_set_data('pnl' / 'yearly-stats')` 的严格子集。4 处轮询和重试是复制粘贴的，处理方式还互不一致（pnl 会轮询，通用版不轮询） | 合并为 `get_alpha_recordset(alpha_id, type)`，共用一个遵守 Retry-After 的轮询器，并支持裁剪或摘要 |
| 相关性 / 提交检查 | `check_correlation`、`get_submission_check`（以及内部的 `get_production_correlation` / `get_self_correlation`） | `get_submission_check` 等于 `check_correlation` 加 `get_alpha_details` 拼在一起，返回体过大。prod 和 self 两个方法是复制粘贴的 | 合并为 `check_alpha`：基于 `GET /alphas/{id}/check` 轮询，另可选拉取 `correlations/{prod,self,power-pool}` 明细 |
| 模拟进度 | `check_simulation_progress`、`lookINTO_SimError_message` | 后者是前者的劣化子集：不校验 URL，会把还在运行的模拟报成失败 | 删掉 `lookINTO_SimError_message`，把"批量查询多个 location"合并进前者 |
| 模拟创建 | `create_simulation`、`create_multi_simulation` | multi 在工具层另写了一套 settings 拼装，没复用 `SimulationSettings`，默认值和 PYTHON 分支的行为也不一致 | 合并为 `simulate(alphas: list)`：1 条走单模拟，2–10 条走 multi，共用 settings 模型 |
| 当前用户 / 登录状态 | `authenticate`、`manage_config(get)`、`get_user_profile(self)`；`get_leaderboard` 和 `get_user_competitions` 每次先请求一次 `GET /users/self` | 三个工具都返回"我是谁 / 是否已登录"。目录明确说 `userId` 可以直接写 `self`，所以预查询完全多余 | 保留一个 `brain_status`（`GET /authentication` 就能拿到 user.id、expiry 和 permissions），其他地方直接用 `self` |
| Activities 家族 | `get_user_activities`、`get_pyramid_multipliers`、`get_pyramid_alphas`、`get_daily_and_quarterly_payment`、`value_factor_trendScore` | 同一资源族拆成了 5 个工具，其中一个还调错了端点 | 合并为 `get_activity(name, …)`，覆盖 diversity、pyramid-*、base/other-payment、simulations、submissions；多样性或 value factor 改用官方 `getActivityDiversity` 和 `getConsultantPerformance` |
| 比赛 / 活动 | `get_user_competitions`、`get_competition_details`、`get_competition_agreement`、`get_events` | 按 id 查和列表查分成了几个薄包装 | 合并为 `get_competition(id?)`，并补上 `listCompetitions`（目前没有任何工具能发现可参加的比赛） |
| 文档 / 论坛 / 知识库 | `get_documentations`、`get_documentation_page`、`get_glossary_terms`、`search_forum_posts`、`read_forum_post` | 和同一个 `.mcp.json` 里的 **brain-rag MCP**（`rag_search` / `rag_fetch`，已索引论坛和教程）职责重叠。论坛抓取也和 `wq-rag/wq-doc-forum` 的实现重复，而且缺少后者已经修掉的若干问题（FORUM-16） | 文档合并为 `get_documentation(page_id?)`。论坛检索交给 brain-rag；wqmcp 只保留"按 URL 读单帖"，或者整组下线 |
| 纯本地转换 | `expand_nested_data` | 不调用任何 BRAIN 接口，需要 LLM 把大段数据回传，而且对 `{schema, records}` 这种列式结构不起作用。pandas 依赖只为它存在 | 删除 |

### 2.2 死代码和遗留参数

- **遗留凭据参数**：`authenticate`、`get_daily_and_quarterly_payment`、`get_glossary_terms`、`search_forum_posts`、`read_forum_post` 共有 10 个 email/password 参数，全部被忽略，却还沿着 5 层调用链往下传（RED-8）。
- **配置文件代码**：`manage_config`、`load_config`、`save_config` 唯一的用途是读写那份已经被忽略的 credentials。临时文件兜底会丢配置，写入失败也被吞掉（RED-9、SIM-20）。
- **重复的认证检查**：`ensure_authenticated()` 在客户端方法里调用了 33 次，和 `_request` 里的 `_ensure_session` 完全重复（RED-14、AUTH-20）。
- **无依据的回退路径**：`get_pyramid_alphas` 在 404 时回退到 `/users/self/pyramid/alphas` 和 `/activities/pyramid-alphas`，两个路径都不在目录里，而主端点目录已实测确认。错误提示中还列出了一个从未请求过的路径（AUTH-14、RED-17）。
- **没有效果的参数**：`get_datafields` 的 `theme` 参数接收了但从来不发送（DATA-4）；`get_datasets` 默认发送 `theme=false`，目录没有这个值（DATA-5）。
- **不可达代码**：`get_operators` 做了两层 list→dict 包装，外层分支永远走不到（DATA-9）；另有未使用的 import、未使用的 `ForumClient.session`、三层重复的"先记日志再抛出"（RED-18、FORUM-15）。
- **薄包装**：36 个 wrapper 和约 30 个客户端方法用的是同一套样板，其中很多 wrapper 只是把调用原样转发（RED-14）。

### 2.3 工具面开销

- 现状：40 个工具，126 个参数，去缩进后 docstring 约 14k 字符。每个连上来的 MCP 客户端都要把这些加载进上下文。
- 详略两极：`get_user_alphas` 的 docstring 有 2.4k 字符，另有 17 个不到 100 字符。其中 `get_record_set_data` 没列出合法的 `record_set_name`；`value_factor_trendScore` 写了一个并不存在的 `p_max` 参数；`create_simulation` 漏写了 6 个参数（RED-20、AUTH-18）。
- 建议合并为约 20 个工具：

`brain_status` · `simulate` · `get_simulation` · `list_alphas` · `get_alpha` · `get_alpha_recordset` · `check_alpha` · `update_alpha` · `submit_alpha`（带轮询和确认） · `get_datasets` · `get_datafields` · `get_operators` · `get_platform_setting_options` · `preview_super_selection` · `get_activity` · `get_diversity`（改用官方接口） · `get_competition` · `get_leaderboard` · `get_messages` · `get_documentation` · `get_alpha_performance`（before-and-after）

同时：枚举参数改用 `Literal`，让 schema 自带可选值；每个 docstring 控制在 300–600 字符；返回结果默认只保留关键字段。

---

## 三、目录中尚未使用、值得接入的接口

按对 alpha 研究 Agent 的价值排序：

1. **`GET /alphas/{id}/check`**（getAlphaChecks）：平台权威的提交前检查，返回全部 `is.checks`（PASS/FAIL/PENDING，带 limit 和 value），需要按 Retry-After 轮询。可以替代现在两个工具的客户端近似判断。
2. **`GET /alphas/{id}/submit`**（pollAlphaSubmission）：提交是异步的，没有这一步就不知道到底成功没有。
3. **`GET /users/self/activities/diversity`**（getActivityDiversity）和 **`GET /users/{id}/consultant`**（getConsultantPerformance）：官方的多样性数据和 valueFactor，可以替代 `value_factor_trendScore` 的 N+1 近似计算。
4. **`DELETE /simulations/{id}`**（deleteSimulation）：遇到 RATE_LIMITED 或孤儿模拟时用来释放并发槽，要做成需要显式确认的工具。
5. **`GET /users/{id}/settings/simulation`**（getSimulationSettings）：读取用户在平台上的默认模拟设置，代替写死的 `NONE/0/0`，让 MCP 跑出的结果和网页端可比（SIM-10）。
6. **`GET /data-sets/search`**、**`GET /data-fields/{id}`**、**`GET /data-fields/summary`**、**`GET /data-categories`**、**`GET /suggest/fields`**：分别用于关键词一次同时搜数据集和字段、查单个字段的覆盖率、拿全量字段名校验表达式、拿分类过滤器的取值、拿 SuperAlpha 可用的字段。
7. **`GET /users/self/alphas/{id}/before-and-after-performance`**（getSelfAlphaPerformance）：替代未收录的 `/performance-comparison`。
8. **`GET /users/self/alphas/summary`**（version=4.0）：一次拿到各阶段和状态的 alpha 数量，不必翻页拉取全部 alpha。
9. **`GET /alphas/{id}/correlations/power-pool`**：目录已实测，代码目前完全不支持。
10. **Alpha List（tags）系列**：`listSelfTags`、`listTagAlphas`、`getTagInnerCorrelation`、`getTagSelfCorrelation`、`createTag`、`patchTag`。由服务端计算一批候选 alpha 之间的相关性，以及它们与已提交 alpha 的相关性，可以替代逐个 alpha 做检查的 N+1 请求。
11. **`GET /competitions`**（listCompetitions）、`GET /competitions/{id}/boards/{type}`、`GET /competitions/{id}/submissions`：补上比赛工作流，目前连"有哪些比赛可以参加"都查不到。
12. **`PATCH /alphas`**（bulkPatchAlphas）：批量设置 favorite/hidden/color。现有的 `set_alpha_properties` 连 hidden 和 favorite 都不支持。
13. **`GET /users/self/messages/summary`**、**`GET /users/{id}/osmosis/summary`**、**`GET /users/{id}/consultant/summary`**：都是很轻的请求，适合 Agent 开局时做一次账户体检。
14. **`GET /search`**（searchPlatform）、**`GET /video-courses`**：官方搜索和带文字稿的视频课程，可以部分替代 Playwright 论坛抓取。`type` 的取值目录没写，需要实测。

---

## 四、建议的修复顺序

1. **安全**（半天）：
   - progress/location 只接受 `https://api.worldquantbrain.com/simulations/` 前缀；`read_forum_post` 只接受 support 域名。
   - 所有路径参数先用 `quote(x, safe='')` 编码，并按 id 格式校验。
   - 删除 `lookINTO_SimError_message`、email/password 参数和 `manage_config`。
   - 服务改为监听 127.0.0.1，或者前置反代做鉴权。
2. **提交链路**：
   - `submit_alpha` 改为 POST 后按 Retry-After 轮询 `GET /submit`，返回解析后的 `is.checks`。
   - `check_alpha` 基于 `GET /check` 实现。
   - 抽出一个共用的 Retry-After 轮询器（识别空响应体，429 退避，4xx 不重试），替换现有 4 处复制粘贴的循环。
3. **端点纠错**：
   - `get_user_activities` 改为请求 `/activities/diversity`。
   - `get_user_profile` 改为请求 `/users/{id}/profile`。
   - `performance_comparison` 改为请求 before-and-after-performance。
   - `run_selection` 的参数名改为 `settings.*`，先实测确认。
4. **分页**：所有列表工具都暴露 `limit`、`offset`、`order`，返回 `count`、`has_more`、`next_offset`，并裁剪长字段。
5. **横切**：
   - `_request` 按接口设置 Accept 版本头。
   - 失败时 raise，让 MCP 返回 `isError`，错误结构统一。
   - 用一个共享令牌桶处理 429。
   - `POST /simulations` 立刻持久化 Location，防止模拟变成孤儿。
6. **收敛工具面**：按第二节合并到约 20 个工具，论坛和文档类检索交给 brain-rag。

---

## 五、接口映射表（工具 → 实际调用 → 目录 operationId）


来源：冗余分析 agent 对全部 40 个工具的逐一梳理（经复核）。

| 工具 | 实际调用的端点 | 目录 operationId | 备注 |
|---|---|---|---|
| `authenticate` | credd GET {CREDD_URL}/cookies（强制刷新）+ GET /authentication | getAuthentication | email/password 形参被忽略；仅 200 视为成功（目录列 200/204 均为成功）；与 manage_config(get)、get_user_profile 功能重叠 |
| `manage_config` | 本地文件 user_config.json 读写 + GET /users/self + GET /authentication | getUser (userId=self) + getAuthentication | get 分支每次 2 次网络请求；set 分支写入的配置只有 forum 工具读取 credentials，而下游忽略该值 |
| `create_simulation` | POST /simulations | createSimulation | 只提交；经 SimulationSettings/SimulationData 构造 payload |
| `check_simulation_progress` | GET {progress_url}（/simulations/{id}）+ 子模拟 GET /simulations/{childId} + GET /alphas/{alphaId} | getSimulation + getAlpha | 遵循 Retry-After；与 lookINTO_SimError_message 读同一 URL |
| `get_alpha_details` | GET /alphas/{alphaId} | getAlpha | 纯透传 |
| `get_datasets` | GET /data-sets | listDatasets | docstring 未说明 search 参数 |
| `get_datafields` | GET /data-fields | listDataFields | 硬编码 limit=50/offset=0，未暴露分页 |
| `get_alpha_pnl` | GET /alphas/{alphaId}/recordsets/pnl（最多 5 次固定退避重试） | getAlphaRecordset (recordsetType=pnl) | 是 get_record_set_data(name='pnl') 的严格子集，只多了一段复制粘贴的重试循环 |
| `get_user_alphas` | GET /users/self/alphas | listAlphas | 工具默认 stage=IS，客户端方法默认 stage=OS，两者不一致 |
| `submit_alpha` | POST /alphas/{alphaId}/submit | submitAlpha（未实现 pollAlphaSubmission） | 返回 response.__dict__；失败时返回 False 而非抛出 |
| `value_factor_trendScore` | GET /users/self/alphas?stage=OS&limit=500 + N 次 GET /alphas/{id}（串行）+ GET /users/self/activities/pyramid-multipliers | listAlphas + getAlpha×N + getActivityPyramidMultipliers | N+1 串行请求；listAlphas 结果已含 classifications/pyramids；getActivityPyramidAlphas 已给出按 pyramid 的计数 |
| `get_events` | GET /events | listEvents | 纯透传 |
| `get_leaderboard` | GET /users/self（未传 user_id 时）+ GET /consultant/boards/leader | getUser + listConsultantBoard (boardType=leader) | 每次调用都重复解析 user id；boardType 硬编码 leader；未暴露 limit/offset/order/aggregate |
| `get_operators` | GET /operators | listOperators | 工具层的 list 分支是死代码（客户端方法已包装为 dict） |
| `run_selection` | GET /simulations/super-selection | getSimulationSuperSelection | 名称为 run，实际是只读预览 |
| `get_user_profile` | GET /users/{userId}（默认 self） | getUser（不是 getUserProfile） | 名为 profile，实际调用含 email/telephone 的完整用户记录；目录说明他人 ID 返回 403 |
| `get_documentations` | GET /tutorials | listTutorials | 与 get_documentation_page 可合并；与 brain-rag 的 rag_search(source=tutorial) 功能重叠 |
| `get_messages` | GET /users/self/messages | listSelfMessages | 含本地 base64 图片落盘处理 |
| `get_glossary_terms` | credd 强制刷新 + GET /authentication + Playwright 打开 support.worldquantbrain.com/hc/en-us/articles/4902349883927 | not in catalog（support 站点 HTML 抓取） | 每次启动 Chrome；email/password 为死参数 |
| `search_forum_posts` | credd 强制刷新 + GET /authentication + Playwright GET support.worldquantbrain.com/hc/{locale}/search?page=N | not in catalog（support 站点 HTML 抓取） | 与 brain-rag 的 rag_search 功能重叠 |
| `read_forum_post` | credd 强制刷新 + GET /authentication + Playwright GET support.worldquantbrain.com/hc/zh-cn/community/posts/{id}?page=N | not in catalog（support 站点 HTML 抓取） | 与 brain-rag 的 rag_fetch 功能重叠 |
| `get_alpha_yearly_stats` | GET /alphas/{alphaId}/recordsets/yearly-stats（最多 5 次固定退避重试） | getAlphaRecordset (recordsetType=yearly-stats) | 是 get_record_set_data 的严格子集 |
| `check_correlation` | GET /alphas/{alphaId}/correlations/prod 与/或 /correlations/self（串行，每次最多 5×20s 固定等待） | getAlphaCorrelation | correlation_type 取值为 production/self/both，与 API 的 prod/self/power-pool 不一致；传 'prod' 会静默跳过并返回 all_passed=True |
| `get_submission_check` | check_correlation(both) 的全部请求 + GET /alphas/{alphaId} | getAlphaCorrelation + getAlpha（未使用 getAlphaChecks） | 是 check_correlation + get_alpha_details 的拼接；未调用平台权威检查接口 GET /alphas/{id}/check |
| `set_alpha_properties` | PATCH /alphas/{alphaId} | patchAlpha | 纯透传 |
| `get_record_sets` | GET /alphas/{alphaId}/recordsets | listAlphaRecordsets | 可并入统一的 recordset 工具 |
| `get_record_set_data` | GET /alphas/{alphaId}/recordsets/{recordsetType} | getAlphaRecordset | 无任何重试/Retry-After 处理，而同端点的 get_alpha_pnl/get_alpha_yearly_stats 有重试 |
| `get_user_activities` | GET /users/{userId}/activities?grouping= | listSelfActivities（仅 self、无参数）；docstring 描述的是 getActivityDiversity | user_id 必填但目录只记录 self；grouping 属于 /users/self/activities/diversity |
| `get_pyramid_multipliers` | GET /users/self/activities/pyramid-multipliers | getActivityPyramidMultipliers | 纯透传 |
| `get_pyramid_alphas` | GET /users/self/activities/pyramid-alphas；404 时回退 GET /users/self/pyramid/alphas、GET /activities/pyramid-alphas | getActivityPyramidAlphas（两个回退端点不在目录中，无法验证） | 主端点目录已实测确认；回退分支是遗留代码 |
| `get_user_competitions` | GET /users/self（未传 user_id 时）+ GET /users/{userId}/competitions | getUser + listUserCompetitions | 目录说明 userId 可直接用 self，预查询多余 |
| `get_competition_details` | GET /competitions/{competitionId} | getCompetition | 纯透传 |
| `get_competition_agreement` | GET /competitions/{competitionId}/agreement | not in catalog（可能是 hidden，无法验证） | 未发现 404 处理或回退 |
| `get_platform_setting_options` | OPTIONS /simulations | getSimulationOptions | 本地解析 choices 树 |
| `performance_comparison` | GET /alphas/{alphaId}/performance-comparison | not in catalog（目录中相近的是 getSelfAlphaPerformance / getCompetitionAlphaPerformance） | 无法验证；命名不带 get_ 前缀 |
| `expand_nested_data` | 无 BRAIN 调用（本地 pandas.json_normalize） | not in catalog | 纯本地转换，调用方必须把整份数据作为参数回传 |
| `get_documentation_page` | GET /tutorial-pages/{pageId} | getTutorialPage | 可与 get_documentations 合并 |
| `create_multi_simulation` | POST /simulations（请求体为数组） | createSimulation | 在工具层内联拼接 settings，绕开 BrainApiClient 和 SimulationSettings |
| `get_daily_and_quarterly_payment` | GET /users/self/activities/base-payment + GET /users/self/activities/other-payment | getSelfActivity (activityName=base-payment / other-payment) | 目录要求 base-payment 用 Accept version=3.0，代码未设置 Accept；异常被吞成 'no data'；email/password 为死参数 |
| `lookINTO_SimError_message` | 对每个传入 URL 执行 GET（串行，无 URL 校验） | getSimulation | 功能是 check_simulation_progress 的子集，并会把运行中的模拟误报为错误 |

---

## 六、全部发现明细

严重度按复核后的结果标注。复核结论含义：
- **已确认**：复核 agent 自己核对了代码和目录，结论成立。
- **部分成立**：核心问题成立，但严重度、细节或建议有修正，修正内容附在条目后。
- **复核补充**：复核 agent 在核对时发现的遗漏问题。
- **critic 补充**：完整性 critic 补充的跨组问题。


### AUTH · 认证 / 账户 / 活动 / 顾问榜（20 条：高 1 / 中 5 / 低 14）

#### AUTH-1 ·【高】get_user_activities 声称返回“活动多样性”，实际调用的是分类清单端点，grouping 参数不起作用

- 类别：`wrong-endpoint` · 复核：已确认
- 代码位置：`platform_functions.py:1506-1520（方法），2457-2462（MCP tool）`
- 目录依据：listSelfActivities 第 1197-1319 行；getActivityDiversity 第 1759-1916 行
- 证据：代码：`"""Get user activity diversity data."""` … `params['grouping'] = grouping` … `self._request('get', f"{self.base_url}/users/{user_id}/activities", params=params)`。目录第 1199 行 `GET /users/self/activities` 的参数写的是“无路径或查询参数。”（第 1209 行），响应只有 results[].name 枚举 `base-payment / other-payment / referrals / simulations / submissions`（第 1249-1255 行）。grouping 属于另一个接口：第 1761 行 `GET /users/self/activities/diversity`，参数表第 1773 行为 `grouping | query | 否 | string | … 实测 region、region,delay、dataCategory,region,delay 均返回 200`。另外，目录只记录了 self 路径，/users/{他人id}/activities 未收录，无法验证。
- 影响：LLM 调用 get_user_activities(user_id='self', grouping='region') 想拿各 region/delay 的 alphaCount 和 dataDiversity 检查结果，实际只拿到 5 个分类名（Base Payment 等）。grouping 被服务端忽略，返回的结果与 docstring 完全不一致，Agent 会误以为自己“没有多样性数据”。另外 user_id 是必填参数，而实际只支持 self。
- 建议：把方法改为 `GET /users/self/activities/diversity`，保留 grouping（逗号分隔，文档中写明可选值 dataCategory/region/delay），并删除 user_id 参数或把它固定为 self。如果还需要分类清单，另设 list_activities，或直接把它并入 payment 工具。

#### AUTH-2 ·【中】get_user_profile 调用的是 getUser（完整私有记录），查询他人返回 403；公开档案端点是 /users/{userId}/profile

- 类别：`wrong-endpoint` · 复核：部分成立（原评 高）
- 代码位置：`platform_functions.py:957-967（方法），2259-2272（MCP tool）`
- 目录依据：getUser 第 513-957 行（参数说明在第 527 行）；getUserProfile 第 960-1056 行
- 证据：代码：`response = await self._request('get', f"{self.base_url}/users/{user_id}")`，tool 的 docstring 为 `user_id: User ID (default: "self" for current user)`。目录第 527 行：`userId … 完整用户记录按本人权限校验；使用另一用户 ID 的只读实测请求返回 HTTP 403。` 目录第 963/975 行：`GET /users/{userId}/profile` … `本人及另一用户 ID 的只读实测请求均返回公开档案；只包含公开字段子集。` getUser 的 schema 含 `email`、`telephone`、`address`、`settings`、`auxiliary.campaign` 等字段（第 545-947 行）。
- 影响：传入排行榜上其他用户的 id 时，工具永远返回 {"error": "...403..."}，也就是说它的文档用例全部失败。传 self 时，会把邮箱、电话、住址、营销来源等 PII 整体塞进 LLM 上下文，而 alpha 研究用不到这些字段。
- 建议：user_id 不是 self 时改用 `GET /users/{userId}/profile`（返回 id/address.country/education.university/geniusLevel）。self 也应默认走 profile，或只返回 id、geniusLevel/level 等白名单字段；需要完整记录时另设显式参数，并在文档中说明只对本人有效。
- 复核修正：代码 962 行请求 GET /users/{user_id}。目录第 527 行写明，用另一用户 ID 请求 getUser 实测返回 403；第 975 行写明 /users/{userId}/profile 对本人和他人都返回公开子集。getUser schema 含 email/telephone/address/settings/auxiliary（第 548-936 行）。这些核心事实都成立。但 tool 的默认值是 self，这条最常用的路径能正常工作，只有“查他人”会失败；PII 也只是用户本人的数据进入 LLM 上下文。定为 high 偏重，调整为 medium。
- 修正后的建议：user_id 不是 self 时改用 GET /users/{userId}/profile。self 默认也走 profile，或者只返回 id、geniusLevel/level 等白名单字段；如果确实需要完整记录，再增加显式开关，并注明只对本人有效。

#### AUTH-3 ·【中】value_factor_trendScore 取详情失败时静默 continue，但 N 仍计入该 alpha，导致 S_A 与多样性分数被系统性压低

- 类别：`error-handling` · 复核：部分成立（原评 高）
- 代码位置：`platform_functions.py:837-862`
- 目录依据：getAlpha 第 8482-9330 行（目录未给出 429 说明）
- 证据：代码：`for a in regular:\n    try:\n        detail = await self.get_alpha_details(a.get('id'))\n    except Exception:\n        continue` 之后是 `N = len(regular)` 和 `S_A = (A / N) if N > 0 else 0.0`。get_alpha_details 会对任何 4xx/5xx 执行 `response.raise_for_status()`。
- 影响：一次对几百个 alpha 顺序请求 /alphas/{id}，其中只要一部分因限流、超时或 5xx 失败，这些 alpha 就既不计入 A，也不计入 pyramid，却仍留在分母 N 里。结果 S_A 偏低、P 偏低，diversity_score 可能被低估一个数量级，而且返回值里没有任何提示。Agent 会据此做出错误的提交策略判断。
- 建议：直接使用 listAlphas 结果中自带的 classifications/pyramids（见 AUTH-6），彻底消除逐个请求详情。如果仍要逐个请求，应统计 failed_ids，把 N 改为成功条数或在返回中加 `incomplete: true, failed: k`；对 429 按 Retry-After 退避重试，不要吞掉异常。
- 复核修正：837-841 行对 get_alpha_details 的异常直接 continue，而 get_alpha_details（537-547 行）遇到非 2xx 会 raise_for_status 并抛出；861 行 N = len(regular) 仍把失败的 alpha 计入分母，因此 S_A 被低估，P/S_H 也基于不完整样本。以上已核实。CreddSession 只对 401 做一次自愈，没有 429/Retry-After 处理，所以限流时确实会出现这种情况。不过这是启发式指标工具，偏差大小取决于失败比例，“低估一个数量级”属于夸大，调整为 medium。
- 修正后的建议：优先按 AUTH-6 直接使用 listAlphas 结果中的字段，去掉逐个取详情。如果保留逐个请求，应统计 failed_ids，并以成功条数作为 N，或在返回中加入 incomplete/failed_count；遇到 429 时按 Retry-After 做有限次重试。

#### AUTH-4 ·【中】value_factor_trendScore 只取 listAlphas 的第一页（limit=500），不检查 count/next，窗口内 alpha 多时静默截断

- 类别：`pagination` · 复核：部分成立（原评 高）
- 代码位置：`platform_functions.py:825-831（调用 get_user_alphas，定义在 673-712）`
- 目录依据：listAlphas 第 7592-8480 行（参数表 7602-7611，运行行为 8465-8473）
- 证据：代码：`alphas_resp = await self.get_user_alphas(stage='OS', limit=500, submission_start_date=start_date, submission_end_date=end_date)`，接着 `alphas = alphas_resp['results']`，没有读取 `count` 或 `next`。目录第 7606 行：`limit | query | 否 | integer | 默认 10 | 可选；实测非正数或非数字会被忽略并回退到默认每页 10 项。`；运行行为为 `"pagination": "limit_offset", "defaultLimit": 10, "invalidNumericPagination": "falls_back_to_default"`，响应 schema 含 `count` / `next`（第 7630-7640 行）。目录没有给出 limit 上限。
- 影响：只要服务端对 limit 设有上限（目录未记录上限，无法排除），或窗口内提交数超过 500，N、A、P 都只基于部分样本计算，返回值却看不出任何截断，趋势分数因此失真。
- 建议：按 limit_offset 循环分页，直到 `next` 为空或累计条数达到 `count`（每页取服务端实际接受的大小）。在返回中附上 `count` 和 `fetched`，两者不一致时报告 truncated。另外把 REGULAR 过滤下推为服务端参数 `type=REGULAR`（目录第 7609 行的枚举），减少翻页量。
- 复核修正：825-831 行只调用一次 get_user_alphas(limit=500)，没有读取 count/next。目录 listAlphas（第 7606 行，运行行为第 8465-8473 行）为 limit_offset 分页，响应含 count/next，但没有记录 limit 上限。“无分页、可能静默截断”这一点属实。不过 OS 窗口通常只有几天，受平台提交频率限制，超过 500 条的情况较少；如果服务端对 limit 有上限并返回 400，工具会显式报错，而不是静默截断。因此 high 偏重。
- 修正后的建议：按 offset 循环分页，直到 next 为空或累计条数达到 count；把 type=REGULAR（目录第 7609 行枚举）下推到服务端；在返回中附上 count/fetched，不一致时标注 truncated。

#### AUTH-6 ·【中】value_factor_trendScore 对每个 alpha 顺序调用 /alphas/{id}（N+1），但 listAlphas 的结果项已经包含 classifications/pyramids/pyramidThemes/tags

- 类别：`concurrency-perf` · 复核：已确认
- 代码位置：`platform_functions.py:837-859`
- 目录依据：listAlphas 结果项 schema：classifications 第 7826-7843 行，pyramids 第 8386-8394 行，pyramidThemes 第 8396-8398 行
- 证据：代码：`for a in regular:\n    try:\n        detail = await self.get_alpha_details(a.get('id'))` 逐个 await，无并发，然后读取 `detail.get('classifications')`、`detail.get('tags')`、`detail.get('pyramids')`、`detail.get('pyramidThemes')`。目录中 listAlphas 的 results[] 项已包含 `"classifications": {"type": "array", …}`、`"pyramids": {"type": ["array","null"] …}`、`"pyramidThemes": {"type": "object" …}`，与 getAlpha 的字段一致（第 9233-9245 行）。
- 影响：几百个 alpha 就要发几百个串行 HTTP 请求（每个受 60s 读超时约束），单次调用可能耗时数分钟，还会提高触发限流的概率，进而引出 AUTH-3 的计数偏差。这些请求完全多余。
- 建议：直接用 listAlphas 返回的 `results` 调用 `_is_atom(a)` 并提取 pyramids，删除逐个取详情的循环。如果某些字段确实只在详情里才有，也应改用 asyncio.gather 加 Semaphore 做有界并发。

#### AUTH-10 ·【中】get_daily_and_quarterly_payment 把所有异常（401/403/5xx/超时）都吞成字符串 "no data"

- 类别：`error-handling` · 复核：已确认
- 代码位置：`platform_functions.py:2690-2727`
- 目录依据：getSelfActivity 第 1322-1680 行
- 证据：代码：`try:\n    base_response = await brain_client._request('get', …/base-payment)\n    base_response.raise_for_status()\n    base_payments = base_response.json()\nexcept Exception:\n    base_payments = "no data"`，other-payment 的写法相同。目录中 activityName 枚举只列出 `base-payment / other-payment / referrals / simulations / submissions`（第 1336 行），成功状态只有 `HTTP 200`。
- 影响：credd 在退避中、cookie 失效、请求超时或权限不足时，Agent 看到的都是“没有收益数据”，无法和真正的零收益区分，可能据此做出错误结论（例如认为 alpha 没有在生产中产生收益）。返回类型也在 dict 和 str 之间变化，不利于程序解析。
- 建议：保留错误信息，例如 `{"error": str(e), "http_status": …}`，只有 200 且内容为空时才表示无数据；两个请求用 asyncio.gather 并发执行；返回前裁剪 records.records（例如只保留最近 N 天，或给出汇总字段 yesterday/current/previous/ytd/total），避免上下文过大。

#### AUTH-5 ·【低】value_factor_trendScore 依赖的 dateSubmitted>/dateSubmitted< 过滤参数不在目录中，窗口是否生效无法验证

- 类别：`wrong-params` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:825；get_user_alphas 699-702`
- 目录依据：listAlphas 参数表第 7604-7611 行
- 证据：代码：`if submission_start_date:\n    params["dateSubmitted>"] = submission_start_date` / `params["dateSubmitted<"] = submission_end_date`。目录 listAlphas 的参数表只有 `limit`、`offset`、`order`、`type`、`stage`、`status` 六项，没有 dateSubmitted 相关过滤；并且目录说明无效的 type/stage/status 值“返回 HTTP 200 空列表”，也就是服务端对未知输入比较宽松。
- 影响：目录不收录 hidden 接口和参数，所以不能断定这个参数无效。但如果服务端忽略它，函数统计的就是全部 OS alpha，而不是指定日期窗口，“趋势”失去意义，而且不会有任何报错。
- 建议：在返回中校验窗口：对每条 result 的 `dateSubmitted`（目录第 7852 行有此字段）再做一次客户端过滤，并在结果里报告被过滤掉的条数。更好的做法是改用目录明确支持日期窗口的 `getActivityPyramidAlphas`（startDate/endDate，第 1928-1934 行）来计算 pyramid 分布。
- 复核修正：699-702 行确实发送 dateSubmitted>/dateSubmitted<，目录参数表（第 7604-7611 行）只有 limit/offset/order/type/stage/status。但目录第 8478 行的源码证据注明“URL 在局部变量 h 中动态构造”，前端动态拼接的过滤参数本来就可能没被收录；同一函数里的 dateCreated> 也不在表中。这类过滤写法在 BRAIN 社区脚本中很常见。审计本身也承认无法断定无效，所以风险只是“未验证”。客户端再按 dateSubmitted 复核一次成本很低，可以作为防御措施，但严重度应为 low。
- 修正后的建议：对返回结果按 dateSubmitted 在客户端再过滤一次，并报告被过滤的条数，作为防御性校验；计算 pyramid 分布时可改用 getActivityPyramidAlphas（startDate/endDate）。

#### AUTH-7 ·【低】value_factor_trendScore 在客户端近似计算“value factor”，而目录中已有返回官方 valueFactor 和金字塔分布的端点；docstring 把近似值说成定义本身

- 类别：`better-endpoint-available` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:794-822（docstring），2161-2176（MCP tool）`
- 目录依据：getConsultantPerformance 第 2812-2953 行；getActivityPyramidAlphas 第 1918-2024 行；getActivityDiversity 第 1759-1916 行
- 证据：docstring：`value factor of a user is defiend by This diversity score, which measures … (S_A) … (S_P) … (S_H). Calculated as their product`。目录第 2814 行 `GET /users/{userId}/consultant` 的响应 `leaderboard` 含 `"weightFactor"`、`"valueFactor"`、`"dataFieldsUsed"`、`"submissionsCount"`、`"meanProdCorrelation"`、`"meanSelfCorrelation"`（第 2854-2890 行）。第 1920 行 `GET /users/self/activities/pyramid-alphas` 支持 `startDate`/`endDate`（date），并返回 `pyramids[].{category,region,delay,alphaCount}`。
- 影响：Agent 会把自行推导的 S_A·S_P·S_H 当成平台的 value factor 来优化，可能与真实值方向相反；而且为此付出 N+1 请求的代价。平台已直接给出 valueFactor，以及按 category×region×delay 统计的窗口内 alphaCount。
- 建议：新增或替换为 get_consultant_performance 工具（GET /users/self/consultant），直接返回 valueFactor/weightFactor 等指标。pyramid 覆盖度和熵改用 getActivityPyramidAlphas 的 alphaCount 计算（P = alphaCount>0 的条目数，P_max = pyramid-multipliers 条目数，二者键一致）。docstring 应明确写成“启发式代理指标，并非平台 valueFactor”。
- 复核修正：目录第 2814 行 GET /users/{userId}/consultant 的 leaderboard 中确实有 valueFactor/weightFactor 等字段，第 1920 行 pyramid-alphas 支持 startDate/endDate。docstring（799 行）把 S_A·S_P·S_H 说成 value factor 的定义，确实有误导性。但官方 valueFactor 是当前时点的累计值，这个工具计算的是指定窗口内的多样性趋势，两者口径不同，不能简单“替换”。另外目录第 2950 行注明路径按权限在 consultant 和 researcher 之间动态选择，非顾问用户可能拿不到 valueFactor。重点应放在修正文档和新增官方指标工具上，定为 low 更合适。
- 修正后的建议：把 docstring 改为“窗口内多样性启发式代理指标，并非平台 valueFactor”。另外新增 get_consultant_performance（GET /users/self/consultant，注明仅顾问可用）返回官方 valueFactor。窗口内的 pyramid 覆盖度和熵可改用 pyramid-alphas 的 alphaCount 计算。

#### AUTH-8 ·【低】P 与 P_max 的口径可能不一致，且 pyramid-multipliers 失败时 P_max 退化为 P，S_P 静默变为 1.0

- 类别：`response-parsing` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:847-876`
- 目录依据：getActivityPyramidMultipliers 第 2028-2131 行；listAlphas pyramids 项第 8386-8394 行
- 证据：代码：`ps = [p.get('name') for p in detail.get('pyramids') if p.get('name')]`（按 name 计数得到 P），`P_max = len(pyramids_list)`（pyramid-multipliers 的条目数），以及 `except Exception:\n    P_max = None … if not P_max or P_max <= 0:\n    P_max = max(P, 1)`。目录 pyramid-multipliers 的条目为 `category{id,name}`、`region`、`delay`、`multiplier`，没有 `name` 字段（第 2058-2090 行）；alpha 的 pyramids 项只写了 `"items": {"type": "object", "additionalProperties": true}`（第 8386-8394 行），字段形状目录未说明。
- 影响：(1) 如果 alpha.pyramids[].name 与 multiplier 条目（category×region×delay）不是一一对应，S_P 的分子分母口径不同，可能大于 1 或被系统性低估，这一点目录无法验证。(2) 只要 multipliers 请求失败（例如 401 后 credd 不可用），S_P 就恒为 1.0，分数被抬高，调用方却看不到任何提示。
- 建议：用同一套键（category.id+region+delay）来统计 P 和 P_max，最好两者都取自 pyramid-alphas 与 pyramid-multipliers。multipliers 失败时应返回 error 或在结果中写 `P_max_source: "fallback"`，不要静默令 S_P=1。
- 复核修正：第 2 点已核实：867-876 行在 get_pyramid_multipliers 失败时令 P_max=None，随后退化为 max(P,1)；只要 P>=1，S_P 就恒为 1.0，而且返回中没有任何提示。第 1 点只是推测：目录中 alpha.pyramids 的 items 为 additionalProperties:true（第 8386-8394 行），没有给出字段形状，无法证明 name 与 category×region×delay 口径不一致；实际上两者很可能都是按 region/delay/category 这一粒度计数的。
- 修正后的建议：multipliers 失败时返回 error，或在结果中加入 P_max_source:"fallback"，并将 S_P 标记为不可靠。P 与 P_max 最好都基于 pyramid-alphas 和 pyramid-multipliers，用同一套键（category.id+region+delay）计算。

#### AUTH-9 ·【低】本组所有 BRAIN 请求都不发送目录要求的 Accept 版本头；base-payment 明确要求 version=3.0

- 类别：`missing-header` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:230-232（_build_session 只设了 User-Agent）；2707、2715（payment 调用）`
- 目录依据：getSelfActivity 第 1328、1336、1661-1675 行；getAuthentication 第 126 行；getActivityPyramidMultipliers/PyramidAlphas/listConsultantBoard 等均为 version=2.0
- 证据：代码：`session.headers.update({'User-Agent': 'Mozilla/5.0 …'})`，整个文件中没有 Accept 头（grep 'Accept' 无结果）。`base_response = await brain_client._request('get', f"{brain_client.base_url}/users/self/activities/base-payment")`。目录第 1328 行：`Accept：application/json;version=2.0 / application/json;version=3.0`；第 1336 行：`base-payment 使用 Accept version=3.0，其余实测分类使用 version=2.0`；运行行为：`"acceptByParameter": {… "values": {"base-payment": "3.0"}, "fallback": "2.0"}`。
- 影响：不带版本头时服务端返回哪个版本的表示，目录没有说明。对于 base-payment，可能拿到旧版 schema（例如缺少 regularAlpha/superAlpha 拆分或 records.schema），被 LLM 误读；一旦服务端升级默认版本，所有工具的返回形状都可能在没有任何告警的情况下改变。
- 建议：在 _request 中支持按调用传入 `headers={'Accept': 'application/json;version=2.0'}`，默认发送 2.0，base-payment 调用单独使用 3.0，按目录逐一对齐各 operation 的版本。
- 复核修正：grep 确认整个文件没有设置 Accept 头，_build_session 只设了 User-Agent（230-232 行）。目录第 1328/1336 行及运行行为第 1661-1675 行注明前端对 base-payment 使用 version=3.0，其余使用 2.0。但目录没有说明不带版本头时服务端返回什么，也没有任何证据表明当前返回的数据有误或缺字段，影响属于推测。建议值得采纳，但 medium 偏重。
- 修正后的建议：_request 支持按调用传入 headers。至少对 base-payment 显式发送 Accept: application/json;version=3.0；其余端点可以按目录逐步对齐，并在改动后实测返回形状。

#### AUTH-11 ·【低】get_leaderboard 取当前用户 id 失败时静默改为查询未过滤榜单（默认第一页 10 行），并当作“我的排名”返回

- 类别：`error-handling` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:743-764`
- 目录依据：listConsultantBoard 第 18876-18990 行（参数表 18890-18898）
- 证据：代码：`user_response = await self._request('get', f"{self.base_url}/users/self")\nif user_response.status_code == 200:\n    …params['user'] = user_data.get('id')` 没有 else 分支，随后 `self._request('get', f"{self.base_url}/consultant/boards/leader", params=params)`。目录：`limit | query | 否 | integer | 默认 10`，`user | query | 否 | string`，运行行为为 `"pagination": "limit_offset"`。
- 影响：/users/self 返回非 200 时（例如 401 且 credd 无新 cookie），函数不报错，而是返回全站榜单前 10 名，LLM 会把别人的指标误当成自己的。另外，为了拿一个 id 要多发一次返回完整 PII 的 getUser 请求；get_user_competitions（1582-1588 行）里也重复了同样的查询逻辑。
- 建议：取 id 失败时直接报错；用户 id 从 GET /authentication 的 `user.id` 获取（目录第 150-160 行）并缓存到 client 上，供 leaderboard/competitions 共用。更进一步，查询“自己的排名”时直接调用 getConsultantPerformance（GET /users/self/consultant），省掉一次查找。
- 复核修正：754-757 行在 /users/self 返回非 200 时没有 else 分支，params 为空，随后请求 /consultant/boards/leader，会得到默认 10 行的未过滤榜单（目录第 18893 行 limit 默认 10）。这一点属实。但审计举的 401 场景不成立：如果 cookie 失效且 credd 无法刷新，随后的 boards 请求同样会 401，raise_for_status 会抛错，最终返回 error。只有 /users/self 出现 5xx 或 403、而 boards 请求成功这种较少见的情况，才会静默返回别人的数据。同样的查询逻辑在 get_user_competitions（1582-1589 行）中也重复出现，这一点属实。
- 修正后的建议：取 id 失败时直接报错；从 GET /authentication 的 user.id 获取用户 id 并缓存，供 leaderboard/competitions 共用。

#### AUTH-12 ·【低】CreddSession 在并发 401 时，每个线程都会串行地向 credd 拉一次 cookie（惊群），没有先检查其他线程是否已经刷新

- 类别：`concurrency-perf` · 复核：已确认（原评 中）
- 代码位置：`platform_functions.py:122-148`
- 目录依据：n/a（credd 为本地守护进程；BRAIN 侧对应 getAuthentication 的 cookie_session 鉴权，第 125 行）
- 证据：代码：`version_used = self._cookie_version` → `resp = super().request(...)` → 401 时 `new_version = self.refresh_cookies()`。而 refresh_cookies 的实现是 `with self._refresh_lock:\n    cookies, version = self._fetch_from_credd()`，拿到锁后无条件请求 credd，只在请求之后才比较 `new_version == version_used`。
- 影响：cookie 过期时，线程池里最多 32 个（WQMCP_HTTP_WORKERS）同时收到 401 的请求会在锁上排队，逐个执行 GET /cookies（每个超时 15s）。第一个线程已经拿到新 cookie，后面的线程还会重复拉取，增加尾延迟，也可能触发 credd 的 `rate_limited`/`backoff`（代码第 117 行正在处理这些情况），之后的重试反而失败。
- 建议：让 refresh_cookies 接收 `seen_version`：拿到锁后如果 `self._cookie_version != seen_version`，说明别的线程已经刷新过，直接返回当前版本，不再请求 credd。request() 中传入 version_used。

#### AUTH-13 ·【低】get_authentication_status 实际读取的是完整 getUser 档案，并与 is_authenticated 在 manage_config 中重复请求；失败时返回 None 掩盖错误

- 类别：`redundancy` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:412-420；manage_config 1830-1838`
- 目录依据：getAuthentication 第 119-219 行；getUser 第 513-957 行
- 证据：代码：`async def get_authentication_status(self) … response = await self._request('get', f"{self.base_url}/users/self")` … `except Exception as e: … return None`；manage_config 中 `auth_status = await brain_client.get_authentication_status()` 之后又执行 `"is_authenticated": await brain_client.is_authenticated()`（GET /authentication）。目录中 getAuthentication 的响应已包含 `user.id`、`token.expiry`、`permissions`（第 146-190 行）。
- 影响：一次 manage_config(get) 要发两个串行请求，还会把邮箱、电话、住址等 PII（getUser schema）返回给 LLM。auth_status 为 None 时看不出是 credd 挂了、401 还是网络错误。authenticate、is_authenticated、get_authentication_status、ensure_authenticated 四个方法职责重叠。
- 建议：合并为一个 get_auth_state()：只调用 GET /authentication，返回 {authenticated, user_id, token_expiry, permissions, error}；manage_config 复用它。删除 get_authentication_status，或把它改为调用 getAuthentication。
- 复核修正：412-420 行 get_authentication_status 请求 /users/self（完整 getUser 记录），异常时返回 None；manage_config（1832-1837 行）还会再串行调用 is_authenticated（GET /authentication）。已核实。目录 getAuthentication（第 146-190 行）已包含 user.id/token.expiry/permissions，可以合并。但这只涉及 manage_config(get) 这一个低频入口，暴露的也是用户本人的 PII，定为 medium 偏重。

#### AUTH-14 ·【低】get_pyramid_alphas 的两个 404 回退路径没有依据，错误提示中还列出了从未请求过的 /pyramid/alphas

- 类别：`dead-code` · 复核：已确认
- 代码位置：`platform_functions.py:1546-1569`
- 目录依据：getActivityPyramidAlphas 第 1918-2024 行
- 证据：代码：`if response.status_code == 404:\n    response = await self._request('get', f"{self.base_url}/users/self/pyramid/alphas", …)\n    if response.status_code == 404:\n        response = await self._request('get', f"{self.base_url}/activities/pyramid-alphas", …)`，而 `"tried_endpoints": [… "/pyramid/alphas"]` 中的 /pyramid/alphas 实际没有被请求。目录第 1920 行记录的主路径 `GET /users/self/activities/pyramid-alphas` 为“实测响应确认”（第 1925 行）；两个回退路径不在目录中（目录不含 hidden 接口，只能判定为“未收录、无法验证”）。
- 影响：主路径已被实测确认，回退分支基本不可达。万一主路径因其他原因 404（例如参数格式错误），会多发两个无意义的请求，并返回一段误导性的“端点在当前 API 版本不可用”建议和一份不准确的尝试清单。
- 建议：删除两个回退分支，改为对主路径的非 2xx 响应返回 {http_status, body}；如果要保留回退，至少让 tried_endpoints 如实反映实际请求过的路径。

#### AUTH-15 ·【低】get_pyramid_alphas 的 startDate/endDate 在目录中是 date 格式，但工具没有说明格式，同组工具的文档还在引导 LLM 传 ISO datetime

- 类别：`wrong-params` · 复核：已确认
- 代码位置：`platform_functions.py:1534-1544；MCP tool 2473-2479`
- 目录依据：getActivityPyramidAlphas 参数表第 1928-1934 行
- 证据：代码：`if start_date:\n    params['startDate'] = start_date` 不做格式转换；tool 的 docstring 只有 `"""Get user's current alpha distribution across pyramid categories."""`。目录：`startDate | query | 否 | string | 格式 date`，`endDate | query | 否 | string | 格式 date`。同组 value_factor_trendScore 的文档示例为 `'2025-08-14T00:00:00Z'`。
- 影响：LLM 很可能传入 `2025-08-14T00:00:00Z`。服务端如何处理 datetime，目录没有说明（可能 400、被忽略，或按不同时区截断），最终可能返回全量区间而不报错。
- 建议：在 docstring 中写明 YYYY-MM-DD，并在方法里把带 'T' 的输入截取为日期部分；说明省略时的默认窗口（目录未说明，需实测）。

#### AUTH-16 ·【低】authenticate 与 get_daily_and_quarterly_payment 仍暴露会被忽略的 email/password 参数，诱导 LLM 在工具调用中传明文密码

- 类别：`security` · 复核：已确认
- 代码位置：`platform_functions.py:1797-1816、2690-2701、355-358`
- 目录依据：createAuthentication 第 222-318 行（鉴权方式为 basic，代码并不使用）
- 证据：代码：`async def authenticate(email: Optional[str] = "", password: Optional[str] = "")`，docstring 为 `password: Ignored (kept for backward compatibility; credentials live in credd)`；`async def get_daily_and_quarterly_payment(email: str = "", password: str = "")`。MCP 不调用 POST /authentication（目录第 224-228 行，鉴权为 `basic`）。
- 影响：这些参数没有任何作用，却出现在工具 schema 中。模型可能向用户索要密码，或把配置里的密码填进参数，导致密码进入 MCP 调用日志和对话记录，而登录本来完全由 credd 负责。
- 建议：从两个工具的签名中删除 email/password。如果需要兼容旧调用，在服务端丢弃这些参数，并且不在 schema 中暴露 password 字段。

#### AUTH-17 ·【低】authenticate/is_authenticated 只把 200 视为已认证，而目录中 GET /authentication 的 200 与 204 都是成功；permissions 被丢弃

- 类别：`response-parsing` · 复核：部分成立
- 代码位置：`platform_functions.py:367-385、396-401`
- 目录依据：getAuthentication 第 138-190 行
- 证据：代码：`if response.status_code == 200: … return {'user': …, 'token_expiry': …}` 否则 `raise Exception(f"credd cookie did not pass BRAIN validation (HTTP {response.status_code}) …")`；is_authenticated 为 `return response.status_code == 200`。目录：`成功状态：HTTP 200、HTTP 204`，`响应模式：json_or_empty`，schema 必填字段含 `"permissions"`。
- 影响：如果服务端返回 204（目录已确认存在这种成功状态），工具会报告“credd cookie 未通过校验”，并建议用户去做生物识别，属于误报。permissions（例如 CONSULTANT）决定了顾问类接口能否使用（目录第 2950 行说明“路径由权限动态选择 consultant 或 researcher”），这里却没有暴露给 Agent。
- 建议：把 200 和 204 都视为成功（204 时 user/expiry 为空）；在返回中加入 `permissions`，供后续工具选择 consultant 或 researcher 路径。
- 复核修正：368 行和 401 行只把 200 视为已认证。目录第 140-142 行写的是成功状态 HTTP 200、HTTP 204，响应模式 json_or_empty；返回中也确实丢弃了 permissions（第 175-181 行）。但目录没有说明 204 在什么情况下出现（也可能是前端的兜底处理），实际能否遇到 204 无法确认。影响有限，维持 low。
- 修正后的建议：把 204 也视为成功，但注明此时 user/expiry 可能为空；在返回中加入 permissions。

#### AUTH-18 ·【低】value_factor_trendScore 的 MCP docstring 写了并不存在的 p_max 参数

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`platform_functions.py:2161-2176`
- 目录依据：n/a
- 证据：tool 签名为 `async def value_factor_trendScore(start_date: str, end_date: str)`，docstring 却写着 `- p_max: optional integer total number of pyramid categories for normalization`；方法 docstring 则说 `P_max … is derived from the platform pyramid-multipliers endpoint and not supplied by callers.`
- 影响：LLM 按文档传入 p_max 时，会触发参数校验错误，或者以为归一化基数可以手动调整。
- 建议：删除 docstring 中的 p_max 说明（或者真正实现这个参数），并与方法 docstring 保持一致。

#### AUTH-19 ·【低】get_leaderboard 把 boardType 写死为 leader，且没有暴露 limit/offset/order/aggregate，无法翻页，也看不到其他榜单

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`platform_functions.py:759；MCP tool 2194-2206`
- 目录依据：getConsultantBoardOptions 第 18805-18870 行；listConsultantBoard 第 18876-18990 行
- 证据：代码：`self._request('get', f"{self.base_url}/consultant/boards/leader", params=params)`，tool 只有 `user_id` 一个参数，docstring 为 `🏅 Get leaderboard data.`。目录：`boardType | path | 是 | string | 枚举 leader / spc / power-pool / referral`，`order | … 可使用 - 前缀降序`，`aggregate | … 枚举 user / university / country`，`limit | 默认 10`。
- 影响：Agent 无法查看 power-pool/spc 榜单，也无法按国家或大学聚合、按字段排序或翻页。user_id 不是 self 时的结果行字段因 board 而异（目录第 18950 行附近有说明），但工具没有给出任何说明。
- 建议：增加 board_type（枚举）、limit、offset、order、aggregate 参数，并在 docstring 中注明可用的 order 字段来自 OPTIONS /consultant/boards/{boardType}；或者拆出一个专门的 get_my_consultant_performance（见 AUTH-7）。

#### AUTH-20 ·【低】ensure_authenticated 与 _request 内部的 _ensure_session 完全重复；authenticate 首次调用会向 credd 连续拉两次 cookie

- 类别：`redundancy` · 复核：部分成立
- 代码位置：`platform_functions.py:406-410、249、361-365`
- 目录依据：n/a
- 证据：代码：`async def ensure_authenticated(self):\n    await self._ensure_session()`，而 `_request` 第一行就是 `session = await self._ensure_session()`。authenticate 中 `session = await self._ensure_session()`（冷启动时 CreddSession.__init__ 会调用 refresh_cookies）之后又执行 `await loop.run_in_executor(self._executor, session.refresh_cookies)`。
- 影响：每个方法开头的 `await self.ensure_authenticated()` 都是空操作，给人一种“每次调用前都做了认证检查”的错觉。冷启动时 authenticate 会让 credd 连续响应两次；如果 GET /authentication 再返回 401，还会有第三次拉取。
- 建议：删除 ensure_authenticated 及各处调用（或在文档中明确说明它只是懒加载）；authenticate 在会话刚创建时跳过强制 refresh。
- 复核修正：ensure_authenticated（406-410 行）只调用 _ensure_session，与 _request 第一行（249 行）重复。但它的 docstring 已明确写着“Intentionally cheap: no ... per-call network probe”，谈不上“给人错觉”，只是冗余。冷启动时 authenticate 会重复拉取 cookie，已核实：_ensure_session 创建 CreddSession 时，__init__ 会调用一次 refresh_cookies（93 行），365 行又强制 refresh 一次。
- 修正后的建议：删除 ensure_authenticated 及各处调用（或保留并注明只做懒加载）；authenticate 仅在会话已经存在时才强制 refresh，刚创建的会话跳过这一步。


### DATA · 数据集 / 数据字段 / 算子 / 搜索（11 条：高 2 / 中 3 / 低 6）

#### DATA-1 ·【高】get_datafields 把 limit=50、offset=0 写死，大数据集的字段被静默截断，而且无法翻页

- 类别：`pagination` · 复核：已确认
- 代码位置：`platform_functions.py:582-590（params），2031-2062（工具签名没有 limit/offset）`
- 目录依据：listDataFields，目录 4319-4320（limit/offset），4556-4557（响应 required count/results）
- 证据：代码：`'limit': '50',
'offset': '0'`；MCP 工具签名只有 instrument_type/region/delay/universe/theme/dataset_id/data_type/search，没有任何分页参数。目录 4319：`| limit | query | 否 | integer | 默认 20；范围 1..+∞ | 每页数量。` 4320：`| offset | query | 否 | integer | 默认 0；范围 0..+∞ | 起始偏移。` 响应 schema 要求返回 `"count"` 和 `"results"`（4556-4557），也就是说服务端支持 limit/offset 分页，并会告诉你总数。
- 影响：一个数据集的字段数超过 50 时（例如大型基本面/模型数据集常有几百个字段），调用方只能看到前 50 个。count 虽然也返回了，但工具没有 offset 参数，LLM 代理既取不到后面的页，也不会得到截断提示，extraNote 只提示“结果为 0 时检查参数”。结果是字段探索系统性漏掉大部分字段，alpha 研究的覆盖面被严重缩小。search 返回的结果也同样只有前 50 条。
- 建议：在客户端方法和 MCP 工具里暴露 limit（默认 50）和 offset；或者加一个 fetch_all/max_fields 参数，按 count 用有界并发（例如 asyncio.gather 加 Semaphore）拉取后续 offset 页。响应里追加 `has_more = offset+len(results) < count` 和 `next_offset` 提示，让 LLM 明确知道还有下一页。

#### DATA-2 ·【高】get_datasets 不发送 limit/offset，只能拿到服务端默认的前 20 个数据集

- 类别：`pagination` · 复核：部分成立
- 代码位置：`platform_functions.py:555-566；工具 2002-2028`
- 目录依据：listDatasets，目录 3346-3347，3599-3605
- 证据：代码 params 只有 `'instrumentType','region','delay','universe','theme'`（外加可选 search），没有 limit/offset。目录 3346：`| limit | query | 否 | integer | 默认 20；范围 1..+∞ | 每页数量。` 目录 3601-3604 运行行为：`{"pagination": "limit_offset"}`。
- 影响：一个 region/universe/delay 组合通常有几十到上百个数据集，工具每次只返回服务端默认的第一页（20 个），也不告诉调用方还有更多。代理会以为可用数据集只有这 20 个，漏掉大量数据集，包括 pyramidMultiplier 高、alphaCount 低的“冷门”数据集。
- 建议：加 limit/offset 参数（以及 order，见 DATA-7），返回里附上 count/has_more/next_offset；或者提供 fetch_all 选项，按 count 分页并发拉取，同时为减小体积裁剪 description/researchPapers 等长字段。
- 复核修正：代码 555-566 确实没发 limit/offset。目录 3346-3347 写明默认 limit=20，3601-3604 写明 pagination 是 limit_offset，所以工具只能拿到第一页，也无法翻页，核心成立。需要更正的是“也不告诉调用方还有更多”这一说法：响应 JSON 原样返回（568-570），schema 含 count（3383），调用方能看到 count 大于 len(results)，只是没有办法继续取。
- 修正后的建议：加 limit（默认调大，例如 50）和 offset 参数，并提供 order；返回里明确附上 has_more/next_offset；如需全量，按 count 有界并发分页，同时裁剪 description/researchPapers 等长字段。

#### DATA-4 ·【中】get_datafields 的 theme 参数被接收但从未发送，属于误导性的空参数

- 类别：`dead-code` · 复核：已确认
- 代码位置：`platform_functions.py:576, 582-597；工具 2036, 2051`
- 目录依据：listDataFields，目录 4334
- 证据：签名：`theme: str = "false",`；工具 docstring：`theme: Theme filter`；但 params 构造（582-597）只写了 instrumentType/region/delay/universe/limit/offset/type/dataset.id/search，没有 'theme'。目录 4334：`| theme | query | 否 | string | — | 主题筛选。 |`
- 影响：代理按 docstring 传 theme 想筛选主题（例如比赛或主题乘数相关字段）时，参数被悄悄丢掉，返回的是未过滤的结果，代理会误以为已经按主题过滤。目录明确支持这个过滤器，所以这是一个本可用却失效的能力。
- 建议：只有在 theme 非空且不是遗留的 "false" 时才发送 `params['theme'] = theme`，默认值改为 None；或者直接删掉该参数并更新 docstring。get_datasets 也要一起处理（见 DATA-5）。

#### DATA-7 ·【中】数据集/字段工具没有暴露目录中对研究最有价值的过滤和排序参数，docstring 也不完整

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`platform_functions.py:549-561, 575-597；工具 2002-2028, 2031-2062`
- 目录依据：listDatasets 3353-3366；listDataFields 4330-4340
- 证据：代码只支持 instrument_type/region/delay/universe/theme/search（datafields 另有 dataset_id/data_type）。目录 listDataFields 还支持 `alphaCount>`/`alphaCount<`（4330-4331 附近）、`category`、`subcategory`、`coverage>`/`coverage<`、`dateCoverage>`/`<`、`dateCreated>`/`<`、`order`（4340：`排序字段；前缀 - 表示降序。`）；listDatasets 还支持 `valueScore>`、`alphaCount<`、`category`、`coverage>`、`order` 等（3353-3366）。另外，get_datasets 工具的 docstring 没有 search 参数的说明（2016-2021 只列了 instrument_type..theme）。
- 影响：alpha 研究代理最常见的需求是“按 coverage 过滤、找 alphaCount 低/userCount 低的冷门字段、按 valueScore 排序数据集”。现在只能靠截断后的前 20/50 条在客户端判断（结合 DATA-1/2，基本无法实现），导致重复挖掘拥挤字段、错过高乘数数据集。
- 建议：新增可选参数：category、subcategory、order、coverage_min/max、alpha_count_max/min、value_score_min（datasets），映射到目录里的 `coverage>` 等参数名；docstring 写明 coverage 取值范围是 0–1（目录 3360 说明）以及 order 的 '-' 降序语义；补上 search 参数说明。

#### DATA-8 ·【中】get_operators 一次返回全部算子，不能按 scope/category 过滤，也没有缓存

- 类别：`tool-api-design` · 复核：部分成立
- 代码位置：`platform_functions.py:910-926；工具 2213-2226`
- 目录依据：listOperators，目录 5064-5171（schema 5083-5140，示例 5143-5157）
- 证据：代码：`response = await self._request('get', f"{self.base_url}/operators")`，然后原样返回整个列表。目录 schema 中每个算子都有 `"scope"`（数组，示例 `"REGULAR","COMBO","SELECTION"`）、`"category"`、`"definition"`、`"description"`、`"documentation"`、`"level"`，并且 `"additionalProperties": false`；运行行为（5162-5164）：`没有从前端确认到额外的分页、轮询或重定向规则。`
- 影响：每次调用都把完整的算子列表（含 description/documentation）塞进 LLM 上下文，而且每次都访问网络。代理可能在 REGULAR 表达式里用了只允许 COMBO/SELECTION 的算子，或者用了 level 不允许的算子，工具没有帮它区分。多个并发客户端反复拉取这份几乎不变的静态数据，是不必要的负载。
- 建议：在客户端缓存 /operators（TTL 可以设为小时级）；工具增加 scope（REGULAR/COMBO/SELECTION）、category、name_contains 过滤，以及 brief=True 模式（只返回 name/definition/scope）；docstring 说明 scope/level 的含义。
- 复核修正：代码 915-923 每次都直接请求 /operators，没有缓存也没有过滤。目录 5083-5140 显示返回数组，每项含 scope/category/level 等字段，而且接口没有查询参数（5078），所以过滤只能在客户端做，这一点建议的方向是对的。需要更正两处：一是 documentation 在示例里是 null，schema 是 string|null，很可能只是文档链接，“包含完整 documentation 撑大上下文”的说法有夸大；二是 level 的具体语义目录没有说明（示例是 'ALL'），“用了 level 不允许的算子”无法从目录验证。缓存和 brief/scope 过滤的建议合理。
- 修正后的建议：在客户端缓存 /operators（小时级 TTL，并发请求合并成一次）；工具增加 scope/category/name_contains 过滤和 brief 模式（只返回 name/definition/scope）；level 的语义在没有实测前不要写进 docstring。

#### DATA-3 ·【低】data_type 默认值 "" 会发送空的 type= 参数；只有魔法值 'ALL' 才会省略，docstring 也没说明

- 类别：`wrong-params` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:577, 592-593；工具 2038, 2053`
- 目录依据：listDataFields，目录 4331
- 证据：代码：`data_type: str = ""` 和 `if data_type != 'ALL':
    params['type'] = data_type`。工具 docstring：`data_type: Type of data (e.g., "MATRIX",'VECTOR','GROUP')`。目录 4331：`| type | query | 否 | string | 枚举 MATRIX / VECTOR / GROUP / UNIVERSE / SYMBOL | 字段类型；前端的 all 只表示不发送该参数。 |`
- 影响：默认调用会带上 `type=`（空字符串），这个值不在目录枚举内。目录没有说明服务端怎么处理空值（可能被忽略，也可能过滤成 0 条），行为无法验证。LLM 按直觉传 "all"、"ALL " 或小写 "matrix" 时，也会原样发出非法枚举值。docstring 没提 'ALL' 的特殊语义，也漏掉了 UNIVERSE/SYMBOL 两个合法值。
- 建议：把默认值改为 None，只有值非空且（大写化后）不等于 'ALL' 时才发送 type；在客户端按 {MATRIX, VECTOR, GROUP, UNIVERSE, SYMBOL} 校验，非法值直接报错；docstring 列出完整枚举，并说明“不传即全部”。
- 复核修正：代码 577/592-593 属实：data_type 默认是 ""，而且只有等于 'ALL' 时才不发送，所以默认调用会发出 type=。requests 只会丢弃值为 None 的参数，空字符串会被编码成 type=。目录 4331 的枚举是 MATRIX/VECTOR/GROUP/UNIVERSE/SYMBOL，docstring 确实漏了 UNIVERSE/SYMBOL，也没说明 'ALL' 的语义。不过这是每次默认调用都会走的路径：如果服务端把空 type 过滤成 0 条，早就会暴露出来，更可能的情况是服务端忽略空值。没有证据表明它造成了实际故障，所以 medium 偏高。
- 修正后的建议：默认值改为 None；只有值非空、且 strip().upper() 后不等于 'ALL' 时才发送；发送前先大写化，再按 5 个枚举值校验；docstring 列出完整枚举，并说明不传即全部。

#### DATA-5 ·【低】get_datasets 默认发送 theme=false，这个值在目录中没有记录，无法验证

- 类别：`wrong-params` · 复核：已确认
- 代码位置：`platform_functions.py:550, 560；工具 2007, 2020`
- 目录依据：listDatasets，目录 3359
- 证据：代码：`theme: str = "false"` … `'theme': theme`。目录 3359：`| theme | query | 否 | string | — | 主题筛选；可行值取自数据集或字段返回的 themes。 |`。目录没有说明 "false" 这个取值的语义。
- 影响："false" 看起来沿用了旧版脚本的习惯写法。按目录描述，theme 是主题 ID 过滤器，传字符串 "false" 可能被忽略，也可能被当成一个不存在的主题。目前没有证据表明它会导致 0 结果（不能断言它有问题），但语义不透明。LLM 看到 docstring “Theme filter” 也不知道该传什么值。
- 建议：默认不发送 theme（None），只有调用方显式给出主题 ID 时才发送；docstring 说明取值来自数据集/字段结果中的 themes。如果确实需要 "false" 的旧语义，请用实测结果加注释说明。

#### DATA-6 ·【低】/data-sets、/data-fields、/operators 请求都没带目录记录的版本化 Accept 头

- 类别：`missing-header` · 复核：已确认
- 代码位置：`platform_functions.py:230-231（session 只设置 User-Agent），566, 600, 915`
- 目录依据：listDatasets 3338；listDataFields 4312；listOperators 5072
- 证据：代码：`session.headers.update({'User-Agent': 'Mozilla/5.0 ...'})`，三个调用都没有传 headers。目录 3338：`Accept：application/json;version=2.0`；4312：`Accept：application/json;version=2.0 / application/json;version=3.0`；5072：`Accept：application/json;version=2.0`。
- 影响：requests 默认发 `Accept: */*`，服务端会按默认版本序列化。目录的 schema 是在 version=2.0 下实测得到的；/data-fields 甚至同时存在 2.0 和 3.0 两种表示。不带版本头时，响应形状依赖服务端默认版本，一旦默认版本变化，解析就可能变脆弱。目前代码读取的字段很少，还没有观察到实际故障。
- 建议：给这三个调用显式传 `headers={'Accept': 'application/json;version=2.0'}`（或者在 _request 里按 endpoint 统一设置），与前端保持一致。

#### DATA-9 ·【低】get_operators 做了两层重复的 list→dict 包装，key 名不一致，外层分支是死代码

- 类别：`redundancy` · 复核：已确认
- 代码位置：`platform_functions.py:919-923；2221-2224`
- 目录依据：listOperators，目录 5083-5086（响应 type: array）
- 证据：客户端：`if isinstance(operators_data, list):
    return {"operators": operators_data, "count": len(operators_data)}
else:
    return operators_data`；工具：`if isinstance(operators, list):
    return {"results": operators, "count": len(operators)}
return operators`。目录 5085：`"type": "array"`。
- 影响：客户端已经把 list 转成了 dict，所以工具层的 isinstance(list) 分支永远不会执行（死代码）。这两个分支用的 key 不同（"operators" 和 "results"），维护者容易误以为输出形状会变化。客户端的 else 分支按目录 schema 同样不会出现。不影响正确性，但属于冗余。
- 建议：只在一处规范化：客户端统一返回 `{"results": [...], "count": n}`（与 /data-sets、/data-fields 的形状一致），删掉工具层的重复判断。

#### DATA-10 ·【低】raise_for_status 丢掉了 BRAIN 的错误响应体；504/429 不重试，工具只返回一句笼统错误

- 类别：`error-handling` · 复核：已确认
- 代码位置：`platform_functions.py:566-573, 600-607, 915-926；工具 2027-2028, 2061-2062, 2225-2226`
- 目录依据：listDatasets 3375（错误状态 HTTP 504）；listDataFields 4348（错误状态 HTTP 400），4560-4563（scope 参数依赖）
- 证据：代码：`response.raise_for_status()` … `except Exception as e: self.log(...); raise`，工具：`return {"error": f"An unexpected error occurred: {str(e)}"}`。目录 3375：`错误状态：HTTP 504`；4348：`错误状态：HTTP 400`；4562：`"parameterDependency": "instrumentType、region、delay、universe 必须同时出现…"`。
- 影响：出现 400（例如 region/universe 组合非法）时，HTTPError 的 str 只有“400 Client Error: Bad Request for url …”，服务端返回的具体原因丢失，LLM 无法自我纠正。/data-sets 已记录会出现 504，但 CreddSession 只对 401 重试，一次偶发网关超时就会直接失败。
- 建议：在出错时把 `status_code` 和截断后的 `response.text` 放进返回的 error 结构里（与 _check_once 的做法一致）；对 504/502/429 做一两次带退避的重试（有 Retry-After 时按它等待）。

#### DATA-11 ·【低】关键词发现需要调用两个带 scope 的工具，而目录中的 /data-sets/search 一次就能返回数据集和字段

- 类别：`better-endpoint-available` · 复核：部分成立
- 代码位置：`platform_functions.py:563-566（get_datasets search），595-600（get_datafields search）`
- 目录依据：searchDatasets，目录 3872-3890，4223-4227（required datasets, fields）
- 证据：目前按关键词搜索要分别调用 `get_datasets(search=...)` 和 `get_datafields(search=...)`，后者还受 50 条截断限制。目录 3874：`GET /data-sets/search`，3886：`| search | query | 是 | string | — | 前端实际使用的搜索参数。允许空字符串；完全没有查询字符串时服务端返回 400。 |`；schema required：`"datasets", "fields"`。
- 影响：代理按主题词（例如 "analyst"、"sentiment"）找数据时，需要多次往返，还要逐个 region/universe 重试。/data-sets/search 一次请求就能返回匹配的数据集和字段，更适合作为发现入口。注意：目录只记录了 search 一个参数，是否支持 scope 过滤不明确。
- 建议：新增工具 search_data(query)，调用 GET /data-sets/search?search=...，返回裁剪后的 datasets/fields（id/name/description/region/delay/universe/coverage/alphaCount），作为发现入口；再用 get_datafields 按 dataset_id 精确分页。
- 复核修正：目录 3872-3890 确认 GET /data-sets/search 只有必填的 search 参数，响应必含 datasets 和 fields（4223-4227），代码里也没有用到这个接口。但目录没有说明它按什么 scope 搜索、返回多少条（运行行为是“没有确认到分页”，4294），所以“省掉逐个 region/universe 重试”“更适合作为发现入口”只是推测，无法验证；它返回的 fields 可能也有条数上限。作为可选的补充工具合理，但不能认定它比现有做法更好。
- 修正后的建议：可以把 search_data(query) 作为一个补充的发现工具，但要先实测它的 scope 语义和结果条数上限，再决定是否推荐为主入口；在 docstring 里说明结果需要再用 get_datafields(dataset_id, 带分页) 精确展开。


### SIM · 模拟 / 设置 / super-selection（21 条：高 1 / 中 8 / 低 12）

#### SIM-2 ·【高】lookINTO_SimError_message / check_simulation_progress 会请求调用方给的任意 URL，可被用来 SSRF 并读取本机 credd 的 cookies

- 类别：`security` · 复核：已确认
- 代码位置：`platform_functions.py:2738-2759；1977-1979；_check_once 261-290；FastMCP host=0.0.0.0 1789-1794`
- 目录依据：getSimulation, catalog 6267（路径应为 /simulations/{simulationId}）
- 证据：lookINTO_SimError_message: `for loc in locations: resp = await brain_client._request('get', loc)` 然后 `results.append({"location": loc, "error": error_msg, "raw": data})`，完全不校验 URL。check_simulation_progress 只做子串判断: `if not progress_url or "worldquantbrain.com" not in str(progress_url)`，所以 `http://127.0.0.1:8762/cookies?x=worldquantbrain.com` 也能通过；_check_once 在没有 alpha 字段时返回 `{..., "raw": body}`。服务监听 `host="0.0.0.0"`，本身没有鉴权。CREDD_URL 默认 http://127.0.0.1:8762，且 `headers = {"X-Auth-Token": CREDD_TOKEN} if CREDD_TOKEN else None`，说明 credd 可以不带 token 运行。
- 影响：任何能访问 8761 端口的 MCP 客户端，或被提示注入的 LLM，都可以让服务器请求内网地址并把响应原样拿回来。如果 credd 没有配置 CREDD_TOKEN，调用 GET /cookies 就能拿到 BRAIN 登录 cookie，账号被接管。即使配了 token，也可以用来探测内网。目录里这个接口的路径是固定的 /simulations/{simulationId}，根本不需要接受任意 URL。
- 建议：两个工具都只接受 simulation id，或者严格解析 URL：scheme 必须是 https，host 必须等于 api.worldquantbrain.com，path 必须匹配 ^/simulations/[A-Za-z0-9]+$，然后由服务端自己拼 URL。_check_multi_children 里的 child URL 也要做同样的校验。另外建议服务默认绑定 127.0.0.1，或者加鉴权。

#### SIM-1 ·【中】run_selection 的 super-selection 查询参数名与目录不一致，region/instrumentType/delay 很可能不生效

- 类别：`wrong-params` · 复核：部分成立（原评 高）
- 代码位置：`platform_functions.py:941-950`
- 目录依据：getSimulationSuperSelection, catalog 6526-6534
- 证据：代码: selection_data = {"selection": selection, "instrumentType": instrument_type, "region": region, "delay": delay, "selectionLimit": selection_limit, "selectionHandling": selection_handling}; self._request('get', f"{self.base_url}/simulations/super-selection", params=selection_data)。目录 6528-6530: "| `settings.instrumentType` | query | 否 | string | 枚举 EQUITY |"、"| `settings.region` | query | 否 | string |"、"| `settings.delay` | query | 否 | integer |"
- 影响：服务端要的参数名带 `settings.` 前缀，代码发的是扁平的 instrumentType/region/delay。服务端大概率直接忽略这几个参数，按默认值或不限区域返回 selection 结果，所以查 region=CHN 可能拿回 USA 或混合区域的 alpha，调用方察觉不到。同时这些参数都是非必填，服务端不会报 400 提醒，错误结果会被当成正确结果使用。
- 建议：改为 params={"settings.instrumentType": instrument_type, "settings.region": region, "settings.delay": delay, "selection": selection, "selectionLimit": selection_limit, "selectionHandling": selection_handling, "limit": limit}，并用一次实测对比返回 alpha 的 settings.region 来确认。
- 复核修正：代码 941-950 确实发送扁平的 instrumentType/region/delay；目录 6528-6530 明确列出 settings.instrumentType/settings.region/settings.delay（来自前端源码 index.6168be46.js:7968 且标注实测响应确认）。仓库内没有任何其它地方用扁平参数名成功调用过 super-selection，所以参数名与目录不一致这一点成立。但"服务端忽略参数、返回其他区域 alpha"没有实测证据（也可能返回 400，或服务端兼容扁平名），而且这个工具只用来预览 selection，真正的 SUPER 模拟用的是请求体里的 settings，不受影响，所以严重性降为 medium。
- 修正后的建议：改用 settings.instrumentType / settings.region / settings.delay 作为 query 参数名，同时加上 limit。上线前实测一次：分别用扁平参数名和带前缀的参数名查 region=CHN，对比返回 results[].settings.region，确认哪种写法生效。

#### SIM-3 ·【中】multi 模拟中某个 child 返回 429/5xx 会被当成已结束，整个 multi 被报成 COMPLETE，这个 child 的结果丢失

- 类别：`polling-retry` · 复核：部分成立（原评 高）
- 代码位置：`platform_functions.py:306-307, 321-333, 527-530`
- 目录依据：getSimulation, catalog 6288（错误状态只列了 HTTP 404）、6447-6451
- 证据：child_state: `if r.status_code >= 400: return {"location": url, "status": "ERROR", "http_status": r.status_code}`；`unfinished = [s for s in states if s["status"] in ("RUNNING", "UNKNOWN")]`。check_simulation_progress 只对顶层结果做重试判断: `transient_5xx = (state.get("status") == "ERROR" and (state.get("http_status") or 0) >= 500)`，对 child 不做。目录里 getSimulation 的错误状态只有 404（文档化的终态错误），429/5xx 是暂时性错误。
- 影响：多个 MCP 客户端同时轮询时，很容易有单个 child 请求碰到 429 或 502。这个 child 会被算作已结束，其余 child 一完成，工具就返回 status=COMPLETE，里面这个 child 显示为 ERROR 且没有 alpha_id。agent 看到 COMPLETE 就会停止轮询，把实际成功的 alpha 当成失败丢掉。
- 建议：child 只有返回 404（或 body 里的 status 为 ERROR/FAIL）才算终态失败。429/5xx/网络异常一律当作 RUNNING/UNKNOWN，并读取该响应的 Retry-After 作为下次轮询间隔。
- 复核修正：代码核实：child_state 在 306-307 行把 status>=400 一律标成 ERROR；321 行的 unfinished 只包含 RUNNING/UNKNOWN，所以 child 的 429/5xx 会被当成已结束。check_simulation_progress 的 transient_5xx（527）只看顶层 state，multi 的 COMPLETE 结果里 http_status 在 child 上，不在顶层，因此不会重试。目录 getSimulation 的错误状态只列了 404。问题成立。但要触发它，必须恰好在其余 child 全部完成的那一次轮询中，这个 child 遇到暂时性错误；而且 alpha 仍然在平台上，并没有真正丢失（只是 agent 拿不到 id），所以降为 medium。
- 修正后的建议：child 返回 404，或者 body.status 为 ERROR/FAIL 时，才算终态失败。429、5xx 和网络异常都记为 UNKNOWN（计入 unfinished），并读取该 child 响应里的 Retry-After。

#### SIM-4 ·【中】创建模拟失败 (400/403/409/503) 时丢掉了 BRAIN 返回的错误原因

- 类别：`error-handling` · 复核：已确认
- 代码位置：`platform_functions.py:485, 506-508, 1947-1948；2661-2662`
- 目录依据：createSimulation, catalog 6236-6237
- 证据：单个模拟: `response.raise_for_status()`，然后外层 `return {"error": f"An unexpected error occurred: {str(e)}"}`，得到的只是 '400 Client Error: Bad Request for url'。multi: `if response.status_code != 201: return {"error": f"Failed to create multisimulation. Status: {response.status_code}"}`。目录: "成功状态：`HTTP 201`"，"错误状态：`HTTP 400`、`HTTP 403`、`HTTP 409`、`HTTP 429`、`HTTP 503`"
- 影响：表达式语法错误、universe 与 region 不匹配、decay/truncation 越界、没有 multi 权限 (403)、平台维护 (503) 这些情况，LLM 都只能看到状态码，看不到 BRAIN 返回的字段级错误说明，没法有针对性地修正参数，只能盲目重试，白白消耗模拟额度。另外 503 可能带 Retry-After，但现在没有像 429 那样返回结构化的等待时间。
- 建议：非 2xx 时返回 {"status":"REJECTED","http_status":..., "body": resp.json() 或 resp.text[:2000], "retry_after_seconds": ...}；503 和 409 按与 429 相同的方式处理并解析 Retry-After。两个创建工具共用同一个错误处理函数。

#### SIM-5 ·【中】单个模拟完成后拉取 alpha 不检查状态码，失败会丢掉 alpha_id；完成结果的 status 与 multi 不一致，模拟的 WARNING/message 被丢弃

- 类别：`response-parsing` · 复核：已确认
- 代码位置：`platform_functions.py:286-294, 520-524`
- 目录依据：getSimulation schema, catalog 6388-6414；getAlpha, catalog 8481-8505
- 证据：`alpha = await self._request('get', f"{self.base_url}/alphas/{alpha_id}")`，接着 `result = alpha.json()`，没有 raise_for_status，也没有把 alpha_id 或模拟 body 带入结果。check_simulation_progress 在等待预算用完时 `if waited >= wait_budget: raise`。getSimulation 完成时的 schema 有 "status"、"message"、"detail"、"details"，代码在有 alpha 时全部没用上。
- 影响：(1) 模拟刚完成时 /alphas/{id} 可能短暂返回 404 或空 body（alpha 还在写入），也可能遇到 429。这时 .json() 抛异常，或者把错误 JSON 当成 alpha 详情返回，工具给出 {"error":...}，alpha_id 就丢了。(2) 单个模拟完成时返回的是 alpha 对象，其中 status 是 alpha 的状态 UNSUBMITTED，而 multi 路径返回 "COMPLETE"，LLM 很难写出统一的判断逻辑。(3) 模拟 status=WARNING 时附带的 message（例如单位不匹配告警）被丢掉。
- 建议：返回 {"status":"COMPLETE","sim_status": body.get('status'),"alpha_id":alpha_id,"message": body.get('message') or body.get('detail'),"alpha": ...}；拉取 alpha 时检查状态码，404/429 就降级为只返回 alpha_id 并提示用 get_alpha_details 重试，不要抛异常。

#### SIM-6 ·【中】multi 轮询每次都查一遍全部 child，忽略父任务和 child 的 Retry-After，间隔固定 5 秒

- 类别：`concurrency-perf` · 复核：已确认
- 代码位置：`platform_functions.py:272-274, 298-320, 330`
- 目录依据：getSimulation 运行行为, catalog 6447-6451
- 证据：`children = body.get("children") or []; if children: return await self._check_multi_children(location, children)`，这一步在检查父任务 `"Retry-After" in resp.headers` 之前执行。_check_multi_children 用 `asyncio.gather(*[child_state(u) for u in child_urls])` 并发请求所有 child；RUNNING 时 `"retry_after_seconds": 5.0` 写死，child 响应里的 Retry-After 值没有使用。目录: `{"polling": true, "retryAfterHeader": true}`
- 影响：10 个 child 的 multi 每轮询一次就是 1+10 个请求，全部完成后还要再发 10 个 GET /alphas。多个客户端同时轮询时请求量成倍增加，更容易触发 429（进而触发 SIM-3 的误判）。服务端建议的等待间隔完全没用上。
- 建议：先按父任务的 Retry-After 轮询父任务（只发 1 个请求），父任务不再带 Retry-After 后再一次性读取 children 状态。确实要查 child 时，下次间隔取各 child Retry-After 的最大值，不要写死 5 秒。

#### SIM-7 ·【中】check_simulation_progress 完成时返回完整 alpha 对象（multi 最多 10 个），占用大量 LLM 上下文

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`platform_functions.py:291-294, 336-353`
- 目录依据：getAlpha, catalog 8481-；getSimulationSuperSelection 中 alpha 对象含 is/test/os/prod/train 等大块字段, catalog 6700-7290
- 证据：`d = await self._request('get', f"{self.base_url}/alphas/{s['alpha_id']}"); return {**s, "details": d.json()}`，返回 `"alpha_results": full`。docstring: "once ALL children finish, alpha_results with each child's full alpha details"
- 影响：完整的 alpha 对象包含 is.checks、各阶段统计、classifications、competitions 等字段，10 个加起来可能有几十 KB。agent 做批量表达式搜索时，每轮都会因此挤占上下文，增加费用，也更容易被截断。另外还要额外发 10 个 /alphas 请求。
- 建议：默认只返回摘要（alpha_id、sharpe、fitness、turnover、returns、drawdown、未通过的 checks），加一个 detail="summary"|"full" 参数；需要完整信息时再调用 get_alpha_details。

#### SIM-8 ·【中】run_selection 不传 limit 也不翻页，只能拿到前 10 条；selection_limit 的说明写错；工具描述与接口实际语义不符

- 类别：`pagination` · 复核：已确认
- 代码位置：`platform_functions.py:928-952；2229-2250`
- 目录依据：getSimulationSuperSelection, catalog 6532-6533, 6549-6583；catalog 16853
- 证据：代码没有 limit/offset 参数，直接 `return response.json()`。docstring: "🎯 Run a selection query to filter instruments."，"selection_limit: Maximum number of results"。目录: "| `limit` | query | 否 | integer | 默认 10 |"，"| `selectionLimit` | query | 否 | integer | 范围 10..1000 |"；响应含 count/next/previous/results；results 里的元素是 alpha（type 枚举 REGULAR/SUPER/RA_PARENT/RA_CHILD，带 is/os 等字段）；16853: "动态基路径也覆盖 ... simulations/super-selection"（limit_offset 分页，defaultLimit 10）
- 影响：LLM 会以为 selection_limit=1000 就能拿到 1000 条，实际每次只返回 10 个 alpha，count 可能是几百。LLM 也会以为这个工具是筛选股票 instrument，实际它返回的是被 SuperAlpha selection 表达式选中的 alpha，这会误导 SuperAlpha 的构建。
- 建议：docstring 改为“预览 SuperAlpha selection 表达式会选中哪些 alpha”；增加 limit/offset 参数（或者只返回 count 加前 N 个 alpha 的精简摘要）；说明 selectionLimit 是 SuperAlpha 最多包含的 alpha 数（10..1000），不是分页大小；如果传了 selection_limit，先在客户端校验范围。

#### SIM-11 ·【中】lookINTO_SimError_message 与 check_simulation_progress 功能重复，而且会把还在运行的模拟报成失败

- 类别：`redundancy` · 复核：已确认
- 代码位置：`platform_functions.py:2728-2766`
- 目录依据：getSimulation, catalog 6295-6309（运行中 body 只有 progress）、6447-6451
- 证据：`data = resp.json() if resp.text else {}`；`error_msg = data.get("error") or data.get("message")`；`if not data.get("alpha"): error_msg = error_msg or "Simulation did not get through, ..."`；`for loc in locations:` 逐个串行 await。非 200 时返回 `"raw": resp.text`，不做截断。
- 影响：运行中的模拟返回 {"progress":0.3}（目录 6295-6309），这个工具会把它报成 "Simulation did not get through"，LLM 可能因此放弃一个正常的模拟或重复提交。对 multi 的父 URL 不会展开 children。多个 URL 串行请求，速度慢，原始 body 不截断还会占满上下文。check_simulation_progress 已经会返回失败模拟的 message 和每个 child 的状态，这两个工具的职责重叠。
- 建议：删除这个工具，把它的用途并入 check_simulation_progress（例如接受 id 列表，并发查询，返回精简的错误信息）；如果要保留，先判断 Retry-After/progress，再判断是否失败，同时读取 detail/details，并截断 raw。

#### SIM-9 ·【低】create_multi_simulation 另写了一套 settings 拼装逻辑，与 create_simulation/SimulationSettings 重复且行为不一致

- 类别：`redundancy` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:2606-2662 与 422-508, 1916-1946`
- 目录依据：createSimulation 数组分支, catalog 5963-5966, 6032-6081
- 证据：multi 内联拼的 dict 只有 `'maxTrade': max_trade`，没有 maxPosition/selectionHandling/selectionLimit/componentActivation，`'type': 'REGULAR'` 写死；成功判断用 `if response.status_code != 201`，单个模拟用 `raise_for_status()`；客户端校验 `if len(alpha_expressions) < 2`。目录中数组分支是 `"minItems": 1`，每一项都可以是 REGULAR 或 SUPER，也都支持 maxPosition 等字段。
- 影响：两个工具对同样的参数产生不同的请求体（multi 永远无法设置 maxPosition），错误处理也不同。以后修复 SIM-4、SIM-15 这类问题都要改两处，很容易漏掉。1 条表达式调用 multi 会被拒绝，但目录并没有这个限制。
- 建议：抽一个 build_simulation_payload(SimulationData) 函数，两个工具都用它。multi 改为用 SimulationData 构建每个元素，最后 POST 一个 list；共用同一套 429/非 201 处理；最小条数改为 1，或者在 docstring 里说明为什么要求至少 2 条。
- 复核修正：重复拼装逻辑已核实：2620-2638 行另拼了一份 settings，没有 maxPosition，type 写死为 REGULAR，非 201 的处理也和单个模拟不同。但目录里数组分支的 minItems:1 来自逆向得到的 schema，不是官方约束；社区普遍认为 BRAIN 的 multi-simulation 要求 2-10 条，2608 行的最小 2 条限制很可能是有意为之。建议里"最小条数改为 1"没有依据，可能导致请求被拒。这是维护层面的冗余，对功能影响有限，降为 low。
- 修正后的建议：抽取共用的 payload 构建函数和错误处理函数（包括 429 与非 2xx 的响应体），multi 补上 max_position 参数。最小 2 条的限制保留，除非实测确认 1 条也能提交，并在 docstring 中说明原因。

#### SIM-10 ·【低】模拟参数默认值写死为 NONE/0/0，与平台及用户自己的默认设置不同；目录里有读取用户默认设置的接口

- 类别：`better-endpoint-available` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:151-169, 1855-1878, 2553-2569`
- 目录依据：getSimulationSettings, catalog 5177-5317；createSimulation 请求示例, catalog 6213-6231
- 证据：代码默认 `decay: float = 0.0`、`neutralization: str = "NONE"`、`truncation: float = 0.0`、`test_period: str = "P0Y0M"`、`visualization: bool = True`。目录中 getSimulationSettings 的示例: "decay": 4, "neutralization": "SUBINDUSTRY", "truncation": 0.08, "testPeriod": "P0Y0M0D", "visualization": false；createSimulation 请求示例也用了同一组值。
- 影响：LLM 没显式传参时，模拟跑的是无中性化、无截断、无衰减的配置，和用户在网页上的实验结果无法对比，sharpe/turnover 往往明显变差，导致误判 alpha 好坏。visualization=True 还可能增加计算开销（目录未说明）。
- 建议：参数默认值改为 None；提交时用 GET /users/self/settings/simulation 的结果补齐未传的字段（可缓存）。或者至少把默认值改成与目录示例一致，并在 docstring 中写明。
- 复核修正：默认值 decay=0、neutralization=NONE、truncation=0、visualization=True 已核实（151-169、1861-1868、2559-2566）。目录 5301-5314 的 decay 4/SUBINDUSTRY/0.08 是去敏后的示例值，不一定是平台或用户默认值；而且 LLM 通常会显式传参，影响有限。visualization 会增加开销在目录里没有依据。另外目录路径是 /users/{userId}/settings/simulation，5192 行说明 userId 是当前用户的 WQ ID，建议里直接写 /users/self/... 未经目录证实。
- 修正后的建议：在 docstring 里写明当前默认值（NONE/0/0）和网页常用配置不同，建议显式传入。如果要读取用户默认设置，先通过 GET /users/self 拿到真实 userId，再调用 GET /users/{userId}/settings/simulation（或先实测 self 是否可用），结果缓存后用来补齐调用方没有传的字段。

#### SIM-12 ·【低】失败原因只读取 message，没有读 getSimulation schema 里的 detail/details 字段

- 类别：`response-parsing` · 复核：已确认
- 代码位置：`platform_functions.py:287-290, 314-317, 2751`
- 目录依据：getSimulation schema, catalog 6406-6414
- 证据：_check_once: `"message": body.get("message")`；_check_multi_children: `"message": b.get("message")`。目录 schema 中同时列出 "details": {"type": "string"}、"detail": {"type": "string"}、"message": {"type": "string"}。
- 影响：如果失败原因放在 detail 或 details 里，multi 的 child 结果只有 status=ERROR、message=None，看不到原因。单个模拟因为还带了 raw 能看到，但 child 路径完全看不到。
- 建议：用 `msg = body.get('message') or body.get('detail') or body.get('details')`；child 返回 4xx 时也附上截断后的 body。

#### SIM-13 ·【低】get_platform_setting_options 按 label 字符串硬匹配、没有带 version=3.0 的 Accept，且丢掉了大部分选项

- 类别：`response-parsing` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:1628-1684`
- 目录依据：getSimulationOptions, catalog 5601-5605, 5625-5684；testPeriod 范围说明 5830
- 证据：`response = await self._request('options', f"{self.base_url}/simulations")`，没有设置 Accept；`settings_options = settings_data['actions']['POST']['settings']['children']`；`if setting['label'] == 'Instrument type': ... elif setting['label'] == 'Region': region_data = setting['choices']['instrumentType']`。目录: "Accept：`application/json;version=3.0`"；目录的去敏示例里 settings 只有 {"type": "nested object", "required": true, "readOnly": false}，没有列出 children（目录对 children 的结构没有说明，只提到 "OPTIONS 给出的范围为 P0Y0M0D 到 P6Y0M0D"）。
- 影响：(1) 按 label 显示文本匹配，前端文案或本地化一变，就会静默返回空字典，或者抛 KeyError 变成泛泛的错误。(2) 没有按目录声明 version=3.0，默认版本的返回结构可能不同（目录无法证实默认版本的结构）。(3) docstring 说用来“校验和修正模拟设置”，但 decay/truncation/lookback 的范围、testPeriod 范围、language、pasteurization/nanHandling/maxTrade/maxPosition、selectionHandling/componentActivation 的枚举都被丢掉，LLM 仍然会传出越界的参数。(4) 每次调用都重新请求，没有缓存，而这个结果几乎不变。
- 建议：带上 headers={'Accept':'application/json;version=3.0'}；按 key（instrumentType/region/universe/delay/neutralization）而不是 label 匹配；对 children 里的其他字段一并输出 choices/min/max；缓存几小时；缺少某个 key 时明确报出 schema 变化。
- 复核修正：按 label 硬匹配（1645-1654）、只输出 instrumentType/region/delay/universe/neutralization、不做缓存，这几点都已核实。但"缺少 version=3.0 的 Accept 会导致结构不对"没有证据：现有解析方式（settings.children[].choices.instrumentType[...].region[...]）和社区广泛使用的 ace_lib 在不带 Accept 头时完全一样，而且能正常工作；目录 5625-5684 的示例并没有给出 children 的结构。直接加上 version=3.0 反而可能改变返回结构，把现有解析弄坏。
- 修正后的建议：改为按 key（instrumentType/region/universe/delay/neutralization）匹配；键缺失时明确报告 schema 变化；把 decay/truncation/testPeriod 等字段的范围和枚举一并输出；结果缓存数小时。Accept version=3.0 先对比实测两个版本的返回结构，再决定是否切换，不要直接加。

#### SIM-14 ·【低】本组所有 BRAIN 请求都没有带目录声明的 Accept 版本头

- 类别：`missing-header` · 复核：部分成立
- 代码位置：`platform_functions.py:230-232（session 只设置了 User-Agent）；471, 950, 1628, 2649, 261, 303, 291, 340`
- 目录依据：getSimulationSettings 5184；getSimulationOptions 5605；createSimulation 5704；getSimulation 6271；getSimulationSuperSelection 6520；getAlpha 8488
- 证据：`session.headers.update({'User-Agent': ...})`，整个文件 grep 不到 Accept。目录里各接口分别写明 "Accept：`application/json;version=2.0`"，OPTIONS /simulations 写明 "Accept：`application/json;version=3.0`"。
- 影响：现在依赖服务端的默认版本；一旦 BRAIN 切换默认版本，返回结构（尤其是 OPTIONS 和 alpha 对象）可能变化，而代码是按字段硬读的，会静默出错。目前没有证据表明已经出问题，所以定为 low。
- 建议：session 默认带 Accept: application/json;version=2.0，OPTIONS /simulations 单独改为 version=3.0。
- 复核修正：230-232 行只设置了 User-Agent，文件里确实没有 Accept 头；目录各接口都标注了 version=2.0，OPTIONS 标注 3.0。但目前代码按服务端默认版本的结构解析，并且可以工作（参见 SIM-13）。在 session 上全局加 Accept 头，会同时改变本文件所有其它接口的返回版本，存在大面积回归风险，这条建议不能直接执行。
- 修正后的建议：先逐个接口实测：带与不带目录声明的 Accept 头时，返回结构是否一致。确认一致后再按接口显式传 headers，不要在 session 上全局设置；OPTIONS /simulations 要和 SIM-13 的解析一起验证。

#### SIM-15 ·【低】decay 类型为 float、默认 testPeriod 格式与目录不一致，且没有做范围/枚举校验，出错时报错信息又被丢掉（见 SIM-4）

- 类别：`wrong-params` · 复核：已确认
- 代码位置：`platform_functions.py:156, 164, 166, 1861-1864, 2559`
- 目录依据：createSimulation, catalog 5751-5755, 5759-5767, 5798-5803, 5828-5831
- 证据：`decay: float = 0.0`、`testPeriod: Optional[str] = "P0Y0M"`、`selectionLimit: int = 1000`，都没有校验。目录: decay `"type": "integer", "minimum": 0, "maximum": 512`；truncation `minimum 0, maximum 1`；lookback `0..1024`；selectionLimit `10..1000`；testPeriod "OPTIONS 给出的范围为 P0Y0M0D 到 P6Y0M0D"。
- 影响：LLM 传入 decay=2.5 或 truncation=5 时会发出不合法的请求体，服务端返回 400，但因为 SIM-4 看不到原因。'P0Y0M' 目前大概率能被接受（目录未说明），但它不在目录给出的格式范围内。
- 建议：decay 改为 int，并在 pydantic 上加 Field(ge=0, le=512) 等约束（truncation、lookback、selectionLimit 同理）；testPeriod 默认值改为 'P0Y0M0D'；枚举字段用 Literal 类型，让 MCP schema 直接把合法值告诉 LLM。

#### SIM-17 ·【低】只要有 Retry-After 头就判定为运行中，没有像 _retry_after_seconds 的注释那样判断值是否为 0

- 类别：`polling-retry` · 复核：部分成立
- 代码位置：`platform_functions.py:276-283, 312-313`
- 目录依据：getSimulation 运行行为, catalog 6447-6451
- 证据：`if "Retry-After" in resp.headers: return {"status": "RUNNING", ..., "retry_after_seconds": _retry_after_seconds(resp) or 5.0}`。目录只写了 `"retryAfterHeader": true`，没有说明完成时是去掉这个头还是返回 0。
- 影响：如果 BRAIN 在完成时仍然带 Retry-After: 0（很多社区客户端按 `== 0` 判断完成），这个工具会一直返回 RUNNING，永远拿不到结果。
- 建议：改为 `ra = _retry_after_seconds(resp); if ra > 0 or (not body.get('alpha') and 'progress' in body and 'status' not in body): RUNNING`，否则按完成处理。
- 复核修正：276 行和 312 行只判断 Retry-After 头是否存在，确实没有判断值是否为 0；54-60 行 _retry_after_seconds 的注释也提示过这类头值的陷阱。但目录 6447-6451 只写了 retryAfterHeader:true，没有证据表明 BRAIN 完成时会返回 Retry-After: 0（常见实现以头缺失作为完成信号），所以只是防御性改进。原建议里用 'status' not in body 这类启发式判断过于复杂。
- 修正后的建议：把运行中的判断改为 `_retry_after_seconds(resp) > 0`（头缺失或值为 0 都视为已完成），child 路径同样修改。

#### SIM-18 ·【低】轮询 GET /simulations 时遇到 429 会立即当作终态 ERROR 返回，只有 5xx 会重试

- 类别：`polling-retry` · 复核：已确认
- 代码位置：`platform_functions.py:262-264, 527-530`
- 目录依据：getSimulation, catalog 6288（目录只列了 404；对 GET 的 429 没有说明）
- 证据：`if resp.status_code >= 400: return {"status": "ERROR", "http_status": resp.status_code, ...}`；`transient_5xx = (state.get("status") == "ERROR" and (state.get("http_status") or 0) >= 500)`
- 影响：多个客户端同时轮询时，一次 429 会让 check_simulation_progress 直接返回 status=ERROR，LLM 可能以为模拟失败而重新提交，占用本来就有限的并发槽。
- 建议：429 和 5xx 同样当作暂时性错误，按 Retry-After 等待后重试；返回的 status 用 THROTTLED，与真正的 ERROR 区分开。

#### SIM-19 ·【低】manage_config(get) 会把配置文件原样返回，可能包含明文 credentials；set 可以写入任意键

- 类别：`security` · 复核：部分成立（原评 中）
- 代码位置：`platform_functions.py:1830-1847；2323-2326`
- 目录依据：n/a（本地配置，目录未涉及）
- 证据：`config = load_config(); ... return {"config": config, "auth_status": auth_status, ...}`；论坛工具读取 `credentials = config.get("credentials", {}); password = password or credentials.get("password", "")`，说明配置文件的约定格式里就有明文密码。服务监听 0.0.0.0 且没有鉴权。
- 影响：只要配置文件里还留着旧的 credentials（现在登录已由 credd 负责，这些字段是历史遗留），任何 MCP 客户端调用 manage_config(get) 就能读到 BRAIN 的明文密码，而且密码会进入 LLM 上下文和日志。set 也可以被用来写入任意内容。
- 建议：get 返回前去掉或打码 credentials/password/token 类字段；set 只允许白名单里的键；既然 credd 已经负责登录，就删除配置文件里的 credentials 约定，论坛工具也不要再读取它。
- 复核修正：1830-1838 行 get 会原样返回配置；2323-2326 等处的论坛工具读取 credentials.password，说明历史上的配置格式里有明文密码。但仓库里没有 user_config.json，而且 authenticate 已经忽略 email/password（1814、356-358），所以只有当部署环境里残留了旧配置文件时才会泄露。set 可以写任意键，但只影响这个本地 JSON 文件，而文件的唯一消费者是已经失效的 credentials 透传，实际影响很小。
- 修正后的建议：get 返回前过滤掉 credentials/password/token 类字段；同时删除论坛工具读取 config.credentials 的遗留代码，彻底去掉明文凭据的约定；set 改为白名单键，或者直接删掉。

#### SIM-20 ·【低】配置文件相关代码基本没用了：路径与注释不符，临时文件兜底会丢配置，写入失败被吞掉；manage_config(get) 对同一登录状态发了 2 个请求

- 类别：`dead-code` · 复核：已确认
- 代码位置：`platform_functions.py:1739-1785, 1832-1838`
- 目录依据：getUser catalog 512-；getAuthentication catalog 119-
- 证据：docstring 写的是 "falls back to ~/.brain_mcp_config.json"，实际是 `config_path = Path(__file__).parent / "user_config.json"`；兜底 `return tempfile.NamedTemporaryFile(delete=False).name`，每次都是新的随机文件；save_config 遇到 `except IOError as e: logger.error(...)` 后，manage_config 仍然 `return config`，看起来像保存成功。get 分支同时调用 `get_authentication_status()` (GET /users/self) 和 `is_authenticated()` (GET /authentication)。
- 影响：写入失败或走了临时文件兜底时，配置会静默丢失，而工具返回的是“已更新”的配置。配置文件现在唯一的读取方是已废弃的 email/password 透传，没有任何模拟设置从这里读取，所以 manage_config 对模拟流程没有实际作用，却可能让 LLM 以为能通过它改模拟默认值。
- 建议：如果需要持久化模拟默认值，改用目录里的 getSimulationSettings/updateSimulationSettings，存到服务端；否则删除 manage_config 和配置文件相关代码，只保留 authenticate。get 分支只请求一次 /authentication 即可。

#### SIM-M1 ·【低】缺少取消模拟的工具：遇到 RATE_LIMITED 时只能被动等待，无法释放并发槽

- 类别：`better-endpoint-available` · 复核：复核补充
- 代码位置：`platform_functions.py:472-484, 2651-2660（没有任何对 /simulations/{id} 的 DELETE 调用）`
- 目录依据：deleteSimulation, catalog 6462-6500
- 证据：429 分支返回的 note 是 "Retry create_simulation after ~Ns, or first finish/check the running ones"，整个文件 grep 不到对 simulations 的 'delete' 请求。目录 6465 行有 "`DELETE /simulations/{simulationId}`"，成功状态 HTTP 200，错误状态 401/404，敏感等级标为"高风险操作"。
- 影响：agent 提交了错误或不再需要的模拟（例如参数写错的 multi）后，只能等它跑完才能释放账户的并发槽，期间所有新的 create_simulation 都返回 RATE_LIMITED。多个客户端共用一个账户时，这个问题会更严重。
- 建议：新增 cancel_simulation(simulation_id) 工具，只接受 id（格式校验后由服务端拼出 /simulations/{id} 的 URL），调用 DELETE。由于目录把它标为高风险操作，docstring 里要写明只用于取消自己刚提交的模拟。

#### SIM-M2 ·【低】multi 完成后拉取 alpha 详情时不检查状态码，错误响应会被当成 details 返回

- 类别：`response-parsing` · 复核：复核补充
- 代码位置：`platform_functions.py:339-343`
- 目录依据：getAlpha, catalog 8481-
- 证据：`d = await self._request('get', f"{self.base_url}/alphas/{s['alpha_id']}"); return {**s, "details": d.json()}`，没有 raise_for_status，也没有判断 status_code。
- 影响：/alphas/{id} 返回 404/429 并带 JSON 错误体（例如 {"detail":"Not found."}）时，这个错误体会原样放进 details，child 仍然标记为 COMPLETE，看起来像拿到了 alpha 详情，LLM 会读到一个没有任何指标的"alpha"。如果错误体不是 JSON，就会走 except 分支，这种情况倒是处理了。
- 建议：status_code != 200 时不填 details，改为返回 {"details_error": {"http_status": ..., "body": text[:300]}}，并提示调用方用 get_alpha_details(alpha_id) 重试。


### ALPHA · Alpha 列表 / 详情 / 修改 / 提交 / 检查（15 条：高 2 / 中 8 / 低 5）

#### ALPHA-1 ·【高】get_submission_check 没有使用平台的 /alphas/{id}/check，all_passed 只看相关性 < 0.7，结论会误导

- 类别：`better-endpoint-available` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1421-1442（包装层 2405-2410），阈值在 1334 与 1402`
- 目录依据：getAlphaChecks GET /alphas/{alphaId}/check (catalog 12552-12714)；schema 中 is.checks[].result enum PASS/FAIL/PENDING (12592-12600)，limit/value (12602-12613)
- 证据：代码：`correlation_checks = await self.check_correlation(alpha_id, correlation_type="both")` ... `'all_passed': correlation_checks['all_passed']`；check_correlation 中 `threshold: float = 0.7` 与 `passes_check = max_correlation < threshold`。目录 12552-12556：`GET /alphas/{alphaId}/check`，operationId `getAlphaChecks`；返回 `is.checks` 数组，每项含 `name`、`result`（enum `PASS`/`FAIL`/`PENDING`）、`limit`、`value`，另有 `is.selfCorrelation`（records/schema/min/max）；运行行为 12700-12706：`"polling": true, "retryAfterHeader": true`。
- 影响：工具名为“Comprehensive pre-submission check”，但 all_passed 只取决于 prod/self 两个相关性的 max 是否 < 0.7。LOW_SHARPE、LOW_FITNESS、换手率、集中度、子宇宙等平台检查全部没有参与判定，平台下发的 limit 也被写死的 0.7 取代。一个 Sharpe 不达标、相关性低的 alpha 会得到 all_passed=True，agent 接着调用 submit_alpha，结果被平台拒绝或浪费提交配额。反过来，平台对相关性的真实判定规则以 checks 的 result/limit 为准，客户端的 0.7 可能误判为 FAIL。
- 建议：改为调用 GET /alphas/{id}/check，并按 Retry-After 轮询：响应带 Retry-After 时等待后重试（可复用 check_simulation_progress 的 wait_budget 模式），不带 Retry-After 时再解析 JSON。all_passed 定义为“is.checks 中没有 FAIL，也没有 PENDING”，同时返回 FAIL/PENDING 项的 name/limit/value 精简列表。相关性细节直接取 is.selfCorrelation.max，不再在客户端重新计算。

#### ALPHA-2 ·【高】submit_alpha 只发一次 POST 就报告成功，没有按 Retry-After 轮询 GET /submit，也不检查最终 checks

- 类别：`polling-retry` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:714-729；包装层 2142-2158`
- 目录依据：submitAlpha POST /alphas/{alphaId}/submit (catalog 12211, 响应 12231-12235, 运行行为 12361-12371)；pollAlphaSubmission GET /alphas/{alphaId}/submit (12382, 运行行为 12532-12541)
- 证据：代码：`response = await self._request('post', f"{self.base_url}/alphas/{alpha_id}/submit")` `response.raise_for_status()` `self.log(f"Alpha {alpha_id} submitted successfully", "SUCCESS")`。目录 12233-12235：`成功状态：HTTP 200、HTTP 201、HTTP 202、HTTP 204`，`响应模式：json_or_empty`；12363-12370：`"poll": "pollAlphaSubmission", "retryAfterHeader": true, ... "completion": "收到不含 Retry-After 的 2xx 响应后解析 JSON"`；最终响应 schema 为 `is.checks[]`，result enum 含 `FAIL`。
- 影响：POST 返回 201/202 且带 Retry-After 只表示“已受理、正在检查”，代码却马上当作提交成功。平台随后可能在 GET /submit 的最终响应中用 is.checks FAIL 拒绝提交（例如相关性或 Sharpe 不达标），agent 仍会认为已提交，后续研究和统计都会出错。这是有不可逆副作用的操作，工具还缺少“先 /check 再提交”的保护。
- 建议：实现完整流程：POST 之后只要响应带 Retry-After，就按其秒数 GET /alphas/{id}/submit 轮询，并设置总时长上限和“进行中”返回（附 retry_after_seconds，参考 check_simulation_progress）；在最终不带 Retry-After 的响应上解析 is.checks，存在 FAIL 时返回 success=false 和失败项。另外建议在提交前强制或可选调用 GET /alphas/{id}/check，并在 docstring 写明这是不可逆操作。

#### ALPHA-3 ·【中】submit_alpha 返回 requests.Response.__dict__，签名写 bool，MCP 输出不可用，还可能把响应头或 cookie 暴露给 LLM

- 类别：`response-parsing` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:714, 725；包装层 2155-2156`
- 目录依据：submitAlpha 响应 schema (catalog 12236-12341, 示例 12343-12359)
- 证据：代码：`async def submit_alpha(self, alpha_id: str) -> bool:` ... `return response.__dict__`；包装层：`success = await brain_client.submit_alpha(alpha_id)` `return {"success": success}`。目录规定响应体是 JSON `{"is": {"checks": [...]}}`，或者为空（json_or_empty）。
- 影响：Response.__dict__ 包含 `_content`(bytes)、`raw`(urllib3 响应)、`request`(PreparedRequest)、`connection`(HTTPAdapter)、`cookies`(RequestsCookieJar)、`headers`、`elapsed` 等非 JSON 对象。FastMCP 序列化时要么报错（结构化输出校验），要么把这些对象 str 化后塞进上下文，真正有用的 is.checks 却没有解析。如果响应带 Set-Cookie，jar/headers 的字符串形式会把会话 cookie 值写进 LLM 对话记录。`{"success": {...}}` 只要是非空 dict 就为真，agent 读不出实际结果。
- 建议：返回结构化结果：`{"success": bool, "status_code": int, "checks": is.checks 精简列表, "failed": [...]}`；响应体为空或进行中时按 ALPHA-2 的方式处理。签名改为 Dict[str, Any]，绝不返回 Response 对象的内部字段。

#### ALPHA-4 ·【中】submit_alpha 吞掉所有异常，只返回 False；429/401/500 的原因和 Retry-After 全部丢失

- 类别：`error-handling` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:727-729；包装层 2157-2158`
- 目录依据：submitAlpha 错误状态 (catalog 12234)
- 证据：代码：`except Exception as e:` `self.log(f"❌ Failed to submit alpha: {str(e)}", "ERROR")` `return False`；包装层 `except Exception as e: return {"error": ...}` 因此永远不会触发。目录 12234：`错误状态：HTTP 401、HTTP 429、HTTP 500`。
- 影响：agent 只拿到 `{"success": false}`，无法区分限流（应按 Retry-After 稍后重试）、会话失效（credd 问题）和服务端错误，也看不到平台返回的拒绝原因。这可能导致盲目重复提交，也可能直接放弃本来合格的 alpha。包装层的 except 分支成了死代码。
- 建议：不要吞异常：对 4xx/5xx 返回 status_code、响应体前若干字符和解析后的 Retry-After；对 429 明确给出 retry_after_seconds。删除或恢复包装层的错误路径，保证两层只有一处负责错误。
- 复核修正：客户端 submit_alpha 在 727-729 行 `except Exception` 后只 `return False`，所以 429/500 的状态码、响应体和 Retry-After 都丢了，这一点成立。说“包装层 except 是死代码”不准确：`await self.ensure_authenticated()`（第 716 行）在 try 之外，credd 不可用（CreddUnavailable）时异常会传到包装层 2157 行，由它返回 error。另外 401 已由 CreddSession 自愈重试一次，只有重试后仍是 401 才会被吞掉。
- 修正后的建议：客户端不要吞 HTTP 异常，对 4xx/5xx 返回 status_code、响应体摘要和 retry_after_seconds（429 时）。包装层的 except 需要保留，用来处理 credd 不可用等会话初始化异常，不要删除。

#### ALPHA-5 ·【中】get_submission_check 串行拉取 prod/self 相关性，每次固定 sleep 20s、最多 5 次，完全忽略 Retry-After

- 类别：`polling-retry` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1218-1266（prod），1278-1326（self），1353-1361（串行循环），1427`
- 目录依据：getAlphaCorrelation 运行行为 (catalog 12828-12842)
- 证据：代码：`max_retries = 5` `retry_delay = 20  # seconds`，空响应和异常一律 `await asyncio.sleep(retry_delay)`；check_correlation 中 `for check_type in check_types:` 依次 `await self.get_production_correlation(...)`、`await self.get_self_correlation(...)`。目录 12830-12841：`"polling": true, "retryAfterHeader": true, "handledStatuses": [401, 410, 412, 429, 503]`。
- 影响：相关性计算进行中时，平台返回带 Retry-After 的空体。代码不读 Retry-After，只按固定 20s 等待；两个相关性串行，最坏约 2×(4×20s + 请求耗时) ≈ 3 分钟以上，超过多数 MCP 客户端的调用超时，阻塞 agent。429 被 raise_for_status 当成普通异常，同样按 20s 重试，不遵守服务端节流；410/412 这类语义错误也会被无意义地重试 5 次。多个客户端并发调用时会放大请求量。
- 建议：改用 ALPHA-1 的 /check（一次轮询同时拿到 checks 和 selfCorrelation）；如果仍需相关性端点，用 asyncio.gather 并发 prod/self，按 _retry_after_seconds 等待并设置总预算，410/412 直接返回明确错误，429 按 Retry-After 退避，超出预算时返回 RUNNING 和 retry_after_seconds，交给调用方稍后再查。

#### ALPHA-6 ·【中】check_correlation 解析 max：值为 null 时 float(None) 崩溃；records 回退逻辑会把非相关性列当作相关性

- 类别：`response-parsing` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1365-1399`
- 目录依据：getAlphaCorrelation 响应 schema max/min (catalog 12776-12787)，示例 12795-12824
- 证据：代码：`elif 'max' in correlation_data:` `max_correlation = float(correlation_data['max'])`；回退分支：`for v in row: ... vf = float(v); if -1.0 <= vf <= 1.0: candidate_max = ...`。目录 12782-12787：`"max": {"type": ["number", "null"]}`；records 各列的含义由 `schema.properties` 定义（示例列 `id`、`correlation`）。
- 影响：没有可比 alpha 等场景下 max 为 null，`float(None)` 抛出 TypeError，整个 get_submission_check 失败（只返回泛化的 error）。回退分支会对一行中所有落在 [-1,1] 的数值取最大值，而 records 行里可能还有 returns、turnover、fitness 等数值列，得到的“最大相关性”是错的，导致 pass/fail 判断错误。代码还优先读取 `schema.max`，但目录的 schema 对象里并没有这个字段。
- 建议：只读取顶层 `max`，null 视为“无可比 alpha / 相关性为 0”并单独标注；如需从 records 推导，先根据 `schema.properties` 找到 name == 'correlation' 的列下标，只读这一列。删除按数值范围猜测的逻辑。
- 复核修正：null 崩溃成立：目录 12776-12787 中顶层 max 类型为 number|null；代码 1368-1372 行先查 schema.max（目录 schema 没有这个字段，但 additionalProperties=true，只是多余，不会出错），然后 `float(correlation_data['max'])`，max 为 null 时抛 TypeError，整个 get_submission_check 失败。records 回退分支的风险被夸大了：只有响应缺少 max 键时才会走到这里，而目录示例和实测都返回了 max。目录示例的 records 只有 [id, correlation] 两列，“可能包含 returns/turnover/fitness 等列”在目录中没有证据。按 [-1,1] 范围猜测列的写法确实脆弱，但属于次要问题。
- 修正后的建议：主修复：max 为 null 时，不要 float(None)，标注为“无可比 alpha/无相关性数据”，passes_check 按 True 处理，并在返回中注明。records 回退属于次要修复：根据 schema.properties 中 name=='correlation' 的列下标读取，删除按数值范围猜列的逻辑。

#### ALPHA-7 ·【中】check_correlation 的 correlation_type 与目录枚举不一致，未知值被静默跳过，all_passed 反而为 True

- 类别：`wrong-params` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1348-1361, 1415；工具 2397`
- 目录依据：getAlphaCorrelation 参数 correlationType (catalog 12731)
- 证据：代码：`if check_type == "production": ... elif check_type == "self": ... else: continue`，之后 `results['all_passed'] = all_passed`（初始为 True）。目录 12731：`correlationType | path | 是 | string | 枚举 self / power-pool / prod | self=自相关，power-pool=Power Pool 相关性，prod=生产相关性；三种取值均已实测。`
- 影响：agent 按平台术语传入 correlation_type="prod" 或 "power-pool" 时，循环直接 continue，checks 为空，all_passed=True。这是静默的假阳性，可能让 agent 误以为相关性检查通过而去提交。power-pool 相关性完全不支持。
- 建议：接受目录枚举 self/prod/power-pool（兼容 production → prod），未知值直接报错；all_passed 至少覆盖一项检查才可为 True。

#### ALPHA-8 ·【中】get_submission_check 等于 check_correlation + get_alpha_details 的拼接，功能冗余，返回体过大

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1427-1437；工具 2397-2410、1986`
- 目录依据：n/a（与 getAlphaChecks 12552 对比）
- 证据：代码：`checks = {'correlation_checks': correlation_checks, 'alpha_details': alpha_details, 'all_passed': ...}`，其中 correlation_checks 的每个 check 都保留 `'correlation_data': correlation_data`（完整的 records/schema）。已单独暴露的 MCP 工具 `check_correlation`（2397）和 `get_alpha_details`（1986）返回的是同样数据。
- 影响：三个工具重叠，LLM 难以选择。get_submission_check 一次返回 prod 与 self 两份完整相关性 records（self 可能有数百行）外加整份 alpha 详情（settings/regular/is/test/pyramids/classifications……），很容易占用数万 token，挤掉上下文，而 agent 真正需要的只是一组 PASS/FAIL 项。
- 建议：按 ALPHA-1 让 get_submission_check 基于 /check 只返回精简的检查摘要（name/result/limit/value 与 selfCorrelation.max）；原始 records 通过可选参数（如 include_raw=False）按需返回；check_correlation 保留为唯一的相关性明细工具，alpha 详情由 get_alpha_details 负责，不再内联。

#### ALPHA-9 ·【中】get_user_alphas 没有暴露目录记载的 type/status 过滤，且强制带 stage；客户端与包装层默认值不一致

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:673-700（client 默认 stage="OS"），2081-2131（工具默认 stage="IS"）`
- 目录依据：listAlphas 参数表 (catalog 7609-7611)，getUserAlphaOptions (12081, 示例 12108-12194)
- 证据：代码：`params = {"stage": stage, "limit": limit, "offset": offset}`；client `stage: str = "OS"`，工具 `stage: str = "IS"`；docstring 只列出 `"IS"`、`"OS"`。目录 7609-7611：`type | query | 否 | 枚举 REGULAR / SUPER / RA_PARENT / RA_CHILD`、`stage | query | 否 | 枚举 IS / OS / PROD`、`status | query | 否 | 枚举 UNSUBMITTED / ACTIVE / DECOMMISSIONED`，并注明`无效值返回 HTTP 200 空列表`。
- 影响：agent 无法只列出 SUPER alpha，也无法按 UNSUBMITTED/ACTIVE/DECOMMISSIONED 过滤；无法省略 stage 做跨阶段查询；docstring 没提 PROD。stage 拼错时平台返回 200 空列表，agent 会误以为“没有 alpha”。client 的默认值 "OS" 永远用不到，是误导性的死默认值。
- 建议：增加 `type: Optional[str]`、`status: Optional[str]`，stage 改为 Optional，None 时不传；在本地按目录枚举校验 stage/type/status，非法值直接报错，避免静默返回空列表；统一两层默认值，docstring 补上 PROD。

#### ALPHA-11 ·【中】get_user_alphas 默认返回 30 个完整 alpha 对象，没有字段裁剪，LLM 上下文开销很大

- 类别：`tool-api-design` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:707-709；工具 2083`
- 目录依据：listAlphas 响应 schema results.items (catalog 7667-8420)
- 证据：代码：`response = await self._request('get', ..., params=params)` `return response.json()`，工具 `limit: int = 30`。目录中每个 result 含 settings、regular/selection/combo、classifications、is（含 checks、investabilityConstrained 等）、test/os/prod、pyramids、pyramidThemes、team 等大对象（7667-8420）。
- 影响：一次调用可能返回几十 KB 的 JSON，agent 在做列表或筛选时大量 token 花在用不到的字段上，翻几页就会挤占上下文。
- 建议：默认返回精简视图（id、type、stage、status、dateCreated、name、tags、settings 核心字段、is.sharpe/fitness/turnover/returns、FAIL 的 checks 名称），另加 `full: bool=False` 选项返回原始对象；同时透传 count/next，方便翻页。
- 复核修正：707-709 行直接 `return response.json()`，工具默认 limit=30（目录 7607 的平台默认值是 10），每个 result 都是含 settings/is/checks/pyramids 等字段的完整对象，上下文开销大这一点成立。但建议里“透传 count/next”已经实现：返回的是完整的响应体，已包含 count/next/previous（目录 7631-7645）。
- 修正后的建议：默认返回精简视图，增加 full=False 开关，默认 limit 降到 10。count/next 目前已随完整响应体返回，精简视图中保留即可。

#### ALPHA-10 ·【低】范围内所有请求都没有发送目录规定的版本化 Accept 头（listAlphas 需要 version=4.0）

- 类别：`missing-header` · 复核：部分成立（原评 中）
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:542, 707, 721, 1475（全文件没有设置 Accept；_build_session 228-236 只设了 User-Agent）`
- 目录依据：listAlphas Accept (7598)；getAlpha (8488)；patchAlpha (9337)；submitAlpha (12217)；getAlphaChecks (12558)
- 证据：代码 _build_session：`session.headers.update({'User-Agent': 'Mozilla/5.0 ...'})`，没有任何 Accept。目录 7598：`Accept：application/json;version=4.0`（listAlphas）；8488/9337/12217：`Accept：application/json;version=2.0`。
- 影响：目录中的 schema、字段和分页行为都是在指定版本下实测的。不带版本时服务端使用哪个默认版本，目录没有说明，代码实际得到的响应结构可能与目录不同，平台调整默认版本时解析会悄悄失效（例如 listAlphas 的 v4.0 结构）。这是脆弱依赖。
- 建议：在 _request 中增加按端点设置的 Accept 参数：listAlphas/listRelatedAlphas/getUserAlphaOptions 用 `application/json;version=4.0`，getAlpha/patchAlpha/submit/check/correlations 用 `application/json;version=2.0`。
- 复核修正：代码确实没有设置 Accept：_build_session 228-236 行只设了 User-Agent，CreddSession 也不设 Accept，requests 默认发送 */*。目录 7598/8488/9337/12217/12558 列出了 version=4.0/2.0。但目录没有说明不带版本时的行为，开头也没有解释版本语义，“响应结构会与目录不同”只是推测，目前没有证据表明现有解析因此出错；现有代码主要读取 results/count/is 等通用字段。这是一个脆弱性和对齐问题，不是已证实的故障，所以降为 low。
- 修正后的建议：在 _request 支持按调用传入 headers，给 listAlphas 发送 `application/json;version=4.0`，给 getAlpha/patchAlpha/submit/check/correlations 发送 `application/json;version=2.0`。改之前用一次实测对比带版本与不带版本的响应差异，确认收益后再全面推广。

#### ALPHA-12 ·【低】get_user_alphas 使用的 dateCreated>/<、dateSubmitted>/<、hidden 过滤以及 order 示例不在目录中，无法核实，docstring 语义也有出入

- 类别：`undocumented-endpoint` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:694-705；docstring 2101-2120`
- 目录依据：listAlphas 参数表 (catalog 7604-7611)
- 证据：代码：`params["dateCreated>"] = start_date`、`params["dateCreated<"] = end_date`、`params["dateSubmitted>"] = ...`、`params["hidden"] = str(hidden).lower()`；docstring：`Filters for alphas created on or after this date`，order 示例 `"name"`、`"-dateSubmitted"`。目录参数表只列出 limit/offset/order/type/stage/status；7608：`order ... 实测 dateCreated 与 -dateCreated 可用，无效排序字段返回 HTTP 400`。
- 影响：这些参数目录没有记载（可能是 hidden 可见性端点参数，并非一定无效）。如果服务端不认识，结果是静默的不过滤；`>` 是否包含边界（docstring 写的是 on or after）也无法确认。order 示例中只有 dateCreated 系列经过实测，其他字段一旦无效会得到 400。
- 建议：在 docstring 中注明这些过滤属于“未文档化、未核实”；order 示例改为已实测的 `dateCreated`/`-dateCreated`，其余字段标为待验证；docstring 不要断言 `>` 包含边界；建议用一次实测确认过滤确实生效（例如比较 count）。

#### ALPHA-13 ·【低】set_alpha_properties：tags 类型与目录不一致，osmosisPoints 不在目录中，本地范围校验缺乏依据

- 类别：`wrong-params` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1452-1467；工具 2420-2422`
- 目录依据：patchAlpha 请求体 schema tags (catalog 9371-9377)
- 证据：代码：`tags: Optional[List[str]] = None`、`"osmosisPoints": osmosis_points`、`if osmosis_points is not None and not (1 <= osmosis_points <= 100000): raise ValueError(...)`。目录 9371-9377：`"tags": {"type": "array", "items": {"type": "object", "additionalProperties": true}}`；body schema 中没有 osmosisPoints（只有响应 schema 10193 中有 `osmosisPoints: number|null`），body 为 `"additionalProperties": true`。
- 影响：目录示例中 tags 为空数组，无法确认字符串数组是否被接受。如果平台要求对象数组，PATCH 会失败或静默忽略 tags。osmosisPoints 是否能通过 PATCH 写入、1..100000 的范围从何而来，目录都没有说明；本地校验可能拒绝合法值，也可能放过非法值。
- 建议：用一次实测确认 tags 的元素格式和 osmosisPoints 能否写入，再据此调整类型；在拿到证据前，docstring 标明“未经目录确认”，删除或注释本地范围校验的来源。
- 复核修正：目录 9371-9377 的 body schema 中，tags.items 为 object；body 没有 osmosisPoints，但 additionalProperties=true；代码 1454-1455 行对 osmosis_points 做 1..100000 的本地校验，来源不明。字面上的不一致成立。不过目录的依据只是前端逆向推断，示例里 tags 都是空数组，对象元素并未经过实测确认；社区里普遍把 BRAIN tags 当字符串数组用。因此说“PATCH 会失败或静默忽略 tags”缺乏证据，只能列为待核实项。
- 修正后的建议：不要按目录把 tags 改成对象数组，先实测确认元素格式。osmosisPoints 的可写性和范围也需要实测，在此之前 docstring 标注为“未经目录确认”。

#### ALPHA-14 ·【低】set_alpha_properties 无法清空字段，可能发送空 body，缺少 favorite/hidden，也没说明 Super Alpha 描述长度要求

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1462-1475；工具 docstring 2424`
- 目录依据：patchAlpha 请求体 (catalog 9350-9425)：name/color/category 为 string|null，minProperties 1 (9423)，描述长度 (9401, 9414)；bulkPatchAlphas (10269-10333)
- 证据：代码：`data = {k: v for k, v in option_map.items() if v is not None}`，然后无条件 `self._request('patch', ..., json=data)`。目录 9356-9369 的 color/name/category 为 `["string", "null"]`；9423：`"minProperties": 1`；9401：`提交 Super Alpha 前，前端要求 Selection Expression 描述不少于 100 个字符。`；10296-10309 的 bulkPatchAlphas 支持 `favorite`/`hidden`/`color`。
- 影响：None 被过滤掉，agent 无法把 name/color/category 置空；所有参数都没传时会发送 `{}` 并得到 4xx，浪费一次请求；无法隐藏或收藏 alpha（整理大量 IS alpha 时常用），批量整理只能逐个调用；为 SUPER 写描述的 agent 不知道少于 100 字符会导致提交失败。
- 建议：用哨兵值区分“不修改”和“置空”；data 为空时在本地报错；为 hidden/favorite 增加批量工具（基于 PATCH /alphas，body 为 [{id, favorite|hidden|color}]）；在 docstring 注明 selection/combo 描述需不少于 100 字符（SUPER 提交前）。

#### ALPHA-M1 ·【低】相关性一直在计算时返回 {}，check_correlation 报误导性的 KeyError，而不是返回“仍在计算”

- 类别：`error-handling` · 复核：复核补充
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1231-1236（prod 返回 {}）、1291-1296（self 返回 {}）、1397-1399（KeyError）`
- 目录依据：getAlphaCorrelation 运行行为 polling/retryAfterHeader (catalog 12828-12842)
- 证据：get_production_correlation/get_self_correlation 连续 5 次拿到空响应后执行 `return {}`；check_correlation 中 `{}` 仍是 dict，但既没有 max 也没有 records，于是走到 `raise KeyError("Correlation response missing 'schema.max' or top-level 'max' and no 'records' to derive from")`。get_submission_check 再原样抛出，包装层返回 `{"error": "An unexpected error occurred: ...missing 'schema.max'..."}`。目录 12830-12833 说明这个端点是带 Retry-After 的轮询端点，空响应表示仍在计算。
- 影响：平台相关性计算较慢时，agent 等待约 80-160 秒后，收到的错误看起来像“响应格式/解析错误”，而实际是“仍在计算，稍后再查”。agent 可能误以为 alpha 或工具已损坏，而不是稍后重试；之前的等待也全部白费。
- 建议：get_*_correlation 超出等待预算时，返回明确的 `{"status": "RUNNING", "retry_after_seconds": ...}` 哨兵，不要返回 {}。check_correlation 和 get_submission_check 遇到这个哨兵时，对应项标为 PENDING，all_passed=False，不再抛 KeyError。


### ANLY · Alpha 相关性 / recordsets / 表现对比（13 条：高 2 / 中 6 / 低 5）

#### ANLY-1 ·【高】check_correlation 对未知 correlation_type 静默跳过并返回 all_passed=True（例如传 'prod' 或 'power-pool'）

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1346-1361, 1402-1413; wrapper 2397-2398`
- 目录依据：getAlphaCorrelation, catalog L12732
- 证据：代码：`if correlation_type == "both": check_types = ["production", "self"] else: check_types = [correlation_type]` ... `if check_type == "production": ... elif check_type == "self": ... else: continue`，随后 `all_passed = True` 保持不变，`results['all_passed'] = all_passed`。工具 docstring 只写 "Check alpha correlation against production alphas, self alphas, or both."，没有给出合法字符串。目录 L12732：`correlationType ... 枚举 self / power-pool / prod`——API 本身的取值是 `prod` 而不是 `production`。
- 影响：LLM 按 API 命名传 correlation_type="prod"、"power-pool"、"Self" 或拼错时，工具不发任何请求，直接返回 {checks: {}, all_passed: true}。agent 会据此判定相关性检查已通过并提交高相关 alpha，属于错误结果。
- 建议：把 correlation_type 改为 Literal["both","prod","self","power-pool"]（同时接受 "production" 作为别名），遇到未知值时抛 ValueError；checks 为空时 all_passed 不能为 True。docstring 中写明合法值，并补充 power-pool 支持（目录 L12732 已实测）。

#### ANLY-2 ·【高】recordset/correlation 轮询忽略 Retry-After，按固定节奏重试，耗尽后把“仍在计算”当成空数据返回 {}

- 类别：`polling-retry` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:614-632, 1155-1173, 1219-1236, 1279-1296`
- 目录依据：getAlphaCorrelation 运行行为 L12830-12841；getAlphaRecordset 运行行为 L13123-13133
- 证据：代码 get_alpha_pnl：`retry_delay = 2`、`# Some alphas may return 204 No Content or an empty body`、`if not text: ... await asyncio.sleep(retry_delay); retry_delay *= 1.5 ... else: ... return {}`；get_production_correlation/get_self_correlation：`retry_delay = 20` 且每次 `await asyncio.sleep(retry_delay)`，从不读取 `Retry-After`（全文件只有 simulation 相关代码 L280/L476/L2652 调用了 `_retry_after_seconds`）。目录 L12832-12833：`"polling": true, "retryAfterHeader": true`（correlations）；L13127-13128 同样适用于 recordsets。目录成功状态只列 `HTTP 200`（L12740、L13014），没有 204。
- 影响：服务端返回空 body + Retry-After 表示仍在计算。PnL/yearly-stats 只等约 16s（2+3+4.5+6.75），之后 get_alpha_pnl 返回 `{}`；MCP 层把它当成正常结果，LLM 看到的是“没有 PnL 数据”，不会知道其实是超时。相关性固定每次等 20s：服务端要求更短时白等，要求更长时 80s 后放弃。放弃后 check_correlation 拿到 `{}` 并抛出误导性的 `KeyError("Correlation response missing 'schema.max'...")`，没有说明是仍在计算。多个 MCP 客户端并发时，这种不遵守服务端节奏的重试也会加重 BRAIN 负载。
- 建议：统一实现一个 `_poll_json(url)`：响应带 Retry-After 时 sleep `_retry_after_seconds(resp)`（加上限和总超时），不带 Retry-After 的 2xx 才解析 JSON（与目录 L12370/L12540 中 “响应不含 Retry-After 时停止轮询并解析 JSON” 的语义一致）。超时后返回结构化的 `{status: "PENDING", retry_after_seconds, note}`（参照 _check_once 的做法），不要返回 `{}` 或 KeyError。

#### ANLY-3 ·【中】对 401/404/410/412/429 等 HTTP 错误一律盲目重试，单次 check_correlation 最多空耗约 160s

- 类别：`error-handling` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1224-1226 与 1262-1270（prod）；1284-1286 与 1322-1330（self）；620-621 与 660-668；1161-1162 与 1201-1209`
- 目录依据：getAlphaCorrelation 错误状态 L12741、handledStatuses L12834-12840；getAlphaRecordset 错误状态 L13015
- 证据：代码：`response.raise_for_status()` 抛出的 HTTPError 被外层 `except Exception as e: if attempt < max_retries - 1: ... await asyncio.sleep(retry_delay); continue` 捕获，状态码不做区分。目录 L12741：`错误状态：HTTP 401、HTTP 410、HTTP 412、HTTP 429、HTTP 503`；L13015：`错误状态：HTTP 401、HTTP 404`。
- 影响：alpha_id 不存在（404）、alpha 已不可用（410）或前置条件不满足（412）时，每种相关性都要等 4×20s=80s 才报错；correlation_type=both 时顺序执行，约 160s，还没算每次请求的超时，很可能超过 MCP 客户端的超时时间。429 在 2s/20s 后立即重试，不看 Retry-After，多客户端下会形成重试风暴。401 时 CreddSession 已经刷新过一次 cookie，外层循环每次重试还会再去请求 credd /cookies。
- 建议：只对 429/503/网络超时重试，并按 Retry-After 退避；401/404/410/412 立即失败，返回带 http_status 和 body 摘要的结构化错误（例如 412 应说明前置条件未满足）。

#### ANLY-5 ·【中】用客户端阈值 0.7 重算“是否通过”，没有使用服务端权威的 GET /alphas/{id}/check

- 类别：`better-endpoint-available` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1334, 1402; get_submission_check 1427-1435`
- 目录依据：getAlphaChecks L12554-12714
- 证据：代码：`passes_check = max_correlation < threshold`（threshold 默认 0.7），且 get_submission_check 用 `'all_passed': correlation_checks['all_passed']` 作为提交前检查的结论。目录 L12554 `GET /alphas/{alphaId}/check`，响应 `is.checks[]` 包含 `name`、`result` 枚举 `PASS/FAIL/PENDING`、`limit`、`value`，另有 `is.selfCorrelation{records,schema,min,max}`；运行行为 `"polling": true, "retryAfterHeader": true`。全文件没有调用 `/check`。
- 影响：平台的相关性判定规则（阈值、例外条件等）由服务端给出，在 `limit`/`result` 中返回。客户端固定用 0.7 且严格小于，可能与平台结论不一致，例如服务端判 PASS 而本工具判 FAIL，或相反，会误导提交决策。get_submission_check 的 all_passed 只看相关性，其余提交检查全部忽略，名称有误导性。
- 建议：新增 get_alpha_checks，按 Retry-After 轮询 /alphas/{id}/check，返回精简的 `[{name,result,limit,value}]`；check_correlation/get_submission_check 改为以 /check 的 SELF_CORRELATION/PROD_CORRELATION 等结果为准，correlations/* 只用于展示明细。

#### ANLY-6 ·【中】performance_comparison 调用未收录路径 /alphas/{id}/performance-comparison，并把 teamId/competition 当作 query 参数

- 类别：`undocumented-endpoint` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1690-1704; wrapper 2522-2527`
- 目录依据：getSelfAlphaPerformance L13148；getCompetitionAlphaPerformance L13644；源码证据 L13684
- 证据：代码：`params = {"teamId": team_id, "competition": competition}` 和 `self._request('get', f"{self.base_url}/alphas/{alpha_id}/performance-comparison", params=params)`。目录记录的前后表现接口是 `GET /users/self/alphas/{alphaId}/before-and-after-performance`（L13148）和 `GET /competitions/{competitionId}/alphas/{alphaId}/before-and-after-performance`（L13644），L13684 注明 `同一动态 URL 构造器按 teamId/scoringCompetition 选择个人、团队或比赛路径`。目录不含 performance-comparison 路径（目录不收录 hidden 接口，因此无法断定它不存在）。
- 影响：该路径无法用目录验证，代码中也没有注释、fallback 或 404 处理能证明它可用。按目录，teamId/competition 决定的是路径而不是 query 参数，所以即使该路径存在，这两个参数也可能不起作用。工具 docstring "Get performance comparison data" 没有说明返回结构，agent 拿不到目录中有文档的 stats.before/after（sharpe、fitness、turnover 等）、yearlyStats、pnl 对比。
- 建议：改为：competition 为空时调用 getSelfAlphaPerformance，有 competition 时调用 getCompetitionAlphaPerformance（目录 L13663-13667 标注其响应模式为 `empty`，需要处理空体并考虑轮询）；teamId 路径目录未收录，如需支持应标记为未验证。旧路径只作为带明确日志的 fallback，或者删除。

#### ANLY-7 ·【中】get_record_sets / get_record_set_data 没有任何轮询或空体处理，与 get_alpha_pnl 对同一端点的处理不一致

- 类别：`polling-retry` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1482-1504`
- 目录依据：listAlphaRecordsets 运行行为 L12974-12981；getAlphaRecordset 运行行为 L13123-13133
- 证据：代码：`response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/recordsets/{record_set_name}"); response.raise_for_status(); return response.json()`。目录 L12977-12979 与 L13127-13128：`"polling": true, "retryAfterHeader": true`。
- 影响：刚模拟完的 alpha 的 recordset 还在生成时，服务端返回空 body + Retry-After，此时 `response.json()` 抛 JSONDecodeError，工具返回 `{"error": "An unexpected error occurred: Expecting value: line 1 column 1 (char 0)"}`，LLM 无法判断是该稍后重试还是数据不存在。而 get_alpha_pnl 对同一个端点（recordsets/pnl）会重试，行为不一致。
- 建议：复用 ANLY-2 中的统一 `_poll_json`；仍在计算时返回 `{status:'PENDING', retry_after_seconds}`。

#### ANLY-8 ·【中】同一端点被多个工具/方法重复封装：get_alpha_pnl/get_alpha_yearly_stats 与 get_record_set_data 重复，prod/self 相关性方法是复制粘贴

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:609-666, 1150-1212, 1214-1272, 1274-1332, 1494-1504; wrappers 2065, 2389, 2449`
- 目录依据：getAlphaRecordset L12993；getAlphaCorrelation L12719
- 证据：get_alpha_pnl 请求 `/alphas/{alpha_id}/recordsets/pnl`，get_alpha_yearly_stats 请求 `/alphas/{alpha_id}/recordsets/yearly-stats`，get_record_set_data 请求 `/alphas/{alpha_id}/recordsets/{record_set_name}`，三者对应同一个 operationId getAlphaRecordset，其 `recordsetType` 枚举为 `pnl / sharpe / turnover / daily-pnl / yearly-stats`（L13006）。get_production_correlation 与 get_self_correlation 约 60 行逐字相同，只有 `correlations/prod` 与 `correlations/self` 不同；目录中是同一个 operation getAlphaCorrelation，用 path 参数区分 self/power-pool/prod（L12732）。四个方法各有一份相同的重试循环，结尾都有 `# This should never be reached ... return {}` 死代码。
- 影响：约 240 行重复代码。三个 MCP 工具（get_alpha_pnl、get_alpha_yearly_stats、get_record_set_data）指向同一端点但错误处理不同（见 ANLY-2/ANLY-7），LLM 选哪个工具得到的可靠性不一样，工具列表也被占用。power-pool 等新取值每次都要再复制一份。
- 建议：合并为 `get_alpha_recordset(alpha_id, name: Literal[...])` 和 `get_alpha_correlation(alpha_id, type: Literal['self','prod','power-pool'])`，两者共用 `_poll_json`；get_alpha_pnl/get_alpha_yearly_stats 要么删除，要么保留为一行调用的薄别名；删除不可达的 `return {}`。

#### ANLY-9 ·【中】PnL/daily-pnl/相关性明细原样返回全部 records，check_correlation 还会内嵌两份完整相关性数据，LLM 上下文开销大

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:637-639, 1404-1408, 1500-1501`
- 目录依据：getAlphaRecordset 示例 L13090-13120；getAlphaCorrelation schema L12750-12756
- 证据：代码：get_alpha_pnl 中 `return pnl_data`；check_correlation 中 `results['checks'][check_type] = {'max_correlation': ..., 'passes_check': ..., 'correlation_data': correlation_data}`。目录 pnl 的 records 每行为 `[date, pnl, equal-weight-pnl]`（L13093-13119），即每个交易日一行；correlation 的 records 为每个被比较 alpha 一行（L12817-12822）。
- 影响：多年回测的日度 PnL 有数千行，数十 KB 以上的 JSON 会直接进入模型上下文，挤占推理空间、增加费用。check_correlation 只需要回答“是否通过、最大相关 alpha 是谁”，却把 prod 和 self 的完整数据都塞进结果。
- 建议：默认返回摘要：PnL 给出起止日期、累计 PnL、年度汇总、可选降采样（如周/月）；相关性给出 max/min 和 top-N 最相关的 alpha。另加 `include_raw: bool=False` 参数，需要时再返回全量。

#### ANLY-4 ·【低】check_correlation 取 max 的逻辑与目录 schema 不符：schema.max 分支是死代码，max 为 null 时崩溃，records 兜底会误取任意数值

- 类别：`response-parsing` · 复核：部分成立（原评 中）
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1364-1398`
- 目录依据：getAlphaCorrelation 响应 Schema L12744-12795，示例 L12799-12825
- 证据：代码：`schema = correlation_data.get('schema') or {}; if isinstance(schema, dict) and 'max' in schema: max_correlation = float(schema['max']) elif 'max' in correlation_data: max_correlation = float(correlation_data['max'])`；records 兜底：`for v in row: vf = float(v); if -1.0 <= vf <= 1.0: candidate_max = ...`。目录 schema 中 `schema` 只有 `title/name/properties`（L12757-12775），`max` 在顶层且 `"type": ["number", "null"]`（L12782-12787）；records 每行按 schema.properties 排列，例如 `["ALPHA_ID_EXAMPLE", 0.12]`。
- 影响：(1) `'max' in schema` 按目录不会成立，是死分支。(2) 顶层 `max` 为 null 时（例如没有可比较的 alpha），`'max' in correlation_data` 为真，`float(None)` 抛 TypeError，整个工具报错，不会落到 records 分支，也不会判定通过。(3) records 兜底对行内所有值做 [-1,1] 过滤，不按 schema.properties 找 `correlation` 列。只要某列是计数或区间边界（0、1 这类值），就会被当成相关性，结果可能是误判为失败，或数值不对。
- 建议：直接读顶层 `max`；为 None 时按 schema.properties 找到 name=='correlation' 的列索引再求 max，records 也为空时返回 `max_correlation: null, passes_check: true, note: 'no comparable alphas'`（或交给 /check 判定）。删除 schema.max 分支。
- 复核修正：代码 L1366-1398 与描述一致。目录 L12757-12775 的 schema 对象里只有 title/name/properties，所以 schema.max 分支按目录确实到不了。顶层 max 的类型是 [number,null]（L12782-12787），`'max' in correlation_data` 为真时执行 float(None) 会抛 TypeError，这一点成立。不过它的结果是工具报错，不会误判为通过。records 兜底只在顶层完全没有 max 键时才走到；目录的实测示例（L12823-12824）总是带 min/max，这条路径实际很少执行。三点都成立，但实际影响有限，属于解析健壮性问题，severity 应降为 low。原建议里“records 也为空时返回 passes_check: true”不妥，会在没有数据时报告通过，与 ANLY-1 是同一类假阳性。
- 修正后的建议：直接读顶层 max；max 为 None 时，按 schema.properties 找到 name=='correlation' 的列索引，在该列上求最大值；records 也为空时返回 {max_correlation: null, passes_check: null, status: 'UNKNOWN', note: 'no comparable alphas'}，交给 /alphas/{id}/check 判定，不要默认 passes_check=true。删除 schema.max 分支。

#### ANLY-10 ·【低】本组所有请求都没有发送目录要求的 Accept: application/json;version=2.0

- 类别：`missing-header` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:230-232（session 头只设置 User-Agent）；620, 1161, 1225, 1285, 1487, 1499, 1699`
- 目录依据：getAlphaCorrelation L12723；listAlphaRecordsets L12860；getAlphaRecordset L12997；getSelfAlphaPerformance L13152
- 证据：代码：`session.headers.update({'User-Agent': 'Mozilla/5.0 ...'})`，本组请求都没有传 headers。目录：`Accept：application/json;version=2.0`。
- 影响：目录没有说明不带版本 Accept 时服务端会返回哪个版本。如果默认版本与 2.0 的结构不同（例如 max/min 字段或 records 形态），ANLY-4 中的解析就会出错；平台升级默认版本时也可能在不知情的情况下改变结构。目前没有观察到实际故障，因此列为低。
- 建议：在 _request 中按 operation 显式设置 Accept（本组为 version=2.0；如果新增 getSelfAlphaSummary，它要求 version=4.0，见 L13695）。

#### ANLY-11 ·【低】check_correlation 顺序等待 prod 和 self 两个相关性，总耗时叠加

- 类别：`concurrency-perf` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1355-1361`
- 目录依据：getAlphaCorrelation L12719
- 证据：代码：`for check_type in check_types: if check_type == "production": correlation_data = await self.get_production_correlation(alpha_id) elif check_type == "self": correlation_data = await self.get_self_correlation(alpha_id)`。
- 影响：两个相关性彼此独立，服务端也分别轮询，顺序执行会让最坏耗时从约 80s 变成约 160s，增加 MCP 调用超时的概率。
- 建议：使用 `asyncio.gather(..., return_exceptions=True)` 并发获取，每种类型的失败分别报告，不要一个失败就让整体失败。

#### ANLY-12 ·【低】expand_nested_data 作为 MCP 工具作用有限：需要 LLM 把大数据回传，且对本组的 {schema, records} 列式结构不起作用

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1708-1719; wrapper 2533-2537`
- 目录依据：n/a（getAlphaRecordset L13064-13078 说明 records 为数组的数组）
- 证据：代码：`df = pd.json_normalize(data, sep='_')`，参数 `data: List[Dict[str, Any]]`；目录 L13066：`"description": "每行元素顺序与 schema.properties 一致。"`，records 是数组的数组，不是 dict 列表。
- 影响：本组所有端点（recordsets、correlations、before-and-after-performance）返回的都是 schema+records 列式结构。json_normalize 无法把它变成带列名的表，真正需要的是把 schema.properties[].name 与 records 按位置配对。另外，该工具要求 LLM 把已拿到的大数据作为参数再发回服务器，token 翻倍；pandas 在事件循环里同步运行，大输入时会阻塞其他客户端。
- 建议：删除该工具，或在服务端内部提供 `_records_to_rows(schema, records)`，由 recordset/correlation 工具直接返回带列名的精简行（可与 ANLY-9 的摘要一起做）。

#### ANLY-13 ·【低】get_record_set_data 的 record_set_name/alpha_id 未校验，也没有文档说明合法值

- 类别：`wrong-params` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1494-1499; wrapper 2449-2450`
- 目录依据：getAlphaRecordset 参数 L13006
- 证据：代码：`f"{self.base_url}/alphas/{alpha_id}/recordsets/{record_set_name}"`，docstring 只有 "Get data from a specific record set."；目录 L13006：`recordsetType ... 枚举 pnl / sharpe / turnover / daily-pnl / yearly-stats | 从 listAlphaRecordsets 的 results[].name 取得`。
- 影响：LLM 容易猜错名称（如 'PnL'、'yearly_stats'），得到 404（目录 L13015）。任意字符串直接拼进 URL 路径，可以构造出本组以外的 GET 路径（例如包含 '/' 或 '..'），并携带会话 cookie 发出请求。
- 建议：在 docstring 中列出枚举并校验（或提示先调用 get_record_sets）；对 alpha_id 做 `^[A-Za-z0-9]+$` 校验，路径段做 URL quote。


### MISC · 事件 / 消息 / 比赛 / 教程 / Alpha List(tags)（9 条：高 0 / 中 2 / 低 7）

#### MISC-2 ·【中】get_messages 在 limit=None 时只拿到服务端默认的 10 条，docstring 暗示未限制；也没有 read/type/order 过滤

- 类别：`pagination` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1076-1083; wrapper 2289-2303`
- 目录依据：listSelfMessages L14144-14148, L14272-14279
- 证据：代码：`if limit is not None: params['limit'] = limit` / `if offset > 0: params['offset'] = offset`，wrapper 文档写的是 `limit: Maximum number of messages to return` 和 `Messages for the current user, optionally limited by count`。catalog：`limit | query | 否 | integer | 默认 10`（L14144），`type ... 枚举 ANNOUNCEMENT / NOTIFICATION`（L14147），`read | query | 否 | boolean | ... true=已读，false=未读`（L14148），`order ... 默认 -dateCreated`（L14146），运行行为 `"defaultLimit": 10`（L14277）。
- 影响：LLM 不传 limit 时以为拿到了全部消息，实际只有最新 10 条，count/next 也没有提示它还有下一页。想查“未读的比赛/提交通知”时只能把所有消息拉下来在客户端过滤，而每条 description 都是带 HTML 甚至内嵌图片的长文本，上下文开销很大。
- 建议：暴露 `read: Optional[bool]`、`type: Optional[Literal['ANNOUNCEMENT','NOTIFICATION']]`、`order` 参数并原样透传；docstring 写明默认只返回 10 条，并提示 LLM 查看 `count`/`next`；可选增加 `max_description_chars`，截断或去掉 HTML 的 description。
- 复核修正：代码 L1076-1083 在 limit 为 None 时不传 limit。catalog L14144 写明 limit 默认 10，运行行为 L14277 为 defaultLimit 10，L14146-14148 另有 order/type/read 三个过滤参数，代码都没有暴露。wrapper L2295 的“Maximum number of messages to return”和返回说明“optionally limited by count”确实会让人以为不传 limit 就不受限。但 impact 里“count/next 也没有提示它还有下一页”不对：代码把 response.json() 原样返回（只改了 results 并加上 image_handling），count/next/previous 都在结果里，LLM 能看到。问题出在 docstring 没有提醒它去看这两个字段。缺少 read/type 过滤导致上下文开销大，这一点成立。
- 修正后的建议：透传 read（Optional[bool]）、type（ANNOUNCEMENT/NOTIFICATION）和 order。docstring 写明服务端默认每页 10 条，需要翻页时查看 count/next 并调整 offset。可选增加一个参数，用来截断 description 或去掉其中的 HTML。

#### MISC-3 ·【中】get_events 声称返回“events and competitions”，但 /events 只有线上/线下活动；只取第一页 10 条，且默认按 -start 排序（包含已结束的活动）

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:731-741; wrapper 2180-2191`
- 目录依据：listEvents L13836-13844, L13906-13911, L14015-14022
- 证据：代码：`"""Get available events and competitions."""`、`response = await self._request('get', f"{self.base_url}/events")`（无参数），wrapper 写的是 `🏆 Get available events and competitions.`。catalog：`limit ... 默认 10`（L13838）、`order ... 默认 -start；枚举 start / -start`（L13839）、`start>= ... 起始时间下界`（L13840）、`type ... 枚举 ONLINE / OFFLINE`（L13841），results.type enum 只有 `ONLINE`/`OFFLINE`（L13906-13911）。比赛由另一个端点 `GET /competitions`（listCompetitions，L16862-16864）提供。
- 影响：LLM 会把 get_events 当成发现比赛的入口，但返回的是研讨会之类的活动，找不到可参加的比赛 ID。而且只拿到按开始时间倒序的前 10 条，可能夹着已结束的活动，也可能漏掉即将开始的。
- 建议：把 docstring 改成“列出 BRAIN 线上/线下活动（讲座、研讨会），不包含比赛”；暴露 `limit`、`order`、`start_after`（映射为 `start>=`）、`type`、`language`；另外新增基于 listCompetitions 的 list_competitions 工具（见 unused endpoints）。

#### MISC-1 ·【低】get_messages 默认把消息里的 base64 图片写到服务端磁盘，并把服务端相对路径返回给远程 MCP 客户端

- 类别：`security` · 复核：部分成立（原评 中）
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1001-1002, 1045-1058, 1105; 服务监听 L1789-1794, L2772-2773`
- 目录依据：listSelfMessages, catalog L14217-14219（description: string）
- 证据：代码：`image_handling = os.environ.get("BRAIN_MESSAGE_IMAGE_MODE", "placeholder").lower()`、`save_dir = pathlib.Path("message_images")`、`with open(file_path, "wb") as f: f.write(base64.b64decode(b64_data))`、`replacement = f"[Image extracted -> {file_path}]"`，并且在 `run_in_executor(None, _sanitize_all)` 里执行。服务是 `FastMCP(..., host="0.0.0.0", port="8761")` + `transport="streamable-http"`。catalog 只把 description 定义成 string（L14217-14219），对图片内嵌格式没有说明。
- 影响：(1) 一个读操作带来了写磁盘的副作用：文件写在进程 CWD 下的相对目录 message_images/，从不清理，每次调用都重复解码、覆盖写入。(2) 客户端通过 HTTP 远程连接，拿到的 `message_images/<id>_1.png` 是服务端本地的相对路径，客户端打不开，这个“提取”对 LLM 没有用处。(3) 多个客户端并发调用时，会在默认线程池里同时写同名文件，产生竞态，可能写出截断的图片。(4) 每张图最多允许约 5MB 解码（`len(b64_data) > 7_000_000` 才拒绝），在 CPU 和内存上都是无意义的开销。
- 建议：默认改成 ignore（只把 <img data:...> 换成 `[image: png, ~NKB]` 占位），不再写磁盘；如果确实需要图片，另外提供一个按 message_id 和序号返回单张图片（MCP image content）的工具。至少应把写入目录改成带绝对路径的可配置目录，并在工具 docstring 中说明。
- 复核修正：已核对 platform_functions.py L1001-1002（默认 BRAIN_MESSAGE_IMAGE_MODE=placeholder，save_dir=pathlib.Path("message_images")）、L1045-1058（base64 解码后写入 file_path，replacement=f"[Image extracted -> {file_path}]"）、L1105（run_in_executor(None, _sanitize_all)），以及 L1789-1794 的 host="0.0.0.0"、L2772 的 streamable-http。确实存在：读操作带写盘副作用；返回的是相对服务端 CWD 的路径，客户端无法使用；解码开销没有意义。不过有两处说法不成立。(1) 并发写同名文件不会留下截断文件。每个线程各自 open("wb")，从 offset 0 写入完全相同的字节，全部 close 后文件内容完整，最多只是写入过程中短暂可读到截断内容。(2) 文件名是 {message_id}_{idx}，重复调用只会覆盖，不会无限增长，磁盘占用上限就是消息图片总量。catalog L14217-14219 只把 description 定义为 string，这一点无误。这个问题本质上是工具设计和资源浪费，并没有可利用的安全漏洞，归为 security/medium 偏高。
- 修正后的建议：类别改成 tool-api-design，严重度 low。默认改用 ignore 模式，只替换成 `[image: png, ~NKB]` 这样的占位符，不解码也不写盘。确实需要图片时，另外提供一个按 message_id 和序号返回 MCP image content 的工具。如果保留写盘模式，写入目录应可配置且为绝对路径，并在 docstring 中说明路径只在服务端有效。

#### MISC-4 ·【低】get_competition_agreement 调用的 /competitions/{id}/agreement 不在 catalog 中，无法验证；404 被当成“unexpected error”，也没有回退

- 类别：`undocumented-endpoint` · 复核：部分成立（原评 中）
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1610-1620; wrapper 2497-2503`
- 目录依据：not in catalog；对比 getCompetition L17138-17161 与 schema 字段 description/faq/helpText/scoring（L17015-17047 同构）
- 证据：代码：`response = await self._request('get', f"{self.base_url}/competitions/{competition_id}/agreement")` + `response.raise_for_status()`，wrapper 统一返回 `{"error": f"An unexpected error occurred: {str(e)}"}`。catalog 全文 grep 'agreement' 无结果。Competition 域只收录了 listCompetitions/getCompetition/listCompetitionLevels/listUserCompetitions/boards/submissions/spc 这几个操作。getCompetition 的响应已经带有 `description`、`faq`、`helpText`、`scoring` 字段。
- 影响：catalog 不收录 hidden 接口，所以不能断定这个端点坏了。但它是否存在、返回什么结构都无法确认。如果某个比赛没有 agreement，或者端点不存在，LLM 只会看到一条泛化的错误，得不到规则信息，而这些信息很可能就在 get_competition_details 返回的 description/faq 里。这两个工具的功能也可能重叠。
- 建议：遇到 404/403 时返回结构化结果 `{"available": false, "http_status": ...}`，并提示改看 get_competition_details 的 description/faq/helpText；或者干脆把 agreement 合并进 get_competition_details（增加 include_agreement 参数），先用真实账号验证端点是否存在再决定是否保留这个独立工具。
- 复核修正：已核对代码 L1610-1620 和 wrapper L2497-2503。catalog 全文 grep 'agreement' 无结果；Competition 域的 operationId 只有 listCompetitions/getCompetition/listCompetitionLevels/listUserCompetitions/getCompetitionBoardOptions/listCompetitionBoard/competition submissions/SPC 这几个，确实没有 agreement。仓库内也没有其他地方引用或验证过这个端点。getCompetition 的 schema 包含 description、scoring、faq、helpText 字段（L17150 之后的 properties 部分）。所以“无法验证、与 get_competition_details 可能重叠”成立。但 impact 里“LLM 只会看到一条泛化的错误”说过头了：raise_for_status 抛出的 HTTPError 文本形如“404 Client Error: Not Found for url: ...”，wrapper 用 str(e) 原样透出，状态码是看得到的。由于端点是否存在未知，而 catalog 又不收录 hidden 端点，这个问题定为 low 更合适。
- 修正后的建议：先用真实账号确认 /competitions/{id}/agreement 是否存在。如果不存在，删除这个工具，并在 get_competition_details 的 docstring 中说明规则在 description/faq/helpText/scoring 字段里。如果存在，遇到 404 时返回 {"available": false, "http_status": 404}，并提示改用 get_competition_details。

#### MISC-5 ·【低】get_user_competitions 为了拿用户 ID 多发一次 GET /users/self，而 catalog 明确 userId 可直接写 self；失败时还会静默回退

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1581-1591`
- 目录依据：listUserCompetitions L17465-17477
- 证据：代码：`if not user_id: user_response = await self._request('get', f"{self.base_url}/users/self") ... else: user_id = 'self'`，随后调用 `f"{self.base_url}/users/{user_id}/competitions"`。catalog L17477：`userId | path | 是 | string | — | WQ 用户 ID；使用 self 可读取当前登录用户。`
- 影响：每次调用多一次往返（多占一个线程池名额），还多了一个失败点：/users/self 返回非 200 时会被静默吞掉。get_leaderboard（L751-757）也有同样的模式。
- 建议：直接写 `user_id = user_id or 'self'`，删掉预取 /users/self 的代码；同时修正 get_leaderboard 里同样的写法。

#### MISC-6 ·【低】范围内所有请求都没有带 catalog 标注的 Accept: application/json;version=2.0

- 类别：`missing-header` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:230-232（session 只设置了 User-Agent），736, 974, 1083, 1591, 1603, 1615, 1728`
- 目录依据：listEvents L13830；listSelfMessages L14136；listUserCompetitions L17471；getCompetition L17144；listTutorials L19972；getTutorialPage L20150
- 证据：代码：`session.headers.update({'User-Agent': 'Mozilla/5.0 ...'})`，整个文件 grep 'Accept' 无结果。catalog 每个在范围内的操作都写着 `- Accept：\`application/json;version=2.0\``，并且 events/messages 的 results item schema 是 `"additionalProperties": false`（L13980, L14244）。
- 影响：catalog 的 schema 和实测样例都是在带版本头的前提下得到的。不带版本头时，服务端可能返回默认或旧版本的结构，catalog 对此没有说明，因此无法保证字段一致（例如 description、tags、read）。以后服务端切换默认版本时，这里会静默出错。
- 建议：在 _request 里按端点设置 Accept，或者全局默认 `application/json;version=2.0`，个别端点单独覆盖（例如 listTagAlphas 是 version=4.0，见 L15973）。
- 复核修正：已核对 L229-231：session 只设置了 User-Agent，全文件没有设置 Accept，requests 默认发送 Accept: */*。catalog 在范围内的每个操作都标注了 Accept: application/json;version=2.0（例如 L13830、L14136、L17144、L19972、L20150），listTagAlphas 标的是 version=4.0（L15973）。但 catalog 没有任何地方说明不带版本头时服务端返回什么（grep “Accept 头”等关键字无结果），而且这些工具目前看起来在不带头的情况下都能正常用，所以“字段不一致”只是推测，没有证据。另外这是全局性问题，不属于本组特有。
- 修正后的建议：作为全局加固项处理：在 _request 中默认设置 Accept: application/json;version=2.0，并允许个别端点覆盖（例如 tags alphas 用 4.0）。上线前先用真实账号对比带头和不带头的响应是否一致，不宜直接定性为缺陷。

#### MISC-7 ·【低】competition_id / page_id 直接拼进 URL，urllib3 会规范化 ../，这两个工具可被用作任意 BRAIN GET 代理

- 类别：`security` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1603, 1615, 1728`
- 目录依据：getCompetition L17152；getTutorialPage L20158
- 证据：代码：`f"{self.base_url}/competitions/{competition_id}"`、`f"{self.base_url}/tutorial-pages/{page_id}"`。本地验证：`parse_url('https://api.worldquantbrain.com/competitions/../users/self/messages').path` 输出 `/users/self/messages`（urllib3 2.6.3）。
- 影响：如果 LLM 受到提示注入（例如论坛内容或消息正文里带指令），传入 `../users/self/...` 这样的值，就能用用户的 cookie 读取任意 GET 端点，并把结果回显到上下文里。只限 GET，影响有限，但绕过了工具本身的语义边界。
- 建议：用 `urllib.parse.quote(competition_id, safe='')` 转义路径参数，或者用正则校验 ID 格式（例如 `^[A-Za-z0-9_-]+$`）。
- 复核修正：已核对 L1603、L1615、L1728 的 f-string 拼接，并在本地用 requests 2.33.1 / urllib3 2.6.3 复现：`Request('GET','https://api.worldquantbrain.com/competitions/../users/self/messages').prepare().url` 规范化为 `https://api.worldquantbrain.com/users/self/messages`。另外含 '?' 的 ID 可以注入 query 参数（例如 'x?limit=1/agreement' 会原样变成 query）。所以现象本身属实。但所谓“绕过语义边界”在这个服务里意义不大：check_simulation_progress（L510、wrapper L1951）直接接收任意 progress_url 并用带 cookie 的 session 发 GET，出错时回显 body[:500]，没有 alpha 字段时回显 raw body。任意 GET 的能力早就存在，而且还不限主机。只修这两个工具作用有限。
- 修正后的建议：对 competition_id/page_id 用 urllib.parse.quote(x, safe='') 转义，或者按 ^[A-Za-z0-9_-]+$ 校验。更重要的是在 _request 或 check_simulation_progress 中统一校验 URL 必须以 self.base_url 开头，否则就算修了这两处也无济于事。

#### MISC-8 ·【低】get_documentation_page 的 docstring 没有说明 page_id 必须取自 get_documentations 的 results[].pages[].id，而不是 results[].id

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:2542-2548, 1723-1728`
- 目录依据：listTutorials L20020-20023, L20043-20046
- 证据：wrapper：`"""Retrieve detailed content of a specific documentation page/article."""`。catalog：results[].id `"文档教程标识；不应作为 /tutorial/{tutorialId} 的参数。"`（L20022），pages[].id `"用于请求 /tutorial-pages/{pageId}。"`（L20045）。
- 影响：LLM 很容易把教程 ID（results[].id）当作 page_id 传入，只会拿到一个泛化错误，白白浪费调用。
- 建议：docstring 写明“page_id 取自 get_documentations() 返回的 results[].pages[].id”；返回内容可以按块类型压缩（例如把 SIMULATION_EXAMPLE 提取成 expression+settings 列表，把 IMAGE 只保留 title/url），减少上下文占用。

#### MISC-9 ·【低】get_documentations / get_user_competitions 返回分页结构但只取第一页，没有暴露 limit/offset

- 类别：`pagination` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:974, 1591`
- 目录依据：listTutorials L19980（limit 默认 50）；listUserCompetitions 响应含 count/next/previous（L17495-17510），参数表只有 userId（L17477）
- 证据：代码：`self._request('get', f"{self.base_url}/tutorials")`、`self._request('get', f"{self.base_url}/users/{user_id}/competitions")`，都不带参数，也不看 `next`。catalog：`/tutorials` 的 `limit | query | 否 | integer | 默认 50`；listUserCompetitions 的运行行为写着“没有从前端确认到额外的分页、轮询或重定向规则”（L17723-17725）。
- 影响：目前样例 count=11（L20103），所以默认 50 条够用。以后教程或比赛数量超过默认页大小时会被静默截断。listUserCompetitions 是否支持 limit/offset，catalog 没有说明。
- 建议：get_documentations 显式传 `limit` 并在 `next` 非空时继续翻页；get_user_competitions 在 `next` 非空时跟随 next URL（或至少在结果里附加 `truncated: true` 提示）。


### FORUM · 论坛 / 术语表（support.worldquantbrain.com）（20 条：高 0 / 中 10 / 低 10）

#### FORUM-1 ·【中】未走 Zendesk SSO、也不检测“未被授权”页：仅注入 BRAIN cookie 很可能拿不到论坛（顾问区）会话，失败时只表现为超时或静默的空/部分结果

- 类别：`auth-session` · 复核：部分成立（原评 高）
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:128-162, 211-218, 321-322`
- 目录依据：n/a（catalog 不覆盖 support.worldquantbrain.com）；getAuthentication 行 119-126 仅校验 api 域会话
- 证据：wqmcp 只做 cookie 注入后直接访问页面：`await context.add_cookies(playwright_cookies)` → `await page.goto(initial_url)` → `await page.wait_for_selector('.post-body, .article-body', timeout=15000)`。同仓库的 wq-rag/wq-doc-forum/sync/forum/browser.py 在注入同样 cookie 之后还有一步显式 SSO：`await page.goto(f"{settings.support_base}/access/sso", ...)`，并轮询直到 `"support.worldquantbrain.com/hc" in page.url`；它的 detail_sync.py:85-98 还专门检测 `_FORBIDDEN_MARKERS = ("您未被授权", "Unauthorized", "未被授权访问")`，注释写着“先检测"未被授权"页：节省 3×45s 的徒劳重试”。wqmcp 两步都没有。catalog 对 Zendesk 站点没有任何说明，无法据此核实。
- 影响：如果顾问区帖子需要 Zendesk 会话（姊妹项目的实现说明确实需要），read_forum_post 会一直等到 15s 超时，然后返回“Timeout waiting for selector”这类笼统报错；search_forum_posts 的搜索页可能只返回公开内容，或落到登录/未授权页后被当作“无更多结果”，最后返回 success:True 和空或不完整的结果。Agent 无法分辨“没有帖子”和“没登录”这两种情况。因为是通过同仓库代码推断而非实测，定级时按“可能”处理。
- 建议：复用 wq-doc-forum 的 forum_browser()：注入 cookie 后先 goto /access/sso，等跳转回 /hc 再继续。每次 goto 之后检查 page.title()/URL，识别出未授权或登录页时明确报 `forum_auth_required` 错误，不要按超时或“无结果”处理。
- 复核修正：已核对：forum_functions.py:128-162 只做了 cookie 注入（add_cookies），随后 321-322 直接 goto 并 wait_for_selector，既没有 SSO 步骤，也没有检测未授权页。姊妹项目 wq-rag/wq-doc-forum/sync/forum/browser.py:44-54 在注入 cookie 后确实会 goto {support_base}/access/sso 并轮询 URL 直到包含 /hc；detail_sync.py:86-97 有 _FORBIDDEN_MARKERS 检测。两边注入的都是 BRAIN API cookie，姊妹项目仍需显式 SSO，这说明 Zendesk 会话需要单独建立。catalog 不覆盖 support 域，这一点无法实测。公开帖子和术语表在匿名状态下大概率仍可访问，所以失效范围只在需要登录的内容，并非整个工具不可用。结论是推断，没有实测，定级降为 medium。
- 修正后的建议：注入 cookie 后先 goto /access/sso，等 URL 回到 /hc 再继续（与 wq-doc-forum 保持一致）。每次 goto 后检查 page.title()/URL 中的未授权标记，命中时返回明确的 forum_auth_required 错误。修改前先实测匿名访问与 SSO 后访问的差异，确认这一步的必要性。

#### FORUM-2 ·【中】get_text(strip=True) 会把段落、<br>、<pre> 中的多行直接拼在一起，帖子正文和评论里的 Alpha 表达式/代码被破坏

- 类别：`response-parsing` · 复核：部分成立（原评 高）
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:339, 386, 261`
- 目录依据：n/a
- 证据：`post_data['body'] = body_element.get_text(strip=True) if body_element else 'Body not found'`（339）；`'body': body_element.get_text(strip=True) if body_element else ''`（386）；`snippet = snippet_element.get_text(strip=True)`（261）。BeautifulSoup 的 get_text(strip=True) 在没有 separator 时，会把每个文本节点 strip 后用空字符串连接。例如 `<pre>x = ts_mean(returns, 20)<br>group_neutralize(x, industry)</pre>` 会变成 `x = ts_mean(returns, 20)group_neutralize(x, industry)`；`<p>a</p><p>b</p>` 会变成 `ab`。
- 影响：Agent 从论坛读模板或表达式（docstring 示例本身就是“新人求模板”）时，拿到的是多行粘在一起的表达式，分号、换行、语句边界都丢了，直接拿去模拟就会出语法错误或得到错误的表达式；段落之间也没有空格，可读性和检索效果都变差。
- 建议：改用 `get_text(separator='\n', strip=True)`。对 <pre>/<code> 块单独提取并保留原始换行，可以把代码块单独放进一个 `code_blocks` 字段返回。
- 复核修正：339/386/261 确实是 get_text(strip=True)，没有 separator。BS4 在这种情况下会把每个文本节点 strip 后用空串拼接，所以 <p>a</p><p>b</p> 变成 ab、<br> 分隔的多行被粘连，这些都属实。不过 <pre>/<code> 中以字面换行书写的单个文本节点只会去掉首尾空白，内部换行能保留下来，所以“代码块一律被破坏”的说法过重。实际受害的是 <br> 分行和分段落书写的表达式。这是正确性问题，但不至于 high。
- 修正后的建议：改为 get_text(separator='\n', strip=True)。另外把 pre/code 块单独用 get_text() 原样提取（不加 strip），放进 code_blocks 字段返回。

#### FORUM-3 ·【中】wait_for_selector 默认要求元素可见，而帖子的 .post-body 常处于隐藏状态；此外没有评论或搜索翻到末页时，每次都要白等 10-15 秒超时

- 类别：`polling-retry` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:215, 322, 327, 362`
- 目录依据：n/a
- 证据：`await page.wait_for_selector('.post-body, .article-body', timeout=15000)` 连续出现了两次（322、327，第二次是重复的）；`await page.wait_for_selector('.comment-list', timeout=10000)`（362）；`await page.wait_for_selector('ul.search-results-list', timeout=15000)`（215）。这些调用都没有传 state，Playwright 默认 state='visible'。同仓库 wq-doc-forum/sync/forum/detail_sync.py:100-104 有注释：`# state=attached: 元素在 DOM 中就算 ready，不要求 CSS visible（社区帖的 .post-body 经常是 hidden，BS4 仍能解析）`，对应代码是 `state="attached"`。
- 影响：.post-body 隐藏时 read_forum_post 会直接超时失败。没有评论的帖子要多等 10 秒；每次搜索翻到最后一页之后还要再等 15 秒才会退出循环。并发调用时，这些浏览器会长时间占着资源。
- 建议：改用 `state='attached'`，删掉重复的 wait。评论区改为先判断 `.comment-list` 是否存在，或者直接解析 page.content()；搜索页等 `ul.search-results-list, .search-results-none`（或整页 load 完成）之后再解析，不要靠超时来判断“没有结果”。

#### FORUM-4 ·【中】search_query 直接拼进 URL，没有编码，含 & # + % 的查询会被截断或改写

- 类别：`wrong-params` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:207`
- 目录依据：n/a
- 证据：`search_url = f"{self.base_url}/hc/{locale}/search?page={page_num}&query={search_query}#results"`
- 影响：查询 `rank & ts_mean` 时，服务端只收到 `query=rank `；`group#1` 中 # 之后的部分会变成 fragment；`a+b` 会被解析成 `a b`；含 % 的字符串可能形成非法的转义序列。结果是静默地搜了另一个词，Agent 还以为这就是原始查询的结果。
- 建议：用 `urllib.parse.urlencode({'page': page_num, 'query': search_query})` 构造查询串，并把 locale 作为工具参数暴露出来（现在 BrainApiClient.search_forum_posts 没有透传 locale）。

#### FORUM-5 ·【中】read_forum_post 的参数叫 article_id，但裸 ID 一律被拼成 /community/posts/ 路径；文章类 ID 会 404 或超时，而且首屏状态码没有检查

- 类别：`wrong-endpoint` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:311-322；alpha-optimize/wqmcp/platform_functions.py:2362-2376`
- 目录依据：n/a
- 证据：`initial_url = f"https://support.worldquantbrain.com/hc/zh-cn/community/posts/{post_url_or_id}"`（314）；首次 `await page.goto(initial_url)`（321）没有检查 response.status。工具签名是 `read_forum_post(article_id: str, ...)`，docstring 写的是“Get a specific forum post by article ID”；而 search_forum_posts 的结果和术语表本身都是 `/hc/.../articles/<id>`（例如 174 行的 4902349883927）。
- 影响：Agent 拿搜索结果中某篇 article 的 ID，或按参数名理解传入 article ID 时，访问的会是 community/posts/<article_id>，结果是 404 页面，要等 15 秒超时才报一个看不出原因的错误。docstring 里也没有说明 include_comments 参数。
- 建议：参数改名为 `post_url_or_id`，并在 docstring 中说明：只接受 community post 的 ID，或者任意完整 URL；article 请传完整链接。首次 goto 之后检查 `response.status`，404/403 时返回明确错误。也可以支持 `kind: post|article` 参数。

#### FORUM-7 ·【中】每次调用都新起 Playwright 驱动和一个 Google Chrome 进程，没有并发上限，没有缓存，也没有总超时；第 1 页还会加载两次

- 类别：`concurrency-perf` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:139, 166, 195, 306, 321, 352-358`
- 目录依据：n/a
- 证据：`async with async_playwright() as p:`（每个方法各一次）；`browser = await p.chromium.launch(channel="chrome", headless=True, args=['--no-sandbox'])`（139）。read_full_forum_post 先 `await page.goto(initial_url)`，随后评论循环又从 `page_num = 1` 开始 `goto(f"{base_url}?page={page_num}#comments")`，同一页加载了两次。评论循环是 `while True:`，没有页数上限；术语表每次调用都重新启动浏览器抓取同一篇静态文章。
- 影响：streamable-http 下多个客户端并发调用时，每个调用都占用约 200MB 以上的 Chrome 进程加一个 node driver，可能耗尽内存和 CPU。channel="chrome" 要求机器上装有品牌版 Google Chrome，只执行过 `playwright install chromium` 的服务器会在 launch 时直接失败。评论多的帖子要连续加载几十页，很容易超过 MCP 客户端的超时时间。
- 建议：进程内复用同一个 browser（懒加载单例，配合 asyncio.Semaphore 限制并发 page 数）。launch 时优先用默认 chromium，失败再回退到 chrome。术语表结果做 TTL 缓存（例如 24 小时）。给评论翻页设置 max_comment_pages，并给整个调用加 asyncio.wait_for 总超时。评论从第 2 页开始翻，复用首屏已解析的第 1 页。

#### FORUM-9 ·【中】三个论坛工具仍暴露 email/password 参数，但这两个参数全链路都被忽略；wrapper 还会从 user_config.json 读出明文密码往下传

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:2307-2328, 2334-2357, 2362-2384, 1124-1148, 355-358；alpha-optimize/wqmcp/forum_functions.py:128, 134`
- 目录依据：n/a
- 证据：工具签名：`async def get_glossary_terms(email: str = "", password: str = "")`，docstring 写的是 `password: Your BRAIN platform password (optional if in config)`；函数体是 `password = password or credentials.get("password", "")`，代码注释自己承认 `email/password are legacy pass-throughs and may be empty`；最终落到 authenticate，其 docstring 为 `email/password arguments are accepted for backward compatibility but ignored`。
- 影响：LLM 会按 docstring 的提示去要用户的密码，或者把密码作为工具参数传进来。密码因此进入对话记录和 MCP 日志，却完全不起作用，这是一个无意义的敏感信息暴露面。load_config 每次调用还会从磁盘读取明文密码。ForumClient 和 BrainApiClient 的签名也都被迫保留这两个无用参数。
- 建议：从三个 MCP 工具、BrainApiClient 的三个直通方法以及 ForumClient 各方法中删除 email/password 参数，同时删掉 load_config/credentials 相关代码。docstring 改为说明“登录状态由 credd 提供”。

#### FORUM-10 ·【中】返回体积没有上限：read_forum_post 返回全部评论和完整正文，search 默认 50 条且 max_results 不设上限，容易撑爆 LLM 上下文

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:193, 206, 349-403；alpha-optimize/wqmcp/platform_functions.py:2334-2335`
- 目录依据：n/a
- 证据：`async def search_forum_posts(..., max_results: int = 50, ...)`，循环条件是 `while len(search_results) < max_results:`，没有上限；read_full_forum_post 的评论循环是 `while True:`，全部评论都 append 进 `comments`，每条都带完整 body；返回 `{"success": True, "post": post_data, "comments": comments, "total_comments": len(comments)}`。另外 max_results<=0 时仍然会先完成认证、启动浏览器，然后返回空结果。
- 影响：热门帖子有几百条评论时，一次调用就可能返回数万 token；搜索结果里每条都带 snippet 和 breadcrumbs，50 条也不小。Agent 的上下文会被迅速挤占甚至截断，还要为浏览器翻页付出很长的延迟。
- 建议：增加 `max_comments`（默认例如 30）、`comment_offset`，以及正文/评论的截断长度（例如 `max_body_chars`）。max_results 默认改为 10，上限设为例如 50，并做参数校验。返回里加上 `truncated`/`has_more` 字段，方便 Agent 按需继续翻页。

#### FORUM-11 ·【中】术语表解析启发式很脆弱：内联链接会被切成新词条，首字母大写的短定义行会被误判为词条，“ago”子串过滤会误删正常定义，页面结构变化时静默返回 []

- 类别：`response-parsing` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:52-66, 68-115, 174`
- 目录依据：n/a（catalog 中没有术语表接口）
- 证据：`lines = article_body.get_text(separator='\n').split('\n')` 会在每个内联标签（<a>/<strong>/<em>/<code>）处断行；`_looks_like_term` 只要 `is_short and ... (starts_with_capital or has_all_caps)` 就判定为词条；过滤条件 `"ago" not in term["definition"]` 按子串匹配，会误删含 diagonal、Chicago、hexagon 等词的定义；`len(term["definition"]) > 10` 会丢掉很短的定义；`if not article_body: return []` 在页面改版时返回空列表，但不会报错；URL 硬编码为 `.../articles/4902349883927-Click-here-for-a-list-of-terms-and-their-definitions`。
- 影响：定义中出现 `... uses the <a>Fast Expression</a> language` 时，“Fast Expression”会被当成一个新词条，原定义被截断，后半句还被错挂到假词条下面。以大写单词开头且不超过 80 字符的定义行（例如 “Sharpe ratio divided by ...”）也会被切成新词条。一旦 Zendesk 主题或 DOM 改版，工具会返回 [] 而不是报错，Agent 会以为术语表本来就是空的。
- 建议：改为基于 DOM 结构解析（例如 <strong>/<h3>/<dt> 作为词条，后续兄弟节点作为定义），不再依赖逐行文本启发式。去掉“ago”子串过滤，改为只在导航区域外解析。找不到 `.article-body` 或解析出 0 个词条时明确报错。另外可以考虑优先使用 Zendesk Help Center JSON 接口（`/api/v2/help_center/{locale}/articles/{id}.json`，不在 catalog 中，需要实测能否访问）获取 body HTML，从而省掉浏览器。

#### FORUM-12 ·【中】翻页过程中任何异常都被当作“到末页”，仍然返回 success:True 和部分结果；get_glossary_terms 出错时返回的是一个看起来像词条的列表

- 类别：`error-handling` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:210-218, 289-295, 357-365；alpha-optimize/wqmcp/platform_functions.py:2329-2331`
- 目录依据：n/a
- 证据：搜索循环：`except Exception as e: log(f"Could not load search results on page {page_num}: {e}", "INFO"); break`，之后仍然 `return {"success": True, "results": search_results, "total_found": len(search_results)}`；评论循环：`except Exception as e: log(f"Could not load page {page_num}: {e}. Assuming end of comments.", "INFO"); break`；response 为 None 时（goto 的同文档导航）访问 `response.status` 会抛 AttributeError，同样被当作结束。get_glossary_terms 的 wrapper 为 `return [{"error": str(e)}]`。
- 影响：网络抖动、单页超时、被限流或出现登录页时，Agent 会收到“成功”的截断结果，而 total_found 只是本次返回的条数，不是总命中数，Agent 很可能据此得出“论坛里只有 N 条相关讨论”的错误结论。术语表出错时返回的是 list[dict]，与正常返回结构相同，LLM 可能把 error 当成一个词条。
- 建议：区分“自然结束”（无结果区块或 404）和“异常中断”，异常时返回 `partial: true` 和 `error` 字段，或者重试一次。total_found 改名为 returned_count。get_glossary_terms 统一返回 dict：`{"terms": [...]}` / `{"error": ...}`。

#### FORUM-6 ·【低】每次论坛调用都强制 credd 刷新 cookie 并请求 /authentication，并发时容易触发 credd 的 rate_limited/backoff；另外 status 检查是死代码

- 类别：`auth-session` · 复核：部分成立（原评 中）
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:131-137；alpha-optimize/wqmcp/platform_functions.py:355-395, 406-410`
- 目录依据：getAuthentication · GET /authentication（catalog 行 119-126）
- 证据：forum_functions.py:134 `auth_result = await brain_client.authenticate(email, password)`；authenticate 内部是 `await loop.run_in_executor(self._executor, session.refresh_cookies)`（365），接着 `self._request('get', f"{self.base_url}/authentication")`。platform_functions 自己的 ensure_authenticated 注释写着“Intentionally cheap: no global lock and no per-call network probe — an expired cookie self-heals inside CreddSession”。_fetch_from_credd 中明确有 `elif code in ("backoff", "rate_limited")` 这一错误分支。另外 authenticate 失败时会直接抛异常，所以 `if auth_result.get('status') != 'authenticated'` 永远不会成立。
- 影响：多个 MCP 客户端并发搜论坛或读帖时，每次调用都要额外打两次网络请求，还会强制轮换全局 cookie jar。只要 credd 返回 rate_limited/backoff，论坛工具就整体失败，哪怕现有 cookie 完全有效。术语表页面本来很可能是公开的，credd 一挂也拿不到。这和 BRAIN API 其他工具采用的“按需自愈”策略不一致。
- 建议：把 authenticate() 换成 `await brain_client.ensure_authenticated()`（只保证 session 存在），不再强制 refresh。如果确实要校验，给 /authentication 结果加一个短 TTL 缓存。公开页面（术语表）的认证失败应降级为匿名访问，而不是直接抛错。删除死代码检查。
- 复核修正：forum_functions.py:134 每次调用都执行 brain_client.authenticate。platform_functions.py:365 会强制 refresh_cookies（向 credd 发 GET /cookies），367 再发 GET /authentication。ensure_authenticated（406-410）的注释明确说要避免每次调用都做网络探测。authenticate 只有两种结果：返回 status='authenticated'，或者抛异常（377-395），所以 135 行的检查是死代码，这一点属实。不过按 platform_functions 注释，credd 只在 cookie 真过期时才重新登录，GET /cookies 本身开销很低；credd 处于 backoff/rate_limited 时现有 cookie 往往也已失效，“cookie 有效却整体失败”的场景并不常见。结论是多了两次请求、行为与其他工具不一致，定级 low 更合适。
- 修正后的建议：改用 await brain_client.ensure_authenticated()，让 CreddSession 在 401 时自愈，不再强制刷新。删除 135-136 行的死代码。如果论坛在 SSO 阶段需要校验会话，可以用短 TTL 缓存 is_authenticated() 的结果。

#### FORUM-8 ·【低】BeautifulSoup 解析整页 HTML 和 O(n²) 的评论去重都在事件循环里同步执行，与模块“nothing blocks the shared event loop”的承诺相矛盾

- 类别：`concurrency-perf` · 复核：部分成立（原评 中）
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:179, 221, 329, 367, 390；alpha-optimize/wqmcp/platform_functions.py:10-12, 2323`
- 目录依据：n/a
- 证据：async 方法中直接执行 `terms = _parse_glossary_terms(content)`、`soup = BeautifulSoup(content, 'html.parser')`、`comment_soup = BeautifulSoup(await page.content(), 'html.parser')`，并用列表线性查找 `if comment_data not in comments:` 去重。模块 docstring 写着：“all BRAIN I/O runs in a bounded thread pool ... nothing blocks the shared event loop”。platform_functions 中 get_messages 的 sanitize 已经用 `run_in_executor(None, _sanitize_all)` 规避了这个问题，论坛代码没有跟进。wrapper 里的 `load_config()` 也是同步读文件。
- 影响：大型 Zendesk 页面（术语表长文、几百条评论的帖子）用 html.parser 解析需要数百毫秒，评论去重则是平方级复杂度。这段时间会阻塞整个 FastMCP 事件循环，其他客户端正在进行的模拟轮询、请求都会一起卡住。
- 建议：把解析函数改成纯函数，通过 `await loop.run_in_executor(None, parse, html)` 调用。评论去重改用 set 存 (author, body, date) 元组。或者在浏览器端用 page.eval_on_selector_all 直接抽取结构化数据，减少 Python 端的 HTML 解析。
- 复核修正：179/221/329/367 的 BeautifulSoup 解析确实在事件循环里同步执行，而 platform_functions 在 1113 行对 get_messages 已经用 run_in_executor(None, ...) 处理过同类问题，前后矛盾属实。但评论去重 `comment_data not in comments` 虽然是 O(n²)，n 通常只有几十到几百，开销可以忽略。load_config 只是读一个小 JSON，也可以忽略。单次 html.parser 解析大约阻塞几十到几百毫秒，相对整个调用动辄数秒到数十秒的浏览器耗时，对其他客户端的影响有限，定级 low。
- 修正后的建议：把 _parse_glossary_terms 以及搜索、帖子、评论的解析抽成纯函数，通过 await loop.run_in_executor(None, fn, html) 执行。去重顺手改成 set，但这不是重点。

#### FORUM-13 ·【低】评论翻页的终止依赖 Zendesk 把越界页钳到最后一页，去重键会误删合法的重复评论

- 类别：`pagination` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:353-398`
- 目录依据：n/a
- 证据：终止条件只有 404、无 `.comment`、异常，以及 `if new_comments_found_on_page == 0 and page_num > 1: break`。去重键是整条 `comment_data`（author, body, date），其中 date 取自 `.comment-meta .meta-data` 的文本，通常是相对时间。同仓库 wq-doc-forum 额外用了 `len(new_records) < 30` 作为末页判据。
- 影响：如果越界页返回的是空列表而非最后一页，循环也能终止，但总要多请求一页；如果某页因为排序变化或新评论出现条目漂移，可能漏抓或重抓。同一作者在同一相对日期发了相同内容（如 “+1”、“同问”）时，会被误去重。
- 建议：从首屏分页组件（pagination）读出总页数后按页抓取，或者以不足一页（Zendesk 默认 30 条）作为末页判据。去重改用 Zendesk comment 元素的 id 属性（例如 `id="community_comment_..."`）。

#### FORUM-14 ·【低】把全部 BRAIN 会话 cookie（domain=.worldquantbrain.com、secure=False）注入浏览器，会发送到第三方托管的 support 子域，并且依赖私有属性 cookie._rest

- 类别：`security` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:143-159；alpha-optimize/wqmcp/platform_functions.py:128`
- 目录依据：n/a
- 证据：CreddSession 的写法是 `jar.set(name, value, domain=_BRAIN_COOKIE_DOMAIN, path="/")`，其中 `_BRAIN_COOKIE_DOMAIN = ".worldquantbrain.com"`，jar.set 默认 secure=False、expires=None、_rest={'HttpOnly': None}。forum 代码原样转发这些 cookie：`'domain': cookie.domain, 'secure': cookie.secure, 'httpOnly': 'HttpOnly' in cookie._rest`。
- 影响：BRAIN 会话 token 会随每个请求发给 support.worldquantbrain.com（Zendesk 托管）。如果 SSO 是走 /access/sso 跳回 BRAIN 域完成的，Zendesk 主机其实不需要这个 cookie，属于不必要的凭据外泄面。secure=False 意味着万一跟随到 http 链接，cookie 会以明文发送。_rest 是 http.cookiejar 的私有属性，将来可能失效。
- 建议：注入时把 domain 限定为 api/platform 域（例如 `.api.worldquantbrain.com`、`platform.worldquantbrain.com`），显式设置 `secure=True`、`httpOnly=True`，SSO 交给 /access/sso 的跳转链完成。不要再读取 `_rest`。
- 复核修正：platform_functions.py:69 的 _BRAIN_COOKIE_DOMAIN 是 '.worldquantbrain.com'，jar.set 没有传 secure，所以 secure=False。forum 代码原样转发这些 cookie，support 子域会收到 BRAIN cookie，属实。但 requests 的 create_cookie 默认 rest={'HttpOnly': None}，所以 152 行算出的 httpOnly 已经是 True，“需要显式设置 httpOnly”的说法不成立。另外，/access/sso 跳转链本身需要平台域上的 cookie；在没实测 SSO 链路经过哪些主机之前，把 domain 收窄到 .api.worldquantbrain.com 可能导致 SSO 失败。
- 修正后的建议：显式设置 secure=True，并通过 cookie.has_nonstandard_attr('HttpOnly') 或直接写 True 来设置 httpOnly，不要读私有属性 _rest。domain 收窄前先实测 /access/sso 的跳转链（至少覆盖 api/platform 两个主机），确认 SSO 不受影响后再收窄。

#### FORUM-15 ·【低】死代码与冗余层：ForumClient.session 从未使用、多个无用 import、三层重复的“记日志后再抛出”、BrainApiClient 纯直通方法

- 类别：`dead-code` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:7-18, 20-27, 120-126, 184-187；alpha-optimize/wqmcp/platform_functions.py:1124-1148`
- 目录依据：n/a
- 证据：`self.session = requests.Session()`，注释写着“mainly used for the initial authentication via brain_client”，实际没有任何地方引用；`import asyncio`、`import time`、`import os`、`import requests`（只为这个无用 session 服务）均未使用；stdout 重设 utf-8 的代码块在两个模块里重复出现。ForumClient 的 `log(...ERROR); raise`，BrainApiClient 的 `self.log(f"Failed to ..."); raise`，以及 wrapper 的 `return {"error": ...}` 使同一个错误被记录三次。BrainApiClient.get_glossary_terms/search_forum_posts/read_forum_post 只做直通，还丢掉了 locale 参数。
- 影响：误导维护者，让人以为论坛存在一条独立的 requests 认证路径。日志噪声重复。直通层让参数（如 locale）无法透传，修改签名时要同时改三处。
- 建议：删除 ForumClient.session 和无用 import。MCP 工具直接调用 forum_client（或者让 BrainApiClient 只保留一处错误处理）。把 locale 等参数透传上来。

#### FORUM-16 ·【低】与同仓库 wq-rag/wq-doc-forum 的论坛抓取实现重复，而且 wqmcp 版本缺少后者已修复的若干问题

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:117-410`
- 目录依据：n/a
- 证据：wq-rag/wq-doc-forum/sync/forum/browser.py 使用完全相同的 cookie 转换代码（`"httpOnly": "HttpOnly" in c._rest`、`channel="chrome"`），并额外实现了 /access/sso；detail_sync.py 实现了未授权检测、`state="attached"`、重试和末页判据；README 说明它会把论坛同步到本地 `data/forum/index.json` 和 `data/forum/posts/{post_id}.json`。
- 影响：两套几乎相同的抓取逻辑分头维护，wqmcp 这一份没有跟上已知修复（见 FORUM-1、FORUM-3、FORUM-13）。另外本地已经有定期同步的结构化论坛语料，wqmcp 每次搜索却仍要现场起浏览器爬取。
- 建议：把 forum 浏览器和解析逻辑抽成共享包供两边复用。search_forum_posts/read_forum_post 可以优先查询 wq-doc-forum 同步好的本地 JSON（或基于它的 RAG 索引），只在需要实时数据时才回退到在线抓取。

#### FORUM-17 ·【低】与 catalog 中文档类接口职责重叠但边界不清：术语表/文章抓取对比 listTutorials/getTutorialPage，论坛搜索对比 searchPlatform（GET /search）

- 类别：`better-endpoint-available` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:2306-2331, 2333-2359（对照 969-979、1723-1733、2275-2285、2543-2548）`
- 目录依据：searchPlatform 行 19086-19169；listTutorials 行 19965-20141；getTutorialPage 行 20143 起；listVideoCourses 行 20677 起
- 证据：catalog 行 19089-19104：`GET /search`，参数有 `query`（必填）、`type`、`offset`，响应 Schema 为按类别分组的 `{count,next,previous,results}`（行 19117-19152），运行行为写的是“没有从前端确认到额外的分页、轮询或重定向规则”（19163），没有给出 type 的可选值，也没有说明是否覆盖论坛内容。catalog 行 20186-20200 说明 getTutorialPage 返回结构化 content，块类型为 `TEXT/HEADING/IMAGE/EQUATION/SIMULATION_EXAMPLE`。catalog 不覆盖 support.worldquantbrain.com，也没有术语表接口。
- 影响：Agent 查概念定义或官方用法时，很可能先调用耗时十几秒、依赖浏览器的 get_glossary_terms/search_forum_posts，而官方文档其实可以通过 get_documentations/get_documentation_page 用廉价的 JSON API 拿到（且包含 SIMULATION_EXAMPLE 示例表达式）。/search 可能提供平台内全局检索，但 catalog 中无法确认它是否包含论坛内容，因此不能断言它能替代论坛搜索。
- 建议：在工具 docstring 中划清边界：官方概念和用法优先用 get_documentations/get_documentation_page，论坛只用于社区经验。新增一个 search_platform 工具封装 GET /search（带 Accept: application/json;version=2.0，并处理 offset/next 分页），先实测 type 的取值和覆盖范围，再决定能否分流一部分论坛搜索需求。

#### FORUM-18 ·【低】论坛前置认证调用的 GET /authentication 没有带 catalog 要求的 Accept 版本头

- 类别：`missing-header` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:367`
- 目录依据：getAuthentication 行 119-126
- 证据：代码：`response = await self._request('get', f"{self.base_url}/authentication")`，整个文件中 grep 不到任何 Accept 头设置。catalog 行 122-126：`GET /authentication` - operationId：`getAuthentication` … - Accept：`application/json;version=2.0`。
- 影响：目前靠服务端默认版本工作；一旦默认版本变化，authenticate 读取的 `data.get('user')`、`token.expiry` 等字段可能失效，进而导致所有论坛工具的前置检查失败。影响较小。
- 建议：在 _build_session 中统一设置 `Accept: application/json;version=2.0`，或者按接口传入。更好的做法是按 FORUM-6 让论坛路径不再依赖 /authentication。
- 复核修正：catalog 122-126 行给出 getAuthentication 的 Accept 为 application/json;version=2.0，对 platform_functions.py grep 'Accept' 没有任何命中，缺失属实。但 authenticate（367-376）只根据 status_code==200 判断是否通过，user/token 字段只放进返回值，论坛代码根本不读这些字段。所以“默认版本一变，所有论坛工具的前置检查都会失败”的影响说法不成立。这是一个横跨整个文件的 Accept 头问题，不是论坛特有的。
- 修正后的建议：在 _build_session 中统一加上 Accept: application/json;version=2.0，作为全局修复。论坛路径按 FORUM-6 的建议不再调用 /authentication 即可。

#### FORUM-19 ·【低】locale 硬编码且各处不一致：搜索和读帖用 zh-cn，术语表用 en-us，工具层无法修改

- 类别：`wrong-params` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:174, 193, 314；alpha-optimize/wqmcp/platform_functions.py:1132-1136`
- 目录依据：n/a
- 证据：`async def search_forum_posts(self, ..., locale: str = "zh-cn")`，但 BrainApiClient 调用时是 `forum_client.search_forum_posts(email, password, search_query, max_results)`，没有传 locale；读帖是 `.../hc/zh-cn/community/posts/{post_url_or_id}`；术语表是 `.../hc/en-us/articles/4902349883927-...`。
- 影响：Zendesk 的文章搜索一般按当前 locale 过滤（对本站点未实测），在 zh-cn 下可能搜不到只有英文版的官方文章，结果不完整且不会有任何提示。用户无法切换到英文区检索。
- 建议：把 locale 作为 MCP 工具参数（默认 zh-cn 或 en-us），一路透传到 ForumClient；读帖路径也使用同一个 locale，或者直接使用搜索结果给出的完整链接。

#### FORUM-M1 ·【低】read_forum_post 接受任意 http(s) URL，没有校验主机，会让以 --no-sandbox 运行的 Chrome 访问任意地址（包括内网）

- 类别：`security` · 复核：复核补充
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:311-312, 139, 321`
- 目录依据：n/a
- 证据：`if post_url_or_id.startswith('http'): initial_url = post_url_or_id`，之后直接 `await page.goto(initial_url)`；浏览器的启动参数是 `p.chromium.launch(channel="chrome", headless=True, args=['--no-sandbox'])`。代码没有检查 URL 的 host 是否为 support.worldquantbrain.com。
- 影响：论坛帖子或评论里的链接、或者被提示注入诱导的 Agent，都可能让 MCP 服务器上关闭了沙箱的 Chrome 访问任意外网或内网地址（例如 http://127.0.0.1:8762 或其他本机服务），形成盲 SSRF，并把关闭沙箱的浏览器暴露给不受信任的页面。页面内容只有在匹配 .post-body/.article-body 时才会返回，所以直接读取数据的能力有限，但导航和发出 GET 请求本身不受限制。
- 建议：只接受 scheme 为 https、host 为 support.worldquantbrain.com 的 URL，路径限定在 /hc/ 下，其余一律拒绝。同时去掉 --no-sandbox，或者只在容器环境中启用它。


### RED · 整体冗余与工具面设计（22 条：高 3 / 中 13 / 低 6）

#### RED-1 ·【高】value_factor_trendScore 对每个 alpha 串行发一次 GET /alphas/{id}（N+1），但 listAlphas 已返回所需字段

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:825-859, 866-873`
- 目录依据：listAlphas（目录 7591 行，响应 schema 含 classifications 7826 行、pyramids/pyramidThemes 8386-8400 行）；getActivityPyramidAlphas（1917 行，返回 pyramids[].category/region/delay/alphaCount，支持 startDate/endDate）
- 证据：代码：`alphas_resp = await self.get_user_alphas(stage='OS', limit=500, ...)` 之后 `for a in regular: detail = await self.get_alpha_details(a.get('id'))`，接着只读取 `detail.get('classifications')`、`detail.get('pyramids')`、`detail.get('pyramidThemes')`。目录 listAlphas 的 results 项 schema 已含 `"classifications"`、`"pyramids"`、`"pyramidThemes"`。
- 影响：窗口内有 300 个 REGULAR alpha 时，会串行发 300 次 GET /alphas/{id}，每次都占 _executor 线程并受 60s 读超时约束，工具调用耗时可达数分钟，并在多客户端并发时抢占共享的 32 线程池。这些字段其实已在第一次列表响应中。另外 `except Exception: continue` 会让失败的 alpha 被静默丢弃，N 却仍把它计入，得分因此偏低。
- 建议：直接用 listAlphas 返回的 results 计算 _is_atom 与 pyramid 计数，删除逐个 get_alpha_details。per_pyramid_counts 也可以直接用 GET /users/self/activities/pyramid-alphas?startDate=&endDate= 的 alphaCount。limit=500 需要改为按 limit/offset 分页（目录中的默认值为 10，未记录最大值）。

#### RED-2 ·【高】get_alpha_pnl / get_alpha_yearly_stats 是 get_record_set_data 的严格子集，且 4 份重试循环是复制粘贴、均未遵守 Retry-After

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:609-671, 1150-1212, 1214-1272, 1274-1332, 1494-1504; 工具 2064-2078, 2388-2394, 2448-2454`
- 目录依据：getAlphaRecordset（12990 行，recordsetType 枚举 pnl/sharpe/turnover/daily-pnl/yearly-stats，运行行为 polling:true, retryAfterHeader:true）；getAlphaCorrelation（12716 行，polling:true, retryAfterHeader:true, handledStatuses 401/410/412/429/503）
- 证据：get_alpha_pnl 请求 `f"{self.base_url}/alphas/{alpha_id}/recordsets/pnl"`，get_alpha_yearly_stats 请求 `.../recordsets/yearly-stats`，get_record_set_data 请求 `.../recordsets/{record_set_name}`，三者是同一个端点。四个方法（pnl、yearly-stats、prod correlation、self correlation）的循环结构逐行相同：`max_retries = 5`、`if not text: ... await asyncio.sleep(retry_delay)`、`except Exception as parse_err:`，区别只在 retry_delay=2*1.5^n 或固定 20。get_record_set_data 则完全没有重试：`response.raise_for_status(); return response.json()`。
- 影响：约 240 行重复代码。同一端点的行为因调用的工具而不同：LLM 调用 get_record_set_data(alpha,'pnl') 时，recordset 若仍在计算（空 body 并带 Retry-After），会在 response.json() 处抛 JSONDecodeError；换成 get_alpha_pnl 则会重试。所有循环都忽略服务端的 Retry-After，而且对 4xx（例如 404 或权限错误）也会重试 5 次：pnl 空等约 16s，correlation 空等 80s。
- 建议：抽出一个 `_poll_json(url, budget)` 助手：按 Retry-After 休眠，只对空 body、Retry-After、5xx 和 429 重试，遇到 4xx 立即返回，并复用 check_simulation_progress 的 wait_seconds 预算模型。然后删除 get_alpha_pnl、get_alpha_yearly_stats 两个工具，保留一个 `get_alpha_recordset(alpha_id, name: Literal['pnl','sharpe','turnover','daily-pnl','yearly-stats'] | None)`，name 为空时列出 recordsets，从而合并 get_record_sets。

#### RED-3 ·【高】check_correlation 与 get_submission_check 高度重叠，都没有使用平台权威检查接口 GET /alphas/{id}/check；取值 'prod' 会被静默跳过并返回 all_passed=True

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1334-1419（尤其 1348-1361、1402、1413）, 1421-1442; 工具 2396-2410`
- 目录依据：getAlphaChecks（12551 行，响应 is.checks[] {name, result: PASS/FAIL/PENDING, limit, value} 与 is.selfCorrelation，运行行为 polling:true, retryAfterHeader:true）；getAlphaCorrelation（12716 行，correlationType 枚举 self / power-pool / prod）
- 证据：`if correlation_type == "both": check_types = ["production", "self"] else: check_types = [correlation_type]`，随后 `if check_type == "production": ... elif check_type == "self": ... else: continue`，最后 `results['all_passed'] = all_passed`（初值为 True）。get_submission_check 的实现只有 `correlation_checks = await self.check_correlation(alpha_id, correlation_type="both")` 加上 `alpha_details = await self.get_alpha_details(alpha_id)`。工具 docstring 仅有 "Comprehensive pre-submission check."（35 字符）和 "Check alpha correlation against production alphas, self alphas, or both."，没有列出合法取值。
- 影响：LLM 按 API 命名传 correlation_type="prod" 或 "power-pool" 时，循环 continue，checks 为空，结果仍为 all_passed=True，这是错误的放行信号。get_submission_check 只多了一次 get_alpha_details，并且把整份 alpha JSON 塞进结果，token 开销大。两者都用本地阈值 0.7 自行判定，没有读取平台给出的 PASS/FAIL/PENDING。prod 和 self 两个请求串行执行，最坏情况 2×(4×20s sleep + 5×60s 读超时)。名称自称“全面预提交检查”，实际缺少平台的其余检查项。
- 建议：合并为一个 `check_alpha(alpha_id, correlations: list[Literal['self','prod','power-pool']] = [])`：主体调用 GET /alphas/{id}/check（按 Retry-After 轮询）返回 is.checks，需要时用 asyncio.gather 并行拉取 /correlations/{type}。对未知 correlation_type 返回错误，不再静默 continue。删除 get_submission_check。

#### RED-4 ·【中】lookINTO_SimError_message 是 check_simulation_progress 的劣化子集，并会把运行中的模拟误报为错误

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:2728-2766 vs 255-294, 296-353, 1950-1981`
- 目录依据：getSimulation（6264 行，运行行为 polling:true, retryAfterHeader:true；响应含 status/alpha/children/message）
- 证据：lookINTO_SimError_message：`for loc in locations: resp = await brain_client._request('get', loc)` 之后 `if not data.get("alpha"): error_msg = error_msg or "Simulation did not get through, ..."`。_check_once 已经处理同一个 URL：`if "Retry-After" in resp.headers: return {"status": "RUNNING", ...}`，失败时返回 `{"status": body.get("status", "ERROR"), "message": body.get("message"), "raw": body}`，多模拟子项也会返回 `"message": b.get("message")`。
- 影响：对仍在运行的模拟（200、带 Retry-After、尚无 alpha），lookINTO 会返回 “Simulation did not get through”，LLM 可能因此误判失败并重复提交，占用账户的并发模拟槽位。它按 URL 串行请求，而 _check_multi_children 用的是 gather。两个工具用途几乎相同，LLM 难以在二者之间选择。
- 建议：删除 lookINTO_SimError_message，改为让 check_simulation_progress 接受 `progress_url: str | list[str]`，用 gather 并行调用 _check_once。_check_once 失败分支已包含 BRAIN 的 message，无需另设工具。

#### RED-5 ·【中】lookINTO_SimError_message 不校验 URL；check_simulation_progress 仅做子串校验，二者都可被用来请求内网地址（SSRF）

- 类别：`other` · 复核：部分成立（原评 高）
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:2739-2741, 1977, 1789-1794, 97`
- 目录依据：n/a
- 证据：lookINTO：`resp = await brain_client._request('get', loc)` 之后原样返回 `"raw": resp.text`，没有任何校验。check_simulation_progress 的校验只是 `if not progress_url or "worldquantbrain.com" not in str(progress_url)`，属于子串匹配。服务监听 `host="0.0.0.0"`。credd 的鉴权头是可选的：`headers = {"X-Auth-Token": CREDD_TOKEN} if CREDD_TOKEN else None`。
- 影响：任何能连到 8761 端口的 MCP 客户端都可以调用 lookINTO_SimError_message(["http://127.0.0.1:8762/cookies"])，也可以把 check_simulation_progress 的参数写成 "http://127.0.0.1:8762/cookies?x=worldquantbrain.com"。若 credd 未配置 token，响应体（BRAIN 会话 cookie）会被原样返回给调用方。即便配置了 token，这两个工具仍可用来探测内网服务。
- 建议：只接受 `https://api.worldquantbrain.com/simulations/` 前缀，或只接受 simulation_id 再在服务端拼接 URL。删除 lookINTO_SimError_message，与 RED-4 一并处理。
- 复核修正：代码层面的 SSRF 成立：2741 行对任意 loc 调用 brain_client._request('get', loc)，并在 2746 行原样返回 resp.text；1977 行只做子串检查 'worldquantbrain.com' in progress_url，类似 http://127.0.0.1:8762/cookies?x=worldquantbrain.com 的地址可以通过，_check_once 在无 alpha 时会返回 raw body（289-290 行）。CreddSession 的 session.headers 只含 User-Agent，所以这个请求不带 X-Auth-Token；只有 credd 未配置 token（97 行 CREDD_TOKEN 默认为空）时才会泄露 cookie。不过 wq-rag/brain_rag_service/README.md 236-240 行写明，部署模型是由 HTTPS 反代校验 Bearer Token（服务本身不鉴权）。能调用这些工具的客户端本来就可以提交 alpha、修改属性，完全控制该 BRAIN 账户，SSRF 带来的增量是窃取可离线复用的会话 cookie，以及探测内网。真正把攻击面放大的是 0.0.0.0 监听可以绕过反代（见 RED-M1）。因此下调为 medium。
- 修正后的建议：check_simulation_progress 只接受 simulation_id，或用 urllib.parse 严格校验 scheme=https、netloc=api.worldquantbrain.com、path 以 /simulations/ 开头。删除 lookINTO_SimError_message（与 RED-4 一并处理）。同时让 credd 强制要求 CREDD_TOKEN，服务改为监听 127.0.0.1（见 RED-M1）。

#### RED-6 ·【中】create_multi_simulation 在工具层内联拼接 settings，与 SimulationSettings / BrainApiClient.create_simulation 重复且行为不一致

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:2552-2685（尤其 2618-2649、2661-2662） vs 151-178, 422-508, 1854-1948`
- 目录依据：createSimulation（5697 行）
- 证据：create_simulation 用 `SimulationSettings(...)` 加 `settings_dict.pop(...)` 按 language/type 裁剪字段，并在 `response.raise_for_status()` 后返回。create_multi_simulation 则在工具函数内部手写 `settings: Dict[str, Any] = {'instrumentType': ..., 'maxTrade': max_trade}`，直接调用私有方法 `brain_client._request('post', ...)`，失败判断是 `if response.status_code != 201: return {"error": f"Failed to create multisimulation. Status: {response.status_code}"}`。两个工具各自复制了一段 429 → RATE_LIMITED 的处理。
- 影响：两处 payload 规则需要分别维护，已经出现差异：multi 不支持 max_position，也不支持 SUPER。multi 失败时丢弃了 BRAIN 返回的错误 body（例如表达式语法错误），LLM 只能看到状态码。两个工具的 docstring 合计约 3.6k 字符，参数大量重复（23+16 个）。
- 建议：合并为一个 `simulate(regular: str | list[str], settings...)`，或至少让 multi 复用 SimulationSettings 和一个 `BrainApiClient.submit_simulations(payload)` 方法，统一处理 429、非 2xx 和 Location 头，失败时返回 resp.text 的前 N 字符。

#### RED-7 ·【中】三个工具都返回“当前用户/登录状态”，另有两处每次调用都预查询 GET /users/self

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:355-394, 396-420, 750-757, 1582-1589, 1830-1838, 957-967`
- 目录依据：getAuthentication（119 行，响应含 user.id）；getUser（512 行，含 email/telephone，他人 ID 返回 403）；getUserProfile（960 行）；listUserCompetitions（17464 行：“使用 self 可读取当前登录用户”）
- 证据：get_leaderboard：`user_response = await self._request('get', f"{self.base_url}/users/self")` 后取 `params['user'] = user_data.get('id')`。get_user_competitions：`user_response = await self._request('get', f"{self.base_url}/users/self")` 后再请求 `/users/{user_id}/competitions`。manage_config(get)：`auth_status = await brain_client.get_authentication_status()`（GET /users/self）加 `await brain_client.is_authenticated()`（GET /authentication）。authenticate 也会调用 GET /authentication 并返回 `'user': data.get('user', {})`。get_user_profile 默认请求 `/users/self`。
- 影响：get_user_competitions 的预查询完全多余，因为目录明确说明 /users/self/competitions 可用。get_leaderboard 每次多一次往返。authenticate、manage_config(get)、get_user_profile 三个工具的用途重叠，LLM 在“检查登录”和“我是谁”这类问题上无法判断该选哪个。get_user_profile 名为 profile，实际调用 getUser，会把 email/telephone 等个人数据带进上下文；传入他人 ID 时按目录会返回 403，真正的公开资料接口 /users/{id}/profile 没有被使用。
- 建议：get_user_competitions 直接请求 /users/self/competitions。首次 GET /authentication 后缓存 user.id，供 leaderboard 使用。把 authenticate、manage_config(get)、get_user_profile(self) 合并为一个 `brain_status` 工具。如确需查看他人资料，改用 getUserProfile。

#### RED-8 ·【中】5 个工具共 10 个遗留 email/password 参数全部被忽略，并沿 5 层调用链传递

- 类别：`dead-code` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1797, 2307, 2323-2328, 2334, 2350-2357, 2362, 2378-2384, 2690, 1124-1148, 355-358; forum_functions.py:128-134, 164, 193, 304`
- 目录依据：n/a
- 证据：authenticate 的 docstring 写明 `email: Ignored (kept for backward compatibility; credentials live in credd)`，方法注释为 `email/password arguments are accepted for backward compatibility but ignored`。forum 工具：`email = email or credentials.get("email", "")` → `brain_client.get_glossary_terms(email, password)` → `forum_client.get_glossary_terms(email, password)` → `_get_browser_context(p, email, password)` → `brain_client.authenticate(email, password)`，到这里被丢弃。get_glossary_terms 的 docstring 仍写着 `email: Your BRAIN platform email address (optional if in config)`。
- 影响：每个工具 schema 都额外暴露两个无效参数。docstring 仍在引导 LLM 或用户“提供密码”，可能导致用户把明文密码发给模型，或写进 user_config.json（见 RED-9）。5 层函数签名也因此平白变长。
- 建议：从所有工具签名、BrainApiClient 的 forum 方法和 ForumClient 方法中删除 email/password。_get_browser_context 不再接收凭据。删除对 load_config()['credentials'] 的读取。

#### RED-9 ·【中】manage_config 与 load_config/save_config 已成遗留功能：唯一的消费者是被忽略的 credentials；get 分支还会原样回显配置文件

- 类别：`dead-code` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1739-1785, 1818-1850`
- 目录依据：n/a
- 证据：`_resolve_config_path` 的 docstring 写 `falls back to ~/.brain_mcp_config.json`，代码实际是 `config_path = Path(__file__).parent / "user_config.json"`。写路径失败时 `return tempfile.NamedTemporaryFile(delete=False).name`，这是每次都不同的随机文件，而读取仍走原路径。manage_config(get) 返回 `{"config": config, "auth_status": ..., "is_authenticated": ...}`。grep 结果显示 load_config 只被 manage_config 和 3 个 forum 工具使用，后者只读 `config.get("credentials", {})`，而这个值在下游被忽略。
- 影响：工具面上多一个没有实际配置语义的工具。set 分支允许任意客户端向服务器目录写入任意 JSON。get 分支会把文件内容（包括旧版本残留的明文密码）返回给任何连入的 MCP 客户端。docstring 与实现不符，回退到临时文件时写入的配置永远读不回来。is_authenticated 与 get_authentication_status 也只被 manage_config 使用。
- 建议：删除 manage_config、_resolve_config_path、load_config、save_config、is_authenticated、get_authentication_status。状态查询由 RED-7 中的 brain_status 提供。
- 复核修正：核心成立：1744 行 docstring 写回退到 ~/.brain_mcp_config.json，1750 行实际用 Path(__file__).parent/'user_config.json'。load_config 只被 manage_config 和三个 forum 工具使用（grep 命中 1831/1844/2323/2352/2379 行），后者读取的 credentials 在下游被忽略。get 分支会回显整个配置文件，set 分支允许客户端写入任意 JSON。is_authenticated 和 get_authentication_status 只被 manage_config 使用（1832、1837 行）。但“回退到临时文件时写入的配置永远读不回来”基本不可达：config_path.parent 就是脚本所在目录，必然存在，mkdir(parents=True, exist_ok=True) 对已存在目录不会抛异常；目录不可写时是后面的 open() 失败，只记一条 logger.error。
- 修正后的建议：删除 manage_config、_resolve_config_path、load_config、save_config，并删除 forum 工具中对 credentials 的读取。is_authenticated 和 get_authentication_status 并入 RED-7 的 brain_status。tempfile 分支属于死代码，一并删除即可，不必作为独立缺陷处理。

#### RED-10 ·【中】forum_functions 在运行时 `from platform_functions import brain_client`，会以新模块名重新执行整个服务器文件，生成第二个 BrainApiClient 和线程池

- 类别：`concurrency-perf` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:131, 134, 143; platform_functions.py:47, 192-195, 1735, 1789, 2770-2774`
- 目录依据：n/a
- 证据：platform_functions.py 通过 `if __name__ == "__main__": mcp.run(transport="streamable-http")` 以脚本方式启动，模块名是 __main__。forum_functions.py 在方法内部执行 `from platform_functions import brain_client`，此时 sys.modules 中没有 'platform_functions'，于是整份文件被再执行一遍，重新创建 `brain_client = BrainApiClient()`（含新的 `ThreadPoolExecutor(max_workers=32)`）和一个新的 `FastMCP(...)`，并再注册 40 个工具，只是不会 run。
- 影响：首次调用 forum 工具时会创建第二个客户端：额外占用最多 32 个线程和一个独立的 credd 会话。forum 工具刷新的是这个影子客户端的 cookie，主服务的 brain_client 不受影响。logging.basicConfig 等模块级副作用也会执行两次。这些额外开销难以察觉，也会让排障时对全局状态的判断出错。
- 建议：改为依赖注入：`ForumClient(cookie_provider)`，由 platform_functions 在构造时传入 `brain_client`。或者把 BrainApiClient 移到独立模块（例如 brain_client.py），入口文件只负责 mcp.run。

#### RED-11 ·【中】每次 forum 调用都强制刷新 credd cookie、访问 /authentication，并启动一个完整的 Chrome 进程

- 类别：`concurrency-perf` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/forum_functions.py:128-162, 166-170, 195-199, 306-316; platform_functions.py:360-367`
- 目录依据：getAuthentication（119 行）
- 证据：`_get_browser_context` 内调用 `auth_result = await brain_client.authenticate(email, password)`，authenticate 中执行 `await loop.run_in_executor(self._executor, session.refresh_cookies)`（强制从 credd 拉取）和 `await self._request('get', f"{self.base_url}/authentication")`，之后 `browser = await p.chromium.launch(channel="chrome", headless=True, ...)`。三个 forum 工具每次调用都走这条路径。
- 影响：并发的 forum 请求会各自启动一个 Chrome（每个数百 MB），并各自强制请求 credd /cookies。credd 若进入 rate_limited 或 backoff，forum 工具直接失败，同时换掉所有正在进行的 BRAIN 请求共用的 cookie jar。设计上本应由 401 自愈机制触发的刷新，在这里变成了每次调用都刷新。
- 建议：去掉 forum 路径中的 authenticate 调用，直接用现有 session 的 cookie（若过期，401 自愈即可）。在进程内复用单个 browser 实例并限制并发数，或者改由已接入的 brain-rag MCP 提供检索（见 RED-12）。
- 复核修正：forum_functions.py 134 行每次都调用 brain_client.authenticate，其中 365 行强制执行 refresh_cookies（请求 credd /cookies），367 行 GET /authentication；139 行每次启动一个 Chrome，三个 forum 工具都走这条路径，都成立。credd 处于 backoff/rate_limited 时 CreddUnavailable 会被转成异常，forum 工具随之失败，也成立。但“同时换掉所有正在进行的 BRAIN 请求共用的 cookie jar”与 RED-10 矛盾：按脚本方式启动时，forum 用的是影子 brain_client，换的是影子客户端的 jar，主服务不受影响。即便是同一个客户端，refresh_cookies 也是原子地重新绑定 jar（129 行），对进行中的请求无害。
- 修正后的建议：去掉 forum 路径中的 authenticate 调用，改为注入主 brain_client 的 cookie（配合 RED-10 的依赖注入）。进程内复用一个 browser，用 asyncio.Semaphore 限制并发，或者按 RED-12 交给 brain-rag。

#### RED-14 ·【中】36 个工具包装函数和约 30 个客户端方法共用完全相同的样板代码；ensure_authenticated 调用 33 次，均为冗余

- 类别：`redundancy` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:406-410, 249（例：537-547、1482-1504、2440-2454）`
- 目录依据：n/a
- 证据：grep 统计：`await self.ensure_authenticated()` 出现 33 次，`An unexpected error occurred` 34 次，`response.raise_for_status()` 32 次，`self.log(f"Failed to` 41 次。ensure_authenticated 的实现只有 `await self._ensure_session()`，而 _request 第一行已经是 `session = await self._ensure_session()`。典型的一对：客户端 `try: response = await self._request('get', ...); response.raise_for_status(); return response.json() except Exception as e: self.log(...); raise`，工具层 `try: return await brain_client.get_record_sets(alpha_id) except Exception as e: return {"error": f"An unexpected error occurred: {str(e)}"}`。
- 影响：大约 600 行样板代码。错误信息只包含 raise_for_status 生成的状态码和 URL，丢掉了 BRAIN 的错误 body（例如 detail、validation 信息），LLM 无法根据错误信息自我修正。返回的错误结构也不统一：`{"error"}`、`[{"error"}]`、`{"success": False}`、`"no data"` 并存。
- 建议：加一个 `_get_json(method, path, **kw)`：统一处理 4xx/5xx，返回 `{"error", "http_status", "body"}`。再写一个装饰器 `@brain_tool` 统一包装异常。删除 ensure_authenticated 及其 33 处调用。大多数“客户端方法 + 工具”对都能缩成 3 行的工具函数。

#### RED-15 ·【中】submit_alpha 返回 requests.Response.__dict__，失败时返回 False，且没有实现目录中的提交轮询

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:714-729, 2141-2158`
- 目录依据：submitAlpha（12210 行，成功状态 200/201/202/204，运行行为 poll: pollAlphaSubmission、retryAfterHeader:true）；pollAlphaSubmission（12381 行）
- 证据：`return response.__dict__`，失败分支 `except Exception as e: ... return False`，方法签名写的却是 `-> bool`。工具层 `success = await brain_client.submit_alpha(alpha_id); return {"success": success}`，其 except 分支永远不会被触发。
- 影响：__dict__ 中包含 _content（bytes）、raw（urllib3 响应对象）、connection、request（PreparedRequest）、cookies（RequestsCookieJar）等内部对象。FastMCP 序列化时要么失败，要么以 str() 回退输出一大段噪声；若响应里带 Set-Cookie，cookie 值也可能随 jar 的字符串表示返回（取决于序列化实现，属于可能情况）。提交失败时 LLM 只能看到 `{"success": false}`，没有任何原因。202 加 Retry-After 表示异步提交尚在进行，工具却不会轮询。
- 建议：返回 `{status_code, retry_after, body(json 或 text[:500])}`。遇到 Retry-After 时，与 check_simulation_progress 一样通过 GET /alphas/{id}/submit 做有预算的轮询。失败时返回 `{"error", "http_status", "body"}`。

#### RED-16 ·【中】activities 家族被拆成 5 个工具，其中 get_user_activities 的路径与参数和描述不符

- 类别：`tool-api-design` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1506-1520, 1522-1532, 1534-1575, 2456-2479, 2689-2726`
- 目录依据：listSelfActivities（1196 行，无查询参数）；getSelfActivity（1321 行，activityName 枚举 base-payment/other-payment/referrals/simulations/submissions，base-payment 需 Accept version=3.0）；getActivityDiversity（1758 行，grouping 参数）；getActivityPyramidAlphas（1917 行）；getActivityPyramidMultipliers（2027 行）
- 证据：get_user_activities 的 docstring 是 "Get user activity diversity data."，实际请求 `f"{self.base_url}/users/{user_id}/activities"` 并带上 `params['grouping'] = grouping`，而目录中 grouping 属于 /users/self/activities/diversity，/users/self/activities 本身没有参数。get_daily_and_quarterly_payment 直接调用私有方法 `brain_client._request('get', .../users/self/activities/base-payment)`，没有设置 Accept，异常被吞为 `base_payments = "no data"`。
- 影响：LLM 想查看多样性时调用 get_user_activities，得到的是分类列表而非多样性数据，并且必须先设法获得 user_id（参数必填、没有默认值）。按目录，base-payment 需要 version=3.0，当前请求可能拿到的是旧版结构，或者请求失败后只返回 "no data" 而没有原因。get_pyramid_multipliers、get_pyramid_alphas、get_daily_and_quarterly_payment、get_user_activities 都是 /users/self/activities/* 的只读查询。
- 建议：合并为 `get_activity(name: Literal['diversity','pyramid-alphas','pyramid-multipliers','base-payment','other-payment','simulations','submissions','referrals'] | None, grouping=None, start_date=None, end_date=None)`，按 name 设置 Accept 版本，name 为空时列出所有分类。

#### RED-20 ·【中】工具面开销：40 个工具、126 个参数、约 15k 字符 docstring；描述详略两极，有不存在的参数和漏写的参数，可合并为约 20 个工具

- 类别：`tool-api-design` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1796-2766（全部 @mcp.tool）；典型：2092-2131, 2571-2605, 2169, 1880-1911, 2010-2024, 2405, 2449, 2457`
- 目录依据：n/a
- 证据：用 ast 统计：40 个 @mcp.tool，126 个形参，docstring 合计 15,339 字符。最大的是 get_user_alphas（2,471）、create_multi_simulation（2,132）、create_simulation（1,666）、check_simulation_progress（1,172）；另有 22 个不足 100 字符，例如 get_user_activities 33 字符、get_submission_check 和 get_alpha_yearly_stats 各 35 字符、get_record_set_data 36 字符。value_factor_trendScore 的 docstring 写着 `- p_max: optional integer total number of pyramid categories`，函数并没有这个参数。create_simulation 的 docstring 没有说明 pasteurization、max_trade、selection_handling、selection_limit、component_activation、max_position，get_datasets 没有说明 search，get_record_set_data 没有列出合法的 record_set_name。
- 影响：加上 JSON Schema（Optional 参数会生成 anyOf/null），全部工具定义估计 25-30k 字符，约 6-8k token，每个 MCP 客户端每轮对话都要加载。详略失衡的结果是：大工具浪费 token，小工具信息不足以做选择，或会用错枚举值（correlation_type、record_set_name、grouping），不存在的 p_max 参数也会误导调用。
- 建议：建议合并后的工具集（约 20 个）：brain_status ← authenticate、manage_config、get_user_profile(self)；simulate ← create_simulation、create_multi_simulation；get_simulation ← check_simulation_progress、lookINTO_SimError_message；get_alpha ← get_alpha_details；list_alphas ← get_user_alphas；get_alpha_recordset ← get_record_sets、get_record_set_data、get_alpha_pnl、get_alpha_yearly_stats；check_alpha ← check_correlation、get_submission_check（基于 GET /alphas/{id}/check）；update_alpha ← set_alpha_properties；submit_alpha（带轮询）；get_datasets；get_datafields；get_operators；get_platform_setting_options；preview_super_selection ← run_selection；get_activity ← get_user_activities、get_pyramid_multipliers、get_pyramid_alphas、get_daily_and_quarterly_payment；get_diversity_score ← value_factor_trendScore；get_competition ← get_user_competitions、get_competition_details、get_competition_agreement、get_events；get_leaderboard；get_messages；get_documentation ← get_documentations、get_documentation_page；forum ← search_forum_posts、read_forum_post、get_glossary_terms（或交给 brain-rag）；get_alpha_performance_comparison ← performance_comparison。直接删除 expand_nested_data。枚举参数改用 Literal，让 schema 自带可选值；每个 docstring 控制在 300-600 字符，并补齐漏写的参数。
- 复核修正：用 ast 复核：40 个 @mcp.tool，126 个形参，与原文一致。按 ast.get_docstring（已去缩进）统计，docstring 合计约 14.0k 字符，不足 100 字符的有 17 个（原文为 15,339 字符、22 个，可能是统计口径不同）。get_user_activities 33、get_alpha_yearly_stats 35、get_submission_check 35、get_record_set_data 36 字符，这几个数值核实无误。2169 行 docstring 列出了不存在的 p_max 参数；create_simulation 的 docstring（1889-1906 行）漏写了 pasteurization、max_trade、selection_handling、selection_limit、component_activation、max_position；get_datasets 的 docstring（2015-2020 行）漏写了 search。这些都成立。合并建议合理，但依赖 RED-2 至 RED-16 各项先落地。
- 修正后的建议：建议本身不变。统计数字改为约 14k 字符（去缩进后）、17 个 docstring 不足 100 字符。

#### RED-M1 ·【中】服务监听 0.0.0.0 且本身不做鉴权，可以绕过文档所说的反代 Bearer 校验，放大了 RED-5 和 RED-9 的暴露面

- 类别：`security` · 复核：复核补充
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1789-1794`
- 目录依据：n/a
- 证据：代码：`mcp = FastMCP("brain-platform-mcp", "A server for interacting with the WorldQuant BRAIN platform", host="0.0.0.0", port="8761")`，没有配置任何 auth 或 token_verifier，文件中也没有出现 Authorization 或 Bearer。/home/user/wqbrain/.mcp.json 会发送 `"Authorization": "Bearer ${BRAIN_PLATFORM_MCP_TOKEN:-}"`；wq-rag/brain_rag_service/README.md 写明“服务经 HTTPS 反代暴露，并由反代校验 Bearer Token（服务本身不鉴权）”。
- 影响：服务监听所有网卡。同一网络中任何能访问 8761 端口的主机都可以绕过反代，直接调用 submit_alpha、set_alpha_properties、manage_config(set/get) 等工具，也能利用 RED-5 的 SSRF 请求 credd 或其他内网服务。反代上的 Bearer 校验因此形同虚设。
- 建议：默认改为 host="127.0.0.1"（可用环境变量覆盖），只让反代访问。或者在 FastMCP 中配置 token 校验，直接对比 BRAIN_PLATFORM_MCP_TOKEN，与 .mcp.json 发送的 Bearer 头对应。

#### RED-12 ·【低】文档和论坛类的 5 个工具与同一 .mcp.json 中的 brain-rag MCP（rag_search/rag_fetch）功能重叠

- 类别：`redundancy` · 复核：部分成立（原评 中）
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:969-979, 1723-1733, 2274-2285, 2306-2386, 2542-2548; /home/user/wqbrain/.mcp.json; wq-rag/brain_rag_service/server.py:68-146`
- 目录依据：listTutorials（19965 行）；getTutorialPage（20143 行）
- 证据：wqmcp 提供 get_documentations（GET /tutorials）、get_documentation_page（GET /tutorial-pages/{id}）、search_forum_posts、read_forum_post、get_glossary_terms（Playwright 抓取 support 站点）。.mcp.json 同时配置了 `"brain-rag"` 和 `"brain-platform"`。brain-rag 的 server.py 提供 `rag_search(query, top_k, source)`，其中 `source: 来源过滤 forum_post / forum_comment / tutorial`，另有 `rag_fetch(ref)`。此外，glossary 本身是一篇 support 文章，read_forum_post 已支持传入 http URL（`if post_url_or_id.startswith('http'): initial_url = post_url_or_id`）。
- 影响：客户端同时加载两套检索工具，LLM 需要在 search_forum_posts 与 rag_search、get_documentation_page 与 rag_fetch 之间做选择。Playwright 版本慢，每次启动 Chrome，也更脆弱（依赖 CSS 选择器）。get_documentations 和 get_documentation_page 本身也是“列表 + 详情”两个工具，可以合并为一个。
- 建议：若 brain-rag 在部署中始终可用，从 wqmcp 中移除 search_forum_posts、read_forum_post、get_glossary_terms；否则至少合并为一个 `forum(query=None, post=None)` 工具。get_documentations 与 get_documentation_page 合并为 `get_documentation(page_id: str | None = None)`。
- 复核修正：.mcp.json 确实同时配置了 brain-rag 和 brain-platform。wq-rag/brain_rag_service/server.py 的 rag_search 支持 source 取 forum_post / forum_comment / tutorial，rag_fetch 支持 post:/tutorial: 引用，功能上与 wqmcp 的 forum/tutorial 工具重叠。但 brain-rag 检索的是 wq-doc-forum 的离线爬取快照（brain_index/config.py 53-55 行的 DATA_ROOT、forum/posts、kb/tutorials），而 wqmcp 的工具读取实时内容（新帖、新评论、最新 tutorial）。两者并非等价，直接删除会丢失时效性。get_documentations/get_documentation_page 可以合并这一点成立。
- 修正后的建议：保留一条实时路径，把 search_forum_posts、read_forum_post、get_glossary_terms 合并为一个 forum 工具，并在 docstring 中写明“优先用 brain-rag 的 rag_search；需要最新帖子或评论时再用本工具”。get_documentations 与 get_documentation_page 合并为 get_documentation(page_id=None)。

#### RED-13 ·【低】expand_nested_data 是纯本地的 pandas 转换，不应放在 MCP 工具面上，而且 pandas 依赖只为它存在

- 类别：`tool-api-design` · 复核：部分成立（原评 中）
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:38, 1706-1719, 2530-2538`
- 目录依据：n/a
- 证据：`async def expand_nested_data(self, data: List[Dict[str, Any]], preserve_original: bool = True)` 的实现只是 `df = pd.json_normalize(data, sep='_')` 加 `pd.concat([original_df, df], axis=1)`，没有任何 BRAIN 请求。grep 显示 `pd` 只在 import 和这个方法中出现（共 4 处）。
- 影响：LLM 要使用它，必须把前一个工具的完整输出作为参数再发一遍，得到的结果又比输入更大（preserve_original=True 时同时包含原列和展开后的列），token 成本约为三倍。pandas 是很重的依赖，拖慢服务启动。这个 async 方法里的 CPU 计算直接在事件循环上执行。
- 建议：删除该工具和 pandas 依赖。需要展平时由客户端或 LLM 自行处理，或者在具体工具（例如 get_user_alphas）上加 `flatten: bool` 选项，在服务端的 executor 中执行。
- 复核修正：1708-1719 行只做 pd.json_normalize 和 concat，没有任何 BRAIN 请求；grep 显示 pd 只出现在 38 行和 1711-1714 行，pandas 依赖确实只为这个方法存在。它是 async 函数，却在事件循环上直接做 CPU 计算。以上都成立。但只有 LLM 主动调用时才会产生开销，“约三倍 token”也是粗估。这是工具面上的冗余，影响有限，定为 low 更合适。

#### RED-17 ·【低】get_pyramid_alphas 的两个 404 回退端点不在目录中，而主端点已实测确认；tried_endpoints 还列出了一个从未请求过的路径

- 类别：`dead-code` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1546-1569`
- 目录依据：getActivityPyramidAlphas（1917 行，响应证据：实测响应确认）
- 证据：`response = await self._request('get', f"{self.base_url}/users/self/pyramid/alphas", ...)` 与 `.../activities/pyramid-alphas` 这两个回退路径不在目录中（目录排除了 hidden 端点，因此无法断言它们不存在）。返回值 `"tried_endpoints": [..., "/pyramid/alphas"]` 中列出了代码从未请求过的 /pyramid/alphas。
- 影响：主端点既然已实测确认，回退分支基本不可达；一旦走到，会额外多出 2 次请求，并返回一份误导性的已尝试端点清单。
- 建议：删除回退分支，只保留 /users/self/activities/pyramid-alphas，404 时如实返回 http_status 和 body。

#### RED-18 ·【低】未使用的 import、函数内重复 import、ForumClient.session 未使用、get_operators 包装层的 list 分支不可达

- 类别：`dead-code` · 复核：部分成立
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:18-22, 999, 1004, 2728, 919-923, 2221-2224; forum_functions.py:7, 10, 12, 16-17, 121-126`
- 目录依据：listOperators（5064 行）
- 证据：platform_functions 中 `from bs4 import BeautifulSoup`、`from dataclasses import dataclass, asdict`、`from datetime import datetime, timedelta`、`Union` 都没有被使用（grep 仅命中 import 行）。get_messages 内部又写了 `import re, base64, pathlib` 和 `from typing import Tuple`，而 re 和 Tuple 在文件顶部已导入。第 2728 行在文件中间写 `from typing import Sequence`。forum_functions 中 `import asyncio`、`import time`、`import os`、`Optional` 没有被使用，`self.session = requests.Session()` 从未被读取。客户端 get_operators 已执行 `if isinstance(operators_data, list): return {"operators": ..., "count": ...}`，工具层的 `if isinstance(operators, list): return {"results": ...}` 因此永远不会执行。
- 影响：启动时多加载 bs4 等模块，也增加阅读负担。死分支使用的键名（results）与实际返回的键名（operators）不一致，维护者容易被误导。
- 建议：删除上述 import、ForumClient.session 以及 get_operators 工具中的 list 分支；Sequence 并入顶部的 typing import。
- 复核修正：platform_functions.py 中 BeautifulSoup、dataclass/asdict、datetime/timedelta、Union 都只出现在 import 行（datetime 另外只出现在 docstring 文本里）。999 行重复 import re，1004 行重复 from typing import Tuple，2728 行在文件中间 import Sequence。forum_functions.py 中 asyncio、time、os、Optional 只出现在 import 行，self.session 只在 123-124 行赋值、从未读取。get_operators 客户端在 920-921 行已把 list 包成 dict，工具层 2222-2223 行的 list 分支不可达。以上都成立。但“启动时多加载 bs4”不成立：forum_functions.py 14 行本身就在顶层 import bs4，而 platform_functions.py 47 行无条件导入 forum_functions，删掉 platform_functions 里的 bs4 import 并不能少加载 bs4。
- 修正后的建议：删除上述未使用的 import，删除 ForumClient.session（随之删除 forum_functions 中的 requests import），删除 get_operators 工具中的 list 分支，并把 Sequence 并入顶部的 typing import。这项清理的收益在可读性，不在启动开销。

#### RED-19 ·【低】命名不一致，容易让 LLM 选错工具

- 类别：`tool-api-design` · 复核：部分成立（原评 中）
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:2730, 2161, 2522, 2275, 2229, 2259, 2397, 2405, 1951`
- 目录依据：n/a
- 证据：`lookINTO_SimError_message` 混用大写单词和驼峰；`value_factor_trendScore` 混用 snake_case 和 camelCase，实际计算的是多样性分数（docstring 写的是 "Compute ... the diversity score"），并非 value factor；`performance_comparison` 缺少 get_ 前缀；`get_documentations` 与 `get_documentation_page` 单复数不一致；`run_selection` 实际是只读 GET 预览；`get_user_profile` 实际调用 getUser；`check_` 前缀同时用于“轮询”（check_simulation_progress）和“校验”（check_correlation），而 `get_submission_check` 又用 get_ 前缀表达校验。
- 影响：LLM 按名称的语义选工具。例如想看模拟为什么失败，会被 lookINTO_SimError_message 的名字吸引而跳过 check_simulation_progress（见 RED-4）；想看 value factor 时会调用一个其实只算多样性代理指标的工具；想做提交前检查时，三个 check 类工具都可能被选中。
- 建议：统一采用 动词_对象 的 snake_case 命名，例如 get_simulation、list_alphas、get_alpha_checks、get_diversity_score、get_alpha_performance_comparison、get_documentation、preview_super_selection。合并后按 RED-20 的清单重命名。
- 复核修正：命名问题逐一核对属实：lookINTO_SimError_message（2730 行）、value_factor_trendScore（2161 行，docstring 写的是 diversity score）、performance_comparison、get_documentations 与 get_documentation_page 单复数不一致、run_selection 实为 GET /simulations/super-selection（950 行）、get_user_profile 实为 /users/{id}，check_ 前缀同时表示轮询和校验。但这些属于可用性和可读性问题；真正会误导 LLM 的实质缺陷（lookINTO 误报、check_correlation 静默放行）已在 RED-3、RED-4 中覆盖。单独看命名，影响为 low。

#### RED-21 ·【低】get_leaderboard 硬编码 boardType=leader 且未暴露分页；lookINTO 与 check_correlation 在可以并发的地方串行执行

- 类别：`concurrency-perf` · 复核：已确认
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:743-764, 1355-1361, 2739`
- 目录依据：listConsultantBoard（18893 行，boardType 枚举 leader/spc/power-pool/referral，limit 默认 10，offset、order、aggregate、user 参数）
- 证据：`response = await self._request('get', f"{self.base_url}/consultant/boards/leader", params=params)`，params 中只有 user。check_correlation 用 `for check_type in check_types:` 串行 await 两个最长各约 80s 的轮询。lookINTO 用 `for loc in locations:` 串行请求。
- 影响：leaderboard 工具只能查 leader 榜，拿不到 spc、power-pool、referral 榜，也无法翻页或排序。correlation 的两个独立请求串行执行，最坏耗时翻倍。
- 建议：get_leaderboard 增加 board_type、limit、offset、order、aggregate 参数（使用 Literal 枚举）。独立请求改用 asyncio.gather 并发（_executor 本身已有上限）。


### CRIT · 跨组问题（completeness critic）（7 条：高 0 / 中 4 / 低 3）

#### CRIT-1 ·【中】alpha_id 直接拼进写操作的 URL，requests/urllib3 会规范化 ../，set_alpha_properties 与 submit_alpha 因此能对任意 BRAIN 路径发 PATCH/POST（MISC-7 只覆盖了 GET 工具）

- 类别：`security` · 复核：critic 补充
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1475（PATCH /alphas/{alpha_id}）、721（POST /alphas/{alpha_id}/submit）；工具入口 2413-2438、2141-2158`
- 目录依据：patchAlpha（L9330-9345，路径参数 alphaId 为 string）；submitAlpha（L12210-12225）；bulkPatchAlphas（L10268）；createCompetitionSubmission（index L94）；patchTag（index L81）
- 证据：代码：`response = await self._request('patch', f"{self.base_url}/alphas/{alpha_id}", json=data)`，以及 `response = await self._request('post', f"{self.base_url}/alphas/{alpha_id}/submit")`，两处都没有对 alpha_id 做任何校验。本地实测（requests 2.33.1 / urllib3 2.6.3）：`.../alphas/../alphas` → `https://api.worldquantbrain.com/alphas`（即 bulkPatchAlphas）；`.../alphas/../tags/T1` → `/tags/T1`（patchTag）；`.../alphas/../competitions/C1/submissions#/submit` → `/competitions/C1/submissions#/submit`，其中 fragment 不会发到服务器，实际请求的是 POST /competitions/C1/submissions（createCompetitionSubmission）。目录里 alphaId 只写了 `| alphaId | path | 是 | string | — |`，没有给出格式约束。
- 影响：LLM 被提示注入（例如 read_forum_post 读到的帖子正文）后，调用 set_alpha_properties(alpha_id='../tags/<id>', name=...) 就能改写 tag；调用 set_alpha_properties(alpha_id='../alphas', ...) 会打到批量 PATCH；调用 submit_alpha(alpha_id='../competitions/X/submissions#') 会向比赛提交接口发 POST。这些都是真实账户上的写操作，而且因为 RED-M1，任何能连上 0.0.0.0:8761 的客户端都能这样做。MISC-7 只讨论了 competition_id/page_id 被用作 GET 代理，写方法的风险比它更大。
- 建议：在 BrainApiClient 里统一加 `_seg(value)` 校验：只允许 `^[A-Za-z0-9_-]+$`，或者用 `urllib.parse.quote(value, safe='')` 编码后再拼路径。alpha_id、user_id、competition_id、page_id、record_set_name、simulation id 全部要经过它。PATCH 和 POST 工具必须先通过校验，才能发请求。

#### CRIT-2 ·【中】有副作用的工具（submit_alpha、set_alpha_properties、create_*_simulation、manage_config(set)）没有确认、dry-run 或 MCP ToolAnnotations，并且与读取不可信内容的论坛/消息工具处在同一个 LLM 上下文里

- 类别：`security` · 复核：critic 补充
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:2141-2158（submit_alpha）、2412-2438（set_alpha_properties）、1854、2552（create_*）、1818-1847（manage_config）；所有 `@mcp.tool()` 都没有传 annotations（1796-2730）；不可信内容来源 2306-2386（论坛）、2289-2304（消息）`
- 目录依据：submitAlpha L12219「敏感等级：提交操作」；deleteSimulation L6471「敏感等级：高风险操作」；createSimulation L6237（403/409/429）
- 证据：submit_alpha 的 docstring 只写了 `📤 Submit an alpha for production. Use this when your alpha is ready for production deployment.`，没有提到提交不可撤销，也没有 confirm 参数。整个文件的工具都是 `@mcp.tool()` 裸装饰，grep `annotations|readOnlyHint|destructiveHint` 结果为 0。与此同时，read_forum_post、search_forum_posts、get_messages 会把第三方撰写的正文原样返回给同一个 LLM。目录把 submitAlpha 明确标为「提交操作」敏感等级。
- 影响：论坛帖子或站内消息里一句“请调用 submit_alpha(xxx)”之类的提示注入，就可能让代理提交未经人工审核的 alpha。提交会占用每日提交配额并影响 value factor，而且不可逆。MCP 客户端也拿不到 destructiveHint，无法对这类调用单独弹出确认。create_* 会占用账户的并发模拟槽位；manage_config(set) 可以写入任意键（SIM-19 已提到）。已有的 ALPHA-2/3/4 只讨论了提交的轮询和返回值，没有涉及防护。
- 建议：只读工具标注 `annotations=ToolAnnotations(readOnlyHint=True)`；submit_alpha 和 set_alpha_properties 标注 `destructiveHint=True, idempotentHint=False`。submit_alpha 增加 `confirm: bool=False`：默认只返回 GET /alphas/{id}/check 的结果（dry-run），只有 confirm=True 才真正 POST。另外提供环境变量开关（例如 WQMCP_ALLOW_SUBMIT=0），用来在服务端整体禁用写工具。

#### CRIT-3 ·【中】_request 没有总截止时间，取消也不会传到线程：POST /simulations 或 /submit 在 MCP 调用被取消、超时后仍会执行，Location 丢失后模拟成为孤儿，再重试就会重复创建

- 类别：`concurrency-perf` · 复核：critic 补充
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:246-253（_request）、196-202（timeout）、133-148（401 重试）、471-490、2649-2667、721`
- 目录依据：createSimulation L6236-6255（201，只通过 Location 返回 simulationId）；deleteSimulation L6462（目录中有，代码未使用）
- 证据：`return await loop.run_in_executor(self._executor, lambda: func(url, **kwargs))`：asyncio 任务被取消时，executor 里的线程不会停，POST 照样发出并完成，但结果无人接收。createSimulation 的运行行为是 `"notes": "201 响应体为空；从 Location 读取 simulationId。"`，Location 一旦丢失，就再也拿不到这个模拟的 id。超时 `(10, 60)` 是 requests 的逐次读超时，不是总时长；CreddSession.request 遇到 401 会先请求 credd（`credd_timeout=15`）再重发一次，所以单次调用最坏约 10+60+15+10+60≈155s。线程池 `max_workers=32` 在满载后，排队等待时间也没有上限。
- 影响：MCP 客户端超时或用户中断时，模拟已经在 BRAIN 上运行，但代理不知道它的 progress_url，通常会再调用一次 create_simulation。这会产生重复模拟，占满每账户并发槽位，随后只能拿到 RATE_LIMITED（见 472-484），而且没有工具能找回或删除孤儿模拟（SIM-M1 只指出缺少取消工具，没有指出取消本身导致孤儿）。submit_alpha 也有同样问题：客户端看到的是超时，实际上可能已经提交成功。
- 建议：对非幂等的 POST，在线程内先把 Location 记到服务端的 recent_submissions 缓存里（按表达式+settings 的哈希索引），并提供工具让代理找回。给 _request 包一层 `asyncio.wait_for` 作为总截止时间，并返回可识别的超时错误。创建模拟前先按哈希去重。另外提供 cancel_simulation（DELETE /simulations/{id}）。

#### CRIT-5 ·【中】所有工具都捕获异常后当作正常结果返回，MCP 的 isError 永远为 false；错误形状有五六种互不一致，还会把内部 credd 地址和 biometric_url 回传给远程客户端

- 类别：`error-handling` · 复核：critic 补充
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:1815-1816、1947-1948、1998-1999 等全部工具包装层；2156（{"success": False}）；2331（[{"error"}] 列表）；477-484（{"status":"RATE_LIMITED"}）；1560-1569（{"error","tried_endpoints"}）；2711（"no data"）；2766（lookINTO 不捕获）；111-119、388-391（错误文本内容）`
- 目录依据：n/a（MCP 协议层问题；BRAIN 侧的错误状态见 createSimulation L6237、getAlphaCorrelation L12741 等）
- 证据：包装层统一写成 `except Exception as e: return {"error": f"An unexpected error occurred: {str(e)}"}`。FastMCP 只有在工具抛出异常时才会设置 isError=True，这里全部被吞掉。实际出现的失败形状包括：`{"error": ...}`、`{"success": False}`、`[{"error": ...}]`、`{"status": "RATE_LIMITED", ...}`、`{"status": "ERROR", "http_status": ...}`、`"no data"`、`{"error": "Pyramid alphas endpoint not found", ...}`。CreddUnavailable 的文本里有 `f"credd unreachable at {CREDD_URL} ..."`，以及 `f". Complete it in a browser at {body['biometric_url']} then POST {CREDD_URL}/complete-biometric"`，会通过 authenticate 等工具原样返回给远程调用方。
- 影响：MCP 客户端或编排层无法按 isError 统一处理失败，LLM 需要为每个工具猜一种失败格式，容易把 {"success": False} 或 "no data" 当成正常数据继续推理。服务监听 0.0.0.0（RED-M1），任何能连接的客户端都能拿到内网 credd 地址和账户的生物识别验证链接。已有的 DATA-10、SIM-4、ALPHA-4、AUTH-10 只分别指出单个工具丢失了错误信息，没有指出协议层 isError 语义和形状不统一这一全局问题。
- 建议：定义统一的 `BrainToolError(code, http_status, retry_after, detail)`，在包装层直接 raise，让 FastMCP 设置 isError。可恢复的状态（RATE_LIMITED、RUNNING）保留结构化成功返回，但对它们统一使用 `status` 字段。返回给客户端的错误文本要过滤 CREDD_URL 和 biometric_url，这类信息只写入服务端日志。

#### CRIT-4 ·【低】cookie jar 没有设置 secure，并且挂载了 http:// adapter：check_simulation_progress 接受 http://api.worldquantbrain.com/...，会以明文发送会话 cookie；CreddSession 用 URL 子串判断是否为 BRAIN，非 BRAIN 主机返回 401 也会触发 credd 刷新

- 类别：`auth-session` · 复核：critic 补充
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:128（jar.set 未传 secure）、229（session.mount("http://", adapter)）、140（`"worldquantbrain.com" in str(url)`）、1977（工具层子串校验）`
- 目录依据：getSimulation L6264-6279（BRAIN API 仅以 https base 使用）；n/a（安全属性）
- 证据：代码：`jar.set(name, value, domain=_BRAIN_COOKIE_DOMAIN, path="/")`（没有 secure=True），以及 `session.mount("http://", adapter)`。本地实测：同样设置的 jar 在 prepare_request('GET','http://api.worldquantbrain.com/simulations/x') 时，Cookie 头为 `t=v`，说明 cookie 会走明文 HTTP。工具层只校验 `"worldquantbrain.com" not in str(progress_url)`，http 协议可以通过。CreddSession 的判断是 `if not (resp.status_code == 401 and "worldquantbrain.com" in str(url))`，所以 `https://attacker/x?worldquantbrain.com` 返回 401 时也会调用 refresh_cookies()。
- 影响：LLM（或被注入的提示）把 progress_url 写成 http://，BRAIN 会话 cookie 就会在网络上明文传输，即使服务端随后 301 跳到 https，cookie 也已经发出去了。攻击者还可以通过 SIM-2/RED-5 所说的任意 URL，让自己的主机返回 401，反复触发 credd 拉取；credd 进入 rate_limited/backoff 后，所有工具都会失败。SIM-2/RED-5 只讨论了 SSRF 读取内网，没有覆盖这两点。
- 建议：jar.set 时加 `secure=True`，只挂载 https adapter。URL 校验改为 `urlparse(u).scheme == 'https' and urlparse(u).hostname == 'api.worldquantbrain.com'`，check_simulation_progress、lookINTO 和 CreddSession 的 401 判定都用这一个函数。

#### CRIT-6 ·【低】没有账户级的共享限流或退避：32 个工作线程加上 gather 扇出，会让多个 MCP 客户端同时向同一个 BRAIN 账户发起突发请求；一处收到 429 后，其他调用不会跟着退避

- 类别：`polling-retry` · 复核：critic 补充
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:192-195（max_workers=32）、226（pool_maxsize=32）、320（gather 所有 child）、345（gather 所有 alpha 详情）、133-148（CreddSession 只处理 401）`
- 目录依据：createSimulation L6237（429）；getSimulationSuperSelection L6543（429）；submitAlpha L12234（429）；getAlphaCorrelation L12741（429）；createAuthentication L243（429）
- 证据：`states = list(await asyncio.gather(*[child_state(u) for u in child_urls]))` 和 `full = list(await asyncio.gather(*[with_details(s) for s in states]))`：一次 multi 轮询最多同时发出 10 个 GET，完成时再发 10 个。多个客户端同时轮询时，突发量成倍增加。_request 和 CreddSession 里都没有 Semaphore、令牌桶或全局 `backoff_until`，429 只由各个调用点自行处理（有的立即返回，有的盲目重试，有的当作 ERROR）。目录里多个接口都声明了 429。
- 影响：几个代理并行研究时，同一账户容易触发 BRAIN 的 429。由于没有共享的冷却时间，其他协程会继续请求，429 越积越多，相关性和模拟状态被误判（见 SIM-3、SIM-18、ANLY-3 的具体表现）。本条指出的是它们共同的根因：缺少全局限流和退避层。
- 建议：在 _request 中加入按账户共享的 asyncio.Semaphore（例如 8）和全局 `_backoff_until`：任何响应为 429 时，按 Retry-After 设置冷却，冷却期间所有请求先等待。gather 扇出也要受这个信号量约束。

#### CRIT-7 ·【低】create_* 的请求体与目录 schema 不一致：PYTHON 会删掉目录中标为 required 的 unitHandling/nanHandling；multi 强制 2..10 条且只允许 REGULAR，而目录的数组形式是 minItems 1，并允许 SUPER

- 类别：`wrong-params` · 复核：critic 补充
- 代码位置：`alpha-optimize/wqmcp/platform_functions.py:441-444（PYTHON 去掉 unitHandling/nanHandling/testPeriod）、2633-2638、2608-2611（2..10 限制）、2641（写死 'type': 'REGULAR'）`
- 目录依据：createSimulation L5840-5853（settings.required 含 unitHandling、nanHandling）；L5963-5966（`"type": "array", "minItems": 1`）；L5969-5973（数组项 type 枚举 REGULAR/SUPER）
- 证据：代码：`if language == "PYTHON": for k in ('unitHandling', 'nanHandling', 'testPeriod'): settings_dict.pop(k, None)`；`if len(alpha_expressions) < 2: return {"error": "At least 2 alpha expressions are required"}`；`'type': 'REGULAR'`。目录：`"required": ["instrumentType", ..., "unitHandling", "nanHandling", "language", "visualization"]`；数组分支是 `"minItems": 1`，没有 maxItems，数组项 type 的 `"enum": ["REGULAR", "SUPER"]`。
- 影响：PYTHON 模拟缺少目录要求的字段，可能收到 400，而 SIM-4 已指出 400 的原因会被丢掉，LLM 看不到问题所在。不过代码注释说明这是有意为之，可能来自实测，需要验证后再下结论。multi 工具不能批量跑 SUPER，也不能提交单条数组；上限 10 在目录里没有依据，也可能与实际上限不符。SIM-9 和 SIM-15 讨论的是拼装逻辑重复和 decay/testPeriod，没有覆盖这几点。
- 建议：用 unitHandling=VERIFY、nanHandling=OFF 实测 PYTHON 请求：如果服务端接受，就保留这两个字段；如果拒绝，在代码注释里写明证据。multi 改为接受通用的 SimulationData 列表（允许 SUPER），下限改为 1。上限改成可配置，并在遇到 400 时透传服务端给出的原因。


---

## 七、被复核推翻的发现

- **SIM-16**：PYTHON 语言会删掉 unitHandling/nanHandling，而目录把这两个字段标为 required（无法核实）
  - 推翻理由：442 行注释 "PYTHON payload omits these FASTEXPR-only fields" 表明这是按已知的 PYTHON 请求体有意删掉的。目录的 required 列表来自 OPTIONS，而目录自己在 5961 行说明 "OPTIONS 将这些分支字段都标为 required，实际应按 type 选择对应分支"，可见 OPTIONS 的 required 不区分分支，不能作为 PYTHON 请求必须带 unitHandling/nanHandling 的依据。没有任何证据表明 PYTHON 请求因此被拒绝，finding 自己也承认只能算需要实测的风险。

---

## 八、当前代码的状态（main 与 v2 合并版）

v2 分支（`claude/wqmcp-review-refactor`）曾经独立重写了客户端，并对照新代码把 138 条逐条复核了一遍。那份状态表留在该分支的历史提交 8d91167 里，行号指向 v2 的 `brain_client.py` / `server.py`。

合并时以 main 的 `BrainApiClient` 为准，因为 main 上有新增功能：ProdMemo、RAA、simulationMode、表达式检查、per_alpha_settings、精简结果行。v2 的独立客户端没有保留，两套客户端并存本身就是冗余。v2 的修复已经逐项移植到 main 的客户端（提交 c198b13、4ee86ed），v2 的工具合并方案在这次合并中落地。

- **冗余（第二节、RED 组）**：工具由 48 个合并为 30 个，新旧对照见 README.md。
  - 三个建模拟工具合为 `create_simulation`，用 `type` × `mode` 区分：single / multi / concurrent × REGULAR / PYTHON / SUPER / REGION_AGNOSTIC。
  - 进度查询与错误查询合为 `get_simulation`，支持一次查多个 id。
  - 检查与相关性合为 `check_alpha`。
  - 四个 recordset 工具合为 `get_alpha_recordset`。
  - 六个账户活动工具合为 `get_activity`。
  - 比赛、文档的列表和详情各合为一个工具。
  - 删除了 `expand_nested_data`、`manage_config`，以及论坛工具里没有作用的 email / password 参数。
- **新增能力**：
  - `cancel_simulation`（DELETE /simulations/{id}）。
  - `update_alpha` 的批量 favorite / hidden / color（PATCH /alphas）。
  - `check_alpha(check="power-pool")`。
  - `list_alphas` 的 status / type 过滤。
- **提交安全**：`submit_alpha` 默认只做预检，`confirm=True` 才真正提交。`WQMCP_READ_ONLY` 和 `WQMCP_ALLOW_SUBMIT` 覆盖所有写 BRAIN 的操作；本地 ProdMemo 数据库的维护（`prodmemo_manage`）不受这两个开关控制。
- **仍需实测**（`scripts/live_regression.py`）：
  - SIM-1：super-selection 参数名，用 `--selection`。
  - SIM-17：完成时 Retry-After 的取值，用 `--writes`。
  - ALPHA-12：未记载的列表过滤，默认运行即覆盖。
  - ALPHA-13：tags 格式和 osmosisPoints，用 `--writes`。
  - MISC-4：比赛 agreement 接口，默认运行即覆盖。
  - FORUM-1：SSO 和页面选择器，用 `--forum`。
  - 合并新增：multi-simulation 能否包含 RAA 条目，用 `--probe-multi-raa`。
- 这一节没有像 v2 那样把 138 条逐条重新复核。如果需要合并版的逐条状态表，需要再跑一轮复核。

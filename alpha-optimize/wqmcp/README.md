# wqmcp：WorldQuant BRAIN MCP 服务

`platform_functions.py` 是入口，提供 30 个工具，由原来的 48 个合并而来，新旧对照见下文。登录由 credd（creds-daemon）负责，本服务从不接触 BRAIN 密码。`API_AUDIT.md` 是对照 WQ API Catalog 1.15.3 做的接口审计，下文的问题编号都指向它。

```bash
pip install -r requirements.txt             # 论坛工具还需要 `playwright install chromium`
python platform_functions.py                 # 默认 streamable-http，地址 http://127.0.0.1:8761/mcp
WQMCP_TRANSPORT=stdio python platform_functions.py
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CREDD_URL` / `CREDD_TOKEN` | `http://127.0.0.1:8762` / 空 | credd 的地址和令牌 |
| `WQMCP_HOST` / `WQMCP_PORT` | `127.0.0.1` / `8761` | 监听地址和端口。服务本身不做鉴权；要对外开放，就设成 `0.0.0.0` 并在前面加一个带鉴权的反代 |
| `WQMCP_ALLOWED_HOSTS` | 空 | 反代转发过来的 Host 名，逗号分隔（用于 DNS rebinding 保护） |
| `WQMCP_TRANSPORT` | `streamable-http` | 也可设为 `stdio` 或 `sse` |
| `WQMCP_READ_ONLY` | `0` | 设为 `1` 时，所有会写 BRAIN 的操作都直接返回错误：建模拟、取消模拟、改 alpha 属性、提交 |
| `WQMCP_ALLOW_SUBMIT` | `1` | 设为 `0` 时 `submit_alpha(confirm=True)` 报错；`confirm=False` 的预检仍可用 |
| `WQMCP_ACCEPT_VERSIONS` | `0` | 设为 `1` 时按目录发送带版本号的 Accept 头。默认关闭，因为线上验证过的是不带版本号的请求；先用 `scripts/live_regression.py --accept-versions` 实测，再决定是否打开 |
| `WQMCP_CORR_MAX_ALPHAS` | `2` | prod 通道同一时刻最多算几个 alpha，其余排队（`WQMCP_CORR_PROD_MAX` 可单独覆盖） |
| `WQMCP_CORR_SELF_MAX` | `1` | self（含 power-pool）通道同一时刻最多算几个。prod 和 self 各走各的通道，慢的 self 不占 prod 的名额 |
| `WQMCP_CORR_SELF_SLOT_SECONDS` | `600` | 一个 self 占通道这么久还没结果，就让出名额、回队列（保留原来的排队时间）；prod 一直占到出结果 |
| `WQMCP_CORR_MIN_INTERVAL` | `0.5` | 相关性和提交检查接口的请求之间至少间隔几秒 |
| `WQMCP_CORR_BACKGROUND` | `1` | 查询在后台继续排队和轮询，不随那次工具调用结束。设为 `0` 时查询随调用结束 |
| `WQMCP_CORR_QUEUE_SECONDS` | `3600` | 一个查询在后台最多存活多久，超时仍无结果就放弃 |
| `WQMCP_CORR_ROTATE` | `0` | 默认查询不会丢位置，一直占着名额等结果。设为 `1` 时，占满 `WQMCP_CORR_MAX_SLOT_SECONDS`（默认 600）还没结果就排到队尾 |
| `WQMCP_CHECK_MAX_ALPHAS` | `1` | 提交检查单独一条通道，同一时刻最多算几个 |
| `WQMCP_CORR_COOLDOWN_SECONDS` / `WQMCP_CORR_COOLDOWN_MAX` | `120` / `1800` | 判定限流后完全停止请求的时长，连续无结果时逐次加倍到上限。设为 `0` 则只降速、不停 |
| `WQMCP_SUBMIT_QUEUE` | `1` | 模拟名额满时，请求进入本服务的先进先出队列等待。设为 `0` 则直接返回 `RATE_LIMITED` |
| `WQMCP_SUBMIT_QUEUE_MAX` / `WQMCP_SUBMIT_QUEUE_SECONDS` | `50` / `7200` | 发枪队列最多排几个请求、一个请求最多等多久 |
| `WQMCP_SUBMIT_QUEUE_INTERVAL` | `15` | 名额满时每隔几秒重试队首的请求 |
| `WQMCP_MAX_WAIT_SECONDS` | `40` | 任何工具单次调用最多等多久。连接在静默约 45 秒后会断开，所以 `wait_seconds` 超过这个值会被截到这个值 |
| `WQMCP_TOOL_DEADLINE_SECONDS` | `75` | 只读工具超过这个时间没有结果就放弃并返回 `timed_out`，不会一直挂着 |
| `WQMCP_COMPACT_EXPR_CHARS` | `1500` | 精简行里表达式最多保留多少字符，`0` 表示不截断 |
| `WQMCP_CORR_AGING_SECONDS` | `300` | 从第一次提出算起，排队超过这个时间的查询排在普通查询前面（`priority="high"` 之后） |
| `WQMCP_CORR_ABANDON_SECONDS` | `1200` | 排队中的查询这么久没人再来问，就自动出队 |
| `WQMCP_RESULTS_DIR` | `<wqmcp>/results` | 带 `tag` 的模拟结果写到这里的 `<tag>.jsonl` |
| `WQMCP_WATCH_SECONDS` / `WQMCP_WATCH_INTERVAL` | `10800` / `30` | 带 `tag` 的模拟在后台最多跟踪多久、每隔几秒查一次 |
| `WQMCP_STALE_MULTI_SECONDS` / `WQMCP_STALE_SINGLE_SECONDS` | `1200` / `600` | FULL 模拟的进度这么久没动，就标为 `stale` |
| `WQMCP_STALE_QUICK_MULTI_SECONDS` / `WQMCP_STALE_QUICK_SINGLE_SECONDS` | `480` / `300` | QUICK 模拟的同一阈值 |
| `WQMCP_STATE_FILE` | `<wqmcp>/state/server_state.json` | 重启后要保留的状态：每个模拟提交了什么、tag、已写入的结果、发枪队列。设为空则只放内存 |
| `WQMCP_PROGRESS_SECONDS` | `10` | 调用进行中每隔几秒向客户端发一次进度通知 |
| `WQMCP_SLOW_CALL_SECONDS` | `50` | 超过这个时长的调用在日志里记为 WARNING |
| `WQMCP_LOOP_STALL_SECONDS` | `2` | 事件循环卡住超过这个时长，就把卡住的位置和正在等待的调用写进日志 |
| `WQMCP_AUTO_RETRY_GLITCH` | `1` | multi 的子项全部失败且没有任何原因时，把每一项单独重发一次（找出坏的那项）。设为 `0` 关闭 |
| `WQMCP_CORR_STALL_SECONDS` | `300` | 有 alpha 在算、却这么久没有任何结果出来，就判定为被限流 |
| `WQMCP_CORR_THROTTLED_INTERVAL` | `60` | 被限流期间每个查询的轮询间隔（秒） |
| `WQMCP_CORR_HOLD_SECONDS` | `90` | 只在关闭后台模式时有用：返回 PENDING 之后名额为这个 alpha 保留多久 |
| `WQMCP_CORR_CACHE_SECONDS` | `1800` | 已算出的相关性在内存里保留多久，期间重复查询直接返回（结果带 `cached: true`）。设为 `0` 关闭 |
| `BRAIN_MESSAGE_IMAGE_MODE` | `ignore` | 消息里的内嵌图片默认直接去掉。设为 `placeholder` 时会把图片写到服务端磁盘 |
| `WQMCP_BASE_URL` | `https://api.worldquantbrain.com` | 只给测试用 |

论坛相关的变量（`WQMCP_FORUM_*`、`WQMCP_GLOSSARY_TTL`）见 `forum_functions.py` 顶部。ProdMemo 的数据库配置见 `prodmemo_db.py`。

## 回测：`create_simulation`

所有回测都走这一个工具，由两个参数组合出各种方式：

- `type`：回测什么
  - `REGULAR`：普通 alpha，写在 `expressions` 里。设 `language="PYTHON"` 时，每一条是 Python 源码，同时必须给 `lookback`。
  - `SUPER`（别名 `SA`）：SuperAlpha，用 `combo` + `selection`。两者可以都是列表，按位置配对；也可以一边是单个字符串，自动配给另一边的每一项。
  - `REGION_AGNOSTIC`（别名 `RA`、`RAA`）：一个 FASTEXPR 表达式同时跑 GLB/USA/ASI/EUR，产出一个 RA_PARENT 和最多 4 个 RA_CHILD。
    - region 固定为 `ALL`，delay 固定为 1，universe 只能是 `LARGE`/`MEDIUM`/`SMALL`。
    - maxTrade 和 maxPosition 不能同时为 ON。
    - 一个 RAA 占 4 个并发名额。
- `mode`：几个一起怎么跑
  - `single`：一次只跑一个。
  - `multi`：2-10 个 REGULAR（FASTEXPR 或 PYTHON）放进一个 multi-simulation，只发一次 POST，拿到一个父 id。
    - BRAIN 在任意一个子项失败时会取消整批，所以发送前会先检查 FASTEXPR 表达式，有问题就整批拒绝。
  - `concurrent`：1-10 个独立模拟并行提交，每个有自己的 id，互不影响。
    - 每项各自给出状态：超出账户并发上限的项为 `RATE_LIMITED`，稍后原样重提即可；被 BRAIN 拒绝的项为 `ERROR`，并附带原因。
    - 整体状态：
      - 全部接受：`SUBMITTED`。
      - 部分接受：`PARTIAL`。
      - 一个都没接受，但其中有被限流的项：`RATE_LIMITED`。
      - 全部被拒：`ERROR`。
  - `auto`（默认）：一个就用 `single`；多个 REGULAR 用 `multi`；多个 SUPER/RAA 用 `concurrent`。

| | single | multi | concurrent |
|---|---|---|---|
| REGULAR · FASTEXPR | ✓ | ✓（表达式检查不过则整批拒绝） | ✓（检查结果只作为 `lint_warnings`） |
| REGULAR · PYTHON | ✓ | ✓ | ✓ |
| SUPER (SA) | ✓ | ✗ 返回错误，建议改用 concurrent | ✓ |
| REGION_AGNOSTIC (RA/RAA) | ✓ | ✗ 同上 | ✓（每个占 4 个名额） |

multi 只收 REGULAR，因为线上验证过的只有这种。BRAIN 是否接受在 multi 里放 RAA 条目，可以用 `scripts/live_regression.py --probe-multi-raa` 实测：探测请求若被接受，会立即取消。

没有传的设置按 type 取默认值：
- REGULAR/SUPER：USA / TOP3000 / delay 1 / decay 0 / NONE / truncation 0 / visualization 开
- RAA：MEDIUM / decay 10 / SLOW_AND_FAST / truncation 0.08 / visualization 关

`settings_used` 回显基础设置，即未被 `per_alpha_settings` 覆盖时实际发出的设置；各项的覆盖列在 `children` 里。PYTHON 只适用于 REGULAR。`simulation_mode="QUICK"` 只算核心指标，会强制关闭 visualization，这样跑出的 alpha 不能直接提交。

`per_alpha_settings[i]` 覆盖第 i 项的设置。只给一个表达式（或一对 combo/selection）时，会按条目重复这个表达式，一次调用就能扫参：

```python
create_simulation(expressions="rank(x)", per_alpha_settings=[{"decay": 3}, {"decay": 5}, {"neutralization": "MARKET"}])   # multi
create_simulation(expressions="rank(x)", type="RAA", per_alpha_settings=[{"universe": "LARGE"}, {"universe": "SMALL"}])  # concurrent
create_simulation(type="SA", combo="combo_expr", selection=["sel_a", "sel_b"])                                           # concurrent
```

提交后用 `get_simulation(simulation_ids=[...], wait_seconds=60)` 查询：
- 可以一次传多个 id，适合 concurrent 模式。
- 不传 id 时列出本服务最近创建的模拟。
- 模拟失败时直接返回 BRAIN 的错误信息，原来的 `lookINTO_SimError_message` 因此不再需要。
- multi 里有子项失败时状态为 `FINISHED_WITH_ERRORS`：BRAIN 会因此取消其余子项。
- RAA 完成后返回父 id，以及每个区域子 alpha 的一行指标。

`cancel_simulation` 可以取消排队中或运行中的模拟，释放名额。

## 工具一览（31 个）

| 分组 | 工具 |
|---|---|
| 账户 | `brain_status` |
| 回测 | `create_simulation` · `get_simulation` · `cancel_simulation` · `get_platform_setting_options` · `preview_super_selection` |
| Alpha | `list_alphas` · `get_alpha` · `get_alpha_recordset` · `check_alpha` · `submit_alpha` · `update_alpha` · `get_alpha_performance` · `compare_alphas` |
| 数据 | `get_datasets` · `get_datafields` · `get_operators` |
| 账户活动 / 社区 | `get_activity` · `get_leaderboard` · `get_competitions` · `get_events` · `get_messages` · `get_documentation` |
| 论坛 | `search_forum_posts` · `read_forum_post` · `get_glossary_terms` |
| ProdMemo | `prodmemo_sync` · `prodmemo_check` · `prodmemo_get` · `prodmemo_stats` · `prodmemo_manage` |

每个工具都带 MCP 注解：
- 是否只读、是否破坏性（如 `submit_alpha`、`cancel_simulation`、`prodmemo_manage`）、是否访问 BRAIN。
- `check_alpha` 等把测得的相关性顺手写进本地 ProdMemo 缓存，这不计为写操作。
- `prodmemo_sync` 和 `prodmemo_check` 会读取 BRAIN，并写入本地数据库。

执行中出错时返回 `{"error": ...}` 字典，而不是 MCP 错误，因为 `scripts/prodmemo_daily_sync.py` 等调用方依赖这种格式。参数类型不对时，MCP 框架在调用工具之前就会拒绝，这类情况仍返回 MCP 错误。

## 旧工具 → 新工具

| 旧工具（main） | 新工具 |
|---|---|
| `authenticate`、`manage_config` | `brain_status(refresh=...)`。manage_config 读写的明文配置文件已不再使用 |
| `create_simulation`、`create_multi_simulation`、`create_raa_simulation` | `create_simulation`（`type` × `mode`，见上文） |
| `check_simulation_progress`、`lookINTO_SimError_message` | `get_simulation` |
| `run_selection` | `preview_super_selection` |
| `get_user_alphas` | `list_alphas`（新增 status / alpha_type 过滤、`next_offset`，默认返回精简行） |
| `get_alpha_details`、`get_raa_alpha` | `get_alpha`（RA_PARENT 自动展开子 alpha；`full=True` 返回原始对象） |
| `get_alpha_pnl`、`get_alpha_yearly_stats`、`get_record_sets`、`get_record_set_data` | `get_alpha_recordset(recordset="pnl" / "yearly-stats" / …)` |
| `get_submission_check`、`check_correlation` | `check_alpha(check="submission" / "prod" / "self" / "power-pool" / "correlation" / "all")` |
| `submit_alpha` | `submit_alpha`，**必须带 `confirm=True` 才真正提交**；默认只做预检 |
| `set_alpha_properties` | `update_alpha`（一次可改多个 alpha 的 favorite / hidden / color） |
| `performance_comparison` | `get_alpha_performance` |
| `get_user_activities`、`get_pyramid_alphas`、`get_pyramid_multipliers`、`get_daily_and_quarterly_payment`、`value_factor_trendScore`、`get_user_profile` | `get_activity(kind="diversity" / "pyramid-alphas" / "pyramid-multipliers" / "payments" / "diversity-score" / "profile")` |
| `get_user_competitions`、`get_competition_details`、`get_competition_agreement` | `get_competitions(competition_id=..., include_agreement=...)` |
| `get_documentations`、`get_documentation_page` | `get_documentation(page_id=...)` |
| `prodmemo_sync_status` | `prodmemo_sync(mode="status")` |
| `expand_nested_data` | 删除（纯数据变换，调用方自己就能做） |
| `get_datasets`、`get_datafields`、`get_operators`、`get_events`、`get_leaderboard`、`get_messages`、`get_platform_setting_options`、论坛 3 个工具、其余 ProdMemo 工具 | 保留原名 |

其他行为变化：
- 论坛工具去掉了没有作用的 email / password 参数。`get_glossary_terms` 改为返回 `{"terms": [...], "count": n}`。
- `get_operators` 默认返回精简字段，可按 category / name 过滤；需要完整对象时传 `detail=True`。
- 错误信息不再带 "An unexpected error occurred" 前缀。

ProdMemo 仍然直接使用客户端方法（`get_user_alphas`、`get_alpha_pnl`、`get_alpha_details`、`get_production_correlation` 等），这些方法及其约定都没有改变。

## 在 main 基础上的修复（沿用）

**安全**
- 所有拼进 URL 的 id 参数都先校验再编码，`../` 之类会被拒绝。
- `get_simulation` 和 `cancel_simulation` 只接受 BRAIN 域名下的 `/simulations/<id>`，也可以直接传模拟 id。
- 服务默认只监听 127.0.0.1。论坛工具只访问 support 站点本身的地址。
- credd 处于退避期时，并发的 401 不会轮流去打 credd。

**提交**
- `submit_alpha(confirm=True)` 提交后按 Retry-After 一直跟踪到最终的检查报告。
- 结果状态为 SUBMITTED、SUBMITTED_WITH_PENDING_CHECKS、REJECTED、PENDING、RATE_LIMITED、ERROR 之一。
- 同一个 alpha 并发调用只发一次 POST；PENDING 之后再调用只会继续轮询。

**轮询与结果**
- 下列查询按 Retry-After 轮询：recordset、PnL、yearly-stats、before-and-after、检查、相关性。
  - PnL 和 yearly-stats 的客户端方法仍按 ProdMemo 的约定：返回 `{}` 表示还在算，出错抛异常。
  - 工具层在超时仍未算完时返回 `status: PENDING`。
- multi 模拟的父任务还带 Retry-After 时，不逐个查子任务。
- 子任务被限流（429）或服务端出错（5xx）时算作"仍在运行"，不会把整批报成 COMPLETE。

**相关性防限流**

同时让 BRAIN 算多个 alpha 的 prod / self 相关性，会让整个账户被限流。BRAIN 限流时不一定返回 429：更常见的是对每个 alpha 都一直回"空内容 + `Retry-After: 1`"，请求越多越不出结果。所有连到本服务的客户端共用一个队列：

- **排队**：同一时刻只让 `WQMCP_CORR_MAX_ALPHAS` 个 alpha 在算，其余按先后顺序等。排队中的 alpha 不会向 BRAIN 发任何请求。
- **后台慢慢等**：查询提交后在后台继续排队和轮询，和那次工具调用是否还在等无关。
  - 调用超时先返回 `PENDING`，稍后用同样的参数再调一次即可。
  - 再调时的回答是三种之一：已有结果（`cached: true`）、正在算（`computing_for_seconds`、`polls`）、还在排队（`queued: true`、`queue_position`）。
  - 算出的 prod / self 会自动写回 ProdMemo，不需要有人等着。
- **看队列**：`check_alpha(check="queue")` 不访问 BRAIN，只返回队列现状。每个 PENDING 回答里也带同样的 `queue`，`brain_status` 里是 `correlation_queue`。
  - `computing`：正在算的 alpha、查的是哪几项、已经算了多久、轮询了几次。
  - `waiting`：排队的 alpha，按顺序列出位置和已等时长。
  - `throttled`：当前是否判定为被限流；`max_at_a_time`：当前允许同时算几个。
- **取消**：`check_alpha(check="cancel", alpha_id=...)`，返回被取消的条目和取消后的队列。
  - `alpha_id` 填某个 alpha：取消它，不管它在排队还是正在算。
  - 填 `"waiting"`：取消所有还没开始算的。
  - 填 `"all"`：全部取消。
  - 取消只是让本服务停止排队和轮询。BRAIN 已经开始的计算不会因此停止，只是结果不再被取走。
  - 正在等这个查询的调用会收到 `status: CANCELLED`。取消之后可以重新提交。
- **识别限流并冷却**：有 alpha 在算却连续 `WQMCP_CORR_STALL_SECONDS` 秒（默认 300，平台算一条相关性常常就要几分钟）没有任何结果，或者收到 429，就判定为被限流。限流时继续请求只会让 BRAIN 更慢，所以：
  - 冷却：完全停止相关性请求 5 分钟，队列原样保留，被打断的 alpha 排在最前面。提交检查不受冷却影响（见下）。
  - 试探：冷却结束后只放 1 个 alpha，每 `WQMCP_CORR_THROTTLED_INTERVAL` 秒查一次。
  - 有结果就恢复正常速度；`WQMCP_CORR_STALL_SECONDS` 秒内还没结果就再冷却，时长加倍（5、10、20、30 分钟）。
  - 冷却期间的相关性调用照样等满 `wait_seconds`：冷却在这期间结束、结果又算出来了，就直接返回结果；否则返回 `cooling_down` 和 `resumes_in_seconds`，并附上 `local_estimate`（ProdMemo 的本地估计，仅供初筛）。冷却的时间不计入查询的存活时间。
  - 冷却从 2 分钟起步，连续无结果时加倍（2、4、8、16、30 分钟）。
  - 手动控制：`check_alpha(check="cooldown", wait_seconds=600)` 立即开始冷却，`check_alpha(check="resume")` 立即恢复。
- **不丢位置**：查询一直占着名额等结果，不会因为等得久被挪到队尾。
- **排队顺序**：被冷却打断的 → `priority="high"` 的 → 只查 PROD 的、或已等满 5 分钟的 → 其余；同一档内先来先算。
- **自动出队**：排队中的查询 20 分钟没人再来问，就自动出队，不用手动取消。
- **部分结果**：prod 已经算完、self 还在算时，状态是 `PARTIAL`，已完成的部分照常返回。
- **提交检查单独排队，且不受冷却影响**：`/check` 里的 IS 检查（Sharpe、fitness、子池、近 2 年、cluster 等）不需要算相关性，所以冷却期间照常请求 BRAIN。
  - 平台还在算相关性时，先把已有的 IS 结果返回，`is_passed` 是不含相关性的结论。
  - PROD 是 ERROR 或 PENDING 时，`prod_fallback` 依次取：本服务缓存的测量值、ProdMemo 里存的平台值、prod 相关性接口，并注明来源；`all_passed_with_fallback` 是把它算进去的结论。都没有时附 `local_estimate`。
- **预计等待时间**：有查询算完之后，排队中的回答带 `estimated_wait_seconds`，正在算的带 `estimated_remaining_seconds`，按最近查询的耗时中位数估算。被限流时这个估计不可靠。
- 正常情况下轮询间隔从 1 秒逐步拉长到 15 秒；同一个 alpha 的相同查询共用一个任务。
- `check_alpha(check="submission")` 随调用结束，不在后台继续。

**发枪排队**

账户的模拟名额满时，`create_simulation` 不再返回 `RATE_LIMITED`，而是把请求放进本服务的队列：
- 回答是 `status: QUEUED`，带 `queue_id`（如 `Q7`）和 `queue_position`。
- 队列先进先出：队首的请求被 BRAIN 接受之前，后面的不会发出；队列非空时新请求直接排到队尾。
- `queue_id` 可以当模拟 id 用：`get_simulation("Q7")` 在排队时返回位置和已等时长，发出后返回模拟的进度和结果；`cancel_simulation("Q7")` 把它从队列里取出。
- 不带参数的 `get_simulation()` 列出队列里的全部请求。
- `mode="concurrent"` 时，被接受的项是 `SUBMITTED`，没名额的项是 `QUEUED`。
- `queue=False` 保持原来的行为。队列只在内存里，服务重启后丢失。

**调用的可观测性**
- 每次工具调用在日志里记一行：编号、工具、参数摘要、结果（ok / error / deadline / cancelled）和耗时；超过 50 秒记为 WARNING。
- 事件循环卡住超过 2 秒时，把卡住的位置和正在等待的调用写进日志。
- 调用进行中每 10 秒向客户端发一次进度通知（客户端要求时），避免客户端因"长时间无响应"放弃调用。
- `brain_status` 的 `server` 给出运行的版本（`release`）、启动时间和工具列表指纹（`tools_hash`）。客户端的工具列表缺少文档里写的参数时，是客户端还拿着旧版本的工具列表，重连 MCP 即可。
- 发布重启时，正在处理中的调用会被断开，客户端要等到自己的空闲超时才会报错。重启后的服务从状态文件恢复发枪队列、运行时长、tag 和结果记录。

**结果记到服务器上的文件**

`create_simulation(..., tag="G-r3", labels=[...])`：
- 每个完成的结果行（序号、label、id、指标、fails、warns、完整设置、完整表达式）追加到服务器的 `results/G-r3.jsonl`。
- 服务在后台跟踪带 tag 的模拟，没人轮询也会写入。
- 取回时不经过对话、不耗 token：`curl -s http://<MCP 地址>:<端口>/results/G-r3.jsonl >> results.jsonl`。`?since=<行数>` 跳过已取的行，`?format=tsv` 每行一个 alpha。

**模拟状态**
- 运行中返回 `running_seconds`，不再返回没有意义的 `progress`（BRAIN 只给 0.1、0.15、0.35 这几档）。
- 进度长时间没动时标 `stale: true` 和 `stalled_seconds`，建议取消后重发。阈值：FULL multi 20 分钟、FULL single 10 分钟、QUICK multi 8 分钟、QUICK single 5 分钟。平台不告诉是哪个子项慢，提示里建议拆成 concurrent 重发来定位。
- 取消一个已经结束的模拟时，返回 `already_complete: true` 和它的结果，不再报 400。
- 完成结果里的 `diagnostics`：
  - `no_effect`：同一表达式只差一个设置、结果却完全相同，说明这个设置没有生效，并给出可能原因（decay：信号在无数据日是 NaN 且 `nanHandling=OFF`，或 FAST 中性化）。
  - `decay_not_applied`：decay ≥ 20 但换手仍大于 1。
  - `concentrated_weight`：平台不给这项的数值，集中给出一次诊断建议。
- `create_simulation(resubmit="<模拟 id>")`：把某次提交原样再发一次（设置、tag、label 都相同）。
- multi 的子项全部失败且都没有原因时，不再原样重发整批（坏字段会让它再失败一次），而是每一项单独发一次（经过发枪队列）。原 id 返回 `RETRIED` 和 `retried_as`（各项的新 id 或队列号），之后查原 id 按原顺序合并各项结果。单独也失败且没有原因的项带 `diagnosis`：列出表达式里的字段，提示用 `get_datafields` 核对该 region / universe / delay 下是否真有数据。带 `tag` 时各项仍按原序号写入日志。
- BRAIN 对已跑完的模拟返回 404 时，按表达式和设置从 alpha 列表里找回结果（`recovered: true`）。
- `get_simulation(format="tsv")`：结果压成一段 TSV，是最短的答复。多个 id 共用一个等待预算；等待被截到 40 秒时答复里有 `wait_capped_to`。

**表达式参数的编码**

有些客户端会把"字符串或列表"类型的参数多编码一层：表达式变成带引号的 `"rank(x)"`（平台会把它当成字符串常量，所有子项报 `found None`），或者整个列表变成一个字符串。服务端会还原，并在 `warnings` 里说明。以引号开头的表达式也会被预检拦下。

**多枪结果**
- 每行带 `index`，即它是请求里的第几项。
- BRAIN 用一个已存在的 alpha 作答时（表达式被它视为相同，例如字段别名），该行带 `reused_alpha: true` 和 `submitted_expr`（提交时的原始表达式）；这时 `id` 和 `expr` 是旧 alpha 的。
- 两项得到同一个 alpha 时，后一行带 `duplicate_of_index`。缺少子项时有 `missing_children`。
- 有子项失败时，`errors[]` 列出失败项的序号、BRAIN 的原因和出错位置；没有原因的子项是被连带取消的。

**发枪前检查**

按 BRAIN 自己的算子列表和字段目录检查（都会缓存）。multi 模式下有问题就整批不发；single / concurrent 只警告：
- 括号不配对；带默认值的参数用了位置写法（如 `hump(x, 0.01)`）。`ts_backfill(x, 10)` 是合法写法，实测平台接受。
- 算子不存在（如 `vec_median`），会给出名字相近的算子。
- 数据字段不存在，或在该项的 region / delay 下没有数据。字段详情最多列 50 个地区组合，列满时不下结论。
- VECTOR 字段没有经过 `vec_*` 聚合就用了（平台报 "does not support event inputs"）。字段类型以平台为准。
- 需要分组的参数给了数值（如 `densify(rank(x))`、`group_rank(x, rank(y))`）。
- `max_ops`：表达式的算子数超过它时在 `warnings` 里提示，按 BRAIN 的 `operatorCount` 口径计算（在 300 条真实 alpha 上与平台一致）。

在本账户 765 条跑成功的 alpha 上检查，零误报。

- multi：有问题就整批不发。single / concurrent：有问题的项不发，列在 `refused` 里，其余照发。`force=True` 全部照发。
- `ops_est` 给出每一项按平台口径估算的算子数；设了 `max_ops` 时，超出的项同样不发。
- 对以 `_min/_max/_mean/_avg/_median` 结尾的字段做 `equal(x, 常数)` 或 `x == 常数` 时，在 `warnings` 里提示信号很可能是常量（IND 轮这类比较 16 枪里 15 枪是常量；`*_std` 字段 158 枪没有一枪是常量，不提示）。只提示，不拦截。

**精简行**
- `fails`：数值没达到门槛的检查项，写成 `SHARPE 1.4<1.58`，不管 BRAIN 判的是 FAIL 还是 WARNING（BRAIN 在同一批里对同类未达标项判得不一致）。BRAIN 没判 FAIL 的带标记，如 `(W)` 表示它只判了 WARNING。没有 LOW_/HIGH_ 前缀但带数值的 FAIL 项（如 IS_LADDER_SHARPE）同样写出数值：`IS_LADDER 1.2<1.58`；BRAIN 没给数值的写 `名字 (no value)`。
- QUICK 行没有 CLUSTER_TEST：给 `cluster_est` = 0.76 × Sharpe（186 条 FULL alpha 上 CLUSTER / Sharpe 的中位数，p10 0.65、p90 0.94），估计值可能低于门槛时给 `cluster_risk`（`likely` / `possible`），好在跑 FULL 前就发现。
- CONCENTRATED_WEIGHT 失败的行带 `long_short`（多 / 空持仓数），作为平台不给数值时的代理指标。
- TSV 里 `set` 的键顺序固定（universe、decay、neutralization、truncation、…），逐项覆盖不会打乱顺序。
- `warns`：其余的 WARNING 项（如 CLUSTER_TEST）。
- 没有值的键不输出（如 FULL 模式下的 `robust_sharpe`）。
- 同一批子项共同的设置只在父级 `set` 出现一次，各行只列不同的项；`nanHandling`、`pasteurization`、`maxPosition`、`unitHandling` 与默认不同时也会列出。`set` 取自平台实际使用的设置。
- 平台实际使用的设置与提交的不同时，行里有 `settings_mismatch`。
- 同一表达式、同一设置之前跑过 QUICK 时，FULL 的行带 `same_as_quick`（那次 QUICK 的 alpha id）。
- 从未持仓的空信号（换手和 Sharpe 都为 0）标 `empty_signal: true`，这时 fails 没有参考价值，需要检查表达式本身（如 `equal(x, 0)` 而字段从不为 0）。
- 查询多个 id 时，本客户端已经完整拿到过的完成结果只给一行摘要（`returned_before: true`）；`only_pending=True` 让所有已完成的都只给摘要。单独查某个 id 时总是给完整结果。
- 同一个模拟的并发查询合并成一次。
- FULL multi 运行中，平台只返回一个进度数，不列子项，所以没有子项完成数。
- 被连带取消的子项压成 `cancelled: [序号]`。
- 说明文字每次调用只出现一次，不在每行重复。

**提交检查的结论**
- 账户级的限制（如 `REGULAR_SUBMISSION`：当日提交额度用完）单独放在 `account_blockers`，不计入 `failed` 和 `is_passed`。
- 同名同值的重复检查行只保留一行。
- `SELF` / `PROD_CORRELATION` 是平台在提交检查内部计算的，不经过本服务的相关性队列，所以在队列里看不到它们。
- 有检查项是 ERROR 时，`all_passed` 为 null，不给结论。
- PROD_CORRELATION 是 ERROR 时，备用端点的值放在 `prod_fallback` 里并注明来源，不计入 `all_passed`。
- `prod_source` 说明 `prod_correlation` 来自提交检查还是备用端点。

**QUICK 模式忽略 maxTrade**

实测 QUICK 下 `max_trade="ON"` 和 `"OFF"` 的结果逐位相同，而 FULL 下差别很大（Sharpe 6.39 → 2.33）。这样的项会出现在 `create_simulation` 返回的 `warnings` 里。

**其他**
- `compare_alphas(alpha_ids=[...])`：用各自的 PnL 在本地算 2–10 条 alpha 之间的相关性，未提交的 alpha 也可以，不消耗相关性请求。相关性 ≥ 0.999 的一对标 `identical`，之后其中一条测出的 prod / self 直接给另一条用（`reused_from`），不再排队。
- `prodmemo_check` 的 `prod_est`：
  - 按区域拟合（该区域标定点不少于 8 个时），否则用全部区域并注明。例如 EUR 的 PROD 平均比全局公式高 0.09，而且 pool 在 EUR 几乎预测不了 PROD。
  - 带 `range`（90% 区间）和标定点数；标定点不足时不给单点值。
  - 本地池里没有接近的 alpha、pool 在该区域没有预测力、或区间太宽时，`confidence` 为 `low`。
  - 斜率低于 0.1（`PRODMEMO_EST_MIN_SLOPE`）时不给单点值，只给区间和 `no_point_estimate` 说明：EUR 的斜率只有 0.01~0.03，所有 alpha 都会被估成约 .71，实测却在 .56~.77。
- `get_datasets` 每页 20 条、默认只返回关键字段，`detail=True` 返回完整对象，`category` 按类目过滤。
- `get_datafields` 默认精简（id、type、coverage、dateCoverage、userCount、alphaCount、描述前 80 字），每页最多 500 条（内部按 BRAIN 的上限 50 分页取）。
- prod 相关性平台返回的是分布直方图（每 0.1 一档有多少条生产 alpha），不是逐条 alpha：返回 0.5 以上的非空档 `histogram_top` 和超过阈值的条数 `alphas_over_threshold`。self 相关性照旧给最相关的 3 条。
- `get_alpha_recordset(alpha_id, "yearly-stats")` 是这条 alpha 自己的逐年数据；`get_alpha_performance` 是加入前后的组合数据。
- `list_alphas` 和 `get_activity(kind="diversity-score")` 的日期可以只写 `2026-06-29`。BRAIN 只接受带时区的完整时间，工具会补成当天 00:00:00 或 23:59:59（UTC）。
- `check_alpha` 查 QUICK 模式的 alpha 时，会说明 BRAIN 不支持检查这类 alpha，需要用 FULL 模式重跑。
- `update_alpha(name="")` 清空名字。BRAIN 不接受空字符串，工具改发 null。

**接口纠错**
- diversity 改用 `/activities/diversity`。
- 查看他人档案改用公开 `/profile` 接口。
- before-and-after 改用目录记载的接口。
- `get_activity(kind="pyramid-alphas")` 去掉了没有依据的回退路径。
- diversity-score 分页读取，不再逐个请求 alpha。
- payments 出错时两部分各自报原因。

## 2026-09-30 实战问题的修复（EUR 第四轮）

- **调用按时返回**：只读工具的每次调用有自己的时限（`wait_seconds + 20` 秒，最少 30，最多 `WQMCP_TOOL_DEADLINE_SECONDS`），HTTP 超时、重试等待、限流冷却都不会超过它。查模拟时 BRAIN 慢，就返回上一次看到的状态（`last_known: true`、`seen_seconds_ago`），不再挂到 75 秒报 UNKNOWN。服务端日志显示 319 秒无响应那次服务端按时回了，是客户端连接断开（与 ECONNRESET 一致）。
- **prod / self 分通道**：见上面的环境变量。`check_alpha(check="correlation")` 先把 self 放进队列，然后等 prod：prod 一出就返回，self 在后台继续。
- **队列顺序**：`priority="high"` → 冷却后回来试探的那一个 → 排队超过 5 分钟的（从第一次提出算，回队列不重新计时）→ 先来先到。队列里每行带 `rank_reason`、`lane`、`client`、`estimated_wait_seconds`；`lanes` 给各通道的上限和占用。
- **谁在占槽**：每个 MCP 会话自动得到 `c1`、`c2`… 的名字，也可以 `check_alpha(client="J")` 自己起名，队列里显示出来。
- **端点健康度**：队列快照里的 `endpoint_health`：`ok` / `slow` / `down`，最近几次 prod 的耗时、距上次出结果多久、按 QUICK / FULL 分开的耗时中位数。`down` 时建议先用 `compare_alphas` 在本地排除同轴。
- **提交检查**：PROD 是 ERROR 时，时间够就隔 3 秒再查一次 `/check`。prod 已在后台队列里时也取它的结果作 `prod_fallback`。回答里有单一结论 `verdict`（pass / fail / unknown）和 `verdict_basis`；有账户限制时加 `submittable_now: false`。POWER_POOL_* 检查放进 `power_pool_checks`，不影响常规 alpha 的结论。
- **整批无原因失败**：改为逐项单独重发（见“模拟状态”）。
- **说明截断**：客户端会截断过长的工具说明，`tag` 等参数原本在 4000 字之后。`create_simulation`、`check_alpha`、`get_simulation` 的说明压缩到约 2000 字，关键参数放在最前。没带 tag 的 multi 完成时，每个会话提示一次 `tag` 的用法。
- **算子数**：负数字面量的符号（`subtract(-1, x)`、`if_else(c, 1, -1)`、`x = -0.5`）不计为算子，与平台 `operatorCount` 一致。
- 修掉的隐患：轮询退避 `1.6 ** polls` 在轮询约 1500 次后溢出，后台任务会崩；取消一个 alpha 时会连带清掉它在另一通道的名额。

## 2026-09-30b 实战问题的修复（EUR 第五轮）

- **labels 与预检**：`labels` 按请求的条数校验（在预检拦下某项之前），被拦下的项不再让整批一条都不发；各项仍按原序号取自己的 label。只给一个 label 时用于整批。
- **旧 alpha / 去重落盘**：single（包括 concurrent 的每一项）也检查 BRAIN 是否用已有 alpha 作答，和 multi 一样给出 `reused_alpha`、`submitted_expr`、`warning`，jsonl 里都有。TSV 末尾新增两列 `reused`（`old_alpha` 或 `dup_of=<序号>`）和 `submitted_expr`；原有列的位置不变，按位置解析的脚本不用改。
- **谁的模拟**：`create_simulation` / `get_simulation` 都有 `client` 参数，默认是会话自动分到的名字（`c1`、`c2`…）。`get_simulation()` 列表每行带 `tag` 和 `client`，可以用 `tag=` / `client=` 过滤，保留最近 300 条。排队中的提交也显示 tag 和 client。
- **并行轮询**：同一会话同时查同一个 id，结果只给第一个完整行，其余返回 `returned_before`；之后再问一次照样给完整结果。
- **预检一次报全**：BRAIN 查字段慢时，没查完的字段以前被静默跳过，下一次提交才暴露。现在在 `warnings` 里写明哪些字段没来得及查，查询在后台继续，下一次预检就有结果。预检本身出错时也会提示“已跳过”。
- **不可用的算子**：本账号没有 `ts_min` / `ts_max` / `ts_median`。前两个直接给出等价写法（`subtract(x, ts_min_diff(x, d))`），其余按名字相似度给出候选。
- **CONCENTRATED_WEIGHT**：FULL 结果会取该 alpha 的 `coverage` 记录集（QUICK 没有），列出覆盖股票数不到中位数一半的交易日（`coverage_gaps`，节假日不算）。M 路实测的根因就是这种缺口日，`ts_backfill` 可以补上。
- **compare_alphas(anchors=[...])**：只算“候选 × anchors”，不再算 anchors 之间；`by_candidate` 给出每个候选最高的一对。
- **ProdMemo 点估计**：斜率低于 0.10 时停止给点估计，要回到 0.15 以上才恢复（`PRODMEMO_EST_RESUME_SLOPE`），不会在 0.10 附近来回跳。
- `ops_est` 对重复出现的子表达式按出现次数计，与平台一致（在有重复的真实 alpha 上核对过）。
- 上一轮报告里 QUICK 行的 IS_LADDER 不带数值：落盘记录显示都是 06:02Z 切换之前旧版本的行，新版本之后都带数值。没有数值的 IS_LADDER 是 PENDING（平台还没算），不计入 fails。

## 测试

```bash
pip install -r requirements-dev.txt
python -m pytest           # 进程内运行 fake BRAIN + credd，不访问真实平台
```

## 真实平台回归

```bash
export CREDD_URL=http://127.0.0.1:8762 CREDD_TOKEN=...
python scripts/live_regression.py                     # 只读
python scripts/live_regression.py --writes            # 另跑 single / multi / concurrent 各一组 QUICK 模拟，并对 alpha 属性改写再还原
python scripts/live_regression.py --raa --forum --prodmemo --selection "<表达式>"
python scripts/live_regression.py --probe-multi-raa   # 实测 multi 里能否放 RAA 条目（接受则立即取消）
python scripts/live_regression.py --accept-versions   # 带版本号 Accept 头再跑一遍，对比两次结果
```

脚本从不提交 alpha。报告写入 `live_regression_report.json`，其中 INFO 项就是目录里没法确定、需要实测给出结论的点。

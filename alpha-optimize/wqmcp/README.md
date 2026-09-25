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

## 工具一览（30 个）

| 分组 | 工具 |
|---|---|
| 账户 | `brain_status` |
| 回测 | `create_simulation` · `get_simulation` · `cancel_simulation` · `get_platform_setting_options` · `preview_super_selection` |
| Alpha | `list_alphas` · `get_alpha` · `get_alpha_recordset` · `check_alpha` · `submit_alpha` · `update_alpha` · `get_alpha_performance` |
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

**接口纠错**
- diversity 改用 `/activities/diversity`。
- 查看他人档案改用公开 `/profile` 接口。
- before-and-after 改用目录记载的接口。
- `get_activity(kind="pyramid-alphas")` 去掉了没有依据的回退路径。
- diversity-score 分页读取，不再逐个请求 alpha。
- payments 出错时两部分各自报原因。

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

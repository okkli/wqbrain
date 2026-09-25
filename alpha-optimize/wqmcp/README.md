# wqmcp：WorldQuant BRAIN MCP 服务

`platform_functions.py` 是入口，提供 48 个工具。登录由 credd（creds-daemon）负责，本服务从不
接触 BRAIN 密码。`API_AUDIT.md` 是对照 WQ API Catalog 1.15.3 做的接口审计，下文提到的问题编号
都指向它。

```bash
pip install mcp requests pandas pydantic beautifulsoup4 playwright psycopg2-binary
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
| `WQMCP_READ_ONLY` | `0` | 设为 `1` 时，所有会写 BRAIN 的工具都直接返回错误：建模拟、改 alpha 属性、提交 |
| `WQMCP_ALLOW_SUBMIT` | `1` | 设为 `0` 时禁止 `submit_alpha`，`get_submission_check` 仍可用 |
| `WQMCP_ACCEPT_VERSIONS` | `0` | 设为 `1` 时按目录发送带版本号的 Accept 头。默认关闭，因为线上验证过的是不带版本号的请求；先用 `scripts/live_regression.py --accept-versions` 实测，再决定是否打开 |
| `BRAIN_MESSAGE_IMAGE_MODE` | `ignore` | 消息里的内嵌图片默认直接去掉。设为 `placeholder` 时会把图片写到服务端磁盘（旧版的默认行为） |
| `WQMCP_BASE_URL` | `https://api.worldquantbrain.com` | 只给测试用 |

论坛相关的变量（`WQMCP_FORUM_*`、`WQMCP_GLOSSARY_TTL`）见 `forum_functions.py` 顶部。

## 这一轮优化（基于 main：工具名和已有参数都不变）

**安全**
- 所有拼进 URL 的 id 参数都先校验再编码，`../` 之类会被拒绝。
- `check_simulation_progress` 和 `lookINTO_SimError_message` 只接受 BRAIN 域名下的 `/simulations/<id>`，也可以直接传模拟 id。
- 服务默认只监听 127.0.0.1。
- `manage_config` 返回的配置会把密码、令牌等字段打码。
- 论坛工具不再从配置文件读取明文密码，并且只访问 support 站点本身的地址。
- credd 处于退避期时，并发的 401 不会轮流去打 credd。

**提交**
- `submit_alpha` 提交后会按 Retry-After 一直跟踪到最终的检查报告。
- 返回 `success` 和 `status`，取值为 SUBMITTED、REJECTED、PENDING、RATE_LIMITED、ERROR 之一；被拒时列出没通过的检查。
- 同一个 alpha 并发调用只会发一次 POST；返回 PENDING 后再调用，只会继续轮询，不会重复提交。
- 新增可选参数 `wait_seconds`，默认 60 秒。

**轮询与结果**
- `get_alpha_pnl`、`get_alpha_yearly_stats`、`get_record_sets`、`get_record_set_data`、`performance_comparison` 改为按 Retry-After 轮询。
  - 前两个仍保持 ProdMemo 依赖的约定：返回 `{}` 表示平台还在计算，真正出错时抛异常。
  - 后三个在超时仍未算完时返回 `status: PENDING`。
- multi 模拟：父任务还带 Retry-After 时，不再逐个查询子任务。
- 子任务被限流（429）或服务端出错（5xx）时，算作"仍在运行"，不再把整批报成 COMPLETE。

**接口纠错**
- `get_user_activities` 改用 `/activities/diversity`，现在 `grouping` 参数才真正生效。
- `get_user_profile` 查别人时改用公开档案接口，原来的接口会返回 403。
- `performance_comparison` 改用目录里有记载的 before-and-after 接口。
- `get_pyramid_alphas` 去掉了没有依据的回退路径。
- `value_factor_trendScore` 不再逐个请求 alpha 详情，改为分页读取并在本地按时间窗过滤。
- `get_daily_and_quarterly_payment` 出错时返回具体原因，不再只写 "no data"。

**分页**
- `get_datasets` 和 `get_datafields` 新增 `limit`、`offset` 参数。
- `run_selection` 新增 `limit` 参数，并同时发送目录记载的 `settings.*` 参数名和旧的参数名。

**论坛**：换成重写后的 `forum_functions.py`，包括 SSO 登录、浏览器复用、保留正文换行与代码、URL 编码和白名单。三个论坛工具的返回结构保持不变。

工具出错时仍然返回 `{"error": ...}` 字典，而不是抛出 MCP 错误，因为 `scripts/prodmemo_daily_sync.py` 等调用方依赖这种格式。

## 测试

```bash
pip install pytest pytest-asyncio
python -m pytest           # 进程内运行 fake BRAIN + credd，不访问真实平台
```

## 真实平台回归

```bash
export CREDD_URL=http://127.0.0.1:8762 CREDD_TOKEN=...
python scripts/live_regression.py                     # 只读
python scripts/live_regression.py --writes            # 另建几个 QUICK 模式的模拟，并对 alpha 属性做一次改写再还原
python scripts/live_regression.py --raa --forum --prodmemo --selection "<表达式>"
python scripts/live_regression.py --accept-versions   # 带版本号 Accept 头再跑一遍，对比两次结果
```

脚本从不提交 alpha。报告写入 `live_regression_report.json`，其中 INFO 项就是目录里没法确定、需要实测给出结论的点。

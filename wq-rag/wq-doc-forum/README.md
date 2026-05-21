# WorldQuant BRAIN 数据同步工具

拉取 BRAIN 论坛（顾问中文区）和官方知识库（教程 / 算子 / 数据集），输出
结构化 JSON，供下游做 RAG / 分析使用。

## 目录结构

```
wq-doc-forum/
├── forum_sync.py            # 薄入口：python forum_sync.py …
├── kb_sync.py               # 薄入口：python kb_sync.py …
├── sync/                    # 同步代码（包）
│   ├── __main__.py          # python -m sync {forum|kb}
│   ├── common/              # 共享：auth / config / io / logging
│   ├── forum/               # 论坛：browser / parse / list_sync / detail_sync / cli
│   └── kb/                  # 知识库：tutorials / operators / datasets / cli
├── data/                    # 所有抓取产物（输出）
│   ├── forum/
│   │   ├── index.json       # 帖子索引（元信息）
│   │   ├── posts/           # {post_id}.json 每帖一份（正文 + 评论）
│   │   ├── state/           # 同步状态 & 详情检查点
│   │   └── logs/            # 按日滚动日志
│   └── kb/
│       ├── tutorials/       # {page_id}.json 教程页（按稳定 ID 命名）
│       ├── operators/       # {category}.json 算子分类
│       ├── datasets/        # 数据集快照
│       ├── state.json       # 同步状态（lastModified / 内容哈希 / 时间戳）
│       └── logs/
├── requirements.txt
├── .env.example             # 复制为 .env 后填凭据
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
playwright install chromium

cp .env.example .env
# 编辑 .env，填入 BRAIN_EMAIL / BRAIN_PASSWORD
```

也可以直接导出环境变量：

```bash
export BRAIN_EMAIL=...
export BRAIN_PASSWORD=...
```

## 日常同步

```bash
# 论坛增量（每天 1 次，新帖 + 评论数变化的旧帖会回头补）
python forum_sync.py

# 官方知识库增量（每周 1 次）
python kb_sync.py
```

也可以用模块形式：

```bash
python -m sync forum
python -m sync kb
```

## 常用参数

### forum_sync.py

| 参数 | 说明 |
|---|---|
| `--list-only` | 只更新索引，不拉详情 |
| `--full-scan` | 强制翻到最后一页（默认翻到连续 90 个老帖就停） |
| `--force-details` | 强制重新拉取所有详情 |
| `--post <id>` | 只拉单个帖子 |

### kb_sync.py

| 参数 | 说明 |
|---|---|
| `--force` | 强制全量 |
| `--tutorials-only` / `--operators-only` / `--datasets-only` | 限定单一模块 |

## 增量逻辑

**论坛**

| 机制 | 实现 |
|---|---|
| 列表增量 | 翻页时遇到连续 90 个已存在帖子停止；`--full-scan` 关闭 |
| 详情增量 | 跳过已存在且完整的 `data/forum/posts/{id}.json` |
| 评论白名单 | 默认**不爬评论**；只对标题命中 `sync/forum/whitelist.py` 关键词的帖子拉评论 |
| 新评论回补 | 仅白名单帖：本地 `total_comments + 1 < 索引 comments_count` 触发重拉 |
| 失败隔离 | 失败的帖子**不写**到 posts/，只记到 `state/failures.json`；下次跑自动重试 |
| 中断续跑 | 每帖单独保存 + `state/detail_checkpoint.json` 记录最近进度 |
| 失败重试 | 单帖内嵌 3 次重试；跨次运行靠 failures.json + 文件缺失自然触发 |

调整评论白名单：编辑 `sync/forum/whitelist.py`，可改 `KEYWORDS` / `EXTRA_POST_IDS` / `EXCLUDE_POST_IDS`。

**知识库**

| 机制 | 实现 |
|---|---|
| 教程页 | 比对每页 `lastModified`，按 `{page_id}.json` 稳定命名（避免孤儿） |
| 算子 | 按分类做内容 SHA-256，hash 不变则跳过 |
| 数据集 | 整体内容 hash 增量 |
| 孤儿清理 | 远端已删除的教程页/算子分类，本地一并删除 |

## 查看数据

```bash
# 论坛统计
python -c "
import json, pathlib
idx = json.load(open('data/forum/index.json'))
posts = list(pathlib.Path('data/forum/posts').glob('*.json'))
comments = sum(json.load(open(f)).get('total_comments', 0) for f in posts)
print(f'帖子: {len(idx)}, 已拉详情: {len(posts)}, 评论合计: {comments}')
"

# 知识库统计
python -c "
import pathlib
for sub in ('tutorials', 'operators', 'datasets'):
    n = len(list(pathlib.Path(f'data/kb/{sub}').glob('*.json')))
    print(f'{sub}: {n}')
"
```

## 定时同步（可选）

```cron
# 每天 03:00 增量同步论坛
0 3 * * * cd /path/to/luntan && python forum_sync.py
# 每周日 04:00 同步知识库
0 4 * * 0 cd /path/to/luntan && python kb_sync.py
```

日志会写到 `data/{forum,kb}/logs/`，按日期切分。

## 与旧脚本的行为差异

| 项 | 旧版 | 新版 |
|---|---|---|
| `--category` / `--sort_by` | 支持 5 个分类过滤、4 种排序 | **去除**（默认拉全部，简化） |
| `--post <id>` | 仅限定范围，文件已存在不会重拉 | **总是重拉**，便于强制刷新单帖 |
| 失败的帖子 | 写到 `posts/{id}.json` 带 `error` 字段 → 后续跑被跳过 | 不写 posts/，只记 `state/failures.json` |
| 教程文件名 | `001_05_<title>_documentation.json`（sequence 变会留孤儿） | `{page_id}.json`（稳定 ID + 自动清理） |
| operators / datasets 增量 | 只看数量 | 内容 SHA-256，更可靠 |
| 凭据 | 硬编码到 .py | `.env` / 环境变量，缺失时显式报错 |

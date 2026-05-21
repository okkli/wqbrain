"""路径与运行参数的集中配置。

约定：所有数据写到 <repo_root>/data/，所有源码写在 sync/。
仓库根目录通过本文件的相对位置推导，避免依赖运行时 cwd。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    repo_root: Path = REPO_ROOT

    # 论坛
    forum_dir: Path = REPO_ROOT / "data" / "forum"
    forum_posts: Path = REPO_ROOT / "data" / "forum" / "posts"
    forum_index: Path = REPO_ROOT / "data" / "forum" / "index.json"
    forum_state_dir: Path = REPO_ROOT / "data" / "forum" / "state"
    forum_list_state: Path = REPO_ROOT / "data" / "forum" / "state" / "list_sync.json"
    forum_detail_checkpoint: Path = REPO_ROOT / "data" / "forum" / "state" / "detail_checkpoint.json"
    forum_logs: Path = REPO_ROOT / "data" / "forum" / "logs"

    # 知识库
    kb_dir: Path = REPO_ROOT / "data" / "kb"
    kb_tutorials: Path = REPO_ROOT / "data" / "kb" / "tutorials"
    kb_operators: Path = REPO_ROOT / "data" / "kb" / "operators"
    kb_datasets: Path = REPO_ROOT / "data" / "kb" / "datasets"
    kb_state: Path = REPO_ROOT / "data" / "kb" / "state.json"
    kb_logs: Path = REPO_ROOT / "data" / "kb" / "logs"

    def ensure(self) -> None:
        for p in (
            self.forum_posts, self.forum_state_dir, self.forum_logs,
            self.kb_tutorials, self.kb_operators, self.kb_datasets, self.kb_logs,
        ):
            p.mkdir(parents=True, exist_ok=True)


paths = Paths()


@dataclass(frozen=True)
class Settings:
    # 认证
    brain_email: str = os.environ.get("BRAIN_EMAIL", "")
    brain_password: str = os.environ.get("BRAIN_PASSWORD", "")
    api_base: str = "https://api.worldquantbrain.com"
    support_base: str = "https://support.worldquantbrain.com"

    # 论坛
    topic_id: str = "18910956638743"
    topic_slug: str = "顾问专属中文论坛"
    locale: str = "zh-cn"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
    )

    # 限流
    list_page_delay: float = 3.0
    detail_delay: float = 3.0
    comment_page_delay: float = 1.0
    max_retries: int = 3
    consecutive_stop: int = 90  # 连续 N 个已存在帖子就停止翻页


def _load_dotenv() -> None:
    """轻量 .env 读取（无需依赖 python-dotenv）。已存在的环境变量优先。"""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()
settings = Settings(
    brain_email=os.environ.get("BRAIN_EMAIL", ""),
    brain_password=os.environ.get("BRAIN_PASSWORD", ""),
)

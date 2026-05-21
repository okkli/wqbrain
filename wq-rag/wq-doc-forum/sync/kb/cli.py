"""知识库同步 CLI 入口。"""

from __future__ import annotations

import argparse

from ..common.auth import brain_session
from ..common.config import paths
from ..common.logging import get_logger
from .datasets import sync_datasets
from .operators import sync_operators
from .tutorials import sync_tutorials

_log = get_logger("kb", paths.kb_logs)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="kb_sync",
        description="WorldQuant BRAIN 官方知识库增量同步",
    )
    p.add_argument("--force", action="store_true", help="强制全量重新拉取")
    p.add_argument("--tutorials-only", action="store_true")
    p.add_argument("--operators-only", action="store_true")
    p.add_argument("--datasets-only", action="store_true")
    args = p.parse_args(argv)

    paths.ensure()
    session = brain_session()
    _log.info("BRAIN 认证成功")

    only_one = any([args.tutorials_only, args.operators_only, args.datasets_only])

    if not only_one or args.tutorials_only:
        sync_tutorials(session, force=args.force)
    if not only_one or args.operators_only:
        sync_operators(session, force=args.force)
    if not only_one or args.datasets_only:
        sync_datasets(session, force=args.force)

    total = sum(1 for _ in paths.kb_dir.rglob("*.json"))
    _log.info(f"知识库 JSON 总数: {total}（含 state.json）")


if __name__ == "__main__":
    main()

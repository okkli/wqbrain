"""论坛同步 CLI 入口。"""

from __future__ import annotations

import argparse
import asyncio

from ..common.config import paths
from ..common.io import load_json
from ..common.logging import get_logger

_log = get_logger("forum", paths.forum_logs)


async def _run(args: argparse.Namespace) -> None:
    # 延迟 import：让 --help 在未安装 playwright 时也能工作
    from .browser import forum_browser
    from .detail_sync import sync_details
    from .list_sync import sync_list

    paths.ensure()

    async with forum_browser() as ctx:
        # 1) 列表
        if args.post:
            index = load_json(paths.forum_index, {})
            if args.post not in index:
                _log.error(f"index 中未找到 post_id={args.post}")
                return
        else:
            index = await sync_list(ctx, full_scan=args.full_scan)

        # 2) 详情
        if not args.list_only:
            await sync_details(
                ctx, index,
                force=args.force_details,
                only_post=args.post,
            )

    # 3) 概览
    index = load_json(paths.forum_index, {})
    fetched = sum(1 for pid in index if (paths.forum_posts / f"{pid}.json").exists())
    print(
        "\n" + "=" * 50 + "\n"
        f"帖子总数: {len(index)}\n"
        f"已拉详情: {fetched}\n"
        f"数据目录: {paths.forum_dir}\n"
        + "=" * 50
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="forum_sync",
        description="WorldQuant BRAIN 论坛增量同步",
    )
    p.add_argument("--list-only", action="store_true", help="只同步列表索引，不拉详情")
    p.add_argument("--full-scan", action="store_true", help="强制翻到最后一页（默认增量）")
    p.add_argument("--force-details", action="store_true", help="强制重新拉取所有详情")
    p.add_argument("--post", type=str, default=None, help="只拉单个 post_id")
    args = p.parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

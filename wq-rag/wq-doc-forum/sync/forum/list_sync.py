"""列表（索引）增量同步。

增量策略：从第一页开始翻页，遇到连续 settings.consecutive_stop 个已存在
帖子就停止。`--full-scan` 强制翻到底，覆盖罕见情况（旧帖被置顶或重排）。

副作用：更新 data/forum/index.json；同时把可变字段（votes/评论数/类目）
更新到既有条目上，便于详情同步阶段识别"评论数变多需要重新拉"的帖子。
"""

from __future__ import annotations

import asyncio
import time

from ..common.config import paths, settings
from ..common.io import load_json, save_json_atomic
from ..common.logging import get_logger
from .parse import next_page_url, parse_post_list

_log = get_logger("forum", paths.forum_logs)


async def sync_list(context, *, full_scan: bool = False) -> dict[str, dict]:
    index: dict[str, dict] = load_json(paths.forum_index, {})
    page = await context.new_page()

    url = (
        f"{settings.support_base}/hc/{settings.locale}"
        f"/community/topics/{settings.topic_id}-{settings.topic_slug}"
    )

    page_num = 0
    new_count = 0
    updated_count = 0  # 评论/投票数有变化的旧帖
    consecutive_seen = 0

    _log.info(f"开始同步列表 (full_scan={full_scan})")
    while url:
        page_num += 1
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_selector(".striped-list-item", timeout=30000)
        except Exception as e:
            _log.warning(f"列表页 {page_num} 加载失败: {e}")
            break

        html = await page.content()
        posts = parse_post_list(html)
        if not posts:
            break

        for p in posts:
            pid = p["post_id"]
            if pid in index:
                consecutive_seen += 1
                # 检测可变字段变化
                old = index[pid]
                if (old.get("comments_count") != p["comments_count"]
                        or old.get("votes") != p["votes"]):
                    updated_count += 1
                old["votes"] = p["votes"]
                old["comments_count"] = p["comments_count"]
                old["category"] = p["category"]
            else:
                consecutive_seen = 0
                new_count += 1
                index[pid] = p

        _log.info(
            f"  第{page_num}页: 累计{len(index)}帖, 新增{new_count}, 评论/投票变更{updated_count}"
        )

        if not full_scan and consecutive_seen >= settings.consecutive_stop:
            _log.info(f"  连续 {consecutive_seen} 帖已存在，停止翻页")
            break

        nxt = next_page_url(html)
        if nxt:
            await asyncio.sleep(settings.list_page_delay)
            url = nxt
        else:
            url = None

    await page.close()
    save_json_atomic(paths.forum_index, index)
    save_json_atomic(paths.forum_list_state, {
        "last_sync_ts": time.time(),
        "last_sync_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_posts": len(index),
        "new_in_last_run": new_count,
        "changed_in_last_run": updated_count,
        "full_scan": full_scan,
    })
    _log.info(
        f"列表同步完成: 共 {len(index)} 帖, 新增 {new_count}, 评论/投票变更 {updated_count}"
    )
    return index

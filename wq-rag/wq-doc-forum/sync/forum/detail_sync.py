"""帖子详情增量同步。

默认**不爬评论**——白名单见 `whitelist.py`，命中的帖子才会拉评论。
非白名单帖：只存正文 + 元信息；白名单帖：正文 + 全部评论页。

增量策略（优先级递减）：
1. 若 only_post 指定，则只处理该帖（无条件重拉）。
2. 否则，对 index 中的每个帖子，命中任一条件就拉：
   - 详情文件不存在
   - force=True
   - 文件存在但缺 body / 显式 error 字段（历史失败残留）
   - 白名单帖：本地 total_comments + 1 < 索引 comments_count（评论增长回补）

成功才写 data/forum/posts/{post_id}.json；失败只写 state/failures.json，
不污染 posts/，确保下次跑还会自然重试。
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from ..common.config import paths, settings
from ..common.io import load_json, save_json_atomic
from ..common.logging import get_logger
from .parse import parse_comments, parse_post_detail
from .whitelist import should_fetch_comments

_log = get_logger("forum", paths.forum_logs)

_FAILURES_PATH = paths.forum_state_dir / "failures.json"


def _post_path(post_id: str):
    return paths.forum_posts / f"{post_id}.json"


def _needs_fetch(post_id: str, index_entry: dict, *, force: bool) -> bool:
    p = _post_path(post_id)
    if force:
        return True
    # 永久失败（如"未被授权"）：跳过，避免每次跑都浪费 3×45s 重试
    failures = load_json(_FAILURES_PATH, {}) or {}
    if failures.get(post_id, {}).get("permanent"):
        return False
    if not p.exists():
        return True
    try:
        existing = load_json(p, {})
    except Exception:
        return True
    if not isinstance(existing, dict):
        return True
    # 历史失败残留：文件存在但缺正文 / 显式 error 字段
    if "error" in existing or "body" not in existing:
        return True
    # 只有白名单帖子才追踪评论增长（其它帖子根本不存评论）
    if should_fetch_comments(post_id, index_entry.get("title", "")):
        remote_count = index_entry.get("comments_count", 0) or 0
        local_count = existing.get("total_comments", 0) or 0
        if remote_count > local_count + 1:
            return True
    return False


def _record_failure(post_id: str, url: str, err: str, *, permanent: bool = False) -> None:
    """把失败信息追加到 state/failures.json，便于人工排查 + 下次重试参考。"""
    failures = load_json(_FAILURES_PATH, {}) or {}
    failures[post_id] = {
        "url": url,
        "error": err[:500],
        "ts": time.time(),
        "permanent": permanent,
    }
    save_json_atomic(_FAILURES_PATH, failures)


def _clear_failure(post_id: str) -> None:
    failures = load_json(_FAILURES_PATH, {}) or {}
    if post_id in failures:
        failures.pop(post_id, None)
        save_json_atomic(_FAILURES_PATH, failures)


_FORBIDDEN_MARKERS = ("您未被授权", "Unauthorized", "未被授权访问")


async def _fetch_one(page, post_url: str, *, with_comments: bool) -> dict:
    """拉单个帖子详情，带重试。with_comments=True 时翻完所有评论页。"""
    last_err = None
    for retry in range(settings.max_retries):
        try:
            await page.goto(post_url, wait_until="domcontentloaded", timeout=60000)
            # 先检测"未被授权"页：节省 3×45s 的徒劳重试
            title = await page.title()
            if any(m in title for m in _FORBIDDEN_MARKERS):
                return {"error": f"forbidden: {title}", "url": post_url, "permanent": True}
            # state=attached: 元素在 DOM 中就算 ready，不要求 CSS visible
            # （社区帖的 .post-body 经常是 hidden，BS4 仍能解析）
            await page.wait_for_selector(
                ".post-body, .article-body", state="attached", timeout=45000,
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
            if retry < settings.max_retries - 1:
                _log.info(f"  重试 {retry + 1}/{settings.max_retries - 1}")
                await asyncio.sleep(5)
    if last_err is not None:
        return {"error": str(last_err), "url": post_url}

    base_url = post_url.split("?")[0].split("#")[0]
    html = await page.content()
    result = parse_post_detail(html, base_url, with_comments=with_comments)

    if with_comments:
        seen = {f"{c['author']}|{c['body'][:50]}" for c in result["comments"]}
        comment_page = 2
        while True:
            await asyncio.sleep(settings.comment_page_delay)
            c_url = f"{base_url}?page={comment_page}#comments"
            try:
                await page.goto(c_url, wait_until="domcontentloaded", timeout=30000)
            except Exception:
                break
            new_records = parse_comments(await page.content())
            if not new_records:
                break
            added = 0
            for rec in new_records:
                key = f"{rec['author']}|{rec['body'][:50]}"
                if key not in seen:
                    seen.add(key)
                    result["comments"].append(rec)
                    added += 1
            if added == 0 or len(new_records) < 30:
                break
            comment_page += 1
        result["total_comments"] = len(result["comments"])

    result["_fetched_at"] = time.time()
    return result


async def sync_details(
    context,
    index: dict[str, dict],
    *,
    force: bool = False,
    only_post: Optional[str] = None,
) -> tuple[int, int]:
    page = await context.new_page()

    todo: list[tuple[str, dict]] = []
    for pid, entry in index.items():
        if only_post and pid != only_post:
            continue
        if only_post or _needs_fetch(pid, entry, force=force):
            todo.append((pid, entry))

    _log.info(f"待拉取详情: {len(todo)} 帖 (force={force}, only_post={only_post})")
    done = fail = 0

    for i, (pid, entry) in enumerate(todo):
        with_comments = should_fetch_comments(pid, entry.get("title", ""))
        flag = "[+评论]" if with_comments else ""
        _log.info(f"[{i + 1}/{len(todo)}] {entry.get('title', '')[:40]} {flag}")
        detail = await _fetch_one(page, entry["link"], with_comments=with_comments)

        if "error" not in detail:
            save_json_atomic(_post_path(pid), detail)
            _clear_failure(pid)
            entry["_fetched_at"] = detail["_fetched_at"]
            entry["_comments_fetched"] = detail.get("total_comments", 0)
            done += 1
            cmt = f", {detail.get('total_comments', 0)}评论" if with_comments else ""
            _log.info(f"  OK: {len(detail.get('body', ''))}字{cmt}")
        else:
            # 失败：不污染 posts/，只记到 failures.json
            is_perm = bool(detail.get("permanent"))
            _record_failure(pid, entry["link"], detail["error"], permanent=is_perm)
            fail += 1
            tag = "PERM-FAIL" if is_perm else "FAIL"
            _log.warning(f"  {tag}: {detail['error'][:80]}")

        save_json_atomic(paths.forum_detail_checkpoint, {
            "last_post_id": pid,
            "progress": i + 1,
            "total_in_run": len(todo),
            "ts": time.time(),
        })
        if (i + 1) % 50 == 0:
            save_json_atomic(paths.forum_index, index)

        await asyncio.sleep(settings.detail_delay)

    await page.close()
    save_json_atomic(paths.forum_index, index)
    _log.info(f"详情同步完成: 成功 {done}, 失败 {fail}")
    return done, fail

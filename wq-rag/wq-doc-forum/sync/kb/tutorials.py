"""官方教程页同步：每 page 一个 {page_id}.json，按 lastModified 增量。

文件按稳定 ID 命名，避免上游 sequence 变化导致本地产生孤儿副本。
"""

from __future__ import annotations

import time

import requests

from ..common.config import paths, settings
from ..common.io import load_json, save_json_atomic
from ..common.logging import get_logger

_log = get_logger("kb", paths.kb_logs)


def _fetch_all_tutorials(session: requests.Session) -> list[dict]:
    out: list[dict] = []
    url = f"{settings.api_base}/tutorials"
    while url:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("results", []))
        nxt = data.get("next")
        if nxt and nxt.startswith("http://"):
            nxt = nxt.replace("http://", "https://")
        url = nxt
    return out


def sync_tutorials(session: requests.Session, *, force: bool = False) -> dict:
    _log.info("=== 同步 Tutorials ===")
    state = load_json(paths.kb_state, {})
    page_ts: dict[str, str] = state.get("tutorial_pages", {})

    tutorials = _fetch_all_tutorials(session)
    _log.info(f"上游 tutorials: {len(tutorials)} 个")

    saved = skipped = failed = 0
    seen_ids: set[str] = set()

    for tut in tutorials:
        cat = tut.get("category", "documentation")
        tut_title = tut.get("title", "")
        tut_seq = tut.get("sequence", 0)

        for page_idx, page_ref in enumerate(tut.get("pages", [])):
            page_id = page_ref.get("id")
            if not page_id:
                continue
            seen_ids.add(page_id)
            server_mtime = page_ref.get("lastModified", "")

            if not force and page_ts.get(page_id) == server_mtime:
                skipped += 1
                continue

            try:
                r = session.get(f"{settings.api_base}/tutorial-pages/{page_id}",
                                timeout=30)
                r.raise_for_status()
                detail = r.json()
            except Exception as e:
                failed += 1
                _log.warning(f"  失败 [{page_id}]: {e}")
                continue

            record = {
                "id": page_id,
                "title": detail.get("title", page_ref.get("title", page_id)),
                "category": cat,
                "tutorial": tut_title,
                "tutorial_sequence": tut_seq,
                "page_sequence": detail.get("sequence", page_idx),
                "lastModified": detail.get("lastModified", server_mtime),
                "content": detail.get("content", detail.get("code", "")),
            }
            save_json_atomic(paths.kb_tutorials / f"{page_id}.json", record)
            page_ts[page_id] = server_mtime
            saved += 1
            time.sleep(0.3)

    # 清理孤儿：远端已删除的 page，本地一并删除并从状态里移除
    pruned = 0
    if seen_ids:
        for f in paths.kb_tutorials.glob("*.json"):
            if f.stem not in seen_ids:
                f.unlink()
                page_ts.pop(f.stem, None)
                pruned += 1
    else:
        _log.warning("上游返回空集，跳过孤儿清理（防误删全部本地数据）")

    state["tutorial_pages"] = page_ts
    state["last_tutorial_sync"] = time.time()
    save_json_atomic(paths.kb_state, state)

    _log.info(f"Tutorials 完成: 更新 {saved}, 跳过 {skipped}, 失败 {failed}, 清理孤儿 {pruned}")
    return {"saved": saved, "skipped": skipped, "failed": failed, "pruned": pruned}

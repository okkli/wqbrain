"""官方算子同步：按 category 分文件，用内容哈希做增量。"""

from __future__ import annotations

import hashlib
import json
import time

import requests

from ..common.config import paths, settings
from ..common.io import load_json, save_json_atomic
from ..common.logging import get_logger

_log = get_logger("kb", paths.kb_logs)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:80]


def _hash(payload) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sync_operators(session: requests.Session, *, force: bool = False) -> dict:
    _log.info("=== 同步 Operators ===")
    r = session.get(f"{settings.api_base}/operators", timeout=30)
    r.raise_for_status()
    data = r.json()
    operators = data if isinstance(data, list) else data.get("results", [])
    _log.info(f"上游 operators: {len(operators)} 个")

    # 按 category 分组
    categories: dict[str, list[dict]] = {}
    for op in operators:
        cat = op.get("category", "other")
        categories.setdefault(cat, []).append(op)

    state = load_json(paths.kb_state, {})
    old_hashes: dict[str, str] = state.get("operator_hashes", {})
    new_hashes: dict[str, str] = {}

    saved = skipped = 0
    for cat, ops in categories.items():
        h = _hash(ops)
        new_hashes[cat] = h
        if not force and old_hashes.get(cat) == h:
            skipped += 1
            continue
        record = {
            "id": f"operators_{cat}",
            "title": f"{cat} operators ({len(ops)})",
            "category": "operators",
            "subcategory": cat,
            "count": len(ops),
            "operators": ops,
        }
        save_json_atomic(
            paths.kb_operators / f"{_safe_filename(cat)}.json",
            record,
        )
        saved += 1

    # 清理远端已消失的 category（防御：上游空响应时不清理）
    pruned = 0
    if categories:
        valid_files = {_safe_filename(c) + ".json" for c in categories}
        for f in paths.kb_operators.glob("*.json"):
            if f.name not in valid_files:
                f.unlink()
                pruned += 1
    else:
        _log.warning("上游 operators 为空，跳过孤儿清理（防误删全部本地数据）")

    state["operator_hashes"] = new_hashes
    state["operator_counts"] = {k: len(v) for k, v in categories.items()}
    state["last_operator_sync"] = time.time()
    save_json_atomic(paths.kb_state, state)

    _log.info(f"Operators 完成: 更新 {saved}, 跳过 {skipped}, 清理 {pruned}")
    return {"saved": saved, "skipped": skipped, "pruned": pruned}

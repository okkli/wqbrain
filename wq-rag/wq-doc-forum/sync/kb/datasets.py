"""官方数据集同步（USA EQUITY TOP3000 D1 默认配置）。

datasets API 没有官方的 lastModified 字段，用整体内容哈希做增量；
若需要细粒度（按 dataset_id 单文件），可在此扩展。
"""

from __future__ import annotations

import hashlib
import json
import time

import requests

from ..common.config import paths, settings
from ..common.io import load_json, save_json_atomic
from ..common.logging import get_logger

_log = get_logger("kb", paths.kb_logs)

_DEFAULT_QUERY = {
    "instrument_type": "EQUITY",
    "region": "USA",
    "delay": 1,
    "universe": "TOP3000",
}


def _hash(payload) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sync_datasets(session: requests.Session, *, force: bool = False) -> dict:
    _log.info("=== 同步 Datasets ===")

    datasets = None
    for path in ("/data", "/datasets"):
        try:
            r = session.get(f"{settings.api_base}{path}",
                            params=_DEFAULT_QUERY, timeout=30)
            if r.status_code == 200:
                data = r.json()
                datasets = data if isinstance(data, list) else data.get("results", [])
                break
        except Exception:
            continue

    if datasets is None:
        _log.warning("Datasets API 不可用，跳过（可用 MCP get_datasets 兜底）")
        return {"saved": 0, "skipped": 0, "reason": "api_unavailable"}

    _log.info(f"上游 datasets: {len(datasets)} 个")

    state = load_json(paths.kb_state, {})
    h = _hash(datasets)
    if not force and state.get("dataset_hash") == h:
        _log.info("Datasets 无变化，跳过")
        return {"saved": 0, "skipped": 1}

    record = {
        "id": "datasets_usa_equity_d1_top3000",
        "title": "Datasets - USA EQUITY TOP3000 D1",
        "category": "datasets",
        "query": _DEFAULT_QUERY,
        "count": len(datasets),
        "datasets": datasets,
    }
    save_json_atomic(
        paths.kb_datasets / "usa_equity_d1_top3000.json",
        record,
    )

    state["dataset_hash"] = h
    state["dataset_count"] = len(datasets)
    state["last_dataset_sync"] = time.time()
    save_json_atomic(paths.kb_state, state)

    _log.info(f"Datasets 完成: {len(datasets)} 个")
    return {"saved": 1, "skipped": 0}

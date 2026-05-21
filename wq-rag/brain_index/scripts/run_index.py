"""一键脚本：setup_qdrant → index_all。

用法:
    python3 scripts/run_index.py              # 增量
    python3 scripts/run_index.py --full       # 重嵌已索引内容
    python3 scripts/run_index.py --recreate   # 删集合重建（小心！）
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import setup_qdrant
import indexer


def main() -> int:
    recreate = "--recreate" in sys.argv
    full = "--full" in sys.argv
    setup_qdrant.setup(recreate=recreate)
    indexer.index_all(incremental=not full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

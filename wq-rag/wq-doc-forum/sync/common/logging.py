"""统一日志：同时输出到 stdout 和按天滚动的日志文件。

每个子领域（forum/kb）使用独立的 logger 命名空间，写入各自的 logs/ 目录。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_FMT = "[%(asctime)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_initialized: set[str] = set()


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _initialized:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            log_dir / f"{name.replace('.', '_')}_{time.strftime('%Y%m%d')}.log",
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _initialized.add(name)
    return logger

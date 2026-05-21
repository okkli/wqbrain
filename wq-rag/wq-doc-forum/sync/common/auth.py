"""BRAIN 平台认证。"""

from __future__ import annotations

import base64

import requests

from .config import settings


def brain_session() -> requests.Session:
    """Basic-Auth 登录 BRAIN API，返回带 cookie 的 Session。"""
    if not settings.brain_email or not settings.brain_password:
        raise RuntimeError(
            "未配置 BRAIN_EMAIL / BRAIN_PASSWORD。请编辑 .env 或导出环境变量。"
        )
    s = requests.Session()
    s.headers.update({"User-Agent": settings.user_agent})
    cred = base64.b64encode(
        f"{settings.brain_email}:{settings.brain_password}".encode()
    ).decode()
    r = s.post(
        f"{settings.api_base}/authentication",
        headers={"Authorization": f"Basic {cred}"},
        timeout=30,
    )
    if r.status_code != 201:
        raise RuntimeError(f"BRAIN 认证失败: HTTP {r.status_code} {r.text[:200]}")
    return s

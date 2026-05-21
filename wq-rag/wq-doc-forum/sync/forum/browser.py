"""Playwright 浏览器启动 + BRAIN SSO 登录。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from playwright.async_api import async_playwright

from ..common.auth import brain_session
from ..common.config import settings
from ..common.logging import get_logger

_log = get_logger("forum")


@asynccontextmanager
async def forum_browser():
    """异步上下文管理器：启动浏览器并完成 SSO，退出时清理资源。"""
    session = brain_session()
    _log.info("BRAIN API 认证成功")

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        channel="chrome", headless=True, args=["--no-sandbox"],
    )
    context = await browser.new_context(user_agent=settings.user_agent)

    # 把 requests 拿到的 cookie 注入到 Playwright
    api_cookies = []
    for c in session.cookies:
        cd = {
            "name": c.name, "value": c.value,
            "domain": c.domain, "path": c.path,
            "secure": c.secure,
            "httpOnly": "HttpOnly" in c._rest,
            "sameSite": "Lax",
        }
        if c.expires:
            cd["expires"] = c.expires
        api_cookies.append(cd)
    await context.add_cookies(api_cookies)

    page = await context.new_page()
    _log.info("SSO 登录中...")
    try:
        await page.goto(f"{settings.support_base}/access/sso",
                        wait_until="domcontentloaded", timeout=60000)
    except Exception:
        pass
    for _ in range(20):
        await asyncio.sleep(3)
        if "support.worldquantbrain.com/hc" in page.url:
            break
    _log.info(f"SSO 完成: {page.url}")
    await page.close()

    try:
        yield context
    finally:
        await browser.close()
        await pw.stop()

#!/usr/bin/env python3
"""
WorldQuant BRAIN MCP Server - Python Version
A comprehensive Model Context Protocol (MCP) server for WorldQuant BRAIN platform integration.

Login: managed by the credd daemon (brain-serve/creds-daemon) over its HTTP interface.
This server holds no password and never logs in by itself — it calls GET {CREDD_URL}/cookies
(CREDD_URL default http://127.0.0.1:8762) authenticated by the X-Auth-Token header
(CREDD_TOKEN env var), injects the cookies into its session, and self-heals on BRAIN 401
by re-pulling once (deduped via X-Cookie-Version). Safe for many concurrent MCP clients
(streamable-http): all BRAIN I/O runs in a bounded thread pool with real per-request
timeouts; nothing blocks the shared event loop.
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import re
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import sys
import math
import io
import threading
import time
# try set gbk problem
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    elif hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass

import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from pathlib import Path

# Import the new forum client
from forum_functions import forum_client
from prodmemo_service import prodmemo_client
from prodmemo_calc import extract_platform_correlation_stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _retry_after_seconds(response: requests.Response) -> float:
    """Parse the Retry-After header. Header values are strings — never compare
    them to int 0 directly. Missing/unparsable → 0.0."""
    try:
        return float(response.headers.get("Retry-After", 0))
    except (TypeError, ValueError):
        return 0.0


# Transient statuses a GET may retry (BRAIN's rate limit and gateway hiccups).
_RETRYABLE_STATUS = (429, 502, 503, 504)


def _http_error_detail(response: requests.Response, context: str = "") -> str:
    """One-line HTTP error including the API's response body (truncated).

    BRAIN's 4xx bodies carry the actual rejection reason, e.g.
    [{"settings":{"universe":["\"TOP9999\" is not a valid choice."]}}] —
    raise_for_status() drops them, leaving an undiagnosable "400 Bad Request"."""
    detail = (response.text or "").strip()
    if len(detail) > 1000:
        detail = detail[:1000] + "…"
    msg = f"{response.status_code} {response.reason} for url: {response.url}"
    if context:
        msg = f"{context}: {msg}"
    if detail:
        msg += f" — body: {detail}"
    return msg


# Region Agnostic Alpha (RAA): one simulation fans out into up to 4 region
# children (GLB/USA/ASI/EUR). Only these three pseudo-universes are accepted and
# delay must be 1; the platform rejects anything else outright.
RAA_UNIVERSES = ("LARGE", "MEDIUM", "SMALL")


def _raa_child_row(alpha: Dict[str, Any]) -> Dict[str, Any]:
    """Compact per-region metric row for one RA child alpha (parent carries no metrics)."""
    is_ = alpha.get("is") or {}
    checks = is_.get("checks") or []
    settings = alpha.get("settings") or {}

    def check_value(name: str):
        return next((c.get("value") for c in checks if c.get("name") == name), None)

    return {
        "alpha_id": alpha.get("id"),
        "region": settings.get("region"),
        "universe": settings.get("universe"),
        "sharpe": is_.get("sharpe"),
        "fitness": is_.get("fitness"),
        "turnover": is_.get("turnover"),
        "returns": is_.get("returns"),
        "drawdown": is_.get("drawdown"),
        "margin_bps": round((is_.get("margin") or 0) * 1e4, 2),
        "sharpe_2y": check_value("LOW_2Y_SHARPE"),
        "sub_universe_sharpe": check_value("LOW_SUB_UNIVERSE_SHARPE"),
        "fails": [c.get("name") for c in checks if c.get("result") == "FAIL"],
        "warnings": [c.get("name") for c in checks
                     if c.get("result") == "WARNING" and "MATCH" not in (c.get("name") or "")],
    }


def _short_check(name: Optional[str]) -> str:
    """LOW_2Y_SHARPE -> 2Y, HIGH_TURNOVER -> HTURNOVER (same shorthand as bq.py)."""
    return (name or "").replace("LOW_", "").replace("_SHARPE", "").replace("HIGH_", "H")


_FLIP_NOTE = ("if you got a negative alpha sharpe, you can just add a minus sign in front of "
              "the last line of the Alpha to flip then think the next step.")
_COMPACT_NOTE = ("Compact rows: margin in bps; robust/sub/y2_sharpe and cluster are the "
                 "LOW_ROBUST_UNIVERSE / LOW_SUB_UNIVERSE / LOW_2Y / CLUSTER_TEST check values; "
                 "fails lists failed checks. Call get_alpha_details(id) for the full object, or "
                 "check_simulation_progress(compact=False). " + _FLIP_NOTE)


# Settings echoed in a compact row, so children of a mixed-settings batch can be told apart.
_COMPACT_SETTING_KEYS = ("universe", "decay", "neutralization", "truncation", "maxTrade")


def _compact_alpha_row(alpha: Dict[str, Any]) -> Dict[str, Any]:
    """One-line summary of a finished alpha — the handful of numbers an
    optimisation loop actually reads, instead of the full ~5KB alpha object."""
    is_ = alpha.get("is") or {}
    checks = is_.get("checks") or []
    regular = alpha.get("regular") or {}
    settings = alpha.get("settings") or {}

    def check_value(name: str):
        return next((c.get("value") for c in checks if c.get("name") == name), None)

    return {
        "id": alpha.get("id"),
        "ops": regular.get("operatorCount"),
        "sharpe": is_.get("sharpe"),
        "fitness": is_.get("fitness"),
        "turnover": is_.get("turnover"),
        "margin_bps": round((is_.get("margin") or 0) * 1e4, 2),
        "robust_sharpe": check_value("LOW_ROBUST_UNIVERSE_SHARPE"),
        "sub_sharpe": check_value("LOW_SUB_UNIVERSE_SHARPE"),
        "y2_sharpe": check_value("LOW_2Y_SHARPE"),
        "cluster": check_value("CLUSTER_TEST"),
        "fails": [_short_check(c.get("name")) for c in checks if c.get("result") == "FAIL"],
        "set": {k: settings.get(k) for k in _COMPACT_SETTING_KEYS if k in settings},
        "expr": (regular.get("code") or "")[:110],
    }


def _finite(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _correlation_stats(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """max/min of a correlations payload: top-level max first (always present on a
    finished response), then schema.max, then the records' correlation column."""
    if _finite(data.get("max")) is not None:
        return {"max": _finite(data.get("max")), "min": _finite(data.get("min"))}
    schema = data.get("schema") or {}
    if isinstance(schema, dict) and _finite(schema.get("max")) is not None:
        return {"max": _finite(schema.get("max")), "min": _finite(schema.get("min"))}
    return extract_platform_correlation_stats(data)


def _correlation_top_rows(data: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """Top-n most correlated rows of a {schema, records} payload, as dicts."""
    schema = data.get("schema") or {}
    columns = [p.get("name") for p in schema.get("properties") or [] if isinstance(p, dict)]
    if "correlation" not in columns:
        return []
    idx = columns.index("correlation")
    rows = [r for r in data.get("records") or []
            if isinstance(r, list) and len(r) > idx and _finite(r[idx]) is not None]
    rows.sort(key=lambda r: float(r[idx]), reverse=True)
    return [dict(zip(columns, r)) for r in rows[:n]]


# --- credd (creds-daemon) HTTP interface ------------------------------------
# Login state comes from the credd daemon over its HTTP API (GET /cookies with
# an X-Auth-Token header) — this process never sees the BRAIN password and does
# no login of its own. See brain-serve/creds-daemon/README.md for the contract.
CREDD_URL = os.environ.get("CREDD_URL", "http://127.0.0.1:8762").rstrip("/")
CREDD_TOKEN = os.environ.get("CREDD_TOKEN", "")
_BRAIN_COOKIE_DOMAIN = ".worldquantbrain.com"


class CreddUnavailable(RuntimeError):
    """credd itself is unreachable / returned an error (distinct from a BRAIN 401)."""


class CreddSession(requests.Session):
    """requests.Session whose BRAIN cookies come from credd's HTTP interface.

    - cookies are pulled from ``GET {CREDD_URL}/cookies`` (``X-Auth-Token`` header
      when ``CREDD_TOKEN`` is set) and injected as one atomic jar rebind;
    - a BRAIN 401 self-heals: re-pull from credd once and retry, deduped by the
      ``X-Cookie-Version`` response header so a credd in backoff can't cause a
      401→refetch→401 spin;
    - thread-safe: refresh is serialized by a lock and the jar is swapped whole,
      so worker threads never observe a partially-cleared jar.
    """

    def __init__(self, *, credd_timeout: float = 15) -> None:
        super().__init__()
        self._credd_timeout = credd_timeout
        self._refresh_lock = threading.Lock()
        self._cookie_version: Optional[str] = None
        self.refresh_cookies()

    def _fetch_from_credd(self) -> Tuple[Dict[str, str], Optional[str]]:
        """GET /cookies from credd, mapping its error contract to clear messages."""
        headers = {"X-Auth-Token": CREDD_TOKEN} if CREDD_TOKEN else None
        try:
            r = requests.get(f"{CREDD_URL}/cookies", headers=headers,
                             timeout=self._credd_timeout)
        except requests.RequestException as exc:
            raise CreddUnavailable(
                f"credd unreachable at {CREDD_URL} (is creds-daemon running?): {exc}"
            ) from exc
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                body = {"error": "error", "detail": r.text[:200]}
            code = body.get("error")
            msg = f"credd returned {r.status_code}: {code} — {body.get('detail')}"
            if code == "biometric_required" and body.get("biometric_url"):
                msg += (f". Complete it in a browser at {body['biometric_url']} "
                        f"then POST {CREDD_URL}/complete-biometric")
            elif code == "unauthorized":
                msg += ". Check that this process's CREDD_TOKEN matches credd's"
            elif code in ("backoff", "rate_limited"):
                msg += f". Retry after ~{body.get('retry_after', '?')}s"
            raise CreddUnavailable(msg)
        return r.json(), r.headers.get("X-Cookie-Version")

    def refresh_cookies(self) -> Optional[str]:
        """Replace the whole cookie jar from credd; returns the cookie version."""
        with self._refresh_lock:
            cookies, version = self._fetch_from_credd()
            jar = requests.cookies.RequestsCookieJar()
            for name, value in cookies.items():
                jar.set(name, value, domain=_BRAIN_COOKIE_DOMAIN, path="/")
            self.cookies = jar  # atomic rebind — never a partially-cleared jar
            self._cookie_version = version
            return version

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        # Capture the version BEFORE the attempt: if another thread refreshes the
        # jar while our request is in flight, comparing against the post-failure
        # version would wrongly skip the retry ("no newer cookie") even though
        # our failed request never used the refreshed jar.
        version_used = self._cookie_version
        resp = super().request(method, url, *args, **kwargs)
        if not (resp.status_code == 401 and "worldquantbrain.com" in str(url)):
            return resp
        try:
            new_version = self.refresh_cookies()
        except CreddUnavailable:
            return resp  # credd down → surface the original 401, don't mask it
        if version_used is not None and new_version == version_used:
            return resp  # credd has no newer cookie (likely in backoff) → don't loop
        return super().request(method, url, *args, **kwargs)

SIMULATION_MODES = ("QUICK", "FULL")


def _normalize_simulation_mode(simulation_mode: Optional[str], visualization: bool):
    """Validate simulationMode and apply the platform rule that QUICK mode
    must be submitted with visualization=false.

    Returns (mode_or_None, visualization) or raises ValueError.
    """
    if simulation_mode is None or str(simulation_mode).strip() == "":
        return None, visualization
    mode = str(simulation_mode).strip().upper()
    if mode not in SIMULATION_MODES:
        raise ValueError(
            f"simulation_mode must be one of {list(SIMULATION_MODES)}, got '{simulation_mode}'"
        )
    if mode == "QUICK":
        visualization = False
    return mode, visualization


# Pydantic models for type safety
class SimulationSettings(BaseModel):
    instrumentType: str = "EQUITY"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    decay: float = 0.0
    neutralization: str = "NONE"
    truncation: float = 0.0
    pasteurization: str = "ON"
    unitHandling: Optional[str] = "VERIFY"
    nanHandling: Optional[str] = "OFF"
    language: str = "FASTEXPR"
    visualization: bool = True
    testPeriod: Optional[str] = "P0Y0M"
    # "QUICK" (fast feedback: core metrics only, no visualizations / Theme /
    # Competition / correlation checks, not directly submittable) or "FULL".
    # None -> field omitted from the payload (platform default, i.e. FULL).
    simulationMode: Optional[str] = None
    selectionHandling: str = "POSITIVE"
    selectionLimit: int = 1000
    maxTrade: str = "OFF"
    maxPosition: str = "OFF"
    componentActivation: str = "IS"
    # PYTHON-language specific
    lookback: Optional[int] = None

class SimulationData(BaseModel):
    type: str = "REGULAR"  # "REGULAR" or "SUPER"
    settings: SimulationSettings
    regular: Optional[str] = None
    combo: Optional[str] = None
    selection: Optional[str] = None

class BrainApiClient:
    """WorldQuant BRAIN API client with comprehensive functionality."""
    
    def __init__(self):
        self.base_url = "https://api.worldquantbrain.com"
        # Login state lives in credd (creds-daemon). The session is a lazily
        # created CreddSession: cookies come from credd, a BRAIN 401 self-heals
        # by re-pulling cookies from credd (thread-safe, atomic jar swap).
        self.session: Optional[CreddSession] = None
        self._session_lock = asyncio.Lock()
        # Dedicated bounded executor for all BRAIN HTTP I/O, so slow calls can't
        # exhaust the loop's default thread pool shared with other work.
        self._executor = ThreadPoolExecutor(
            max_workers=int(os.environ.get("WQMCP_HTTP_WORKERS", "32")),
            thread_name_prefix="wqmcp-http",
        )
        # (connect, read) timeout applied to every request unless overridden.
        # NOTE: requests.Session has no working `.timeout` attribute — it must
        # be passed per request (see _request).
        self.request_timeout = (
            float(os.environ.get("WQMCP_CONNECT_TIMEOUT", "10")),
            float(os.environ.get("WQMCP_READ_TIMEOUT", "60")),
        )
        # Monotonic deadline set by a GET 429; every GET waits it out first.
        self._cooldown_until = 0.0
    
    def log(self, message: str, level: str = "INFO"):
        """Log messages to stderr to avoid MCP protocol interference."""
        try:
            # Try to print with original message first
            print(f"[{level}] {message}", file=sys.stderr)
        except UnicodeEncodeError:
            # Fallback: remove problematic characters and try again
            try:
                safe_message = message.encode('ascii', 'ignore').decode('ascii')
                print(f"[{level}] {safe_message}", file=sys.stderr)
            except Exception:
                # Final fallback: just print the level and a safe message
                print(f"[{level}] Log message", file=sys.stderr)
        except Exception:
            # Final fallback: just print the level and a safe message
            print(f"[{level}] Log message", file=sys.stderr)

    def _build_session(self) -> CreddSession:
        """Blocking: pull cookies from credd and build the session. Run in executor."""
        session = CreddSession()  # fetches cookies from credd; raises CreddUnavailable if down
        adapter = HTTPAdapter(
            pool_connections=4,
            pool_maxsize=int(os.environ.get("WQMCP_POOL_MAXSIZE", "32")),
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session

    async def _ensure_session(self) -> CreddSession:
        """Lazily create the credd-backed session (first tool call pays the cost)."""
        if self.session is not None:
            return self.session
        async with self._session_lock:
            if self.session is None:
                loop = asyncio.get_running_loop()
                self.session = await loop.run_in_executor(self._executor, self._build_session)
                self.log(f"Session initialized from credd ({CREDD_URL})", "SUCCESS")
            return self.session

    async def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Run a synchronous requests call in the bounded thread pool with a real
        timeout, so a hung call can never freeze the event loop or leak a thread.

        GETs are retried twice on 429/502/503/504 and network errors, sleeping for
        BRAIN's Retry-After (bounded). A GET 429 also starts an account-wide
        cooldown so concurrent callers back off together. Writes are never
        retried: a POST /simulations 429 means "slots full" and is reported to the
        caller as RATE_LIMITED instead.
        """
        session = await self._ensure_session()
        kwargs.setdefault('timeout', self.request_timeout)
        loop = asyncio.get_running_loop()
        func = getattr(session, method)
        is_get = method.lower() == 'get'
        attempts = 3 if is_get else 1
        for attempt in range(attempts):
            last = attempt == attempts - 1
            cooldown = self._cooldown_until - time.monotonic()
            if is_get and cooldown > 0:
                await asyncio.sleep(min(cooldown, 30.0))
            try:
                resp = await loop.run_in_executor(self._executor, lambda: func(url, **kwargs))
            except requests.RequestException:
                if last:
                    raise
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            if not is_get or last or resp.status_code not in _RETRYABLE_STATUS:
                return resp
            wait = _retry_after_seconds(resp) or 2.0 * (attempt + 1)
            if resp.status_code == 429:
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + min(wait, 30.0))
            self.log(f"GET {url} -> {resp.status_code}, retrying in {min(wait, 15.0):.0f}s", "WARNING")
            await asyncio.sleep(min(wait, 15.0))
        return resp

    async def _check_once(self, location: str, compact: bool = True) -> Dict[str, Any]:
        """One status check of a simulation location — single or multi.

        A multi-simulation parent exposes a `children` list; each child is an
        ordinary simulation, so multi handling composes out of the single case.
        compact=True returns one _compact_alpha_row per finished alpha instead of
        the full alpha object.
        """
        resp = await self._request('get', location)
        if resp.status_code >= 400:
            return {"status": "ERROR", "http_status": resp.status_code,
                    "progress_url": location, "body": (resp.text or "")[:500]}
        try:
            body = resp.json() if (resp.text or "").strip() else {}
        except ValueError:
            return {"status": "ERROR", "progress_url": location,
                    "error": "Simulation endpoint returned non-JSON body",
                    "body": (resp.text or "")[:500]}

        children = body.get("children") or []

        # RAA: the parent simulation carries BOTH a parent `alpha` and per-region
        # `children` simulations. Handle it before the generic multi branch, which
        # would report the children and silently drop the parent (the submittable id).
        if str(body.get("type") or "").upper() in ("REGION_AGNOSTIC", "RA_PARENT") or (children and body.get("alpha")):
            if body.get("alpha"):
                return {**await self.get_raa_alpha(body["alpha"]),
                        "status": "COMPLETE", "progress_url": location}
            if "Retry-After" in resp.headers or str(body.get("status") or "").upper() in ("RUNNING", "PENDING"):
                return {
                    "status": "RUNNING",
                    "type": "REGION_AGNOSTIC",
                    "progress": body.get("progress"),
                    "total_children": len(children) or 4,
                    "retry_after_seconds": _retry_after_seconds(resp) or 5.0,
                    "progress_url": location,
                    "note": "RAA still running (up to 4 region children); check again after retry_after_seconds.",
                }
            # Settled without a parent alpha → fall through so the per-child
            # simulation errors are surfaced instead of a bare status.

        if children:
            return await self._check_multi_children(location, children, compact)

        if "Retry-After" in resp.headers:
            return {
                "status": "RUNNING",
                "progress": body.get("progress"),
                "retry_after_seconds": _retry_after_seconds(resp) or 5.0,
                "progress_url": location,
                "note": "Still running; check again after retry_after_seconds (do other work meanwhile).",
            }

        # Finished single simulation
        alpha_id = body.get("alpha")
        if not alpha_id:
            # Failed simulation: surface BRAIN's own error message.
            return {"status": body.get("status", "ERROR"), "progress_url": location,
                    "message": body.get("message"), "raw": body}
        alpha = await self._request('get', f"{self.base_url}/alphas/{alpha_id}")
        alpha.raise_for_status()
        if compact:
            return {"status": "COMPLETE", "progress_url": location,
                    "alpha": _compact_alpha_row(alpha.json()), "note": _COMPACT_NOTE}
        result = alpha.json()
        result['note'] = _FLIP_NOTE
        return result

    async def _check_multi_children(self, location: str, children: List[str],
                                    compact: bool = True) -> Dict[str, Any]:
        """Check all children of a multi-simulation concurrently."""
        child_urls = [c if str(c).startswith('http') else f"{self.base_url}/simulations/{c}"
                      for c in children]

        async def child_state(url: str) -> Dict[str, Any]:
            try:
                r = await self._request('get', url)
            except Exception as e:
                return {"location": url, "status": "UNKNOWN", "error": str(e)}
            if r.status_code >= 400:
                return {"location": url, "status": "ERROR", "http_status": r.status_code}
            try:
                b = r.json() if (r.text or "").strip() else {}
            except ValueError:
                b = {}
            if "Retry-After" in r.headers:
                return {"location": url, "status": "RUNNING", "progress": b.get("progress")}
            alpha_id = b.get("alpha")
            if not alpha_id:
                return {"location": url, "status": b.get("status", "ERROR"),
                        "message": b.get("message")}
            return {"location": url, "status": "COMPLETE", "alpha_id": alpha_id}

        states = list(await asyncio.gather(*[child_state(u) for u in child_urls]))
        unfinished = [s for s in states if s["status"] in ("RUNNING", "UNKNOWN")]
        if unfinished:
            done_n = len(states) - len(unfinished)
            return {
                "status": "RUNNING",
                "type": "MULTI",
                "completed_children": done_n,
                "total_children": len(states),
                "children": states,
                "retry_after_seconds": 5.0,
                "progress_url": location,
                "note": "Multi-simulation still running; check again after retry_after_seconds.",
            }

        # All children settled — fetch alpha details for the completed ones concurrently.
        async def with_details(s: Dict[str, Any]) -> Dict[str, Any]:
            if s["status"] != "COMPLETE":
                return s
            try:
                d = await self._request('get', f"{self.base_url}/alphas/{s['alpha_id']}")
                d.raise_for_status()
                if compact:
                    return {"status": "COMPLETE", **_compact_alpha_row(d.json())}
                return {**s, "details": d.json()}
            except Exception as e:
                return {**s, "error": f"failed to fetch alpha details: {e}"}

        full = list(await asyncio.gather(*[with_details(s) for s in states]))
        return {
            "status": "COMPLETE",
            "type": "MULTI",
            "total_children": len(full),
            "alpha_results": full,
            "progress_url": location,
            "note": _COMPACT_NOTE if compact else _FLIP_NOTE,
        }

    async def authenticate(self, email: str = "", password: str = "") -> Dict[str, Any]:
        """Refresh login state from credd. Credentials live only in the
        creds-daemon — email/password arguments are accepted for backward
        compatibility but ignored."""
        self.log("🔐 Refreshing BRAIN login state from credd...", "INFO")
        try:
            session = await self._ensure_session()
            loop = asyncio.get_running_loop()
            # Force-pull the latest cookies from credd (credd itself re-auths
            # in place only when actually stale, with anti-lockout backoff).
            await loop.run_in_executor(self._executor, session.refresh_cookies)

            response = await self._request('get', f"{self.base_url}/authentication")
            if response.status_code == 200:
                data = {}
                try:
                    data = response.json()
                except ValueError:
                    pass
                return {
                    'user': data.get('user', {}),
                    'status': 'authenticated',
                    'token_expiry': (data.get('token') or {}).get('expiry'),
                    'message': 'Authenticated via credd (creds-daemon)',
                    'credd_url': CREDD_URL,
                }
            raise Exception(
                f"credd cookie did not pass BRAIN validation (HTTP {response.status_code}). "
                f"Check credd /status at {CREDD_URL} — it may be in backoff or awaiting "
                f"biometric verification (complete it via its biometric_url, then POST /complete-biometric)."
            )
        except CreddUnavailable as e:
            self.log(f"❌ credd unavailable: {e}", "ERROR")
            raise Exception(
                f"credd (login daemon) unavailable at {CREDD_URL}: {e}. "
                f"Start creds-daemon (brain-serve/creds-daemon/run.sh) or set CREDD_URL."
            )
        except Exception as e:
            self.log(f"❌ Authentication failed: {str(e)}", "ERROR")
            raise

    async def is_authenticated(self) -> bool:
        """Check the credd-backed session against BRAIN (401 self-heals once inside
        CreddSession before we see the result)."""
        try:
            response = await self._request('get', f"{self.base_url}/authentication")
            return response.status_code == 200
        except Exception as e:
            self.log(f"❌ Error checking authentication: {str(e)}", "ERROR")
            return False

    async def ensure_authenticated(self):
        """Make sure the credd-backed session exists. Intentionally cheap: no
        global lock and no per-call network probe — an expired cookie self-heals
        inside CreddSession when a request hits 401 (re-pull from credd + retry)."""
        await self._ensure_session()
    
    async def get_authentication_status(self) -> Optional[Dict[str, Any]]:
        """Get current authentication status and user info."""
        try:
            response = await self._request('get', f"{self.base_url}/users/self")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get auth status: {str(e)}", "ERROR")
            return None
    
    async def create_simulation(self, simulation_data: SimulationData) -> Dict[str, str]:
        """Create a new simulation on BRAIN platform."""
        await self.ensure_authenticated()
        
        try:
            self.log("🚀 Creating simulation...", "INFO")
            
            # Prepare settings based on simulation type
            settings_dict = simulation_data.settings.model_dump()

            # Remove fields based on simulation type
            if simulation_data.type in ("REGULAR", "REGION_AGNOSTIC"):
                # Remove SUPER-specific fields for REGULAR / RAA
                settings_dict.pop('selectionHandling', None)
                settings_dict.pop('selectionLimit', None)
                settings_dict.pop('componentActivation', None)

            # Remove fields based on expression language
            language = (settings_dict.get('language') or '').upper()
            if language == "PYTHON":
                # PYTHON payload omits these FASTEXPR-only fields
                for k in ('unitHandling', 'nanHandling', 'testPeriod'):
                    settings_dict.pop(k, None)
            else:
                # Non-PYTHON languages don't carry lookback
                settings_dict.pop('lookback', None)

            # Filter out None values from settings
            settings_dict = {k: v for k, v in settings_dict.items() if v is not None}
            
            # Prepare simulation payload
            payload = {
                'type': simulation_data.type,
                'settings': settings_dict
            }
            
            # Add type-specific fields
            if simulation_data.type in ("REGULAR", "REGION_AGNOSTIC"):
                if simulation_data.regular:
                    payload['regular'] = simulation_data.regular
            elif simulation_data.type == "SUPER":
                if simulation_data.combo:
                    payload['combo'] = simulation_data.combo
                if simulation_data.selection:
                    payload['selection'] = simulation_data.selection
            
            # Filter out None values from entire payload
            payload = {k: v for k, v in payload.items() if v is not None}
            
            response = await self._request('post', f"{self.base_url}/simulations", json=payload)
            if response.status_code == 429:
                # Per-account concurrent simulation slots are full (e.g. other
                # sessions' simulations still running) — return structured info
                # instead of a raw error so the client can back off sensibly.
                retry_after = _retry_after_seconds(response) or 30.0
                return {
                    "status": "RATE_LIMITED",
                    "retry_after_seconds": retry_after,
                    "note": ("BRAIN's per-account concurrent simulation limit is reached "
                             "(another simulation is still running on this account). "
                             f"Retry create_simulation after ~{int(retry_after)}s, or first "
                             "finish/check the running ones; you can do other work meanwhile."),
                }
            if response.status_code >= 400:
                # Surface BRAIN's rejection reason (invalid settings choice,
                # blank expression, ...) instead of a bare "400 Bad Request".
                raise Exception(_http_error_detail(response, "simulation rejected"))
            response.raise_for_status()

            location = response.headers.get('Location', '')
            if not location:
                raise Exception("BRAIN returned no Location header for the submitted simulation")
            simulation_id = location.split('/')[-1]

            self.log(f"Simulation submitted with ID: {simulation_id}", "SUCCESS")

            # Submit-only: return immediately so the MCP client is never parked
            # on a long-running HTTP call. Progress/result via check_simulation_progress.
            return {
                "status": "SUBMITTED",
                "simulation_id": simulation_id,
                "progress_url": location,
                "note": ("Simulation is running asynchronously (typically 1-5 minutes). "
                         "Call check_simulation_progress with this progress_url to get "
                         "progress (0.0-1.0) and, once finished, the full alpha result. "
                         "You can do other work between checks."),
            }

        except Exception as e:
            self.log(f"❌ Failed to create simulation: {str(e)}", "ERROR")
            raise

    async def check_simulation_progress(self, location: str, wait_seconds: float = 0,
                                        compact: bool = True) -> Dict[str, Any]:
        """Check a submitted simulation (single OR multi) once — or keep checking
        within a bounded wait budget — returning progress while running and full
        alpha details once finished. Transient 5xx/network errors are retried
        while wait budget remains instead of aborting the wait."""
        await self.ensure_authenticated()

        wait_budget = max(0.0, min(float(wait_seconds or 0), 120.0))
        waited = 0.0
        while True:
            try:
                state = await self._check_once(location, compact)
            except Exception as e:
                if waited >= wait_budget:
                    raise
                state = {"status": "RUNNING", "retry_after_seconds": 5.0,
                         "progress_url": location, "transient_error": str(e)}
            transient_5xx = (state.get("status") == "ERROR"
                             and (state.get("http_status") or 0) >= 500)
            if state.get("status") != "RUNNING" and not transient_5xx:
                return state
            if waited >= wait_budget:
                return state
            wait = max(min(state.get("retry_after_seconds") or 5.0, wait_budget - waited), 1.0)
            await asyncio.sleep(wait)
            waited += wait
    
    async def get_raa_alpha(self, parent_alpha_id: str) -> Dict[str, Any]:
        """Summarise an RA parent alpha: settings/expression plus one metric row per
        region child. An RA_PARENT carries no metrics of its own — all performance
        lives on its RA_CHILD alphas, so they are fetched concurrently."""
        await self.ensure_authenticated()

        parent_resp = await self._request('get', f"{self.base_url}/alphas/{parent_alpha_id}")
        parent_resp.raise_for_status()
        parent = parent_resp.json()
        child_ids = parent.get("children") or []
        if not child_ids:
            return {"type": parent.get("type"), "parent_alpha_id": parent_alpha_id,
                    "error": "Alpha has no children — it is not an RA parent alpha.",
                    "parent": parent}

        async def child_row(cid: str) -> Dict[str, Any]:
            try:
                r = await self._request('get', f"{self.base_url}/alphas/{cid}")
                r.raise_for_status()
                return _raa_child_row(r.json())
            except Exception as e:
                return {"alpha_id": cid, "error": str(e)}

        rows = list(await asyncio.gather(*[child_row(c) for c in child_ids]))
        settings = parent.get("settings") or {}
        ok = [r for r in rows if not r.get("error")]
        no_fail = [r for r in ok if not r.get("fails")]
        clean = [r for r in no_fail if not r.get("warnings")]
        return {
            "type": "REGION_AGNOSTIC",
            "parent_alpha_id": parent_alpha_id,
            "expression": (parent.get("regular") or {}).get("code"),
            "settings": {k: settings.get(k) for k in
                         ("universe", "delay", "decay", "neutralization", "truncation",
                          "maxTrade", "maxPosition")},
            "children": rows,
            "children_without_fails": len(no_fail),
            "children_without_warnings": len(clean),
            "note": ("Submission needs >=2 children with no FAIL. Then run "
                     "check_correlation on those children: one child passing PROD "
                     "correlation lets all passing children be submitted together "
                     "(the whole RAA counts as a single submission). Use "
                     "get_submission_check on the PARENT id — children cannot be "
                     "checked individually."),
        }

    async def get_alpha_details(self, alpha_id: str) -> Dict[str, Any]:
        """Get detailed information about an alpha."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get alpha details: {str(e)}", "ERROR")
            raise
    
    async def get_datasets(self, instrument_type: str = "EQUITY", region: str = "USA",
                          delay: int = 1, universe: str = "TOP3000", theme: str = "false", search: Optional[str] = None) -> Dict[str, Any]:
        """Get available datasets."""
        await self.ensure_authenticated()
        
        try:
            params = {
                'instrumentType': instrument_type,
                'region': region,
                'delay': delay,
                'universe': universe,
                'theme': theme
            }
            
            if search:
                params['search'] = search
            
            response = await self._request('get', f"{self.base_url}/data-sets", params=params)
            response.raise_for_status()
            response_json = response.json()
            response_json['extraNote'] = "if your returned result is 0, you may want to check your parameter by using get_platform_setting_options tool to got correct parameter"
            return response_json
        except Exception as e:
            self.log(f"Failed to get datasets: {str(e)}", "ERROR")
            raise
    
    async def get_datafields(self, instrument_type: str = "EQUITY", region: str = "USA",
                            delay: int = 1, universe: str = "TOP3000", theme: str = "false",
                            dataset_id: Optional[str] = None, data_type: str = "",
                            search: Optional[str] = None) -> Dict[str, Any]:
        """Get available data fields."""
        await self.ensure_authenticated()
        
        try:
            params = {
                'instrumentType': instrument_type,
                'region': region,
                'delay': delay,
                'universe': universe,
                'limit': '50',
                'offset': '0'
            }
            
            if data_type != 'ALL':
                params['type'] = data_type
            
            if dataset_id:
                params['dataset.id'] = dataset_id
            if search:
                params['search'] = search
            
            response = await self._request('get', f"{self.base_url}/data-fields", params=params)
            response.raise_for_status()
            response_json = response.json()
            response_json['extraNote'] = "if your returned result is 0, you may want to check your parameter by using get_platform_setting_options tool to got correct parameter"
            return response_json
        except Exception as e:
            self.log(f"Failed to get datafields: {str(e)}", "ERROR")
            raise
    
    async def get_alpha_pnl(self, alpha_id: str) -> Dict[str, Any]:
        """Get PnL data for an alpha with retry logic."""
        await self.ensure_authenticated()
        
        max_retries = 5
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.log(f"Attempting to get PnL for alpha {alpha_id} (attempt {attempt + 1}/{max_retries})", "INFO")
                
                response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/recordsets/pnl")
                response.raise_for_status()

                # Some alphas may return 204 No Content or an empty body
                text = (response.text or "").strip()
                if not text:
                    if attempt < max_retries - 1:
                        self.log(f"Empty PnL response for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                        continue
                    else:
                        self.log(f"Empty PnL response after {max_retries} attempts for {alpha_id}", "WARNING")
                        return {}
                
                try:
                    pnl_data = response.json()
                    if pnl_data:
                        self.log(f"Successfully retrieved PnL data for alpha {alpha_id}", "SUCCESS")
                        return pnl_data
                    else:
                        if attempt < max_retries - 1:
                            self.log(f"Empty PnL JSON for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 1.5
                            continue
                        else:
                            self.log(f"Empty PnL JSON after {max_retries} attempts for {alpha_id}", "WARNING")
                            return {}
                            
                except Exception as parse_err:
                    if attempt < max_retries - 1:
                        self.log(f"PnL JSON parse failed for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5
                        continue
                    else:
                        self.log(f"PnL JSON parse failed for {alpha_id} after {max_retries} attempts: {parse_err}", "WARNING")
                        return {}
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    self.log(f"Failed to get alpha PnL for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds: {str(e)}", "WARNING")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                else:
                    self.log(f"Failed to get alpha PnL for {alpha_id} after {max_retries} attempts: {str(e)}", "ERROR")
                    raise
        
        # This should never be reached, but just in case
        return {}
    
    async def get_user_alphas(
        self,
        stage: str = "OS",
        limit: int = 30,
        offset: int = 0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        submission_start_date: Optional[str] = None,
        submission_end_date: Optional[str] = None,
        order: Optional[str] = None,
        hidden: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Get user's alphas with advanced filtering."""
        await self.ensure_authenticated()
        
        try:
            params = {
                "stage": stage,
                "limit": limit,
                "offset": offset,
            }
            if start_date:
                params["dateCreated>"] = start_date
            if end_date:
                params["dateCreated<"] = end_date
            if submission_start_date:
                params["dateSubmitted>"] = submission_start_date
            if submission_end_date:
                params["dateSubmitted<"] = submission_end_date
            if order:
                params["order"] = order
            if hidden is not None:
                params["hidden"] = str(hidden).lower()

            response = await self._request('get', f"{self.base_url}/users/self/alphas", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get user alphas: {str(e)}", "ERROR")
            raise
    
    async def submit_alpha(self, alpha_id: str) -> bool:
        """Submit an alpha for production."""
        await self.ensure_authenticated()
        
        try:
            self.log(f"📤 Submitting alpha {alpha_id} for production...", "INFO")
            
            response = await self._request('post', f"{self.base_url}/alphas/{alpha_id}/submit")
            response.raise_for_status()

            self.log(f"Alpha {alpha_id} submitted successfully", "SUCCESS")
            return response.__dict__
            
        except Exception as e:
            self.log(f"❌ Failed to submit alpha: {str(e)}", "ERROR")
            return False
    
    async def get_events(self) -> Dict[str, Any]:
        """Get available events and competitions."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/events")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get events: {str(e)}", "ERROR")
            raise
    
    async def get_leaderboard(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get leaderboard data."""
        await self.ensure_authenticated()
        
        try:
            params = {}
            
            if user_id:
                params['user'] = user_id
            else:
                # Get current user ID if not specified
                user_response = await self._request('get', f"{self.base_url}/users/self")
                if user_response.status_code == 200:
                    user_data = user_response.json()
                    params['user'] = user_data.get('id')

            response = await self._request('get', f"{self.base_url}/consultant/boards/leader", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get leaderboard: {str(e)}", "ERROR")
            raise

    def _is_atom(self, detail: Optional[Dict[str, Any]]) -> bool:
        """Match atom detection used in extract_regular_alphas.py:
        - Primary signal: 'classifications' entries containing 'SINGLE_DATA_SET'
        - Fallbacks: tags list contains 'atom' or classification id/name contains 'ATOM'
        """
        if not detail or not isinstance(detail, dict):
            return False

        classifications = detail.get('classifications') or []
        for c in classifications:
            cid = (c.get('id') or c.get('name') or '')
            if isinstance(cid, str) and 'SINGLE_DATA_SET' in cid:
                return True

        # Fallbacks
        tags = detail.get('tags') or []
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, str) and t.strip().lower() == 'atom':
                    return True

        for c in classifications:
            cid = (c.get('id') or c.get('name') or '')
            if isinstance(cid, str) and 'ATOM' in cid.upper():
                return True

        return False

    async def value_factor_trendScore(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Compute diversity score for regular alphas in a date range.

        Description:
        This function calculate the diversity of the users' submission, by checking the diversity, we can have a good understanding on the valuefactor's trend.
        value factor of a user is defiend by This diversity score, which measures three key aspects of work output: the proportion of works
        with the "Atom" tag (S_A), atom proportion, the breadth of pyramids covered (S_P), and how evenly works
        are distributed across those pyramids (S_H). Calculated as their product, it rewards
        strong performance across all three dimensions—encouraging more Atom-tagged works,
        wider pyramid coverage, and balanced distribution—with weaknesses in any area lowering
        the total score significantly.

        Inputs (hints for AI callers):
        - start_date (str): ISO UTC start datetime, e.g. '2025-08-14T00:00:00Z'
        - end_date (str): ISO UTC end datetime, e.g. '2025-08-18T23:59:59Z'
        - Note: this tool always uses 'OS' (submission dates) to define the window; callers do not need to supply a stage.
                - Note: P_max (total number of possible pyramids) is derived from the platform
                    pyramid-multipliers endpoint and not supplied by callers.

        Returns (compact JSON): {
            'diversity_score': float,
            'N': int,  # total regular alphas in window
            'A': int,  # number of Atom-tagged works (is_single_data_set)
            'P': int,  # pyramid coverage count in the sample
            'P_max': int, # used max for normalization
            'S_A': float, 'S_P': float, 'S_H': float,
            'per_pyramid_counts': {pyramid_name: count}
        }
        """
        # Fetch user alphas (always use OS / submission dates per product policy)
        await self.ensure_authenticated()
        alphas_resp = await self.get_user_alphas(stage='OS', limit=500, submission_start_date=start_date, submission_end_date=end_date)

        if not isinstance(alphas_resp, dict) or 'results' not in alphas_resp:
            return {'error': 'Unexpected response from get_user_alphas', 'raw': alphas_resp}

        alphas = alphas_resp['results']
        regular = [a for a in alphas if a.get('type') == 'REGULAR']

        # Fetch details for each regular alpha
        pyramid_list = []
        atom_count = 0
        per_pyramid = {}
        for a in regular:
            try:
                detail = await self.get_alpha_details(a.get('id'))
            except Exception:
                continue

            is_atom = self._is_atom(detail)
            if is_atom:
                atom_count += 1

            # Extract pyramids
            ps = []
            if isinstance(detail.get('pyramids'), list):
                ps = [p.get('name') for p in detail.get('pyramids') if p.get('name')]
            else:
                pt = detail.get('pyramidThemes') or {}
                pss = pt.get('pyramids') if isinstance(pt, dict) else None
                if pss and isinstance(pss, list):
                    ps = [p.get('name') for p in pss if p.get('name')]

            for p in ps:
                pyramid_list.append(p)
                per_pyramid[p] = per_pyramid.get(p, 0) + 1

        N = len(regular)
        A = atom_count
        P = len(per_pyramid)

        # Determine P_max similarly to the script: use pyramid multipliers if available
        P_max = None
        try:
            pm = await self.get_pyramid_multipliers()
            if isinstance(pm, dict) and 'pyramids' in pm:
                pyramids_list = pm.get('pyramids') or []
                P_max = len(pyramids_list)
        except Exception:
            P_max = None

        if not P_max or P_max <= 0:
            P_max = max(P, 1)

        # Component scores
        S_A = (A / N) if N > 0 else 0.0
        S_P = (P / P_max) if P_max > 0 else 0.0

        # Entropy
        S_H = 0.0
        if P <= 1 or not per_pyramid:
            S_H = 0.0
        else:
            total_occ = sum(per_pyramid.values())
            H = 0.0
            for cnt in per_pyramid.values():
                q = cnt / total_occ if total_occ > 0 else 0
                if q > 0:
                    H -= q * math.log2(q)
            max_H = math.log2(P) if P > 0 else 1
            S_H = (H / max_H) if max_H > 0 else 0.0

        diversity_score = S_A * S_P * S_H

        return {
            'diversity_score': diversity_score,
            'N': N,
            'A': A,
            'P': P,
            'P_max': P_max,
            'S_A': S_A,
            'S_P': S_P,
            'S_H': S_H,
            'per_pyramid_counts': per_pyramid
        }

    async def get_operators(self) -> Dict[str, Any]:
        """Get available operators for alpha creation."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/operators")
            response.raise_for_status()
            operators_data = response.json()
            
            # Ensure we return a dictionary format even if API returns a list
            if isinstance(operators_data, list):
                return {"operators": operators_data, "count": len(operators_data)}
            else:
                return operators_data
        except Exception as e:
            self.log(f"Failed to get operators: {str(e)}", "ERROR")
            raise
            
    async def run_selection(
        self,
        selection: str,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        delay: int = 1,
        selection_limit: int = 1000,
        selection_handling: str = "POSITIVE"
    ) -> Dict[str, Any]:
        """Run a selection query to filter instruments."""
        await self.ensure_authenticated()
        
        try:
            selection_data = {
                "selection": selection,
                "instrumentType": instrument_type,
                "region": region,
                "delay": delay,
                "selectionLimit": selection_limit,
                "selectionHandling": selection_handling
            }
            
            response = await self._request('get', f"{self.base_url}/simulations/super-selection", params=selection_data)
            if response.status_code >= 400:
                raise Exception(_http_error_detail(response, "selection rejected"))
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to run selection: {str(e)}", "ERROR")
            raise

    async def get_user_profile(self, user_id: str = "self") -> Dict[str, Any]:
        """Get user profile information."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/users/{user_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get user profile: {str(e)}", "ERROR")
            raise
            
    async def get_documentations(self) -> Dict[str, Any]:
        """Get available documentations and learning materials."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/tutorials")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get documentations: {str(e)}", "ERROR")
            raise
            
    async def get_messages(self, limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
        """Get messages for the current user with optional pagination.

        Image / large binary payload mitigation:
          Some messages embed base64 encoded images (e.g. <img src="data:image/png;base64,..."/>).
          Returning full base64 can explode token usage for an LLM client. We post-process each
          message description and (by default) extract embedded base64 images to disk and replace
          them with lightweight placeholders while preserving context.

        Strategies (environment driven in future – currently parameterless public API):
          - placeholder (default): save images to message_images/ and replace with marker text.
          - ignore: strip image tags entirely, leaving a note.
          - keep: leave description unchanged (unsafe for LLM token limits).

        A message dict gains an 'extracted_images' list when images are processed.
        """
        await self.ensure_authenticated()

        import re, base64, pathlib

        image_handling = os.environ.get("BRAIN_MESSAGE_IMAGE_MODE", "placeholder").lower()
        save_dir = pathlib.Path("message_images")

        from typing import Tuple
        def process_description(desc: str, message_id: str) -> Tuple[str, List[str]]:
            try:
                if not desc or image_handling == "keep":
                    return desc, []
                attachments: List[str] = []
                # Regex to capture full <img ...> tag with data URI
                img_tag_pattern = re.compile(r"<img[^>]+src=\"(data:image/[^\"]+)\"[^>]*>", re.IGNORECASE)
                # Iterate over unique matches to avoid double work
                matches = list(img_tag_pattern.finditer(desc))
                if not matches:
                    # Additional heuristic: very long base64-looking token inside quotes followed by </img>
                    # (legacy format noted by user sample). Replace with placeholder.
                    heuristic_pattern = re.compile(r"([A-Za-z0-9+/]{500,}={0,2})\"\s*</img>")
                    if image_handling != "keep" and heuristic_pattern.search(desc):
                        placeholder = "[Embedded image removed - large base64 sequence truncated]"
                        return heuristic_pattern.sub(placeholder + "</img>", desc), []
                    return desc, []

                # Ensure save directory exists only if we will store something
                if image_handling == "placeholder" and not save_dir.exists():
                    try:
                        save_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        self.log(f"Could not create image save directory: {e}", "WARNING")

                new_desc = desc
                for idx, match in enumerate(matches, start=1):
                    data_uri = match.group(1)  # data:image/...;base64,XXXX
                    if not data_uri.lower().startswith("data:image"):
                        continue
                    # Split header and base64 payload
                    if "," not in data_uri:
                        continue
                    header, b64_data = data_uri.split(",", 1)
                    mime_part = header.split(";")[0]  # data:image/png
                    ext = "png"
                    if "/" in mime_part:
                        ext = mime_part.split("/")[1]
                    safe_ext = (ext or "img").split("?")[0]
                    placeholder_text = "[Embedded image]"
                    if image_handling == "ignore":
                        replacement = f"[Image removed: {safe_ext}]"
                    elif image_handling == "placeholder":
                        # Try decode & save
                        file_name = f"{message_id}_{idx}.{safe_ext}"
                        file_path = save_dir / file_name
                        try:
                            # Guard extremely large strings (>5MB ~ 6.7M base64 chars) to avoid memory blow
                            if len(b64_data) > 7_000_000:
                                raise ValueError("Image too large to decode safely")
                            with open(file_path, "wb") as f:
                                f.write(base64.b64decode(b64_data))
                            attachments.append(str(file_path))
                            replacement = f"[Image extracted -> {file_path}]"
                        except Exception as e:
                            self.log(f"Failed to decode embedded image in message {message_id}: {e}", "WARNING")
                            replacement = "[Image extraction failed - content omitted]"
                    else:  # keep
                        replacement = placeholder_text  # shouldn't be used since early return, but safe
                    # Replace only the matched tag (not global) – use re.sub with count=1 on substring slice
                    # Safer to operate on new_desc using the exact matched string
                    original_tag = match.group(0)
                    new_desc = new_desc.replace(original_tag, replacement, 1)
                return new_desc, attachments
            except UnicodeEncodeError as ue:
                self.log(f"Unicode encoding error in process_description: {ue}", "WARNING")
                return desc, []
            except Exception as e:
                self.log(f"Error in process_description: {e}", "WARNING")
                return desc, []

        try:
            params = {}
            if limit is not None:
                params['limit'] = limit
            if offset > 0:
                params['offset'] = offset

            response = await self._request('get', f"{self.base_url}/users/self/messages", params=params)
            response.raise_for_status()
            data = response.json()

            # Post-process results for image handling. base64 decode (multi-MB per
            # image) + file writes + regex are CPU/disk work — run in the executor
            # so they never stall the event loop shared by all MCP clients.
            results = data.get('results', [])

            def _sanitize_all():
                for msg in results:
                    try:
                        desc = msg.get('description')
                        processed_desc, attachments = process_description(desc, msg.get('id', 'msg'))
                        if attachments or desc != processed_desc:
                            msg['description'] = processed_desc
                            if attachments:
                                msg['extracted_images'] = attachments
                            else:
                                # If changed but no attachments (ignore mode) mark sanitized
                                msg['sanitized'] = True
                    except UnicodeEncodeError as ue:
                        self.log(f"Unicode encoding error sanitizing message {msg.get('id')}: {ue}", "WARNING")
                        # Keep original description if encoding fails
                        continue
                    except Exception as inner_e:
                        self.log(f"Failed to sanitize message {msg.get('id')}: {inner_e}", "WARNING")

            # Default executor (not self._executor): keeps CPU/disk sanitize work
            # from competing with other clients' HTTP calls for pool threads.
            await asyncio.get_running_loop().run_in_executor(None, _sanitize_all)
            data['results'] = results
            data['image_handling'] = image_handling
            return data
        except UnicodeEncodeError as ue:
            self.log(f"Failed to get messages due to encoding error: {str(ue)}", "ERROR")
            raise
        except Exception as e:
            self.log(f"Failed to get messages: {str(e)}", "ERROR")
            raise

    async def get_glossary_terms(self, email: str, password: str) -> List[Dict[str, str]]:
        """Get glossary terms from forum."""
        try:
            return await forum_client.get_glossary_terms(email, password)
        except Exception as e:
            self.log(f"Failed to get glossary terms: {str(e)}", "ERROR")
            raise

    async def search_forum_posts(self, email: str, password: str, search_query: str, 
                                 max_results: int = 50) -> Dict[str, Any]:
        """Search forum posts."""
        try:
            return await forum_client.search_forum_posts(email, password, search_query, max_results)
        except Exception as e:
            self.log(f"Failed to search forum posts: {str(e)}", "ERROR")
            raise

    async def read_forum_post(self, email: str, password: str, article_id: str, 
                              include_comments: bool = True) -> Dict[str, Any]:
        """Get forum post."""
        try:
            return await forum_client.read_full_forum_post(email, password, article_id, include_comments)
        except Exception as e:
            self.log(f"Failed to read forum post: {str(e)}", "ERROR")
            raise
    
    async def get_alpha_yearly_stats(self, alpha_id: str) -> Dict[str, Any]:
        """Get yearly statistics for an alpha with retry logic."""
        await self.ensure_authenticated()
        
        max_retries = 5
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.log(f"Attempting to get yearly stats for alpha {alpha_id} (attempt {attempt + 1}/{max_retries})", "INFO")
                
                response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/recordsets/yearly-stats")
                response.raise_for_status()

                # Check if response has content
                text = (response.text or "").strip()
                if not text:
                    if attempt < max_retries - 1:
                        self.log(f"Empty yearly stats response for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5  # Exponential backoff
                        continue
                    else:
                        self.log(f"Empty yearly stats response after {max_retries} attempts for {alpha_id}", "WARNING")
                        return {}
                
                try:
                    yearly_stats = response.json()
                    if yearly_stats:
                        self.log(f"Successfully retrieved yearly stats for alpha {alpha_id}", "SUCCESS")
                        return yearly_stats
                    else:
                        if attempt < max_retries - 1:
                            self.log(f"Empty yearly stats JSON for {alpha_id}, retrying in {retry_delay} seconds...", "WARNING")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 1.5
                            continue
                        else:
                            self.log(f"Empty yearly stats JSON after {max_retries} attempts for {alpha_id}", "WARNING")
                            return {}
                            
                except Exception as parse_err:
                    if attempt < max_retries - 1:
                        self.log(f"Yearly stats JSON parse failed for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds...", "WARNING")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5
                        continue
                    else:
                        self.log(f"Yearly stats JSON parse failed for {alpha_id} after {max_retries} attempts: {parse_err}", "WARNING")
                        return {}
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    self.log(f"Failed to get alpha yearly stats for {alpha_id} (attempt {attempt + 1}), retrying in {retry_delay} seconds: {str(e)}", "WARNING")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                else:
                    self.log(f"Failed to get alpha yearly stats for {alpha_id} after {max_retries} attempts: {str(e)}", "ERROR")
                    raise
        
        # This should never be reached, but just in case
        return {}
        
    async def _poll_correlation(self, alpha_id: str, kind: str, max_wait: float) -> Dict[str, Any]:
        """Poll GET /alphas/{id}/correlations/{kind} ("prod" or "self") until it
        settles or max_wait runs out, keeping the three outcomes apart:

        - DONE: a 2xx without Retry-After and with a body; `max` read from the
          top-level `max`, then schema.max, then the records' correlation column.
        - PENDING: still computing (2xx + Retry-After, or an empty body) when the
          wait budget ran out — call again later, this is NOT a failure.
        - ERROR: a 4xx/5xx (after _request's own retries), non-JSON body, or a
          body carrying only an error message.
        """
        url = f"{self.base_url}/alphas/{alpha_id}/correlations/{kind}"
        deadline = time.monotonic() + max(0.0, min(float(max_wait or 0), 300.0))
        while True:
            try:
                resp = await self._request('get', url)
            except requests.RequestException as e:
                return {"status": "ERROR", "error": f"network error: {e}"}
            if resp.status_code >= 400:
                return {"status": "ERROR", "http_status": resp.status_code,
                        "error": _http_error_detail(resp)}
            ra = _retry_after_seconds(resp)
            text = (resp.text or "").strip()
            if text and "Retry-After" not in resp.headers:
                try:
                    data = resp.json()
                except ValueError:
                    return {"status": "ERROR", "error": "non-JSON body", "body": text[:300]}
                if isinstance(data, dict) and data:
                    stats = _correlation_stats(data)
                    if stats is None and (data.get("message") or data.get("detail") or data.get("error")):
                        return {"status": "ERROR",
                                "error": str(data.get("message") or data.get("detail") or data.get("error"))[:300]}
                    return {"status": "DONE", "data": data,
                            "max": (stats or {}).get("max"), "min": (stats or {}).get("min")}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"status": "PENDING", "retry_after_seconds": ra or 10.0}
            await asyncio.sleep(max(1.0, min(ra or 5.0, remaining)))

    async def get_production_correlation(self, alpha_id: str, max_wait: float = 100) -> Dict[str, Any]:
        """Raw production-correlation payload. {} = platform still computing
        (retry later); raises on a real failure. ProdMemo relies on this contract."""
        await self.ensure_authenticated()
        r = await self._poll_correlation(alpha_id, "prod", max_wait)
        if r["status"] == "ERROR":
            raise Exception(f"prod correlation failed for {alpha_id}: {r.get('error')}")
        return r.get("data") or {}

    async def get_self_correlation(self, alpha_id: str, max_wait: float = 100) -> Dict[str, Any]:
        """Raw self-correlation payload; same contract as get_production_correlation."""
        await self.ensure_authenticated()
        r = await self._poll_correlation(alpha_id, "self", max_wait)
        if r["status"] == "ERROR":
            raise Exception(f"self correlation failed for {alpha_id}: {r.get('error')}")
        return r.get("data") or {}

    async def _record_platform_corr(self, alpha_id: str, kind: str, max_v: Any,
                                    min_v: Any = None, source: str = "platform") -> None:
        """Write an officially measured correlation back into ProdMemo, so it becomes
        a reference point. Best effort: ProdMemo's database being down must never
        break the platform tool that measured the value."""
        try:
            await prodmemo_client.record_platform_corr(alpha_id, kind, max_v, min_v, source)
        except Exception as e:
            self.log(f"ProdMemo write-back skipped for {alpha_id} ({kind}): {e}", "WARNING")

    async def check_correlation(self, alpha_id: str, correlation_type: str = "both",
                                threshold: float = 0.7, max_wait: float = 60,
                                include_data: bool = False) -> Dict[str, Any]:
        """Prod and/or self correlation, polled concurrently within max_wait.

        Each type reports status DONE / PENDING / ERROR; all_passed is only a
        boolean once every requested type is DONE (None otherwise). Measured
        values are written back into ProdMemo.
        """
        await self.ensure_authenticated()
        aliases = {"production": "prod", "prod": "prod", "self": "self"}
        ctype = (correlation_type or "both").lower()
        if ctype == "both":
            kinds = ["prod", "self"]
        elif ctype in aliases:
            kinds = [aliases[ctype]]
        else:
            raise ValueError("correlation_type must be 'prod'/'production', 'self' or 'both'")

        polled = await asyncio.gather(*[self._poll_correlation(alpha_id, k, max_wait) for k in kinds])
        checks: Dict[str, Any] = {}
        for kind, r in zip(kinds, polled):
            name = "production" if kind == "prod" else "self"
            entry: Dict[str, Any] = {"status": r["status"]}
            if r["status"] == "DONE":
                mx = r.get("max")
                entry["max_correlation"] = mx
                entry["passes_check"] = (mx < threshold) if mx is not None else None
                entry["top"] = _correlation_top_rows(r.get("data") or {}, 3)
                if include_data:
                    entry["correlation_data"] = r.get("data")
                if mx is not None:
                    await self._record_platform_corr(alpha_id, kind, mx, r.get("min"))
            elif r["status"] == "PENDING":
                entry["retry_after_seconds"] = r.get("retry_after_seconds")
                entry["note"] = "Platform is still computing; call again later (not a failure)."
            else:
                entry.update({k: r[k] for k in ("error", "http_status", "body") if k in r})
            checks[name] = entry

        statuses = [c["status"] for c in checks.values()]
        status = "ERROR" if "ERROR" in statuses else ("PENDING" if "PENDING" in statuses else "DONE")
        passes = [c.get("passes_check") for c in checks.values()]
        all_passed = all(passes) if status == "DONE" and None not in passes else None
        return {"alpha_id": alpha_id, "threshold": threshold, "status": status,
                "all_passed": all_passed, "checks": checks}

    async def get_submission_check(self, alpha_id: str, max_wait: float = 60) -> Dict[str, Any]:
        """Platform-authoritative pre-submission check (GET /alphas/{id}/check).

        Polls the Retry-After protocol within max_wait. PROD_CORRELATION often
        comes back as result=ERROR while the platform is busy; in that case the
        dedicated prod-correlation endpoint is tried as a fallback so the answer
        is not lost. A numeric PROD_CORRELATION is written back into ProdMemo.
        """
        await self.ensure_authenticated()
        url = f"{self.base_url}/alphas/{alpha_id}/check"
        deadline = time.monotonic() + max(0.0, min(float(max_wait or 0), 300.0))
        data: Any = None
        while True:
            resp = await self._request('get', url)
            if resp.status_code >= 400:
                return {"alpha_id": alpha_id, "status": "ERROR", "http_status": resp.status_code,
                        "error": _http_error_detail(resp)}
            ra = _retry_after_seconds(resp)
            text = (resp.text or "").strip()
            if text and "Retry-After" not in resp.headers:
                try:
                    data = resp.json()
                except ValueError:
                    return {"alpha_id": alpha_id, "status": "ERROR", "error": "non-JSON body",
                            "body": text[:300]}
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"alpha_id": alpha_id, "status": "PENDING", "retry_after_seconds": ra or 10.0,
                        "note": "Platform is still running the checks; call again later."}
            await asyncio.sleep(max(1.0, min(ra or 5.0, remaining)))

        is_ = (data or {}).get("is") or {}
        checks = [c for c in (is_.get("checks") or []) if isinstance(c, dict)]
        rows = [{k: c.get(k) for k in ("name", "result", "value", "limit") if c.get(k) is not None}
                for c in checks]

        def names(result: str) -> List[str]:
            return [c.get("name") for c in checks if c.get("result") == result]

        failed, pending, errored = names("FAIL"), names("PENDING"), names("ERROR")
        prod = next((c for c in checks if c.get("name") == "PROD_CORRELATION"), None)
        out: Dict[str, Any] = {
            "alpha_id": alpha_id,
            "status": "PENDING" if pending else "DONE",
            "all_passed": (not failed and not pending and not errored) if checks else None,
            "failed": failed, "pending": pending, "errored": errored,
            "checks": rows,
        }
        self_corr = is_.get("selfCorrelation")
        if isinstance(self_corr, dict) and self_corr.get("max") is not None:
            out["self_correlation_max"] = self_corr.get("max")

        prod_value = _finite(prod.get("value")) if prod else None
        if prod and prod.get("result") in ("PASS", "FAIL") and prod_value is not None:
            out["prod_correlation"] = prod_value
            await self._record_platform_corr(alpha_id, "prod", prod_value, source="platform_check")
        elif prod is None or prod.get("result") == "ERROR":
            # /check lost the prod value (busy platform) — ask the dedicated endpoint.
            fallback = await self.check_correlation(alpha_id, "prod", max_wait=min(max_wait, 60))
            out["prod_fallback"] = fallback["checks"].get("production")
            if (out["prod_fallback"] or {}).get("status") == "DONE" and errored == ["PROD_CORRELATION"]:
                passes = out["prod_fallback"].get("passes_check")
                if passes is not None:
                    out["all_passed"] = bool(passes) and not failed and not pending
        return out

    async def set_alpha_properties(
        self,
        alpha_id: str,
        name: Optional[str] = None,
        color: Optional[str] = None,
        category: Optional[str] = None,
        regular_desc: Optional[str] = None,
        selection_desc: Optional[str] = None,
        combo_desc: Optional[str] = None,
        osmosis_points: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update alpha properties (name, color, tags, descriptions, category, osmosis points)."""
        await self.ensure_authenticated()

        try:
            if osmosis_points is not None and not (1 <= osmosis_points <= 100000):
                raise ValueError(f"osmosis_points must be between 1 and 100000, got {osmosis_points}")

            option_map = {
                "name": name,
                "color": color,
                "tags": tags,
                "category": category,
                "osmosisPoints": osmosis_points,
                "regular": {"description": regular_desc} if regular_desc is not None else None,
                "selection": {"description": selection_desc} if selection_desc is not None else None,
                "combo": {"description": combo_desc} if combo_desc is not None else None,
            }
            data = {k: v for k, v in option_map.items() if v is not None}

            response = await self._request('patch', f"{self.base_url}/alphas/{alpha_id}", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to set alpha properties: {str(e)}", "ERROR")
            raise

    async def get_record_sets(self, alpha_id: str) -> Dict[str, Any]:
        """List available record sets for an alpha."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/recordsets")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get record sets: {str(e)}", "ERROR")
            raise

    async def get_record_set_data(self, alpha_id: str, record_set_name: str) -> Dict[str, Any]:
        """Get data from a specific record set."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/recordsets/{record_set_name}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get record set data: {str(e)}", "ERROR")
            raise

    async def get_user_activities(self, user_id: str, grouping: Optional[str] = None) -> Dict[str, Any]:
        """Get user activity diversity data."""
        await self.ensure_authenticated()
        
        try:
            params = {}
            if grouping:
                params['grouping'] = grouping
            
            response = await self._request('get', f"{self.base_url}/users/{user_id}/activities", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get user activities: {str(e)}", "ERROR")
            raise

    async def get_pyramid_multipliers(self) -> Dict[str, Any]:
        """Get current pyramid multipliers showing BRAIN's encouragement levels."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/users/self/activities/pyramid-multipliers")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get pyramid multipliers: {str(e)}", "ERROR")
            raise

    async def get_pyramid_alphas(self, start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get user's current alpha distribution across pyramid categories."""
        await self.ensure_authenticated()
        
        try:
            params = {}
            if start_date:
                params['startDate'] = start_date
            if end_date:
                params['endDate'] = end_date
            
            # Try the user-specific activities endpoint first (like pyramid-multipliers)
            response = await self._request('get', f"{self.base_url}/users/self/activities/pyramid-alphas", params=params)

            # If that fails, try alternative endpoints
            if response.status_code == 404:
                # Try alternative endpoint structure
                response = await self._request('get', f"{self.base_url}/users/self/pyramid/alphas", params=params)

                if response.status_code == 404:
                    # Try yet another alternative
                    response = await self._request('get', f"{self.base_url}/activities/pyramid-alphas", params=params)
                    
                    if response.status_code == 404:
                        # Return an informative error with what we tried
                        return {
                            "error": "Pyramid alphas endpoint not found",
                            "tried_endpoints": [
                                "/users/self/activities/pyramid-alphas",
                                "/users/self/pyramid/alphas", 
                                "/activities/pyramid-alphas",
                                "/pyramid/alphas"
                            ],
                            "suggestion": "This endpoint may not be available in the current API version"
                        }
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get pyramid alphas: {str(e)}", "ERROR")
            raise
            
    async def get_user_competitions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get list of competitions that the user is participating in."""
        await self.ensure_authenticated()
        
        try:
            if not user_id:
                # Get current user ID if not specified
                user_response = await self._request('get', f"{self.base_url}/users/self")
                if user_response.status_code == 200:
                    user_data = user_response.json()
                    user_id = user_data.get('id')
                else:
                    user_id = 'self'

            response = await self._request('get', f"{self.base_url}/users/{user_id}/competitions")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get user competitions: {str(e)}", "ERROR")
            raise
            
    async def get_competition_details(self, competition_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific competition."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/competitions/{competition_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get competition details: {str(e)}", "ERROR")
            raise
            
    async def get_competition_agreement(self, competition_id: str) -> Dict[str, Any]:
        """Get the rules, terms, and agreement for a specific competition."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/competitions/{competition_id}/agreement")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get competition agreement: {str(e)}", "ERROR")
            raise

    async def get_platform_setting_options(self) -> Dict[str, Any]:
        """Get available instrument types, regions, delays, and universes."""
        await self.ensure_authenticated()
        
        try:
            # Use OPTIONS method on simulations endpoint to get configuration options
            response = await self._request('options', f"{self.base_url}/simulations")
            response.raise_for_status()
            
            # Parse the settings structure from the response
            settings_data = response.json()
            settings_options = settings_data['actions']['POST']['settings']['children']
            
            # Extract instrument configuration options
            instrument_type_data = {}
            region_data = {}
            universe_data = {}
            delay_data = {}
            neutralization_data = {}
            
            # Parse each setting type
            for key, setting in settings_options.items():
                if setting['type'] == 'choice':
                    if setting['label'] == 'Instrument type':
                        instrument_type_data = setting['choices']
                    elif setting['label'] == 'Region':
                        region_data = setting['choices']['instrumentType']
                    elif setting['label'] == 'Universe':
                        universe_data = setting['choices']['instrumentType']
                    elif setting['label'] == 'Delay':
                        delay_data = setting['choices']['instrumentType']
                    elif setting['label'] == 'Neutralization':
                        neutralization_data = setting['choices']['instrumentType']
            
            # Build comprehensive instrument options
            data_list = []
            
            for instrument_type in instrument_type_data:
                for region in region_data[instrument_type['value']]:
                    for delay in delay_data[instrument_type['value']]['region'][region['value']]:
                        row = {
                            'InstrumentType': instrument_type['value'],
                            'Region': region['value'],
                            'Delay': delay['value']
                        }
                        row['Universe'] = [
                            item['value'] for item in universe_data[instrument_type['value']]['region'][region['value']]
                        ]
                        row['Neutralization'] = [
                            item['value'] for item in neutralization_data[instrument_type['value']]['region'][region['value']]
                        ]
                        data_list.append(row)
            
            # Return structured data
            return {
                'instrument_options': data_list,
                'total_combinations': len(data_list),
                'instrument_types': [item['value'] for item in instrument_type_data],
                'regions_by_type': {
                    item['value']: [r['value'] for r in region_data[item['value']]]
                    for item in instrument_type_data
                }
            }
            
        except Exception as e:
            self.log(f"Failed to get instrument options: {str(e)}", "ERROR")
            raise
            
    async def performance_comparison(self, alpha_id: str, team_id: Optional[str] = None, 
                                     competition: Optional[str] = None) -> Dict[str, Any]:
        """Get performance comparison data for an alpha."""
        await self.ensure_authenticated()
        
        try:
            params = {"teamId": team_id, "competition": competition}
            params = {k: v for k, v in params.items() if v is not None}
            
            response = await self._request('get', f"{self.base_url}/alphas/{alpha_id}/performance-comparison", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get performance comparison: {str(e)}", "ERROR")
            raise
            
    # --- Helper function for data flattening ---
    
    async def expand_nested_data(self, data: List[Dict[str, Any]], preserve_original: bool = True) -> List[Dict[str, Any]]:
        """Flatten complex nested data structures into tabular format."""
        try:
            df = pd.json_normalize(data, sep='_')
            if preserve_original:
                original_df = pd.DataFrame(data)
                df = pd.concat([original_df, df], axis=1)
                df = df.loc[:,~df.columns.duplicated()]
            return df.to_dict(orient='records')
        except Exception as e:
            self.log(f"Failed to expand nested data: {str(e)}", "ERROR")
            raise
            
    # --- New documentation endpoint ---
    
    async def get_documentation_page(self, page_id: str) -> Dict[str, Any]:
        """Retrieve detailed content of a specific documentation page/article."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/tutorial-pages/{page_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get documentation page: {str(e)}", "ERROR")
            raise

brain_client = BrainApiClient()

# --- Configuration Management ---

def _resolve_config_path(for_write: bool = False) -> str:
    """
    Resolve the configuration file path.
    
    Checks for a file specified by the MCP_CONFIG_FILE environment variable,
    then falls back to ~/.brain_mcp_config.json. If for_write is True,
    it ensures the directory exists.
    """
    if 'MCP_CONFIG_FILE' in os.environ:
        return os.environ['MCP_CONFIG_FILE']
    
    config_path = Path(__file__).parent / "user_config.json"
    
    if for_write:
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
        except (IOError, OSError) as e:
            logger.warning(f"Could not create config directory {config_path.parent}: {e}")
            # Fallback to a temporary file if home is not writable
            import tempfile
            return tempfile.NamedTemporaryFile(delete=False).name
            
    return str(config_path)

def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    config_file = _resolve_config_path()
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading config file {config_file}: {e}")
    return {}

def save_config(config: Dict[str, Any]):
    """Save configuration to file using the resolved config path.
    
    This function now uses the write-enabled path resolver to handle
    cases where the default home directory is not writable.
    """
    config_file = _resolve_config_path(for_write=True)
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except IOError as e:
        logger.error(f"Error saving config file to {config_file}: {e}")

# --- MCP Tool Definitions ---

mcp = FastMCP(
    "brain-platform-mcp",
    "A server for interacting with the WorldQuant BRAIN platform",
    host="0.0.0.0",
    port="8761"
)

@mcp.tool()
async def authenticate(email: Optional[str] = "", password: Optional[str] = "") -> Dict[str, Any]:
    """
    🔐 Verify / refresh the BRAIN login state.

    Login is managed centrally by the credd daemon (creds-daemon) — this MCP no
    longer logs in with a password itself. Calling this tool is optional: every
    other tool self-heals its login automatically. Use it to check that the
    platform connection is healthy.

    Args:
        email: Ignored (kept for backward compatibility; credentials live in credd)
        password: Ignored (kept for backward compatibility; credentials live in credd)

    Returns:
        Authentication status from the credd-backed session
    """
    try:
        return await brain_client.authenticate(email or "", password or "")
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def manage_config(action: str = "get", settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    🔧 Manage configuration settings - get or update configuration.
    
    Args:
        action: Action to perform ("get" to retrieve config, "set" to update config)
        settings: Configuration settings to update (required when action="set")
    
    Returns:
        Current or updated configuration including authentication status
    """
    if action == "get":
        config = load_config()
        auth_status = await brain_client.get_authentication_status()
        
        return {
            "config": config,
            "auth_status": auth_status,
            "is_authenticated": await brain_client.is_authenticated()
        }
    
    elif action == "set":
        if settings is None:
            return {"error": "Settings parameter is required when action='set'"}
        
        config = load_config()
        config.update(settings)
        save_config(config)
        return config
    
    else:
        return {"error": f"Invalid action '{action}'. Use 'get' or 'set'."}

# --- Simulation Tools ---

@mcp.tool()
async def create_simulation(
    type: str = "REGULAR",
    instrument_type: str = "EQUITY",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: float = 0.0,
    neutralization: str = "NONE",
    truncation: float = 0.0,
    test_period: str = "P0Y0M",
    unit_handling: str = "VERIFY",
    nan_handling: str = "OFF",
    language: str = "FASTEXPR",
    visualization: bool = True,
    regular: Optional[str] = None,
    combo: Optional[str] = None,
    selection: Optional[str] = None,
    pasteurization: str = "ON",
    max_trade: str = "OFF",
    selection_handling: str = "POSITIVE",
    selection_limit: int = 1000,
    component_activation: str = "IS",
    max_position: str = "OFF",
    lookback: Optional[int] = None,
    simulation_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    🚀 Submit a new simulation on BRAIN platform (returns immediately).

    This tool submits a simulation with your alpha code and returns right away with
    a simulation_id and a progress_url — it does NOT wait for the simulation to
    finish (simulations typically take 1-5 minutes). To get progress and the final
    result, call check_simulation_progress with the returned progress_url. You can
    do other work between checks.

    Args:
        type: Simulation type ("REGULAR" or "SUPER")
        instrument_type: Type of instruments (e.g., "EQUITY")
        region: Market region (e.g., "USA")
        universe: Universe of stocks (e.g., "TOP3000")
        delay: Data delay (0 or 1)
        decay: Decay value for the simulation
        neutralization: Neutralization method
        truncation: Truncation value
        test_period: Test period (e.g., "P0Y0M"). Ignored when language="PYTHON".
        unit_handling: Unit handling method. Ignored when language="PYTHON".
        nan_handling: NaN handling method. Ignored when language="PYTHON".
        language: Expression language. "FASTEXPR" (default) or "PYTHON".
        visualization: Enable visualization
        regular: Regular simulation code (for REGULAR type). For language="PYTHON" pass a Python source string.
        combo: Combo code (for SUPER type)
        selection: Selection code (for SUPER type)
        lookback: PYTHON-only lookback window (required when language="PYTHON", ignored otherwise).
        simulation_mode: "QUICK" or "FULL" (default None = platform default, FULL).
            QUICK = fast feedback for rapid iteration: returns core metrics only
            (PnL, sub-universe PnL, weight concentration, fitness, IS ladder,
            Sharpe, returns, turnover), skips visualizations and the Theme /
            Competition / correlation checks, and the alpha is NOT directly
            submittable. QUICK forces visualization=False. Use FULL for
            near-submission-ready alphas that need all checks.

    Returns:
        {"status": "SUBMITTED", "simulation_id": ..., "progress_url": ...} — poll
        check_simulation_progress(progress_url) for progress and the final result.
    """
    try:
        if (language or "").upper() == "PYTHON" and lookback is None:
            return {"error": "lookback is required when language='PYTHON'"}
        try:
            simulation_mode, visualization = _normalize_simulation_mode(simulation_mode, visualization)
        except ValueError as e:
            return {"error": str(e)}

        settings = SimulationSettings(
            instrumentType=instrument_type,
            region=region,
            universe=universe,
            delay=delay,
            decay=decay,
            neutralization=neutralization,
            truncation=truncation,
            testPeriod=test_period,
            unitHandling=unit_handling,
            nanHandling=nan_handling,
            language=language,
            visualization=visualization,
            pasteurization=pasteurization,
            maxTrade=max_trade,
            selectionHandling=selection_handling,
            selectionLimit=selection_limit,
            componentActivation=component_activation,
            maxPosition=max_position,
            lookback=lookback,
            simulationMode=simulation_mode,
        )

        sim_data = SimulationData(
            type=type,
            settings=settings,
            regular=regular,
            combo=combo,
            selection=selection
        )

        return await brain_client.create_simulation(sim_data)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def check_simulation_progress(progress_url: str, wait_seconds: float = 0,
                                    compact: bool = True) -> Dict[str, Any]:
    """
    ⏳ Check the progress / result of a submitted simulation — single OR multi.

    Use this after create_simulation or create_multi_simulation returns
    {"status": "SUBMITTED", "progress_url": ...}. The URL type is detected
    automatically:
    - single simulation: progress (0.0-1.0) while running; the alpha when
      finished; BRAIN's error message if the simulation failed.
    - multi-simulation: per-child status with completed_children/total_children
      while running; once ALL children finish, alpha_results with one entry per
      child.
    - RAA (create_raa_simulation): once finished, the parent_alpha_id plus one
      compact metric row per region child (same shape as get_raa_alpha).

    Args:
        progress_url: The progress_url returned by create_simulation /
            create_multi_simulation / create_raa_simulation
            (e.g. "https://api.worldquantbrain.com/simulations/<id>")
        wait_seconds: Optional bounded wait before answering (0 = check once and
            return immediately; max 120). Use e.g. 30-60 to block briefly when you
            have nothing else to do.
        compact: True (default) = one short row per finished alpha: id, ops
            (operator count), sharpe, fitness, turnover, margin_bps,
            robust_sharpe, sub_sharpe, y2_sharpe, cluster, fails (failed checks),
            set (universe/decay/neutralization/truncation/maxTrade) and expr.
            False = the full alpha object (same shape as get_alpha_details).

    Returns:
        {"status": "RUNNING", ...} with progress info while running; final results
        when finished; {"status": "ERROR"/..., "message": ...} on failure.
    """
    try:
        if not progress_url or "worldquantbrain.com" not in str(progress_url):
            return {"error": "progress_url must be the simulations URL returned by create_simulation"}
        return await brain_client.check_simulation_progress(progress_url, wait_seconds, compact)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def create_raa_simulation(
    regular: str,
    universe: str = "MEDIUM",
    decay: float = 10,
    neutralization: str = "SLOW_AND_FAST",
    truncation: float = 0.08,
    pasteurization: str = "ON",
    unit_handling: str = "VERIFY",
    nan_handling: str = "OFF",
    max_trade: str = "OFF",
    max_position: str = "OFF",
    visualization: bool = False,
    simulation_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    🌍 Submit a Region Agnostic Alpha (RAA) simulation — one expression, all regions.

    A single RAA simulation runs the SAME expression in GLB / USA / ASI / EUR at
    once, producing an RA_PARENT alpha with up to 4 RA_CHILD alphas. Returns
    immediately; poll check_simulation_progress(progress_url), which detects RAA
    and reports the parent id plus one metric row per region child.

    Platform rules enforced here (violations fail the simulation outright):
    - region is always "ALL", delay always 1, instrumentType always EQUITY.
    - universe must be an RAA pseudo-universe: LARGE / MEDIUM / SMALL. Per region:
      LARGE  -> ASI MINVOL1M / EUR TOP2500 / GLB MINVOL1M / USA TOP3000
      MEDIUM -> ASI MINVOL10M / EUR TOP1200 / GLB MINVOL10M / USA TOP2000
      SMALL  -> ASI TOP500 / EUR TOP800 / GLB TOPDIV3000 / USA TOP1000
    - max_trade and max_position cannot both be "ON" (ASI children are forced to
      maxTrade by the platform anyway).

    Quota: one RAA consumes 4 concurrent simulation slots but counts as a single
    submission. Submitting requires >=2 children passing all checks, and at least
    one of those children passing PROD correlation.

    Args:
        regular: FASTEXPR alpha expression. Every data field used must exist in at
            least 2 of the 4 regions (fields tagged region "ALL" on the platform);
            a field present in only one region cannot be an RAA.
        universe: "LARGE", "MEDIUM" or "SMALL" (default "MEDIUM").
        decay / neutralization / truncation / pasteurization / unit_handling /
        nan_handling / max_trade / max_position / visualization: same meaning as in
        create_simulation; one setting applies to all four regions.
        simulation_mode: "QUICK" or "FULL" (default None = platform default). See
            create_simulation; QUICK forces visualization=False and is not
            directly submittable.

    Returns:
        {"status": "SUBMITTED", "simulation_id": ..., "progress_url": ...} — poll
        check_simulation_progress(progress_url); or {"status": "RATE_LIMITED", ...}.
    """
    try:
        if (universe or "").upper() not in RAA_UNIVERSES:
            return {"error": f"RAA universe must be one of {RAA_UNIVERSES}, got '{universe}'"}
        if (max_trade or "").upper() == "ON" and (max_position or "").upper() == "ON":
            return {"error": "maxTrade and maxPosition cannot both be ON for an RAA simulation"}
        if not regular:
            return {"error": "regular (the alpha expression) is required"}
        try:
            simulation_mode, visualization = _normalize_simulation_mode(simulation_mode, visualization)
        except ValueError as e:
            return {"error": str(e)}

        settings = SimulationSettings(
            instrumentType="EQUITY",
            region="ALL",
            universe=universe.upper(),
            delay=1,
            decay=decay,
            neutralization=neutralization,
            truncation=truncation,
            pasteurization=pasteurization,
            unitHandling=unit_handling,
            nanHandling=nan_handling,
            language="FASTEXPR",
            visualization=visualization,
            testPeriod=None,  # RAA payload carries no testPeriod
            maxTrade=max_trade,
            maxPosition=max_position,
            simulationMode=simulation_mode,
        )
        sim_data = SimulationData(type="REGION_AGNOSTIC", settings=settings, regular=regular)
        return await brain_client.create_simulation(sim_data)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_raa_alpha(parent_alpha_id: str) -> Dict[str, Any]:
    """
    🌍 Read an existing RAA parent alpha: settings, expression and per-region child metrics.

    Use this for RA_PARENT alphas from earlier runs (check_simulation_progress
    already returns this shape for a freshly finished RAA). The parent itself has
    no metrics — this fetches every RA_CHILD and returns one compact row per region
    (sharpe, fitness, turnover, returns, drawdown, margin_bps, 2Y sharpe,
    sub-universe sharpe, FAIL and WARNING check names) plus how many children pass
    every check.

    Args:
        parent_alpha_id: The RA_PARENT alpha id (not a child id).
    """
    try:
        return await brain_client.get_raa_alpha(parent_alpha_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- Alpha and Data Retrieval Tools ---

@mcp.tool()
async def get_alpha_details(alpha_id: str) -> Dict[str, Any]:
    """
    📋 Get detailed information about an alpha.
    
    Args:
        alpha_id: The ID of the alpha to retrieve
    
    Returns:
        Detailed alpha information
    """
    try:
        return await brain_client.get_alpha_details(alpha_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_datasets(
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: str = "false",
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """
    📚 Get available datasets for research.
    
    Use this to discover what data is available for your alpha research.
    
    Args:
        instrument_type: Type of instruments (e.g., "EQUITY")
        region: Market region (e.g., "USA")
        delay: Data delay (0 or 1)
        universe: Universe of stocks (e.g., "TOP3000")
        theme: Theme filter
    
    Returns:
        Available datasets
    """
    try:
        return await brain_client.get_datasets(instrument_type, region, delay, universe, theme, search)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_datafields(
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: str = "false",
    dataset_id: Optional[str] = None,
    data_type: str = "",
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """
    🔍 Get available data fields for alpha construction.
    
    Use this to find specific data fields you can use in your alpha formulas.
    
    Args:
        instrument_type: Type of instruments (e.g., "EQUITY")
        region: Market region (e.g., "USA")
        delay: Data delay (0 or 1)
        universe: Universe of stocks (e.g., "TOP3000")
        theme: Theme filter
        dataset_id: Specific dataset ID to filter by
        data_type: Type of data (e.g., "MATRIX",'VECTOR','GROUP')
        search: Search term to filter fields
    
    Returns:
        Available data fields
    """
    try:
        return await brain_client.get_datafields(instrument_type, region, delay, universe, theme, dataset_id, data_type, search)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_alpha_pnl(alpha_id: str) -> Dict[str, Any]:
    """
    📈 Get PnL (Profit and Loss) data for an alpha.
    
    Args:
        alpha_id: The ID of the alpha
    
    Returns:
        PnL data for the alpha
    """
    try:
        return await brain_client.get_alpha_pnl(alpha_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_user_alphas(
    stage: str = "IS",
    limit: int = 30,
    offset: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    submission_start_date: Optional[str] = None,
    submission_end_date: Optional[str] = None,
    order: Optional[str] = None,
    hidden: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    👤 Get user's alphas with advanced filtering, pagination, and sorting.

    This tool retrieves a list of your alphas, allowing for detailed filtering based on stage,
    creation date, submission date, and visibility. It also supports pagination and custom sorting.

    Args:
        stage (str): The stage of the alphas to retrieve.
            - "IS": In-Sample (alphas that have not been submitted).
            - "OS": Out-of-Sample (alphas that have been submitted).
            Defaults to "IS".
        limit (int): The maximum number of alphas to return in a single request.
            For example, `limit=50` will return at most 50 alphas. Defaults to 30.
        offset (int): The number of alphas to skip from the beginning of the list.
            Used for pagination. For example, `limit=50, offset=50` will retrieve alphas 51-100.
            Defaults to 0.
        start_date (Optional[str]): The earliest creation date for the alphas to be included.
            Filters for alphas created on or after this date.
            Example format: "2023-01-01T00:00:00Z".
        end_date (Optional[str]): The latest creation date for the alphas to be included.
            Filters for alphas created before this date.
            Example format: "2023-12-31T23:59:59Z".
        submission_start_date (Optional[str]): The earliest submission date for the alphas.
            Only applies to "OS" alphas. Filters for alphas submitted on or after this date.
            Example format: "2024-01-01T00:00:00Z".
        submission_end_date (Optional[str]): The latest submission date for the alphas.
            Only applies to "OS" alphas. Filters for alphas submitted before this date.
            Example format: "2024-06-30T23:59:59Z".
        order (Optional[str]): The sorting order for the returned alphas.
            Prefix with a hyphen (-) for descending order.
            Examples: "name" (sort by name ascending), "-dateSubmitted" (sort by submission date descending).
        hidden (Optional[bool]): Filter alphas based on their visibility.
            - `True`: Only return hidden alphas.
            - `False`: Only return non-hidden alphas.
            If not provided, both hidden and non-hidden alphas are returned.

    Returns:
        Dict[str, Any]: A dictionary containing a list of alpha details under the 'results' key,
        along with pagination information. If an error occurs, it returns a dictionary with an 'error' key.
    """
    try:
        return await brain_client.get_user_alphas(
            stage=stage, limit=limit, offset=offset, start_date=start_date,
            end_date=end_date, submission_start_date=submission_start_date,
            submission_end_date=submission_end_date, order=order, hidden=hidden
        )
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def submit_alpha(alpha_id: str) -> Dict[str, Any]:
    """
    📤 Submit an alpha for production.
    
    Use this when your alpha is ready for production deployment.
    
    Args:
        alpha_id: The ID of the alpha to submit
    
    Returns:
        Submission result
    """
    try:
        success = await brain_client.submit_alpha(alpha_id)
        return {"success": success}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def value_factor_trendScore(start_date: str, end_date: str) -> Dict[str, Any]:
    """Compute and return the diversity score for REGULAR alphas in a submission-date window.
    This function calculate the diversity of the users' submission, by checking the diversity, we can have a good understanding on the valuefactor's trend.
    This MCP tool wraps BrainApiClient.value_factor_trendScore and always uses submission dates (OS).

    Inputs:
        - start_date: ISO UTC start datetime (e.g. '2025-08-14T00:00:00Z')
        - end_date: ISO UTC end datetime (e.g. '2025-08-18T23:59:59Z')
        - p_max: optional integer total number of pyramid categories for normalization

    Returns: compact JSON with diversity_score, N, A, P, P_max, S_A, S_P, S_H, per_pyramid_counts
    """
    try:
        return await brain_client.value_factor_trendScore(start_date=start_date, end_date=end_date)
    except Exception as e:
        return {"error": str(e)}

# --- Community and Events Tools ---

@mcp.tool()
async def get_events() -> Dict[str, Any]:
    """
    🏆 Get available events and competitions.
    
    Returns:
        Available events and competitions
    """
    try:
        return await brain_client.get_events()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_leaderboard(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    🏅 Get leaderboard data.
    
    Args:
        user_id: Optional user ID to filter results
    
    Returns:
        Leaderboard data
    """
    try:
        return await brain_client.get_leaderboard(user_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


# --- Forum Tools ---

@mcp.tool()
async def get_operators() -> Dict[str, Any]:
    """
    🔧 Get available operators for alpha creation.
    
    Returns:
        Dictionary containing operators list and count
    """
    try:
        operators = await brain_client.get_operators()
        if isinstance(operators, list):
            return {"results": operators, "count": len(operators)}
        return operators
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def run_selection(
    selection: str,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    selection_limit: int = 1000,
    selection_handling: str = "POSITIVE",
) -> Dict[str, Any]:
    """
    🎯 Run a selection query to filter instruments.
    
    Args:
        selection: Selection criteria
        instrument_type: Type of instruments
        region: Geographic region
        delay: Delay setting
        selection_limit: Maximum number of results
        selection_handling: How to handle selection results
    
    Returns:
        Selection results
    """
    try:
        return await brain_client.run_selection(
            selection, instrument_type, region, delay, selection_limit, selection_handling
        )
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_user_profile(user_id: str = "self") -> Dict[str, Any]:
    """
    👤 Get user profile information.
    
    Args:
        user_id: User ID (default: "self" for current user)
    
    Returns:
        User profile data
    """
    try:
        return await brain_client.get_user_profile(user_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_documentations() -> Dict[str, Any]:
    """
    📚 Get available documentations and learning materials.
    
    Returns:
        List of documentations
    """
    try:
        return await brain_client.get_documentations()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- Message and Forum Tools ---

@mcp.tool()
async def get_messages(limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
    """
    💬 Get messages for the current user with optional pagination.
    
    Args:
        limit: Maximum number of messages to return (e.g., 10 for top 10 messages)
        offset: Number of messages to skip (for pagination)
    
    Returns:
        Messages for the current user, optionally limited by count
    """
    try:
        return await brain_client.get_messages(limit, offset)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_glossary_terms(email: str = "", password: str = "") -> List[Dict[str, str]]:
    """
    📚 Get glossary terms from WorldQuant BRAIN forum.
    
    Note: This uses Playwright and is implemented in forum_functions.py
    
    Args:
        email: Your BRAIN platform email address (optional if in config)
        password: Your BRAIN platform password (optional if in config)
    
    Returns:
        A list of glossary terms with definitions
    """
    try:
        # Login is credd-backed: the browser context reuses brain_client's session
        # cookies, so email/password are legacy pass-throughs and may be empty.
        config = load_config()
        credentials = config.get("credentials", {})
        email = email or credentials.get("email", "")
        password = password or credentials.get("password", "")

        return await brain_client.get_glossary_terms(email, password)
    except Exception as e:
        logger.error(f"Error in get_glossary_terms tool: {e}")
        return [{"error": str(e)}]

@mcp.tool()
async def search_forum_posts(search_query: str, email: str = "", password: str = "", 
                             max_results: int = 50) -> Dict[str, Any]:
    """
    🔍 Search forum posts on WorldQuant BRAIN support site.
    
    Note: This uses Playwright and is implemented in forum_functions.py
    
    Args:
        search_query: Search term or phrase
        email: Your BRAIN platform email address (optional if in config)
        password: Your BRAIN platform password (optional if in config)
        max_results: Maximum number of results to return (default: 50)
    
    Returns:
        Search results with analysis
    """
    try:
        # Login is credd-backed; email/password are legacy pass-throughs.
        config = load_config()
        credentials = config.get("credentials", {})
        email = email or credentials.get("email", "")
        password = password or credentials.get("password", "")

        return await brain_client.search_forum_posts(email, password, search_query, max_results)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def read_forum_post(article_id: str, email: str = "", password: str = "", 
                          include_comments: bool = True) -> Dict[str, Any]:
    """
    📄 Get a specific forum post by article ID.
    
    Note: This uses Playwright and is implemented in forum_functions.py
    
    Args:
        article_id: The article ID to retrieve (e.g., "32984819083415-新人求模板")
        email: Your BRAIN platform email address (optional if in config)
        password: Your BRAIN platform password (optional if in config)
    
    Returns:
        Forum post content with comments
    """
    try:
        # Login is credd-backed; email/password are legacy pass-throughs.
        config = load_config()
        credentials = config.get("credentials", {})
        email = email or credentials.get("email", "")
        password = password or credentials.get("password", "")

        return await brain_client.read_forum_post(email, password, article_id, include_comments)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_alpha_yearly_stats(alpha_id: str) -> Dict[str, Any]:
    """Get yearly statistics for an alpha."""
    try:
        return await brain_client.get_alpha_yearly_stats(alpha_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def check_correlation(alpha_id: str, correlation_type: str = "both", threshold: float = 0.7,
                            max_wait: float = 60, include_data: bool = False) -> Dict[str, Any]:
    """Platform production and/or self correlation of an alpha.

    Each type comes back with status:
      DONE    — max_correlation, passes_check (max < threshold), top 3 correlated alphas
      PENDING — platform still computing when max_wait ran out; call again later
                (retry_after_seconds). NOT a failure.
      ERROR   — the platform really failed (http_status / error given).
    Top-level status is ERROR if any type errored, else PENDING if any is pending,
    else DONE; all_passed is true/false only when DONE, otherwise null.
    Measured values are saved into ProdMemo as reference points.

    Args:
        alpha_id: Alpha to check
        correlation_type: "prod" (or "production"), "self", or "both" (default)
        threshold: Pass threshold (default 0.7)
        max_wait: Seconds to keep polling while the platform computes (default 60, max 300)
        include_data: Also return the raw correlation payloads (large)
    """
    try:
        return await brain_client.check_correlation(alpha_id, correlation_type, threshold,
                                                    max_wait, include_data)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_submission_check(alpha_id: str, max_wait: float = 60) -> Dict[str, Any]:
    """Platform-authoritative pre-submission check (the same checks as the Submit button).

    Returns status DONE/PENDING, all_passed, failed / pending / errored check
    names, and each check's result/value/limit. When PROD_CORRELATION comes back
    as ERROR (the platform is busy), the dedicated prod-correlation endpoint is
    tried and its result reported in prod_fallback. A measured prod correlation
    is saved into ProdMemo.

    Args:
        alpha_id: Alpha to check
        max_wait: Seconds to keep polling while the platform runs the checks (default 60, max 300)
    """
    try:
        return await brain_client.get_submission_check(alpha_id, max_wait)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def set_alpha_properties(
    alpha_id: str,
    name: Optional[str] = None,
    color: Optional[str] = None,
    category: Optional[str] = None,
    regular_desc: Optional[str] = None,
    selection_desc: Optional[str] = None,
    combo_desc: Optional[str] = None,
    osmosis_points: Optional[int] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Update alpha properties (name, color, tags, descriptions, category, osmosis points)."""
    try:
        return await brain_client.set_alpha_properties(
            alpha_id,
            name,
            color,
            category,
            regular_desc,
            selection_desc,
            combo_desc,
            osmosis_points,
            tags,
        )
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_record_sets(alpha_id: str) -> Dict[str, Any]:
    """List available record sets for an alpha."""
    try:
        return await brain_client.get_record_sets(alpha_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_record_set_data(alpha_id: str, record_set_name: str) -> Dict[str, Any]:
    """Get data from a specific record set."""
    try:
        return await brain_client.get_record_set_data(alpha_id, record_set_name)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_user_activities(user_id: str, grouping: Optional[str] = None) -> Dict[str, Any]:
    """Get user activity diversity data."""
    try:
        return await brain_client.get_user_activities(user_id, grouping)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_pyramid_multipliers() -> Dict[str, Any]:
    """Get current pyramid multipliers showing BRAIN's encouragement levels."""
    try:
        return await brain_client.get_pyramid_multipliers()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_pyramid_alphas(start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, Any]:
    """Get user's current alpha distribution across pyramid categories."""
    try:
        return await brain_client.get_pyramid_alphas(start_date, end_date)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
        
@mcp.tool()
async def get_user_competitions(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Get list of competitions that the user is participating in."""
    try:
        return await brain_client.get_user_competitions(user_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_competition_details(competition_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific competition."""
    try:
        return await brain_client.get_competition_details(competition_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_competition_agreement(competition_id: str) -> Dict[str, Any]:
    """Get the rules, terms, and agreement for a specific competition."""
    try:
        return await brain_client.get_competition_agreement(competition_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def get_platform_setting_options() -> Dict[str, Any]:
    """Discover valid simulation setting options (instrument types, regions, delays, universes, neutralization).

    Use this when a simulation request might contain an invalid/mismatched setting. If an AI or user supplies
    incorrect parameters (e.g., wrong region for an instrument type), call this tool to retrieve the authoritative
    option sets and correct the inputs before proceeding.

    Returns:
        A structured list of valid combinations and choice lists to validate or fix simulation settings.
    """
    try:
        return await brain_client.get_platform_setting_options()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
async def performance_comparison(alpha_id: str, team_id: Optional[str] = None, 
                                 competition: Optional[str] = None) -> Dict[str, Any]:
    """Get performance comparison data for an alpha."""
    try:
        return await brain_client.performance_comparison(alpha_id, team_id, competition)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
        
# --- Dataframe Tool ---

@mcp.tool()
async def expand_nested_data(data: List[Dict[str, Any]], preserve_original: bool = True) -> List[Dict[str, Any]]:
    """Flatten complex nested data structures into tabular format."""
    try:
        return await brain_client.expand_nested_data(data, preserve_original)
    except Exception as e:
        return [{"error": f"An unexpected error occurred: {str(e)}"}]
        
# --- Documentation Tool ---

@mcp.tool()
async def get_documentation_page(page_id: str) -> Dict[str, Any]:
    """Retrieve detailed content of a specific documentation page/article."""
    try:
        return await brain_client.get_documentation_page(page_id)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- Advanced Simulation Tools ---

# Per-alpha override keys for create_multi_simulation: tool-style snake_case
# names map to the API's camelCase; camelCase is accepted as-is. language and
# lookback stay batch-level because they change the payload shape.
_MULTI_OVERRIDE_KEYS = {
    "instrument_type": "instrumentType", "region": "region", "universe": "universe",
    "delay": "delay", "decay": "decay", "neutralization": "neutralization",
    "truncation": "truncation", "pasteurization": "pasteurization",
    "unit_handling": "unitHandling", "nan_handling": "nanHandling",
    "max_trade": "maxTrade", "max_position": "maxPosition", "test_period": "testPeriod",
    "visualization": "visualization", "simulation_mode": "simulationMode",
}
_MULTI_OVERRIDE_KEYS.update({v: v for v in list(_MULTI_OVERRIDE_KEYS.values())})

# Operators whose optional parameters must be passed by keyword: a positional
# optional argument makes that child fail, and a failed child cancels the whole
# multi-simulation. Value = (number of positional args, keyword names after them);
# None = variadic (only the keyword `filter` exists). Ported from bqx.py.
_KWONLY_OPERATORS = {
    "tail": (1, ["lower", "upper", "newval"]), "hump": (1, ["hump"]),
    "winsorize": (1, ["std"]), "quantile": (1, ["driver", "sigma"]), "rank": (1, ["rate"]),
    "ts_backfill": (1, ["lookback", "k"]), "ts_rank": (2, ["constant"]),
    "ts_scale": (2, ["constant"]), "ts_quantile": (2, ["driver"]),
    "group_backfill": (3, ["std"]), "ts_poly_regression": (3, ["k"]),
    "ts_regression": (3, ["lag", "rettype"]), "ts_decay_linear": (2, ["dense"]),
    "normalize": (1, ["useStd", "limit"]), "nan_out": (1, ["lower", "upper"]),
    "ts_returns": (2, ["mode"]), "ts_min_max_diff": (2, ["f"]), "ts_min_max_cps": (2, ["f"]),
    "ts_moment": (2, ["k"]), "kth_element": (2, ["k", "ignore"]), "ts_weighted_decay": (1, ["k"]),
    "hump_decay": (1, ["p"]), "reduce_avg": (1, ["threshold"]),
    "scale": (1, ["scale", "longscale", "shortscale"]),
    "ts_target_tvr_decay": (1, ["lambda_min", "lambda_max", "target_tvr"]),
    "ts_target_tvr_hump": (1, ["lambda_min", "lambda_max", "target_tvr"]),
    "bucket": (1, ["range", "buckets", "skipBoth", "NaNGroup"]),
}


def _lint_expression(expr: str) -> List[str]:
    """Cheap pre-flight check: unbalanced parentheses and optional operator
    arguments passed positionally. Returns a list of problems (empty = OK)."""
    if expr.count("(") != expr.count(")"):
        return ["unbalanced parentheses"]
    problems = []
    for op, (nfixed, kws) in _KWONLY_OPERATORS.items():
        i = 0
        while True:
            i = expr.find(op + "(", i)
            if i < 0:
                break
            if i and (expr[i - 1].isalnum() or expr[i - 1] == "_"):
                i += 1
                continue
            open_at = i + len(op)
            depth, j = 0, open_at
            for j in range(open_at, len(expr)):
                if expr[j] == "(":
                    depth += 1
                elif expr[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
            args, depth, cur, in_quote = [], 0, "", False
            for ch in expr[open_at + 1:j]:
                if ch == '"':
                    in_quote = not in_quote
                if ch == "," and depth == 0 and not in_quote:
                    args.append(cur)
                    cur = ""
                    continue
                if ch in "([" and not in_quote:
                    depth += 1
                elif ch in ")]" and not in_quote:
                    depth -= 1
                cur += ch
            if cur.strip():
                args.append(cur)
            for k, arg in enumerate(args):
                if k >= nfixed and "=" not in arg and arg.strip():
                    kw = kws[min(k - nfixed, len(kws) - 1)]
                    problems.append(f"{op}: argument {k + 1} {arg.strip()!r} must be written as {kw}=...")
            i = j
    return problems


@mcp.tool()
async def create_multi_simulation(
    alpha_expressions: List[str],
    instrument_type: str = "EQUITY",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: int = 1,
    decay: float = 0.0,
    neutralization: str = "NONE",
    truncation: float = 0.0,
    test_period: str = "P0Y0M",
    unit_handling: str = "VERIFY",
    nan_handling: str = "OFF",
    language: str = "FASTEXPR",
    visualization: bool = True,
    pasteurization: str = "ON",
    max_trade: str = "OFF",
    lookback: Optional[int] = None,
    simulation_mode: Optional[str] = None,
    per_alpha_settings: Optional[List[Dict[str, Any]]] = None,
    max_position: Optional[str] = None,
    validate_expressions: bool = True,
) -> Dict[str, Any]:
    """
    🚀 Submit 2-10 regular alpha simulations in a single request (returns immediately).

    This tool submits a multisimulation and returns right away with a
    progress_url — it does NOT wait for completion (typically 3-10 minutes). Poll
    check_simulation_progress(progress_url): it reports per-child progress, then
    one compact row per alpha once finished. Call get_platform_setting_options to
    get the valid options for the simulation.

    Every child may use DIFFERENT settings (the platform accepts per-item
    settings): the keyword arguments below are the base, and per_alpha_settings[i]
    overrides them for alpha_expressions[i]. Sweeping decay / neutralization /
    truncation / maxTrade on one expression is a single batch:
        alpha_expressions=["rank(x)"],
        per_alpha_settings=[{"decay": 3}, {"decay": 5}, {"neutralization": "MARKET"}]
    (a single expression is repeated for every per_alpha_settings entry).

    Args:
        alpha_expressions: 2-10 alpha expressions; or ONE expression plus 2-10
            per_alpha_settings entries. For language="PYTHON" each entry is Python source.
        per_alpha_settings: Optional list of per-child overrides, same length as
            alpha_expressions (use {} for "base settings"). Keys: decay,
            neutralization, truncation, max_trade, max_position, universe, region,
            delay, pasteurization, nan_handling, unit_handling, test_period,
            visualization, simulation_mode, instrument_type (camelCase API names
            such as maxTrade also work). language/lookback are batch-level only.
        instrument_type / region / universe / delay / decay / neutralization /
        truncation / pasteurization / max_trade: base settings (see create_simulation).
        max_position: base maxPosition ("ON"/"OFF"; default None = omitted).
        test_period / unit_handling / nan_handling: ignored when language="PYTHON".
        language: "FASTEXPR" (default) or "PYTHON".
        visualization: Enable visualization (default: True)
        lookback: PYTHON-only lookback window (required when language="PYTHON").
        simulation_mode: "QUICK" or "FULL" (default None = platform default, FULL).
            QUICK = fast feedback (core metrics only, no visualizations, no
            Theme / Competition / correlation checks, not directly submittable);
            forces visualization=False for that child.
        validate_expressions: Lint FASTEXPR expressions before sending (default
            True): unbalanced parentheses, or an optional operator argument passed
            positionally (e.g. ts_backfill(x, 250) instead of lookback=250). One
            bad child makes the platform cancel the WHOLE batch, so the request
            is refused with the problems listed. Set False to send anyway.

    Returns:
        {"status": "SUBMITTED", "type": "MULTI", "multisimulation_id": ...,
         "progress_url": ..., "children": [{"index", "overrides"}...]} — poll
        check_simulation_progress(progress_url); or {"status": "RATE_LIMITED", ...}
        when the account's concurrent simulation slots are full.
    """
    try:
        exprs = list(alpha_expressions or [])
        overrides_list = list(per_alpha_settings or [])
        if overrides_list and len(exprs) == 1:
            exprs = exprs * len(overrides_list)
        if overrides_list and len(overrides_list) != len(exprs):
            return {"error": (f"per_alpha_settings has {len(overrides_list)} entries but there are "
                              f"{len(exprs)} expressions; they must match one-to-one")}
        if len(exprs) < 2:
            return {"error": "At least 2 alpha expressions (or 1 expression + 2 per_alpha_settings) are required"}
        if len(exprs) > 10:
            return {"error": "Maximum 10 alpha expressions allowed per request"}
        if any(not isinstance(e, str) or not e.strip() for e in exprs):
            return {"error": "Every alpha expression must be a non-empty string"}

        is_python = (language or "").upper() == "PYTHON"
        if is_python and lookback is None:
            return {"error": "lookback is required when language='PYTHON'"}

        if validate_expressions and not is_python:
            problems = {i: p for i, p in ((i, _lint_expression(e)) for i, e in enumerate(exprs)) if p}
            if problems:
                return {"error": "Expression pre-check failed — one failing child cancels the whole batch",
                        "problems": [{"index": i, "expr": exprs[i][:120], "issues": p}
                                     for i, p in problems.items()],
                        "note": "Fix the expressions, or pass validate_expressions=False to send anyway."}

        base: Dict[str, Any] = {
            'instrumentType': instrument_type,
            'region': region,
            'universe': universe,
            'delay': delay,
            'decay': decay,
            'neutralization': neutralization,
            'truncation': truncation,
            'pasteurization': pasteurization,
            'language': language,
            'visualization': visualization,
            'maxTrade': max_trade,
        }
        if max_position is not None:
            base['maxPosition'] = max_position
        if simulation_mode is not None:
            base['simulationMode'] = simulation_mode
        if is_python:
            base['lookback'] = lookback
        else:
            base['unitHandling'] = unit_handling
            base['nanHandling'] = nan_handling
            base['testPeriod'] = test_period

        multisimulation_data = []
        children = []
        for i, alpha_expr in enumerate(exprs):
            raw = overrides_list[i] if overrides_list else {}
            if not isinstance(raw, dict):
                return {"error": f"per_alpha_settings[{i}] must be an object, got {type(raw).__name__}"}
            unknown = [k for k in raw if k not in _MULTI_OVERRIDE_KEYS]
            if unknown:
                return {"error": (f"per_alpha_settings[{i}] has unsupported keys {unknown}; "
                                  f"allowed: {sorted(set(_MULTI_OVERRIDE_KEYS.values()))} "
                                  "(or their snake_case forms)")}
            override = {_MULTI_OVERRIDE_KEYS[k]: v for k, v in raw.items() if v is not None}
            settings = {**base, **override}
            if is_python:
                for k in ('unitHandling', 'nanHandling', 'testPeriod'):
                    settings.pop(k, None)
            try:
                mode, settings['visualization'] = _normalize_simulation_mode(
                    settings.pop('simulationMode', None), settings['visualization'])
            except ValueError as e:
                return {"error": f"alpha {i}: {e}"}
            if mode:
                settings['simulationMode'] = mode
            multisimulation_data.append({'type': 'REGULAR', 'settings': settings, 'regular': alpha_expr})
            children.append({"index": i, "overrides": override} if override else {"index": i})

        # Send multisimulation request (must go through _request: a direct
        # session.post here would block the shared event loop for every client)
        response = await brain_client._request('post', f"{brain_client.base_url}/simulations", json=multisimulation_data)

        if response.status_code == 429:
            retry_after = _retry_after_seconds(response) or 30.0
            return {
                "status": "RATE_LIMITED",
                "retry_after_seconds": retry_after,
                "note": ("BRAIN's per-account concurrent simulation limit is reached "
                         "(another simulation is still running on this account). "
                         f"Retry create_multi_simulation after ~{int(retry_after)}s, or first "
                         "finish/check the running ones; you can do other work meanwhile."),
            }
        if response.status_code != 201:
            # Include BRAIN's per-item rejection reasons (400 bodies are a JSON
            # array with one entry per expression) — a bare status is undiagnosable.
            detail = _http_error_detail(response)
            brain_client.log(f"❌ Failed to create multisimulation: {detail}", "ERROR")
            return {"error": f"Failed to create multisimulation. {detail}"}

        location = response.headers.get('Location', '')
        if not location:
            return {"error": "No location header in multisimulation response"}

        # Submit-only: return immediately, same pattern as create_simulation.
        result = {
            "status": "SUBMITTED",
            "type": "MULTI",
            "multisimulation_id": location.split('/')[-1],
            "expected_children": len(exprs),
            "progress_url": location,
            "note": ("Multi-simulation is running asynchronously (typically 3-10 minutes for "
                     f"{len(exprs)} alphas). Call check_simulation_progress with this "
                     "progress_url to get per-child progress and, once finished, one compact "
                     "row per alpha (each row echoes its settings). You can do other work between checks."),
        }
        if overrides_list:
            result["children"] = children
        return result

    except Exception as e:
        return {"error": f"Error creating multisimulation: {str(e)}"}

# --- Payment and Financial Tools ---

@mcp.tool()
async def get_daily_and_quarterly_payment(email: str = "", password: str = "") -> Dict[str, Any]:
    """
    Get daily and quarterly payment information from WorldQuant BRAIN platform.
    
    This function retrieves both base payments (daily alpha performance payments) and
    other payments (competition rewards, quarterly payments, referrals, etc.).

    Args:
        email: Ignored (kept for backward compatibility; login is managed by credd)
        password: Ignored (kept for backward compatibility; login is managed by credd)

    Returns:
        Dictionary containing base payment and other payment data with summaries and detailed records
    """
    try:
        # Get base payments (login comes from credd via the session; no creds needed)
        try:
            base_response = await brain_client._request('get', f"{brain_client.base_url}/users/self/activities/base-payment")
            base_response.raise_for_status()
            base_payments = base_response.json()
        except Exception:
            base_payments = "no data"

        try:
            # Get other payments
            other_response = await brain_client._request('get', f"{brain_client.base_url}/users/self/activities/other-payment")
            other_response.raise_for_status()
            other_payments = other_response.json()
        except Exception:
            other_payments = "no data"
        return {
            "base_payments": base_payments,
            "other_payments": other_payments
        }
        
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

from typing import Sequence
@mcp.tool()
async def lookINTO_SimError_message(locations: Sequence[str]) -> dict:
    """
    Fetch and parse error/status from multiple simulation locations (URLs).
    Args:
        locations: List of simulation result URLs (e.g., /simulations/{id})
    Returns:
        List of dicts with location, error message, and raw response
    """
    results = []
    for loc in locations:
        try:
            resp = await brain_client._request('get', loc)
            if resp.status_code != 200:
                results.append({
                    "location": loc,
                    "error": f"HTTP {resp.status_code}",
                    "raw": resp.text
                })
                continue
            data = resp.json() if resp.text else {}
            # Try to extract error message or status
            error_msg = data.get("error") or data.get("message")
            # If alpha ID is missing, include that info
            if not data.get("alpha"):
                error_msg = error_msg or "Simulation did not get through, if you are running a multisimulation, check the other children location in your request"
            results.append({
                "location": loc,
                "error": error_msg,
                "raw": data
            })
        except Exception as e:
            results.append({
                "location": loc,
                "error": str(e),
                "raw": None
            })
    return {"results": results}


# --- ProdMemo: local Self/Pool/Prod correlation memory -----------------------
# Thin wrappers over prodmemo_service (see docs/PRODMEMO_IMPLEMENTATION.md).
# The service holds no BRAIN client of its own — inject this module's, which
# already carries credd-backed auth and 401 self-healing.
prodmemo_client.fetcher = brain_client


@mcp.tool()
async def prodmemo_sync(mode: str = "incremental") -> Dict[str, Any]:
    """Sync submitted alphas and their PnL into the local ProdMemo database.

    Runs in the background and returns immediately — poll prodmemo_sync_status.
    'incremental' probes the remote count first and only fetches what is missing;
    'full' re-walks every alpha; 'stop' cancels a run in progress.

    Args:
        mode: "incremental" (default), "full", or "stop"
    """
    try:
        return await prodmemo_client.start_sync(mode)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def prodmemo_sync_status() -> Dict[str, Any]:
    """Progress and final state of the most recent ProdMemo sync."""
    try:
        return await prodmemo_client.sync_status()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def prodmemo_check(alpha_id: str = "", alpha_ids: Optional[List[str]] = None,
                         run_platform_check: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """Estimate Prod Correlation locally for one or many alphas (no platform quota).

    Per alpha (compact row):
      prod_est          — empirical estimate prod ≈ a + b × pool (the most useful
                          number; default a=0.428 b=1.139, refitted automatically
                          once >= 8 alphas have a measured platform Prod)
      pool / self       — local Pool / Self correlation max
      prod_lower_bound  — conditional lower bound from reference curves; > 0.7
                          proves the platform check would fail
      platform_prod     — platform-measured Prod, if known
      recommendation    — "skip" (bound > 0.7), "check", or "insufficient_data"
    Platform Prod values measured by check_correlation / get_submission_check
    are written back automatically and become new reference/calibration points.

    Args:
        alpha_id: One alpha to evaluate
        alpha_ids: Several alphas (up to 20) — use instead of alpha_id
        run_platform_check: Also query the platform for Prod/Self correlation and
            store the result (costs platform time; improves future estimates)
        verbose: Return the full per-alpha report (local details, witness,
            calibration, resolved values) instead of the compact row
    """
    try:
        ids = list(alpha_ids or []) + ([alpha_id] if alpha_id else [])
        result = await prodmemo_client.check_many(ids, run_platform_check, verbose)
        return result["results"][0] if len(result["results"]) == 1 and not alpha_ids else result
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def prodmemo_get(alpha_id: str = "", stale_only: bool = False,
                       above: float = 0.0, group_key: str = "",
                       limit: int = 100) -> Dict[str, Any]:
    """Inspect stored ProdMemo state — one alpha in full, or a filtered list.

    Each entry reports sync state (metadata/PnL present, PnL last date), platform
    correlations, local correlations with a live `stale` flag, and the resolved
    value per metric (platform Ⓟ preferred, non-stale local Ⓛ as fallback, ≥ for
    the Prod lower bound).

    Args:
        alpha_id: Return the full status card for this alpha; empty for a list
        stale_only: List only alphas whose local results need recomputing
        above: List only alphas whose highest resolved correlation is >= this
        group_key: Restrict to one Region|Universe|Delay group, e.g. "USA|TOP3000|D1"
        limit: Maximum rows for list mode
    """
    try:
        return await prodmemo_client.get(alpha_id=alpha_id, stale_only=stale_only,
                                         above=above, group_key=group_key, limit=limit)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def prodmemo_stats() -> Dict[str, Any]:
    """Counts held in the ProdMemo database, including how many alphas are usable
    as Prod lower-bound reference curves (valid_reference_count)."""
    try:
        return await prodmemo_client.stats()
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
async def prodmemo_manage(action: str, alpha_id: str = "", data: str = "") -> Dict[str, Any]:
    """Maintain the ProdMemo store: export, import, or clear correlation data.

    'import' accepts the WebDataScope browser extension's Corr JSON export — the
    only way to carry over platform Prod values captured in the browser, which
    cannot be re-derived server-side. Clearing is NOT reversible: 'clear_corrs'
    drops correlations but keeps alphas/PnL, 'clear_sync' does the opposite (the
    surviving local correlations then read as stale).

    Args:
        action: "export" | "import" | "clear_corrs" | "clear_sync" | "delete"
        alpha_id: Alpha to drop, for action="delete"
        data: Corr JSON payload, for action="import"
    """
    try:
        return await prodmemo_client.manage(action, alpha_id=alpha_id, data=data)
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


# --- Main entry point ---
if __name__ == "__main__":
    print("running the server")
    mcp.run(
         transport="streamable-http"
    )

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

import asyncio
import collections
import functools
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import quote, urlsplit
import re
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
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel

# Forum client (support.worldquantbrain.com, headless browser); created below
# with this module's BRAIN session cookies injected.
from forum_functions import ForumClient
from prodmemo_calc import extract_platform_correlation_stats
try:
    from prodmemo_service import prodmemo_client
except ImportError as _prodmemo_import_error:  # psycopg2 missing: keep the other tools working
    class _ProdMemoUnavailable:
        fetcher = None
        _reason = f"ProdMemo is unavailable ({_prodmemo_import_error}); pip install psycopg2-binary"

        def __getattr__(self, name):
            async def unavailable(*args, **kwargs):
                raise RuntimeError(self._reason)
            return unavailable

    prodmemo_client = _ProdMemoUnavailable()

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

# --- Deployment switches (see README.md) -------------------------------------
BRAIN_BASE_URL = os.environ.get("WQMCP_BASE_URL", "https://api.worldquantbrain.com").rstrip("/")
_BRAIN_PARTS = urlsplit(BRAIN_BASE_URL)
# Kill switches: READ_ONLY blocks every write tool; ALLOW_SUBMIT=0 blocks submissions.
READ_ONLY = os.environ.get("WQMCP_READ_ONLY", "0") == "1"
ALLOW_SUBMIT = os.environ.get("WQMCP_ALLOW_SUBMIT", "1") != "0"
# Versioned Accept headers from the API catalog. Off by default: the unversioned
# requests are what has been verified in production; turn on after a live check.
SEND_ACCEPT_VERSIONS = os.environ.get("WQMCP_ACCEPT_VERSIONS", "0") == "1"

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$")


def _seg(value: Any, name: str) -> str:
    """Validate + percent-encode one URL path segment taken from a tool argument.

    Rejects '/', '..', '?', '#', whitespace etc., so an id can never make a request
    address another BRAIN endpoint (e.g. alpha_id='../users/self')."""
    text = str(value).strip() if value is not None else ""
    if not _SEGMENT_RE.match(text) or ".." in text:
        raise ValueError(f"invalid {name}: {value!r}")
    return quote(text, safe="")


def _simulation_url(ref: Any) -> str:
    """Normalise a simulation id or progress URL to the canonical BRAIN URL.

    Only the BRAIN API host and the /simulations/<id> path are accepted, so a
    tool argument cannot make this server fetch arbitrary URLs (SSRF)."""
    text = str(ref or "").strip()
    if "://" in text:
        parts = urlsplit(text)
        m = re.fullmatch(r"/simulations/([^/]+)/?", parts.path or "")
        if (parts.scheme != _BRAIN_PARTS.scheme or parts.hostname != _BRAIN_PARTS.hostname
                or parts.port != _BRAIN_PARTS.port or not m or parts.query or parts.fragment):
            raise ValueError(f"not a BRAIN simulation URL: {text!r}")
        text = m.group(1)
    return f"{BRAIN_BASE_URL}/simulations/{_seg(text, 'simulation id')}"


_ACCEPT_RULES = tuple((m, re.compile(p), v) for m, p, v in (
    ("OPTIONS", r"^/simulations$", "3.0"),
    ("GET", r"^/users/self/alphas$", "4.0"),
    ("GET", r"^/users/self/alphas/summary$", "4.0"),
    ("GET", r"^/alphas/[^/]+/alphas$", "4.0"),
    ("GET", r"^/tags/[^/]+/alphas$", "4.0"),
    ("GET", r"^/suggest/fields$", "4.0"),
    ("GET", r"^/data-fields/summary$", "3.0"),
    ("GET", r"^/users/self/activities/base-payment$", "3.0"),
))


def _accept_header(method: str, url: str) -> Optional[str]:
    """The catalog's versioned Accept header for a BRAIN URL (None when disabled)."""
    if not SEND_ACCEPT_VERSIONS or not str(url).startswith(BRAIN_BASE_URL):
        return None
    path = urlsplit(str(url)).path
    for m, pattern, version in _ACCEPT_RULES:
        if m == method.upper() and pattern.match(path):
            return f"application/json;version={version}"
    return "application/json;version=2.0"


def _json_or_none(response: requests.Response) -> Any:
    try:
        return response.json() if (response.text or "").strip() else None
    except ValueError:
        return None


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
                 "fails lists failed checks. Call get_alpha(id) for the full alpha, or "
                 "get_simulation(compact=False). " + _FLIP_NOTE)


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


def _pyramid_key(p: Dict[str, Any]) -> str:
    """Same key for an alpha's pyramid entry and a pyramid-multipliers row."""
    if p.get("name"):
        return str(p["name"])
    cat = p.get("category") or {}
    cat_name = (cat.get("name") or cat.get("id")) if isinstance(cat, dict) else cat
    return f"{p.get('region')}/D{p.get('delay')}/{cat_name}"


def _alpha_pyramid_keys(alpha: Dict[str, Any]) -> List[str]:
    ps = alpha.get("pyramids")
    if not isinstance(ps, list):
        pt = alpha.get("pyramidThemes") or {}
        ps = pt.get("pyramids") if isinstance(pt, dict) else None
    return [_pyramid_key(p) for p in ps or [] if isinstance(p, dict)]


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
_BRAIN_HOST = _BRAIN_PARTS.hostname or ""
_BRAIN_COOKIE_DOMAIN = (".worldquantbrain.com" if _BRAIN_HOST.endswith("worldquantbrain.com")
                        else _BRAIN_HOST)


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
        # (stale version, monotonic time) of the last refresh attempt: concurrent
        # 401s while credd is in backoff must not each hit credd in turn.
        self._last_attempt: Tuple[Optional[str], float] = (None, 0.0)
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

    def refresh_cookies(self, stale_version: Optional[str] = None) -> Optional[str]:
        """Replace the whole cookie jar from credd; returns the cookie version.

        With stale_version (the version a failed request used): skip the fetch if
        another thread already moved past it, or if the same stale version was
        tried in the last 10s (credd in backoff hands out the same cookie)."""
        with self._refresh_lock:
            if stale_version is not None:
                if self._cookie_version != stale_version:
                    return self._cookie_version
                tried_version, tried_at = self._last_attempt
                if tried_version == stale_version and time.monotonic() - tried_at < 10.0:
                    return self._cookie_version
            try:
                cookies, version = self._fetch_from_credd()
            finally:
                self._last_attempt = (stale_version, time.monotonic())
            jar = requests.cookies.RequestsCookieJar()
            secure = _BRAIN_PARTS.scheme == "https"
            for name, value in cookies.items():
                jar.set(name, value, domain=_BRAIN_COOKIE_DOMAIN, path="/", secure=secure)
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
        if not (resp.status_code == 401 and urlsplit(str(url)).hostname == _BRAIN_HOST):
            return resp
        try:
            new_version = self.refresh_cookies(stale_version=version_used)
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
        self.base_url = BRAIN_BASE_URL
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
        # submit_alpha bookkeeping: one POST per alpha, resumable polling.
        self._submit_locks: Dict[str, asyncio.Lock] = {}
        self._pending_submits: set = set()
        self._submit_results: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        # Fire-and-forget work (ProdMemo write-backs) kept alive until done.
        self._background: set = set()
        # Simulations created by this process (newest first), for get_simulation().
        self.recent_simulations: collections.deque = collections.deque(maxlen=50)
    
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
        session.mount(f"{_BRAIN_PARTS.scheme}://", adapter)
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
        accept = _accept_header(method, url)
        if accept:
            kwargs['headers'] = {"Accept": accept, **(kwargs.get('headers') or {})}
        loop = asyncio.get_running_loop()
        func = getattr(session, method)
        is_get = method.lower() == 'get'
        attempts = 3 if is_get else 1
        for attempt in range(attempts):
            last = attempt == attempts - 1
            cooldown = self._cooldown_until - time.monotonic()
            if is_get and cooldown > 0:
                await asyncio.sleep(min(cooldown, 30.0))  # cooldown itself is capped at 30s
            try:
                resp = await loop.run_in_executor(self._executor, lambda: func(url, **kwargs))
            except (requests.ConnectionError, requests.Timeout):
                # Transient; anything else (bad URL, TLS config...) will not heal.
                if last or not is_get:
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

    async def cookie_list(self) -> List[Dict[str, Any]]:
        """Current BRAIN session cookies (for the forum's headless browser)."""
        session = await self._ensure_session()
        return [{"name": c.name, "value": c.value, "domain": c.domain, "path": c.path,
                 "secure": True, "httpOnly": True} for c in session.cookies]

    async def _poll(self, url: str, max_wait: float) -> Dict[str, Any]:
        """Follow BRAIN's Retry-After protocol on a GET until done or max_wait runs out.

        Returns {"status": "DONE", "data": <json or None>} once a 2xx arrives without
        Retry-After, {"status": "PENDING", "retry_after_seconds": s} while the
        platform is still computing (or busy: 429/5xx after _request's retries),
        and {"status": "ERROR", "http_status": n, "error": ...} on any other 4xx.
        """
        deadline = time.monotonic() + max(0.0, min(float(max_wait or 0), 300.0))
        while True:
            try:
                resp = await self._request('get', url)
            except requests.RequestException as e:
                resp, net_error = None, str(e)
            if resp is not None and 400 <= resp.status_code < 500 and resp.status_code != 429:
                return {"status": "ERROR", "http_status": resp.status_code, "error": _http_error_detail(resp),
                        "json": _json_or_none(resp)}
            busy = resp is None or resp.status_code == 429 or resp.status_code >= 500
            ra = _retry_after_seconds(resp) if resp is not None else 0.0
            if not busy and "Retry-After" not in resp.headers:
                text = (resp.text or "").strip()
                if not text:
                    return {"status": "DONE", "data": None}
                try:
                    return {"status": "DONE", "data": resp.json()}
                except ValueError:
                    return {"status": "ERROR", "error": "non-JSON body", "body": text[:300]}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                out = {"status": "PENDING", "retry_after_seconds": max(ra, 5.0 if busy else 1.0)}
                if busy:
                    out["busy"] = net_error if resp is None else f"HTTP {resp.status_code}"
                return out
            await asyncio.sleep(max(1.0, min(ra or 5.0, remaining)))

    async def _check_once(self, location: str, compact: bool = True) -> Dict[str, Any]:
        """One status check of a simulation location — single or multi.

        A multi-simulation parent exposes a `children` list; each child is an
        ordinary simulation, so multi handling composes out of the single case.
        compact=True returns one _compact_alpha_row per finished alpha instead of
        the full alpha object.
        """
        location = _simulation_url(location)
        resp = await self._request('get', location)
        if resp.status_code == 429 or resp.status_code >= 500:
            # Still busy after _request's own retries: not a verdict on the simulation.
            return {"status": "RUNNING", "http_status": resp.status_code, "progress_url": location,
                    "retry_after_seconds": _retry_after_seconds(resp) or 10.0,
                    "note": "BRAIN is busy; check again after retry_after_seconds."}
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
            running = ("Retry-After" in resp.headers
                       or str(body.get("status") or "").upper() in ("RUNNING", "PENDING"))
            if body.get("alpha") and not running:
                summary = await self.get_raa_alpha(body["alpha"])
                status = "ERROR" if summary.get("error") else (
                    "RUNNING" if summary.get("children_pending") else "COMPLETE")
                return {**summary, "status": status, "progress_url": location}
            if running or body.get("alpha"):
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

        if children and "Retry-After" in resp.headers:
            # Parent still running: honour its Retry-After instead of fanning out
            # one request per child on every check.
            return {
                "status": "RUNNING", "type": "MULTI", "progress": body.get("progress"),
                "total_children": len(children),
                "retry_after_seconds": _retry_after_seconds(resp) or 5.0,
                "progress_url": location,
                "note": "Multi-simulation still running; check again after retry_after_seconds.",
            }
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
                    "message": body.get("message") or body.get("detail") or body.get("details"),
                    "raw": body}
        alpha = await self._request('get', f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}")
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
        async def child_state(child: Any) -> Dict[str, Any]:
            try:
                url = _simulation_url(child)
            except ValueError as e:
                return {"location": str(child), "status": "ERROR", "error": str(e)}
            try:
                r = await self._request('get', url)
            except Exception as e:
                return {"location": url, "status": "UNKNOWN", "error": str(e)}
            if r.status_code == 429 or r.status_code >= 500:
                # Busy is not finished: counting it as settled would report the
                # whole batch COMPLETE and silently drop this child's result.
                return {"location": url, "status": "UNKNOWN", "http_status": r.status_code}
            if r.status_code >= 400:
                return {"location": url, "status": "ERROR", "http_status": r.status_code,
                        "error": (r.text or "")[:300]}
            try:
                b = r.json() if (r.text or "").strip() else {}
            except ValueError:
                b = {}
            if "Retry-After" in r.headers:
                return {"location": url, "status": "RUNNING", "progress": b.get("progress")}
            alpha_id = b.get("alpha")
            if not alpha_id:
                return {"location": url, "status": b.get("status", "ERROR"),
                        "message": b.get("message") or b.get("detail") or b.get("details")}
            return {"location": url, "status": "COMPLETE", "alpha_id": alpha_id}

        states = list(await asyncio.gather(*[child_state(c) for c in children]))
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
                d = await self._request('get', f"{self.base_url}/alphas/{_seg(s['alpha_id'], 'alpha id')}")
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
        """Current login as reported by GET /authentication (user id, expiry, permissions)."""
        try:
            response = await self._request('get', f"{self.base_url}/authentication")
            response.raise_for_status()
            return response.json() if (response.text or "").strip() else {}
        except Exception as e:
            self.log(f"Failed to get auth status: {str(e)}", "ERROR")
            return None
    
    @staticmethod
    def simulation_payload(simulation_data: SimulationData) -> Dict[str, Any]:
        """The POST /simulations item for one alpha: settings trimmed to what its
        type and language carry, None values dropped."""
        settings_dict = simulation_data.settings.model_dump()
        if simulation_data.type in ("REGULAR", "REGION_AGNOSTIC"):
            # SUPER-only fields
            for k in ('selectionHandling', 'selectionLimit', 'componentActivation'):
                settings_dict.pop(k, None)
        if (settings_dict.get('language') or '').upper() == "PYTHON":
            # PYTHON payload omits these FASTEXPR-only fields
            for k in ('unitHandling', 'nanHandling', 'testPeriod'):
                settings_dict.pop(k, None)
        else:
            settings_dict.pop('lookback', None)  # PYTHON-only
        payload: Dict[str, Any] = {
            'type': simulation_data.type,
            'settings': {k: v for k, v in settings_dict.items() if v is not None},
        }
        if simulation_data.type in ("REGULAR", "REGION_AGNOSTIC"):
            payload['regular'] = simulation_data.regular
        elif simulation_data.type == "SUPER":
            payload['combo'] = simulation_data.combo
            payload['selection'] = simulation_data.selection
        return {k: v for k, v in payload.items() if v is not None}

    @staticmethod
    def _rate_limited(response: requests.Response) -> Dict[str, Any]:
        # Per-account concurrent simulation slots are full (e.g. other sessions'
        # simulations still running): structured info so the client can back off.
        retry_after = _retry_after_seconds(response) or 30.0
        return {
            "status": "RATE_LIMITED",
            "retry_after_seconds": retry_after,
            "note": ("BRAIN's per-account concurrent simulation limit is reached "
                     "(other simulations are still running on this account). "
                     f"Retry after ~{int(retry_after)}s, or first finish/check the running "
                     "ones (get_simulation); you can do other work meanwhile."),
        }

    def _remember_simulation(self, simulation_id: str, kind: str, sim_type: str, count: int) -> None:
        self.recent_simulations.appendleft({
            "simulation_id": simulation_id, "kind": kind, "type": sim_type, "alphas": count,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    async def _post_simulation(self, body: Any, what: str) -> Any:
        """POST /simulations; returns the RATE_LIMITED dict or (simulation_id, location)."""
        await self.ensure_authenticated()
        response = await self._request('post', f"{self.base_url}/simulations", json=body)
        if response.status_code == 429:
            return self._rate_limited(response)
        if response.status_code >= 400:
            # Surface BRAIN's rejection reason (invalid settings choice, blank
            # expression, per-item errors of a multi...) instead of a bare 400.
            raise Exception(_http_error_detail(response, f"{what} rejected"))
        location = response.headers.get('Location', '')
        if not location:
            raise Exception(f"BRAIN returned no Location header for the submitted {what}")
        return location.rstrip('/').split('/')[-1], location

    async def create_simulation(self, simulation_data: SimulationData) -> Dict[str, Any]:
        """Submit one simulation (REGULAR, SUPER or REGION_AGNOSTIC) and return at once."""
        self.log("🚀 Creating simulation...", "INFO")
        posted = await self._post_simulation(self.simulation_payload(simulation_data), "simulation")
        if isinstance(posted, dict):
            return posted
        simulation_id, location = posted
        self._remember_simulation(simulation_id, "single", simulation_data.type, 1)
        self.log(f"Simulation submitted with ID: {simulation_id}", "SUCCESS")
        # Submit-only: the MCP client is never parked on a long-running HTTP call.
        return {
            "status": "SUBMITTED",
            "simulation_id": simulation_id,
            "progress_url": location,
            "note": ("Simulation is running asynchronously (typically 1-5 minutes; RAA longer). "
                     "Call get_simulation with this simulation_id to get progress and, once "
                     "finished, the result. You can do other work between checks."),
        }

    async def create_multi_simulation(self, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit 2-10 simulation items as ONE multi-simulation (one POST, one parent id).
        BRAIN cancels the whole batch when any child fails."""
        self.log(f"🚀 Creating multi-simulation ({len(payloads)} alphas)...", "INFO")
        posted = await self._post_simulation(payloads, "multi-simulation")
        if isinstance(posted, dict):
            return posted
        simulation_id, location = posted
        self._remember_simulation(simulation_id, "multi", payloads[0].get("type", "REGULAR"), len(payloads))
        return {
            "status": "SUBMITTED",
            "type": "MULTI",
            "simulation_id": simulation_id,
            "multisimulation_id": simulation_id,
            "expected_children": len(payloads),
            "progress_url": location,
            "note": ("Multi-simulation is running asynchronously (typically 3-10 minutes for "
                     f"{len(payloads)} alphas). Call get_simulation with this simulation_id to get "
                     "per-child progress and, once finished, one compact row per alpha (each row "
                     "echoes its settings). You can do other work between checks."),
        }

    async def cancel_simulation(self, ref: str) -> Dict[str, Any]:
        """DELETE a queued/running simulation to free its account slot."""
        await self.ensure_authenticated()
        url = _simulation_url(ref)
        response = await self._request('delete', url)
        if response.status_code >= 400:
            raise Exception(_http_error_detail(response, "cancel simulation"))
        return {"simulation_id": url.rsplit('/', 1)[-1], "cancelled": True,
                "http_status": response.status_code}

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
    
    async def get_raa_alpha(self, parent_alpha_id: str,
                            parent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Summarise an RA parent alpha: settings/expression plus one metric row per
        region child. An RA_PARENT carries no metrics of its own — all performance
        lives on its RA_CHILD alphas, so they are fetched concurrently. `parent` is
        the already-fetched parent object, if the caller has it."""
        await self.ensure_authenticated()

        if parent is None:
            parent_resp = await self._request('get', f"{self.base_url}/alphas/{_seg(parent_alpha_id, 'alpha id')}")
            parent_resp.raise_for_status()
            parent = parent_resp.json()
        child_ids = parent.get("children") or []
        if not child_ids:
            return {"type": parent.get("type"), "parent_alpha_id": parent_alpha_id,
                    "error": "Alpha has no children — it is not an RA parent alpha.",
                    "parent": parent}

        async def child_row(cid: str) -> Dict[str, Any]:
            try:
                r = await self._request('get', f"{self.base_url}/alphas/{_seg(cid, 'alpha id')}")
                r.raise_for_status()
                return _raa_child_row(r.json())
            except Exception as e:
                return {"alpha_id": cid, "error": str(e)}

        rows = list(await asyncio.gather(*[child_row(c) for c in child_ids]))
        settings = parent.get("settings") or {}
        # A child without checks yet has not finished computing: it must not count
        # as "no FAIL" (that would suggest the RAA is ready to submit).
        pending = [r for r in rows if not r.get("error") and r.get("sharpe") is None and not r.get("fails")]
        for r in pending:
            r["pending"] = True
        ok = [r for r in rows if not r.get("error") and not r.get("pending")]
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
            "children_pending": len(pending),
            "note": ("Submission needs >=2 children with no FAIL. Then run "
                     "check_alpha(check='prod') on those children: one child passing PROD "
                     "correlation lets all passing children be submitted together "
                     "(the whole RAA counts as a single submission). Use "
                     "check_alpha(check='submission') and submit_alpha on the PARENT id — "
                     "children cannot be checked individually."),
        }

    async def get_alpha_details(self, alpha_id: str) -> Dict[str, Any]:
        """Get detailed information about an alpha."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get alpha details: {str(e)}", "ERROR")
            raise
    
    async def get_datasets(self, instrument_type: str = "EQUITY", region: str = "USA",
                          delay: int = 1, universe: str = "TOP3000", theme: str = "false", search: Optional[str] = None,
                          limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
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
            if limit:
                params['limit'] = max(1, min(int(limit), 50))
            if offset:
                params['offset'] = max(0, int(offset))
            
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
                            search: Optional[str] = None, limit: int = 50,
                            offset: int = 0) -> Dict[str, Any]:
        """Get available data fields (paged: limit <= 100, offset)."""
        await self.ensure_authenticated()
        
        try:
            params = {
                'instrumentType': instrument_type,
                'region': region,
                'delay': delay,
                'universe': universe,
                'limit': max(1, min(int(limit or 50), 100)),
                'offset': max(0, int(offset or 0)),
            }
            
            if data_type and data_type != 'ALL':
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
    
    async def get_alpha_pnl(self, alpha_id: str, max_wait: float = 30) -> Dict[str, Any]:
        """PnL recordset, following BRAIN's Retry-After protocol.

        Contract (ProdMemo relies on it): {} = still being generated, call again
        later; raises on a real failure (4xx)."""
        await self.ensure_authenticated()
        r = await self._poll(f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}/recordsets/pnl", max_wait)
        if r["status"] == "ERROR":
            raise Exception(f"PnL failed for {alpha_id}: {r.get('error')}")
        return (r.get("data") or {}) if r["status"] == "DONE" else {}

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
        status: Optional[str] = None,
        alpha_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get user's alphas with advanced filtering (stage/status/type, dates, hidden)."""
        await self.ensure_authenticated()
        
        try:
            params = {
                "limit": limit,
                "offset": offset,
            }
            if stage:
                params["stage"] = stage
            if status:
                params["status"] = status
            if alpha_type:
                params["type"] = alpha_type
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
    
    async def submit_alpha(self, alpha_id: str, wait_seconds: float = 60) -> Dict[str, Any]:
        """Submit an alpha and follow BRAIN's asynchronous submission to its verdict.

        POST /alphas/{id}/submit starts it; GET on the same URL is polled while it
        carries Retry-After, and the final body lists every check. Concurrent calls
        for one alpha share one POST; a call after PENDING resumes polling instead of
        submitting again. A 4xx carrying checks is reported as REJECTED."""
        await self.ensure_authenticated()
        url = f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}/submit"
        lock = self._submit_locks.setdefault(alpha_id, asyncio.Lock())
        async with lock:
            recent = self._submit_results.get(alpha_id)
            if recent and time.monotonic() - recent[0] < 600:
                return {**recent[1], "note": "Result of the submission made in the last 10 minutes."}
            if alpha_id not in self._pending_submits:
                self.log(f"📤 Submitting alpha {alpha_id}...", "INFO")
                resp = await self._request('post', url)
                if resp.status_code == 429:
                    return {"success": False, "alpha_id": alpha_id, "status": "RATE_LIMITED",
                            "retry_after_seconds": _retry_after_seconds(resp) or 30.0}
                if resp.status_code >= 400:
                    return self._submit_verdict(alpha_id, _json_or_none(resp), resp)
                self._pending_submits.add(alpha_id)
                if "Retry-After" not in resp.headers:
                    return self._finish_submit(alpha_id, _json_or_none(resp))
            r = await self._poll(url, wait_seconds)
            if r["status"] == "PENDING":
                return {"success": False, "alpha_id": alpha_id, "status": "PENDING",
                        "retry_after_seconds": r["retry_after_seconds"],
                        "note": "Submission is being processed. Call submit_alpha again to keep "
                                "polling; it will not submit twice."}
            if r["status"] == "ERROR":
                self._pending_submits.discard(alpha_id)  # a fixed alpha can be resubmitted
                return self._submit_verdict(alpha_id, r.get("json"), None, r)
            return self._finish_submit(alpha_id, r.get("data"))

    def _finish_submit(self, alpha_id: str, data: Any) -> Dict[str, Any]:
        self._pending_submits.discard(alpha_id)
        result = self._submit_verdict(alpha_id, data, None)
        self._submit_results[alpha_id] = (time.monotonic(), result)
        return result

    @staticmethod
    def _submit_verdict(alpha_id: str, body: Any, resp: Optional[requests.Response],
                        poll_error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        checks = [c for c in (((body or {}).get("is") or {}).get("checks") or []) if isinstance(c, dict)] \
            if isinstance(body, dict) else []
        failed = [c.get("name") for c in checks if c.get("result") == "FAIL"]
        pending = [c.get("name") for c in checks if c.get("result") == "PENDING"]
        rows = [{k: c.get(k) for k in ("name", "result", "value", "limit") if c.get(k) is not None}
                for c in checks]
        http_status = resp.status_code if resp is not None else (poll_error or {}).get("http_status")
        if resp is not None or poll_error:
            if not checks:  # an error without a check report
                detail = _http_error_detail(resp, "submit") if resp is not None else poll_error.get("error")
                return {"success": False, "alpha_id": alpha_id, "status": "ERROR",
                        "http_status": http_status, "error": detail}
            return {"success": False, "alpha_id": alpha_id, "status": "REJECTED",
                    "http_status": http_status, "failed": failed, "checks": rows}
        if failed:
            return {"success": False, "alpha_id": alpha_id, "status": "REJECTED",
                    "failed": failed, "pending": pending, "checks": rows}
        return {"success": True, "alpha_id": alpha_id,
                "status": "SUBMITTED_WITH_PENDING_CHECKS" if pending else "SUBMITTED",
                "pending": pending, "checks": rows}

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
                own_id = await self._self_id_or_none()
                if not own_id:
                    raise Exception("could not determine your user id from GET /authentication")
                params['user'] = own_id

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
                t = t.get('name') if isinstance(t, dict) else t
                if isinstance(t, str) and t.strip().lower() == 'atom':
                    return True

        for c in classifications:
            cid = (c.get('id') or c.get('name') or '')
            if isinstance(cid, str) and 'ATOM' in cid.upper():
                return True

        return False

    async def value_factor_trendScore(self, start_date: str, end_date: str,
                                      max_alphas: int = 2000) -> Dict[str, Any]:
        """Diversity score of the REGULAR alphas submitted in [start_date, end_date].

        S_A = share of Atom (single-data-set) alphas, S_P = pyramids covered /
        pyramids available (pyramid-multipliers), S_H = normalised entropy of the
        pyramid distribution; diversity_score = S_A * S_P * S_H. A client-side
        estimate of the value-factor trend, not BRAIN's official valueFactor.

        listAlphas already returns classifications and pyramids, so this pages
        through it (no per-alpha requests) and re-applies the date window, since
        the dateSubmitted filters are not documented.
        """
        await self.ensure_authenticated()
        alphas: List[Dict[str, Any]] = []
        offset, page, complete = 0, 100, True
        while True:
            resp = await self.get_user_alphas(stage='OS', limit=page, offset=offset,
                                              submission_start_date=start_date,
                                              submission_end_date=end_date)
            batch = (resp or {}).get('results') or []
            alphas.extend(batch)
            offset += len(batch)
            count = (resp or {}).get('count')
            if not batch or (count is not None and offset >= count):
                break
            if len(alphas) >= max_alphas:
                complete = False
                break

        def in_window(a: Dict[str, Any]) -> bool:
            ds = str(a.get('dateSubmitted') or '')
            if not ds:
                return True
            return str(start_date)[:10] <= ds[:10] <= str(end_date)[:10]

        regular = [a for a in alphas if a.get('type', 'REGULAR') == 'REGULAR' and in_window(a)]
        filtered_out = len([a for a in alphas if not in_window(a)])
        atom_count = sum(1 for a in regular if self._is_atom(a))
        per_pyramid: Dict[str, int] = {}
        for a in regular:
            for p in _alpha_pyramid_keys(a):
                per_pyramid[p] = per_pyramid.get(p, 0) + 1

        N, A, P = len(regular), atom_count, len(per_pyramid)
        P_max = None
        try:
            pm = await self.get_pyramid_multipliers()
            if isinstance(pm, dict):
                P_max = len({_pyramid_key(x) for x in pm.get('pyramids') or [] if isinstance(x, dict)}) or None
        except Exception as e:
            self.log(f"pyramid multipliers unavailable: {e}", "WARNING")
        S_A = (A / N) if N else 0.0
        S_P = (P / P_max) if P_max else None
        S_H = 0.0
        if P > 1:
            total = sum(per_pyramid.values())
            H = -sum((c / total) * math.log2(c / total) for c in per_pyramid.values() if c)
            S_H = H / math.log2(P)
        result = {
            'diversity_score': (S_A * S_P * S_H) if S_P is not None else None,
            'N': N, 'A': A, 'P': P, 'P_max': P_max,
            'S_A': S_A, 'S_P': S_P, 'S_H': S_H,
            'per_pyramid_counts': per_pyramid,
            'complete': complete,
        }
        notes = []
        if filtered_out:
            notes.append(f"{filtered_out} alphas outside the window were ignored")
        if S_P is None:
            notes.append("P_max unknown (pyramid-multipliers failed); diversity_score not computed")
        if not complete:
            notes.append(f"only the first {len(alphas)} alphas were counted")
        if notes:
            result['note'] = "; ".join(notes)
        return result

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
        selection_handling: str = "POSITIVE",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Preview which alphas a SuperAlpha selection expression picks."""
        await self.ensure_authenticated()
        
        try:
            selection_data = {
                "selection": selection,
                # The API catalog documents settings.* names; the flat ones are what
                # v1 sent. Both are sent until a live check shows which is honoured.
                "settings.instrumentType": instrument_type,
                "settings.region": region,
                "settings.delay": delay,
                "instrumentType": instrument_type,
                "region": region,
                "delay": delay,
                "selectionLimit": selection_limit,
                "selectionHandling": selection_handling,
                "limit": limit,
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
        """Your own full record (user_id='self'), or another user's public profile.

        GET /users/{id} answers 403 for anyone but yourself, so other ids use the
        public GET /users/{id}/profile (id, country, university, genius level)."""
        await self.ensure_authenticated()
        uid = _seg(user_id or "self", "user id")
        path = f"/users/{uid}" if uid == "self" else f"/users/{uid}/profile"
        response = await self._request('get', f"{self.base_url}{path}")
        if response.status_code >= 400:
            raise Exception(_http_error_detail(response, "user profile"))
        return response.json()

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

        # "ignore" (default) strips embedded images; "placeholder" saves them to the
        # server's disk (only useful when the MCP client runs on the same machine).
        image_handling = os.environ.get("BRAIN_MESSAGE_IMAGE_MODE", "ignore").lower()
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

    async def get_alpha_yearly_stats(self, alpha_id: str, max_wait: float = 30) -> Dict[str, Any]:
        """Yearly-stats recordset; same contract as get_alpha_pnl ({} = still computing)."""
        await self.ensure_authenticated()
        r = await self._poll(f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}/recordsets/yearly-stats",
                             max_wait)
        if r["status"] == "ERROR":
            raise Exception(f"yearly stats failed for {alpha_id}: {r.get('error')}")
        return (r.get("data") or {}) if r["status"] == "DONE" else {}

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
        url = f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}/correlations/{kind}"
        deadline = time.monotonic() + max(0.0, min(float(max_wait or 0), 300.0))
        while True:
            try:
                resp = await self._request('get', url)
            except requests.RequestException as e:
                resp, net_error = None, str(e)
            busy = resp is None or resp.status_code in (429, 503) or resp.status_code >= 500
            if resp is not None and resp.status_code >= 400 and not busy:
                return {"status": "ERROR", "http_status": resp.status_code,
                        "error": _http_error_detail(resp)}
            ra = _retry_after_seconds(resp) if resp is not None else 0.0
            text = "" if busy else (resp.text or "").strip()
            if busy and deadline - time.monotonic() <= 0:
                # Throttled (the catalog lists 429/503 for this polling endpoint):
                # report "not ready yet", not a failure.
                return {"status": "PENDING", "retry_after_seconds": ra or 10.0,
                        "busy": net_error if resp is None else f"HTTP {resp.status_code}"}
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
        a reference point. Best effort and in the background: for a Prod value the
        write-back also backfills the alpha's metadata + PnL (minutes for a PnL that
        is still generating), and a slow or down database must never delay or break
        the platform tool that measured the value."""
        async def write_back() -> None:
            try:
                await asyncio.wait_for(
                    prodmemo_client.record_platform_corr(alpha_id, kind, max_v, min_v, source),
                    timeout=float(os.environ.get("PRODMEMO_WRITEBACK_TIMEOUT", "300")))
            except Exception as e:  # includes TimeoutError
                self.log(f"ProdMemo write-back skipped for {alpha_id} ({kind}): {e!r}", "WARNING")

        task = asyncio.create_task(write_back())
        self._background.add(task)  # keep a reference until it finishes
        task.add_done_callback(self._background.discard)

    async def check_correlation(self, alpha_id: str, correlation_type: str = "both",
                                threshold: float = 0.7, max_wait: float = 60,
                                include_data: bool = False) -> Dict[str, Any]:
        """Prod and/or self correlation, polled concurrently within max_wait.

        Each type reports status DONE / PENDING / ERROR; all_passed is only a
        boolean once every requested type is DONE (None otherwise). Measured
        values are written back into ProdMemo.
        """
        await self.ensure_authenticated()
        aliases = {"production": "prod", "prod": "prod", "self": "self",
                   "power-pool": "power-pool", "power_pool": "power-pool"}
        ctype = (correlation_type or "both").lower()
        if ctype == "both":
            kinds = ["prod", "self"]
        elif ctype in aliases:
            kinds = [aliases[ctype]]
        else:
            raise ValueError("correlation_type must be 'prod'/'production', 'self', 'power-pool' or 'both'")

        polled = await asyncio.gather(*[self._poll_correlation(alpha_id, k, max_wait) for k in kinds])
        checks: Dict[str, Any] = {}
        for kind, r in zip(kinds, polled):
            name = {"prod": "production", "power-pool": "power_pool"}.get(kind, kind)
            entry: Dict[str, Any] = {"status": r["status"]}
            if r["status"] == "DONE":
                mx = r.get("max")
                entry["max_correlation"] = mx
                entry["passes_check"] = (mx < threshold) if mx is not None else None
                entry["top"] = _correlation_top_rows(r.get("data") or {}, 3)
                if include_data:
                    entry["correlation_data"] = r.get("data")
                if mx is not None and kind in ("prod", "self"):  # ProdMemo keeps prod/self only
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
        url = f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}/check"
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

            response = await self._request('patch', f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to set alpha properties: {str(e)}", "ERROR")
            raise

    async def update_alphas_bulk(self, alpha_ids: List[str], fields: Dict[str, Any]) -> Dict[str, Any]:
        """favorite / hidden / color for many alphas in one PATCH /alphas request."""
        await self.ensure_authenticated()
        body = [{"id": _seg(a, 'alpha id'), **fields} for a in alpha_ids]
        response = await self._request('patch', f"{self.base_url}/alphas", json=body)
        if response.status_code >= 400:
            raise Exception(_http_error_detail(response, "bulk alpha update"))
        return {"alpha_ids": list(alpha_ids), "updated": sorted(fields)}

    async def get_record_sets(self, alpha_id: str, max_wait: float = 20) -> Dict[str, Any]:
        """List available record sets for an alpha (polls while BRAIN prepares them)."""
        await self.ensure_authenticated()
        r = await self._poll(f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}/recordsets", max_wait)
        if r["status"] == "ERROR":
            raise Exception(r.get("error"))
        if r["status"] == "PENDING":
            return {"status": "PENDING", "retry_after_seconds": r["retry_after_seconds"],
                    "note": "BRAIN is still preparing the record sets; call again later."}
        return r.get("data") or {}

    async def get_record_set_data(self, alpha_id: str, record_set_name: str,
                                  max_wait: float = 30) -> Dict[str, Any]:
        """One record set (pnl, sharpe, turnover, daily-pnl, yearly-stats, ...)."""
        await self.ensure_authenticated()
        url = (f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}/recordsets/"
               f"{_seg(record_set_name, 'record set name')}")
        r = await self._poll(url, max_wait)
        if r["status"] == "ERROR":
            raise Exception(r.get("error"))
        if r["status"] == "PENDING":
            return {"status": "PENDING", "retry_after_seconds": r["retry_after_seconds"],
                    "note": "BRAIN is still computing this record set; call again later."}
        return r.get("data") or {}

    async def get_user_activities(self, user_id: str, grouping: Optional[str] = None) -> Dict[str, Any]:
        """Activity diversity: alpha counts and data-diversity PASS/FAIL per group.

        GET /users/self/activities/diversity (grouping e.g. "region,delay" or
        "dataCategory,region,delay"). The old /users/{id}/activities endpoint only
        returns category names and ignored `grouping`. Only the current user is
        supported by the platform."""
        await self.ensure_authenticated()
        if str(user_id or "self").strip() not in ("self", "", await self._self_id_or_none()):
            raise ValueError("activity diversity is only available for the current user (user_id='self')")
        params = {}
        if grouping:
            if not re.fullmatch(r"[A-Za-z]+(,[A-Za-z]+)*", grouping):
                raise ValueError("grouping must be comma-separated field names, e.g. 'region,delay'")
            params['grouping'] = grouping
        response = await self._request('get', f"{self.base_url}/users/self/activities/diversity",
                                       params=params)
        if response.status_code >= 400:
            raise Exception(_http_error_detail(response, "activity diversity"))
        return response.json()

    async def get_payments(self) -> Dict[str, Any]:
        """Base payments (daily) and other payments (quarterly, competitions,
        referrals). Each half reports its own error instead of hiding the other."""
        async def fetch(kind: str) -> Any:
            try:
                resp = await self._request('get', f"{self.base_url}/users/self/activities/{kind}")
                if resp.status_code >= 400:
                    return {"error": _http_error_detail(resp, kind)}
                return resp.json() if (resp.text or "").strip() else {}
            except Exception as e:
                return {"error": f"{kind}: {e}"}

        await self.ensure_authenticated()
        base_payments, other_payments = await asyncio.gather(fetch("base-payment"), fetch("other-payment"))
        return {"base_payments": base_payments, "other_payments": other_payments}

    async def _self_id_or_none(self) -> Optional[str]:
        try:
            status = await self.get_authentication_status() or {}
            return (status.get("user") or {}).get("id")
        except Exception:
            return None

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
        """Your alpha distribution across pyramid categories (dates as YYYY-MM-DD)."""
        await self.ensure_authenticated()
        params = {}
        for key, value in (("startDate", start_date), ("endDate", end_date)):
            if value:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value[:10]):
                    raise ValueError(f"{key} must start with YYYY-MM-DD, got {value!r}")
                params[key] = value[:10]
        response = await self._request('get', f"{self.base_url}/users/self/activities/pyramid-alphas",
                                       params=params)
        if response.status_code >= 400:
            raise Exception(_http_error_detail(response, "pyramid alphas"))
        return response.json()

    async def get_user_competitions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get list of competitions that the user is participating in."""
        await self.ensure_authenticated()
        
        try:
            if not user_id:
                user_id = 'self'  # the API accepts "self" directly

            response = await self._request('get', f"{self.base_url}/users/{_seg(user_id, 'user id')}/competitions")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get user competitions: {str(e)}", "ERROR")
            raise
            
    async def get_competition_details(self, competition_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific competition."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/competitions/{_seg(competition_id, 'competition id')}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get competition details: {str(e)}", "ERROR")
            raise
            
    async def get_competition_agreement(self, competition_id: str) -> Dict[str, Any]:
        """Get the rules, terms, and agreement for a specific competition."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/competitions/{_seg(competition_id, 'competition id')}/agreement")
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
                                     competition: Optional[str] = None,
                                     max_wait: float = 30) -> Dict[str, Any]:
        """Portfolio before/after adding this alpha (sharpe, fitness, turnover, ...).

        Uses the catalogued before-and-after-performance endpoints (the competition
        variant when `competition` is given). The old /performance-comparison path is
        not a documented endpoint; team_id has no equivalent there and is ignored."""
        await self.ensure_authenticated()
        aid = _seg(alpha_id, 'alpha id')
        if competition:
            url = (f"{self.base_url}/competitions/{_seg(competition, 'competition id')}/alphas/"
                   f"{aid}/before-and-after-performance")
        else:
            url = f"{self.base_url}/users/self/alphas/{aid}/before-and-after-performance"
        r = await self._poll(url, max_wait)
        if r["status"] == "ERROR":
            raise Exception(r.get("error"))
        if r["status"] == "PENDING":
            return {"status": "PENDING", "retry_after_seconds": r["retry_after_seconds"]}
        out = r.get("data")
        if out is None:
            return {"status": "EMPTY", "note": "BRAIN returned no body for this alpha."}
        if team_id and isinstance(out, dict):
            out["note"] = "team_id is not supported by the before-and-after endpoint and was ignored."
        return out

    # --- New documentation endpoint ---
    
    async def get_documentation_page(self, page_id: str) -> Dict[str, Any]:
        """Retrieve detailed content of a specific documentation page/article."""
        await self.ensure_authenticated()
        
        try:
            response = await self._request('get', f"{self.base_url}/tutorial-pages/{_seg(page_id, 'page id')}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log(f"Failed to get documentation page: {str(e)}", "ERROR")
            raise

brain_client = BrainApiClient()
forum_client = ForumClient(brain_client.cookie_list)

# ProdMemo (prodmemo_service) holds no BRAIN client of its own: inject this one,
# which already carries credd-backed auth and 401 self-healing.
prodmemo_client.fetcher = brain_client


# --- Response shaping ---------------------------------------------------------

_SUMMARY_SETTING_KEYS = ("instrumentType", "region", "universe", "delay", "decay", "neutralization",
                         "truncation", "pasteurization", "nanHandling", "unitHandling", "language",
                         "testPeriod", "maxTrade", "maxPosition", "lookback", "selectionHandling",
                         "selectionLimit", "componentActivation")
_SUMMARY_IS_METRICS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
                       "longCount", "shortCount", "pnl", "bookSize", "startDate")


def _code_of(part: Any) -> Any:
    return part.get("code") if isinstance(part, dict) else part


def _alpha_summary(alpha: Dict[str, Any]) -> Dict[str, Any]:
    """One alpha without the bulky parts: identity, all settings, full code, IS
    metrics and every check (name/result/value/limit)."""
    if not isinstance(alpha, dict):
        return alpha
    settings = alpha.get("settings") or {}
    is_ = alpha.get("is") or {}
    checks = [c for c in is_.get("checks") or [] if isinstance(c, dict)]
    out: Dict[str, Any] = {
        "id": alpha.get("id"), "type": alpha.get("type"), "stage": alpha.get("stage"),
        "status": alpha.get("status"), "name": alpha.get("name"),
        "dateCreated": alpha.get("dateCreated"), "dateSubmitted": alpha.get("dateSubmitted"),
        "settings": {k: settings.get(k) for k in _SUMMARY_SETTING_KEYS if settings.get(k) is not None},
    }
    for part in ("regular", "combo", "selection"):
        code = _code_of(alpha.get(part))
        if code:
            out[part] = code
    out["is"] = {k: is_.get(k) for k in _SUMMARY_IS_METRICS if is_.get(k) is not None}
    if checks:
        out["is"]["checks"] = [{k: c.get(k) for k in ("name", "result", "value", "limit") if c.get(k) is not None}
                               for c in checks]
        out["is"]["failed"] = [c.get("name") for c in checks if c.get("result") == "FAIL"]
    if alpha.get("tags"):
        out["tags"] = [t.get("name", t) if isinstance(t, dict) else t for t in alpha["tags"]]
    if alpha.get("classifications"):
        out["classifications"] = [c.get("id") or c.get("name") for c in alpha["classifications"]
                                  if isinstance(c, dict)]
    if _alpha_pyramid_keys(alpha):
        out["pyramids"] = _alpha_pyramid_keys(alpha)
    for key in ("grade", "favorite", "hidden", "color", "category", "osmosisPoints"):
        if alpha.get(key) not in (None, False, ""):
            out[key] = alpha.get(key)
    return {k: v for k, v in out.items() if v not in (None, {}, [])}


def _alpha_list_row(alpha: Dict[str, Any]) -> Dict[str, Any]:
    """Listing row: the compact metric row plus what tells alphas apart in a list."""
    if not isinstance(alpha, dict):
        return alpha
    row = {"type": alpha.get("type"), "stage": alpha.get("stage"), "status": alpha.get("status"),
           "dateCreated": alpha.get("dateCreated"), "dateSubmitted": alpha.get("dateSubmitted"),
           "name": alpha.get("name"), **_compact_alpha_row(alpha)}
    if alpha.get("type") == "SUPER":
        row["expr"] = {"combo": _code_of(alpha.get("combo")), "selection": _code_of(alpha.get("selection"))}
    return {k: v for k, v in row.items() if v not in (None, [], {}, "")}


def _as_list(value: Any, name: str) -> List[Any]:
    """A tool argument that takes one value or a list of them."""
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        return [value]
    if isinstance(value, list):
        return list(value)
    raise ValueError(f"{name} must be a string or a list of strings")


# --- Simulation planning ------------------------------------------------------

# Accepted spellings of the simulation type (case-insensitive).
SIMULATION_TYPES = {
    "REGULAR": "REGULAR",
    "SUPER": "SUPER", "SA": "SUPER", "SUPERALPHA": "SUPER",
    "REGION_AGNOSTIC": "REGION_AGNOSTIC", "RA": "REGION_AGNOSTIC", "RAA": "REGION_AGNOSTIC",
}
SIMULATION_BATCH_MODES = ("auto", "single", "multi", "concurrent")
MAX_SIMULATIONS_PER_CALL = 10

# Defaults for the settings create_simulation leaves as None. RAA has its own:
# region ALL / delay 1 are fixed, and only LARGE / MEDIUM / SMALL universes exist.
_TYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "REGULAR": {"instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 1,
                "decay": 0.0, "neutralization": "NONE", "truncation": 0.0, "visualization": True},
    "REGION_AGNOSTIC": {"instrumentType": "EQUITY", "region": "ALL", "universe": "MEDIUM", "delay": 1,
                        "decay": 10.0, "neutralization": "SLOW_AND_FAST", "truncation": 0.08,
                        "visualization": False},
}
_TYPE_DEFAULTS["SUPER"] = _TYPE_DEFAULTS["REGULAR"]
_RAA_FIXED = {"instrumentType": "EQUITY", "region": "ALL", "delay": 1}

# per_alpha_settings keys: tool-style snake_case names map to the API's camelCase;
# camelCase is accepted as-is. language and lookback stay call-level because they
# change the payload shape.
_OVERRIDE_KEYS = {
    "instrument_type": "instrumentType", "region": "region", "universe": "universe",
    "delay": "delay", "decay": "decay", "neutralization": "neutralization",
    "truncation": "truncation", "pasteurization": "pasteurization",
    "unit_handling": "unitHandling", "nan_handling": "nanHandling",
    "max_trade": "maxTrade", "max_position": "maxPosition", "test_period": "testPeriod",
    "visualization": "visualization", "simulation_mode": "simulationMode",
    "selection_handling": "selectionHandling", "selection_limit": "selectionLimit",
    "component_activation": "componentActivation",
}
_OVERRIDE_KEYS.update({v: v for v in list(_OVERRIDE_KEYS.values())})


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


def _strip_strings_and_comments(expr: str) -> str:
    """Blank out quoted strings ('..' or "..") and # comments, keeping offsets."""
    out, quote_char, in_comment = [], None, False
    for ch in expr:
        if in_comment:
            in_comment = ch != "\n"
            out.append(ch if ch == "\n" else " ")
        elif quote_char:
            if ch == quote_char:
                quote_char = None
            out.append(" ")
        elif ch in ("'", '"'):
            quote_char = ch
            out.append(" ")
        elif ch == "#":
            in_comment = True
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _lint_expression(expr: str) -> List[str]:
    """Cheap pre-flight check: unbalanced parentheses and optional operator
    arguments passed positionally. Returns a list of problems (empty = OK)."""
    expr = _strip_strings_and_comments(expr)
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


def _check_raa_settings(settings: Dict[str, Any], prefix: str) -> None:
    """Platform rules for REGION_AGNOSTIC (violations fail the simulation outright)."""
    for key, fixed in _RAA_FIXED.items():
        if str(settings.get(key)).strip().upper() != str(fixed):
            raise ValueError(f"{prefix}an RAA simulation always runs with {key}={fixed!r}, "
                             f"got {settings.get(key)!r}")
        settings[key] = fixed
    universe = str(settings.get("universe") or "").strip().upper()
    if universe not in RAA_UNIVERSES:
        raise ValueError(f"{prefix}RAA universe must be one of {list(RAA_UNIVERSES)}, "
                         f"got {settings.get('universe')!r}")
    settings["universe"] = universe
    if (str(settings.get("maxTrade")).upper() == "ON"
            and str(settings.get("maxPosition")).upper() == "ON"):
        raise ValueError(f"{prefix}maxTrade and maxPosition cannot both be ON for an RAA simulation")


def _finalize_settings(sim_type: str, settings: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    settings = dict(settings)
    if sim_type == "REGION_AGNOSTIC":
        _check_raa_settings(settings, prefix)
    try:
        settings["simulationMode"], settings["visualization"] = _normalize_simulation_mode(
            settings.get("simulationMode"), settings.get("visualization"))
    except ValueError as e:
        raise ValueError(f"{prefix}{e}")
    return settings


def _lint_problems(codes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Lint each distinct FASTEXPR expression once (a settings sweep repeats one)."""
    seen: Dict[str, int] = {}
    rows = []
    for i, code in enumerate(codes):
        expr = code.get("regular")
        if expr is None or expr in seen:
            continue
        seen[expr] = i
        problems = _lint_expression(expr)
        if problems:
            rows.append({"index": i, "expr": expr[:120], "issues": problems})
    return rows


# --- MCP server -----------------------------------------------------------------

def _transport_security():
    """DNS-rebinding protection: FastMCP enables it for loopback binds; extend it to
    the host names a reverse proxy forwards (WQMCP_ALLOWED_HOSTS, comma-separated)."""
    from mcp.server.transport_security import TransportSecuritySettings
    extra = [h.strip() for h in os.environ.get("WQMCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    if not extra:
        if WQMCP_HOST not in ("127.0.0.1", "localhost", "::1"):
            logger.warning("WQMCP_HOST=%s without WQMCP_ALLOWED_HOSTS: no DNS-rebinding protection "
                           "and no authentication; put an authenticating reverse proxy in front.",
                           WQMCP_HOST)
        return None
    return TransportSecuritySettings(
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", *extra],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
                         *[f"https://{h}" for h in extra], *[f"http://{h}" for h in extra]])


# Default to loopback: this server has no authentication of its own. Set
# WQMCP_HOST=0.0.0.0 (behind an authenticating proxy) for remote clients.
WQMCP_HOST = os.environ.get("WQMCP_HOST", "127.0.0.1")
WQMCP_PORT = int(os.environ.get("WQMCP_PORT", "8761"))

mcp = FastMCP(
    "brain-platform-mcp",
    instructions=(
        "WorldQuant BRAIN research tools. Typical loop: get_platform_setting_options -> "
        "get_datasets / get_datafields / get_operators -> create_simulation (mode single / multi / "
        "concurrent; type REGULAR, SUPER (SA) or REGION_AGNOSTIC (RA/RAA); language FASTEXPR or "
        "PYTHON) -> get_simulation (poll with wait_seconds) -> check_alpha -> "
        "submit_alpha(confirm=True). Long-running BRAIN jobs answer RUNNING / PENDING with "
        "retry_after_seconds: call the same tool again instead of waiting idle. A failing tool "
        "returns {\"error\": ...}. prodmemo_* tools estimate Prod correlation locally."
    ),
    host=WQMCP_HOST,
    port=WQMCP_PORT,
    transport_security=_transport_security(),
)

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
WRITE_IDEMPOTENT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True,
                                   openWorldHint=True)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True)
LOCAL_READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
LOCAL_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)


def _tool(annotations: ToolAnnotations):
    """Register an MCP tool whose failures come back as {"error": ...}: callers such
    as scripts/prodmemo_daily_sync.py read that shape rather than MCP's isError."""
    def register(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                return {"error": str(e) or repr(e)}
        return mcp.tool(annotations=annotations)(wrapper)
    return register


def _write_guard(what: str) -> Optional[Dict[str, Any]]:
    """Error payload when writes are disabled (WQMCP_READ_ONLY=1), else None."""
    if READ_ONLY:
        return {"error": f"{what} is disabled: this server runs with WQMCP_READ_ONLY=1"}
    return None


# --- Account ------------------------------------------------------------------

@_tool(READ)
async def brain_status(refresh: bool = False) -> Dict[str, Any]:
    """
    🔐 Check the BRAIN login (managed by the credd daemon; this server holds no password).

    Every other tool self-heals an expired login, so this is only needed to diagnose
    a connection problem.

    Args:
        refresh: Force-pull fresh cookies from credd first; on failure the error says
            why (credd down, token mismatch, backoff, biometric verification pending).

    Returns:
        authenticated, user, token_expiry, plus the server's read_only / allow_submit switches.
    """
    switches = {"credd_url": CREDD_URL, "read_only": READ_ONLY, "allow_submit": ALLOW_SUBMIT}
    if refresh:
        auth = await brain_client.authenticate()
        return {"authenticated": True, **auth, **switches}
    status = await brain_client.get_authentication_status()
    if status is None:
        return {"authenticated": False, **switches,
                "note": "BRAIN rejected the session or credd is unreachable; call "
                        "brain_status(refresh=True) for the exact reason."}
    return {"authenticated": True, "user": status.get("user"),
            "token_expiry": (status.get("token") or {}).get("expiry"),
            "permissions": status.get("permissions"), **switches}


# --- Simulation -----------------------------------------------------------------

@_tool(WRITE)
async def create_simulation(
    expressions: Union[str, List[str], None] = None,
    type: str = "REGULAR",
    mode: str = "auto",
    combo: Union[str, List[str], None] = None,
    selection: Union[str, List[str], None] = None,
    language: str = "FASTEXPR",
    lookback: Optional[int] = None,
    instrument_type: Optional[str] = None,
    region: Optional[str] = None,
    universe: Optional[str] = None,
    delay: Optional[int] = None,
    decay: Optional[float] = None,
    neutralization: Optional[str] = None,
    truncation: Optional[float] = None,
    pasteurization: str = "ON",
    unit_handling: str = "VERIFY",
    nan_handling: str = "OFF",
    test_period: str = "P0Y0M",
    max_trade: str = "OFF",
    max_position: str = "OFF",
    visualization: Optional[bool] = None,
    simulation_mode: Optional[str] = None,
    selection_handling: str = "POSITIVE",
    selection_limit: int = 1000,
    component_activation: str = "IS",
    per_alpha_settings: Optional[List[Dict[str, Any]]] = None,
    validate_expressions: bool = True,
) -> Dict[str, Any]:
    """
    🚀 Submit simulations (backtests) — returns immediately; poll get_simulation.

    TYPE (what is simulated):
      - "REGULAR": expressions = alpha expression(s). language="PYTHON" makes each
        entry Python source (lookback then required).
      - "SUPER" (alias "SA"): combo + selection expressions. Pass lists (paired
        one-to-one, or one side a single string) to run several SuperAlphas.
      - "REGION_AGNOSTIC" (aliases "RA", "RAA"): one FASTEXPR expression run in
        GLB / USA / ASI / EUR at once -> an RA_PARENT alpha with up to 4 RA_CHILD
        alphas. region is always "ALL" and delay 1; universe must be LARGE, MEDIUM
        or SMALL (LARGE -> ASI MINVOL1M / EUR TOP2500 / GLB MINVOL1M / USA TOP3000,
        MEDIUM -> ASI MINVOL10M / EUR TOP1200 / GLB MINVOL10M / USA TOP2000,
        SMALL -> ASI TOP500 / EUR TOP800 / GLB TOPDIV3000 / USA TOP1000); maxTrade
        and maxPosition cannot both be ON. Every data field must exist in >= 2 of
        the 4 regions. One RAA takes 4 concurrent simulation slots.

    MODE (how several are run):
      - "single": exactly one simulation.
      - "multi": 2-10 REGULAR alphas (FASTEXPR or PYTHON) in ONE multi-simulation
        request (one parent id). BRAIN cancels the whole batch if any child fails,
        so FASTEXPR expressions are linted first and a problem refuses the batch.
      - "concurrent": 1-10 independent simulations submitted in parallel, one id
        each — for SUPER / REGION_AGNOSTIC batches, or REGULAR alphas that must not
        share one failure. Items beyond the account's concurrent-simulation limit
        come back RATE_LIMITED (resubmit those later).
      - "auto" (default): single for one item; multi for 2+ REGULAR; concurrent
        for 2+ SUPER / REGION_AGNOSTIC.

    SETTINGS: arguments left as None take the type's defaults — REGULAR / SUPER:
    region USA, universe TOP3000, delay 1, decay 0, neutralization NONE,
    truncation 0, visualization True; REGION_AGNOSTIC: universe MEDIUM, decay 10,
    neutralization SLOW_AND_FAST, truncation 0.08, visualization False. Use
    get_platform_setting_options for valid region / universe / neutralization
    combinations. test_period / unit_handling / nan_handling are dropped for
    PYTHON; selection_* / component_activation apply to SUPER only.
    simulation_mode: "QUICK" (core metrics only, no visualizations, no Theme /
    Competition / correlation checks, NOT directly submittable; forces
    visualization=False) or "FULL"; None = platform default (FULL).

    PER-ITEM SETTINGS: per_alpha_settings[i] overrides the settings for item i;
    with a single expression (or combo/selection pair) it is repeated for every
    entry, so a sweep is one call:
        expressions=["rank(x)"], per_alpha_settings=[{"decay": 3}, {"decay": 5},
        {"neutralization": "MARKET"}]
    Keys: decay, neutralization, truncation, max_trade, max_position, universe,
    region, delay, pasteurization, nan_handling, unit_handling, test_period,
    visualization, simulation_mode, instrument_type, selection_handling,
    selection_limit, component_activation (camelCase API names such as maxTrade
    also work). language / lookback are call-level only.

    validate_expressions: lint FASTEXPR expressions (unbalanced parentheses, an
    optional operator argument passed positionally such as ts_backfill(x, 250)
    instead of lookback=250). It blocks a multi batch; for single / concurrent the
    problems come back as lint_warnings and the simulations are still sent.

    Returns:
        {"status": "SUBMITTED", "mode", "type", "simulation_id" (single / multi) or
        "simulation_ids" (concurrent), "settings_used", "next": the get_simulation
        call}; "RATE_LIMITED" with retry_after_seconds when the account's slots are
        full; "PARTIAL" when only some concurrent items were accepted.
    """
    guard = _write_guard("create_simulation")
    if guard:
        return guard
    sim_type = SIMULATION_TYPES.get(str(type or "").strip().upper())
    if not sim_type:
        raise ValueError(f"type must be one of {sorted(SIMULATION_TYPES)}, got {type!r}")
    mode = str(mode or "auto").strip().lower()
    if mode not in SIMULATION_BATCH_MODES:
        raise ValueError(f"mode must be one of {list(SIMULATION_BATCH_MODES)}, got {mode!r}")
    language = str(language or "FASTEXPR").strip().upper()
    is_python = language == "PYTHON"
    if is_python and lookback is None:
        raise ValueError("lookback is required when language='PYTHON'")
    if is_python and sim_type == "REGION_AGNOSTIC":
        raise ValueError("REGION_AGNOSTIC simulations take a FASTEXPR expression (language='FASTEXPR')")

    # 1. The code of each simulation.
    if sim_type == "SUPER":
        if expressions:
            raise ValueError("SUPER (SA) simulations use combo + selection, not expressions")
        combos, selections = _as_list(combo, "combo"), _as_list(selection, "selection")
        if not combos or not selections:
            raise ValueError("SUPER (SA) simulations need both combo and selection")
        if len(combos) > 1 and len(selections) > 1 and len(combos) != len(selections):
            raise ValueError(f"{len(combos)} combos and {len(selections)} selections: pass lists of "
                             "the same length, or a single string on one side")
        codes = [{"combo": combos[i if len(combos) > 1 else 0],
                  "selection": selections[i if len(selections) > 1 else 0]}
                 for i in range(max(len(combos), len(selections)))]
    else:
        if combo or selection:
            raise ValueError(f"combo / selection are for type SUPER (SA); {sim_type} uses expressions")
        codes = [{"regular": e} for e in _as_list(expressions, "expressions")]
        if not codes:
            raise ValueError("expressions is required: one alpha expression (or Python source) per simulation")
    for i, code in enumerate(codes):
        if any(not isinstance(v, str) or not v.strip() for v in code.values()):
            raise ValueError(f"item {i}: every expression / combo / selection must be a non-empty string")

    # 2. Per-item overrides; a single code is repeated for each entry (a sweep).
    overrides_list = list(per_alpha_settings or [])
    if overrides_list and len(codes) == 1:
        codes = codes * len(overrides_list)
    if overrides_list and len(overrides_list) != len(codes):
        raise ValueError(f"per_alpha_settings has {len(overrides_list)} entries but there are "
                         f"{len(codes)} simulations; they must match one-to-one")
    n = len(codes)
    if n > MAX_SIMULATIONS_PER_CALL:
        raise ValueError(f"at most {MAX_SIMULATIONS_PER_CALL} simulations per call, got {n}")

    # 3. Mode.
    if mode == "auto":
        mode = "single" if n == 1 else ("multi" if sim_type == "REGULAR" else "concurrent")
    if mode == "single" and n != 1:
        raise ValueError(f"mode='single' takes one simulation, got {n}; use mode='multi' or 'concurrent'")
    if mode == "multi":
        if sim_type != "REGULAR":
            raise ValueError("a multi-simulation holds REGULAR alphas only (FASTEXPR or PYTHON); use "
                             "mode='concurrent' to run several SUPER / REGION_AGNOSTIC simulations in parallel")
        if n < 2:
            raise ValueError("mode='multi' needs 2-10 alphas (or 1 expression + 2-10 per_alpha_settings)")

    # 4. Settings: type defaults <- explicit arguments <- per-item overrides.
    explicit = {"instrumentType": instrument_type, "region": region, "universe": universe,
                "delay": delay, "decay": decay, "neutralization": neutralization,
                "truncation": truncation, "visualization": visualization}
    base: Dict[str, Any] = {**_TYPE_DEFAULTS[sim_type], **{k: v for k, v in explicit.items() if v is not None}}
    base.update(pasteurization=pasteurization, unitHandling=unit_handling, nanHandling=nan_handling,
                testPeriod=None if sim_type == "REGION_AGNOSTIC" else test_period,  # RAA carries none
                maxTrade=max_trade, maxPosition=max_position, language=language,
                lookback=lookback if is_python else None, simulationMode=simulation_mode,
                selectionHandling=selection_handling, selectionLimit=selection_limit,
                componentActivation=component_activation)

    items: List[SimulationData] = []
    children: List[Dict[str, Any]] = []
    for i, code in enumerate(codes):
        raw = overrides_list[i] if overrides_list else {}
        if not isinstance(raw, dict):
            raise ValueError(f"per_alpha_settings[{i}] must be an object, got {raw.__class__.__name__}")
        unknown = [k for k in raw if k not in _OVERRIDE_KEYS]
        if unknown:
            raise ValueError(f"per_alpha_settings[{i}] has unsupported keys {unknown}; allowed: "
                             f"{sorted(set(_OVERRIDE_KEYS.values()))} (or their snake_case forms)")
        override = {_OVERRIDE_KEYS[k]: v for k, v in raw.items() if v is not None}
        settings = _finalize_settings(sim_type, {**base, **override}, f"item {i}: " if n > 1 else "")
        items.append(SimulationData(type=sim_type, settings=SimulationSettings(**settings), **code))
        children.append({"index": i, "overrides": override} if override else {"index": i})

    lint = _lint_problems(codes) if validate_expressions and not is_python else []
    if lint and mode == "multi":
        return {"error": "Expression pre-check failed — one failing child cancels the whole multi-simulation",
                "problems": lint,
                "note": "Fix the expressions, pass validate_expressions=False to send anyway, or use "
                        "mode='concurrent' so each alpha runs on its own."}

    payloads = [brain_client.simulation_payload(item) for item in items]
    if mode == "single":
        result = await brain_client.create_simulation(items[0])
        ids = [result["simulation_id"]] if result.get("simulation_id") else []
    elif mode == "multi":
        result = await brain_client.create_multi_simulation(payloads)
        ids = [result["simulation_id"]] if result.get("simulation_id") else []
    else:
        result = await _submit_concurrently(items, children)
        ids = result.get("simulation_ids") or []
    result = {"mode": mode, "type": sim_type, **result}
    if mode == "multi" or (mode == "concurrent" and overrides_list):
        result["children"] = children
    try:  # the settings of an item without overrides (all of them when there are none)
        result["settings_used"] = brain_client.simulation_payload(SimulationData(
            type=sim_type, settings=SimulationSettings(**_finalize_settings(sim_type, base, "")),
            **codes[0]))["settings"]
    except ValueError:  # base itself invalid, but every item overrides the bad value
        pass
    if lint:
        result["lint_warnings"] = lint
    if ids:
        result["next"] = f"get_simulation(simulation_ids={ids!r}, wait_seconds=60)"
    return result


async def _submit_concurrently(items: List[SimulationData], children: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One POST per simulation, all at once; each item reports its own outcome."""
    async def submit(i: int, item: SimulationData) -> Dict[str, Any]:
        try:
            r = await brain_client.create_simulation(item)
        except Exception as e:
            r = {"status": "ERROR", "error": str(e)}
        return {**children[i], **{k: r[k] for k in ("status", "simulation_id", "progress_url",
                                                      "retry_after_seconds", "error") if k in r}}

    rows = list(await asyncio.gather(*(submit(i, item) for i, item in enumerate(items))))
    ids = [r["simulation_id"] for r in rows if r.get("status") == "SUBMITTED"]
    if len(ids) == len(rows):
        status = "SUBMITTED"
    elif ids:
        status = "PARTIAL"
    elif all(r.get("status") == "RATE_LIMITED" for r in rows):
        status = "RATE_LIMITED"
    else:
        status = "ERROR"
    out: Dict[str, Any] = {"status": status, "submitted": len(ids), "total": len(rows),
                           "simulation_ids": ids, "simulations": rows}
    if status == "ERROR":
        out["error"] = "no simulation was accepted; see simulations[].error"
    elif status != "SUBMITTED":
        out["retry_after_seconds"] = max((r.get("retry_after_seconds") or 0) for r in rows) or 30.0
        out["note"] = ("Items with status RATE_LIMITED hit the account's concurrent-simulation limit "
                       "(an RAA takes 4 slots): resubmit just those once running simulations finish.")
    return out


@_tool(READ)
async def get_simulation(simulation_ids: Union[str, List[str], None] = None, wait_seconds: float = 0,
                         compact: bool = True) -> Dict[str, Any]:
    """
    ⏳ Progress / result of simulations — single, multi, RAA, or several at once.

    Args:
        simulation_ids: One or more simulation ids (or the progress_url values)
            returned by create_simulation. Omit to list the simulations this server
            created recently.
        wait_seconds: Keep polling up to this long before answering (0 = check once;
            max 120). Use 30-60 to block briefly when there is nothing else to do.
        compact: True (default) = one short row per finished alpha: id, ops, sharpe,
            fitness, turnover, margin_bps, robust / sub / y2_sharpe, cluster, fails,
            set (universe / decay / neutralization / truncation / maxTrade) and expr.
            False = the full alpha object.

    Returns:
        For one id: RUNNING with progress (multi: completed_children / children)
        and retry_after_seconds; COMPLETE with the alpha (multi: alpha_results, one
        row per child; RAA: parent_alpha_id plus one metric row per region child);
        or the failure status with BRAIN's own error message. For several ids:
        {"status": RUNNING / COMPLETE / FINISHED_WITH_ERRORS, "simulations": [...]}.
    """
    refs = _as_list(simulation_ids, "simulation_ids")
    if not refs:
        recent = list(brain_client.recent_simulations)
        return {"recent_simulations": recent,
                "note": ("Pass simulation_ids to check them." if recent
                         else "No simulations were created by this server since it started.")}
    if len(refs) > 20:
        raise ValueError("at most 20 simulation ids per call")
    if len(refs) == 1:
        try:
            url = _simulation_url(refs[0])
        except ValueError as e:
            return {"error": f"{e}. Pass the simulation_id (or progress_url) returned by create_simulation"}
        return await brain_client.check_simulation_progress(url, wait_seconds, compact)

    async def one(ref: Any) -> Dict[str, Any]:
        try:
            url = _simulation_url(ref)
        except ValueError as e:
            return {"simulation_id": str(ref), "status": "ERROR", "error": str(e)}
        try:
            state = await brain_client.check_simulation_progress(url, wait_seconds, compact)
        except Exception as e:  # e.g. a network error after the wait ran out: not a verdict
            state = {"status": "UNKNOWN", "error": str(e)}
        if "progress_url" not in state and state.get("id"):
            # compact=False on a finished single: the raw alpha, whose own
            # "status" (UNSUBMITTED, ACTIVE...) is not the simulation's.
            state = {"status": "COMPLETE", "progress_url": url, "alpha": state}
        return {"simulation_id": url.rsplit("/", 1)[-1], **state}

    states = list(await asyncio.gather(*(one(r) for r in dict.fromkeys(refs))))
    counts = collections.Counter(str(s.get("status")) for s in states)
    if counts["RUNNING"] or counts["UNKNOWN"]:
        status = "RUNNING"
    elif counts["COMPLETE"] == len(states):
        status = "COMPLETE"
    else:
        status = "FINISHED_WITH_ERRORS"
    out: Dict[str, Any] = {"status": status, "counts": dict(counts), "simulations": states}
    if status == "RUNNING":
        out["retry_after_seconds"] = min((s.get("retry_after_seconds") or 5.0) for s in states
                                         if s.get("status") in ("RUNNING", "UNKNOWN"))
    return out


@_tool(DESTRUCTIVE)
async def cancel_simulation(simulation_id: str) -> Dict[str, Any]:
    """
    🛑 Cancel a queued / running simulation (DELETE) to free an account simulation slot.

    Args:
        simulation_id: The simulation id (or progress_url) from create_simulation.
    """
    guard = _write_guard("cancel_simulation")
    if guard:
        return guard
    return await brain_client.cancel_simulation(simulation_id)


@_tool(READ)
async def get_platform_setting_options() -> Dict[str, Any]:
    """Valid simulation settings: every instrument type / region / delay combination
    with its universes and neutralizations. Call it to fix an invalid or mismatched
    setting before simulating."""
    return await brain_client.get_platform_setting_options()


@_tool(READ)
async def preview_super_selection(
    selection: str,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    selection_limit: int = 1000,
    selection_handling: str = "POSITIVE",
    limit: int = 10,
    compact: bool = True,
) -> Dict[str, Any]:
    """
    🎯 Preview which of your alphas a SuperAlpha selection expression would pick.

    Args:
        selection: SuperAlpha selection expression
        instrument_type / region / delay: the SuperAlpha's settings
        selection_limit: Max number of alphas the selection may pick (10-1000)
        selection_handling: "POSITIVE", "NON_ZERO" or "NON_NAN"
        limit: How many selected alphas to return
        compact: One short row per alpha (False = full alpha objects)
    """
    data = await brain_client.run_selection(selection, instrument_type, region, delay,
                                            selection_limit, selection_handling, limit)
    if compact and isinstance(data, dict) and isinstance(data.get("results"), list):
        data = {**data, "results": [_alpha_list_row(a) for a in data["results"]]}
    return data


# --- Alphas -------------------------------------------------------------------

@_tool(READ)
async def list_alphas(
    stage: Optional[str] = "IS",
    status: Optional[str] = None,
    alpha_type: Optional[str] = None,
    limit: int = 30,
    offset: int = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    submission_start_date: Optional[str] = None,
    submission_end_date: Optional[str] = None,
    order: Optional[str] = None,
    hidden: Optional[bool] = None,
    compact: bool = True,
) -> Dict[str, Any]:
    """
    👤 List your alphas with filters, sorting and pagination.

    Args:
        stage: "IS" (unsubmitted, default), "OS" (submitted), "PROD"; None / "" = any
        status: e.g. "UNSUBMITTED", "ACTIVE", "DECOMMISSIONED"
        alpha_type: "REGULAR", "SUPER", "RA_PARENT" or "RA_CHILD"
        limit / offset: page size (1-100) and number of alphas to skip; the answer
            carries count (total) and next_offset (None on the last page)
        start_date / end_date: creation-date window, e.g. "2025-01-01T00:00:00Z"
        submission_start_date / submission_end_date: submission-date window (OS)
        order: e.g. "-dateCreated", "-dateSubmitted", "name" (prefix - = descending)
        hidden: True = only hidden alphas, False = only visible ones, None = both
        compact: One short row per alpha (metrics, failed checks, settings, code);
            False = full alpha objects
    """
    limit = max(1, min(int(limit or 30), 100))
    offset = max(0, int(offset or 0))
    data = await brain_client.get_user_alphas(
        stage=stage or None, limit=limit, offset=offset, start_date=start_date, end_date=end_date,
        submission_start_date=submission_start_date, submission_end_date=submission_end_date,
        order=order, hidden=hidden, status=status, alpha_type=alpha_type)
    results = data.get("results") or []
    count = data.get("count")
    out = {**data, "next_offset": (offset + len(results)
                                   if results and (count is None or offset + len(results) < count) else None)}
    if compact:
        out["results"] = [_alpha_list_row(a) for a in results]
    return out


@_tool(READ)
async def get_alpha(alpha_id: str, full: bool = False) -> Dict[str, Any]:
    """
    📋 One alpha: settings, full code, IS metrics and every check.

    An RA_PARENT (region-agnostic) alpha carries no metrics of its own: its answer
    is one metric row per region child (sharpe, fitness, turnover, returns,
    drawdown, margin_bps, 2Y and sub-universe sharpe, FAIL / WARNING check names)
    plus how many children pass every check.

    Args:
        alpha_id: The alpha id
        full: Return the raw BRAIN object (for an RA parent: added as "parent")
    """
    alpha = await brain_client.get_alpha_details(alpha_id)
    if alpha.get("type") in ("RA_PARENT", "REGION_AGNOSTIC") or alpha.get("children"):
        summary = await brain_client.get_raa_alpha(alpha_id, parent=alpha)
        if full:
            summary["parent"] = alpha
        else:
            summary.pop("parent", None)
        return summary
    return alpha if full else _alpha_summary(alpha)


@_tool(READ)
async def get_alpha_recordset(alpha_id: str, recordset: Optional[str] = None,
                              wait_seconds: float = 30, max_rows: int = 0) -> Dict[str, Any]:
    """
    📈 An alpha's time series / tables: pnl, daily-pnl, sharpe, turnover, yearly-stats, ...

    Args:
        alpha_id: The alpha id
        recordset: e.g. "pnl", "yearly-stats", "daily-pnl", "sharpe", "turnover";
            omit to list the record sets available for this alpha
        wait_seconds: How long to keep polling while BRAIN computes it (max 300)
        max_rows: Keep only the most recent rows (0 = all)

    Returns:
        BRAIN's {schema, records}; status PENDING means BRAIN is still computing
        it: call again after retry_after_seconds.
    """
    if not recordset:
        data = await brain_client.get_record_sets(alpha_id, wait_seconds)
        return {"alpha_id": alpha_id, **(data if isinstance(data, dict) else {"results": data})}
    data = await brain_client.get_record_set_data(alpha_id, recordset, wait_seconds)
    data = data if isinstance(data, dict) else {"result": data}
    records = data.get("records")
    if max_rows and isinstance(records, list) and len(records) > max_rows:
        data = {**data, "records": records[-max_rows:],
                "truncated": f"showing the last {max_rows} of {len(records)} rows"}
    return {"alpha_id": alpha_id, "recordset": recordset, **data}


_CHECK_KINDS = ("submission", "correlation", "prod", "self", "power-pool", "all")


@_tool(READ)
async def check_alpha(alpha_id: str, check: str = "submission", wait_seconds: float = 60,
                      threshold: float = 0.7, include_data: bool = False) -> Dict[str, Any]:
    """
    ✅ BRAIN's checks for an alpha: pre-submission checks and / or correlations.

    Args:
        alpha_id: The alpha id (for an RAA use the PARENT id for "submission")
        check:
            "submission" (default) — the same checks as the Submit button: status
                DONE / PENDING, all_passed, failed / pending / errored names and
                each check's result / value / limit. When PROD_CORRELATION comes
                back as ERROR (platform busy) the prod-correlation endpoint is
                asked instead (prod_fallback).
            "prod" / "self" / "power-pool" — that correlation: max_correlation,
                passes_check (max < threshold), the 3 most correlated alphas.
            "correlation" — prod and self together.
            "all" — submission and correlation together.
          Correlation status DONE / PENDING (still computing, call again — NOT a
          failure) / ERROR. Measured prod / self values are saved into ProdMemo.
        wait_seconds: How long to keep polling while BRAIN computes (max 300)
        threshold: Correlation pass threshold (default 0.7)
        include_data: Also return the raw correlation payloads (large)
    """
    kind = str(check or "submission").strip().lower()
    if kind not in _CHECK_KINDS + ("production", "power_pool", "both"):
        raise ValueError(f"check must be one of {list(_CHECK_KINDS)}, got {check!r}")
    if kind == "submission":
        return await brain_client.get_submission_check(alpha_id, wait_seconds)
    if kind != "all":
        ctype = "both" if kind in ("correlation", "both") else kind
        return await brain_client.check_correlation(alpha_id, ctype, threshold, wait_seconds, include_data)
    submission, correlation = await asyncio.gather(
        brain_client.get_submission_check(alpha_id, wait_seconds),
        brain_client.check_correlation(alpha_id, "both", threshold, wait_seconds, include_data),
        return_exceptions=True)
    return {"alpha_id": alpha_id,
            "submission": ({"error": str(submission)} if isinstance(submission, BaseException) else submission),
            "correlation": ({"error": str(correlation)} if isinstance(correlation, BaseException) else correlation)}


@_tool(DESTRUCTIVE)
async def submit_alpha(alpha_id: str, confirm: bool = False, wait_seconds: float = 60) -> Dict[str, Any]:
    """
    📤 Submit an alpha to BRAIN. Irreversible, so confirm=False (default) is a dry run.

    confirm=False returns the pre-submission checks (as check_alpha) without
    submitting. confirm=True submits and follows BRAIN's asynchronous submission
    (Retry-After) up to wait_seconds (max 300).

    Returns:
        success and status: SUBMITTED, SUBMITTED_WITH_PENDING_CHECKS, REJECTED
        (see failed / checks), PENDING (still processing — call again with
        confirm=True; it resumes polling and never submits twice), RATE_LIMITED
        or ERROR.

    Args:
        alpha_id: The alpha to submit (for an RAA: the RA_PARENT id)
        confirm: True = really submit
        wait_seconds: How long to follow the submission before answering
    """
    if not confirm:
        report = await brain_client.get_submission_check(alpha_id, wait_seconds)
        return {**report, "dry_run": True,
                "note": "Not submitted: these are BRAIN's pre-submission checks. Call "
                        "submit_alpha(alpha_id, confirm=True) to submit."}
    guard = _write_guard("submit_alpha")
    if guard:
        return guard
    if not ALLOW_SUBMIT:
        return {"error": "submissions are disabled on this server (WQMCP_ALLOW_SUBMIT=0); "
                         "submit_alpha(confirm=False) still shows whether the alpha would pass"}
    return await brain_client.submit_alpha(alpha_id, wait_seconds)


@_tool(WRITE_IDEMPOTENT)
async def update_alpha(
    alpha_ids: Union[str, List[str]],
    name: Optional[str] = None,
    color: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    regular_desc: Optional[str] = None,
    selection_desc: Optional[str] = None,
    combo_desc: Optional[str] = None,
    osmosis_points: Optional[int] = None,
    favorite: Optional[bool] = None,
    hidden: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    ✏️ Update alpha properties.

    favorite / hidden (and color, for several alphas) apply to every id in one
    bulk request. name, category, tags, descriptions and osmosis_points apply to
    a single alpha.

    Args:
        alpha_ids: One alpha id or a list (up to 100)
        name / color / category / tags: alpha metadata ([] clears tags)
        regular_desc / selection_desc / combo_desc: descriptions (SUPER: selection / combo)
        osmosis_points: 1-100000
        favorite / hidden: flags
    """
    guard = _write_guard("update_alpha")
    if guard:
        return guard
    ids = [str(a).strip() for a in _as_list(alpha_ids, "alpha_ids")]
    if not ids or len(ids) > 100:
        raise ValueError("alpha_ids must hold 1-100 alpha ids")
    single = {"name": name, "category": category, "tags": tags, "regular_desc": regular_desc,
              "selection_desc": selection_desc, "combo_desc": combo_desc, "osmosis_points": osmosis_points}
    single = {k: v for k, v in single.items() if v is not None}
    if single and len(ids) != 1:
        raise ValueError(f"{sorted(single)} can only be set on one alpha at a time")
    bulk = {k: v for k, v in (("favorite", favorite), ("hidden", hidden),
                               ("color", color if len(ids) > 1 else None)) if v is not None}
    if not single and not bulk and color is None:
        raise ValueError("nothing to update")
    out: Dict[str, Any] = {"alpha_ids": ids}
    if len(ids) == 1 and (single or color is not None):
        updated = await brain_client.set_alpha_properties(
            ids[0], name=name, color=color, category=category, regular_desc=regular_desc,
            selection_desc=selection_desc, combo_desc=combo_desc, osmosis_points=osmosis_points,
            tags=tags)
        out["alpha"] = _alpha_summary(updated) if isinstance(updated, dict) else updated
    if bulk:
        out.update(await brain_client.update_alphas_bulk(ids, bulk))
    return out


@_tool(READ)
async def get_alpha_performance(alpha_id: str, competition_id: Optional[str] = None,
                                wait_seconds: float = 30) -> Dict[str, Any]:
    """
    📊 How adding this alpha changes your portfolio: before / after stats (sharpe,
    fitness, turnover, returns, drawdown, margin), yearly stats and PnL.

    Args:
        alpha_id: The alpha id
        competition_id: Show the competition's before / after view instead
        wait_seconds: How long to keep polling while BRAIN computes it
    """
    return await brain_client.performance_comparison(alpha_id, None, competition_id, wait_seconds)


# --- Data -----------------------------------------------------------------------

@_tool(READ)
async def get_datasets(
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: str = "false",
    search: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    📚 Datasets available for a region / delay / universe.

    Args:
        instrument_type / region / delay / universe: see get_platform_setting_options
        theme: Theme filter
        search: Free-text search
        limit: Page size (1-50; default = BRAIN's 20). Response `count` is the total.
        offset: Skip this many datasets (paging)
    """
    return await brain_client.get_datasets(instrument_type, region, delay, universe, theme, search,
                                           limit, offset)


@_tool(READ)
async def get_datafields(
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: str = "false",
    dataset_id: Optional[str] = None,
    data_type: str = "",
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    🔍 Data fields usable in alpha expressions.

    Args:
        instrument_type / region / delay / universe: see get_platform_setting_options
        theme: Theme filter
        dataset_id: Only fields of this dataset
        data_type: "MATRIX", "VECTOR", "GROUP" ... ("" / "ALL" = any)
        search: Search term
        limit: Page size (1-100, default 50). Response `count` is the total.
        offset: Skip this many fields (paging)
    """
    return await brain_client.get_datafields(instrument_type, region, delay, universe, theme, dataset_id,
                                             data_type, search, limit, offset)


@_tool(READ)
async def get_operators(category: Optional[str] = None, name: Optional[str] = None,
                        detail: bool = False) -> Dict[str, Any]:
    """
    🔧 Operators available in expressions.

    Args:
        category: e.g. "Arithmetic", "Time Series", "Cross Sectional", "Group"
        name: Substring match on the operator name
        detail: Full objects (complete descriptions, documentation links) instead of
            name / category / definition / scope / description (first 200 chars)
    """
    data = await brain_client.get_operators()
    ops = data.get("operators") if isinstance(data, dict) and "operators" in data else data
    if not isinstance(ops, list):
        return data
    picked = [op for op in ops if isinstance(op, dict)
              and (not category or str(op.get("category", "")).lower() == category.lower())
              and (not name or name.lower() in str(op.get("name", "")).lower())]
    rows = picked if detail else [
        {"name": op.get("name"), "category": op.get("category"), "definition": op.get("definition"),
         "scope": op.get("scope"), "description": (op.get("description") or "")[:200]}
        for op in picked]
    return {"count": len(rows), "results": rows,
            "categories": sorted({str(op.get("category")) for op in ops
                                  if isinstance(op, dict) and op.get("category")})}


# --- Account activity -----------------------------------------------------------

_ACTIVITY_KINDS = ("diversity", "pyramid-alphas", "pyramid-multipliers", "payments",
                   "diversity-score", "profile")


@_tool(READ)
async def get_activity(kind: str, grouping: Optional[str] = None, start_date: Optional[str] = None,
                       end_date: Optional[str] = None, user_id: str = "self") -> Dict[str, Any]:
    """
    🧭 Your BRAIN activity, scores and profile.

    Args:
        kind:
            "diversity" — alpha counts and data-diversity PASS / FAIL per group
                (grouping e.g. "region,delay" or "dataCategory,region,delay").
            "pyramid-alphas" — your alphas per pyramid category (start_date /
                end_date as YYYY-MM-DD).
            "pyramid-multipliers" — BRAIN's current pyramid multipliers.
            "payments" — base payments (daily) and other payments (quarterly,
                competitions, referrals).
            "diversity-score" — client-side estimate of the value-factor trend for
                the REGULAR alphas submitted between start_date and end_date (ISO,
                both required): diversity_score = S_A * S_P * S_H with N, A, P,
                P_max and per-pyramid counts.
            "profile" — your full record (user_id "self") or another user's public
                profile.
        grouping / start_date / end_date / user_id: see kind
    """
    kind = str(kind or "").strip().lower()
    if kind == "diversity":
        return await brain_client.get_user_activities(user_id or "self", grouping)
    if kind == "pyramid-alphas":
        return await brain_client.get_pyramid_alphas(start_date, end_date)
    if kind == "pyramid-multipliers":
        return await brain_client.get_pyramid_multipliers()
    if kind == "payments":
        return await brain_client.get_payments()
    if kind == "diversity-score":
        if not start_date or not end_date:
            raise ValueError("diversity-score needs start_date and end_date")
        return await brain_client.value_factor_trendScore(start_date=start_date, end_date=end_date)
    if kind == "profile":
        return await brain_client.get_user_profile(user_id or "self")
    raise ValueError(f"kind must be one of {list(_ACTIVITY_KINDS)}, got {kind!r}")


# --- Community ------------------------------------------------------------------

@_tool(READ)
async def get_leaderboard(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    🏅 Consultant leaderboard row for a user (default: you).

    Args:
        user_id: Another user's id
    """
    return await brain_client.get_leaderboard(user_id)


@_tool(READ)
async def get_competitions(competition_id: Optional[str] = None, include_agreement: bool = False,
                           user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    🏆 Competitions: the ones a user takes part in, or one competition's details.

    Args:
        competition_id: Return this competition's details instead of the list
        include_agreement: With competition_id, also fetch its rules / agreement
        user_id: Whose competitions to list (default: you)
    """
    if not competition_id:
        return await brain_client.get_user_competitions(user_id)
    details = await brain_client.get_competition_details(competition_id)
    if include_agreement:
        try:
            details = {**details, "agreement": await brain_client.get_competition_agreement(competition_id)}
        except Exception as e:  # the agreement is best effort
            details = {**details, "agreement": {"error": str(e)}}
    return details


@_tool(READ)
async def get_events() -> Dict[str, Any]:
    """🗓️ BRAIN events (webinars, meetups)."""
    return await brain_client.get_events()


@_tool(READ)
async def get_messages(limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
    """
    💬 Your BRAIN announcements and notifications (embedded images are stripped).

    Args:
        limit: Maximum number of messages to return
        offset: Number of messages to skip (paging)
    """
    return await brain_client.get_messages(limit, offset)


@_tool(READ)
async def get_documentation(page_id: Optional[str] = None) -> Dict[str, Any]:
    """
    📖 Official BRAIN documentation.

    Args:
        page_id: Omit for the tutorials with their pages (ids + titles); pass a page
            id for that page's content.
    """
    if page_id:
        return await brain_client.get_documentation_page(page_id)
    return await brain_client.get_documentations()


# --- Forum (support.worldquantbrain.com, headless browser) ----------------------

@_tool(READ)
async def search_forum_posts(search_query: str, max_results: int = 50) -> Dict[str, Any]:
    """
    🔍 Search the BRAIN community forum.

    Args:
        search_query: Search term or phrase
        max_results: Maximum number of results (default 50)
    """
    res = await forum_client.search_posts(search_query, max_results=max_results)
    return {**res, "success": True, "total_found": res.get("count")}


@_tool(READ)
async def read_forum_post(article_id: str, include_comments: bool = True) -> Dict[str, Any]:
    """
    📄 Read one forum post or article (body keeps line breaks and code) with its comments.

    Args:
        article_id: Post / article id (e.g. "32984819083415-新人求模板"), "posts/<id>",
            "articles/<id>" or a support.worldquantbrain.com URL
        include_comments: Also return the comments
    """
    res = await forum_client.read_post(article_id, include_comments=include_comments)
    return {**res, "success": True}


@_tool(READ)
async def get_glossary_terms() -> Dict[str, Any]:
    """📚 BRAIN glossary terms and definitions from the support site (cached)."""
    return await forum_client.get_glossary_terms()


# --- ProdMemo: local Self / Pool / Prod correlation memory ----------------------
# Thin wrappers over prodmemo_service (see docs/PRODMEMO_IMPLEMENTATION.md).

@_tool(LOCAL_WRITE)
async def prodmemo_sync(mode: str = "incremental") -> Dict[str, Any]:
    """
    Sync submitted alphas and their PnL into the local ProdMemo database.

    Runs in the background and returns immediately — poll prodmemo_sync("status").

    Args:
        mode: "incremental" (default: probes the remote count first and fetches only
            what is missing), "full" (re-walks every alpha), "stop" (cancels a run
            in progress) or "status" (progress / final state of the latest run)
    """
    if str(mode or "").strip().lower() == "status":
        return await prodmemo_client.sync_status()
    return await prodmemo_client.start_sync(mode)


@_tool(LOCAL_READ)
async def prodmemo_check(alpha_id: str = "", alpha_ids: Optional[List[str]] = None,
                         run_platform_check: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """
    Estimate Prod Correlation locally for one or many alphas (no platform quota).

    Per alpha (compact row):
      prod_est          — empirical estimate prod ≈ a + b × pool (the most useful
                          number; default a=0.428 b=1.139, refitted automatically
                          once >= 8 alphas have a measured platform Prod)
      pool / self       — local Pool / Self correlation max
      prod_lower_bound  — conditional lower bound from reference curves; > 0.7
                          proves the platform check would fail
      platform_prod     — platform-measured Prod, if known
      recommendation    — "skip" (bound > 0.7), "check", or "insufficient_data"
    Platform Prod values measured by check_alpha are written back automatically
    and become new reference / calibration points.

    Args:
        alpha_id: One alpha to evaluate
        alpha_ids: Several alphas (up to 20) — use instead of alpha_id
        run_platform_check: Also query the platform for Prod / Self correlation and
            store the result (costs platform time; improves future estimates)
        verbose: Return the full per-alpha report (local details, witness,
            calibration, resolved values) instead of the compact row
    """
    ids = list(alpha_ids or []) + ([alpha_id] if alpha_id else [])
    result = await prodmemo_client.check_many(ids, run_platform_check, verbose)
    return result["results"][0] if len(result["results"]) == 1 and not alpha_ids else result


@_tool(LOCAL_READ)
async def prodmemo_get(alpha_id: str = "", stale_only: bool = False,
                       above: float = 0.0, group_key: str = "",
                       limit: int = 100) -> Dict[str, Any]:
    """
    Inspect stored ProdMemo state — one alpha in full, or a filtered list.

    Each entry reports sync state (metadata / PnL present, PnL last date), platform
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
    return await prodmemo_client.get(alpha_id=alpha_id, stale_only=stale_only,
                                     above=above, group_key=group_key, limit=limit)


@_tool(LOCAL_READ)
async def prodmemo_stats() -> Dict[str, Any]:
    """Counts held in the ProdMemo database, including how many alphas are usable
    as Prod lower-bound reference curves (valid_reference_count)."""
    return await prodmemo_client.stats()


@_tool(LOCAL_WRITE)
async def prodmemo_manage(action: str, alpha_id: str = "", data: str = "") -> Dict[str, Any]:
    """
    Maintain the ProdMemo store: export, import, or clear correlation data.

    'import' accepts the WebDataScope browser extension's Corr JSON export — the
    only way to carry over platform Prod values captured in the browser, which
    cannot be re-derived server-side. Clearing is NOT reversible: 'clear_corrs'
    drops correlations but keeps alphas / PnL, 'clear_sync' does the opposite (the
    surviving local correlations then read as stale).

    Args:
        action: "export" | "import" | "clear_corrs" | "clear_sync" | "delete"
        alpha_id: Alpha to drop, for action="delete"
        data: Corr JSON payload, for action="import"
    """
    return await prodmemo_client.manage(action, alpha_id=alpha_id, data=data)


# --- Main entry point ---
if __name__ == "__main__":
    transport = os.environ.get("WQMCP_TRANSPORT", "streamable-http")
    if transport != "stdio":  # stdout carries the protocol in stdio mode
        print(f"running the server on http://{WQMCP_HOST}:{WQMCP_PORT}/mcp", file=sys.stderr)
    mcp.run(transport=transport)

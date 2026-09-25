"""WorldQuant BRAIN HTTP client used by the wqmcp MCP server.

Endpoint behaviour follows "WQ API Catalog 1.15.3" (reverse-engineered from the
BRAIN web frontend, not an official contract). See API_AUDIT.md for the findings
this module addresses.

Login is owned by the credd daemon: this process never sees the BRAIN password.
Cookies are pulled from ``GET {CREDD_URL}/cookies`` and re-pulled once when BRAIN
answers 401. All HTTP I/O runs in a bounded thread pool so that many concurrent
MCP clients never block the shared event loop.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlsplit

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger("wqmcp")

BASE_URL = os.environ.get("WQMCP_BASE_URL", "https://api.worldquantbrain.com").rstrip("/")
CREDD_URL = os.environ.get("CREDD_URL", "http://127.0.0.1:8762").rstrip("/")
CREDD_TOKEN = os.environ.get("CREDD_TOKEN", "")
# Send the catalog's versioned Accept header per endpoint (set to 0 to disable).
SEND_ACCEPT_VERSIONS = os.environ.get("WQMCP_ACCEPT_VERSIONS", "1") != "0"
# Fill unspecified simulation settings from the user's saved platform defaults.
USE_PLATFORM_DEFAULTS = os.environ.get("WQMCP_USE_PLATFORM_DEFAULTS", "1") != "0"

MAX_WAIT_SECONDS = 300.0

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class BrainError(Exception):
    """Base class for errors surfaced to MCP clients."""


class InvalidArgument(BrainError, ValueError):
    """A tool argument failed validation before any request was sent."""


class CreddUnavailable(BrainError):
    """credd itself is unreachable or returned an error (distinct from a BRAIN 401)."""


class BrainAPIError(BrainError):
    """BRAIN answered with an HTTP error status."""

    def __init__(self, status: int, method: str, path: str, message: str,
                 retry_after: Optional[float] = None) -> None:
        self.status = status
        self.method = method
        self.path = path
        self.message = message
        self.retry_after = retry_after
        hint = f" (retry after ~{int(retry_after)}s)" if retry_after else ""
        super().__init__(f"BRAIN {method} {path} -> HTTP {status}: {message}{hint}")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$")
_TEST_PERIOD_RE = re.compile(r"^P\d+Y\d+M(\d+D)?$")


def seg(value: Any, name: str) -> str:
    """Validate and percent-encode one URL path segment (ids, names).

    Rejects anything that could change the path (``/``, ``..``, ``?``, ``#``,
    whitespace), so a tool argument can never address another endpoint.
    """
    text = str(value).strip() if value is not None else ""
    if not _SEGMENT_RE.match(text) or ".." in text:
        raise InvalidArgument(f"invalid {name}: {value!r}")
    return quote(text, safe="")


def check_date(value: Optional[str], name: str, *, allow_time: bool = False) -> Optional[str]:
    if value is None or value == "":
        return None
    pattern = _DATETIME_RE if allow_time else _DATE_RE
    if not pattern.match(value):
        fmt = "ISO 8601 date or date-time" if allow_time else "YYYY-MM-DD"
        raise InvalidArgument(f"{name} must be {fmt}, got {value!r}")
    return value


def retry_after_seconds(response: requests.Response) -> float:
    """Parse Retry-After (seconds). Missing / unparsable -> 0.0."""
    try:
        return max(0.0, float(response.headers.get("Retry-After", 0)))
    except (TypeError, ValueError):
        return 0.0


def clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    return max(lo, min(hi, v))


def error_message(response: requests.Response) -> str:
    """Best-effort human-readable error from a BRAIN error response."""
    text = (response.text or "").strip()
    if not text:
        return response.reason or "no response body"
    try:
        body = response.json()
    except ValueError:
        return text[:500]
    if isinstance(body, dict):
        for key in ("detail", "message", "error", "details"):
            val = body.get(key)
            if isinstance(val, str) and val:
                return val[:500]
        parts = []
        for key, val in body.items():
            if isinstance(val, list):
                val = "; ".join(str(v) for v in val)
            parts.append(f"{key}: {val}")
        return ", ".join(parts)[:500] or text[:500]
    return str(body)[:500]


def trim(text: Any, limit: int) -> Any:
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + "…"
    return text


def page_meta(data: Any, limit: int, offset: int) -> Dict[str, Any]:
    count = data.get("count") if isinstance(data, dict) else None
    results = (data.get("results") if isinstance(data, dict) else None) or []
    has_more = (count is not None and offset + len(results) < count) or bool(
        isinstance(data, dict) and data.get("next"))
    meta: Dict[str, Any] = {"count": count, "limit": limit, "offset": offset, "returned": len(results),
                            "has_more": has_more}
    if has_more:
        meta["next_offset"] = offset + len(results)
    return meta


# Accept versions from the catalog; everything else uses 2.0.
_ACCEPT_RULES: Tuple[Tuple[str, re.Pattern, str], ...] = tuple(
    (m, re.compile(p), v) for m, p, v in (
        ("OPTIONS", r"^/simulations$", "3.0"),
        ("GET", r"^/users/self/alphas$", "4.0"),
        ("GET", r"^/users/self/alphas/summary$", "4.0"),
        ("GET", r"^/alphas/[^/]+/alphas$", "4.0"),
        ("OPTIONS", r"^/users/[^/]+/alphas$", "4.0"),
        ("GET", r"^/tags/[^/]+/alphas$", "4.0"),
        ("GET", r"^/suggest/fields$", "4.0"),
        ("GET", r"^/data-fields/summary$", "3.0"),
        ("GET", r"^/alphas/[^/]+/tutorial$", "3.0"),
        ("GET", r"^/users/self/activities/base-payment$", "3.0"),
    )
)


def accept_header(method: str, path: str) -> str:
    if not SEND_ACCEPT_VERSIONS:
        return "application/json"
    for m, pattern, version in _ACCEPT_RULES:
        if m == method.upper() and pattern.match(path):
            return f"application/json;version={version}"
    return "application/json;version=2.0"


# --------------------------------------------------------------------------- #
# credd-backed requests session
# --------------------------------------------------------------------------- #


class CreddSession(requests.Session):
    """requests.Session whose BRAIN cookies come from credd's HTTP interface.

    - cookies are pulled from ``GET {CREDD_URL}/cookies`` (``X-Auth-Token`` when
      ``CREDD_TOKEN`` is set) and swapped in as one jar;
    - a BRAIN 401 re-pulls once and retries. Concurrent 401s refresh only once:
      a thread whose request used an older cookie version just retries with the
      jar another thread already refreshed;
    - credd's ``X-Cookie-Version`` prevents a 401 -> refetch -> 401 spin while
      credd is in backoff.
    """

    def __init__(self, base_url: str = BASE_URL, *, credd_timeout: float = 15) -> None:
        super().__init__()
        parts = urlsplit(base_url)
        self._brain_host = parts.hostname or ""
        self._secure = parts.scheme == "https"
        self._cookie_domain = (".worldquantbrain.com" if self._brain_host.endswith("worldquantbrain.com")
                               else self._brain_host)
        self._credd_timeout = credd_timeout
        self._refresh_lock = threading.Lock()
        self._cookie_version: Optional[str] = None
        self.refresh_cookies()

    def _fetch_from_credd(self) -> Tuple[Dict[str, str], Optional[str]]:
        headers = {"X-Auth-Token": CREDD_TOKEN} if CREDD_TOKEN else None
        try:
            r = requests.get(f"{CREDD_URL}/cookies", headers=headers, timeout=self._credd_timeout)
        except requests.RequestException as exc:
            raise CreddUnavailable(f"credd unreachable (is creds-daemon running?): {exc}") from exc
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                body = {"error": "error", "detail": r.text[:200]}
            code = body.get("error")
            msg = f"credd returned {r.status_code}: {code} - {body.get('detail')}"
            if code == "biometric_required":
                msg += ". Complete the biometric check in credd (see its /status), then retry"
            elif code == "unauthorized":
                msg += ". CREDD_TOKEN does not match credd's token"
            elif code in ("backoff", "rate_limited"):
                msg += f". Retry after ~{body.get('retry_after', '?')}s"
            raise CreddUnavailable(msg)
        try:
            cookies = r.json()
        except ValueError as exc:
            raise CreddUnavailable("credd returned a non-JSON cookie payload") from exc
        if not isinstance(cookies, dict):
            raise CreddUnavailable("credd returned an unexpected cookie payload")
        return cookies, r.headers.get("X-Cookie-Version")

    @property
    def cookie_version(self) -> Optional[str]:
        return self._cookie_version

    def refresh_cookies(self, stale_version: Optional[str] = None) -> Optional[str]:
        """Replace the cookie jar from credd and return the new cookie version.

        With ``stale_version``: if the jar has already moved past that version
        (another thread refreshed while we waited for the lock), skip the fetch.
        """
        with self._refresh_lock:
            if stale_version is not None and self._cookie_version != stale_version:
                return self._cookie_version
            cookies, version = self._fetch_from_credd()
            jar = requests.cookies.RequestsCookieJar()
            for name, value in cookies.items():
                jar.set(name, str(value), domain=self._cookie_domain, path="/", secure=self._secure)
            self.cookies = jar
            self._cookie_version = version
            return version

    def cookie_list(self) -> List[Dict[str, Any]]:
        return [{"name": c.name, "value": c.value, "domain": c.domain, "path": c.path,
                 "secure": c.secure, "httpOnly": True} for c in self.cookies]

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        version_used = self._cookie_version
        resp = super().request(method, url, *args, **kwargs)
        if resp.status_code != 401 or urlsplit(str(url)).hostname != self._brain_host:
            return resp
        try:
            new_version = self.refresh_cookies(stale_version=version_used)
        except CreddUnavailable:
            return resp  # surface the original 401
        if version_used is not None and new_version == version_used:
            return resp  # credd has nothing newer (likely in backoff)
        return super().request(method, url, *args, **kwargs)


# --------------------------------------------------------------------------- #
# Response shaping
# --------------------------------------------------------------------------- #

_IS_METRICS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
               "longCount", "shortCount", "pnl", "bookSize", "startDate")
_SETTING_KEYS = ("instrumentType", "region", "universe", "delay", "decay", "neutralization",
                 "truncation", "pasteurization", "nanHandling", "unitHandling", "language",
                 "testPeriod", "maxTrade", "maxPosition", "lookback", "selectionHandling",
                 "selectionLimit", "componentActivation")


def _code(part: Any) -> Any:
    if isinstance(part, dict):
        return part.get("code")
    return part


def summarize_checks(checks: Any) -> Dict[str, Any]:
    items = [c for c in (checks or []) if isinstance(c, dict)]
    compact = [{k: c.get(k) for k in ("name", "result", "limit", "value") if c.get(k) is not None}
               for c in items]
    failed = [c["name"] for c in compact if c.get("result") == "FAIL"]
    pending = [c["name"] for c in compact if c.get("result") == "PENDING"]
    return {"checks": compact, "failed": failed, "pending": pending,
            "all_passed": bool(compact) and not failed and not pending}


def summarize_alpha(alpha: Dict[str, Any]) -> Dict[str, Any]:
    """Compact view of an alpha: identity, settings, code, IS metrics and checks."""
    if not isinstance(alpha, dict):
        return alpha
    settings = alpha.get("settings") or {}
    is_block = alpha.get("is") or {}
    out: Dict[str, Any] = {
        "id": alpha.get("id"),
        "type": alpha.get("type"),
        "stage": alpha.get("stage"),
        "status": alpha.get("status"),
        "name": alpha.get("name"),
        "dateCreated": alpha.get("dateCreated"),
        "dateSubmitted": alpha.get("dateSubmitted"),
        "settings": {k: settings.get(k) for k in _SETTING_KEYS if settings.get(k) is not None},
    }
    for part in ("regular", "combo", "selection"):
        code = _code(alpha.get(part))
        if code:
            out[part] = code
    out["is"] = {k: is_block.get(k) for k in _IS_METRICS if is_block.get(k) is not None}
    if is_block.get("checks") is not None:
        out["is"].update(summarize_checks(is_block.get("checks")))
    tags = alpha.get("tags")
    if tags:
        out["tags"] = [t.get("name", t) if isinstance(t, dict) else t for t in tags]
    classifications = alpha.get("classifications")
    if classifications:
        out["classifications"] = [c.get("id") or c.get("name") for c in classifications if isinstance(c, dict)]
    pyramids = alpha.get("pyramids")
    if pyramids:
        out["pyramids"] = [pyramid_key(p) for p in pyramids if isinstance(p, dict)]
    for key in ("grade", "favorite", "hidden", "color", "category", "osmosisPoints"):
        if alpha.get(key) not in (None, False, ""):
            out[key] = alpha.get(key)
    return {k: v for k, v in out.items() if v not in (None, {}, [])}


def pyramid_key(p: Dict[str, Any]) -> str:
    """Stable key for a pyramid cell, shared by alpha pyramids and multipliers."""
    if p.get("name"):
        return str(p["name"])
    cat = p.get("category") or {}
    cat_name = cat.get("name") or cat.get("id") if isinstance(cat, dict) else cat
    return f"{p.get('region')}/D{p.get('delay')}/{cat_name}"


def summarize_records(data: Any, max_rows: int) -> Any:
    """Column/row view of a {schema, records} payload, keeping the last max_rows rows."""
    if not isinstance(data, dict) or "records" not in data:
        return data
    schema = data.get("schema") or {}
    columns = [p.get("name") for p in schema.get("properties") or [] if isinstance(p, dict)]
    records = data.get("records") or []
    out: Dict[str, Any] = {"name": schema.get("name"), "columns": columns, "total_rows": len(records)}
    if max_rows and len(records) > max_rows:
        out["rows"] = records[-max_rows:]
        out["truncated"] = f"showing the last {max_rows} of {len(records)} rows"
    else:
        out["rows"] = records
    for key in ("min", "max"):
        if key in data:
            out[key] = data[key]
    return out


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


@dataclass
class PollResult:
    done: bool
    data: Any = None
    retry_after: float = 0.0
    status: int = 0


_HARD_SIM_DEFAULTS: Dict[str, Any] = {
    "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 1, "decay": 0,
    "neutralization": "NONE", "truncation": 0.0, "pasteurization": "ON", "unitHandling": "VERIFY",
    "nanHandling": "OFF", "language": "FASTEXPR", "visualization": False, "testPeriod": "P0Y0M",
    "maxTrade": "OFF", "maxPosition": "OFF", "selectionHandling": "POSITIVE", "selectionLimit": 1000,
    "componentActivation": "IS",
}
_ON_OFF = ("ON", "OFF")
_SETTING_ENUMS: Dict[str, Sequence[str]] = {
    "instrumentType": ("EQUITY",), "pasteurization": _ON_OFF, "nanHandling": _ON_OFF,
    "unitHandling": ("VERIFY",), "maxTrade": _ON_OFF, "maxPosition": _ON_OFF,
    "language": ("FASTEXPR", "PYTHON"), "selectionHandling": ("POSITIVE", "NON_ZERO", "NON_NAN"),
    "componentActivation": ("IS", "OS"),
}
_SUPER_ONLY = ("selectionHandling", "selectionLimit", "componentActivation")
_FASTEXPR_ONLY = ("unitHandling", "nanHandling", "testPeriod")

ACTIVITY_KINDS = ("list", "diversity", "pyramid-alphas", "pyramid-multipliers", "base-payment",
                  "other-payment", "referrals", "simulations", "submissions")
BOARD_TYPES = ("leader", "spc", "power-pool", "referral")
CORRELATION_TYPES = ("self", "prod", "power-pool")
RECORDSET_TYPES = ("pnl", "sharpe", "turnover", "daily-pnl", "yearly-stats")


class BrainClient:
    """Async facade over the BRAIN REST API (requests + bounded thread pool)."""

    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        parts = urlsplit(self.base_url)
        self._scheme = parts.scheme
        self._host = parts.hostname or ""
        self.session: Optional[CreddSession] = None
        self._session_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=int(os.environ.get("WQMCP_HTTP_WORKERS", "32")),
            thread_name_prefix="wqmcp-http",
        )
        self.request_timeout = (
            float(os.environ.get("WQMCP_CONNECT_TIMEOUT", "10")),
            float(os.environ.get("WQMCP_READ_TIMEOUT", "60")),
        )
        # Account-wide cooldown after a 429 so concurrent callers back off together.
        self._cooldown_until = 0.0
        self._user_id: Optional[str] = None
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._pending_submits: Dict[str, float] = {}
        # Every simulation this process created (even if the MCP call was cancelled).
        self.recent_simulations: deque = deque(maxlen=50)

    # ---------------------------------------------------------------- session

    def _build_session(self) -> CreddSession:
        session = CreddSession(self.base_url)
        adapter = HTTPAdapter(pool_connections=4,
                              pool_maxsize=int(os.environ.get("WQMCP_POOL_MAXSIZE", "32")))
        session.mount(f"{self._scheme}://", adapter)
        session.headers.update({"User-Agent": "wqmcp/2.0 (+requests)"})
        return session

    async def _ensure_session(self) -> CreddSession:
        if self.session is not None:
            return self.session
        async with self._session_lock:
            if self.session is None:
                loop = asyncio.get_running_loop()
                self.session = await loop.run_in_executor(self._executor, self._build_session)
                logger.info("session initialised from credd")
            return self.session

    async def cookie_list(self) -> List[Dict[str, Any]]:
        session = await self._ensure_session()
        return session.cookie_list()

    def _cached(self, key: str, ttl: float) -> Any:
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < ttl:
            return hit[1]
        return None

    def _store(self, key: str, value: Any) -> Any:
        self._cache[key] = (time.monotonic(), value)
        return value

    # ------------------------------------------------------------------ HTTP

    async def request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None,
                      json: Any = None, on_response=None, accept: Optional[str] = None) -> requests.Response:
        """Send one request. ``path`` must be built from literals and seg() values."""
        if not path.startswith("/"):
            raise InvalidArgument(f"bad path {path!r}")
        session = await self._ensure_session()
        wait = self._cooldown_until - time.monotonic()
        if wait > 0:
            await asyncio.sleep(min(wait, 30.0))
        headers = {"Accept": accept or accept_header(method, path)}
        url = self.base_url + path
        method_l = method.lower()

        def send() -> requests.Response:
            resp = session.request(method_l, url, params=params, json=json, headers=headers,
                                   timeout=self.request_timeout)
            if on_response is not None:
                on_response(resp)  # runs even if the awaiting task was cancelled
            return resp

        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(self._executor, send)
        if resp.status_code == 429 and not (method.upper() == "POST" and path == "/simulations"):
            ra = retry_after_seconds(resp) or 5.0
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + min(ra, 60.0))
        return resp

    @staticmethod
    def _parse(resp: requests.Response, method: str, path: str) -> Any:
        if resp.status_code >= 400:
            raise BrainAPIError(resp.status_code, method, path, error_message(resp),
                                retry_after_seconds(resp) or None)
        if not (resp.text or "").strip():
            return None
        try:
            return resp.json()
        except ValueError as exc:
            raise BrainAPIError(resp.status_code, method, path, "response is not JSON") from exc

    async def call(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None,
                   json: Any = None) -> Any:
        """Request + raise on HTTP error + parse JSON (None for an empty body)."""
        resp = await self.request(method, path, params=params, json=json)
        return self._parse(resp, method.upper(), path)

    async def poll(self, path: str, wait_seconds: float, *, first: Optional[requests.Response] = None,
                   first_method: str = "GET") -> PollResult:
        """Follow the catalog's Retry-After protocol for GET ``path``.

        In progress = 2xx with ``Retry-After`` > 0. Done = 2xx without it (the
        body is then parsed). 429 and 5xx are retried while the wait budget lasts;
        other 4xx raise immediately (no blind retries).
        """
        budget = clamp(wait_seconds, 0, MAX_WAIT_SECONDS, 0)
        deadline = time.monotonic() + budget
        resp = first
        method = first_method if first is not None else "GET"
        backoff = 2.0
        while True:
            if resp is None:
                resp = await self.request("GET", path)
                method = "GET"
            status = resp.status_code
            ra = retry_after_seconds(resp)
            remaining = deadline - time.monotonic()
            if status == 429 or status >= 500:
                if remaining <= 0:
                    if status == 429:
                        return PollResult(False, None, ra or 5.0, status)
                    raise BrainAPIError(status, method, path, error_message(resp), ra or None)
                sleep_for = ra or backoff
                backoff = min(backoff * 2, 30.0)
            elif status >= 400:
                raise BrainAPIError(status, method, path, error_message(resp), ra or None)
            elif ra > 0:
                if remaining <= 0:
                    data = None
                    try:
                        data = resp.json() if (resp.text or "").strip() else None
                    except ValueError:
                        pass
                    return PollResult(False, data, ra, status)
                sleep_for = ra
            else:
                return PollResult(True, self._parse(resp, method, path), 0.0, status)
            await asyncio.sleep(max(1.0, min(sleep_for, max(remaining, 1.0))))
            resp = None

    # ------------------------------------------------------------------ auth

    async def status(self, refresh: bool = False) -> Dict[str, Any]:
        session = await self._ensure_session()
        if refresh:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, session.refresh_cookies)
        resp = await self.request("GET", "/authentication")
        if resp.status_code in (401, 403):
            return {"authenticated": False, "http_status": resp.status_code,
                    "hint": "credd's cookie was rejected by BRAIN; check credd /status "
                            "(backoff or pending biometric check)."}
        data = self._parse(resp, "GET", "/authentication") or {}
        user_id = (data.get("user") or {}).get("id")
        if user_id:
            self._user_id = user_id
        return {"authenticated": True, "user_id": user_id,
                "token_expiry": (data.get("token") or {}).get("expiry"),
                "permissions": data.get("permissions")}

    async def user_id(self) -> str:
        if self._user_id:
            return self._user_id
        try:
            await self.status()
        except BrainError:
            pass
        return self._user_id or "self"

    # ------------------------------------------------------------ simulation

    async def platform_defaults(self) -> Optional[Dict[str, Any]]:
        cached = self._cached("sim_defaults", 600)
        if cached is not None:
            return cached
        uid = await self.user_id()
        try:
            data = await self.call("GET", f"/users/{seg(uid, 'user id')}/settings/simulation")
        except BrainError as exc:
            logger.warning("could not read platform simulation defaults: %s", exc)
            return None
        return self._store("sim_defaults", data if isinstance(data, dict) else None)

    async def build_settings(self, sim_type: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        settings = dict(_HARD_SIM_DEFAULTS)
        if USE_PLATFORM_DEFAULTS:
            platform = await self.platform_defaults() or {}
            for key, val in platform.items():
                if key in settings and val is not None:
                    settings[key] = val
        for key, val in overrides.items():
            if val is not None:
                settings[key] = val
        if sim_type == "SUPER":
            # Platform defaults may carry "OFF" for these SUPER-only enums.
            for key in ("selectionHandling", "componentActivation"):
                if settings.get(key) not in _SETTING_ENUMS[key]:
                    settings[key] = _HARD_SIM_DEFAULTS[key]
        else:
            for key in _SUPER_ONLY:
                settings.pop(key, None)
        language = str(settings.get("language") or "FASTEXPR").upper()
        settings["language"] = language
        if language == "PYTHON":
            for key in _FASTEXPR_ONLY:
                settings.pop(key, None)
            if settings.get("lookback") is None:
                raise InvalidArgument("lookback is required when language='PYTHON'")
        else:
            settings.pop("lookback", None)
        self._validate_settings(settings)
        return settings

    @staticmethod
    def _validate_settings(s: Dict[str, Any]) -> None:
        for key, allowed in _SETTING_ENUMS.items():
            if key in s and s[key] not in allowed:
                raise InvalidArgument(f"{key} must be one of {list(allowed)}, got {s[key]!r}")
        decay = s.get("decay")
        if isinstance(decay, float) and decay.is_integer():
            s["decay"] = decay = int(decay)
        if not isinstance(decay, int) or not 0 <= decay <= 512:
            raise InvalidArgument(f"decay must be an integer in 0..512, got {decay!r}")
        if not 0 <= float(s.get("truncation", 0)) <= 1:
            raise InvalidArgument("truncation must be in 0..1")
        if "lookback" in s and not 0 <= int(s["lookback"]) <= 1024:
            raise InvalidArgument("lookback must be in 0..1024")
        if "selectionLimit" in s and not 10 <= int(s["selectionLimit"]) <= 1000:
            raise InvalidArgument("selectionLimit must be in 10..1000")
        if "testPeriod" in s and not _TEST_PERIOD_RE.match(str(s["testPeriod"])):
            raise InvalidArgument("testPeriod must look like P0Y0M or P1Y0M0D")
        if s.get("delay") not in (0, 1):
            raise InvalidArgument("delay must be 0 or 1")

    def _record_simulation(self, resp: requests.Response, count: int) -> None:
        if resp.status_code == 201 and resp.headers.get("Location"):
            location = resp.headers["Location"]
            self.recent_simulations.appendleft({
                "simulation_id": location.rstrip("/").split("/")[-1],
                "alphas": count, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

    async def create_simulations(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            raise InvalidArgument("nothing to simulate")
        if len(items) > 10:
            raise InvalidArgument("at most 10 alphas per multi-simulation")
        body: Any = items[0] if len(items) == 1 else items
        resp = await self.request("POST", "/simulations", json=body,
                                  on_response=lambda r: self._record_simulation(r, len(items)))
        if resp.status_code == 429:
            ra = retry_after_seconds(resp) or 30.0
            return {"status": "RATE_LIMITED", "retry_after_seconds": ra,
                    "note": "The account's concurrent simulation slots are full. Retry after "
                            "retry_after_seconds, or cancel_simulation a running one you no longer need."}
        if resp.status_code != 201:
            raise BrainAPIError(resp.status_code, "POST", "/simulations", error_message(resp),
                                retry_after_seconds(resp) or None)
        location = resp.headers.get("Location", "")
        if not location:
            raise BrainAPIError(201, "POST", "/simulations", "no Location header in the response")
        sim_id = location.rstrip("/").split("/")[-1]
        return {"status": "SUBMITTED", "simulation_id": sim_id, "alphas": len(items),
                "multi": len(items) > 1}

    def simulation_id_from(self, ref: str) -> str:
        """Accept a bare simulation id or a progress URL on the BRAIN API host only."""
        ref = str(ref or "").strip()
        if "://" in ref:
            parts = urlsplit(ref)
            m = re.fullmatch(r"/simulations/([^/]+)/?", parts.path or "")
            if parts.scheme != self._scheme or parts.hostname != self._host or not m \
                    or parts.query or parts.fragment:
                raise InvalidArgument(f"not a BRAIN simulation URL: {ref!r}")
            ref = m.group(1)
        seg(ref, "simulation id")
        return ref

    async def _alpha_summary(self, alpha_id: str) -> Dict[str, Any]:
        resp = await self.request("GET", f"/alphas/{seg(alpha_id, 'alpha id')}")
        if resp.status_code >= 400:
            return {"id": alpha_id, "error": f"HTTP {resp.status_code}: {error_message(resp)}"}
        try:
            return summarize_alpha(resp.json())
        except ValueError:
            return {"id": alpha_id, "error": "alpha response is not JSON"}

    async def _simulation_once(self, sim_id: str, include_alpha: bool) -> Dict[str, Any]:
        path = f"/simulations/{seg(sim_id, 'simulation id')}"
        resp = await self.request("GET", path)
        status = resp.status_code
        if status == 429 or status >= 500:
            return {"simulation_id": sim_id, "status": "UNKNOWN", "http_status": status,
                    "retry_after_seconds": retry_after_seconds(resp) or 5.0,
                    "note": "transient BRAIN error; check again"}
        body = self._parse(resp, "GET", path) or {}
        ra = retry_after_seconds(resp)
        children = body.get("children") or []
        if children:
            states = await asyncio.gather(*[self._child_state(str(c)) for c in children])
            unfinished = [s for s in states if s["status"] in ("RUNNING", "UNKNOWN")]
            if include_alpha and not unfinished:
                async def with_alpha(s: Dict[str, Any]) -> Dict[str, Any]:
                    if s.get("alpha_id"):
                        return {**s, "alpha": await self._alpha_summary(s["alpha_id"])}
                    return s
                states = await asyncio.gather(*[with_alpha(s) for s in states])
            out = {"simulation_id": sim_id, "multi": True,
                   "status": "RUNNING" if unfinished else (body.get("status") or "COMPLETE"),
                   "completed_children": len(states) - len(unfinished), "total_children": len(states),
                   "children": list(states)}
            if unfinished:
                out["retry_after_seconds"] = max([ra] + [s.get("retry_after_seconds", 0) for s in unfinished]) or 5.0
            return out
        if ra > 0:
            return {"simulation_id": sim_id, "status": "RUNNING", "progress": body.get("progress"),
                    "retry_after_seconds": ra}
        out = {"simulation_id": sim_id, "status": body.get("status") or ("COMPLETE" if body.get("alpha") else "ERROR"),
               "alpha_id": body.get("alpha")}
        for key in ("message", "detail", "details"):
            if body.get(key):
                out[key] = body[key]
        if body.get("regular"):
            out["regular"] = _code(body["regular"])
        if include_alpha and body.get("alpha"):
            out["alpha"] = await self._alpha_summary(body["alpha"])
        return {k: v for k, v in out.items() if v is not None}

    async def _child_state(self, child: str) -> Dict[str, Any]:
        child_id = child.rstrip("/").split("/")[-1]
        try:
            return await self._simulation_once(child_id, False)
        except BrainAPIError as exc:
            return {"simulation_id": child_id, "status": "ERROR", "http_status": exc.status,
                    "message": exc.message}

    async def simulations(self, refs: Sequence[str], wait_seconds: float = 0,
                          include_alpha: bool = True) -> List[Dict[str, Any]]:
        ids = [self.simulation_id_from(r) for r in refs]
        budget = clamp(wait_seconds, 0, MAX_WAIT_SECONDS, 0)
        deadline = time.monotonic() + budget
        results: Dict[str, Dict[str, Any]] = {}
        pending = list(dict.fromkeys(ids))
        while True:
            states = await asyncio.gather(*[self._simulation_once(i, include_alpha) for i in pending])
            for i, s in zip(pending, states):
                results[i] = s
            pending = [i for i in pending if results[i]["status"] in ("RUNNING", "UNKNOWN")]
            remaining = deadline - time.monotonic()
            if not pending or remaining <= 0:
                break
            nap = min(r.get("retry_after_seconds", 5.0) for r in (results[i] for i in pending))
            await asyncio.sleep(max(1.0, min(nap, remaining)))
        return [results[i] for i in dict.fromkeys(ids)]

    async def cancel_simulation(self, ref: str) -> Dict[str, Any]:
        sim_id = self.simulation_id_from(ref)
        await self.call("DELETE", f"/simulations/{seg(sim_id, 'simulation id')}")
        return {"simulation_id": sim_id, "cancelled": True}

    async def setting_options(self) -> Dict[str, Any]:
        cached = self._cached("sim_options", 3600)
        if cached is None:
            data = await self.call("OPTIONS", "/simulations") or {}
            parsed = parse_setting_options(data)
            if not parsed.get("instrument_options") and SEND_ACCEPT_VERSIONS:
                # The v3.0 sample in the catalog is abridged; v1 parsed the unversioned shape.
                resp = await self.request("OPTIONS", "/simulations", accept="application/json")
                alt = parse_setting_options(self._parse(resp, "OPTIONS", "/simulations") or {})
                if alt.get("instrument_options"):
                    parsed = alt
            cached = self._store("sim_options", parsed)
        return {**cached, "platform_defaults": await self.platform_defaults()}

    async def super_selection(self, selection: str, *, instrument_type: str, region: Optional[str],
                              delay: Optional[int], selection_limit: Optional[int],
                              selection_handling: Optional[str], limit: int) -> Dict[str, Any]:
        params: Dict[str, Any] = {"selection": selection, "settings.instrumentType": instrument_type,
                                  "limit": int(clamp(limit, 1, 100, 10))}
        if region:
            params["settings.region"] = region
        if delay is not None:
            params["settings.delay"] = delay
        if selection_limit is not None:
            params["selectionLimit"] = int(clamp(selection_limit, 10, 1000, 1000))
        if selection_handling:
            params["selectionHandling"] = selection_handling
        data = await self.call("GET", "/simulations/super-selection", params=params) or {}
        return {**page_meta(data, params["limit"], 0),
                "results": [summarize_alpha(a) for a in data.get("results") or []]}

    # ------------------------------------------------------------------ alphas

    async def list_alphas(self, *, stage: Optional[str], status: Optional[str], alpha_type: Optional[str],
                          limit: int, offset: int, order: Optional[str], created_after: Optional[str],
                          created_before: Optional[str], submitted_after: Optional[str],
                          submitted_before: Optional[str], hidden: Optional[bool],
                          full: bool) -> Dict[str, Any]:
        limit = int(clamp(limit, 1, 100, 20))
        offset = int(clamp(offset, 0, 10**9, 0))
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        for key, val in (("stage", stage), ("status", status), ("type", alpha_type), ("order", order)):
            if val:
                params[key] = val
        # Date filters and `hidden` are not in the catalog (hidden params); kept from v1.
        for key, val in (("dateCreated>", created_after), ("dateCreated<", created_before),
                         ("dateSubmitted>", submitted_after), ("dateSubmitted<", submitted_before)):
            if check_date(val, key, allow_time=True):
                params[key] = val
        if hidden is not None:
            params["hidden"] = str(hidden).lower()
        data = await self.call("GET", "/users/self/alphas", params=params) or {}
        results = data.get("results") or []
        return {**page_meta(data, limit, offset),
                "results": results if full else [summarize_alpha(a) for a in results]}

    async def alpha_summary_counts(self) -> Any:
        return await self.call("GET", "/users/self/alphas/summary")

    async def get_alpha(self, alpha_id: str, full: bool = False) -> Dict[str, Any]:
        data = await self.call("GET", f"/alphas/{seg(alpha_id, 'alpha id')}") or {}
        return data if full else summarize_alpha(data)

    async def recordset(self, alpha_id: str, name: Optional[str], wait_seconds: float,
                        max_rows: int) -> Dict[str, Any]:
        base = f"/alphas/{seg(alpha_id, 'alpha id')}/recordsets"
        if not name:
            res = await self.poll(base, wait_seconds)
            if not res.done:
                return {"status": "PENDING", "retry_after_seconds": res.retry_after}
            return {"alpha_id": alpha_id, "recordsets": [r.get("name") for r in (res.data or {}).get("results") or []]}
        res = await self.poll(f"{base}/{seg(name, 'recordset')}", wait_seconds)
        if not res.done:
            return {"alpha_id": alpha_id, "recordset": name, "status": "PENDING",
                    "retry_after_seconds": res.retry_after,
                    "note": "BRAIN is still computing this recordset; call again later."}
        if res.data is None:
            return {"alpha_id": alpha_id, "recordset": name, "status": "EMPTY"}
        return {"alpha_id": alpha_id, "recordset": name, "status": "DONE",
                **summarize_records(res.data, int(clamp(max_rows, 0, 10000, 300)))}

    async def correlation(self, alpha_id: str, kind: str, wait_seconds: float, top: int = 5) -> Dict[str, Any]:
        if kind not in CORRELATION_TYPES:
            raise InvalidArgument(f"correlation type must be one of {list(CORRELATION_TYPES)}")
        res = await self.poll(f"/alphas/{seg(alpha_id, 'alpha id')}/correlations/{kind}", wait_seconds)
        if not res.done:
            return {"type": kind, "status": "PENDING", "retry_after_seconds": res.retry_after}
        data = res.data or {}
        out: Dict[str, Any] = {"type": kind, "status": "DONE", "max": data.get("max"), "min": data.get("min")}
        schema = data.get("schema") or {}
        columns = [p.get("name") for p in schema.get("properties") or [] if isinstance(p, dict)]
        records = data.get("records") or []
        if "correlation" in columns:
            idx = columns.index("correlation")
            ranked = sorted((r for r in records if isinstance(r, list) and len(r) > idx
                             and isinstance(r[idx], (int, float))), key=lambda r: r[idx], reverse=True)
            out["top"] = [dict(zip(columns, r)) for r in ranked[:top]]
        out["rows"] = len(records)
        return out

    async def check_alpha(self, alpha_id: str, wait_seconds: float,
                          correlations: Sequence[str] = ()) -> Dict[str, Any]:
        for kind in correlations:
            if kind not in CORRELATION_TYPES:
                raise InvalidArgument(f"correlation type must be one of {list(CORRELATION_TYPES)}")
        check_task = self.poll(f"/alphas/{seg(alpha_id, 'alpha id')}/check", wait_seconds)
        corr_tasks = [self.correlation(alpha_id, k, wait_seconds) for k in dict.fromkeys(correlations)]
        check, *corrs = await asyncio.gather(check_task, *corr_tasks)
        out: Dict[str, Any] = {"alpha_id": alpha_id}
        if not check.done:
            out.update(status="PENDING", retry_after_seconds=check.retry_after,
                       note="BRAIN is still running the checks; call check_alpha again later.")
        else:
            is_block = (check.data or {}).get("is") or {}
            summary = summarize_checks(is_block.get("checks"))
            out.update(status="DONE", **summary)
            self_corr = is_block.get("selfCorrelation")
            if isinstance(self_corr, dict) and self_corr.get("max") is not None:
                out["self_correlation_max"] = self_corr.get("max")
        if corrs:
            out["correlations"] = corrs
        return out

    async def submit(self, alpha_id: str, confirm: bool, wait_seconds: float) -> Dict[str, Any]:
        path = f"/alphas/{seg(alpha_id, 'alpha id')}/submit"
        if not confirm:
            result = await self.check_alpha(alpha_id, wait_seconds)
            result["dry_run"] = True
            result["note"] = ("Not submitted. These are BRAIN's pre-submission checks; call "
                              "submit_alpha(alpha_id, confirm=True) to actually submit.")
            return result
        started = self._pending_submits.get(alpha_id)
        first = None
        if not started or time.monotonic() - started > 3600:
            first = await self.request("POST", path)
            if first.status_code == 429:
                return {"alpha_id": alpha_id, "status": "RATE_LIMITED",
                        "retry_after_seconds": retry_after_seconds(first) or 30.0}
            if first.status_code >= 400:
                raise BrainAPIError(first.status_code, "POST", path, error_message(first),
                                    retry_after_seconds(first) or None)
            self._pending_submits[alpha_id] = time.monotonic()
        res = await self.poll(path, wait_seconds, first=first, first_method="POST")
        if not res.done:
            return {"alpha_id": alpha_id, "status": "PENDING", "retry_after_seconds": res.retry_after,
                    "note": "Submission is being processed. Call submit_alpha(alpha_id, confirm=True) "
                            "again to keep polling; it will not submit twice."}
        self._pending_submits.pop(alpha_id, None)
        summary = summarize_checks(((res.data or {}).get("is") or {}).get("checks"))
        if summary["failed"]:
            state = "REJECTED"
        elif summary["pending"]:
            state = "SUBMITTED_WITH_PENDING_CHECKS"
        else:
            state = "SUBMITTED"  # includes an empty 2xx body (accepted without a report)
        return {"alpha_id": alpha_id, "status": state, **summary}

    async def update_alpha(self, alpha_ids: Sequence[str], fields: Dict[str, Any],
                           bulk_fields: Dict[str, Any]) -> Dict[str, Any]:
        ids = [str(a).strip() for a in alpha_ids]
        for a in ids:
            seg(a, "alpha id")
        if not ids:
            raise InvalidArgument("alpha_ids is empty")
        if not fields and not bulk_fields:
            raise InvalidArgument("nothing to update")
        if fields and len(ids) != 1:
            raise InvalidArgument("name/category/tags/descriptions/osmosis_points can only be set on one alpha at a time")
        out: Dict[str, Any] = {"alpha_ids": ids}
        if bulk_fields:
            body = [{"id": a, **bulk_fields} for a in ids]
            await self.call("PATCH", "/alphas", json=body)
            out["updated_bulk_fields"] = sorted(bulk_fields)
        if fields:
            data = await self.call("PATCH", f"/alphas/{seg(ids[0], 'alpha id')}", json=fields)
            out["alpha"] = summarize_alpha(data or {})
        return out

    async def alpha_performance(self, alpha_id: str, competition_id: Optional[str]) -> Any:
        if competition_id:
            path = (f"/competitions/{seg(competition_id, 'competition id')}/alphas/"
                    f"{seg(alpha_id, 'alpha id')}/before-and-after-performance")
        else:
            path = f"/users/self/alphas/{seg(alpha_id, 'alpha id')}/before-and-after-performance"
        data = await self.call("GET", path)
        if data is None:
            return {"alpha_id": alpha_id, "status": "EMPTY",
                    "note": "BRAIN returned no body (the catalog marks the competition variant as empty)."}
        if isinstance(data, dict):
            for key in ("yearlyStats", "pnl"):
                block = data.get(key)
                if isinstance(block, dict):
                    data[key] = {side: summarize_records(v, 50) for side, v in block.items()}
        return data

    # -------------------------------------------------------------------- data

    async def datasets(self, params: Dict[str, Any], limit: int, offset: int) -> Dict[str, Any]:
        limit = int(clamp(limit, 1, 50, 20))
        offset = int(clamp(offset, 0, 10**9, 0))
        q = {k: v for k, v in params.items() if v not in (None, "")}
        q.update(limit=limit, offset=offset)
        data = await self.call("GET", "/data-sets", params=q) or {}
        results = [_compact_dataset(d) for d in data.get("results") or []]
        out = {**page_meta(data, limit, offset), "results": results}
        if not results:
            out["hint"] = "No datasets; check region/delay/universe with get_platform_setting_options."
        return out

    async def datafields(self, params: Dict[str, Any], limit: int, offset: int) -> Dict[str, Any]:
        limit = int(clamp(limit, 1, 100, 50))
        offset = int(clamp(offset, 0, 10**9, 0))
        q = {k: v for k, v in params.items() if v not in (None, "")}
        q.update(limit=limit, offset=offset)
        data = await self.call("GET", "/data-fields", params=q) or {}
        results = [_compact_field(f) for f in data.get("results") or []]
        out = {**page_meta(data, limit, offset), "results": results}
        if not results:
            out["hint"] = "No fields; check region/delay/universe with get_platform_setting_options."
        return out

    async def datafield(self, field_id: str) -> Any:
        return await self.call("GET", f"/data-fields/{seg(field_id, 'field id')}")

    async def search_data(self, query: str, limit: int) -> Dict[str, Any]:
        data = await self.call("GET", "/data-sets/search", params={"search": query}) or {}
        limit = int(clamp(limit, 1, 100, 20))
        datasets = data.get("datasets") or []
        fields = data.get("fields") or []
        return {"datasets": [_compact_dataset(d) for d in datasets[:limit]], "datasets_total": len(datasets),
                "fields": [_compact_field(f) for f in fields[:limit]], "fields_total": len(fields)}

    async def operators(self) -> List[Dict[str, Any]]:
        cached = self._cached("operators", 3600)
        if cached is None:
            data = await self.call("GET", "/operators") or []
            if isinstance(data, dict):
                data = data.get("results") or data.get("operators") or []
            cached = self._store("operators", data)
        return cached

    # -------------------------------------------------------------- activities

    async def activity(self, kind: str, *, grouping: Optional[str], start_date: Optional[str],
                       end_date: Optional[str], since: Optional[str], max_rows: int) -> Any:
        if kind not in ACTIVITY_KINDS:
            raise InvalidArgument(f"kind must be one of {list(ACTIVITY_KINDS)}")
        params: Dict[str, Any] = {}
        if kind == "list":
            path = "/users/self/activities"
        else:
            path = f"/users/self/activities/{kind}"
        if kind == "diversity" and grouping:
            if not re.fullmatch(r"[A-Za-z]+(,[A-Za-z]+)*", grouping):
                raise InvalidArgument("grouping must be comma-separated field names, e.g. 'region,delay'")
            params["grouping"] = grouping
        if kind == "pyramid-alphas":
            if check_date(start_date, "start_date"):
                params["startDate"] = start_date
            if check_date(end_date, "end_date"):
                params["endDate"] = end_date
        if kind in ("simulations", "submissions") and check_date(since, "since"):
            params["date>"] = since
        data = await self.call("GET", path, params=params or None)
        if isinstance(data, dict) and isinstance(data.get("records"), dict):
            data["records"] = summarize_records(data["records"], int(clamp(max_rows, 0, 5000, 60)))
        return data

    async def consultant(self) -> Any:
        uid = await self.user_id()
        return await self.call("GET", f"/users/{seg(uid, 'user id')}/consultant")

    async def diversity_score(self, start: str, end: str, max_alphas: int = 2000) -> Dict[str, Any]:
        """Client-side diversity score of OS REGULAR alphas submitted in [start, end].

        Uses the pyramids/classifications already present in listAlphas results
        (no per-alpha requests). Approximation; see value_factor in the result for
        BRAIN's official number.
        """
        check_date(start, "start_date", allow_time=True)
        check_date(end, "end_date", allow_time=True)
        alphas: List[Dict[str, Any]] = []
        offset, page, complete = 0, 100, True
        while True:
            data = await self.call("GET", "/users/self/alphas", params={
                "stage": "OS", "type": "REGULAR", "limit": page, "offset": offset,
                "dateSubmitted>": start, "dateSubmitted<": end}) or {}
            batch = data.get("results") or []
            alphas.extend(batch)
            offset += len(batch)
            count = data.get("count")
            if not batch or (count is not None and offset >= count):
                break
            if len(alphas) >= max_alphas:
                complete = False
                break
        regular = [a for a in alphas if a.get("type", "REGULAR") == "REGULAR"]
        per_pyramid: Dict[str, int] = {}
        atoms = 0
        for a in regular:
            if _is_atom(a):
                atoms += 1
            for p in a.get("pyramids") or []:
                if isinstance(p, dict):
                    key = pyramid_key(p)
                    per_pyramid[key] = per_pyramid.get(key, 0) + 1
        n, p_count = len(regular), len(per_pyramid)
        p_max: Optional[int] = None
        try:
            pm = await self.activity("pyramid-multipliers", grouping=None, start_date=None,
                                     end_date=None, since=None, max_rows=0) or {}
            p_max = len({pyramid_key(x) for x in pm.get("pyramids") or [] if isinstance(x, dict)}) or None
        except BrainError as exc:
            logger.warning("pyramid multipliers unavailable: %s", exc)
        s_a = atoms / n if n else 0.0
        s_p = p_count / p_max if p_max else None
        s_h = 0.0
        if p_count > 1:
            total = sum(per_pyramid.values())
            h = -sum((c / total) * math.log2(c / total) for c in per_pyramid.values() if c)
            s_h = h / math.log2(p_count)
        out: Dict[str, Any] = {
            "window": {"start": start, "end": end}, "N": n, "A": atoms, "P": p_count, "P_max": p_max,
            "S_A": round(s_a, 4), "S_P": None if s_p is None else round(s_p, 4), "S_H": round(s_h, 4),
            "diversity_score": None if s_p is None else round(s_a * s_p * s_h, 4),
            "per_pyramid_counts": per_pyramid, "complete": complete,
        }
        if s_p is None:
            out["note"] = "P_max unknown (pyramid-multipliers failed); diversity_score not computed."
        if not complete:
            out["note"] = f"Only the first {len(alphas)} alphas in the window were counted."
        return out

    async def leaderboard(self, board: str, *, limit: int, offset: int, order: Optional[str],
                          aggregate: Optional[str], user: Optional[str], mine: bool) -> Dict[str, Any]:
        if board not in BOARD_TYPES:
            raise InvalidArgument(f"board must be one of {list(BOARD_TYPES)}")
        limit = int(clamp(limit, 1, 100, 10))
        offset = int(clamp(offset, 0, 10**9, 0))
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if order:
            params["order"] = order
        if aggregate:
            params["aggregate"] = aggregate
        if user or mine:
            params["user"] = user or await self.user_id()
        data = await self.call("GET", f"/consultant/boards/{board}", params=params) or {}
        return {**page_meta(data, limit, offset), "results": data.get("results") or []}

    async def consultant_summary(self) -> Any:
        uid = await self.user_id()
        return await self.call("GET", f"/users/{seg(uid, 'user id')}/consultant/summary")

    async def message_summary(self) -> Any:
        return await self.call("GET", "/users/self/messages/summary")

    # ------------------------------------------------------------ competitions

    async def competitions(self, *, competition_id: Optional[str], scope: str, limit: int, offset: int,
                           include_agreement: bool) -> Any:
        if competition_id:
            cid = seg(competition_id, "competition id")
            detail = await self.call("GET", f"/competitions/{cid}")
            out: Dict[str, Any] = {"competition": detail}
            if include_agreement:
                # Not in the catalog (hidden or removed); best effort.
                resp = await self.request("GET", f"/competitions/{cid}/agreement")
                if resp.status_code == 404:
                    out["agreement"] = None
                    out["agreement_note"] = "no agreement endpoint/record for this competition (404)"
                else:
                    out["agreement"] = self._parse(resp, "GET", f"/competitions/{cid}/agreement")
            return out
        limit = int(clamp(limit, 1, 100, 20))
        offset = int(clamp(offset, 0, 10**9, 0))
        if scope == "mine":
            data = await self.call("GET", "/users/self/competitions") or {}
        else:
            params: Dict[str, Any] = {"limit": limit, "offset": offset}
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if scope == "active":
                params["endDate!<"] = now
            elif scope == "ended":
                params["endDate<"] = now
            data = await self.call("GET", "/competitions", params=params) or {}
        results = [{k: c.get(k) for k in ("id", "name", "status", "startDate", "endDate", "signUpStartDate",
                                          "signUpEndDate", "teamBased", "scoring", "submissions")}
                   for c in data.get("results") or []]
        return {**page_meta(data, limit, offset), "results": results}

    async def events(self, *, limit: int, offset: int, order: Optional[str], event_type: Optional[str],
                     language: Optional[str], upcoming_only: bool) -> Dict[str, Any]:
        limit = int(clamp(limit, 1, 100, 10))
        offset = int(clamp(offset, 0, 10**9, 0))
        params: Dict[str, Any] = {"limit": limit, "offset": offset,
                                  "order": order or ("start" if upcoming_only else "-start")}
        if event_type:
            params["type"] = event_type
        if language:
            params["language"] = language
        if upcoming_only:
            params["start>="] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data = await self.call("GET", "/events", params=params) or {}
        results = [{k: trim(e.get(k), 500) for k in ("id", "title", "type", "start", "end", "timezone",
                                                     "language", "description", "register", "recording",
                                                     "city", "country") if e.get(k) is not None}
                   for e in data.get("results") or []]
        return {**page_meta(data, limit, offset), "results": results}

    async def messages(self, *, limit: int, offset: int, unread_only: bool, message_type: Optional[str],
                       order: Optional[str]) -> Dict[str, Any]:
        limit = int(clamp(limit, 1, 100, 10))
        offset = int(clamp(offset, 0, 10**9, 0))
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if unread_only:
            params["read"] = "false"
        if message_type:
            params["type"] = message_type
        if order:
            params["order"] = order
        data = await self.call("GET", "/users/self/messages", params=params) or {}
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _sanitize_messages, data.get("results") or [])
        return {**page_meta(data, limit, offset), "results": results}

    # ---------------------------------------------------------- documentation

    async def documentation(self, page_id: Optional[str], limit: int) -> Any:
        if page_id:
            data = await self.call("GET", f"/tutorial-pages/{seg(page_id, 'page id')}") or {}
            return _compact_tutorial_page(data)
        limit = int(clamp(limit, 1, 200, 50))
        data = await self.call("GET", "/tutorials", params={"limit": limit}) or {}
        results = [{"id": t.get("id"), "title": t.get("title"), "category": t.get("category"),
                    "pages": [{"id": p.get("id"), "title": p.get("title")} for p in t.get("pages") or []]}
                   for t in data.get("results") or []]
        return {**page_meta(data, limit, 0), "results": results}


# --------------------------------------------------------------------------- #
# Pure helpers (module level so tests can use them)
# --------------------------------------------------------------------------- #


def _is_atom(alpha: Dict[str, Any]) -> bool:
    for c in alpha.get("classifications") or []:
        cid = str((c.get("id") or c.get("name") or "") if isinstance(c, dict) else c)
        if "SINGLE_DATA_SET" in cid or "ATOM" in cid.upper():
            return True
    for t in alpha.get("tags") or []:
        name = t.get("name") if isinstance(t, dict) else t
        if isinstance(name, str) and name.strip().lower() == "atom":
            return True
    return False


def _compact_dataset(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: d.get(k) for k in ("id", "name", "region", "delay", "universe", "coverage", "dateCoverage",
                                 "valueScore", "userCount", "alphaCount", "fieldCount", "pyramidMultiplier",
                                 "dateUpdated")}
    out["description"] = trim(d.get("description"), 300)
    for key in ("category", "subcategory"):
        val = d.get(key)
        out[key] = val.get("id") if isinstance(val, dict) else val
    if d.get("themes"):
        out["themes"] = d["themes"]
    return {k: v for k, v in out.items() if v is not None}


def _compact_field(f: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: f.get(k) for k in ("id", "type", "region", "delay", "universe", "coverage", "dateCoverage",
                                 "userCount", "alphaCount", "pyramidMultiplier")}
    out["description"] = trim(f.get("description"), 300)
    ds = f.get("dataset")
    out["dataset"] = ds.get("id") if isinstance(ds, dict) else ds
    for key in ("category", "subcategory"):
        val = f.get(key)
        out[key] = val.get("id") if isinstance(val, dict) else val
    if f.get("themes"):
        out["themes"] = f["themes"]
    return {k: v for k, v in out.items() if v is not None}


def parse_setting_options(data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten OPTIONS /simulations into valid instrument/region/delay/universe combos.

    Tolerant of shape differences: anything it cannot interpret is skipped and
    the raw choice lists are still returned.
    """
    post = ((data or {}).get("actions") or {}).get("POST") or {}
    settings = post.get("settings") or {}
    children = settings.get("children") if isinstance(settings, dict) else None
    if not isinstance(children, dict):
        return {"instrument_options": [], "note": "OPTIONS /simulations returned no settings.children",
                "raw_keys": sorted(post)}

    by_label: Dict[str, Any] = {}
    enums: Dict[str, Any] = {}
    for key, spec in children.items():
        if not isinstance(spec, dict):
            continue
        label = spec.get("label") or key
        by_label[label] = spec.get("choices")
        choices = spec.get("choices")
        if isinstance(choices, list):
            enums[key] = [c.get("value") for c in choices if isinstance(c, dict)]
        for bound in ("minValue", "maxValue", "min_value", "max_value"):
            if bound in spec:
                enums.setdefault(f"{key}_range", {})[bound] = spec[bound]

    def nested(label: str) -> Dict[str, Any]:
        val = by_label.get(label)
        return val.get("instrumentType", {}) if isinstance(val, dict) else {}

    rows: List[Dict[str, Any]] = []
    instrument_types = by_label.get("Instrument type") or []
    regions, delays, universes, neutralizations = (nested("Region"), nested("Delay"),
                                                   nested("Universe"), nested("Neutralization"))
    for it in instrument_types if isinstance(instrument_types, list) else []:
        itv = it.get("value") if isinstance(it, dict) else None
        for region in regions.get(itv) or []:
            rv = region.get("value")
            for delay in ((delays.get(itv) or {}).get("region") or {}).get(rv) or []:
                rows.append({
                    "instrumentType": itv, "region": rv, "delay": delay.get("value"),
                    "universe": [u.get("value") for u in
                                 ((universes.get(itv) or {}).get("region") or {}).get(rv) or []],
                    "neutralization": [n.get("value") for n in
                                       ((neutralizations.get(itv) or {}).get("region") or {}).get(rv) or []],
                })
    return {"instrument_options": rows, "total_combinations": len(rows), "enums": enums}


_IMG_RE = re.compile(r"<img[^>]+src=\"data:image/[^\"]+\"[^>]*>", re.IGNORECASE)
_B64_RE = re.compile(r"[A-Za-z0-9+/]{500,}={0,2}")


def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip embedded base64 images (they explode LLM context) and trim bodies."""
    out = []
    for msg in messages:
        desc = msg.get("description")
        if isinstance(desc, str):
            desc, n = _IMG_RE.subn("[image removed]", desc)
            desc = _B64_RE.sub("[binary data removed]", desc)
            desc = trim(desc, 4000)
        out.append({k: v for k, v in {"id": msg.get("id"), "type": msg.get("type"), "title": msg.get("title"),
                                      "dateCreated": msg.get("dateCreated"), "read": msg.get("read"),
                                      "description": desc}.items() if v is not None})
    return out


def _compact_tutorial_page(page: Dict[str, Any]) -> Dict[str, Any]:
    blocks = []
    for block in page.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind, value = block.get("type"), block.get("value")
        if kind == "HEADING" and isinstance(value, dict):
            blocks.append({"type": kind, "text": value.get("content"), "level": value.get("level")})
        elif kind == "IMAGE" and isinstance(value, dict):
            blocks.append({"type": kind, "title": value.get("title"), "url": value.get("url")})
        else:
            blocks.append({"type": kind, "value": value})
    return {"id": page.get("id"), "title": page.get("title"), "category": page.get("category"),
            "lastModified": page.get("lastModified"), "content": blocks}

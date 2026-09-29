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
import json
from datetime import datetime
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
from prodmemo_calc import (extract_platform_correlation_stats, normalize_pnl, rolling_window_start,
                           calculate_forward_filled_returns, pearson_correlation)
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


def _poll_delay(retry_after: float, polls: int, remaining: float) -> float:
    """Seconds to wait before poll number `polls` + 1. BRAIN answers "Retry-After: 1"
    for as long as a correlation or check is computing (often minutes), and
    polling every second burns the account's request budget into 429s. So the
    header is a lower bound and the interval grows 1s -> 15s: quick results
    are still picked up quickly."""
    backoff = min(1.6 ** polls, 15.0)
    return max(1.0, min(max(retry_after, backoff), remaining))


# What a PENDING result tells the caller to wait: BRAIN's "1" is a polling hint
# for a client that is already waiting, not a useful delay before a new tool call.
_PENDING_RETRY_SECONDS = 10.0

# Submission checks that are limits of the account, not properties of the alpha.
_ACCOUNT_CHECKS = {"REGULAR_SUBMISSION", "SUPER_SUBMISSION", "SUBMISSION_LIMIT", "DAILY_SUBMISSION"}

# BRAIN's prod correlation limit for a submission.
PROD_THRESHOLD = 0.7


class CorrelationGate:
    """Account-wide queue for BRAIN's correlation / check endpoints.

    Asking for the prod or self correlation of several alphas at once gets the
    whole account rate limited. BRAIN does not always say so with a 429: a
    throttled account just keeps answering "empty body, Retry-After: 1" for
    every alpha, for as long as the requests keep coming. Every client of this
    server shares one gate, which keeps the pressure down:

    - a queue: at most `max_alphas` alphas are computed at a time, the others
      wait first come, first served. `snapshot()` shows who is computing and
      who waits;
    - jobs live in the background (`background`): a request keeps its place and
      keeps being polled after the tool call that made it returned PENDING, for
      up to `queue_seconds`; the result is remembered for `cache_seconds`, so
      asking again later collects it;
    - a request never loses its place: it keeps its slot until BRAIN answers or
      the job's time is over (`rotate` hands a slot on after `max_slot_seconds`
      instead, sending the alpha to the back; off by default);
    - order: jobs stopped by a cooldown, then priority="high" requests, then
      requests for the prod correlation alone (or anything that has waited
      `aging_seconds`), then the rest, first come first served within each;
    - the submission check has a lane of its own (`check_max`). A cooldown
      stops the correlation lane only: /check answers its IS checks without
      computing correlations, so it keeps working while BRAIN throttles;
    - a queued request nobody has asked about for `abandon_seconds` is dropped;
    - the time earlier jobs took gives waiting requests an estimate;
    - throttle detection and cooldown: no result for `stall_seconds` while
      something is computing (or a 429) means rate limiting, and asking a
      throttled BRAIN for more only makes it slower. So everything stops for
      `cooldown_seconds`: no request is sent, the queue is kept. Then one
      alpha probes, polled every `throttled_interval` seconds; a result means
      back to normal, no result within `stall_seconds` means the next
      cooldown, twice as long (up to `cooldown_max`);
    - pacing (`min_interval` between requests), growing poll intervals, and
      overlapping identical requests share one job.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Settings from the environment, and an empty queue / cache."""
        env = os.environ.get
        self.max_alphas = max(1, int(env("WQMCP_CORR_MAX_ALPHAS", "2")))
        self.min_interval = max(0.0, float(env("WQMCP_CORR_MIN_INTERVAL", "0.5")))
        self.hold_seconds = max(0.0, float(env("WQMCP_CORR_HOLD_SECONDS", "90")))
        self.cache_seconds = max(0.0, float(env("WQMCP_CORR_CACHE_SECONDS", "1800")))
        self.max_slot_seconds = max(1.0, float(env("WQMCP_CORR_MAX_SLOT_SECONDS", "600")))
        self.background = env("WQMCP_CORR_BACKGROUND", "1") != "0"
        self.queue_seconds = max(1.0, float(env("WQMCP_CORR_QUEUE_SECONDS", "3600")))
        # BRAIN often needs minutes for one correlation: only a long silence is throttling.
        self.stall_seconds = max(1.0, float(env("WQMCP_CORR_STALL_SECONDS", "300")))
        self.throttled_interval = max(1.0, float(env("WQMCP_CORR_THROTTLED_INTERVAL", "60")))
        # 0 = never pause, only slow down
        self.cooldown_seconds = max(0.0, float(env("WQMCP_CORR_COOLDOWN_SECONDS", "120")))
        self.cooldown_max = max(self.cooldown_seconds, float(env("WQMCP_CORR_COOLDOWN_MAX", "1800")))
        self.rotate = env("WQMCP_CORR_ROTATE", "0") != "0"
        self.check_max = max(1, int(env("WQMCP_CHECK_MAX_ALPHAS", "1")))
        self.aging_seconds = max(0.0, float(env("WQMCP_CORR_AGING_SECONDS", "300")))
        self.abandon_seconds = max(1.0, float(env("WQMCP_CORR_ABANDON_SECONDS", "1200")))
        self._asked: Dict[Tuple[str, str], float] = {}    # (alpha, kind) -> last time a caller asked
        self._high: set = set()                          # alphas asked for with priority="high"
        self._durations: collections.deque = collections.deque(maxlen=30)   # seconds a job took
        self._cooldown_until = 0.0
        self._cooldown_length = 0.0     # length of the current / last cooldown
        self._cooldown_level = 0        # cooldowns in a row without a result
        self._cooldown_total = 0.0      # seconds of cooldown ever started (jobs do not age in it)
        self._cooldown_reason = ""
        # How long a caller waits past its own budget for the first answer.
        self.grace_seconds = 5.0
        # (alpha, kind) -> {"admitted", "seen", "polls"}
        self._active: Dict[Tuple[str, str], Dict[str, float]] = {}
        # tickets, oldest first: (number, alpha, kind, enqueued at)
        self._waiting: List[Tuple[int, str, str, float]] = []
        self._tickets = 0
        self._next_request = 0.0
        self._last_result: Optional[float] = None    # last time BRAIN gave an answer
        self._pending_since: Optional[float] = None  # computing without an answer since
        self._throttled_until = 0.0                  # after an explicit 429
        self._cache: "collections.OrderedDict[Tuple[str, str], Tuple[float, Dict[str, Any]]]" = \
            collections.OrderedDict()
        self.inflight: Dict[Tuple[str, str], "asyncio.Task"] = {}

    # -- queue ---------------------------------------------------------------
    @staticmethod
    def lane(kind: str) -> str:
        """The submission check queues apart from the correlations."""
        return "check" if kind == "check" else "corr"

    def computing(self, lane: Optional[str] = None) -> set:
        """Alphas BRAIN is (as far as we know) still computing, in one lane or both."""
        now = time.monotonic()
        for key in [k for k, slot in self._active.items() if now - slot["seen"] > self.hold_seconds]:
            del self._active[key]
        if not self._active and not self._waiting:
            self._pending_since = None
        return {alpha for (alpha, kind), slot in self._active.items()
                if (lane is None or self.lane(kind) == lane)
                and (not self.rotate or now - slot["admitted"] <= self.max_slot_seconds)}

    def cooling(self) -> float:
        """Seconds of cooldown left (0 = requests may be sent)."""
        return max(0.0, self._cooldown_until - time.monotonic())

    def start_cooldown(self, seconds: Optional[float] = None, reason: str = "") -> float:
        """Stop every request for a while. Without `seconds` the length doubles
        with every cooldown in a row that was not followed by a result."""
        now = time.monotonic()
        self._cooldown_level += 1
        if seconds is None:
            seconds = min(self.cooldown_seconds * 2 ** (self._cooldown_level - 1), self.cooldown_max)
        seconds = max(0.0, float(seconds))
        left = self.cooling()
        self._cooldown_total += max(0.0, seconds - left)
        self._cooldown_until = max(self._cooldown_until, now + seconds)
        self._cooldown_length = max(seconds, left)
        self._cooldown_reason = reason
        self._pending_since = None      # the probe after the cooldown gets a fresh clock
        return self.cooling()

    def resume(self) -> None:
        """End the cooldown and forget the throttling: full speed again."""
        self._cooldown_until = 0.0
        self._throttled_until = 0.0
        self._cooldown_level = 0
        self._pending_since = time.monotonic() if self._active else None

    def check_stall(self) -> bool:
        """Called before every request: True = a cooldown is on (it may just have
        started because BRAIN has not answered for stall_seconds)."""
        if self.cooling() > 0:
            return True
        now = time.monotonic()
        if (self.cooldown_seconds > 0 and self._pending_since is not None
                and now - self._pending_since > self.stall_seconds):
            silent = int(now - self._pending_since)
            self.start_cooldown(reason=f"BRAIN returned no result for {silent}s")
            return True
        return False

    def throttled(self) -> bool:
        now = time.monotonic()
        if now < self._throttled_until or self.cooling() > 0 or self._cooldown_level > 0:
            return True
        return self._pending_since is not None and now - self._pending_since > self.stall_seconds

    def limit(self, lane: str = "corr") -> int:
        """How many alphas may be computed at once right now."""
        if lane == "check":
            return self.check_max      # /check is not throttled with the correlation endpoints
        if self.cooling() > 0:
            return 0
        if self.throttled():
            return 1
        return self.max_alphas

    def _order(self, lane: str) -> List[Tuple[int, str, str, float]]:
        """The waiting tickets of a lane in the order they are served: jobs that
        already had a turn, then alphas asked for their prod correlation only,
        then everybody else; first come, first served within each."""
        tickets = [t for t in self._waiting if self.lane(t[2]) == lane]
        kinds: Dict[str, set] = {}
        since: Dict[str, float] = {}
        for _, alpha, kind, at in tickets:
            kinds.setdefault(alpha, set()).add(kind)
            since[alpha] = min(since.get(alpha, at), at)
        now = time.monotonic()

        def rank(ticket: Tuple[int, str, str, float]) -> Tuple[int, int, int, int]:
            number, alpha = ticket[0], ticket[1]
            favoured = kinds[alpha] == {"prod"} or now - since[alpha] >= self.aging_seconds
            return (0 if number < 0 else 1, 0 if alpha in self._high else 1,
                    0 if favoured else 1, abs(number))
        return sorted(tickets, key=rank)

    def asked(self, alpha_id: str, kind: str, priority: str = "normal") -> None:
        """A caller wants this result (again)."""
        self._asked[(alpha_id, kind)] = time.monotonic()
        if str(priority or "").lower() == "high":
            self._high.add(alpha_id)

    def abandoned(self, alpha_id: str, kind: str) -> bool:
        """True when nobody has asked for this for abandon_seconds."""
        at = self._asked.get((alpha_id, kind))
        return at is not None and time.monotonic() - at > self.abandon_seconds

    def typical_seconds(self) -> Optional[float]:
        """How long a job took lately (median), None before the first one finished."""
        if not self._durations:
            return None
        ordered = sorted(self._durations)
        return float(ordered[len(ordered) // 2])

    def estimated_wait(self, alpha_id: str, kind: str = "prod") -> Optional[int]:
        """Seconds until a waiting alpha starts, from how long jobs took lately."""
        typical = self.typical_seconds()
        position = self.position(alpha_id, kind)
        if typical is None or not position:
            return None
        lane = self.lane(kind)
        now = time.monotonic()
        slots = max(1, self.limit(lane) or 1)
        left = sorted(max(0.0, typical - (now - slot["admitted"]))
                      for (alpha, k), slot in self._active.items() if self.lane(k) == lane)
        busy = len(self.computing(lane))
        free_in = [0.0] * max(0, slots - busy) + left[:slots]
        free_in = (free_in + [typical] * slots)[:slots]
        rounds, place = divmod(position - 1, slots)
        return int(self.cooling() + sorted(free_in)[place] + rounds * typical)

    def enqueue(self, alpha_id: str, kind: str = "", front: bool = False) -> Tuple[int, str, str, float]:
        """front=True: a job that had its turn and was stopped by a cooldown keeps it."""
        self._tickets += 1
        ticket = (self._tickets, alpha_id, kind, time.monotonic())
        if front:
            head = len([t for t in self._waiting if t[0] < 0])
            ticket = (-self._tickets, alpha_id, kind, ticket[3])
            self._waiting.insert(head, ticket)
        else:
            self._waiting.append(ticket)
        return ticket

    def leave(self, ticket: Tuple[int, str, str, float]) -> None:
        if ticket in self._waiting:
            self._waiting.remove(ticket)

    def admit(self, ticket: Tuple[int, str, str, float], kind: str) -> bool:
        alpha_id = ticket[1]
        lane = self.lane(kind)
        if lane == "corr" and self.cooling() > 0:
            return False
        computing = self.computing(lane)
        if alpha_id not in computing:
            if len(computing) >= self.limit(lane):
                return False
            ahead = next((t for t in self._order(lane) if t[1] not in computing), ticket)
            if ahead[1] != alpha_id:
                return False   # somebody else goes first
            for key in [k for k in self._active if k[0] == alpha_id]:
                del self._active[key]   # an expired slot of this alpha: it starts over
        self.touch(alpha_id, kind)
        self.leave(ticket)
        return True

    def touch(self, alpha_id: str, kind: str, polled: bool = False) -> None:
        now = time.monotonic()
        slot = self._active.get((alpha_id, kind))
        if slot is None:
            slot = self._active[(alpha_id, kind)] = {"admitted": now, "seen": now, "polls": 0}
        slot["seen"] = now
        if polled:
            slot["polls"] += 1
        if self._pending_since is None and self.cooling() <= 0 and self.lane(kind) == "corr":
            self._pending_since = now      # the throttle clock only watches correlations

    def release(self, alpha_id: str, kind: str, finished: bool = False) -> None:
        slot = self._active.pop((alpha_id, kind), None)
        if finished and slot is not None and self.lane(kind) == "corr":
            self._durations.append(time.monotonic() - slot["admitted"])
        if finished:
            self._asked.pop((alpha_id, kind), None)
            if not any(k[0] == alpha_id for k in self._active) and \
                    not any(t[1] == alpha_id for t in self._waiting):
                self._high.discard(alpha_id)

    def answered(self) -> None:
        """BRAIN produced a result: whatever throttling there was is over."""
        now = time.monotonic()
        self._last_result = now
        self._throttled_until = 0.0
        self._cooldown_level = 0
        self._pending_since = now if self._active else None

    def rate_limited(self, retry_after: float = 0.0) -> None:
        """BRAIN answered 429 on one of these endpoints."""
        self._throttled_until = max(self._throttled_until,
                                    time.monotonic() + max(retry_after, self.throttled_interval))
        if self.cooldown_seconds > 0 and self.cooling() <= 0:
            length = min(self.cooldown_seconds * 2 ** self._cooldown_level, self.cooldown_max)
            self.start_cooldown(max(retry_after, length), reason="BRAIN answered 429")

    def position(self, ticket_or_alpha: Any, kind: str = "prod") -> int:
        """1 = next in line; counts the alphas (not requests) waiting ahead in the lane."""
        if isinstance(ticket_or_alpha, tuple):
            alpha_id, kind = ticket_or_alpha[1], ticket_or_alpha[2] or kind
        else:
            alpha_id = str(ticket_or_alpha)
        lane = self.lane(kind)
        computing = self.computing(lane)
        order = list(dict.fromkeys(t[1] for t in self._order(lane) if t[1] not in computing))
        return order.index(alpha_id) + 1 if alpha_id in order else 0

    def poll_delay(self, retry_after: float, polls: int, remaining: float) -> float:
        if self.throttled():
            return max(1.0, min(max(retry_after, self.throttled_interval), remaining))
        return _poll_delay(retry_after, polls, remaining)

    # -- what callers are told -------------------------------------------------
    def describe(self, alpha_id: str, kind: str) -> Dict[str, Any]:
        """The PENDING answer for a request whose job is still going on."""
        now = time.monotonic()
        slot = self._active.get((alpha_id, kind)) if alpha_id in self.computing() else None
        keeps = (" This server keeps polling in the background; call again later to collect the result."
                 if self.background else " Call again later.")
        cooling = self.cooling() if self.lane(kind) == "corr" else 0.0
        if cooling > 0:
            out = {"status": "PENDING", "cooling_down": True, "resumes_in_seconds": int(cooling) + 1,
                   "retry_after_seconds": float(int(cooling) + 1),
                   "note": (f"BRAIN is rate limiting ({self._cooldown_reason or 'cooldown'}): every correlation "
                            f"request is paused for another {int(cooling) + 1}s, because asking a throttled "
                            "BRAIN only slows it down further."
                            + (" The request keeps its place in the queue." if self.background else ""))}
            position = self.position(alpha_id, kind)
            if position:
                out["queue_position"] = position
                wait = self.estimated_wait(alpha_id, kind)
                if wait is not None:
                    out["estimated_wait_seconds"] = wait
            return out
        typical = self.typical_seconds()
        if slot is not None:
            out = {"status": "PENDING", "computing_for_seconds": int(now - slot["admitted"]),
                   "polls": int(slot["polls"]),
                   "retry_after_seconds": self.throttled_interval if self.throttled() else _PENDING_RETRY_SECONDS,
                   "note": "BRAIN has not answered yet (not a failure)." + keeps}
            if typical is not None:
                out["estimated_remaining_seconds"] = int(max(0.0, typical - (now - slot["admitted"])))
            return out
        out = {"status": "PENDING", "queued": True, "queue_position": self.position(alpha_id, kind),
               "retry_after_seconds": 30.0,
               "note": (f"Not started yet: waiting in line (at most {self.limit(self.lane(kind))} alphas "
                        "are computed at a time, to stay under BRAIN's rate limit)."
                        + (" It keeps its place in the background; call again later." if self.background
                           else " Call again later."))}
        wait = self.estimated_wait(alpha_id, kind)
        if wait is not None:
            out["estimated_wait_seconds"] = wait
            out["retry_after_seconds"] = float(max(15, min(wait, 300)))
        return out

    def snapshot(self) -> Dict[str, Any]:
        """Who is being computed and who waits, oldest first."""
        now = time.monotonic()
        live = self.computing()
        computing: Dict[str, Dict[str, Any]] = {}
        for (alpha, kind), slot in self._active.items():
            if alpha not in live:
                continue
            row = computing.setdefault(alpha, {"alpha_id": alpha, "checks": [], "for_seconds": 0, "polls": 0})
            row["checks"].append(kind)
            row["for_seconds"] = max(row["for_seconds"], int(now - slot["admitted"]))
            row["polls"] += int(slot["polls"])
        typical = self.typical_seconds()
        if typical is not None:
            for row in computing.values():
                row["estimated_remaining_seconds"] = int(max(0.0, typical - row["for_seconds"]))
        waiting: Dict[str, Dict[str, Any]] = {}
        for lane in ("corr", "check"):
            busy = self.computing(lane)
            for ticket in self._order(lane):
                _, alpha, kind, since = ticket
                if alpha in busy:
                    continue
                key = f"{lane}:{alpha}"
                if key not in waiting:
                    waiting[key] = {"position": self.position(alpha, kind), "alpha_id": alpha,
                                    "checks": [], "waiting_seconds": int(now - since)}
                    if lane == "check":
                        waiting[key]["lane"] = "submission check"
                    wait = self.estimated_wait(alpha, kind)
                    if wait is not None:
                        waiting[key]["estimated_wait_seconds"] = wait
                waiting[key]["checks"].append(kind)
        throttled = self.throttled()
        cooling = self.cooling()
        out: Dict[str, Any] = {
            "throttled": throttled,
            "cooling_down": cooling > 0,
            "max_at_a_time": self.limit(),
            "typical_seconds_per_alpha": None if typical is None else int(typical),
            "computing": sorted(computing.values(), key=lambda r: -r["for_seconds"]),
            "waiting": list(waiting.values()),
        }
        if self._last_result is not None:
            out["last_result_seconds_ago"] = int(now - self._last_result)
        if cooling > 0:
            out["cooldown"] = {"resumes_in_seconds": int(cooling) + 1,
                               "length_seconds": int(self._cooldown_length),
                               "in_a_row": self._cooldown_level, "reason": self._cooldown_reason}
            out["note"] = (f"Cooling down: no correlation request is sent for another {int(cooling) + 1}s. "
                           "The queue is kept; afterwards one alpha probes whether BRAIN answers again. "
                           "check_alpha(check=\"resume\") ends the cooldown now.")
        elif self._cooldown_level > 0:
            out["note"] = (f"Probing after {self._cooldown_level} cooldown(s): one alpha, polled every "
                           f"{int(self.throttled_interval)}s. A result restores full speed; no result within "
                           f"{int(self.stall_seconds)}s starts a longer cooldown.")
        elif throttled:
            silent = int(now - self._pending_since) if self._pending_since is not None else None
            out["note"] = ((f"BRAIN has returned no correlation for {silent}s" if silent is not None
                            else "BRAIN answered 429")
                           + ": treated as rate limiting. One alpha at a time, polled every "
                           f"{int(self.throttled_interval)}s, until BRAIN answers again. Nothing is lost: "
                           "the queue is worked through in order.")
        return out

    # -- pacing --------------------------------------------------------------
    async def pace(self) -> None:
        """Wait for the turn to send a request (never called during a cooldown)."""
        now = time.monotonic()
        at = max(now, self._next_request)
        self._next_request = at + self.min_interval
        if at > now:
            await asyncio.sleep(at - now)

    # -- cache ---------------------------------------------------------------
    def cached(self, key: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cache_seconds:
            return hit[1]
        self._cache.pop(key, None)
        return None

    def remember(self, key: Tuple[str, str], result: Dict[str, Any]) -> None:
        if self.cache_seconds <= 0:
            return
        self._cache[key] = (time.monotonic(), result)
        self._cache.move_to_end(key)
        while len(self._cache) > 200:
            self._cache.popitem(last=False)


_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_HAS_TZ_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def _iso_datetime(value: Any, end_of_day: bool = False) -> str:
    """BRAIN's alpha date filters only take ISO 8601 datetimes WITH a timezone
    (a bare "2026-06-29" is a 400). Accept a plain date too: it becomes the start
    (or, for an upper bound, the end) of that day in UTC; a datetime without a
    timezone is read as UTC."""
    text = str(value).strip()
    if _DATE_ONLY_RE.fullmatch(text):
        return f"{text}T{'23:59:59' if end_of_day else '00:00:00'}Z"
    if "T" in text and not _HAS_TZ_RE.search(text):
        return text + "Z"
    return text


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
_COMPACT_NOTE = ("Rows: margin in bps; robust/sub/y2_sharpe, cluster = check values; fails = checks "
                 "that miss their limit (\"SHARPE 1.4<1.58\"; \"(W)\" = BRAIN only warned) or BRAIN "
                 "failed; warns = other warnings. Absent keys are empty. A negative sharpe: negate "
                 "the expression.")

# Longest expression echoed in a compact row (0 = never cut).
_COMPACT_EXPR_CHARS = int(os.environ.get("WQMCP_COMPACT_EXPR_CHARS", "1500"))


def _plain(message: Any) -> Any:
    """BRAIN's error text without its markup (<linkToCommonErrorMessages>Learn more</...>)."""
    if not isinstance(message, str):
        return message
    return re.sub(r"\s*<(\w+)>[^<]*</\1>", "", message).strip()


def _compact_field(f: Dict[str, Any]) -> Dict[str, Any]:
    """The parts of a data field row that differ from field to field."""
    row = {"id": f.get("id"), "type": f.get("type"), "coverage": f.get("coverage"),
           "dateCoverage": f.get("dateCoverage"), "userCount": f.get("userCount"),
           "alphaCount": f.get("alphaCount"), "desc": _cut(f.get("description") or "", 80)}
    return {k: v for k, v in row.items() if v not in (None, "")}


def _cut(text: str, limit: int) -> str:
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}…(+{len(text) - limit} chars)"


def _limit_miss(check: Dict[str, Any]) -> Optional[str]:
    """ "SHARPE 1.4<1.58" when a LOW_ / HIGH_ check's value is on the wrong side of
    its limit, whatever result BRAIN gave it (it answers WARNING, not FAIL, for some)."""
    name = check.get("name") or ""
    value, limit = check.get("value"), check.get("limit")
    if isinstance(value, bool) or isinstance(limit, bool) \
            or not isinstance(value, (int, float)) or not isinstance(limit, (int, float)):
        return None
    if (name.startswith("LOW_") and value < limit) or (name.startswith("HIGH_") and value > limit):
        return f"{_short_check(name)} {value:g}{'<' if name.startswith('LOW_') else '>'}{limit:g}"
    return None


def _check_lists(checks: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """(fails, warns) of an alpha's checks, in short names.

    BRAIN is not consistent: within one batch it marks the same kind of miss
    (sharpe 1.14 below 1.58) FAIL for one alpha and WARNING for another. So a
    value on the wrong side of its limit is a fail whatever BRAIN called it;
    BRAIN's own verdict stays as a mark when it was not FAIL ("(W)" = WARNING). warns is what remains: the
    WARNING checks without a limit to miss (CLUSTER_TEST, ...)."""
    fails: List[str] = []
    warns: List[str] = []
    for c in checks:
        result, name = c.get("result"), c.get("name") or ""
        miss = _limit_miss(c)
        if miss:
            fails.append(miss + {"FAIL": "", "WARNING": " (W)"}.get(result, f" ({str(result)[:1]})"))
        elif result == "FAIL":
            fails.append(_short_check(name))
        elif result == "WARNING" and "MATCHES_" not in name:
            warns.append(_short_check(name))
    return fails, warns


# Settings echoed in a compact row, so children of a mixed-settings batch can be told apart.
_COMPACT_SETTING_KEYS = ("universe", "decay", "neutralization", "truncation", "maxTrade", "mode")
# Echoed as well, but only when they differ from BRAIN's default.
_COMPACT_EXTRA_SETTINGS = {"nanHandling": "OFF", "pasteurization": "ON", "maxPosition": "OFF",
                           "unitHandling": "VERIFY"}


def _compact_alpha_row(alpha: Dict[str, Any]) -> Dict[str, Any]:
    """One-line summary of a finished alpha — the handful of numbers an
    optimisation loop actually reads, instead of the full ~5KB alpha object."""
    is_ = alpha.get("is") or {}
    checks = is_.get("checks") or []
    regular = alpha.get("regular") or {}
    settings = alpha.get("settings") or {}

    def check_value(name: str):
        return next((c.get("value") for c in checks if c.get("name") == name), None)

    fails, warns = _check_lists([c for c in checks if isinstance(c, dict)])
    # No position ever taken: an expression that is constant (e.g. equal(x, 0) on a
    # field that is never 0). Its fails say little; the expression needs rethinking.
    empty = (is_.get("turnover") in (0, 0.0) and is_.get("sharpe") in (0, 0.0, None)) or \
        (is_.get("longCount") == 0 and is_.get("shortCount") == 0)
    if settings.get("simulationMode"):
        settings = {**settings, "mode": settings["simulationMode"]}
    row = {
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
        "fails": fails,
        "warns": warns,
        "set": {**{k: settings.get(k) for k in _COMPACT_SETTING_KEYS if k in settings},
                **{k: settings[k] for k, default in _COMPACT_EXTRA_SETTINGS.items()
                   if settings.get(k) not in (None, default)}},
        "expr": _cut(regular.get("code") or "", _COMPACT_EXPR_CHARS),
    }
    if empty:
        row["empty_signal"] = True
    # Absent means "nothing": FULL runs have no robust sharpe, most rows no fails.
    return {k: v for k, v in row.items() if v not in (None, [], {}, "")}


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


def _correlation_histogram(data: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """BRAIN answers the prod correlation with a histogram (how many production
    alphas per 0.1 band), not with the alphas themselves. The top bands and how
    many production alphas are above the threshold."""
    schema = data.get("schema") or {}
    columns = [p.get("name") for p in schema.get("properties") or [] if isinstance(p, dict)]
    if not {"min", "max", "alphas"} <= set(columns):
        return {}
    i_min, i_max, i_n = columns.index("min"), columns.index("max"), columns.index("alphas")
    bands = []
    for row in data.get("records") or []:
        if isinstance(row, list) and len(row) > max(i_min, i_max, i_n):
            lo, hi, n = _finite(row[i_min]), _finite(row[i_max]), row[i_n]
            if lo is not None and hi is not None and isinstance(n, int):
                bands.append((lo, hi, n))
    if not bands:
        return {}
    high = {f"{lo:g}..{hi:g}": n for lo, hi, n in bands if lo >= 0.5 and n}
    return {"histogram_top": high,
            "alphas_over_threshold": sum(n for lo, _, n in bands if lo >= threshold - 1e-9)}


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
        # ProdMemo estimates handed out while BRAIN had no answer: alpha -> (at, row)
        self._estimates: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._estimate_tasks: Dict[str, "asyncio.Task"] = {}
        # One gate for every client of this server: see CorrelationGate.
        self.correlation_gate = CorrelationGate()
        # Simulations created by this process (newest first), for get_simulation().
        self.recent_simulations: collections.deque = collections.deque(maxlen=50)
        # What was sent for each of them: a result row can then say which item of
        # the request it belongs to, and whether BRAIN answered with another alpha.
        self._submitted: "collections.OrderedDict[str, Dict[str, Any]]" = collections.OrderedDict()
        # Submissions waiting for a free simulation slot, first in, first out.
        self.submit_queue: List[Dict[str, Any]] = []
        self._queue_numbers = 0
        self._queue_worker: Optional["asyncio.Task"] = None
        self._queue_done: "collections.OrderedDict[str, Dict[str, Any]]" = collections.OrderedDict()
        # simulation id -> what to do with its results: {"tag", "labels"}
        self._tags: Dict[str, Dict[str, Any]] = {}
        self._logged: set = set()                     # simulations whose results are in the log
        self._seen: Dict[str, Dict[str, Any]] = {}    # simulation id -> first seen / progress / changed
        self._retried: Dict[str, str] = {}            # simulation id -> the id of its resubmission
        self._watchers: Dict[str, "asyncio.Task"] = {}
        self._checking: Dict[Tuple[str, bool], "asyncio.Task"] = {}
        # finished simulations already given in full to a client session: session -> ids
        self._delivered: "collections.OrderedDict[int, set]" = collections.OrderedDict()
        # (expression + settings, without the mode) -> id of the QUICK alpha it gave
        self._quick_alphas: "collections.OrderedDict[Tuple, str]" = collections.OrderedDict()
        # data field id -> (checked at, combinations it exists for) / None = no such field
        self._field_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
        self._operators: Optional[Tuple[float, Dict[str, str]]] = None
    
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
        polls = 0
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
                out = {"status": "PENDING", "retry_after_seconds": max(ra, _PENDING_RETRY_SECONDS)}
                if busy:
                    out["busy"] = net_error if resp is None else f"HTTP {resp.status_code}"
                return out
            await asyncio.sleep(_poll_delay(ra, polls, remaining))
            polls += 1

    async def _check_once(self, location: str, compact: bool = True) -> Dict[str, Any]:
        """One status check, plus what BRAIN's answer does not say by itself: how
        long it has been running and whether it is stuck, results recovered after
        a 404, a resubmission when BRAIN failed a batch without a reason, compact
        rows, and the results log of a tagged simulation."""
        location = _simulation_url(location)
        sim_id = location.rstrip("/").rsplit("/", 1)[-1]
        if sim_id in self._retried:           # BRAIN failed it without a reason; it was sent again
            new_id = self._retried[sim_id]
            state = await self._check_once(f"{self.base_url}/simulations/{new_id}", compact)
            return {**state, "retried_as": new_id, "note": (
                f"Simulation {sim_id} failed on BRAIN's side (every child FAIL, no message), so it was "
                f"resubmitted once as {new_id}; this is that run. " + str(state.get("note") or "")).strip()}
        key = (sim_id, compact)
        task = self._checking.get(key)
        if task is None or task.done():         # callers polling the same id share one check
            async def run() -> Dict[str, Any]:
                state = await self._check_once_raw(location, compact)
                return await self._after_check(sim_id, location, state, compact)
            task = asyncio.create_task(run())
            self._checking[key] = task
            task.add_done_callback(lambda t, k=key: self._checking.pop(k, None)
                                   if self._checking.get(k) is t else None)
        return dict(await asyncio.shield(task))

    async def _after_check(self, sim_id: str, location: str, state: Dict[str, Any],
                           compact: bool) -> Dict[str, Any]:
        sent = self._submitted.get(sim_id) or {}
        now = time.time()
        if state.get("status") == "ERROR" and state.get("http_status") == 404 and sent:
            found = await self._recover_results(sim_id, compact)
            if found is not None:
                state = found
        if state.get("status") == "RUNNING":
            seen = self._seen.setdefault(sim_id, {"first": now, "progress": None, "changed": now})
            progress = state.pop("progress", None)
            if progress != seen["progress"]:
                seen["progress"], seen["changed"] = progress, now
            started = sent.get("at") or seen["first"]
            state["running_seconds"] = int(now - started)
            is_multi = state.get("type") == "MULTI" or len(sent.get("items") or []) > 1
            limit = STALE_MULTI_SECONDS if is_multi else STALE_SINGLE_SECONDS
            stalled = now - seen["changed"]
            if stalled > limit:
                state.update(stale=True, stalled_seconds=int(stalled), note=(
                    f"Looks stuck: no progress for {int(stalled // 60)} minutes (BRAIN usually finishes a "
                    f"{'multi' if is_multi else 'single'} simulation well within {int(limit // 60)}). It holds "
                    "an account slot: cancel_simulation it and submit again, e.g. "
                    f"create_simulation(resubmit=\"{sim_id}\")."))
            return state
        self._seen.pop(sim_id, None)
        if self._is_platform_glitch(state) and sent.get("payloads") and AUTO_RETRY_GLITCH \
                and sim_id not in self._retried.values():
            retried = await self.resubmit(sim_id)
            if retried.get("simulation_id"):
                self._retried[sim_id] = retried["simulation_id"]
                return {"status": "RETRIED", "retried_as": retried["simulation_id"],
                        "simulation_id": sim_id, "retry_after_seconds": 30.0,
                        "note": ("BRAIN failed every child without giving a reason — a platform hiccup "
                                 "(the same batch usually passes when sent again). It was resubmitted once as "
                                 f"{retried['simulation_id']}; asking for {sim_id} follows the new run.")}
        if compact:
            state = _compact_state(state)
        await self._log_results(sim_id, state)
        return state

    @staticmethod
    def _is_platform_glitch(state: Dict[str, Any]) -> bool:
        """A multi where every child failed and not one says why."""
        rows = state.get("alpha_results") or []
        return (state.get("type") == "MULTI" and len(rows) > 1
                and all(r.get("status") not in ("COMPLETE", None) and not r.get("message") and not r.get("error")
                        for r in rows)
                and not state.get("errors"))

    # --- state that survives a restart -------------------------------------------------
    def save_state(self) -> None:
        """Write what a restart must not lose: what each simulation sent (running
        time, resubmit, recovery), tags and which results are already logged, and
        the submit queue. Written in a thread, a failed write is only logged."""
        if not STATE_FILE:
            return
        state = {
            "version": 1,
            "submitted": [[k, v] for k, v in list(self._submitted.items())[-300:]],
            "tags": {k: v for k, v in self._tags.items() if k in self._submitted},
            "logged": sorted(x for x in self._logged if x in self._submitted),
            "queue": [{k: v for k, v in e.items()} for e in self.submit_queue],
            "queue_numbers": self._queue_numbers,
            "recent": list(self.recent_simulations),
        }

        def write() -> None:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False)
            os.replace(tmp, STATE_FILE)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            try:
                write()
            except OSError as e:
                self.log(f"state file {STATE_FILE}: {e}", "WARNING")
            return

        def done(f: "asyncio.Future") -> None:
            if f.exception():
                self.log(f"state file {STATE_FILE}: {f.exception()}", "WARNING")
        loop.run_in_executor(None, write).add_done_callback(done)

    def load_state(self) -> None:
        """Read the state a previous process left (called once, at start)."""
        if not STATE_FILE or not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, ValueError) as e:
            self.log(f"state file {STATE_FILE} not read: {e}", "WARNING")
            return
        for k, v in state.get("submitted") or []:
            self._submitted[k] = v
        self._tags.update(state.get("tags") or {})
        self._logged.update(state.get("logged") or [])
        self._queue_numbers = max(self._queue_numbers, int(state.get("queue_numbers") or 0))
        self.submit_queue = [e for e in state.get("queue") or [] if e.get("status") == "QUEUED"]
        for row in reversed(state.get("recent") or []):
            self.recent_simulations.appendleft(row)
        self._resume_pending = True

    def resume_after_restart(self) -> None:
        """Once a loop runs: restart the submit queue and the watchers of tagged
        simulations whose results are not logged yet."""
        if not getattr(self, "_resume_pending", False):
            return
        self._resume_pending = False
        if self.submit_queue and (self._queue_worker is None or self._queue_worker.done()):
            self._queue_worker = asyncio.create_task(self._run_submit_queue())
        cutoff = time.time() - WATCH_SECONDS
        for sim_id, meta in list(self._tags.items()):
            if sim_id not in self._logged and (self._submitted.get(sim_id) or {}).get("at", 0) > cutoff:
                self._watch(sim_id)

    def delivered_to(self, session: int) -> set:
        """Finished simulations whose rows this client session already got in full."""
        seen = self._delivered.setdefault(session, set())
        self._delivered.move_to_end(session)
        while len(self._delivered) > 50:
            self._delivered.popitem(last=False)
        return seen

    def _note_quick(self, item: Dict[str, Any], alpha: Dict[str, Any], row: Dict[str, Any]) -> None:
        """Remember QUICK results; mark a FULL result of the same expression and settings."""
        mode = str((alpha.get("settings") or {}).get("simulationMode") or "FULL").upper()
        settings = {k: v for k, v in item["settings"].items() if k not in ("simulationMode", "visualization")}
        key = _match_key(item["expr"], settings)
        if mode == "QUICK":
            self._quick_alphas[key] = str(alpha.get("id"))
            self._quick_alphas.move_to_end(key)
            while len(self._quick_alphas) > 2000:
                self._quick_alphas.popitem(last=False)
        elif key in self._quick_alphas and self._quick_alphas[key] != alpha.get("id"):
            row["same_as_quick"] = self._quick_alphas[key]

    async def resubmit(self, sim_id: str, queue: bool = True) -> Dict[str, Any]:
        """Send exactly what was sent for sim_id again (same items, settings, tag)."""
        sent = self._submitted.get(str(sim_id))
        if not sent or not sent.get("payloads"):
            raise ValueError(f"this server has no record of what simulation {sim_id} sent "
                             "(it only knows the ones it created since its last restart)")
        payloads, meta = sent["payloads"], self._tags.get(str(sim_id))
        if sent.get("kind") == "multi":
            return await self.create_multi_simulation(payloads, queue, meta)
        item = payloads[0]
        data = SimulationData(type=item.get("type", "REGULAR"), settings=SimulationSettings(**item["settings"]),
                              **{k: item[k] for k in ("regular", "combo", "selection") if item.get(k)})
        return await self.create_simulation(data, queue, meta)

    async def _recover_results(self, sim_id: str, compact: bool) -> Optional[Dict[str, Any]]:
        """BRAIN sometimes forgets a simulation (404) that did run. Its alphas are
        still in the alpha list: find them by expression and settings, created
        after the submission."""
        sent = self._submitted.get(sim_id) or {}
        items = sent.get("items") or []
        since = datetime.utcfromtimestamp(sent.get("at", time.time()) - 120).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            listed = await self.get_user_alphas(stage=None, limit=100, start_date=since, order="-dateCreated")
        except Exception:
            return None
        found: Dict[int, Dict[str, Any]] = {}
        used: set = set()
        for index, item in enumerate(items):
            want = _match_key(item["expr"], item["settings"])
            for alpha in listed.get("results") or []:
                if alpha.get("id") in used:
                    continue
                code = ((alpha.get("regular") or alpha.get("combo") or {}).get("code"))
                if _match_key(code, alpha.get("settings") or {}) == want:
                    found[index] = alpha
                    used.add(alpha.get("id"))
                    break
        if not found:
            return None
        rows = []
        for index in range(len(items)):
            alpha = found.get(index)
            if alpha is None:
                rows.append({"index": index, "status": "UNKNOWN", "submitted_expr": _cut(items[index]["expr"], 200)})
            else:
                rows.append({"index": index, "status": "COMPLETE",
                             **(_compact_alpha_row(alpha) if compact else {"details": alpha})})
        multi = sent.get("kind") == "multi"
        out: Dict[str, Any] = {
            "status": "COMPLETE" if len(found) == len(items) else "FINISHED_WITH_ERRORS",
            "recovered": True,
            "note": ("BRAIN answered 404 for this simulation although it ran: the results were recovered "
                     "from your alpha list by expression and settings."),
        }
        if multi:
            out.update(type="MULTI", total_children=len(items), alpha_results=rows)
        else:
            out["alpha"] = rows[0]
        return out

    def _watch(self, sim_id: str) -> None:
        """Follow a tagged simulation in the background until it ends, so its
        results reach the log without anybody polling for them."""
        if sim_id in self._watchers and not self._watchers[sim_id].done():
            return

        async def run() -> None:
            deadline = time.time() + WATCH_SECONDS
            while time.time() < deadline:
                await asyncio.sleep(WATCH_INTERVAL)
                try:
                    state = await self._check_once(f"{self.base_url}/simulations/{sim_id}")
                except Exception:
                    continue
                if state.get("status") not in ("RUNNING", "UNKNOWN", "RETRIED"):
                    return
        try:
            task = asyncio.get_running_loop().create_task(run())
        except RuntimeError:
            return
        self._watchers[sim_id] = task
        task.add_done_callback(lambda t, k=sim_id: self._watchers.pop(k, None))

    async def _log_results(self, sim_id: str, state: Dict[str, Any]) -> None:
        """Append the finished rows of a tagged simulation to results/<tag>.jsonl."""
        meta = self._tags.get(sim_id)
        if not meta or sim_id in self._logged or state.get("status") in ("RUNNING", "UNKNOWN", "RETRIED", None):
            return
        rows = list(state.get("alpha_results") or ([state["alpha"]] if isinstance(state.get("alpha"), dict) else []))
        if not rows and state.get("message"):
            rows = [{"index": 0, "status": state.get("status"), "message": state.get("message")}]
        common = state.get("set") or {}
        labels = meta.get("labels") or []
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        lines = []
        for row in rows:
            index = row.get("index", 0)
            record = {"ts": stamp, "tag": meta["tag"], "simulation_id": sim_id,
                      "index": meta.get("item", index)}    # the item of the original request
            if index < len(labels) and labels[index]:
                record["label"] = labels[index]
            record.update({k: v for k, v in row.items() if k not in ("index", "location")})
            if common:
                record["set"] = {**common, **(row.get("set") or {})}
            lines.append(json.dumps(record, ensure_ascii=False))
        for index in state.get("cancelled") or []:
            lines.append(json.dumps({"ts": stamp, "tag": meta["tag"], "simulation_id": sim_id, "index": index,
                                     "status": "CANCELLED"}, ensure_ascii=False))
        if not lines:
            return
        path = _results_path(meta["tag"])

        def write() -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        try:
            await asyncio.get_running_loop().run_in_executor(None, write)
            self._logged.add(sim_id)
            self.save_state()
        except OSError as e:
            self.log(f"results log {path}: {e}", "WARNING")

    async def _check_once_raw(self, location: str, compact: bool = True) -> Dict[str, Any]:
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
            return await self._check_multi_children(location, children, compact, body)

        if "Retry-After" in resp.headers:
            sent = self._submitted.get(location.rstrip("/").rsplit("/", 1)[-1]) or {}
            out = {
                "status": "RUNNING",
                "progress": body.get("progress"),
                "retry_after_seconds": _retry_after_seconds(resp) or 5.0,
                "progress_url": location,
                "note": "Still running; check again after retry_after_seconds (do other work meanwhile).",
            }
            if sent.get("kind") == "multi":
                out.update(type="MULTI", total_children=len(sent.get("items") or []), note=(
                    "Multi-simulation still running. BRAIN does not list its children before the whole "
                    "batch is done, so there is no per-child count; stale flags a batch that stopped moving."))
            return out

        # Finished single simulation
        alpha_id = body.get("alpha")
        if not alpha_id:
            # Failed simulation: surface BRAIN's own error message.
            out = {"status": body.get("status", "ERROR"), "progress_url": location,
                   "message": _plain(body.get("message") or body.get("detail") or body.get("details"))}
            if isinstance(body.get("location"), dict):
                out["at"] = body["location"]
            if not compact:
                out["raw"] = body
            return out
        alpha = await self._request('get', f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}")
        alpha.raise_for_status()
        if compact:
            return {"status": "COMPLETE", "progress_url": location,
                    "alpha": _compact_alpha_row(alpha.json()), "note": _COMPACT_NOTE}
        return {"status": "COMPLETE", "progress_url": location, "alpha": alpha.json(), "note": _FLIP_NOTE}

    async def _check_multi_children(self, location: str, children: List[str],
                                    compact: bool = True,
                                    parent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check all children of a multi-simulation concurrently. Every row says which
        item of the request it is (index) and what was sent for it."""
        sim_id = location.rstrip("/").rsplit("/", 1)[-1]
        sent = self._submitted.get(sim_id) or {}
        items: List[Dict[str, Any]] = sent.get("items") or []

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
                out = {"location": url, "status": b.get("status", "ERROR"),
                       "message": _plain(b.get("message") or b.get("detail") or b.get("details"))}
                if isinstance(b.get("location"), dict):
                    out["at"] = b["location"]      # line / start / end of the error in the expression
                code = b.get("regular")
                out["sent"] = code.get("code") if isinstance(code, dict) else code
                return out
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
                "children": [{"index": i, **s} for i, s in enumerate(states)],
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
                return {**s, "details": d.json()}
            except Exception as e:
                return {**s, "error": f"failed to fetch alpha details: {e}"}

        detailed = list(await asyncio.gather(*[with_details(s) for s in states]))
        order = _match_children(detailed, items)
        first_row_of: Dict[str, int] = {}
        full: List[Dict[str, Any]] = []
        for position, s in enumerate(detailed):
            index = order[position]
            item = items[index] if index < len(items) else None
            alpha = s.get("details")
            if alpha is None:
                row = {"index": index, **{k: v for k, v in s.items() if k != "sent"}}
                expr = (item or {}).get("expr") or s.get("sent")
                if expr:
                    row["submitted_expr"] = _cut(expr, _COMPACT_EXPR_CHARS)
                full.append(row)
                continue
            row = ({"index": index, "status": "COMPLETE", **_compact_alpha_row(alpha)} if compact
                   else {"index": index, **s})
            alpha_id = str(alpha.get("id") or s.get("alpha_id"))
            if alpha_id in first_row_of:
                row["duplicate_of_index"] = first_row_of[alpha_id]
                row["warning"] = ("BRAIN returned the same alpha for this item as for item "
                                  f"{first_row_of[alpha_id]}: to BRAIN the two are identical.")
            else:
                first_row_of[alpha_id] = index
            if item is not None:
                mismatch = _settings_mismatch(item["settings"], alpha.get("settings") or {})
                if mismatch:
                    row["settings_mismatch"] = mismatch
                self._note_quick(item, alpha, row)
                code = ((alpha.get("regular") or alpha.get("combo") or {}).get("code") or "")
                same = _same_code(code, item["expr"])
                created = _epoch(alpha.get("dateCreated"))
                older = created is not None and created < sent.get("at", 0) - 120
                if not same or older:
                    row["reused_alpha"] = True
                    row["submitted_expr"] = _cut(item["expr"], _COMPACT_EXPR_CHARS)
                    row["warning"] = ((row.get("warning", "") + " ") if row.get("warning") else "") + (
                        "BRAIN answered with an alpha that already existed"
                        + ("" if same else " and whose expression is not the one sent (it treats them as "
                                           "the same, e.g. a field alias)")
                        + f": id and expr are the old alpha's (created {alpha.get('dateCreated')}).")
            full.append(row)
        full.sort(key=lambda r: r["index"])

        failed = [s for s in full if s.get("status") != "COMPLETE" or s.get("error")]
        result = {
            # BRAIN cancels the whole batch when a child fails: never call that COMPLETE.
            "status": "FINISHED_WITH_ERRORS" if failed else "COMPLETE",
            "type": "MULTI",
            "total_children": len(full),
            "failed_children": len(failed),
            "alpha_results": full,
            "progress_url": location,
            "note": _COMPACT_NOTE if compact else _FLIP_NOTE,
        }
        if items and len(full) != len(items):
            got = {r["index"] for r in full}
            result["missing_children"] = [i for i in range(len(items)) if i not in got]
            result["expected_children"] = len(items)
        if failed:
            causes = [{"index": r["index"], "status": r.get("status"),
                       "message": r.get("message") or r.get("error"),
                       **({"at": r["at"]} if r.get("at") else {}),
                       **({"submitted_expr": r["submitted_expr"]} if r.get("submitted_expr") else {})}
                      for r in failed if r.get("message") or r.get("error")]
            parent_message = (parent or {}).get("message") or (parent or {}).get("detail")
            if causes:
                result["errors"] = causes
            elif parent_message:
                result["errors"] = [{"index": None, "message": parent_message}]
            result["note"] = (
                ("errors[] holds BRAIN's reason and the index of the item that caused it. "
                 if result.get("errors") else
                 "BRAIN gave no reason for any child. Run the items one by one (mode=\"concurrent\") "
                 "to find the one it refuses. ")
                + "The children without a message were cancelled because of it: BRAIN cancels a "
                  "whole multi-simulation when one child fails. " + result["note"])
        return result

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
        told = _retry_after_seconds(response)
        retry_after = told or 30.0
        return {
            "status": "RATE_LIMITED",
            "retry_after_seconds": retry_after,
            # BRAIN sends no Retry-After here as a rule: 30 is then this server's guess
            "retry_after_from_brain": bool(told),
            "note": ("BRAIN's per-account concurrent simulation limit is reached "
                     "(other simulations are still running on this account). "
                     f"Retry after ~{int(retry_after)}s, or first finish/check the running "
                     "ones (get_simulation); you can do other work meanwhile."),
        }

    def _remember_simulation(self, simulation_id: str, kind: str, sim_type: str, count: int,
                             payloads: Optional[List[Dict[str, Any]]] = None,
                             meta: Optional[Dict[str, Any]] = None) -> None:
        self.recent_simulations.appendleft({
            "simulation_id": simulation_id, "kind": kind, "type": sim_type, "alphas": count,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if payloads:
            self._submitted[simulation_id] = {
                "at": time.time(), "kind": kind, "payloads": payloads,
                "items": [{"expr": p.get("regular") or p.get("combo") or "",
                           "settings": dict(p.get("settings") or {})} for p in payloads]}
            while len(self._submitted) > 300:
                self._submitted.popitem(last=False)
        if meta and meta.get("tag"):
            self._tags[simulation_id] = meta
            self._watch(simulation_id)
        self.save_state()

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

    # --- submissions waiting for a free simulation slot -----------------------------
    def _queue_submission(self, kind: str, body: Any, what: str, sim_type: str,
                          payloads: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
        if len(self.submit_queue) >= SUBMIT_QUEUE_MAX:
            return {"status": "RATE_LIMITED", "retry_after_seconds": 60.0,
                    "note": f"The account's simulation slots are full and so is this server's submit "
                            f"queue ({SUBMIT_QUEUE_MAX} waiting). Submit again later."}
        self._queue_numbers += 1
        entry = {"queue_id": f"Q{self._queue_numbers}", "status": "QUEUED", "kind": kind, "body": body,
                 "what": what, "type": sim_type, "payloads": payloads, "alphas": len(payloads),
                 "enqueued": time.time(), "attempts": 0, "meta": meta}
        self.submit_queue.append(entry)
        self.save_state()
        if self._queue_worker is None or self._queue_worker.done():
            self._queue_worker = asyncio.create_task(self._run_submit_queue())
        return self.queued_state(entry)

    def queued_submission(self, queue_id: str) -> Optional[Dict[str, Any]]:
        queue_id = str(queue_id).strip().upper()
        return next((e for e in self.submit_queue if e["queue_id"] == queue_id),
                    self._queue_done.get(queue_id))

    def queued_state(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """What a caller is told about a submission that went through the queue."""
        out = {"status": entry["status"], "queue_id": entry["queue_id"], "kind": entry["kind"],
               "alphas": entry["alphas"]}
        if entry["status"] == "QUEUED":
            ahead = self.submit_queue.index(entry)
            out.update(queue_position=ahead + 1, waiting_seconds=int(time.time() - entry["enqueued"]),
                       attempts=entry["attempts"], retry_after_seconds=15.0,
                       note=("The account's simulation slots are full. This request waits in this "
                             "server's queue (first in, first out) and is sent as soon as BRAIN accepts "
                             f"it. Follow it with get_simulation(simulation_ids=\"{entry['queue_id']}\"); "
                             "cancel_simulation takes it out."))
        else:
            out.update({k: entry[k] for k in ("simulation_id", "progress_url", "error", "waited_seconds")
                        if entry.get(k) is not None})
        return out

    def submit_queue_snapshot(self) -> List[Dict[str, Any]]:
        return [{"queue_position": i + 1, "queue_id": e["queue_id"], "kind": e["kind"],
                 "alphas": e["alphas"], "waiting_seconds": int(time.time() - e["enqueued"]),
                 "attempts": e["attempts"],
                 "expr": _cut((e["payloads"][0].get("regular") or e["payloads"][0].get("combo") or ""), 80)}
                for i, e in enumerate(self.submit_queue)]

    def _queue_finished(self, entry: Dict[str, Any], status: str, **fields: Any) -> None:
        if entry in self.submit_queue:
            self.submit_queue.remove(entry)
        entry.update(status=status, waited_seconds=int(time.time() - entry["enqueued"]), **fields)
        entry.pop("body", None)
        self._queue_done[entry["queue_id"]] = entry
        while len(self._queue_done) > 200:
            self._queue_done.popitem(last=False)
        self.save_state()

    def cancel_queued_submission(self, queue_id: str) -> Optional[Dict[str, Any]]:
        entry = self.queued_submission(queue_id)
        if entry is None:
            return None
        if entry["status"] == "QUEUED":
            self._queue_finished(entry, "CANCELLED")
            return {"queue_id": entry["queue_id"], "cancelled": True,
                    "note": "Taken out of the submit queue; it was never sent to BRAIN."}
        return {"queue_id": entry["queue_id"], "cancelled": False, **self.queued_state(entry),
                "note": "No longer in the queue." + (
                    " It was sent: cancel the simulation_id instead." if entry.get("simulation_id") else "")}

    async def _run_submit_queue(self) -> None:
        """Send the queued submissions in order; the first one blocks the others, so
        nothing overtakes."""
        while self.submit_queue:
            entry = self.submit_queue[0]
            if time.time() - entry["enqueued"] > SUBMIT_QUEUE_SECONDS:
                self._queue_finished(entry, "ERROR", error=(
                    f"no simulation slot became free within {int(SUBMIT_QUEUE_SECONDS // 60)} minutes"))
                continue
            try:
                posted = await self._post_simulation(entry["body"], entry["what"])
            except Exception as e:   # BRAIN refused it: not a matter of waiting
                self._queue_finished(entry, "ERROR", error=str(e) or repr(e))
                continue
            if isinstance(posted, dict):
                entry["attempts"] += 1
                told = posted["retry_after_seconds"] if posted.get("retry_after_from_brain") else 0.0
                await asyncio.sleep(min(max(told, SUBMIT_QUEUE_INTERVAL), SUBMIT_QUEUE_MAX_INTERVAL))
                continue
            simulation_id, location = posted
            self._remember_simulation(simulation_id, entry["kind"], entry["type"], entry["alphas"],
                                      entry["payloads"], entry.get("meta"))
            self._queue_finished(entry, "SUBMITTED", simulation_id=simulation_id, progress_url=location)

    async def _submit(self, kind: str, body: Any, what: str, sim_type: str,
                      payloads: List[Dict[str, Any]], queue: bool,
                      meta: Optional[Dict[str, Any]] = None) -> Any:
        """POST now, or queue when the slots are full. Returns (id, location), or the
        RATE_LIMITED / QUEUED dict."""
        if queue and self.submit_queue:
            return self._queue_submission(kind, body, what, sim_type, payloads, meta)   # nobody overtakes
        posted = await self._post_simulation(body, what)
        if isinstance(posted, dict) and queue:
            return self._queue_submission(kind, body, what, sim_type, payloads, meta)
        return posted

    async def create_simulation(self, simulation_data: SimulationData, queue: bool = False,
                                meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Submit one simulation (REGULAR, SUPER or REGION_AGNOSTIC) and return at once."""
        self.log("🚀 Creating simulation...", "INFO")
        payload = self.simulation_payload(simulation_data)
        posted = await self._submit("single", payload, "simulation", simulation_data.type, [payload], queue, meta)
        if isinstance(posted, dict):
            return posted
        simulation_id, location = posted
        self._remember_simulation(simulation_id, "single", simulation_data.type, 1, [payload], meta)
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

    async def create_multi_simulation(self, payloads: List[Dict[str, Any]],
                                      queue: bool = False, meta: Optional[Dict[str, Any]] = None
                                      ) -> Dict[str, Any]:
        """Submit 2-10 simulation items as ONE multi-simulation (one POST, one parent id).
        BRAIN cancels the whole batch when any child fails."""
        self.log(f"🚀 Creating multi-simulation ({len(payloads)} alphas)...", "INFO")
        posted = await self._submit("multi", payloads, "multi-simulation",
                                    payloads[0].get("type", "REGULAR"), payloads, queue, meta)
        if isinstance(posted, dict):
            return posted
        simulation_id, location = posted
        self._remember_simulation(simulation_id, "multi", payloads[0].get("type", "REGULAR"), len(payloads),
                                  payloads, meta)
        return {
            "status": "SUBMITTED",
            "simulation_id": simulation_id,
            "multisimulation_id": simulation_id,
            "expected_children": len(payloads),
            "progress_url": location,
            "note": ("Multi-simulation is running asynchronously (typically 3-10 minutes for "
                     f"{len(payloads)} alphas). Call get_simulation with this simulation_id to get "
                     "per-child progress and, once finished, one compact row per alpha (each row "
                     "echoes its settings). You can do other work between checks."),
        }

    async def field_info(self, name: str) -> Optional[Dict[str, Any]]:
        """{"type": MATRIX / VECTOR / GROUP, "combos": [{region, delay, universe}]};
        None = BRAIN knows no such field. Raises when BRAIN cannot say (then nothing
        is concluded). The type is BRAIN's: a local catalogue may call an event
        (VECTOR) field MATRIX."""
        hit = self._field_cache.get(name)
        if hit and time.monotonic() - hit[0] < 6 * 3600:
            return hit[1]
        response = await self._request('get', f"{self.base_url}/data-fields/{_seg(name, 'data field')}")
        if response.status_code == 404:
            found: Optional[Dict[str, Any]] = None
        elif response.status_code == 200:
            body = response.json()
            found = {"type": str(body.get("type") or "").upper() or None,
                     "combos": [d for d in (body.get("data") or []) if isinstance(d, dict)]}
        else:
            raise Exception(_http_error_detail(response, "data field lookup"))
        self._field_cache[name] = (time.monotonic(), found)
        return found

    async def field_combinations(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """Where a data field exists: [{region, delay, universe}, ...]; None = unknown field."""
        info = await self.field_info(name)
        return None if info is None else info["combos"]

    async def operator_categories(self) -> Dict[str, str]:
        """name -> category of every operator usable in an alpha expression
        (REGULAR scope), cached for a day. {} when BRAIN cannot be asked."""
        if self._operators and time.monotonic() - self._operators[0] < 86400:
            return self._operators[1]
        try:
            data = await self.get_operators()
        except Exception:
            return {}
        rows = data.get("operators") if isinstance(data, dict) else data
        names = {o["name"]: o.get("category") or "" for o in rows or []
                 if isinstance(o, dict) and o.get("name") and "REGULAR" in (o.get("scope") or ["REGULAR"])}
        if names:
            self._operators = (time.monotonic(), names)
        return names

    async def field_problems(self, items: List[Tuple[int, str, Dict[str, Any]]],
                             budget: float = 20.0, max_ops: Optional[int] = None
                             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """(problems, warnings) of FASTEXPR expressions, checked against BRAIN's own
        catalogue: data fields it does not know or has no data for in the item's
        region / delay, operators it does not have, VECTOR fields without a vec_*,
        numbers where a group is needed; warnings: more operators than max_ops.
        items = (index, expression, settings). Best effort: what cannot be looked
        up in time is not reported."""
        names = list(dict.fromkeys(n for _, expr, _ in items for n in _field_candidates(expr)))
        gate = asyncio.Semaphore(4)
        known: Dict[str, Any] = {}

        async def look_up(name: str) -> None:
            async with gate:
                try:
                    known[name] = await self.field_info(name)
                except Exception:
                    pass
        operators: Dict[str, str] = {}

        async def load_operators() -> None:
            operators.update(await self.operator_categories())
        try:
            await asyncio.wait_for(asyncio.gather(load_operators(), *(look_up(n) for n in names)),
                                   timeout=budget)
        except asyncio.TimeoutError:
            pass
        types = {n: info["type"] for n, info in known.items() if info and info.get("type")}
        rows, warnings = [], []
        for index, expr, settings in items:
            issues = []
            region = str(settings.get("region") or "").upper()
            delay = settings.get("delay")
            for name in _field_candidates(expr):
                if name not in known:
                    continue
                combos = None if known[name] is None else known[name]["combos"]
                if known[name] is None:
                    issues.append(f"unknown data field {name!r} (BRAIN has no field of that name; "
                                  "a typo, or a variable that is never assigned?)")
                elif region and region != "ALL" and combos and len(combos) < _FIELD_DATA_ROWS_CAP and not any(
                        str(c.get("region")).upper() == region and c.get("delay") in (None, delay)
                        for c in combos):
                    has = sorted({f"{c.get('region')}/D{c.get('delay')}" for c in combos})
                    issues.append(f"data field {name!r} has no data for {region} delay {delay} "
                                  f"(it exists for {', '.join(has[:8])})")
            issues += _semantic_issues(expr, operators, types)
            if issues:
                rows.append({"index": index, "expr": _cut(expr, 120), "issues": issues})
            if max_ops:
                ops = _estimated_ops(expr)
                if ops > max_ops:
                    warnings.append({"index": index, "expr": _cut(expr, 120),
                                     "issue": f"about {ops} operators by BRAIN's count, more than max_ops={max_ops} "
                                              "(it counts every call, infix and unary operator)"})
        return rows, warnings

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
                          limit: Optional[int] = None, offset: int = 0, detail: bool = False,
                          category: Optional[str] = None) -> Dict[str, Any]:
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
            if category:
                params['category'] = category     # BRAIN filters by category id: model, pv, analyst, ...
            # Without a limit BRAIN sends every dataset at once (hundreds, ~0.5 MB).
            page = max(1, min(int(limit or 20), 50))
            params['limit'] = page
            if offset:
                params['offset'] = max(0, int(offset))

            response = await self._request('get', f"{self.base_url}/data-sets", params=params)
            response.raise_for_status()
            response_json = response.json()
            results = response_json.get('results') or []
            if not detail:
                keep = ('id', 'name', 'category', 'subcategory', 'coverage', 'valueScore', 'userCount',
                        'alphaCount', 'fieldCount', 'pyramidMultiplier', 'themes')
                rows = []
                for d in results:
                    row = {k: d.get(k) for k in keep if d.get(k) not in (None, [], '')}
                    for k in ('category', 'subcategory'):
                        if isinstance(row.get(k), dict):
                            row[k] = row[k].get('id') or row[k].get('name')
                    if isinstance(row.get('themes'), list):
                        row['themes'] = [t.get('name', t) if isinstance(t, dict) else t for t in row['themes']]
                    row['description'] = _cut(d.get('description') or '', 160)
                    rows.append(row)
                response_json['results'] = rows
            count = response_json.get('count')
            shown = max(0, int(offset or 0)) + len(results)
            response_json['next_offset'] = shown if isinstance(count, int) and shown < count else None
            response_json.pop('next', None)
            response_json.pop('previous', None)
            if not results:
                response_json['note'] = ("No dataset for these settings: check region / delay / universe "
                                         "with get_platform_setting_options.")
            return response_json
        except Exception as e:
            self.log(f"Failed to get datasets: {str(e)}", "ERROR")
            raise
    
    async def get_datafields(self, instrument_type: str = "EQUITY", region: str = "USA",
                            delay: int = 1, universe: str = "TOP3000", theme: str = "false",
                            dataset_id: Optional[str] = None, data_type: str = "",
                            search: Optional[str] = None, limit: int = 50,
                            offset: int = 0, compact: bool = False) -> Dict[str, Any]:
        """Data fields, paged. BRAIN answers at most 50 per request; a larger limit
        (up to 500) is served from consecutive requests."""
        await self.ensure_authenticated()
        want = max(1, min(int(limit or 50), 500))
        start = max(0, int(offset or 0))
        base = {'instrumentType': instrument_type, 'region': region, 'delay': delay, 'universe': universe}
        if data_type and data_type != 'ALL':
            base['type'] = data_type
        if dataset_id:
            base['dataset.id'] = dataset_id
        if search:
            base['search'] = search
        results: List[Dict[str, Any]] = []
        count = None
        while len(results) < want:
            page = min(50, want - len(results))
            response = await self._request('get', f"{self.base_url}/data-fields",
                                           params={**base, 'limit': page, 'offset': start + len(results)})
            if response.status_code >= 400:
                raise Exception(_http_error_detail(response, "data fields"))
            body = response.json()
            count = body.get('count', count)
            batch = body.get('results') or []
            results += batch
            if len(batch) < page or (isinstance(count, int) and start + len(results) >= count):
                break
        if compact:
            results = [_compact_field(f) for f in results]
        shown = start + len(results)
        out: Dict[str, Any] = {"count": count, "results": results,
                               "next_offset": shown if isinstance(count, int) and shown < count else None}
        if dataset_id and results and compact:
            out["dataset"] = dataset_id      # said once instead of on every row
        if not results:
            out["note"] = ("No field for these settings: check region / delay / universe with "
                           "get_platform_setting_options, and the dataset id with get_datasets.")
        return out

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
                params["dateCreated>"] = _iso_datetime(start_date)
            if end_date:
                params["dateCreated<"] = _iso_datetime(end_date, end_of_day=True)
            if submission_start_date:
                params["dateSubmitted>"] = _iso_datetime(submission_start_date)
            if submission_end_date:
                params["dateSubmitted<"] = _iso_datetime(submission_end_date, end_of_day=True)
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
                        "note": "Submission is being processed. Call submit_alpha(alpha_id, "
                                "confirm=True) again to keep polling; it will not submit twice."}
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

    async def _wait_for_slot(self, alpha_id: str, kind: str, deadline: Any,
                             front: bool = False) -> Optional[Dict[str, Any]]:
        """Queue for a CorrelationGate slot. None = admitted; otherwise the PENDING
        result to return because the time ran out while still queued. `deadline`
        is a time or a function giving one (a job's deadline moves with cooldowns)."""
        gate = self.correlation_gate
        ticket = gate.enqueue(alpha_id, kind, front=front)
        try:
            while not gate.admit(ticket, kind):
                remaining = (deadline() if callable(deadline) else deadline) - time.monotonic()
                if remaining <= 0:
                    return gate.describe(alpha_id, kind)
                if gate.background and gate.abandoned(alpha_id, kind):
                    return {"status": "CANCELLED", "abandoned": True,
                            "note": (f"Dropped from the queue: nobody asked for it for "
                                     f"{int(gate.abandon_seconds // 60)} minutes. Ask again to queue it anew.")}
                await asyncio.sleep(min(1.0, remaining))
            return None
        finally:
            gate.leave(ticket)

    async def _poll_correlation(self, alpha_id: str, kind: str, max_wait: float,
                                priority: str = "normal") -> Dict[str, Any]:
        """Correlation of one alpha through the CorrelationGate: a remembered
        result, else the job already going on for it, else a new job. The caller
        waits max_wait for the job; the job itself goes on in the background."""
        gate = self.correlation_gate
        alpha_id = str(alpha_id)
        key = (alpha_id, kind)
        hit = gate.cached(key)
        if hit is not None:
            return {**hit, "cached": True}
        gate.asked(alpha_id, kind, priority)
        budget = max(0.0, min(float(max_wait or 0), 300.0))
        task = gate.inflight.get(key)
        if task is None or task.done():
            task = asyncio.create_task(self._correlation_job(alpha_id, kind, budget))
            gate.inflight[key] = task
            task.add_done_callback(
                lambda t, k=key: gate.inflight.pop(k, None) if gate.inflight.get(k) is t else None)
        # During a cooldown the call still waits its wait_seconds: the job goes on as
        # soon as the cooldown ends, and the caller gets the result if it comes in time.
        try:
            # shield: a caller giving up must not cancel the job others wait on
            return dict(await asyncio.wait_for(asyncio.shield(task), timeout=budget + gate.grace_seconds))
        except asyncio.TimeoutError:
            return gate.describe(alpha_id, kind)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise                      # this call itself was cancelled
            return {"status": "CANCELLED",
                    "note": "Taken out of the correlation queue by check_alpha(check=\"cancel\")."}

    async def local_prod_estimate(self, alpha_id: str, wait: float = 8.0) -> Optional[Dict[str, Any]]:
        """ProdMemo's local view of an alpha's prod correlation, for callers who are
        waiting on BRAIN. Best effort: None when ProdMemo cannot answer within `wait`
        (the computation goes on, so asking again later usually finds it)."""
        alpha_id = str(alpha_id)
        hit = self._estimates.get(alpha_id)
        if hit and time.monotonic() - hit[0] < 600:
            return hit[1]
        task = self._estimate_tasks.get(alpha_id)
        if task is None or task.done():
            async def run() -> Optional[Dict[str, Any]]:
                full = await prodmemo_client.check(alpha_id)
                row = prodmemo_client.compact_check(full)
                if "prod_est" not in row:
                    return None
                est = full.get("prod_est") or {}
                out = {k: row.get(k) for k in ("prod_est", "prod_est_confidence", "pool", "self",
                                               "prod_lower_bound", "platform_prod") if row.get(k) is not None}
                for k in ("range", "region", "calibration_points"):
                    if est.get(k) is not None:
                        out[k] = est[k]
                out["note"] = ("ESTIMATE ONLY, from this server's local PnL pool (ProdMemo), not BRAIN's "
                               "measurement. Good for a first screen, not for a submit decision."
                               + (" Low confidence: " + est["note"] if est.get("note") else ""))
                self._estimates[alpha_id] = (time.monotonic(), out)
                return out
            task = asyncio.create_task(run())
            self._estimate_tasks[alpha_id] = task
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=wait)
        except Exception:
            return None

    async def measured_prod(self, alpha_id: str) -> Optional[Dict[str, Any]]:
        """The last prod correlation BRAIN measured for this alpha: this server's
        cache first, then ProdMemo's stored platform value. None when neither has one."""
        hit = self.correlation_gate.cached((str(alpha_id), "prod"))
        if hit is not None and hit.get("max") is not None:
            return {"value": hit["max"], "source": "correlations/prod endpoint (this server's cache)"}
        try:
            stored = await asyncio.wait_for(prodmemo_client.platform_corr(str(alpha_id), "prod"), timeout=5)
        except Exception:
            return None
        if stored and stored.get("max") is not None:
            return {"value": stored["max"], "source": "ProdMemo (a platform value measured earlier)",
                    "measured_at": stored.get("updated")}
        return None

    async def cancel_correlations(self, target: str) -> Dict[str, Any]:
        """Take correlation jobs out of the queue: one alpha, "waiting" (everything
        that has not started) or "all". Only this server's queue and polling stop;
        a computation BRAIN already started is not stopped by it."""
        gate = self.correlation_gate
        target = str(target or "").strip()
        if not target:
            raise ValueError('alpha_id is required: an alpha id, "waiting" or "all"')
        computing = gate.computing()
        mode = target.lower() if target.lower() in ("all", "waiting") else None
        picked = [(key, task) for key, task in gate.inflight.items() if not task.done() and (
            mode == "all" or (mode == "waiting" and key[0] not in computing)
            or (mode is None and key[0] == target))]
        cancelled: Dict[str, Dict[str, Any]] = {}
        for (alpha, kind), task in picked:
            row = cancelled.setdefault(alpha, {"alpha_id": alpha, "checks": [],
                                               "was": "computing" if alpha in computing else "waiting"})
            row["checks"].append(kind)
            task.cancel()
        if picked:   # let the jobs leave the queue and give their slots back
            await asyncio.gather(*[task for _, task in picked], return_exceptions=True)
        for alpha in cancelled:
            for key in [k for k in gate._active if k[0] == alpha and k[1] != "check"]:
                gate.release(*key)
        out: Dict[str, Any] = {"cancelled": list(cancelled.values()), "queue": gate.snapshot()}
        if not cancelled:
            out["note"] = ("Nothing to cancel: no correlation job " +
                           ("is waiting." if mode == "waiting" else "is queued or running." if mode == "all"
                            else f"for alpha {target} is queued or running."))
        elif any(r["was"] == "computing" for r in cancelled.values()):
            out["note"] = ("This server stopped polling. A computation BRAIN already started goes on "
                           "on its side; its result is simply not collected.")
        return out

    async def _correlation_job(self, alpha_id: str, kind: str, budget: float) -> Dict[str, Any]:
        """Queue, poll, and (in background mode) go back to the end of the queue
        when the slot is used up, until BRAIN answers or the job's time is over."""
        gate = self.correlation_gate
        started, cooled = time.monotonic(), gate._cooldown_total

        def deadline() -> float:
            if not gate.background:
                return started + budget
            # a job does not age while everything is paused
            return started + gate.queue_seconds + (gate._cooldown_total - cooled)

        front = False
        while True:
            queued = await self._wait_for_slot(alpha_id, kind, deadline, front=front)
            if queued is not None:
                return queued
            left = deadline() - time.monotonic()
            slot = min(gate.max_slot_seconds, left) if gate.background and gate.rotate else left
            try:
                result = await self._poll_correlation_now(alpha_id, kind, max(0.0, slot),
                                                          limit=max(gate.max_slot_seconds, gate.queue_seconds))
            except BaseException:
                gate.release(alpha_id, kind)
                raise
            front = bool(result.get("cooling_down"))   # stopped by a cooldown: it keeps its turn
            if result["status"] != "PENDING":
                gate.release(alpha_id, kind, finished=result["status"] == "DONE")
                gate.answered()
                if result["status"] == "DONE":
                    gate.remember((alpha_id, kind), result)
                    if result.get("max") is not None and kind in ("prod", "self"):
                        # ProdMemo keeps prod / self: every measured value is a reference point
                        await self._record_platform_corr(alpha_id, kind, result["max"], result.get("min"))
                return result
            if gate.background and deadline() - time.monotonic() > 0:
                gate.release(alpha_id, kind)    # slot used up (or cooldown): back into the queue
                continue
            if gate.background:
                gate.release(alpha_id, kind)    # the job is over: nobody polls this alpha any more
                result["note"] = (f"BRAIN did not answer within {int(gate.queue_seconds // 60)} minutes; "
                                  "the job was dropped. Ask again to queue it anew.")
            elif result.get("busy") or result.get("cooling_down"):
                gate.release(alpha_id, kind)
            else:
                gate.touch(alpha_id, kind)      # BRAIN keeps computing: the slot stays taken
            return result

    async def _poll_correlation_now(self, alpha_id: str, kind: str, max_wait: float,
                                    limit: float = 300.0) -> Dict[str, Any]:
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
        deadline = time.monotonic() + max(0.0, min(float(max_wait or 0), max(limit, 300.0)))
        gate = self.correlation_gate
        polls = 0
        while True:
            if gate.check_stall():
                return gate.describe(str(alpha_id), kind)   # cooldown: nothing is sent
            await gate.pace()
            gate.touch(str(alpha_id), kind, polled=True)
            try:
                resp = await self._request('get', url)
            except requests.RequestException as e:
                resp, net_error = None, str(e)
            busy = resp is None or resp.status_code in (429, 503) or resp.status_code >= 500
            if resp is not None and resp.status_code == 429:
                gate.rate_limited(_retry_after_seconds(resp))
            if resp is not None and resp.status_code >= 400 and not busy:
                return {"status": "ERROR", "http_status": resp.status_code,
                        "error": _http_error_detail(resp)}
            ra = _retry_after_seconds(resp) if resp is not None else 0.0
            text = "" if busy else (resp.text or "").strip()
            if busy and deadline - time.monotonic() <= 0:
                # Throttled (the catalog lists 429/503 for this polling endpoint):
                # report "not ready yet", not a failure.
                return {"status": "PENDING", "retry_after_seconds": max(ra, _PENDING_RETRY_SECONDS),
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
                return {"status": "PENDING", "retry_after_seconds": max(ra, _PENDING_RETRY_SECONDS)}
            await asyncio.sleep(gate.poll_delay(ra, polls, remaining))
            polls += 1

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
                                include_data: bool = False, priority: str = "normal",
                                estimate: bool = True) -> Dict[str, Any]:
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

        polled = await asyncio.gather(*[self._poll_correlation(alpha_id, k, max_wait, priority) for k in kinds])
        checks: Dict[str, Any] = {}
        for kind, r in zip(kinds, polled):
            name = {"prod": "production", "power-pool": "power_pool"}.get(kind, kind)
            entry: Dict[str, Any] = {"status": r["status"]}
            if r["status"] == "DONE":
                mx = r.get("max")
                entry["max_correlation"] = mx
                entry["passes_check"] = (mx < threshold) if mx is not None else None
                top = _correlation_top_rows(r.get("data") or {}, 3)
                if top:
                    entry["top"] = top
                histogram = _correlation_histogram(r.get("data") or {}, threshold)
                if histogram:
                    entry.update(histogram)
                if include_data:
                    entry["correlation_data"] = r.get("data")
                if r.get("cached"):
                    entry["cached"] = True      # measured a short while ago
            elif r["status"] == "CANCELLED":
                entry["note"] = r.get("note")
            elif r["status"] == "PENDING":
                entry["retry_after_seconds"] = r.get("retry_after_seconds")
                entry["note"] = r.get("note") or "Platform is still computing; call again later (not a failure)."
                entry.update({k: r[k] for k in ("queued", "queue_position", "computing_for_seconds",
                                                "polls", "busy", "cooling_down", "resumes_in_seconds")
                              if k in r})
            else:
                entry.update({k: r[k] for k in ("error", "http_status", "body") if k in r})
            checks[name] = entry

        statuses = [c["status"] for c in checks.values()]
        status = next((st for st in ("ERROR", "CANCELLED", "PENDING") if st in statuses), "DONE")
        if status == "PENDING" and "DONE" in statuses:
            status = "PARTIAL"   # some parts are in: they are there to read, the rest follows
        passes = [c.get("passes_check") for c in checks.values()]
        all_passed = all(passes) if status == "DONE" and None not in passes else None
        if status == "PARTIAL" and False in passes:
            all_passed = False   # one measured correlation over the threshold settles it
        out = {"alpha_id": alpha_id, "threshold": threshold, "status": status,
               "all_passed": all_passed, "checks": checks}
        if status in ("PENDING", "PARTIAL"):
            out["queue"] = self.correlation_gate.snapshot()   # who is computed, who waits
        if estimate and (checks.get("production") or {}).get("status") == "PENDING":
            local = await self.local_prod_estimate(alpha_id)
            if local:
                out["local_estimate"] = local
        return out

    async def get_submission_check(self, alpha_id: str, max_wait: float = 60,
                                   priority: str = "normal") -> Dict[str, Any]:
        """Platform-authoritative pre-submission check (GET /alphas/{id}/check).

        Polls the Retry-After protocol within max_wait. PROD_CORRELATION often
        comes back as result=ERROR while the platform is busy; in that case the
        dedicated prod-correlation endpoint is tried as a fallback so the answer
        is not lost. A numeric PROD_CORRELATION is written back into ProdMemo.
        """
        await self.ensure_authenticated()
        url = f"{self.base_url}/alphas/{_seg(alpha_id, 'alpha id')}/check"
        deadline = time.monotonic() + max(0.0, min(float(max_wait or 0), 300.0))
        # /check queues in a lane of its own. A correlation cooldown does not stop
        # it: its IS checks need no correlation, and they are what the caller
        # can act on while BRAIN throttles the correlation endpoints.
        gate = self.correlation_gate
        gate.asked(str(alpha_id), "check", priority)
        queued = await self._wait_for_slot(str(alpha_id), "check", deadline)
        if queued is not None:
            return {"alpha_id": alpha_id, **queued, "queue": gate.snapshot()}
        try:
            data = await self._poll_check(url, alpha_id, deadline)
        finally:
            gate.release(str(alpha_id), "check")
        # The check ends with this call, and so does its place in the lane: holding
        # it would make the check of the next alpha wait for nothing.
        if isinstance(data, dict) and data.get("status") in ("ERROR", "PENDING") and "alpha_id" in data:
            partial = data.pop("partial", None)
            if partial is None:
                if data["status"] == "PENDING":
                    data["queue"] = gate.snapshot()
                return data
            data = partial      # BRAIN still computes, but its IS checks are in: report those
        report = await self._submission_report(alpha_id, data, max_wait)
        if report.get("status") == "PENDING":
            report["queue"] = gate.snapshot()
        else:
            gate.answered()
        return report

    async def _poll_check(self, url: str, alpha_id: str, deadline: float) -> Any:
        """GET /alphas/{id}/check until it settles: the JSON body, or an
        {"alpha_id", "status": ERROR / PENDING, ...} result."""
        gate = self.correlation_gate
        polls = 0
        partial: Any = None
        while True:
            await gate.pace()
            gate.touch(str(alpha_id), "check", polled=True)
            resp = await self._request('get', url)
            if resp.status_code == 429:
                gate.rate_limited(_retry_after_seconds(resp))
            if resp.status_code >= 400:
                out = {"alpha_id": alpha_id, "status": "ERROR", "http_status": resp.status_code,
                       "error": _http_error_detail(resp)}
                if "QUICK" in (resp.text or ""):
                    out["note"] = ("This alpha came from a QUICK simulation, which BRAIN cannot check "
                                   "or submit. Re-run the expression with simulation_mode=\"FULL\" "
                                   "(or omit simulation_mode) and check the new alpha.")
                return out
            ra = _retry_after_seconds(resp)
            text = (resp.text or "").strip()
            if text and "Retry-After" not in resp.headers:
                try:
                    data = resp.json()
                except ValueError:
                    return {"alpha_id": alpha_id, "status": "ERROR", "error": "non-JSON body",
                            "body": text[:300]}
                return data
            if text:     # still running, but with the checks it has so far
                try:
                    body = resp.json()
                    if ((body or {}).get("is") or {}).get("checks"):
                        partial = body
                except ValueError:
                    pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"alpha_id": alpha_id, "status": "PENDING", "retry_after_seconds": max(ra, _PENDING_RETRY_SECONDS),
                        "note": "Platform is still running the checks; call again later.", "partial": partial}
            await asyncio.sleep(gate.poll_delay(ra, polls, remaining))
            polls += 1

    async def _submission_report(self, alpha_id: str, data: Any, max_wait: float) -> Dict[str, Any]:
        """Summarise a finished /check body; fall back to the prod-correlation
        endpoint when /check lost that value."""
        is_ = (data or {}).get("is") or {}
        checks = [c for c in (is_.get("checks") or []) if isinstance(c, dict)]
        rows = [{k: c.get(k) for k in ("name", "result", "value", "limit") if c.get(k) is not None}
                for c in checks]

        def names(result: str) -> List[str]:
            return [c.get("name") for c in checks if c.get("result") == result]

        failed, pending, errored = names("FAIL"), names("PENDING"), names("ERROR")
        # Limits of the account (today's submissions), not properties of the alpha.
        blockers = [n for n in failed if n in _ACCOUNT_CHECKS]
        failed = [n for n in failed if n not in _ACCOUNT_CHECKS]
        # BRAIN sometimes lists the same check twice (MATCHES_THEMES): once is enough.
        rows = [json.loads(r) for r in dict.fromkeys(json.dumps(r, sort_keys=True) for r in
                [{k: c.get(k) for k in ("name", "result", "value", "limit") if c.get(k) is not None}
                 for c in checks])]
        prod = next((c for c in checks if c.get("name") == "PROD_CORRELATION"), None)
        out: Dict[str, Any] = {
            "alpha_id": alpha_id,
            "status": "PENDING" if pending else "DONE",
            "all_passed": (not failed and not pending and not errored) if checks else None,
            "failed": failed, "pending": pending, "errored": errored,
            "checks": rows,
        }
        if blockers:
            out["account_blockers"] = blockers
            out["account_note"] = ("Limits of the account, not of this alpha (e.g. REGULAR_SUBMISSION = "
                                   "today's submissions are used up): it can be submitted once they clear.")
        if any("CORRELATION" in (n or "") for n in pending):
            out["pending_note"] = ("BRAIN computes SELF / PROD_CORRELATION inside this check; they do not "
                                   "show in the correlation queue. Check again later, or "
                                   "check_alpha(check=\"prod\") to follow the prod correlation in the queue.")
        self_corr = is_.get("selfCorrelation")
        if isinstance(self_corr, dict) and self_corr.get("max") is not None:
            out["self_correlation_max"] = self_corr.get("max")

        if errored:
            # A check BRAIN could not run is not a pass: no verdict for the alpha.
            out["all_passed"] = None
        # The checks that need no correlation, decided on their own: usable at once,
        # also while the correlation checks are pending or errored.
        is_checks = [c for c in checks if "CORRELATION" not in (c.get("name") or "")]
        is_checks = [c for c in is_checks if c.get("name") not in _ACCOUNT_CHECKS]
        if is_checks:
            is_failed = [c.get("name") for c in is_checks if c.get("result") == "FAIL"]
            is_open = [c.get("name") for c in is_checks if c.get("result") in ("PENDING", "ERROR")]
            out["is_passed"] = False if is_failed else (None if is_open else True)
        prod_value = _finite(prod.get("value")) if prod else None
        if prod and prod.get("result") in ("PASS", "FAIL") and prod_value is not None:
            out["prod_correlation"] = prod_value
            out["prod_source"] = "submission check (BRAIN's own verdict)"
            await self._record_platform_corr(alpha_id, "prod", prod_value, source="platform_check")
        elif prod is not None and prod.get("result") in ("ERROR", "PENDING"):
            # /check has no prod value (yet): a value BRAIN measured before, else the
            # dedicated endpoint. Reported next to the check, never counted into all_passed.
            known = await self.measured_prod(alpha_id)
            if known is not None:
                out["prod_fallback"] = {"status": "DONE", "max_correlation": known["value"],
                                        "passes_check": known["value"] < PROD_THRESHOLD, **known}
            elif prod.get("result") == "ERROR":
                fallback = await self.check_correlation(alpha_id, "prod", max_wait=min(max_wait, 60),
                                                        estimate=False)
                entry = fallback["checks"].get("production") or {}
                out["prod_fallback"] = {**entry, "source": "correlations/prod endpoint (not the submission check)"}
            fb = out.get("prod_fallback") or {}
            if fb.get("status") == "DONE" and fb.get("max_correlation") is not None:
                out["prod_correlation"] = fb["max_correlation"]
                out["prod_source"] = "fallback: " + fb["source"]
                others_ok = all(c.get("result") in ("PASS", "WARNING") for c in checks
                                if c.get("name") != "PROD_CORRELATION")
                out["all_passed_with_fallback"] = bool(fb.get("passes_check")) and others_ok
            else:
                local = await self.local_prod_estimate(alpha_id)
                if local:
                    out["local_estimate"] = local
            out["note"] = (f"PROD_CORRELATION is {prod.get('result')} in BRAIN's submission check, so all_passed "
                           "is unknown (null). prod_fallback is a prod value from elsewhere (see its source); "
                           "all_passed_with_fallback counts it in. Check again before submitting.")
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
            if name is not None and not name.strip():
                data["name"] = None   # BRAIN refuses "" ("may not be blank"); null clears it

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


def _decoded_codes(value: Any, name: str) -> Tuple[List[str], List[str]]:
    """The expressions of a tool argument, undoing a JSON encoding some clients add
    to a "string or list" argument: '"rank(x)"' (a quoted string, which BRAIN
    reads as a string literal: "found None" for every child) or '["a", "b"]'
    (a list sent as one string). Returns (expressions, notes on what was undone)."""
    notes: List[str] = []

    def unwrap(text: Any, where: str) -> Any:
        if not isinstance(text, str):
            return text
        for _ in range(3):
            stripped = text.strip()
            if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
                try:
                    inner = json.loads(stripped)
                except ValueError:
                    break
                if not isinstance(inner, str):
                    break
                notes.append(f"{where}: removed a layer of JSON quotes around the expression")
                text = inner
                continue
            break
        return text

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except ValueError:
                parsed = None
            if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
                notes.append(f"{name}: a JSON list sent as one string was read as {len(parsed)} expressions")
                value = parsed
    items = _as_list(value, name)
    return [unwrap(x, f"{name}[{i}]" if len(items) > 1 else name) for i, x in enumerate(items)], notes


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
    "ts_backfill": (2, ["k"]), "ts_rank": (2, ["constant"]),
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
    if expr.strip()[:1] in ("'", '"'):
        return ["the expression is a quoted string: BRAIN reads it as a string literal "
                "(\"Expression must have dimensions dates,instruments, found None\")"]
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


# BRAIN's /data-fields/{id} lists at most this many region / delay / universe rows;
# a full list may be cut off, so a region missing from it proves nothing.
_FIELD_DATA_ROWS_CAP = 50

# Arguments that must be a group (industry, sector, bucket(...), densify(...)):
# operator -> positions, 0-based. From BRAIN's operator definitions.
_GROUP_ARGS = {
    "group_rank": (1,), "group_scale": (1,), "group_count": (1,), "group_zscore": (1,),
    "group_std_dev": (1,), "group_sum": (1,), "group_neutralize": (1,), "group_backfill": (1,),
    "group_mean": (2,), "group_extra": (2,), "group_vector_proj": (2,),
    "group_cartesian_product": (0, 1), "densify": (0,),
}
# Operators whose result is a group.
_GROUP_MAKERS = {"bucket", "densify", "group_cartesian_product"}
# Operator categories whose result is a number, never a group.
_NUMERIC_CATEGORIES = {"Arithmetic", "Time Series", "Cross Sectional", "Vector", "Transformational"}
# Infix / unary operators BRAIN counts in operatorCount.
_INFIX_RE = re.compile(r"(?<![=!<>])(?:\*\*|[+*/^]|-|<=|>=|==|!=|&&|\|\||(?<![<>=!])[<>](?!=))")


def _calls(expr: str) -> List[Dict[str, Any]]:
    """Every operator call of an expression (strings and comments blanked out):
    name, where it starts, its argument texts with their offsets, and the name of
    the call it sits in (None at the top level)."""
    text = _strip_strings_and_comments(expr)
    calls: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []      # open calls; plain parentheses are None entries
    frames: List[Optional[Dict[str, Any]]] = []
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*\s*\(|\(|\)|,", text):
        tok = m.group(0)
        if tok == ",":
            call = frames[-1] if frames else None
            if call is not None:
                call["args"].append((call["_arg_start"], m.start()))
                call["_arg_start"] = m.end()
            continue
        if tok == ")":
            if not frames:
                continue
            call = frames.pop()
            if call is not None:
                if call["_arg_start"] < m.start() and text[call["_arg_start"]:m.start()].strip():
                    call["args"].append((call["_arg_start"], m.start()))
                call["end"] = m.end()
                stack.pop()
            continue
        if tok == "(":
            frames.append(None)
            continue
        name = tok.rstrip("( \t\n")
        before = text[:m.start()].rstrip()
        if before.endswith("."):
            frames.append(None)          # a method of an object: not an operator
            continue
        parent = stack[-1]["name"] if stack else None
        call = {"name": name, "start": m.start(), "parent": parent, "args": [], "_arg_start": m.end()}
        calls.append(call)
        stack.append(call)
        frames.append(call)
    for call in calls:
        call["args"] = [(text[a:b].strip(), a, b) for a, b in call["args"]]
        call.pop("_arg_start", None)
    return calls


def _assignments(expr: str) -> Dict[str, str]:
    """name -> right-hand side of `name = ...;` statements."""
    text = _strip_strings_and_comments(expr)
    out = {}
    for stmt in text.split(";"):
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)(.*)$", stmt, re.S)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _estimated_ops(expr: str) -> int:
    """operatorCount as BRAIN counts it, roughly: every operator call plus every
    infix / unary operator (a - b, -x, a > b)."""
    text = _strip_strings_and_comments(expr)
    body = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)", " ", text)       # assignments are not operators
    body = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)[^,)]*", " ", body)  # nor keyword arguments
    calls = len(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", body))
    infix = len(_INFIX_RE.findall(re.sub(r"\d+(\.\d+)?[eE][-+]?\d+", "0", body)))
    return calls + infix


def _semantic_issues(expr: str, operators: Dict[str, str], field_types: Dict[str, str]) -> List[str]:
    """Problems BRAIN refuses the whole batch for that the syntax check cannot see:
    an operator that does not exist, a VECTOR (event) field used without a vec_*
    aggregation, and a non-group where an operator needs a group.
    operators: name -> category; field_types: field -> MATRIX / VECTOR / GROUP."""
    issues: List[str] = []
    calls = _calls(expr)
    assigned = _assignments(expr)
    if operators:
        for name in dict.fromkeys(c["name"] for c in calls):
            if name not in operators:
                close = [o for o in operators if o.startswith(name.split("_")[0] + "_")][:4]
                issues.append(f"unknown operator {name}()" + (f" (did you mean {', '.join(close)}?)" if close else ""))

    def kind_of(arg: str, depth: int = 0) -> Optional[str]:
        """GROUP / NUMBER / None (unknown) for an argument text."""
        arg = arg.strip()
        while arg.startswith("(") and arg.endswith(")"):
            arg = arg[1:-1].strip()
        if re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", arg):
            return "NUMBER"
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", arg):
            if arg in assigned and depth < 3:
                return kind_of(assigned[arg], depth + 1)
            t = field_types.get(arg)
            return None if t is None else ("GROUP" if t == "GROUP" else "NUMBER")
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", arg)
        if m and _balanced_call(arg, m.end() - 1):
            name = m.group(1)
            if name in _GROUP_MAKERS:
                return "GROUP"
            if operators.get(name) in _NUMERIC_CATEGORIES or name.startswith(("ts_", "vec_")):
                return "NUMBER"
            return None
        if re.search(r"[-+*/]", arg):
            return "NUMBER"          # arithmetic of values
        return None

    for call in calls:
        for pos in _GROUP_ARGS.get(call["name"], ()):
            args = call["args"]
            named = next((a for a, _, _ in args if re.match(r"(group|g\d?)\s*=", a)), None)
            arg = named.split("=", 1)[1] if named else (args[pos][0] if pos < len(args) else None)
            if arg is None or "=" in arg.split("(")[0]:
                continue
            if kind_of(arg) == "NUMBER":
                issues.append(f"{call['name']}: argument {pos + 1} {_cut(arg.strip(), 60)!r} must be a group "
                              "(e.g. industry, subindustry, bucket(rank(x), range=\"0,1,0.1\")), not a number")

    # VECTOR (event) fields: BRAIN only takes them inside a vec_* aggregation.
    vector_fields = {f for f, t in field_types.items() if t == "VECTOR"}
    if vector_fields:
        text = _strip_strings_and_comments(expr)
        spans = [(c["start"], c.get("end", len(text)), c["name"]) for c in calls]
        via: Dict[str, str] = {}
        for var, rhs in assigned.items():
            if rhs in vector_fields:
                via[var] = rhs       # a = vector_field; then a is used like the field
        for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text):
            name = m.group(0)
            field = name if name in vector_fields else via.get(name)
            if field is None or text[m.end():].lstrip().startswith(("(", "=")) and not \
                    text[m.end():].lstrip().startswith("=="):
                continue
            inside = [n for a, b, n in spans if a < m.start() < b]
            if not inside:
                continue             # the right-hand side of an assignment: judged where it is used
            if not inside[-1].startswith("vec_"):
                issues.append(f"VECTOR field {field!r}"
                              + (f" (via {name})" if name != field else "")
                              + f" is used in {inside[-1]}() without a vec_* aggregation: BRAIN refuses event "
                              "inputs there. Wrap it, e.g. vec_avg(" + field + ")")
    return list(dict.fromkeys(issues))


def _balanced_call(text: str, open_at: int) -> bool:
    """True when the parenthesis at open_at closes at the very end of text."""
    depth = 0
    for i in range(open_at, len(text)):
        depth += text[i] == "("
        depth -= text[i] == ")"
        if depth == 0:
            return i == len(text) - 1
    return False


def _check_raa_settings(settings: Dict[str, Any], prefix: str) -> None:
    """Platform rules for REGION_AGNOSTIC (violations fail the simulation outright)."""
    for key, fixed in _RAA_FIXED.items():
        value = settings.get(key)
        same = (_finite(value) == fixed if isinstance(fixed, int)
                else str(value).strip().upper() == fixed)
        if not same:
            raise ValueError(f"{prefix}an RAA simulation always runs with {key}={fixed!r}, got {value!r}")
        settings[key] = fixed
    settings["testPeriod"] = None  # the RAA payload carries no testPeriod
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


# Bare words of an expression that are no data fields.
_NOT_FIELDS = {"nan", "inf", "true", "false", "none", "null"}


def _field_candidates(expr: str) -> List[str]:
    """Names in a FASTEXPR expression that must be data fields: not an operator
    (followed by "("), not a keyword argument or an assigned variable (followed by
    "="), not inside a string or comment, not an attribute (after ".")."""
    text = re.sub(r"/\*.*?\*/", " ", expr or "", flags=re.S)
    text = re.sub(r"#[^\n]*", " ", text)
    text = re.sub(r"\"[^\"]*\"|'[^']*'", '""', text)
    assigned, names = set(), []
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text):
        name, rest = m.group(0), text[m.end():].lstrip()
        before = text[:m.start()].rstrip()
        if before.endswith(".") or re.search(r"\d$", text[:m.start()]):
            continue                                 # attribute, or the e of 1e5
        if rest.startswith("(") or rest.startswith("."):
            continue                                 # an operator, or an object such as stats.returns
        if rest.startswith("=") and not rest.startswith("=="):
            assigned.add(name)
            continue
        names.append(name)
    return [n for n in dict.fromkeys(names) if n not in assigned and n.lower() not in _NOT_FIELDS]


def _same_code(a: Optional[str], b: Optional[str]) -> bool:
    """The same expression, whitespace aside."""
    squeeze = lambda t: re.sub(r"\s+", "", t or "")  # noqa: E731
    return squeeze(a) == squeeze(b)


def _epoch(stamp: Any) -> Optional[float]:
    """Seconds since 1970 of an ISO timestamp such as 2026-09-27T02:04:55-04:00."""
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


_MATCH_SETTINGS = ("region", "universe", "delay", "decay", "neutralization", "truncation",
                   "maxTrade", "maxPosition", "pasteurization", "nanHandling")


def _settings_mismatch(sent: Dict[str, Any], used: Dict[str, Any]) -> Dict[str, Any]:
    """Settings BRAIN ran with that differ from what was sent: {key: {sent, brain}}."""
    out = {}
    for k in _MATCH_SETTINGS + ("unitHandling", "language"):
        if k not in sent or k not in used:
            continue
        a, b = sent[k], used[k]
        if isinstance(a, str) and isinstance(b, str):
            same = a.upper() == b.upper()
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            same = abs(float(a) - float(b)) < 1e-9
        else:
            same = a == b
        if not same:
            out[k] = {"sent": a, "brain": b}
    return out


def _match_key(code: Any, settings: Dict[str, Any]) -> Tuple:
    """Expression (whitespace aside) + the settings that make a simulation distinct."""
    return (re.sub(r"\s+", "", code or ""),) + tuple(
        str(settings.get(k)).upper() if isinstance(settings.get(k), str)
        else (float(settings[k]) if isinstance(settings.get(k), (int, float))
              and not isinstance(settings.get(k), bool) else settings.get(k))
        for k in _MATCH_SETTINGS)


def _compact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shorter answer for a finished simulation: settings shared by all rows said
    once, cancelled children as a list of indexes, no URLs nobody needs."""
    state = dict(state)
    state.pop("raw", None)
    rows = state.get("alpha_results")
    if not isinstance(rows, list):
        return state
    kept, cancelled = [], []
    for row in rows:
        row = {k: v for k, v in row.items() if k != "location"}
        if row.get("status") == "CANCELLED" and not row.get("message") and not row.get("error"):
            cancelled.append(row.get("index"))
            continue
        kept.append(row)
    sets = [r.get("set") for r in kept if isinstance(r.get("set"), dict)]
    if len(sets) > 1:
        common = {k: v for k, v in sets[0].items() if all(x.get(k) == v for x in sets[1:])}
        if common:
            state["set"] = common
            for r in kept:
                if isinstance(r.get("set"), dict):
                    rest = {k: v for k, v in r["set"].items() if k not in common}
                    if rest:
                        r["set"] = rest
                    else:
                        r.pop("set")
    state["alpha_results"] = kept
    if cancelled:
        state["cancelled"] = cancelled
    return state


_TSV_COLUMNS = ("index", "id", "ops", "sharpe", "fitness", "turnover", "margin_bps", "sub_sharpe",
                "y2_sharpe", "cluster", "fails", "warns", "set", "expr")


def _tsv(state: Dict[str, Any]) -> Dict[str, Any]:
    """The rows of a finished simulation as one TSV text (header + a line per alpha)."""
    rows = state.get("alpha_results")
    if rows is None and isinstance(state.get("alpha"), dict):
        rows = [{"index": 0, **state["alpha"]}]
    if not isinstance(rows, list):
        return state
    common = state.get("set") or {}

    def cell(row: Dict[str, Any], key: str) -> str:
        value = row.get(key)
        if key == "set":
            value = {**common, **(value or {})}
            return ",".join(f"{k}={v}" for k, v in value.items())
        if key == "id" and value is None:
            return f"{row.get('status', '')}: {row.get('message') or row.get('error') or ''}".strip()
        if isinstance(value, list):
            return ",".join(map(str, value))
        return "" if value is None else str(value).replace("\t", " ").replace("\n", " ")
    lines = ["\t".join(_TSV_COLUMNS)] + ["\t".join(cell(r, k) for k in _TSV_COLUMNS) for r in rows]
    out = {k: v for k, v in state.items() if k not in ("alpha_results", "alpha", "set", "note")}
    out["tsv"] = "\n".join(lines)
    return out


_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _results_path(tag: str) -> str:
    if not _TAG_RE.fullmatch(str(tag or "")):
        raise ValueError("tag must be 1-64 letters, digits, '_', '.' or '-' (starting with a letter or digit)")
    return os.path.join(RESULTS_DIR, f"{tag}.jsonl")


def _match_children(children: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> List[int]:
    """Index of the submitted item each child belongs to. BRAIN lists children in
    the order they were sent; that is checked against expression and settings
    where both are known, and corrected when every child can be told apart."""
    positions = list(range(len(children)))
    if not items or len(items) != len(children):
        return positions

    key_of = _match_key

    wanted = [key_of(i["expr"], i["settings"]) for i in items]
    got = []
    for c in children:
        alpha = c.get("details")
        if alpha is None:
            return positions                     # a failed child has nothing to compare
        got.append(key_of(((alpha.get("regular") or alpha.get("combo") or {}).get("code")),
                          alpha.get("settings") or {}))
    if got == wanted or len(set(wanted)) != len(wanted) or sorted(map(repr, got)) != sorted(map(repr, wanted)):
        return positions
    return [wanted.index(g) for g in got]


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
# Reads BRAIN, writes only the local ProdMemo database (never destructively).
PRODMEMO_BRAIN = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)


# MCP connections drop calls that stay silent for about 45 seconds, so no tool
# waits longer than this, whatever wait_seconds asks for. Nothing is lost by it:
# simulations run on BRAIN and correlation jobs in this server's background queue.
MAX_TOOL_WAIT = max(1.0, float(os.environ.get("WQMCP_MAX_WAIT_SECONDS", "40")))
# Submissions that hit the account's simulation limit wait in this server's
# queue (first in, first out) instead of coming back RATE_LIMITED.
SUBMIT_QUEUE_DEFAULT = os.environ.get("WQMCP_SUBMIT_QUEUE", "1") != "0"
SUBMIT_QUEUE_MAX = int(os.environ.get("WQMCP_SUBMIT_QUEUE_MAX", "50"))
SUBMIT_QUEUE_SECONDS = float(os.environ.get("WQMCP_SUBMIT_QUEUE_SECONDS", "7200"))
SUBMIT_QUEUE_INTERVAL = float(os.environ.get("WQMCP_SUBMIT_QUEUE_INTERVAL", "15"))
SUBMIT_QUEUE_MAX_INTERVAL = float(os.environ.get("WQMCP_SUBMIT_QUEUE_MAX_INTERVAL", "60"))
# A simulation whose progress has not moved for this long is reported as stale.
STALE_MULTI_SECONDS = float(os.environ.get("WQMCP_STALE_MULTI_SECONDS", "1200"))
STALE_SINGLE_SECONDS = float(os.environ.get("WQMCP_STALE_SINGLE_SECONDS", "600"))
# Resubmit (once) a multi-simulation BRAIN failed without saying why.
AUTO_RETRY_GLITCH = os.environ.get("WQMCP_AUTO_RETRY_GLITCH", "1") != "0"
# Results of tagged simulations: results/<tag>.jsonl, followed in the background.
RESULTS_DIR = os.environ.get("WQMCP_RESULTS_DIR",
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
# What the server must not forget over a restart ("" = keep it in memory only).
STATE_FILE = os.environ.get("WQMCP_STATE_FILE",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "server_state.json"))
brain_client.load_state()
WATCH_SECONDS = float(os.environ.get("WQMCP_WATCH_SECONDS", "10800"))
WATCH_INTERVAL = float(os.environ.get("WQMCP_WATCH_INTERVAL", "30"))
# A read-only tool that has not answered by then is given up (BRAIN hanging).
TOOL_DEADLINE = max(MAX_TOOL_WAIT + 10.0, float(os.environ.get("WQMCP_TOOL_DEADLINE_SECONDS", "75")))


_STARTED_AT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _release() -> str:
    """The git revision this server runs: the .release file a deploy writes, else git."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, ".release"), encoding="utf-8") as fh:
            return fh.read().strip() or "unknown"
    except OSError:
        pass
    try:
        import subprocess
        out = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


_RELEASE = _release()


def _server_info() -> Dict[str, Any]:
    """Which release runs since when, and a fingerprint of the tool list: a client
    whose tools_hash differs holds an old tool list and should reconnect."""
    import hashlib
    try:
        tools = mcp._tool_manager.list_tools()
        schema = json.dumps(sorted((t.name, sorted((t.parameters or {}).get("properties", {})))
                                   for t in tools))
        count, digest = len(tools), hashlib.sha1(schema.encode()).hexdigest()[:10]
    except Exception:
        count, digest = None, None
    return {"release": _RELEASE, "started": _STARTED_AT, "tools": count, "tools_hash": digest}


def _session_key() -> int:
    """Which client session this call comes from (0 when unknown)."""
    try:
        return id(mcp.get_context().session)
    except Exception:
        return 0


def _capped(asked: Any) -> Dict[str, Any]:
    """{"wait_capped_to": 40} when the caller asked to wait longer than a call may."""
    try:
        return {"wait_capped_to": int(MAX_TOOL_WAIT)} if float(asked or 0) > MAX_TOOL_WAIT else {}
    except (TypeError, ValueError):
        return {}


def _wait(seconds: Any) -> float:
    """wait_seconds as asked for, capped at MAX_TOOL_WAIT."""
    try:
        return max(0.0, min(float(seconds or 0), MAX_TOOL_WAIT))
    except (TypeError, ValueError):
        return 0.0


# While a call runs, a progress notification goes out this often: a client that
# hears nothing for minutes gives the call up (Claude Code: 300s) and drops the
# connection, although the answer would still have come.
PROGRESS_INTERVAL = float(os.environ.get("WQMCP_PROGRESS_SECONDS", "10"))
# A tool call that takes longer than this is logged as slow.
SLOW_CALL_SECONDS = float(os.environ.get("WQMCP_SLOW_CALL_SECONDS", "50"))
_calls_running: Dict[int, Dict[str, Any]] = {}
_call_numbers = iter(range(1, 1 << 62))


def _args_summary(kwargs: Dict[str, Any]) -> str:
    """A short, log-safe view of a call's arguments."""
    parts = []
    for k, v in kwargs.items():
        if v is None or v == "" or v == [] or k in ("expressions", "combo", "selection", "data"):
            if k in ("expressions", "combo", "selection") and v:
                parts.append(f"{k}=<{len(v) if isinstance(v, list) else 1}>")
            continue
        text = json.dumps(v, ensure_ascii=False, default=str)
        parts.append(f"{k}={text[:60]}")
    return " ".join(parts)[:240]


async def _heartbeat(name: str, started: float) -> None:
    """Progress notifications for a running call (only if the client asked for them)."""
    try:
        ctx = mcp.get_context()
    except Exception:
        return
    while True:
        await asyncio.sleep(PROGRESS_INTERVAL)
        elapsed = time.monotonic() - started
        try:
            await ctx.report_progress(round(elapsed, 1), None, f"{name}: still working ({int(elapsed)}s)")
        except Exception:
            return


def _tool(annotations: ToolAnnotations):
    """Register an MCP tool whose failures come back as {"error": ...}: callers such
    as scripts/prodmemo_daily_sync.py read that shape rather than MCP's isError.
    Read-only tools also get a deadline, so a hanging BRAIN cannot hang the call;
    tools that write are never cut off half way. Every call is logged with its
    duration, and sends progress notifications while it runs."""
    def register(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            _LoopWatchdog.ensure_running()
            brain_client.resume_after_restart()
            number = next(_call_numbers)
            started = time.monotonic()
            _calls_running[number] = {"tool": fn.__name__, "args": _args_summary(kwargs), "started": started}
            beat = asyncio.create_task(_heartbeat(fn.__name__, started))
            outcome = "ok"
            try:
                if annotations.readOnlyHint:
                    result = await asyncio.wait_for(fn(*args, **kwargs), timeout=TOOL_DEADLINE)
                else:
                    result = await fn(*args, **kwargs)
                if isinstance(result, dict) and result.get("error"):
                    outcome = "error"
                return result
            except asyncio.TimeoutError:
                outcome = "deadline"
                return {"status": "UNKNOWN", "timed_out": True, "retry_after_seconds": 15.0,
                        "error": f"no answer from BRAIN within {int(TOOL_DEADLINE)}s; nothing was "
                                 "changed or lost — call again"}
            except asyncio.CancelledError:
                outcome = "cancelled"          # the client went away
                raise
            except Exception as e:
                outcome = "error"
                return {"error": str(e) or repr(e)}
            finally:
                beat.cancel()
                call = _calls_running.pop(number, {})
                took = time.monotonic() - started
                level = logging.WARNING if took >= SLOW_CALL_SECONDS or outcome in ("deadline", "cancelled") \
                    else logging.INFO
                logger.log(level, "call #%d %s %s -> %s in %.1fs", number, fn.__name__,
                           call.get("args", ""), outcome, took)
        return mcp.tool(annotations=annotations)(wrapper)
    return register


class _LoopWatchdog:
    """Notices when the event loop stops running (some code blocking it) and logs
    what the loop thread is doing and which calls are waiting, so a hang can be
    traced from the log afterwards."""
    _started = False
    _beat = 0.0
    _thread_id: Optional[int] = None
    STALL_SECONDS = float(os.environ.get("WQMCP_LOOP_STALL_SECONDS", "2"))

    @classmethod
    def ensure_running(cls) -> None:
        if cls._started:
            return
        cls._started = True
        cls._thread_id = threading.get_ident()
        cls._beat = time.monotonic()
        loop = asyncio.get_running_loop()

        async def tick() -> None:
            while True:
                cls._beat = time.monotonic()
                await asyncio.sleep(0.5)
        loop.create_task(tick())
        threading.Thread(target=cls._watch, name="wqmcp-loop-watchdog", daemon=True).start()

    @classmethod
    def _watch(cls) -> None:
        import traceback
        reported = 0.0
        while True:
            time.sleep(1.0)
            stalled = time.monotonic() - cls._beat
            if stalled < cls.STALL_SECONDS or time.monotonic() - reported < 30:
                continue
            reported = time.monotonic()
            frame = sys._current_frames().get(cls._thread_id)
            stack = "".join(traceback.format_stack(frame)[-12:]) if frame else "(no frame)"
            running = ", ".join(f"#{n} {c['tool']} {int(time.monotonic() - c['started'])}s"
                                for n, c in list(_calls_running.items())[:12])
            logger.warning("event loop blocked for %.1fs; calls waiting: %s\nloop thread:\n%s",
                           stalled, running or "none", stack)


def _write_guard(what: str) -> Optional[Dict[str, Any]]:
    """Error payload when writes are disabled (WQMCP_READ_ONLY=1), else None."""
    if READ_ONLY:
        return {"error": f"{what} is disabled: this server runs with WQMCP_READ_ONLY=1"}
    return None


# --- Account ------------------------------------------------------------------

@mcp.custom_route("/results/{tag}.jsonl", methods=["GET"])
async def results_file(request):
    """The results log of a tag, for curl: ?since=<n> skips the first n lines,
    ?format=tsv gives one TSV line per alpha instead of JSON."""
    from starlette.responses import PlainTextResponse
    tag = request.path_params.get("tag", "")
    try:
        path = _results_path(tag)
    except ValueError as e:
        return PlainTextResponse(str(e) + "\n", status_code=400)
    if not os.path.exists(path):
        return PlainTextResponse("", status_code=200, headers={"X-Lines": "0"})
    try:
        since = max(0, int(request.query_params.get("since", "0") or 0))
    except ValueError:
        return PlainTextResponse("since must be a number of lines\n", status_code=400)

    def read() -> List[str]:
        with open(path, encoding="utf-8") as fh:
            return fh.read().splitlines()
    lines = await asyncio.get_running_loop().run_in_executor(None, read)
    body = lines[since:]
    if request.query_params.get("format") == "tsv":
        rows = []
        for line in body:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        cols = ("ts", "simulation_id", "label") + _TSV_COLUMNS
        out = ["\t".join(cols)]
        for r in rows:
            cells = _tsv({"alpha_results": [r]})["tsv"].split("\n")[1].split("\t")
            out.append("\t".join([str(r.get("ts") or ""), str(r.get("simulation_id") or ""),
                                   str(r.get("label") or "")] + cells))
        text = "\n".join(out) + "\n"
    else:
        text = "".join(line + "\n" for line in body)
    return PlainTextResponse(text, headers={"X-Lines": str(len(lines))})


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
        authenticated, user, token_expiry, the server's read_only / allow_submit
        switches, server (release, started, tools, tools_hash) and correlation_queue
        (who is computed, who waits, throttling). If a parameter described in a
        tool's documentation is missing from your client's tool list, the client
        still holds the tool list of an older release: reconnect the MCP server.
    """
    switches = {"credd_url": CREDD_URL, "read_only": READ_ONLY, "allow_submit": ALLOW_SUBMIT,
                "server": _server_info(), "correlation_queue": brain_client.correlation_gate.snapshot()}
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
    queue: Optional[bool] = None,
    max_ops: Optional[int] = None,
    tag: Optional[str] = None,
    labels: Optional[List[str]] = None,
    resubmit: Optional[str] = None,
    force: bool = False,
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
    QUICK IGNORES max_trade: with max_trade="ON" it returns exactly the numbers of
    max_trade="OFF", while FULL can differ a lot (measured: Sharpe 6.39 -> 2.33).
    Never judge max_trade from a QUICK run; such items come back in "warnings".

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

    validate_expressions: check FASTEXPR expressions before sending, against
    BRAIN's own operator list and field catalogue: unbalanced parentheses; an
    optional operator argument passed positionally (e.g. hump(x, 0.01) instead
    of hump=0.01); an operator BRAIN does not have; a data field it does not
    know or has no data for in the item's region / delay; a VECTOR (event) field
    used without a vec_* aggregation; a number where an operator needs a group
    (group_neutralize(x, rank(y)), densify(close)); an expression wrapped in
    quotes. A problem blocks a multi batch (problems[] names the item); for
    single / concurrent the items with a problem are held back (listed in
    "refused") and the others are sent. force=True sends everything anyway.
    max_ops: hold back (multi: refuse) items with more operators than this,
    counted like BRAIN's operatorCount (every call, infix and unary operator,
    ts_backfill included). "ops_est" gives the count of every item.

    tag: write the results to a log on this server instead of copying them out of
    the answers: every finished row (index, label, id, metrics, fails, warns,
    settings, full expression) is appended to results/<tag>.jsonl, followed in
    the background even when nobody polls. Fetch it without spending tokens:
        curl -s http://<mcp host>:<port>/results/<tag>.jsonl >> results.jsonl
    (?since=<n> skips the first n lines, ?format=tsv gives one TSV line per alpha).
    labels: one free text per item (e.g. the idea behind it), stored as "label".
    resubmit: a simulation id this server created: send exactly the same items
    again (same settings, tag and labels); all other arguments are ignored.

    queue: when the account's simulation slots are full, wait in this server's
    queue instead of coming back RATE_LIMITED (default: on). Requests are sent
    first in, first out as slots free up. The answer is then status "QUEUED"
    with a queue_id ("Q7") and queue_position; get_simulation and
    cancel_simulation take the queue_id like a simulation id. queue=False
    returns RATE_LIMITED as before.

    Returns:
        {"status": "SUBMITTED", "mode", "type", "simulation_id" (single / multi) or
        "simulation_ids" (concurrent), "settings_used" (the base settings; per-item
        overrides are listed in children), "next": the get_simulation call};
        "RATE_LIMITED" with retry_after_seconds when the account's slots are full.
        Concurrent: "PARTIAL" when only some items were accepted, per-item status
        in simulations[] (RATE_LIMITED items can be resubmitted as they are).
    """
    guard = _write_guard("create_simulation")
    if guard:
        return guard
    if resubmit:
        result = await brain_client.resubmit(str(resubmit).strip(), SUBMIT_QUEUE_DEFAULT if queue is None else queue)
        ids = [result[k] for k in ("simulation_id", "queue_id") if result.get(k)][:1]
        if ids:
            result["next"] = f"get_simulation(simulation_ids={ids!r}, wait_seconds=30)"
        return {**result, "resubmitted": str(resubmit).strip()}
    meta = None
    if tag is not None:
        _results_path(tag)                  # validates the tag before anything is sent
        meta = {"tag": tag, "labels": [str(x) if x is not None else "" for x in (labels or [])]}
    elif labels:
        raise ValueError("labels are stored with a tag: pass tag as well")
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
    if is_python and sim_type != "REGULAR":
        raise ValueError(f"language='PYTHON' is for REGULAR alphas; {sim_type} takes FASTEXPR")

    # 1. The code of each simulation.
    if sim_type == "SUPER":
        if expressions:
            raise ValueError("SUPER (SA) simulations use combo + selection, not expressions")
        combos, notes_c = _decoded_codes(combo, "combo")
        selections, notes_s = _decoded_codes(selection, "selection")
        decoded_notes = notes_c + notes_s
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
        exprs, decoded_notes = _decoded_codes(expressions, "expressions")
        codes = [{"regular": e} for e in exprs]
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
                testPeriod=test_period,  # dropped for RAA by _check_raa_settings
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

    warnings = [f"item {i}: QUICK mode ignores maxTrade=ON — the result equals maxTrade=OFF. "
                "Run this item with simulation_mode=\"FULL\" to see what maxTrade does."
                for i, item in enumerate(items)
                if str(item.settings.simulationMode or "").upper() == "QUICK"
                and str(item.settings.maxTrade or "").upper() == "ON"]
    # Pre-check. Every problem is found per distinct expression (or expression +
    # region + delay for the data checks) and then applies to each item using it.
    issues_of: Dict[int, List[str]] = {}

    def add(indexes: List[int], issues: List[str]) -> None:
        for i in indexes:
            bucket = issues_of.setdefault(i, [])
            bucket += [x for x in issues if x not in bucket]

    if validate_expressions and not is_python:
        for row in _lint_problems(codes):
            expr = codes[row["index"]].get("regular")
            add([i for i, c in enumerate(codes) if c.get("regular") == expr], row["issues"])
    if validate_expressions and not is_python and sim_type != "SUPER":
        seen_items: Dict[Tuple[str, str, Any], int] = {}
        for i, item in enumerate(items):
            key = (item.regular or "", str(item.settings.region), item.settings.delay)
            seen_items.setdefault(key, i)
        try:
            unknown, _ = await brain_client.field_problems(
                [(i, key[0], {"region": key[1], "delay": key[2]}) for key, i in seen_items.items()])
        except Exception:   # the pre-check must never stop a submission by failing itself
            unknown = []
        key_of_index = {i: key for key, i in seen_items.items()}
        for row in unknown:
            key = key_of_index[row["index"]]
            add([i for i, item in enumerate(items)
                 if (item.regular or "", str(item.settings.region), item.settings.delay) == key], row["issues"])
    # Operators by BRAIN's operatorCount (calls, infix and unary operators, ts_backfill).
    ops_est = [_estimated_ops(item.regular) if item.regular and not is_python else None for item in items]
    for child, ops in zip(children, ops_est):
        if ops is not None:
            child["ops_est"] = ops
    if max_ops:
        for i, ops in enumerate(ops_est):
            if ops is not None and ops > max_ops:
                add([i], [f"about {ops} operators by BRAIN's count, more than max_ops={max_ops} "
                          "(it counts every call, infix and unary operator, ts_backfill included)"])
    lint = [{"index": i, "expr": _cut(codes[i].get("regular") or codes[i].get("combo") or "", 120),
             "issues": issues_of[i]} for i in sorted(issues_of)]
    use_queue = SUBMIT_QUEUE_DEFAULT if queue is None else bool(queue)
    if lint and mode == "multi" and not force:
        return {"error": "Expression pre-check failed — one failing child cancels the whole multi-simulation",
                "problems": lint, "ops_est": ops_est,
                "note": "Fix the expressions, pass force=True to send anyway, or use mode='concurrent' "
                        "so each alpha runs on its own (items with problems are then held back)."}
    refused: List[Dict[str, Any]] = []
    if lint and mode in ("single", "concurrent") and not force:
        # BRAIN would only turn these into ERROR simulations, each taking a slot.
        refused = [dict(row, status="REFUSED") for row in lint]
        keep = [i for i in range(len(items)) if i not in issues_of]
        if not keep:
            return {"status": "REFUSED", "mode": mode, "type": sim_type, "refused": refused,
                    "ops_est": ops_est,
                    "note": "Nothing was sent: every item failed the pre-check. Fix them, or pass "
                            "force=True to send them anyway."}
        items = [items[i] for i in keep]
        children = [children[i] for i in keep]

    try:
        if meta and meta["labels"] and len(meta["labels"]) != len(items):
            raise ValueError(f"labels has {len(meta['labels'])} entries but there are {len(items)} simulations")
        if mode == "single":
            result = await brain_client.create_simulation(items[0], use_queue, meta)
        elif mode == "multi":
            result = await brain_client.create_multi_simulation(
                [brain_client.simulation_payload(item) for item in items], use_queue, meta)
        else:
            result = await _submit_concurrently(items, children, use_queue, meta)
    except Exception as e:  # BRAIN rejected it: keep the lint warnings next to its reason
        result = {"error": str(e) or repr(e)}
    ids = result.get("simulation_ids") or [
        result[k] for k in ("simulation_id", "queue_id") if result.get(k)][:1]
    result = {**result, "mode": mode, "type": sim_type}
    if mode == "multi" or (mode == "concurrent" and overrides_list):
        result["children"] = children
    try:  # the settings of an item without overrides (all of them when there are none)
        result["settings_used"] = brain_client.simulation_payload(SimulationData(
            type=sim_type, settings=SimulationSettings(**_finalize_settings(sim_type, base, "")),
            **codes[0]))["settings"]
    except ValueError:  # base itself invalid, but every item overrides the bad value
        pass
    if refused:
        result["refused"] = refused
        result["note"] = ((result.get("note") + " ") if result.get("note") else "") + (
            f"{len(refused)} item(s) were not sent: they failed the pre-check (see refused). "
            "force=True sends them anyway.")
    elif lint:
        result["lint_warnings"] = lint
    if any(o is not None for o in ops_est):
        result["ops_est"] = ops_est
    if decoded_notes:
        warnings = list(dict.fromkeys(decoded_notes)) + warnings
    if warnings:
        result["warnings"] = warnings
    if ids:
        result["next"] = f"get_simulation(simulation_ids={ids!r}, wait_seconds=30)"
    return result


async def _submit_concurrently(items: List[SimulationData], children: List[Dict[str, Any]],
                               queue: bool = False, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One POST per simulation, all at once; each item reports its own outcome.
    With the queue, items BRAIN has no slot for wait in it, in the order of the request."""
    keep = ("status", "simulation_id", "progress_url", "retry_after_seconds", "error",
            "queue_id", "queue_position")

    def meta_of(i: int) -> Optional[Dict[str, Any]]:
        if not meta:
            return None
        labels = meta.get("labels") or []
        item = children[i].get("index", i)        # the item's place in the original request
        return {"tag": meta["tag"], "labels": [labels[item]] if item < len(labels) else [], "item": item}

    async def submit(i: int, item: SimulationData) -> Dict[str, Any]:
        try:
            r = await brain_client.create_simulation(item, meta=meta_of(i))
        except Exception as e:
            r = {"status": "ERROR", "error": str(e)}
        return {**children[i], **{k: r[k] for k in keep if k in r}}

    def enqueue(i: int, item: SimulationData) -> Dict[str, Any]:
        payload = brain_client.simulation_payload(item)
        r = brain_client._queue_submission("single", payload, "simulation", item.type, [payload], meta_of(i))
        return {**children[i], **{k: r[k] for k in keep if k in r}}

    if queue and brain_client.submit_queue:          # others wait already: nobody overtakes
        rows = [enqueue(i, item) for i, item in enumerate(items)]
    else:
        rows = list(await asyncio.gather(*(submit(i, item) for i, item in enumerate(items))))
        if queue:
            rows = [enqueue(i, items[i]) if r.get("status") == "RATE_LIMITED" else r
                    for i, r in enumerate(rows)]
    ids = [r["simulation_id"] for r in rows if r.get("status") == "SUBMITTED"]
    queued = [r["queue_id"] for r in rows if r.get("status") == "QUEUED"]
    limited = [r for r in rows if r.get("status") == "RATE_LIMITED"]
    failed = [r for r in rows if r.get("status") not in ("SUBMITTED", "RATE_LIMITED", "QUEUED")]
    if len(ids) == len(rows):
        status = "SUBMITTED"
    elif ids:
        status = "PARTIAL"
    elif queued:
        status = "QUEUED"
    elif limited:
        status = "RATE_LIMITED"  # nothing accepted, but the limited items can simply be retried
    else:
        status = "ERROR"
    out: Dict[str, Any] = {"status": status, "submitted": len(ids), "rate_limited": len(limited),
                           "failed": len(failed), "total": len(rows),
                           "simulation_ids": ids + queued, "simulations": rows}
    notes = []
    if queued:
        out["queued"] = len(queued)
        notes.append(f"{len(queued)} item(s) found no free simulation slot and wait in this server's "
                     "queue (first in, first out); their queue_id works in get_simulation and "
                     "cancel_simulation like a simulation id.")
    if limited:
        out["retry_after_seconds"] = max((r.get("retry_after_seconds") or 0) for r in limited) or 30.0
        notes.append("RATE_LIMITED items hit the account's concurrent-simulation limit (an RAA takes "
                      "4 slots): resubmit just those once running simulations finish.")
    if failed:
        notes.append("Failed items were rejected by BRAIN (see simulations[].error); fix them before "
                      "resubmitting.")
    if status == "ERROR":
        out["error"] = "no simulation was accepted; see simulations[].error"
    if notes:
        out["note"] = " ".join(notes)
    return out


@_tool(READ)
async def get_simulation(simulation_ids: Union[str, List[str], None] = None, wait_seconds: float = 0,
                         compact: bool = True, format: str = "rows", only_pending: bool = False) -> Dict[str, Any]:
    """
    ⏳ Progress / result of simulations — single, multi, RAA, or several at once.

    Args:
        simulation_ids: One or more simulation ids, queue ids ("Q7") or progress_url
            values returned by create_simulation. Omit to list the simulations this
            server created recently and the submissions waiting in its queue.
        wait_seconds: Keep polling up to this long before answering (0 = check once;
            at most 40: longer calls get cut by the connection, the answer then says
            wait_capped_to). One budget for all ids. Call again instead.
        compact: True (default) = one short row per finished alpha: index, id, ops,
            sharpe, fitness, turnover, margin_bps, robust / sub / y2_sharpe, cluster,
            fails, warns, set and expr; keys without a value are left out, settings
            all rows share are given once as "set" next to the rows, cancelled
            children as "cancelled": [indexes]. False = the full alpha object.
        format: "rows" (default) or "tsv": the rows as one TSV text (index, id, ops,
            sharpe, fitness, turnover, margin_bps, sub_sharpe, y2_sharpe, cluster,
            fails, warns, set, expr) — the shortest answer.
        only_pending: with several ids, finished simulations come back as one line
            (id, status, rows) instead of their rows. Without it, a finished
            simulation whose rows this client already got in full comes back as that
            one line too (returned_before: true); ask for it alone to see the rows again.

    Returns:
        For one id: QUEUED with queue_position while it waits for a free slot;
        RUNNING with running_seconds (BRAIN's progress value says nothing) and
        retry_after_seconds, plus stale + stalled_seconds when it has not moved for
        20 minutes (multi) / 10 minutes (single); RETRIED + retried_as when BRAIN
        failed a whole batch without a reason and it was sent once more; COMPLETE with the alpha (multi: alpha_results, one
        row per child with the index of the item it was sent as; reused_alpha +
        submitted_expr when BRAIN answered with an alpha that already existed;
        duplicate_of_index when two items gave the same alpha; missing_children
        when a child is absent — FINISHED_WITH_ERRORS when a child failed, which
        makes BRAIN cancel the rest: errors[] has the index and BRAIN's reason; RAA: parent_alpha_id plus one metric row per region
        child); or the failure status with BRAIN's own error message. For several
        ids: {"status": RUNNING / COMPLETE / FINISHED_WITH_ERRORS, "simulations": [...]}.
    """
    refs = _as_list(simulation_ids, "simulation_ids")
    if not refs:
        recent = list(brain_client.recent_simulations)
        waiting = brain_client.submit_queue_snapshot()
        out = {"recent_simulations": recent,
               "note": ("Pass simulation_ids (or queue ids) to check them." if recent or waiting
                        else "No simulations were created by this server since it started.")}
        if waiting:
            out["submit_queue"] = waiting
        return out
    if len(refs) > 20:
        raise ValueError("at most 20 simulation ids per call")
    fmt = str(format or "rows").strip().lower()
    if fmt not in ("rows", "tsv"):
        raise ValueError(f"format must be 'rows' or 'tsv', got {format!r}")
    wait = _wait(wait_seconds)
    started = time.monotonic()
    deadline = started + wait                  # one budget, shared by every id

    def left() -> float:
        return max(0.0, deadline - time.monotonic())

    def finish(out: Dict[str, Any]) -> Dict[str, Any]:
        if fmt == "tsv":
            if "simulations" in out:
                out["simulations"] = [_tsv(x) for x in out["simulations"]]
            else:
                out = _tsv(out)
        if wait:
            out["waited_seconds"] = int(time.monotonic() - started)
        return {**out, **_capped(wait_seconds)}

    async def queued(ref: str) -> Optional[Dict[str, Any]]:
        """State of a queue id; None = the queue sent it, look at the simulation."""
        while True:
            entry = brain_client.queued_submission(ref)
            if entry is None:
                return {"status": "ERROR", "queue_id": ref,
                        "error": f"no queued submission {ref} (the queue is lost when the server restarts)"}
            if entry["status"] == "SUBMITTED":
                return None
            if entry["status"] != "QUEUED" or time.monotonic() >= deadline:
                return brain_client.queued_state(entry)
            await asyncio.sleep(min(2.0, max(0.1, deadline - time.monotonic())))

    async def state_of(ref: Any) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if re.fullmatch(r"[Qq]\d+", str(ref).strip()):
            ref = str(ref).strip().upper()
            state = await queued(ref)
            if state is not None:
                return {"simulation_id": ref, **state}
            entry = brain_client.queued_submission(ref)
            extra = {"queue_id": ref, "waited_in_queue_seconds": entry.get("waited_seconds")}
            ref = entry["simulation_id"]
        url = _simulation_url(ref)
        state = await brain_client.check_simulation_progress(url, left(), compact)
        return {**extra, **state} if len(refs) == 1 and not extra else \
            {"simulation_id": url.rsplit("/", 1)[-1], **extra, **state}

    if len(refs) == 1:
        try:
            return finish(await state_of(refs[0]))
        except ValueError as e:
            return {"error": f"{e}. Pass the simulation_id (or progress_url) returned by create_simulation"}

    async def one(ref: Any) -> Dict[str, Any]:
        try:
            return await state_of(ref)
        except ValueError as e:
            return {"simulation_id": str(ref), "status": "ERROR", "error": str(e)}
        except Exception as e:  # e.g. a network error after the wait ran out: not a verdict
            return {"simulation_id": str(ref).rsplit("/", 1)[-1], "status": "UNKNOWN", "error": str(e)}

    states = list(await asyncio.gather(*(one(r) for r in dict.fromkeys(refs))))
    delivered = brain_client.delivered_to(_session_key())
    for k, st in enumerate(states):
        finished = st.get("status") in ("COMPLETE", "FINISHED_WITH_ERRORS") and \
            (st.get("alpha_results") is not None or st.get("alpha") is not None)
        if not finished:
            continue
        sid = str(st.get("simulation_id"))
        if only_pending or sid in delivered:
            rows = st.get("alpha_results")
            states[k] = {"simulation_id": sid, "status": st["status"],
                         "rows": len(rows) if isinstance(rows, list) else 1,
                         **({"returned_before": True} if sid in delivered else {})}
        else:
            delivered.add(sid)
    counts = collections.Counter(str(s.get("status")) for s in states)
    if counts["RUNNING"] or counts["UNKNOWN"] or counts["QUEUED"] or counts["RETRIED"]:
        status = "RUNNING"
    elif counts["COMPLETE"] == len(states):
        status = "COMPLETE"
    else:
        status = "FINISHED_WITH_ERRORS"
    notes = list(dict.fromkeys(s.pop("note") for s in states if s.get("note")))
    out: Dict[str, Any] = {"status": status, "counts": dict(counts), "simulations": states}
    if notes:
        out["note"] = " | ".join(notes)   # said once, not on every simulation
    if status == "RUNNING":
        out["retry_after_seconds"] = min((s.get("retry_after_seconds") or 5.0) for s in states
                                         if s.get("status") in ("RUNNING", "UNKNOWN", "QUEUED", "RETRIED"))
    return finish(out)


@_tool(DESTRUCTIVE)
async def cancel_simulation(simulation_id: str) -> Dict[str, Any]:
    """
    🛑 Cancel a queued / running simulation (DELETE) to free an account simulation
    slot, or take a submission out of this server's submit queue.

    Args:
        simulation_id: The simulation id (or progress_url), or the queue_id ("Q7"),
            from create_simulation.
    """
    guard = _write_guard("cancel_simulation")
    if guard:
        return guard
    if re.fullmatch(r"[Qq]\d+", str(simulation_id).strip()):
        done = brain_client.cancel_queued_submission(simulation_id)
        if done is None:
            return {"error": f"no queued submission {simulation_id}"}
        return done
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
        start_date / end_date: creation-date window: "2025-01-01" (whole days, UTC)
            or a full ISO datetime such as "2025-01-01T00:00:00-04:00"
        submission_start_date / submission_end_date: submission-date window (OS),
            same formats
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
    if alpha.get("type") in ("RA_PARENT", "REGION_AGNOSTIC") and not alpha.get("children"):
        out = alpha if full else _alpha_summary(alpha)
        return {**out, "note": "RA parent with no child alphas yet (still simulating, or it failed)."}
    if alpha.get("children"):
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
    📈 THIS alpha's own time series / tables (not the portfolio): pnl, daily-pnl,
    sharpe, turnover, yearly-stats, ...

    Args:
        alpha_id: The alpha id
        recordset: e.g. "pnl", "yearly-stats", "daily-pnl", "sharpe", "turnover";
            omit to list the record sets available for this alpha.
            "yearly-stats" = one row per year of this single alpha: year, pnl,
            bookSize, longCount, shortCount, turnover, sharpe, returns, drawdown,
            margin, fitness, stage.
        wait_seconds: How long to keep polling while BRAIN computes it (at most 40)
        max_rows: Keep only the most recent rows (0 = all)

    Returns:
        BRAIN's {schema, records}; status PENDING means BRAIN is still computing
        it: call again after retry_after_seconds.
    """
    wait_seconds = _wait(wait_seconds)
    if not recordset:
        data = await brain_client.get_record_sets(alpha_id, wait_seconds)
        return {"alpha_id": alpha_id, **(data if isinstance(data, dict) else {"results": data})}
    max_rows = max(0, int(max_rows or 0))
    data = await brain_client.get_record_set_data(alpha_id, recordset, wait_seconds)
    data = data if isinstance(data, dict) else {"result": data}
    records = data.get("records")
    if max_rows and isinstance(records, list) and len(records) > max_rows:
        data = {**data, "records": records[-max_rows:],
                "truncated": f"showing the last {max_rows} of {len(records)} rows"}
    return {"alpha_id": alpha_id, "recordset": recordset, **data}


@_tool(READ)
async def compare_alphas(alpha_ids: List[str], wait_seconds: float = 30, years: int = 4) -> Dict[str, Any]:
    """
    🔀 Correlation between alphas of your own, computed here from their PnL — also
    for alphas that are not submitted, which BRAIN's self correlation leaves out.

    Costs no correlation request: only each alpha's PnL is read. The method is the
    one ProdMemo uses for its local correlations: daily PnL changes over the last
    `years` years, Pearson, on the days both alphas have.

    Args:
        alpha_ids: 2-10 alpha ids
        wait_seconds: How long to wait for BRAIN to produce a PnL it does not have
            yet (an unsubmitted alpha's PnL is made on first request; at most 40)
        years: Length of the window, counted back from the last common day

    Returns:
        pairs (highest correlation first): a, b, correlation, overlap_days;
        max = the highest pair; alphas = per alpha the days of PnL found, or
        status PENDING (call again) / ERROR.
    """
    ids = list(dict.fromkeys(str(a).strip() for a in _as_list(alpha_ids, "alpha_ids") if str(a).strip()))
    if not 2 <= len(ids) <= 10:
        raise ValueError(f"alpha_ids takes 2-10 different alphas, got {len(ids)}")
    years = max(1, min(int(years or 4), 20))
    wait = _wait(wait_seconds)
    gate = asyncio.Semaphore(3)

    async def pnl_of(alpha_id: str) -> Tuple[str, Any]:
        async with gate:
            try:
                data = await brain_client.get_alpha_pnl(alpha_id, wait)
            except Exception as e:
                return alpha_id, {"id": alpha_id, "status": "ERROR", "error": str(e)}
        records = normalize_pnl(data)
        if len(records) < 3:
            return alpha_id, {"id": alpha_id, "status": "PENDING" if not data else "ERROR",
                              **({"note": "BRAIN is still producing the PnL; call again."} if not data
                                 else {"error": f"only {len(records)} days of PnL"})}
        return alpha_id, records

    curves = dict(await asyncio.gather(*(pnl_of(a) for a in ids)))
    alphas = [c if isinstance(c, dict) else
              {"id": a, "status": "OK", "days": len(c), "first_day": c[0][0], "last_day": c[-1][0]}
              for a, c in curves.items()]
    ready = [a for a in ids if isinstance(curves[a], list)]
    pairs = []
    for i, a in enumerate(ready):
        for b in ready[i + 1:]:
            ca, cb = curves[a], curves[b]
            end = min(ca[-1][0], cb[-1][0])
            shorter = [r for r in (ca if ca[-1][0] <= cb[-1][0] else cb) if r[0] <= end]
            start = rolling_window_start(shorter, years)
            dates = sorted({d for d, _ in ca} | {d for d, _ in cb})
            r = pearson_correlation(calculate_forward_filled_returns(ca, dates, start, end),
                                    calculate_forward_filled_returns(cb, dates, start, end))
            row: Dict[str, Any] = {"a": a, "b": b}
            if r is None:
                row.update(correlation=None, note="no common days, or a flat PnL")
            else:
                row.update(correlation=round(r["value"], 4), overlap_days=r["overlapCount"],
                           window=f"{start}..{end}")
            pairs.append(row)
    pairs.sort(key=lambda p: 2.0 if p["correlation"] is None else -p["correlation"])
    out: Dict[str, Any] = {"alphas": alphas, "pairs": pairs}
    measured = [p for p in pairs if p["correlation"] is not None]
    if measured:
        out["max"] = {k: measured[0][k] for k in ("a", "b", "correlation")}
    waiting = [a["id"] for a in alphas if a.get("status") == "PENDING"]
    if waiting:
        out["status"] = "PENDING"
        out["retry_after_seconds"] = 15.0
        out["note"] = f"No PnL yet for {waiting}: BRAIN produces it on first request. Call again."
    else:
        out["status"] = "DONE"
    return out


_CHECK_KINDS = ("submission", "correlation", "prod", "self", "power-pool", "all", "queue", "cancel",
                "cooldown", "resume")


@_tool(READ)
async def check_alpha(alpha_id: str = "", check: str = "submission", wait_seconds: float = 40,
                      threshold: float = 0.7, include_data: bool = False,
                      priority: str = "normal") -> Dict[str, Any]:
    """
    ✅ BRAIN's checks for an alpha: pre-submission checks and / or correlations.

    Correlations are queued: BRAIN rate limits an account that asks for several
    at once (it then answers nothing at all, for every alpha). At most 2 alphas
    are computed at a time and the rest wait in line. A request keeps running in
    the background after this call returned PENDING, so just call again later
    with the same arguments: the answer is then "cached", "computing" (with
    computing_for_seconds) or "queued" (with queue_position). Every PENDING
    answer carries "queue": who is computed, who waits, and whether BRAIN is
    throttling. Order: priority="high" first, then prod-only requests (or any
    that waited 5 minutes), then the rest, first come first served. A queued
    request nobody asks about for 20 minutes is dropped.

    When BRAIN throttles correlations, the server cools down by itself: no
    correlation request for 5 minutes (then 10, 20, 30 if BRAIN still does not
    answer), the queue is kept, then one alpha probes. During a cooldown a
    correlation call answers at once with cooling_down and resumes_in_seconds,
    plus local_estimate (ProdMemo, an estimate only). The submission check is
    NOT paused: its IS checks (sharpe, fitness, sub-universe, 2Y, ...) need no
    correlation and come back with is_passed, even while PROD is pending.

    Args:
        alpha_id: The alpha id (for an RAA use the PARENT id for "submission").
            Not needed for check="queue". For check="cancel": an alpha id,
            "waiting" or "all".
        check:
            "queue" — only show the correlation queue; asks BRAIN nothing.
            "cooldown" — start a cooldown now, for wait_seconds (e.g. 600).
            "resume" — end the cooldown and return to full speed (use it when
                you know BRAIN answers again).
            "cancel" — take correlation jobs out of the queue: alpha_id = one
                alpha (waiting or being computed), "waiting" = every job that has
                not started, "all" = everything. Returns what was cancelled and
                the queue afterwards. It stops this server's polling; BRAIN is
                not told anything. A cancelled alpha can be asked for again.
            "submission" (default) — the same checks as the Submit button: status
                DONE / PENDING, all_passed, is_passed (the checks that need no
                correlation), failed / pending / errored names and each check's
                result / value / limit. When PROD_CORRELATION is ERROR or PENDING,
                prod_fallback gives a prod value measured elsewhere (this server's
                cache, ProdMemo, or the prod endpoint; its source is named) and
                all_passed_with_fallback the verdict that counts it in.
            "prod" / "self" / "power-pool" — that correlation: max_correlation,
                passes_check (max < threshold), the 3 most correlated alphas.
            "correlation" — prod and self together.
            "all" — submission and correlation together.
          Correlation status DONE / PARTIAL (some parts are in: read them, the
          rest follows) / PENDING (still computing, call again — NOT a failure) /
          ERROR. Measured prod / self values are saved into ProdMemo.
        wait_seconds: How long this call waits (at most 40; the answer says so in
            wait_capped_to). Correlations go on in the background afterwards: call
            again to collect them.
        priority: "high" puts this alpha ahead of the "normal" ones in the queue
        threshold: Correlation pass threshold (default 0.7)
        include_data: Also return the raw correlation payloads (large)
    """
    asked_wait, wait_seconds = wait_seconds, _wait(wait_seconds)
    if str(priority or "normal").lower() not in ("normal", "high"):
        raise ValueError(f"priority must be 'normal' or 'high', got {priority!r}")
    kind = str(check or "submission").strip().lower()
    if kind not in _CHECK_KINDS + ("production", "power_pool", "both"):
        raise ValueError(f"check must be one of {list(_CHECK_KINDS)}, got {check!r}")
    if kind == "queue":
        return brain_client.correlation_gate.snapshot()
    if kind == "cancel":
        return await brain_client.cancel_correlations(alpha_id)
    if kind in ("cooldown", "resume"):
        gate = brain_client.correlation_gate
        if kind == "resume":
            gate.resume()
        else:
            gate.start_cooldown(max(30.0, min(float(asked_wait or 0), 7200.0)), reason="started by hand")
        return gate.snapshot()
    if not str(alpha_id or "").strip():
        raise ValueError("alpha_id is required (only check=\"queue\" works without one)")
    capped = _capped(asked_wait)
    if kind == "submission":
        return {**await brain_client.get_submission_check(alpha_id, wait_seconds, priority), **capped}
    if kind != "all":
        ctype = "both" if kind in ("correlation", "both") else kind
        return {**await brain_client.check_correlation(alpha_id, ctype, threshold, wait_seconds, include_data,
                                                       priority), **capped}
    submission, correlation = await asyncio.gather(
        brain_client.get_submission_check(alpha_id, wait_seconds, priority),
        brain_client.check_correlation(alpha_id, "both", threshold, wait_seconds, include_data, priority),
        return_exceptions=True)
    return {"alpha_id": alpha_id,
            "submission": ({"error": str(submission)} if isinstance(submission, BaseException) else submission),
            "correlation": ({"error": str(correlation)} if isinstance(correlation, BaseException) else correlation),
            **capped}


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
    📊 How adding this alpha changes your PORTFOLIO: before / after stats (sharpe,
    fitness, turnover, returns, drawdown, margin), yearly stats and PnL. Every
    number here is the combined portfolio, not the alpha. For the alpha's own
    year-by-year numbers use get_alpha_recordset(alpha_id, "yearly-stats").

    Args:
        alpha_id: The alpha id
        competition_id: Show the competition's before / after view instead
        wait_seconds: How long to keep polling while BRAIN computes it
    """
    return await brain_client.performance_comparison(alpha_id, None, competition_id, _wait(wait_seconds))


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
    detail: bool = False,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    📚 Datasets available for a region / delay / universe, 20 per page.

    Args:
        instrument_type / region / delay / universe: see get_platform_setting_options
        theme: Theme filter
        search: Free-text search
        limit: Page size (1-50, default 20). `count` is the total; `next_offset` is
            the offset of the next page (None on the last one).
        offset: Skip this many datasets (paging)
        detail: False (default) = one short row per dataset (id, name, category,
            coverage, valueScore, alphaCount, fieldCount, userCount,
            pyramidMultiplier, themes, first 160 chars of the description);
            True = BRAIN's full objects (long descriptions, research papers).
        category: Only this category id, e.g. "model", "pv", "analyst", "fundamental",
            "news", "other" (search matches text in any category; this does not).
    """
    return await brain_client.get_datasets(instrument_type, region, delay, universe, theme, search,
                                           limit, offset, detail, category)


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
    compact: bool = True,
) -> Dict[str, Any]:
    """
    🔍 Data fields usable in alpha expressions.

    Args:
        instrument_type / region / delay / universe: see get_platform_setting_options
        theme: Theme filter
        dataset_id: Only fields of this dataset
        data_type: "MATRIX", "VECTOR", "GROUP" ... ("" / "ALL" = any)
        search: Search term
        limit: Page size, 1-500 (default 50). BRAIN answers 50 per request; more are
            fetched in consecutive requests. `count` is the total, `next_offset` the
            offset of the next page.
        offset: Skip this many fields (paging)
        compact: True (default) = id, type, coverage, dateCoverage, userCount,
            alphaCount and the first 80 characters of the description; False =
            BRAIN's full rows (dataset, category, region, themes... on every row)
    """
    return await brain_client.get_datafields(instrument_type, region, delay, universe, theme, dataset_id,
                                             data_type, search, limit, offset, compact)


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
                the REGULAR alphas submitted between start_date and end_date
                (YYYY-MM-DD or ISO datetime, both required): diversity_score = S_A * S_P * S_H with N, A, P,
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

_PRODMEMO_SYNC_MODES = ("incremental", "full", "stop", "status")


@_tool(PRODMEMO_BRAIN)
async def prodmemo_sync(mode: str = "incremental") -> Dict[str, Any]:
    """
    Sync submitted alphas and their PnL into the local ProdMemo database.

    Runs in the background and returns immediately — poll prodmemo_sync("status").

    Args:
        mode: "incremental" (default: probes the remote count first and fetches only
            what is missing), "full" (re-walks every alpha), "stop" (cancels a run
            in progress) or "status" (progress / final state of the latest run)
    """
    mode = str(mode or "incremental").strip().lower()
    if mode not in _PRODMEMO_SYNC_MODES:
        raise ValueError(f"mode must be one of {list(_PRODMEMO_SYNC_MODES)}, got {mode!r}")
    if mode == "status":
        return await prodmemo_client.sync_status()
    return await prodmemo_client.start_sync(mode)


@_tool(PRODMEMO_BRAIN)
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

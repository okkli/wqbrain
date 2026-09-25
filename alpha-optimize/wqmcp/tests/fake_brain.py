"""In-process fake of the BRAIN API + credd, modelled on WQ API Catalog 1.15.3.

Only what the tests need: Retry-After polling, Location headers, multi-simulation
children, 401/429 handling, pagination, and request recording (method, path,
query, Accept header, JSON body) so tests can assert what was sent.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

USER_ID = "U123"


def alpha(alpha_id: str, **extra: Any) -> Dict[str, Any]:
    base = {
        "id": alpha_id, "type": "REGULAR", "author": USER_ID, "stage": "IS", "status": "UNSUBMITTED",
        "settings": {"instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 1,
                     "decay": 4, "neutralization": "SUBINDUSTRY", "truncation": 0.08, "language": "FASTEXPR"},
        "regular": {"code": "rank(close)", "description": None, "operatorCount": 1},
        "is": {"sharpe": 1.4, "fitness": 1.1, "turnover": 0.3, "returns": 0.1, "drawdown": 0.05,
               "margin": 0.001, "longCount": 1500, "shortCount": 1490, "pnl": 1000000, "bookSize": 20000000,
               "checks": [{"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25, "value": 1.4},
                          {"name": "SELF_CORRELATION", "result": "PENDING"}]},
        "classifications": [{"id": "DATA_USAGE:SINGLE_DATA_SET", "name": "Single Data Set Alpha"}],
        "pyramids": [{"name": "USA/D1/PRICE_VOLUME"}],
        "tags": [], "favorite": False, "hidden": False, "dateCreated": "2026-01-01T00:00:00Z",
        "researchNotes": "x" * 5000,
    }
    base.update(extra)
    return base


@dataclass
class FakeState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    requests: List[Dict[str, Any]] = field(default_factory=list)
    cookie_version: int = 1
    valid_token: str = "tok-1"
    credd_calls: int = 0
    counters: Dict[str, int] = field(default_factory=dict)
    sim_slots_full: bool = False
    sims: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    next_sim: int = 1
    alphas: List[Dict[str, Any]] = field(default_factory=list)
    submit_fail: bool = False
    submit_get_403: bool = False
    submit_empty_done: bool = False
    submit_post_delay: float = 0.0
    child_429_once: bool = False
    child_429_times: int = 0  # C1 answers 429 this many times (outlasting client retries)
    child_polls_needed: int = 2
    rate_limit_next_get: int = 0
    credd_same_version: bool = False
    credd_delay: float = 0.0
    corr_412: bool = False
    check_ra_zero_once: bool = False
    saved_language: str = "FASTEXPR"

    def tick(self, key: str) -> int:
        with self.lock:
            self.counters[key] = self.counters.get(key, 0) + 1
            return self.counters[key]

    def calls(self, method: str, path_re: str) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["method"] == method and re.fullmatch(path_re, r["path"])]


Response = Tuple[int, Dict[str, str], Any]


class FakeBrain:
    def __init__(self) -> None:
        self.state = FakeState()
        self.reset()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def reset(self) -> None:
        self.state = FakeState()
        self.state.alphas = [alpha(f"A{i:03d}", stage="OS", status="ACTIVE",
                                   dateSubmitted="2026-02-01T00:00:00Z",
                                   pyramids=[{"name": ["USA/D1/PRICE_VOLUME", "USA/D1/ANALYST",
                                                       "CHN/D1/MODEL"][i % 3]}],
                                   classifications=[{"id": "DATA_USAGE:SINGLE_DATA_SET" if i % 2 else "DATA_USAGE:MULTI",
                                                     "name": "x"}])
                             for i in range(250)]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # ------------------------------------------------------------ routing

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence
                pass

            def _handle(self, method: str) -> None:
                parts = urlsplit(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw.decode()
                query = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parts.query, keep_blank_values=True).items()}
                record = {"method": method, "path": parts.path, "query": query, "body": body,
                          "accept": self.headers.get("Accept"), "cookie": self.headers.get("Cookie", "")}
                with fake.state.lock:
                    fake.state.requests.append(record)
                status, headers, payload = fake.dispatch(method, parts.path, query, body, record)
                data = b"" if payload is None else (payload if isinstance(payload, bytes) else json.dumps(payload).encode())
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                if data:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                if data:
                    self.wfile.write(data)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_PATCH(self):
                self._handle("PATCH")

            def do_DELETE(self):
                self._handle("DELETE")

            def do_OPTIONS(self):
                self._handle("OPTIONS")

        return Handler

    def dispatch(self, method: str, path: str, q: Dict[str, Any], body: Any, rec: Dict[str, Any]) -> Response:
        st = self.state
        if path == "/cookies":
            st.credd_calls += 1
            if st.credd_delay:
                time.sleep(st.credd_delay)
            if st.credd_same_version:  # credd in backoff: same stale cookie, same version
                return 200, {"X-Cookie-Version": "v1"}, {"session": "tok-1"}
            return 200, {"X-Cookie-Version": f"v{st.cookie_version}"}, {"session": st.valid_token}
        if f"session={st.valid_token}" not in rec["cookie"]:
            return 401, {}, {"detail": "Incorrect authentication credentials."}
        if method == "GET" and st.rate_limit_next_get > 0 and path != "/authentication":
            st.rate_limit_next_get -= 1
            return 429, {"Retry-After": "1"}, {"detail": "Too many requests"}
        for (m, pattern), fn in ROUTES.items():
            if m == method:
                match = re.fullmatch(pattern, path)
                if match:
                    return fn(self, q, body, *match.groups())
        return 404, {}, {"detail": "Not found."}


ROUTES: Dict[Tuple[str, str], Callable[..., Response]] = {}


def route(method: str, pattern: str):
    def deco(fn):
        ROUTES[(method, pattern)] = fn
        return fn
    return deco


@route("GET", r"/authentication")
def _auth(fb: FakeBrain, q, body) -> Response:
    return 200, {}, {"user": {"id": USER_ID}, "token": {"expiry": 1900000000}, "permissions": ["TUTORIAL"]}


@route("GET", r"/users/U123/settings/simulation")
def _sim_settings(fb, q, body) -> Response:
    return 200, {}, {"instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 1, "decay": 4,
                     "neutralization": "SUBINDUSTRY", "truncation": 0.08, "lookback": 0, "pasteurization": "ON",
                     "unitHandling": "VERIFY", "nanHandling": "OFF", "selectionHandling": "OFF",
                     "selectionLimit": 1000, "maxTrade": "OFF", "maxPosition": "OFF", "language": fb.state.saved_language,
                     "visualization": False, "testPeriod": "P0Y0M0D", "componentActivation": "OFF"}


@route("POST", r"/simulations")
def _create_sim(fb: FakeBrain, q, body) -> Response:
    st = fb.state
    if st.sim_slots_full:
        return 429, {"Retry-After": "30"}, {"detail": "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"}
    items = body if isinstance(body, list) else [body]
    for it in items:
        decay = it.get("settings", {}).get("decay")
        # Like DRF's IntegerField: 4 and 4.0 are fine, 4.5 is not.
        if not isinstance(decay, (int, float)) or float(decay) != int(decay):
            return 400, {}, {"settings": {"decay": ["A valid integer is required."]}}
        if it.get("regular") == "bad(":
            return 400, {}, {"detail": "Invalid expression: unexpected end"}
    sid = f"S{st.next_sim}"
    st.next_sim += 1
    if isinstance(body, list):
        children = []
        for i, it in enumerate(body):
            cid = f"{sid}C{i}"
            st.sims[cid] = {"polls": 0, "alpha": f"A-{cid}", "regular": it.get("regular")}
            children.append(cid)
        st.sims[sid] = {"children": children}
    elif body.get("type") == "REGION_AGNOSTIC":
        st.sims[sid] = {"polls": 0, "raa": True, "alpha": f"RAP{sid}", "type": "REGION_AGNOSTIC"}
    else:
        st.sims[sid] = {"polls": 0, "alpha": f"A-{sid}", "regular": body.get("regular"),
                        "type": body.get("type")}
    return 201, {"Location": f"{fb.url}/simulations/{sid}"}, None


@route("GET", r"/simulations/super-selection")
def _super_sel(fb, q, body) -> Response:
    return 200, {}, {"count": 1, "next": None, "results": [alpha("A900")]}


@route("GET", r"/simulations/([^/]+)")
def _get_sim(fb: FakeBrain, q, body, sid: str) -> Response:
    st = fb.state
    sim = st.sims.get(sid)
    if sim is None:
        return 404, {}, {"detail": "Not found."}
    if sim.get("raa"):
        sim["polls"] += 1
        if sim["polls"] < 2:
            return 200, {"Retry-After": "1"}, {"progress": 0.4, "type": "REGION_AGNOSTIC"}
        return 200, {}, {"id": sid, "type": "REGION_AGNOSTIC", "status": "COMPLETE", "alpha": sim["alpha"],
                         "children": [f"{sid}R{i}" for i in range(4)]}
    if "children" in sim:
        sim["polls"] = sim.get("polls", 0) + 1
        if sim["polls"] < 2:
            return 200, {"Retry-After": "1"}, {"progress": 0.3}
        return 200, {}, {"children": sim["children"], "status": "COMPLETE"}
    if st.child_429_once and "C1" in sid and not sim.get("rl"):
        sim["rl"] = True
        return 429, {"Retry-After": "1"}, {"detail": "slow down"}
    if st.child_429_times and "C1" in sid:
        st.child_429_times -= 1
        return 429, {"Retry-After": "1"}, {"detail": "slow down"}
    sim["polls"] += 1
    if sim["polls"] < st.child_polls_needed:
        return 200, {"Retry-After": "1"}, {"progress": 0.5}
    if sim.get("regular") == "fail()":
        return 200, {}, {"id": sid, "status": "ERROR", "message": "Attempted to use unknown variable \"foo\"",
                         "alpha": None}
    return 200, {}, {"id": sid, "status": "COMPLETE", "alpha": sim["alpha"], "regular": sim.get("regular")}


@route("DELETE", r"/simulations/([^/]+)")
def _del_sim(fb, q, body, sid) -> Response:
    return 200, {}, None


@route("OPTIONS", r"/simulations")
def _sim_options(fb, q, body) -> Response:
    choices = {
        "instrumentType": {"type": "choice", "label": "Instrument type", "choices": [{"value": "EQUITY", "label": "Equity"}]},
        "region": {"type": "choice", "label": "Region",
                   "choices": {"instrumentType": {"EQUITY": [{"value": "USA"}, {"value": "CHN"}]}}},
        "delay": {"type": "choice", "label": "Delay", "choices": {"instrumentType": {"EQUITY": {"region": {
            "USA": [{"value": 1}, {"value": 0}], "CHN": [{"value": 1}]}}}}},
        "universe": {"type": "choice", "label": "Universe", "choices": {"instrumentType": {"EQUITY": {"region": {
            "USA": [{"value": "TOP3000"}, {"value": "TOP500"}], "CHN": [{"value": "TOP2000U"}]}}}}},
        "neutralization": {"type": "choice", "label": "Neutralization", "choices": {"instrumentType": {"EQUITY": {"region": {
            "USA": [{"value": "NONE"}, {"value": "SUBINDUSTRY"}], "CHN": [{"value": "MARKET"}]}}}}},
    }
    return 200, {}, {"actions": {"POST": {"settings": {"type": "nested object", "children": choices}}}}


@route("GET", r"/users/self/alphas/summary")
def _alpha_summary(fb, q, body) -> Response:
    return 200, {}, {"unsubmitted": 12, "active": 3, "decommissioned": 1}


@route("GET", r"/users/self/alphas")
def _list_alphas(fb: FakeBrain, q, body) -> Response:
    items = fb.state.alphas
    if q.get("stage"):
        items = [a for a in items if a["stage"] == q["stage"]]
    limit = int(q.get("limit", 10))
    offset = int(q.get("offset", 0))
    page = items[offset:offset + limit]
    nxt = "next" if offset + limit < len(items) else None
    return 200, {}, {"count": len(items), "next": nxt, "previous": None, "results": page}


@route("GET", r"/alphas/([^/]+)")
def _get_alpha(fb, q, body, aid) -> Response:
    if aid == "MISSING":
        return 404, {}, {"detail": "Not found."}
    child = re.fullmatch(r"RAP.+C(USA|EUR|ASI|GLB)", aid)
    if child:
        return 200, {}, alpha(aid, type="RA_CHILD", settings={"region": child.group(1), "universe": "TOP2000"})
    if aid.startswith("RAP"):
        return 200, {}, alpha(aid, type="RA_PARENT", children=[f"{aid}C{r}" for r in ("USA", "EUR", "ASI", "GLB")])
    return 200, {}, alpha(aid)


@route("PATCH", r"/alphas/([^/]+)")
def _patch_alpha(fb, q, body, aid) -> Response:
    return 200, {}, alpha(aid, **{k: v for k, v in body.items() if k in ("name", "category", "tags")})


@route("PATCH", r"/alphas")
def _bulk_patch(fb, q, body) -> Response:
    return 200, {}, [alpha(b["id"]) for b in body]


@route("GET", r"/alphas/([^/]+)/check")
def _check(fb: FakeBrain, q, body, aid) -> Response:
    n = fb.state.tick(f"check:{aid}")
    if fb.state.check_ra_zero_once and n == 1:
        return 200, {"Retry-After": "0"}, None  # present-but-zero still means "running"
    if n < 2:
        return 200, {"Retry-After": "1"}, None
    return 200, {}, {"is": {"checks": [{"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25, "value": 1.4},
                                       {"name": "SELF_CORRELATION", "result": "FAIL", "limit": 0.7, "value": 0.82}],
                            "selfCorrelation": {"max": 0.82, "min": 0.1}}}


@route("POST", r"/alphas/([^/]+)/submit")
def _submit(fb: FakeBrain, q, body, aid) -> Response:
    fb.state.tick(f"submit-post:{aid}")
    if fb.state.submit_post_delay:
        time.sleep(fb.state.submit_post_delay)
    return 201, {"Retry-After": "1"}, None


@route("GET", r"/alphas/([^/]+)/submit")
def _submit_poll(fb: FakeBrain, q, body, aid) -> Response:
    if fb.state.tick(f"submit-get:{aid}") < 2:
        return 200, {"Retry-After": "1"}, None
    if fb.state.submit_get_403:
        return 403, {}, {"is": {"checks": [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL",
                                            "limit": 0.5, "value": 0.3}]}}
    if fb.state.submit_empty_done:
        return 200, {}, None
    result = "FAIL" if fb.state.submit_fail else "PASS"
    return 200, {}, {"is": {"checks": [{"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25, "value": 1.4},
                                       {"name": "PROD_CORRELATION", "result": result, "limit": 0.7, "value": 0.5}]}}


@route("GET", r"/alphas/([^/]+)/correlations/(self|prod|power-pool)")
def _corr(fb: FakeBrain, q, body, aid, kind) -> Response:
    if fb.state.corr_412 and kind == "power-pool":
        return 412, {}, {"detail": "Power pool correlation is not available for this alpha."}
    if fb.state.tick(f"corr:{aid}:{kind}") < 2:
        return 200, {"Retry-After": "1"}, None
    return 200, {}, {"schema": {"name": "correlation", "properties": [{"name": "id"}, {"name": "correlation"}]},
                     "records": [["X1", 0.2], ["X2", 0.65], ["X3", None], ["X4", 0.4]], "min": 0.2, "max": 0.65}


@route("GET", r"/alphas/([^/]+)/recordsets")
def _recordsets(fb, q, body, aid) -> Response:
    return 200, {}, {"count": 2, "results": [{"name": "pnl", "title": "PnL"}, {"name": "yearly-stats", "title": "Y"}]}


@route("GET", r"/alphas/([^/]+)/recordsets/([^/]+)")
def _recordset(fb: FakeBrain, q, body, aid, name) -> Response:
    if fb.state.tick(f"rs:{aid}:{name}") < 2:
        return 200, {"Retry-After": "1"}, None
    rows = [[f"2020-01-{(i % 28) + 1:02d}", i * 10, i * 9] for i in range(500)]
    return 200, {}, {"schema": {"name": name, "properties": [{"name": "date"}, {"name": "pnl"}, {"name": "equal-weight-pnl"}]},
                     "records": rows}


@route("GET", r"/users/self/alphas/([^/]+)/before-and-after-performance")
def _perf(fb, q, body, aid) -> Response:
    return 200, {}, {"stats": {"before": {"sharpe": 1.5}, "after": {"sharpe": 1.6}},
                     "yearlyStats": {"before": {"schema": {"properties": [{"name": "year"}]}, "records": [["2025"]]},
                                     "after": {"schema": {"properties": [{"name": "year"}]}, "records": [["2025"]]}}}


@route("GET", r"/data-sets/search")
def _ds_search(fb, q, body) -> Response:
    return 200, {}, {"datasets": [{"id": "fundamental6", "name": "F6", "researchPapers": ["p"] * 20}],
                     "fields": [{"id": "assets", "dataset": {"id": "fundamental6", "name": "F6"}}]}


@route("GET", r"/data-sets")
def _datasets(fb, q, body) -> Response:
    return 200, {}, {"count": 45, "results": [{"id": f"ds{i}", "name": f"DS {i}", "description": "d" * 1000,
                                               "category": {"id": "fundamental", "name": "Fundamental"},
                                               "researchPapers": [{"title": "t"}] * 10} for i in range(int(q.get("limit", 20)))]}


@route("GET", r"/data-fields/([^/]+)")
def _field(fb, q, body, fid) -> Response:
    return 200, {}, {"id": fid, "type": "MATRIX", "data": [{"region": "USA", "coverage": 0.9}]}


@route("GET", r"/data-fields")
def _fields(fb, q, body) -> Response:
    for k in ("instrumentType", "region", "delay", "universe"):
        if k not in q:
            return 400, {}, {"detail": f"{k} is required"}
    return 200, {}, {"count": 120, "results": [{"id": f"f{i}", "type": "MATRIX", "dataset": {"id": "ds1"}}
                                               for i in range(int(q.get("limit", 20)))]}


@route("GET", r"/operators")
def _operators(fb, q, body) -> Response:
    return 200, {}, [{"name": "ts_rank", "category": "Time Series", "scope": ["REGULAR"], "definition": "ts_rank(x, d)",
                      "description": "Rank over time"},
                     {"name": "rank", "category": "Cross Sectional", "scope": ["REGULAR", "COMBO"],
                      "definition": "rank(x)", "description": "Cross-sectional rank"}]


@route("GET", r"/users/self/activities")
def _activities(fb, q, body) -> Response:
    return 200, {}, {"results": [{"name": "base-payment"}, {"name": "simulations"}]}


@route("GET", r"/users/self/activities/(diversity|pyramid-alphas|pyramid-multipliers|base-payment|other-payment|referrals|simulations|submissions)")
def _activity(fb, q, body, kind) -> Response:
    if kind == "pyramid-multipliers":
        return 200, {}, {"pyramids": [{"category": {"id": c, "name": c}, "region": "USA", "delay": 1, "multiplier": 1.2}
                                      for c in ("PRICE_VOLUME", "ANALYST", "MODEL", "NEWS")]}
    if kind in ("base-payment", "simulations"):
        return 200, {}, {"type": "DAILY", "records": {"schema": {"properties": [{"name": "date"}, {"name": "value"}]},
                                                      "records": [[f"d{i}", i] for i in range(100)]}}
    return 200, {}, {"kind": kind, "query": q}


@route("GET", r"/users/([^/]+)/profile")
def _profile(fb, q, body, uid) -> Response:
    return 200, {}, {"id": uid, "address": {"country": "XX"}, "geniusLevel": "GOLD"}


@route("GET", r"/users/self")
def _self_user(fb, q, body) -> Response:
    return 200, {}, {"id": USER_ID, "email": "me@example.com", "telephone": "+1"}


@route("GET", r"/users/U123/consultant")
def _consultant(fb, q, body) -> Response:
    return 200, {}, {"leaderboard": {"valueFactor": 0.8, "weightFactor": 1.1}}


@route("GET", r"/users/U123/consultant/summary")
def _consultant_summary(fb, q, body) -> Response:
    return 200, {}, {"performance": {"currentLevel": "GOLD"}}


@route("GET", r"/users/self/messages/summary")
def _msg_summary(fb, q, body) -> Response:
    return 200, {}, {"unread": 2}


@route("GET", r"/consultant/boards/([^/]+)")
def _board(fb, q, body, board) -> Response:
    return 200, {}, {"count": 1, "next": None, "results": [{"user": q.get("user"), "board": board}]}


@route("GET", r"/users/self/messages")
def _messages(fb, q, body) -> Response:
    img = '<p>Hi</p><img src="data:image/png;base64,' + "A" * 2000 + '"/>'
    return 200, {}, {"count": 30, "next": "n", "results": [{"id": "M1", "type": "NOTIFICATION", "title": "t",
                                                             "description": img, "read": False}]}


@route("GET", r"/competitions")
def _competitions(fb, q, body) -> Response:
    return 200, {}, {"count": 1, "next": None, "results": [{"id": "GAC2026", "name": "GAC", "status": "ACCEPTED",
                                                            "faq": "long" * 100}]}


@route("GET", r"/competitions/([^/]+)")
def _competition(fb, q, body, cid) -> Response:
    return 200, {}, {"id": cid, "name": "GAC"}


@route("GET", r"/users/self/competitions")
def _my_competitions(fb, q, body) -> Response:
    return 200, {}, {"count": 0, "next": None, "results": []}


@route("GET", r"/events")
def _events(fb, q, body) -> Response:
    return 200, {}, {"count": 1, "next": None, "results": [{"id": "E1", "title": "Webinar", "type": "ONLINE"}]}


@route("GET", r"/tutorials")
def _tutorials(fb, q, body) -> Response:
    return 200, {}, {"count": 1, "next": None, "results": [{"id": "T1", "title": "Intro", "category": "Start",
                                                            "pages": [{"id": "P1", "title": "Page 1"}]}]}


@route("GET", r"/tutorial-pages/([^/]+)")
def _tutorial_page(fb, q, body, pid) -> Response:
    return 200, {}, {"id": pid, "title": "Page", "content": [
        {"type": "HEADING", "value": {"level": "H2", "content": "Intro"}},
        {"type": "TEXT", "value": "Hello"}, {"type": "EQUATION", "value": "rank(close)"}]}

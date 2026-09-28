"""Client-side tests for platform_functions: GET retry, correlation polling,
expression lint and compact rows (tool-level behaviour lives in test_platform_tools).

HTTP is faked at the session level (brain_client.session), so the real
_request / polling code runs; asyncio.sleep is patched out to keep it fast.
"""

import asyncio
import json
import os
import sys

import pytest
import requests

pytest.importorskip("mcp")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # forum tools need Playwright, which these tests never touch
    import forum_functions  # noqa: F401
except ImportError:
    import types
    sys.modules["forum_functions"] = types.SimpleNamespace(forum_client=None)

import platform_functions as pf  # noqa: E402


def resp(status=200, body=None, headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    r.headers.update(headers or {})
    r.url = "https://api.worldquantbrain.com/x"
    return r


class FakeSession:
    """Routes by URL substring to a queue of responses (last one repeats)."""

    def __init__(self, routes):
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls = []

    def _next(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs.get("json")))
        for key, queue in self.routes.items():
            if key in url:
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url, **kw):
        return self._next("get", url, **kw)

    def post(self, url, **kw):
        return self._next("post", url, **kw)


@pytest.fixture
def client(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(pf.asyncio, "sleep", fake_sleep)
    written = []

    async def fake_record(alpha_id, kind, max_v, min_v=None, source="platform"):
        written.append((alpha_id, kind, max_v, source))

    monkeypatch.setattr(pf.prodmemo_client, "record_platform_corr", fake_record)
    c = pf.brain_client
    c._cooldown_until = 0.0
    c.correlation_gate.reset()
    c.sleeps = sleeps
    c.written = written
    yield c
    c.session = None


def install(client, routes):
    client.session = FakeSession(routes)
    return client.session


ALPHA = {
    "id": "A1", "settings": {"universe": "TOP500", "decay": 4, "neutralization": "INDUSTRY",
                             "truncation": 0.08, "maxTrade": "OFF", "region": "IND"},
    "regular": {"code": "rank(x)", "operatorCount": 1},
    "is": {"sharpe": 1.5, "fitness": 1.1, "turnover": 0.2, "margin": 0.0012,
           "checks": [{"name": "LOW_ROBUST_UNIVERSE_SHARPE", "result": "PASS", "value": 0.9},
                      {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "value": 0.3},
                      {"name": "LOW_2Y_SHARPE", "result": "PASS", "value": 1.2},
                      {"name": "CLUSTER_TEST", "result": "WARNING", "value": 0.4}]},
}


# --- pure helpers ----------------------------------------------------------

def test_lint_flags_positional_optional_args_and_parens():
    assert pf._lint_expression("ts_backfill(x, lookback=250)") == []
    assert pf._lint_expression("ts_rank(x, 20)") == []
    assert "lookback=" in pf._lint_expression("ts_backfill(x, 250)")[0]
    assert "hump=" in pf._lint_expression("rank(hump(x, 0.01))")[0]
    assert pf._lint_expression("rank(x") == ["unbalanced parentheses"]
    assert pf._lint_expression('bucket(rank(x), range="0,1,0.1")') == []
    assert pf._lint_expression("my_rank(x, 3)") == []


def test_compact_alpha_row():
    row = pf._compact_alpha_row(ALPHA)
    assert row["id"] == "A1" and row["ops"] == 1 and row["margin_bps"] == 12.0
    assert (row["robust_sharpe"], row["sub_sharpe"], row["y2_sharpe"], row["cluster"]) == (0.9, 0.3, 1.2, 0.4)
    assert row["fails"] == ["SUB_UNIVERSE"]
    assert row["set"] == {"universe": "TOP500", "decay": 4, "neutralization": "INDUSTRY",
                          "truncation": 0.08, "maxTrade": "OFF"}
    assert len(json.dumps(row)) < 500


def test_correlation_stats_prefers_top_level_max():
    assert pf._correlation_stats({"max": 0.61, "min": -0.2, "schema": {"max": 0.9}}) == {"max": 0.61, "min": -0.2}
    assert pf._correlation_stats({"schema": {"max": 0.5}})["max"] == 0.5
    data = {"schema": {"properties": [{"name": "id"}, {"name": "correlation"}]},
            "records": [["a", 0.3], ["b", 0.7]]}
    assert pf._correlation_stats(data)["max"] == 0.7
    assert pf._correlation_top_rows(data, 1) == [{"id": "b", "correlation": 0.7}]


# --- _request retry ----------------------------------------------------------

@pytest.mark.asyncio
async def test_get_retries_429_honouring_retry_after(client):
    s = install(client, {"/alphas/A1": [resp(429, headers={"Retry-After": "3"}), resp(200, ALPHA)]})
    out = await client.get_alpha_details("A1")
    assert out["id"] == "A1"
    assert len(s.calls) == 2
    assert 3.0 in client.sleeps
    assert client._cooldown_until > 0



# --- correlation states ------------------------------------------------------


@pytest.mark.asyncio
async def test_check_correlation_polls_until_done_within_budget(client):
    install(client, {"/correlations/prod": [resp(200, b"", {"Retry-After": "1"}),
                                            resp(200, {"max": 0.55})]})
    out = await client.check_correlation("A1", "prod", max_wait=30)
    assert out["status"] == "DONE" and out["all_passed"] is True


@pytest.mark.asyncio
async def test_prod_correlation_contract_for_prodmemo(client):
    install(client, {"/correlations/prod": [resp(200, b"", {"Retry-After": "1"})]})
    assert await client.get_production_correlation("A1", max_wait=0) == {}
    install(client, {"/correlations/prod": [resp(403, {"detail": "no"})]})
    with pytest.raises(Exception):
        await client.get_production_correlation("A1", max_wait=0)






# --- correlation gate: queue, held slots, cache, pacing ---------------------------

COMPUTING = lambda: resp(200, b"", {"Retry-After": "1"})  # noqa: E731


@pytest.mark.asyncio
async def test_gate_queues_beyond_max_alphas_and_holds_the_slot(client):
    gate = client.correlation_gate
    gate.max_alphas = 1
    s = install(client, {"/alphas/A1/correlations": [COMPUTING()], "/alphas/B2/correlations": [COMPUTING()]})
    first = await client.check_correlation("A1", "prod", max_wait=0)
    assert first["checks"]["production"]["status"] == "PENDING"
    assert "queued" not in first["checks"]["production"]

    # BRAIN is still computing A1, so B2 has to wait and sends nothing
    sent = len(s.calls)
    second = await client.check_correlation("B2", "both", max_wait=0)
    for entry in second["checks"].values():
        assert entry["status"] == "PENDING" and entry["queued"] is True
    assert second["status"] == "PENDING" and len(s.calls) == sent

    # asking again for A1 (either correlation) continues it instead of queueing
    again = await client.check_correlation("A1", "both", max_wait=0)
    assert not any(e.get("queued") for e in again["checks"].values())

    # once A1 is finished its slot goes to B2
    s.routes["/alphas/A1/correlations"] = [resp(200, {"max": 0.4})]
    done = await client.check_correlation("A1", "both", max_wait=0)
    assert done["status"] == "DONE"
    third = await client.check_correlation("B2", "prod", max_wait=0)
    assert "queued" not in third["checks"]["production"]


@pytest.mark.asyncio
async def test_gate_frees_a_slot_nobody_polls_anymore(client):
    gate = client.correlation_gate
    gate.max_alphas, gate.hold_seconds = 1, 0.0
    install(client, {"/correlations/prod": [COMPUTING()]})
    await client.check_correlation("A1", "prod", max_wait=0)
    out = await client.check_correlation("B2", "prod", max_wait=0)
    assert "queued" not in out["checks"]["production"]


@pytest.mark.asyncio
async def test_gate_gives_up_a_slot_brain_never_finishes(client, monkeypatch):
    gate = client.correlation_gate
    gate.max_alphas, gate.max_slot_seconds, gate.rotate = 1, 600.0, True
    install(client, {"/correlations/prod": [COMPUTING()]})
    clock = [1000.0]
    monkeypatch.setattr(pf.time, "monotonic", lambda: clock[0])
    for _ in range(10):                      # A1 is asked for every minute, for 9 minutes
        await client.check_correlation("A1", "prod", max_wait=0)
        clock[0] += 60
    clock[0] -= 60
    out = await client.check_correlation("B2", "prod", max_wait=0)
    assert out["checks"]["production"]["queued"] is True
    clock[0] += 61                           # 601s after A1 got its slot
    out = await client.check_correlation("B2", "prod", max_wait=0)
    assert "queued" not in out["checks"]["production"]
    # A1 now waits for its turn like everybody else
    out = await client.check_correlation("A1", "prod", max_wait=0)
    assert out["checks"]["production"]["queued"] is True


@pytest.mark.asyncio
async def test_gate_is_first_come_first_served(client):
    gate = client.correlation_gate
    gate.max_alphas = 1
    gate.touch("A1", "prod")
    b, c = gate.enqueue("B2"), gate.enqueue("C3")
    assert not gate.admit(b, "prod") and not gate.admit(c, "prod")
    gate.release("A1", "prod")
    assert not gate.admit(c, "prod")      # B2 asked first
    assert gate.position(c) == 2 and gate.admit(b, "prod") and gate.position(c) == 1


@pytest.mark.asyncio
async def test_finished_correlation_is_cached_and_overlapping_polls_are_shared(client):
    s = install(client, {"/correlations/prod": [COMPUTING(), resp(200, {"max": 0.61})]})
    a, b = await asyncio.gather(client.check_correlation("A1", "prod", max_wait=30),
                                client.check_correlation("A1", "prod", max_wait=30))
    assert a["checks"]["production"]["max_correlation"] == b["checks"]["production"]["max_correlation"] == 0.61
    assert len(s.calls) == 2              # one shared poll: computing, then done
    again = await client.check_correlation("A1", "prod", max_wait=0)
    assert again["checks"]["production"]["cached"] is True and len(s.calls) == 2

    client.correlation_gate.cache_seconds = 0.0
    await client.check_correlation("A1", "prod", max_wait=0)
    assert len(s.calls) == 3


@pytest.mark.asyncio
async def test_gate_spaces_requests(client):
    gate = client.correlation_gate
    gate.min_interval = 0.5
    install(client, {"/correlations/": [resp(200, {"max": 0.2})]})
    await asyncio.gather(*[client.check_correlation(a, "both", max_wait=0) for a in ("A1", "B2")])
    paced = [w for w in client.sleeps if 0 < w <= 1.5]
    assert len(paced) == 3 and max(paced) > 1.0   # 4 requests, 0.5s apart


@pytest.mark.asyncio
async def test_blank_name_is_sent_as_null(client):
    calls = []

    class Patch(FakeSession):
        def patch(self, url, **kw):
            calls.append(kw.get("json"))
            return resp(200, {"id": "A1"})

    client.session = Patch({})
    await client.set_alpha_properties("A1", name="", tags=[])
    await client.set_alpha_properties("A1", name="momentum v2")
    assert calls == [{"name": None, "tags": []}, {"name": "momentum v2"}]


# --- background queue: keep waiting after the call returned, show the queue --------

@pytest.fixture
def bg_client(monkeypatch):
    """Background mode, with sleeps that only yield so jobs move on quickly."""
    real_sleep = asyncio.sleep
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        await real_sleep(0.001)

    monkeypatch.setattr(pf.asyncio, "sleep", fake_sleep)
    written = []

    async def fake_record(alpha_id, kind, max_v, min_v=None, source="platform"):
        written.append((alpha_id, kind, max_v))

    monkeypatch.setattr(pf.prodmemo_client, "record_platform_corr", fake_record)
    c = pf.brain_client
    c._cooldown_until = 0.0
    gate = c.correlation_gate
    gate.reset()
    gate.background, gate.grace_seconds, gate.min_interval = True, 0.0, 0.0
    c.sleeps, c.written, c.real_sleep = sleeps, written, real_sleep
    yield c
    for task in list(gate.inflight.values()):
        task.cancel()
    gate.reset()
    c.session = None


async def until(client, predicate, seconds=3.0):
    end = pf.time.monotonic() + seconds
    while not predicate():
        assert pf.time.monotonic() < end, "timed out"
        await client.real_sleep(0.01)


@pytest.mark.asyncio
async def test_background_job_finishes_after_the_call_returned(bg_client):
    client, gate = bg_client, bg_client.correlation_gate
    s = install(client, {"/correlations/prod": [COMPUTING() for _ in range(6)] + [resp(200, {"max": 0.66, "min": -0.1})]})
    first = await client.check_correlation("A1", "prod", max_wait=0)
    entry = first["checks"]["production"]
    assert entry["status"] == "PENDING" and "background" in entry["note"]
    assert first["queue"]["computing"][0]["alpha_id"] == "A1" or first["queue"]["waiting"]

    await until(client, lambda: gate.cached(("A1", "prod")) is not None)
    polls = len(s.calls)
    assert polls == 7 and client.written == [("A1", "prod", 0.66)]   # saved without anybody waiting
    later = await client.check_correlation("A1", "prod", max_wait=0)
    assert later["status"] == "DONE" and later["checks"]["production"]["max_correlation"] == 0.66
    assert later["checks"]["production"]["cached"] is True and len(s.calls) == polls
    assert "queue" not in later


@pytest.mark.asyncio
async def test_queue_is_visible_and_worked_through_in_order(bg_client):
    client, gate = bg_client, bg_client.correlation_gate
    gate.max_alphas = 1
    release = {"A1": False}

    class Session(FakeSession):
        def get(self, url, **kw):
            self.calls.append(("get", url, None))
            alpha = url.split("/alphas/")[1].split("/")[0]
            if alpha == "A1" and not release["A1"]:
                return COMPUTING()
            return resp(200, {"max": {"A1": 0.3, "B2": 0.5, "C3": 0.8}[alpha]})

    client.session = s = Session({})
    outs = [await client.check_correlation(a, "prod", max_wait=0) for a in ("A1", "B2", "C3")]
    assert [o["checks"]["production"].get("queue_position") for o in outs] == [None, 1, 2]
    queue = await pf.check_alpha(check="queue")
    assert [r["alpha_id"] for r in queue["computing"]] == ["A1"]
    assert [(r["position"], r["alpha_id"], r["checks"]) for r in queue["waiting"]] == \
        [(1, "B2", ["prod"]), (2, "C3", ["prod"])]
    assert queue["computing"][0]["polls"] >= 1 and queue["throttled"] is False
    assert not [c for c in s.calls if "/alphas/B2" in c[1] or "/alphas/C3" in c[1]]   # nothing sent for them yet

    release["A1"] = True
    await until(client, lambda: all(gate.cached((a, "prod")) for a in ("A1", "B2", "C3")))
    order = list(dict.fromkeys(c[1].split("/alphas/")[1].split("/")[0] for c in s.calls))
    assert order == ["A1", "B2", "C3"]
    done = await client.check_correlation("C3", "prod", max_wait=0)
    assert done["checks"]["production"]["passes_check"] is False
    assert (await pf.check_alpha(check="queue"))["computing"] == []


@pytest.mark.asyncio
async def test_used_up_slot_goes_to_the_back_of_the_queue(bg_client):
    client, gate = bg_client, bg_client.correlation_gate
    gate.max_alphas, gate.max_slot_seconds, gate.queue_seconds, gate.rotate = 1, 0.05, 0.6, True
    s = install(client, {"/correlations/prod": [COMPUTING()]})   # BRAIN never answers
    await client.check_correlation("A1", "prod", max_wait=0)
    await client.check_correlation("B2", "prod", max_wait=0)
    await until(client, lambda: not gate.inflight, seconds=5)
    polled = [c[1].split("/alphas/")[1].split("/")[0] for c in s.calls]
    turns = [a for a, b in zip(polled, [None] + polled) if a != b]
    assert turns[:4] == ["A1", "B2", "A1", "B2"]                 # they take turns
    assert gate.snapshot()["computing"] == [] and gate.cached(("A1", "prod")) is None


def test_no_answer_for_a_while_is_treated_as_rate_limiting(monkeypatch):
    gate = pf.CorrelationGate()
    assert gate.cooldown_seconds == 0        # conftest: slow mode only
    clock = [5000.0]
    monkeypatch.setattr(pf.time, "monotonic", lambda: clock[0])
    gate.touch("A1", "prod", polled=True)
    assert not gate.throttled() and gate.limit() == 2
    assert gate.poll_delay(1.0, 3, 300) == pf._poll_delay(1.0, 3, 300)
    for _ in range(4):                       # polled for four minutes, never an answer
        clock[0] += 60
        gate.touch("A1", "prod", polled=True)
    snap = gate.snapshot()
    assert gate.throttled() and gate.limit() == 1 and snap["throttled"] and snap["max_at_a_time"] == 1
    assert "240s" in snap["note"] and snap["computing"][0] == \
        {"alpha_id": "A1", "checks": ["prod"], "for_seconds": 240, "polls": 5}
    assert gate.poll_delay(1.0, 3, 300) == 60.0 and gate.poll_delay(1.0, 3, 20) == 20.0
    assert not gate.admit(gate.enqueue("B2", "prod"), "prod")

    gate.answered()                          # BRAIN answers again: back to normal
    assert not gate.throttled() and gate.limit() == 2
    assert gate.snapshot()["last_result_seconds_ago"] == 0

    gate.rate_limited(30)                    # an explicit 429 (cooldowns are off here)
    assert gate.throttled() and gate.cooling() == 0
    clock[0] += 61
    gate.touch("A1", "prod")
    assert not gate.throttled()


@pytest.mark.asyncio
async def test_check_alpha_needs_an_alpha_except_for_the_queue(bg_client):
    assert set(await pf.check_alpha(check="queue")) >= {"throttled", "computing", "waiting", "max_at_a_time"}
    out = await pf.check_alpha(check="prod")
    assert "alpha_id is required" in str(out.get("error"))


@pytest.mark.asyncio
async def test_cancel_one_waiting_all(bg_client):
    client, gate = bg_client, bg_client.correlation_gate
    gate.max_alphas = 1
    s = install(client, {"/correlations/": [COMPUTING()]})       # BRAIN never answers
    for alpha in ("A1", "B2", "C3", "D4"):
        await client.check_correlation(alpha, "both", max_wait=0)
    assert [r["alpha_id"] for r in gate.snapshot()["waiting"]] == ["B2", "C3", "D4"]

    out = await pf.check_alpha(alpha_id="C3", check="cancel")     # one alpha that waits
    assert out["cancelled"] == [{"alpha_id": "C3", "checks": ["prod", "self"], "was": "waiting"}]
    assert [(r["position"], r["alpha_id"]) for r in out["queue"]["waiting"]] == [(1, "B2"), (2, "D4")]
    assert "note" not in out

    out = await pf.check_alpha(alpha_id="A1", check="cancel")     # the one being computed
    assert out["cancelled"][0]["was"] == "computing" and "BRAIN already started" in out["note"]
    await until(client, lambda: [r["alpha_id"] for r in gate.snapshot()["computing"]] == ["B2"])
    polled_a1 = len([c for c in s.calls if "/alphas/A1/" in c[1]])
    assert gate.snapshot()["waiting"][0]["alpha_id"] == "D4"      # B2 moved up and started

    out = await pf.check_alpha(alpha_id="waiting", check="cancel")
    assert [r["alpha_id"] for r in out["cancelled"]] == ["D4"]
    assert [r["alpha_id"] for r in out["queue"]["computing"]] == ["B2"] and out["queue"]["waiting"] == []

    out = await pf.check_alpha(alpha_id="all", check="cancel")
    assert [r["alpha_id"] for r in out["cancelled"]] == ["B2"]
    assert out["queue"]["computing"] == [] and out["queue"]["waiting"] == [] and not gate.inflight
    sent = len(s.calls)
    await client.real_sleep(0.05)
    assert len(s.calls) == sent                                   # nothing is polled any more
    assert len([c for c in s.calls if "/alphas/A1/" in c[1]]) == polled_a1

    out = await pf.check_alpha(alpha_id="all", check="cancel")
    assert out["cancelled"] == [] and "Nothing to cancel" in out["note"]
    assert "alpha_id is required" in str((await pf.check_alpha(check="cancel")).get("error"))

    again = await client.check_correlation("C3", "prod", max_wait=0)   # can be asked for again
    assert again["checks"]["production"]["status"] == "PENDING"


@pytest.mark.asyncio
async def test_a_caller_waiting_for_a_cancelled_job_is_told_so(bg_client):
    client, gate = bg_client, bg_client.correlation_gate
    gate.grace_seconds = 0.0
    install(client, {"/correlations/prod": [COMPUTING()]})
    waiting = asyncio.ensure_future(client.check_correlation("A1", "prod", max_wait=30))
    await until(client, lambda: gate.snapshot()["computing"])
    await pf.check_alpha(alpha_id="A1", check="cancel")
    out = await asyncio.wait_for(waiting, timeout=2)
    assert out["status"] == "CANCELLED" and out["checks"]["production"]["status"] == "CANCELLED"
    assert out["all_passed"] is None


# --- cooldown: stop asking a throttled BRAIN, probe, back off further ---------------

def cooling_gate(monkeypatch, clock):
    monkeypatch.setattr(pf.time, "monotonic", lambda: clock[0])
    gate = pf.CorrelationGate()
    gate.cooldown_seconds, gate.cooldown_max, gate.stall_seconds = 300.0, 1800.0, 180.0
    return gate


def test_cooldown_probe_and_longer_cooldowns(monkeypatch):
    clock = [9000.0]
    gate = cooling_gate(monkeypatch, clock)
    gate.touch("A1", "prod", polled=True)
    clock[0] += 170
    assert not gate.check_stall() and gate.cooling() == 0
    clock[0] += 11                                   # 181s without a result
    assert gate.check_stall() and gate.cooling() == 300
    assert gate.limit() == 0 and not gate.admit(gate.enqueue("A1", "self"), "self")   # not even the same alpha
    snap = gate.snapshot()
    assert snap["cooling_down"] and snap["max_at_a_time"] == 0
    assert snap["cooldown"] == {"resumes_in_seconds": 301, "length_seconds": 300, "in_a_row": 1,
                                "reason": "BRAIN returned no result for 181s"}
    told = gate.describe("B2", "prod")
    assert told["cooling_down"] and told["resumes_in_seconds"] == 301 and told["retry_after_seconds"] == 301

    lengths = []
    for _ in range(4):                               # BRAIN stays silent: 300, 600, 1200, 1800, 1800
        clock[0] += gate.cooling() + 1
        assert gate.cooling() == 0 and gate.throttled() and gate.limit() == 1   # probing
        assert "Probing" in gate.snapshot()["note"]
        assert gate.poll_delay(1.0, 0, 900) == 60.0
        gate.touch("A1", "prod", polled=True)        # the probe
        clock[0] += 181
        assert gate.check_stall()
        lengths.append(gate.cooling())
    assert lengths == [600, 1200, 1800, 1800]

    clock[0] += 1801
    gate.touch("A1", "prod", polled=True)
    gate.answered()                                  # the probe got a result
    assert not gate.throttled() and gate.limit() == 2 and gate.snapshot()["cooling_down"] is False
    gate.touch("B2", "prod", polled=True)
    clock[0] += 181
    assert gate.check_stall() and gate.cooling() == 300       # starts from the short one again


def test_a_429_starts_a_cooldown_and_jobs_do_not_age_in_it(monkeypatch):
    clock = [100.0]
    gate = cooling_gate(monkeypatch, clock)
    gate.rate_limited(30)
    assert gate.cooling() == 300 and gate.snapshot()["cooldown"]["reason"] == "BRAIN answered 429"
    gate.rate_limited(900)                           # already cooling: not stacked
    assert gate.cooling() == 300 and gate._cooldown_total == 300
    clock[0] += 301
    gate.rate_limited(900)                           # BRAIN asks for more than our own length
    assert gate.cooling() == 900 and gate._cooldown_total == 1200


def test_front_of_the_queue_after_a_cooldown():
    gate = pf.CorrelationGate()
    gate.enqueue("B2", "prod"); gate.enqueue("C3", "prod")
    gate.enqueue("A1", "prod", front=True); gate.enqueue("A1", "self", front=True)
    assert [t[1:3] for t in gate._waiting] == [("A1", "prod"), ("A1", "self"), ("B2", "prod"), ("C3", "prod")]
    assert gate.position("A1") == 1 and gate.position("C3") == 3


@pytest.mark.asyncio
async def test_nothing_is_sent_during_a_cooldown_and_the_queue_survives(bg_client):
    client, gate = bg_client, bg_client.correlation_gate
    gate.max_alphas, gate.stall_seconds, gate.cooldown_seconds, gate.cooldown_max = 1, 0.1, 0.5, 5.0
    answers = {"on": False}

    class Session(FakeSession):
        def get(self, url, **kw):
            self.calls.append(("get", url, None))
            alpha = url.split("/alphas/")[1].split("/")[0]
            if url.endswith("/check"):
                return resp(200, {"is": {"checks": [{"name": "LOW_SHARPE", "result": "PASS", "value": 2}]}})
            return resp(200, {"max": {"A1": 0.3, "B2": 0.5}.get(alpha, 0.1)}) if answers["on"] else COMPUTING()

    client.session = s = Session({})
    await client.check_correlation("A1", "prod", max_wait=0)
    await client.check_correlation("B2", "prod", max_wait=0)
    await until(client, lambda: gate.cooling() > 0)               # BRAIN silent for 0.1s
    await client.real_sleep(0.02)
    sent = len(s.calls)

    out = await client.check_correlation("B2", "prod", max_wait=30)   # answers at once, asks nothing
    entry = out["checks"]["production"]
    assert entry["cooling_down"] is True and entry["resumes_in_seconds"] >= 1 and entry["queue_position"] == 2
    assert out["queue"]["cooling_down"] and out["queue"]["computing"] == []
    assert [r["alpha_id"] for r in out["queue"]["waiting"]] == ["A1", "B2"]   # A1 keeps its turn
    check = await client.get_submission_check("C3", max_wait=30)
    assert check["status"] == "PENDING" and check["cooling_down"] is True
    await client.real_sleep(0.2)
    assert len(s.calls) == sent and gate.cooling() > 0            # silence during the cooldown

    answers["on"] = True
    await until(client, lambda: gate.cached(("A1", "prod")) and gate.cached(("B2", "prod")))
    order = list(dict.fromkeys(c[1].split("/alphas/")[1].split("/")[0] for c in s.calls[sent:]))
    assert order == ["A1", "B2"]
    snap = gate.snapshot()
    assert not snap["throttled"] and not snap["cooling_down"] and snap["max_at_a_time"] == 1
    assert (await client.get_submission_check("C3", max_wait=5))["status"] == "DONE"


@pytest.mark.asyncio
async def test_cooldown_by_hand_and_resume(bg_client):
    client, gate = bg_client, bg_client.correlation_gate
    s = install(client, {"/correlations/prod": [resp(200, {"max": 0.2})]})
    snap = await pf.check_alpha(check="cooldown", wait_seconds=600)
    assert snap["cooling_down"] and 598 <= snap["cooldown"]["resumes_in_seconds"] <= 601
    assert snap["cooldown"]["reason"] == "started by hand"
    out = await client.check_correlation("A1", "prod", max_wait=0)
    assert out["checks"]["production"]["cooling_down"] is True and s.calls == []

    snap = await pf.check_alpha(check="resume")
    assert snap["cooling_down"] is False and snap["throttled"] is False and snap["max_at_a_time"] == 2
    await until(client, lambda: gate.cached(("A1", "prod")))      # the queued job went on by itself
    assert len(s.calls) == 1


# --- queue order: prod first, a lane for the submission check, estimates, no lost place

def test_prod_only_requests_go_first_and_the_check_has_its_own_lane(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(pf.time, "monotonic", lambda: clock[0])
    gate = pf.CorrelationGate()
    gate.max_alphas, gate.check_max = 1, 1
    gate.touch("RUN", "prod")                          # the only correlation slot is taken
    a_self, a_prod = gate.enqueue("A1", "self"), gate.enqueue("A1", "prod")
    b_prod = gate.enqueue("B2", "prod")
    c_check = gate.enqueue("C3", "check")
    assert [t[1:3] for t in gate._order("corr")] == [("B2", "prod"), ("A1", "self"), ("A1", "prod")]
    assert gate.position("B2") == 1 and gate.position("A1") == 2 and gate.position("C3", "check") == 1
    assert gate.admit(c_check, "check")                # not behind the correlations
    assert not gate.admit(b_prod, "prod")
    gate.release("RUN", "prod")
    assert not gate.admit(a_prod, "prod") and gate.admit(b_prod, "prod")
    snap = gate.snapshot()
    assert [(r["alpha_id"], r["checks"]) for r in snap["computing"]] in (
        [("B2", ["prod"]), ("C3", ["check"])], [("C3", ["check"]), ("B2", ["prod"])])
    assert [(r["position"], r["alpha_id"], r["checks"]) for r in snap["waiting"]] == [(1, "A1", ["self", "prod"])]
    assert a_self in gate._waiting


def test_a_request_keeps_its_slot_and_waiting_ones_get_an_estimate(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(pf.time, "monotonic", lambda: clock[0])
    gate = pf.CorrelationGate()
    gate.max_alphas, gate.max_slot_seconds, gate.hold_seconds = 1, 600.0, 1e9
    assert gate.rotate is False and gate.typical_seconds() is None
    gate.touch("OLD", "prod")
    clock[0] += 60
    gate.release("OLD", "prod", finished=True)         # jobs take a minute
    gate.touch("A1", "prod")
    b, c = gate.enqueue("B2", "prod"), gate.enqueue("C3", "prod")
    clock[0] += 20
    assert gate.estimated_wait("B2") == 40 and gate.estimated_wait("C3") == 100
    told = gate.describe("C3", "prod")
    assert told["queue_position"] == 2 and told["estimated_wait_seconds"] == 100
    assert gate.describe("A1", "prod")["estimated_remaining_seconds"] == 40
    snap = gate.snapshot()
    assert snap["typical_seconds_per_alpha"] == 60 and snap["waiting"][1]["estimated_wait_seconds"] == 100
    clock[0] += 700                                    # longer than max_slot_seconds
    assert gate.computing() == {"A1"} and not gate.admit(b, "prod")   # A1 did not lose its slot
    gate.rotate = True
    assert gate.computing() == set() and gate.admit(b, "prod") and c in gate._waiting


@pytest.mark.asyncio
async def test_a_pending_submission_check_does_not_hold_the_lane(client):
    gate = client.correlation_gate
    gate.check_max = 1
    pending = {"is": {"checks": [{"name": "LOW_SHARPE", "result": "PASS", "value": 2, "limit": 1.25},
                                 {"name": "SELF_CORRELATION", "result": "PENDING"}]}}
    s = install(client, {"/alphas/A1/check": [resp(200, pending)],
                         "/alphas/B2/check": [resp(400, {"detail": "Cannot check submission for QUICK mode alphas"})]})
    first = await client.get_submission_check("A1", max_wait=0)
    assert first["status"] == "PENDING" and first["pending"] == ["SELF_CORRELATION"]
    assert first["queue"]["computing"] == []
    second = await client.get_submission_check("B2", max_wait=0)      # straight away, not queued
    assert second["status"] == "ERROR" and "FULL" in second["note"] and "queued" not in second
    assert len(s.calls) == 2

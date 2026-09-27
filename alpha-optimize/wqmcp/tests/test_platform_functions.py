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
    gate.max_alphas, gate.max_slot_seconds = 1, 600.0
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

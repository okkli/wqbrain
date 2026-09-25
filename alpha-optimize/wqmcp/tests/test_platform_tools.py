"""platform_functions against the fake BRAIN, over a real in-memory MCP session.

The tool surface is the consolidated one (48 -> 30 tools); README.md maps every
old tool to its replacement."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import platform_functions as pf

TOOLS = {
    "brain_status",
    # simulation
    "create_simulation", "get_simulation", "cancel_simulation", "get_platform_setting_options",
    "preview_super_selection",
    # alphas
    "list_alphas", "get_alpha", "get_alpha_recordset", "check_alpha", "submit_alpha", "update_alpha",
    "get_alpha_performance",
    # data
    "get_datasets", "get_datafields", "get_operators",
    # account / community
    "get_activity", "get_leaderboard", "get_competitions", "get_events", "get_messages",
    "get_documentation",
    # forum
    "search_forum_posts", "read_forum_post", "get_glossary_terms",
    # ProdMemo
    "prodmemo_sync", "prodmemo_check", "prodmemo_get", "prodmemo_stats", "prodmemo_manage",
}


@pytest.fixture
def mcp_session(fake, monkeypatch):
    """Factory: the anyio task group must be entered/exited inside the test task."""
    fresh = pf.BrainApiClient()
    monkeypatch.setattr(pf, "brain_client", fresh)
    yield lambda: create_connected_server_and_client_session(pf.mcp._mcp_server)
    fresh._executor.shutdown(wait=False)


@pytest.fixture
def client(fake):
    c = pf.BrainApiClient()
    yield c
    c._executor.shutdown(wait=False)


async def call(session, _tool, **args):
    res = await session.call_tool(_tool, args)
    assert not res.isError, res.content
    text = res.content[0].text if res.content else ""
    return json.loads(text)


def posted(fake):
    return [c["body"] for c in fake.state.calls("POST", "/simulations")]


# ------------------------------------------------------------- contract


async def test_tool_surface(mcp_session):
    async with mcp_session() as s:
        tools = (await s.list_tools()).tools
    assert {t.name for t in tools} == TOOLS
    by_name = {t.name: t for t in tools}
    assert by_name["get_alpha"].annotations.readOnlyHint is True
    assert by_name["submit_alpha"].annotations.destructiveHint is True
    assert by_name["prodmemo_stats"].annotations.openWorldHint is False


def test_server_binds_loopback_by_default():
    assert pf.mcp.settings.host == "127.0.0.1" and pf.mcp.settings.port == 8761


async def test_brain_status(mcp_session, fake):
    async with mcp_session() as s:
        st = await call(s, "brain_status")
        assert st["authenticated"] is True and st["user"]["id"] == "U123"
        assert st["read_only"] is False and st["allow_submit"] is True
        refreshed = await call(s, "brain_status", refresh=True)
        assert refreshed["status"] == "authenticated"
    assert len(fake.state.calls("GET", "/cookies")) == 2  # bootstrap + forced refresh


# ------------------------------------------------------------- security


async def test_arbitrary_urls_and_path_tricks_are_refused(mcp_session, fake):
    async with mcp_session() as s:
        for url in (f"{fake.url}/cookies", "http://127.0.0.1:8762/cookies?x=worldquantbrain.com",
                    "https://evil.example/simulations/S1", f"{fake.url}/simulations/S1/../../cookies"):
            res = await call(s, "get_simulation", simulation_ids=url)
            assert "not a BRAIN simulation URL" in res["error"]
        many = await call(s, "get_simulation", simulation_ids=[f"{fake.url}/cookies", "../x"])
        assert all(r["status"] == "ERROR" for r in many["simulations"])
        bad = await call(s, "get_alpha", alpha_id="../users/self")
        assert "invalid alpha id" in bad["error"]
        bad = await call(s, "update_alpha", alpha_ids="A1/../../users/self", name="x")
        assert "invalid alpha id" in bad["error"]
        bad = await call(s, "cancel_simulation", simulation_id="https://evil.example/simulations/S1")
        assert "not a BRAIN simulation URL" in bad["error"]
    assert not fake.state.calls("GET", "/cookies")[1:]  # only the session bootstrap
    assert not fake.state.calls("GET", "/users/self")
    assert not fake.state.calls("DELETE", ".*")


async def test_write_kill_switches(mcp_session, fake, monkeypatch):
    monkeypatch.setattr(pf, "ALLOW_SUBMIT", False)
    async with mcp_session() as s:
        res = await call(s, "submit_alpha", alpha_id="A1", confirm=True)
        assert "WQMCP_ALLOW_SUBMIT=0" in res["error"]
        dry = await call(s, "submit_alpha", alpha_id="A1", wait_seconds=10)  # dry run still works
        assert dry["dry_run"] is True and dry["status"] == "DONE"
    monkeypatch.setattr(pf, "READ_ONLY", True)
    async with mcp_session() as s:
        for tool, args in (("create_simulation", {"expressions": "rank(close)"}),
                           ("create_simulation", {"expressions": ["rank(a)", "rank(b)"]}),
                           ("create_simulation", {"expressions": "rank(close)", "type": "RAA"}),
                           ("cancel_simulation", {"simulation_id": "S1"}),
                           ("update_alpha", {"alpha_ids": "A1", "name": "x"}),
                           ("submit_alpha", {"alpha_id": "A1", "confirm": True})):
            res = await call(s, tool, **args)
            assert "WQMCP_READ_ONLY=1" in res["error"], tool
    assert not fake.state.calls("POST", r"/simulations|/alphas/.*")
    assert not fake.state.calls("PATCH", ".*") and not fake.state.calls("DELETE", ".*")


# ----------------------------------------------------------- simulations


async def test_single_regular_flow(mcp_session, fake):
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions="rank(close)")
        assert sub["status"] == "SUBMITTED" and sub["mode"] == "single" and sub["type"] == "REGULAR"
        assert sub["settings_used"]["universe"] == "TOP3000" and "selectionLimit" not in sub["settings_used"]
        assert sub["simulation_id"] in sub["next"]
        body = posted(fake)[0]
        assert body["type"] == "REGULAR" and body["regular"] == "rank(close)"
        assert body["settings"]["testPeriod"] == "P0Y0M" and "lookback" not in body["settings"]
        running = await call(s, "get_simulation", simulation_ids=sub["progress_url"])
        assert running["status"] == "RUNNING"
        done = await call(s, "get_simulation", simulation_ids=[sub["simulation_id"]], wait_seconds=10)
        assert done["status"] == "COMPLETE" and done["alpha"]["sharpe"] == 1.4
        recent = await call(s, "get_simulation")
        assert recent["recent_simulations"][0]["simulation_id"] == sub["simulation_id"]


async def test_multi_regular_parent_retry_after_and_busy_child(mcp_session, fake):
    fake.state.child_polls_needed = 1
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions=["rank(a)", "rank(b)", "rank(c)"])
        assert sub["mode"] == "multi" and sub["type"] == "MULTI" and sub["expected_children"] == 3
        fake.state.child_429_times = 3  # outlasts _request's two GET retries
        body = posted(fake)[0]
        assert isinstance(body, list) and len(body) == 3
        first = await call(s, "get_simulation", simulation_ids=sub["simulation_id"])
        assert first["status"] == "RUNNING" and "children" not in first  # parent Retry-After honoured
        assert not fake.state.calls("GET", r"/simulations/S\d+C\d")
        second = await call(s, "get_simulation", simulation_ids=sub["simulation_id"])
        assert second["status"] == "RUNNING" and second["completed_children"] == 2  # busy child != done
        final = await call(s, "get_simulation", simulation_ids=sub["simulation_id"], wait_seconds=10)
        assert final["status"] == "COMPLETE" and len(final["alpha_results"]) == 3


async def test_multi_sweep_overrides_and_lint(mcp_session, fake):
    async with mcp_session() as s:
        bad = await call(s, "create_simulation", expressions=["ts_backfill(x, 250)", "rank(x)"])
        assert "pre-check" in bad["error"] and bad["problems"][0]["index"] == 0
        assert not fake.state.calls("POST", "/simulations")
        ok = await call(s, "create_simulation", expressions="rank(x)",
                        per_alpha_settings=[{"decay": 3}, {"neutralization": "MARKET"},
                                            {"simulation_mode": "QUICK"}])
        assert ok["status"] == "SUBMITTED" and ok["mode"] == "multi"
        assert ok["children"][0] == {"index": 0, "overrides": {"decay": 3}}
        body = posted(fake)[0]
        assert [b["regular"] for b in body] == ["rank(x)"] * 3
        assert body[0]["settings"]["decay"] == 3 and body[1]["settings"]["neutralization"] == "MARKET"
        assert body[2]["settings"]["simulationMode"] == "QUICK" and body[2]["settings"]["visualization"] is False
        unknown = await call(s, "create_simulation", expressions="rank(x)", per_alpha_settings=[{"foo": 1}, {}])
        assert "unsupported keys ['foo']" in unknown["error"]


async def test_single_and_concurrent_lint_only_warns(mcp_session, fake):
    async with mcp_session() as s:
        one = await call(s, "create_simulation", expressions="ts_backfill(x, 250)")
        assert one["status"] == "SUBMITTED" and one["lint_warnings"][0]["issues"]
        conc = await call(s, "create_simulation", expressions=["ts_backfill(x, 250)", "rank(x)"],
                          mode="concurrent")
        assert conc["status"] == "SUBMITTED" and len(conc["lint_warnings"]) == 1
    assert len(posted(fake)) == 3


async def test_python_single_and_multi(mcp_session, fake):
    async with mcp_session() as s:
        missing = await call(s, "create_simulation", expressions="print(1)", language="PYTHON")
        assert "lookback is required" in missing["error"]
        await call(s, "create_simulation", expressions="def alpha(): ...", language="PYTHON", lookback=20)
        multi = await call(s, "create_simulation", expressions=["src_a", "src_b"], language="python",
                           lookback=20)
        assert multi["mode"] == "multi"
    single, batch = posted(fake)
    assert single["settings"]["language"] == "PYTHON" and single["settings"]["lookback"] == 20
    for item in [single, *batch]:
        assert not {"testPeriod", "unitHandling", "nanHandling"} & set(item["settings"])


async def test_super_single_and_concurrent(mcp_session, fake):
    async with mcp_session() as s:
        one = await call(s, "create_simulation", type="SA", combo="combo_a", selection="sel_a",
                         selection_limit=50)
        assert one["type"] == "SUPER" and one["mode"] == "single"
        conc = await call(s, "create_simulation", type="SUPER", combo="combo_a",
                          selection=["sel_a", "sel_b"])
        assert conc["mode"] == "concurrent" and len(conc["simulation_ids"]) == 2
        nope = await call(s, "create_simulation", type="SUPER", combo="c", selection=["a", "b"], mode="multi")
        assert "REGULAR alphas only" in nope["error"]
        half = await call(s, "create_simulation", type="SUPER", combo="c")
        assert "both combo and selection" in half["error"]
        both = await call(s, "create_simulation", type="SUPER", combo="c", selection="s", expressions="x")
        assert "not expressions" in both["error"]
        done = await call(s, "get_simulation", simulation_ids=conc["simulation_ids"], wait_seconds=10)
        assert done["status"] == "COMPLETE" and done["counts"] == {"COMPLETE": 2}
    first, a, b = posted(fake)
    assert first["type"] == "SUPER" and first["combo"] == "combo_a" and first["selection"] == "sel_a"
    assert first["settings"]["selectionLimit"] == 50 and "regular" not in first
    assert {a["selection"], b["selection"]} == {"sel_a", "sel_b"}


async def test_raa_single_concurrent_and_rules(mcp_session, fake):
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions="rank(close)", type="RAA")
        assert sub["type"] == "REGION_AGNOSTIC" and sub["settings_used"]["universe"] == "MEDIUM"
        body = posted(fake)[0]
        assert body["type"] == "REGION_AGNOSTIC" and body["settings"]["region"] == "ALL"
        assert body["settings"]["decay"] == 10 and body["settings"]["neutralization"] == "SLOW_AND_FAST"
        assert "testPeriod" not in body["settings"] and body["settings"]["visualization"] is False
        done = await call(s, "get_simulation", simulation_ids=sub["simulation_id"], wait_seconds=10)
        assert done["status"] == "COMPLETE" and done["type"] == "REGION_AGNOSTIC"
        assert {c["region"] for c in done["children"]} == {"USA", "EUR", "ASI", "GLB"}

        sweep = await call(s, "create_simulation", expressions="rank(x)", type="ra",
                           per_alpha_settings=[{"universe": "large"}, {"universe": "SMALL"}])
        assert sweep["mode"] == "concurrent" and sweep["status"] == "SUBMITTED"
        assert [b["settings"]["universe"] for b in posted(fake)[1:]] == ["LARGE", "SMALL"]

        for args, msg in (({"universe": "TOP3000"}, "RAA universe must be one of"),
                          ({"region": "USA"}, "region='ALL'"),
                          ({"delay": 0}, "delay=1"),
                          ({"max_trade": "ON", "max_position": "ON"}, "cannot both be ON"),
                          ({"language": "PYTHON", "lookback": 5}, "FASTEXPR expression"),
                          ({"mode": "multi", "expressions": ["a", "b"]}, "REGULAR alphas only")):
            res = await call(s, "create_simulation", **{"expressions": "rank(x)", "type": "RAA", **args})
            assert msg in res["error"], (args, res)
        per_item = await call(s, "create_simulation", expressions="rank(x)", type="RAA",
                              per_alpha_settings=[{"universe": "LARGE"}, {"region": "USA"}])
        assert per_item["error"].startswith("item 1: ")
    assert len(posted(fake)) == 3


async def test_concurrent_partial_rate_limit_and_errors(mcp_session, fake):
    fake.state.sim_post_limit = 2
    async with mcp_session() as s:
        res = await call(s, "create_simulation", expressions=["rank(a)", "rank(b)", "rank(c)"],
                         mode="concurrent")
        assert res["status"] == "PARTIAL" and res["submitted"] == 2 and res["total"] == 3
        assert sorted(r["status"] for r in res["simulations"]) == ["RATE_LIMITED", "SUBMITTED", "SUBMITTED"]
        assert res["retry_after_seconds"] == 30
        full = await call(s, "create_simulation", expressions=["rank(a)", "rank(b)"], mode="concurrent")
        assert full["status"] == "RATE_LIMITED" and not full["simulation_ids"]
        single = await call(s, "create_simulation", expressions="rank(a)")
        assert single["status"] == "RATE_LIMITED" and single["retry_after_seconds"] == 30
    fake.reset()
    async with mcp_session() as s:
        bad = await call(s, "create_simulation", expressions=["bad(", "rank(a)"], mode="concurrent",
                         validate_expressions=False)
        assert bad["status"] == "PARTIAL"
        failed = next(r for r in bad["simulations"] if r["status"] == "ERROR")
        assert "Invalid expression" in failed["error"] and failed["index"] == 0
        both_bad = await call(s, "create_simulation", expressions=["bad(", "bad("], mode="concurrent",
                              validate_expressions=False)
        assert both_bad["status"] == "ERROR" and "no simulation was accepted" in both_bad["error"]


async def test_mode_and_type_validation(mcp_session, fake):
    async with mcp_session() as s:
        for args, msg in (({"expressions": ["a", "b"], "mode": "single"}, "takes one simulation"),
                          ({"expressions": "a", "mode": "multi"}, "needs 2-10 alphas"),
                          ({"expressions": [f"rank(x{i})" for i in range(11)]}, "at most 10"),
                          ({"expressions": "a", "mode": "parallel"}, "mode must be one of"),
                          ({"expressions": "a", "type": "WEIRD"}, "type must be one of"),
                          ({"expressions": ["a", " "]}, "non-empty string"),
                          ({"expressions": ["a", "b"], "per_alpha_settings": [{}]}, "one-to-one"),
                          ({}, "expressions is required"),
                          ({"expressions": "a", "combo": "c"}, "are for type SUPER")):
            res = await call(s, "create_simulation", **args)
            assert msg in res["error"], (args, res)
    assert not fake.state.calls("POST", "/simulations")


async def test_rejection_detail_and_failed_simulation_message(mcp_session, fake):
    async with mcp_session() as s:
        rej = await call(s, "create_simulation", expressions="bad(", validate_expressions=False)
        assert "Invalid expression: unexpected end" in rej["error"]
        sub = await call(s, "create_simulation", expressions="fail()")
        done = await call(s, "get_simulation", simulation_ids=sub["simulation_id"], wait_seconds=10)
        assert done["status"] == "ERROR" and "unknown variable" in done["message"]


async def test_get_simulation_many_and_cancel(mcp_session, fake):
    async with mcp_session() as s:
        a = await call(s, "create_simulation", expressions="rank(a)")
        b = await call(s, "create_simulation", expressions="fail()")
        running = await call(s, "get_simulation", simulation_ids=[a["simulation_id"], b["simulation_id"]])
        assert running["status"] == "RUNNING" and running["retry_after_seconds"] >= 1
        mixed = await call(s, "get_simulation", simulation_ids=[a["simulation_id"], b["progress_url"]],
                           wait_seconds=10, compact=False)
        assert mixed["status"] == "FINISHED_WITH_ERRORS"
        states = {x["simulation_id"]: x for x in mixed["simulations"]}
        assert states[a["simulation_id"]]["status"] == "COMPLETE"
        assert states[a["simulation_id"]]["alpha"]["id"].startswith("A-")
        cancelled = await call(s, "cancel_simulation", simulation_id=a["progress_url"])
        assert cancelled == {"simulation_id": a["simulation_id"], "cancelled": True, "http_status": 200}
    assert fake.state.calls("DELETE", f"/simulations/{a['simulation_id']}")


async def test_preview_super_selection(mcp_session, fake):
    async with mcp_session() as s:
        res = await call(s, "preview_super_selection", selection="rank(sharpe)", region="CHN", limit=3)
    q = fake.state.calls("GET", "/simulations/super-selection")[0]["query"]
    assert q["settings.region"] == "CHN" and q["region"] == "CHN" and q["limit"] == "3"
    assert res["results"][0]["id"] == "A900" and "researchNotes" not in json.dumps(res)


# ----------------------------------------------------------------- alphas


async def test_list_alphas_filters_paging_compact(mcp_session, fake):
    async with mcp_session() as s:
        page = await call(s, "list_alphas", stage="OS", status="ACTIVE", alpha_type="REGULAR", limit=100,
                          offset=200)
        assert page["count"] == 250 and len(page["results"]) == 50 and page["next_offset"] is None
        first = await call(s, "list_alphas", stage="OS", limit=100)
        assert first["next_offset"] == 100
        row = first["results"][0]
        assert row["id"] == "A000" and row["sharpe"] == 1.4 and "researchNotes" not in row
        full = await call(s, "list_alphas", stage="", limit=1, compact=False)
        assert "researchNotes" in full["results"][0]
    q = fake.state.calls("GET", "/users/self/alphas")
    assert q[0]["query"]["status"] == "ACTIVE" and q[0]["query"]["type"] == "REGULAR"
    assert "stage" not in q[2]["query"]


async def test_get_alpha_summary_full_and_raa(mcp_session, fake):
    async with mcp_session() as s:
        summary = await call(s, "get_alpha", alpha_id="A1")
        assert summary["regular"] == "rank(close)" and summary["settings"]["decay"] == 4
        assert summary["is"]["checks"][0]["name"] == "LOW_SHARPE" and "researchNotes" not in summary
        full = await call(s, "get_alpha", alpha_id="A1", full=True)
        assert "researchNotes" in full
        raa = await call(s, "get_alpha", alpha_id="RAPX")
        assert raa["type"] == "REGION_AGNOSTIC" and len(raa["children"]) == 4 and "parent" not in raa
    assert len(fake.state.calls("GET", "/alphas/RAPX")) == 1  # the parent is not fetched twice


async def test_recordsets(mcp_session, fake):
    async with mcp_session() as s:
        res = await call(s, "get_alpha_recordset", alpha_id="A1", recordset="sharpe")
        assert len(res["records"]) == 500  # waited out the Retry-After instead of returning {}
        tail = await call(s, "get_alpha_recordset", alpha_id="A1", recordset="pnl", max_rows=10)
        assert len(tail["records"]) == 10 and "last 10 of 500" in tail["truncated"]
        listed = await call(s, "get_alpha_recordset", alpha_id="A1")
        assert [r["name"] for r in listed["results"]] == ["pnl", "yearly-stats"]
        pending = await call(s, "get_alpha_recordset", alpha_id="A2", recordset="turnover", wait_seconds=0)
        assert pending["status"] == "PENDING"


async def test_check_alpha_kinds(mcp_session, fake):
    async with mcp_session() as s:
        sub = await call(s, "check_alpha", alpha_id="A1", wait_seconds=10)
        assert sub["status"] == "DONE" and "checks" in sub
        prod = await call(s, "check_alpha", alpha_id="A1", check="prod", wait_seconds=10)
        assert set(prod["checks"]) == {"production"} and prod["checks"]["production"]["status"] == "DONE"
        pool = await call(s, "check_alpha", alpha_id="A1", check="power-pool", wait_seconds=10)
        assert set(pool["checks"]) == {"power_pool"}
        both = await call(s, "check_alpha", alpha_id="A1", check="correlation", wait_seconds=10)
        assert set(both["checks"]) == {"production", "self"}
        everything = await call(s, "check_alpha", alpha_id="A1", check="all", wait_seconds=10)
        assert everything["submission"]["status"] == "DONE" and "checks" in everything["correlation"]
        bad = await call(s, "check_alpha", alpha_id="A1", check="nope")
        assert "check must be one of" in bad["error"]


async def test_update_alpha_single_and_bulk(mcp_session, fake):
    async with mcp_session() as s:
        one = await call(s, "update_alpha", alpha_ids="A1", name="mine", color="RED", tags=["t"])
        assert one["alpha"]["name"] == "mine"
        many = await call(s, "update_alpha", alpha_ids=["A1", "A2"], favorite=True, color="GREEN")
        assert many["updated"] == ["color", "favorite"]
        refused = await call(s, "update_alpha", alpha_ids=["A1", "A2"], name="x")
        assert "one alpha at a time" in refused["error"]
        nothing = await call(s, "update_alpha", alpha_ids="A1")
        assert "nothing to update" in nothing["error"]
    single = fake.state.calls("PATCH", "/alphas/A1")[0]["body"]
    assert single == {"name": "mine", "color": "RED", "tags": ["t"]}
    assert fake.state.calls("PATCH", "/alphas")[0]["body"] == [
        {"id": "A1", "favorite": True, "color": "GREEN"}, {"id": "A2", "favorite": True, "color": "GREEN"}]


# ------------------------------------------------------------- submission


async def test_submit_dry_run_then_confirm(mcp_session, fake):
    async with mcp_session() as s:
        dry = await call(s, "submit_alpha", alpha_id="A1", wait_seconds=10)
        assert dry["dry_run"] is True and not fake.state.calls("POST", "/alphas/A1/submit")
        res = await call(s, "submit_alpha", alpha_id="A1", confirm=True, wait_seconds=10)
    assert res["success"] is True and res["status"] == "SUBMITTED"
    assert {c["name"] for c in res["checks"]} == {"LOW_SHARPE", "PROD_CORRELATION"}
    assert len(fake.state.calls("POST", "/alphas/A1/submit")) == 1
    assert len(fake.state.calls("GET", "/alphas/A1/submit")) >= 2


async def test_submit_pending_resumes_without_second_post(client, fake):
    first = await client.submit_alpha("A2", wait_seconds=0)
    assert first["status"] == "PENDING" and first["success"] is False
    second = await client.submit_alpha("A2", wait_seconds=10)
    assert second["status"] == "SUBMITTED"
    assert len(fake.state.calls("POST", "/alphas/A2/submit")) == 1


async def test_concurrent_submits_post_once(client, fake):
    fake.state.submit_post_delay = 0.5
    a, b = await asyncio.gather(client.submit_alpha("A3", 10), client.submit_alpha("A3", 10))
    assert a["status"] == b["status"] == "SUBMITTED"
    assert len(fake.state.calls("POST", "/alphas/A3/submit")) == 1


async def test_rejection_is_reported_and_resubmittable(client, fake):
    fake.state.submit_get_403 = True
    res = await client.submit_alpha("A4", wait_seconds=10)
    assert res["status"] == "REJECTED" and res["failed"] == ["LOW_SUB_UNIVERSE_SHARPE"]
    assert "A4" not in client._pending_submits
    fake.state.submit_get_403 = False
    fake.state.counters.clear()
    again = await client.submit_alpha("A4", wait_seconds=10)
    assert again["status"] == "SUBMITTED" and len(fake.state.calls("POST", "/alphas/A4/submit")) == 2


# -------------------------------------------------------------- read paths


async def test_pnl_contract_for_prodmemo(client, fake):
    assert await client.get_alpha_pnl("A1", max_wait=0) == {}  # still computing -> {}
    data = await client.get_alpha_pnl("A1", max_wait=10)
    assert len(data["records"]) == 500
    with pytest.raises(Exception):
        await client.get_alpha_pnl("MISSING/x", max_wait=0)


async def test_activity_kinds(mcp_session, fake):
    async with mcp_session() as s:
        div = await call(s, "get_activity", kind="diversity", grouping="region,delay")
        assert div["query"] == {"grouping": "region,delay"}
        assert fake.state.calls("GET", "/users/self/activities/diversity")
        other = await call(s, "get_activity", kind="diversity", user_id="SOMEONE")
        assert "only available for the current user" in other["error"]
        prof = await call(s, "get_activity", kind="profile", user_id="U999")
        assert prof["geniusLevel"] == "GOLD" and fake.state.calls("GET", "/users/U999/profile")
        pyr = await call(s, "get_activity", kind="pyramid-alphas", start_date="2026-01-01",
                         end_date="2026-02-01T00:00:00Z")
        assert pyr["query"] == {"startDate": "2026-01-01", "endDate": "2026-02-01"}
        mult = await call(s, "get_activity", kind="pyramid-multipliers")
        assert mult["pyramids"]
        pay = await call(s, "get_activity", kind="payments")
        assert "records" in pay["base_payments"] and "error" not in pay["base_payments"]
        needs = await call(s, "get_activity", kind="diversity-score", start_date="2026-01-01")
        assert "needs start_date and end_date" in needs["error"]
        bad = await call(s, "get_activity", kind="nope")
        assert "kind must be one of" in bad["error"]
        perf = await call(s, "get_alpha_performance", alpha_id="A1")
        assert perf["stats"]["after"]["sharpe"] == 1.6


async def test_diversity_score_without_per_alpha_requests(mcp_session, fake):
    async with mcp_session() as s:
        res = await call(s, "get_activity", kind="diversity-score", start_date="2026-01-01",
                         end_date="2026-12-31")
    assert res["N"] == 250 and res["A"] == 125 and res["P"] == 3 and res["P_max"] == 4
    assert res["complete"] and len(fake.state.calls("GET", "/users/self/alphas")) == 3
    assert not fake.state.calls("GET", r"/alphas/[^/]+")


async def test_data_community_docs(mcp_session, fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    async with mcp_session() as s:
        await call(s, "get_datafields", dataset_id="ds1", limit=5, offset=10)
        q = fake.state.calls("GET", "/data-fields")[0]["query"]
        assert q["limit"] == "5" and q["offset"] == "10" and "type" not in q
        await call(s, "get_datasets", limit=5, offset=5)
        assert fake.state.calls("GET", "/data-sets")[0]["query"]["offset"] == "5"
        ops = await call(s, "get_operators", category="time series")
        assert ops["count"] >= 1 and all(o["category"] == "Time Series" for o in ops["results"])
        assert "categories" in ops
        msgs = await call(s, "get_messages", limit=3)
        assert "base64" not in json.dumps(msgs) and not (tmp_path / "message_images").exists()
        lb = await call(s, "get_leaderboard")
        assert lb["results"][0]["user"] == "U123" and not fake.state.calls("GET", "/users/self")
        comps = await call(s, "get_competitions")
        assert "results" in comps and fake.state.calls("GET", "/users/self/competitions")
        comp = await call(s, "get_competitions", competition_id="C1", include_agreement=True)
        assert comp["id"] == "C1" and "agreement" in comp
        docs = await call(s, "get_documentation")
        assert docs["results"]
        page = await call(s, "get_documentation", page_id="P1")
        assert page["id"] == "P1"
        opts = await call(s, "get_platform_setting_options")
        assert opts["total_combinations"] == 3


async def test_accept_versions_are_opt_in(client, fake, monkeypatch):
    await client.get_user_alphas(stage="IS", limit=1)
    assert "version=" not in (fake.state.calls("GET", "/users/self/alphas")[0]["accept"] or "")
    monkeypatch.setattr(pf, "SEND_ACCEPT_VERSIONS", True)
    await client.get_user_alphas(stage="IS", limit=1)
    assert fake.state.calls("GET", "/users/self/alphas")[1]["accept"] == "application/json;version=4.0"


async def test_forum_tools(mcp_session, fake, monkeypatch):
    async def glossary():
        return {"terms": [{"term": "Alpha", "definition": "A signal"}], "count": 1}

    async def search(query, max_results=20, locale="zh-cn"):
        return {"results": [{"title": "t"}], "count": 1}

    monkeypatch.setattr(pf.forum_client, "get_glossary_terms", glossary)
    monkeypatch.setattr(pf.forum_client, "search_posts", search)
    async with mcp_session() as s:
        terms = await call(s, "get_glossary_terms")
        assert terms["terms"][0]["term"] == "Alpha"
        res = await call(s, "search_forum_posts", search_query="sharpe")
    assert res["success"] is True and res["total_found"] == 1


async def test_prodmemo_sync_status_mode(mcp_session, monkeypatch):
    seen = []

    async def sync_status():
        seen.append("status")
        return {"state": "idle"}

    async def start_sync(mode):
        seen.append(mode)
        return {"started": True}

    monkeypatch.setattr(pf.prodmemo_client, "sync_status", sync_status)
    monkeypatch.setattr(pf.prodmemo_client, "start_sync", start_sync)
    async with mcp_session() as s:
        assert (await call(s, "prodmemo_sync", mode="status")) == {"state": "idle"}
        assert (await call(s, "prodmemo_sync"))["started"] is True
    assert seen == ["status", "incremental"]


# ------------------------------------------------------------------ credd


async def test_credd_backoff_does_not_stampede(client, fake):
    await client.get_alpha_details("A1")
    fake.state.valid_token = "tok-2"
    fake.state.credd_same_version = True
    fake.state.credd_delay = 0.3
    t0 = time.monotonic()
    results = await asyncio.gather(*[client.get_alpha_details(f"A{i}") for i in range(8)],
                                   return_exceptions=True)
    assert all(isinstance(r, Exception) for r in results)
    assert fake.state.credd_calls == 2 and time.monotonic() - t0 < 3

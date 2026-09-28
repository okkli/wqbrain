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
from fake_brain import fake_alpha

TOOLS = {
    "brain_status",
    # simulation
    "create_simulation", "get_simulation", "cancel_simulation", "get_platform_setting_options",
    "preview_super_selection",
    # alphas
    "list_alphas", "get_alpha", "get_alpha_recordset", "check_alpha", "submit_alpha", "update_alpha",
    "get_alpha_performance", "compare_alphas",
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
    assert by_name["prodmemo_check"].annotations.openWorldHint is True  # reads BRAIN


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
        assert sub["mode"] == "multi" and sub["type"] == "REGULAR" and sub["expected_children"] == 3
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
        bad = await call(s, "create_simulation", expressions=["hump(x, 0.01)", "rank(x)"])
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
        one = await call(s, "create_simulation", expressions="hump(x, 0.01)")
        assert one["status"] == "SUBMITTED" and one["lint_warnings"][0]["issues"]
        conc = await call(s, "create_simulation", expressions=["hump(x, 0.01)", "rank(x)"],
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
        # concurrent POSTs race each other, so only the set of universes is fixed
        assert sorted(b["settings"]["universe"] for b in posted(fake)[1:]) == ["LARGE", "SMALL"]

        for args, msg in (({"universe": "TOP3000"}, "RAA universe must be one of"),
                          ({"region": "USA"}, "region='ALL'"),
                          ({"delay": 0}, "delay=1"),
                          ({"max_trade": "ON", "max_position": "ON"}, "cannot both be ON"),
                          ({"language": "PYTHON", "lookback": 5}, "REGION_AGNOSTIC takes FASTEXPR"),
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


async def test_multi_with_a_failed_child_is_not_complete(mcp_session, fake):
    fake.state.child_polls_needed = 1
    async with mcp_session() as s:
        multi = await call(s, "create_simulation", expressions=["rank(a)", "fail()"])
        single = await call(s, "create_simulation", expressions="rank(b)")
        one = await call(s, "get_simulation", simulation_ids=multi["simulation_id"], wait_seconds=10)
        assert one["status"] == "FINISHED_WITH_ERRORS" and one["failed_children"] == 1
        both = await call(s, "get_simulation", simulation_ids=[multi["simulation_id"], single["simulation_id"]],
                          wait_seconds=10)
        assert both["status"] == "FINISHED_WITH_ERRORS"
        assert both["counts"] == {"FINISHED_WITH_ERRORS": 1, "COMPLETE": 1}


async def test_single_full_result_reports_complete(mcp_session, fake):
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions="rank(a)")
        done = await call(s, "get_simulation", simulation_ids=sub["simulation_id"], wait_seconds=10,
                          compact=False)
    assert done["status"] == "COMPLETE" and done["alpha"]["status"] == "UNSUBMITTED"
    assert "researchNotes" in done["alpha"]


async def test_concurrent_status_counts(mcp_session, fake):
    fake.state.sim_slots_full = True
    async with mcp_session() as s:
        mixed = await call(s, "create_simulation", expressions=["bad(", "rank(a)"], mode="concurrent",
                           validate_expressions=False)
        assert mixed["status"] == "RATE_LIMITED" and "error" not in mixed
        assert (mixed["rate_limited"], mixed["failed"]) == (1, 1) and mixed["retry_after_seconds"] == 30
        fake.state.sim_slots_full = False
        some = await call(s, "create_simulation", expressions=["bad(", "rank(a)"], mode="concurrent",
                          validate_expressions=False)
        assert some["status"] == "PARTIAL" and "retry_after_seconds" not in some
        assert "rejected by BRAIN" in some["note"] and "RATE_LIMITED" not in some["note"]


async def test_create_simulation_edge_rules(mcp_session, fake):
    async with mcp_session() as s:
        raa = await call(s, "create_simulation", expressions="rank(x)", type="RAA",
                         per_alpha_settings=[{"delay": 1.0, "test_period": "P1Y0M"}, {"delay": "1"}])
        assert raa["status"] == "SUBMITTED"
        assert all("testPeriod" not in b["settings"] and b["settings"]["delay"] == 1 for b in posted(fake))
        sa_py = await call(s, "create_simulation", type="SA", combo="c", selection="s", language="PYTHON",
                           lookback=5)
        assert "is for REGULAR alphas" in sa_py["error"]
        rejected = await call(s, "create_simulation", expressions="bad(")
        assert "Invalid expression" in rejected["error"]
        assert rejected["lint_warnings"][0]["issues"] == ["unbalanced parentheses"]
        assert rejected["mode"] == "single" and "next" not in rejected


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
        empty = await call(s, "get_alpha", alpha_id="RAPEMPTY")
        assert empty["id"] == "RAPEMPTY" and empty["type"] == "RA_PARENT" and "no child alphas" in empty["note"]
    assert len(fake.state.calls("GET", "/alphas/RAPX")) == 1  # the parent is not fetched twice


async def test_recordsets(mcp_session, fake):
    async with mcp_session() as s:
        res = await call(s, "get_alpha_recordset", alpha_id="A1", recordset="sharpe")
        assert len(res["records"]) == 500  # waited out the Retry-After instead of returning {}
        tail = await call(s, "get_alpha_recordset", alpha_id="A1", recordset="pnl", max_rows=10)
        assert len(tail["records"]) == 10 and "last 10 of 500" in tail["truncated"]
        listed = await call(s, "get_alpha_recordset", alpha_id="A1")
        assert [r["name"] for r in listed["results"]] == ["pnl", "yearly-stats"]
        negative = await call(s, "get_alpha_recordset", alpha_id="A1", recordset="pnl", max_rows=-1)
        assert len(negative["records"]) == 500 and "truncated" not in negative
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
    assert "confirm=True" in first["note"]
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
        assert (await call(s, "prodmemo_sync", mode=" STOP "))["started"] is True
        bad = await call(s, "prodmemo_sync", mode="Fulll")
        assert "mode must be one of" in bad["error"]
    assert seen == ["status", "incremental", "stop"]


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


async def test_plain_dates_become_iso_datetimes(mcp_session, fake):
    async with mcp_session() as s:
        page = await call(s, "list_alphas", stage="OS", limit=1, start_date="2026-06-29",
                          end_date="2026-09-27", submission_start_date="2026-06-29T08:00:00",
                          submission_end_date="2026-09-27T23:00:00-04:00")
        assert page["count"] == 250
        score = await call(s, "get_activity", kind="diversity-score", start_date="2026-06-29",
                           end_date="2026-09-27")
        assert "error" not in score
    q = fake.state.calls("GET", "/users/self/alphas")
    assert q[0]["query"]["dateCreated>"] == "2026-06-29T00:00:00Z"
    assert q[0]["query"]["dateCreated<"] == "2026-09-27T23:59:59Z"
    assert q[0]["query"]["dateSubmitted>"] == "2026-06-29T08:00:00Z"
    assert q[0]["query"]["dateSubmitted<"] == "2026-09-27T23:00:00-04:00"
    assert q[1]["query"]["dateSubmitted<"] == "2026-09-27T23:59:59Z"


def test_poll_delay_grows_and_respects_bounds():
    import platform_functions as pf
    waits = [pf._poll_delay(1.0, n, 300) for n in range(12)]
    assert waits[0] == 1.0 and waits == sorted(waits) and waits[-1] == 15.0
    assert sum(1 for _ in waits) and sum(waits[:9]) > 60   # <= ~10 polls per 90s, not 90
    assert pf._poll_delay(30.0, 0, 300) == 30.0             # a longer Retry-After wins
    assert pf._poll_delay(1.0, 8, 4.0) == 4.0               # never past the wait budget
    assert pf._poll_delay(0.0, 0, 0.2) == 1.0


# ------------------------------------------------ fixes from the usage reports

CHECKS = [{"name": "LOW_SHARPE", "result": "WARNING", "limit": 1.58, "value": 1.4},
          {"name": "LOW_FITNESS", "result": "FAIL", "limit": 1.0, "value": 0.8},
          {"name": "HIGH_TURNOVER", "result": "PASS", "limit": 0.7, "value": 0.8},
          {"name": "LOW_TURNOVER", "result": "PASS", "limit": 0.01, "value": 0.8},
          {"name": "CLUSTER_TEST", "result": "WARNING"},
          {"name": "MATCHES_THEMES", "result": "WARNING"},
          {"name": "SELF_CORRELATION", "result": "PENDING"}]


async def test_compact_row_has_warns_and_the_whole_expression(mcp_session, fake):
    fake.state.alpha_checks, fake.state.child_polls_needed = CHECKS, 1
    long_expr = "rank(" + " + ".join(f"ts_mean(close, {n})" for n in range(10, 40)) + ")"
    assert len(long_expr) > 400
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions=long_expr, simulation_mode="QUICK")
        row = (await call(s, "get_simulation", simulation_ids=sub["simulation_id"], wait_seconds=5))["alpha"]
    # a value past its limit is a fail, whatever BRAIN called it; its verdict stays as a mark
    assert row["fails"] == ["SHARPE 1.4<1.58 (W)", "FITNESS 0.8<1", "HTURNOVER 0.8>0.7 (P)"]
    assert row["warns"] == ["CLUSTER_TEST"]
    assert "robust_sharpe" not in row and "sub_sharpe" not in row          # no value: left out
    assert row["expr"] == long_expr and row["set"]["mode"] == "QUICK"
    assert pf._cut("x" * 50, 20) == "x" * 20 + "…(+30 chars)" and pf._cut("abc", 0) == "abc"


async def test_prod_error_gives_no_verdict_and_names_the_source(mcp_session, fake):
    fake.state.check_prod_error = True
    async with mcp_session() as s:
        out = await call(s, "check_alpha", alpha_id="A1", wait_seconds=10)
        assert out["status"] == "DONE" and out["errored"] == ["PROD_CORRELATION"]
        assert out["all_passed"] is None                       # not True, whatever the fallback says
        assert out["prod_fallback"]["max_correlation"] == 0.65 and out["prod_fallback"]["passes_check"] is True
        assert "not the submission check" in out["prod_fallback"]["source"]
        assert out["prod_correlation"] == 0.65 and out["prod_source"].startswith("fallback")
        assert "unknown (null)" in out["note"]
        fake.state.check_prod_error = False
        ok = await call(s, "check_alpha", alpha_id="PRODOK", wait_seconds=10)
        assert ok["all_passed"] is True and ok["prod_correlation"] == 0.55
        assert ok["prod_source"].startswith("submission check") and "prod_fallback" not in ok
        none = await call(s, "check_alpha", alpha_id="A2", wait_seconds=10)   # no prod check at all
        assert "prod_fallback" not in none and "prod_source" not in none
    assert len(fake.state.calls("GET", "/alphas/A2/correlations/prod")) == 0


async def test_quick_mode_with_max_trade_warns(mcp_session, fake):
    async with mcp_session() as s:
        out = await call(s, "create_simulation", expressions="rank(x)",
                         per_alpha_settings=[{"simulation_mode": "QUICK", "max_trade": "ON"},
                                             {"max_trade": "ON"}, {"simulation_mode": "QUICK"}])
        assert out["status"] == "SUBMITTED" and len(out["warnings"]) == 1
        assert out["warnings"][0].startswith("item 0: QUICK mode ignores maxTrade")
        plain = await call(s, "create_simulation", expressions="rank(x)", max_trade="ON")
        assert "warnings" not in plain


async def test_datasets_are_paged_and_short_fields_stop_at_50(mcp_session, fake):
    async with mcp_session() as s:
        page = await call(s, "get_datasets")
        assert len(page["results"]) == 20 and page["count"] == 45 and page["next_offset"] == 20
        row = page["results"][0]
        assert row["category"] == "fundamental" and "researchPapers" not in row
        assert len(row["description"]) < 200 and len(json.dumps(page)) < 8000
        last = await call(s, "get_datasets", limit=50, offset=40)
        assert last["next_offset"] is None
        full = await call(s, "get_datasets", limit=2, detail=True)
        assert len(full["results"][0]["description"]) == 1000 and "researchPapers" in full["results"][0]
        fields = await call(s, "get_datafields", limit=100)
        assert "error" not in fields and len(fields["results"]) == 100 and fields["next_offset"] == 100
        assert set(fields["results"][0]) <= {"id", "type", "coverage", "dateCoverage", "userCount",
                                             "alphaCount", "desc"}
        rest = await call(s, "get_datafields", limit=500, offset=100)
        assert len(rest["results"]) == 20 and rest["next_offset"] is None
        full = await call(s, "get_datafields", limit=2, compact=False)
        assert full["results"][0]["dataset"] == {"id": "ds1"}
        model = await call(s, "get_datasets", category="model")
    assert fake.state.calls("GET", "/data-sets")[0]["query"]["limit"] == "20"
    assert fake.state.calls("GET", "/data-sets")[-1]["query"]["category"] == "model"
    queries = [c["query"] for c in fake.state.calls("GET", "/data-fields")]
    assert [(q["limit"], q["offset"]) for q in queries[:2]] == [("50", "0"), ("50", "50")]
    assert all(int(q["limit"]) <= 50 for q in queries)


async def test_waits_are_capped_and_a_hanging_read_is_given_up(mcp_session, fake, monkeypatch):
    assert pf._wait(300) == pf.MAX_TOOL_WAIT == 40 and pf._wait(5) == 5 and pf._wait(None) == 0
    assert pf._wait("x") == 0 and pf._wait(-3) == 0
    monkeypatch.setattr(pf, "TOOL_DEADLINE", 0.3)
    async with mcp_session() as s:
        async def hang(*a, **k):
            await asyncio.sleep(5)
        monkeypatch.setattr(pf.brain_client, "get_record_sets", hang)
        t0 = time.monotonic()
        out = await call(s, "get_alpha_recordset", alpha_id="A1")
        assert out["timed_out"] is True and out["status"] == "UNKNOWN" and time.monotonic() - t0 < 2


async def test_multi_rows_carry_index_reuse_and_duplicates(mcp_session, fake):
    fake.state.child_polls_needed = 1
    async with mcp_session() as s:
        sweep = await call(s, "create_simulation", expressions="rank(x)",
                           per_alpha_settings=[{"decay": 3}, {"decay": 5}])
        done = await call(s, "get_simulation", simulation_ids=sweep["simulation_id"], wait_seconds=10)
        assert [(r["index"], r["set"]["decay"]) for r in done["alpha_results"]] == [(0, 3), (1, 5)]
        assert not any("reused_alpha" in r or "submitted_expr" in r for r in done["alpha_results"])
        assert "missing_children" not in done

        mixed = await call(s, "create_simulation", expressions=["rank(a)", "alias(x)", "dup()", "dup()"])
        rows = (await call(s, "get_simulation", simulation_ids=mixed["simulation_id"],
                           wait_seconds=10))["alpha_results"]
        assert [r["index"] for r in rows] == [0, 1, 2, 3]
        assert rows[1]["reused_alpha"] is True and rows[1]["submitted_expr"] == "alias(x)"
        assert rows[1]["id"] == "OLD1" and rows[1]["expr"] == "rank(old_name)"
        assert "already existed" in rows[1]["warning"] and "not the one sent" in rows[1]["warning"]
        assert rows[3]["duplicate_of_index"] == 2 and rows[3]["id"] == rows[2]["id"] == "DUP1"
        assert "reused_alpha" not in rows[0] and "duplicate_of_index" not in rows[2]


async def test_multi_failure_names_the_item_and_the_reason(mcp_session, fake):
    fake.state.child_polls_needed = 1
    async with mcp_session() as s:
        multi = await call(s, "create_simulation", expressions=["rank(a)", "fail()", "rank(b)"])
        out = await call(s, "get_simulation", simulation_ids=multi["simulation_id"], wait_seconds=10)
    assert out["status"] == "FINISHED_WITH_ERRORS"
    assert out["errors"] == [{"index": 1, "status": "ERROR", "submitted_expr": "fail()",
                              "message": 'Attempted to use unknown variable "foo"'}]
    assert out["note"].startswith("errors[] holds BRAIN's reason")
    assert out["alpha_results"][1]["index"] == 1 and out["alpha_results"][1]["submitted_expr"] == "fail()"


def test_field_candidates():
    expr = ('a = ts_mean(close, 20); b = rank(a) * volume;  # vwap is only a comment\n'
            'group_neutralize(b, bucket(rank(cap), range="0.1,1,0.1")) + 1e5 * x_1 / returns')
    assert pf._field_candidates(expr) == ["close", "volume", "cap", "x_1", "returns"]
    assert pf._field_candidates("ts_backfill(nan_out(x), lookback=250) == inf ? nan : true") == ["x"]
    assert pf._field_candidates("stats = generate_stats(alpha); stats.returns") == ["alpha"]
    assert pf._field_candidates("/* close */ rank(open)") == ["open"]


async def test_unknown_fields_stop_a_multi_batch_and_warn_otherwise(mcp_session, fake):
    async with mcp_session() as s:
        bad = await call(s, "create_simulation", expressions=["rank(close)", "rank(nofield_x) + ts_rank(y, 250)"])
        assert "pre-check" in bad["error"] and not posted(fake)
        assert bad["problems"] == [{"index": 1, "expr": "rank(nofield_x) + ts_rank(y, 250)", "issues": [
            "ts_rank: argument 2 '250' must be written as constant=...",
            "unknown data field 'nofield_x' (BRAIN has no field of that name; a typo, or a variable "
            "that is never assigned?)"]}] or len(bad["problems"][0]["issues"]) >= 1
        assert any("nofield_x" in i for i in bad["problems"][0]["issues"])

        region = await call(s, "create_simulation", expressions=["rank(ind_only_f)", "rank(close)"])
        assert "has no data for USA delay 1 (it exists for IND/D0, IND/D1)" in region["problems"][0]["issues"][0]
        ind = await call(s, "create_simulation", expressions=["rank(ind_only_f)", "rank(close)"],
                         region="IND", universe="TOP500")
        assert ind["status"] == "SUBMITTED"

        single = await call(s, "create_simulation", expressions="rank(nofield_x)")
        assert single["status"] == "SUBMITTED" and "nofield_x" in single["lint_warnings"][0]["issues"][0]
        forced = await call(s, "create_simulation", expressions=["rank(nofield_x)", "rank(close)"],
                            validate_expressions=False)
        assert forced["status"] == "SUBMITTED" and "lint_warnings" not in forced
    lookups = [c["path"] for c in fake.state.calls("GET", "/data-fields/[^/]+")]
    assert lookups.count("/data-fields/close") == 1            # looked up once, then remembered


async def test_submissions_wait_in_the_queue_first_in_first_out(mcp_session, fake):
    fake.state.sim_slots_full, fake.state.child_polls_needed = True, 1
    async with mcp_session() as s:
        first = await call(s, "create_simulation", expressions="rank(a)", queue=True)
        assert first["status"] == "QUEUED" and first["queue_id"] == "Q1" and first["queue_position"] == 1
        assert first["next"] == "get_simulation(simulation_ids=['Q1'], wait_seconds=30)"
        second = await call(s, "create_simulation", expressions=["rank(b)", "rank(c)"], queue=True)
        third = await call(s, "create_simulation", expressions="rank(d)", queue=True)
        assert (second["queue_id"], second["queue_position"], second["mode"]) == ("Q2", 2, "multi")
        assert third["queue_position"] == 3
        refused = await call(s, "create_simulation", expressions="rank(e)", queue=False)
        assert refused["status"] == "RATE_LIMITED"

        listing = await call(s, "get_simulation")
        assert [(q["queue_position"], q["queue_id"], q["alphas"]) for q in listing["submit_queue"]] == \
            [(1, "Q1", 1), (2, "Q2", 2), (3, "Q3", 1)]
        waiting = await call(s, "get_simulation", simulation_ids="Q3")
        assert waiting["status"] == "QUEUED" and waiting["queue_position"] == 3
        gone = await call(s, "cancel_simulation", simulation_id="q2")
        assert gone["cancelled"] is True
        assert (await call(s, "get_simulation", simulation_ids="Q2"))["status"] == "CANCELLED"
        assert (await call(s, "get_simulation", simulation_ids="Q3"))["queue_position"] == 2

        fake.state.sim_slots_full = False                         # a slot frees up
        done = await call(s, "get_simulation", simulation_ids="Q1", wait_seconds=10)
        assert done["status"] == "COMPLETE" and done["queue_id"] == "Q1" and done["simulation_id"] == "S1"
        assert done["alpha"]["expr"] == "rank(a)"
        both = await call(s, "get_simulation", simulation_ids=["Q1", "Q3"], wait_seconds=10)
        assert both["status"] == "COMPLETE" and [x["simulation_id"] for x in both["simulations"]] == ["S1", "S2"]
        late = await call(s, "cancel_simulation", simulation_id="Q1")
        assert late["cancelled"] is False and late["simulation_id"] == "S1"
        assert "error" in await call(s, "get_simulation", simulation_ids="Q99")
    # accepted in the order they were asked for; the cancelled multi was never accepted
    assert [(k, v.get("regular")) for k, v in fake.state.sims.items()] == [("S1", "rank(a)"), ("S2", "rank(d)")]


async def test_concurrent_items_without_a_slot_are_queued(mcp_session, fake):
    fake.state.sim_post_limit = 1
    async with mcp_session() as s:
        out = await call(s, "create_simulation", expressions=["rank(a)", "rank(b)", "rank(c)"],
                         mode="concurrent", queue=True)
        assert out["status"] == "PARTIAL" and out["submitted"] == 1 and out["queued"] == 2
        assert sorted(r["status"] for r in out["simulations"]) == ["QUEUED", "QUEUED", "SUBMITTED"]
        assert len(out["simulation_ids"]) == 3 and "rate_limited" in out and out["rate_limited"] == 0
        queued = [r for r in out["simulations"] if r["status"] == "QUEUED"]
        assert [r["queue_position"] for r in queued] == [1, 2] and queued[0]["index"] < queued[1]["index"]
        more = await call(s, "create_simulation", expressions="rank(z)", queue=True)
        assert more["status"] == "QUEUED" and more["queue_position"] == 3      # nobody overtakes
        await call(s, "cancel_simulation", simulation_id=more["queue_id"])
        for r in queued:
            await call(s, "cancel_simulation", simulation_id=r["queue_id"])


async def test_compare_alphas_locally(mcp_session, fake):
    days = [f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 28)]
    steps = [((i * 37) % 11) - 5 for i in range(len(days))]
    def curve(scale, shift=0.0):
        total, out = 0.0, []
        for day, step in zip(days, steps):
            total += scale * step + shift
            out.append([day, total])
        return out
    fake.state.pnl = {"A1": curve(1.0), "B2": curve(2.5, 0.5), "C3": curve(-1.0)}
    async with mcp_session() as s:
        out = await call(s, "compare_alphas", alpha_ids=["A1", "B2", "C3"])
        assert out["status"] == "DONE" and len(out["pairs"]) == 3
        assert out["max"] == {"a": "A1", "b": "B2", "correlation": 1.0}
        assert out["pairs"][-1]["correlation"] == -1.0 and out["pairs"][0]["overlap_days"] == len(days) - 1
        assert out["alphas"][0] == {"id": "A1", "status": "OK", "days": len(days),
                                    "first_day": days[0], "last_day": days[-1]}
        assert "alpha_ids takes 2-10" in (await call(s, "compare_alphas", alpha_ids=["A1", "A1"]))["error"]
    assert not fake.state.calls("GET", "/alphas/[^/]+/correlations/.*")     # no correlation request spent


def test_brain_markup_is_removed_from_messages():
    assert pf._plain('Attempted to use unknown variable "x". <linkToCommonErrorMessages>Learn more'
                     '</linkToCommonErrorMessages>') == 'Attempted to use unknown variable "x".'
    assert pf._plain(None) is None and pf._plain("plain <b text") == "plain <b text"


async def test_lint_blocks_unknown_operators_and_warns_on_max_ops(mcp_session, fake):
    async with mcp_session() as s:
        bad = await call(s, "create_simulation", expressions=["vec_median(close)", "rank(close)"])
        assert "pre-check" in bad["error"] and not posted(fake)
        assert bad["problems"][0]["issues"][0].startswith("unknown operator vec_median()")
        ok = await call(s, "create_simulation", expressions=["rank(close)", "rank(-close) - rank(open)"], max_ops=3)
        assert ok["status"] == "SUBMITTED"
        assert ok["warnings"] == ["item 1: about 4 operators by BRAIN's count, more than max_ops=3 "
                                  "(it counts every call, infix and unary operator)"]


async def test_field_region_check_ignores_a_cut_off_list(mcp_session, fake):
    async with mcp_session() as s:
        many = await call(s, "create_simulation", expressions=["rank(wide_field)", "rank(wide_other)"],
                          region="MEA", universe="TOP400")
        assert many["status"] == "SUBMITTED", many         # a cut-off list proves nothing
        few = await call(s, "create_simulation", expressions=["rank(wide_field)", "rank(close)"],
                         region="MEA", universe="TOP400")
        assert "has no data for MEA" in few["problems"][0]["issues"][0]   # a complete list does


# ------------------------------------------------ field report 2026-09-28: P1-3/4/5, P2

async def test_rows_are_short_and_shared_settings_are_said_once(mcp_session, fake):
    fake.state.child_polls_needed = 1
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions="rank(x)", decay=4, neutralization="MARKET",
                         per_alpha_settings=[{}, {"nan_handling": "ON"}, {"pasteurization": "OFF"}])
        out = await call(s, "get_simulation", simulation_ids=sub["simulation_id"], wait_seconds=10)
        assert out["set"]["decay"] == 4 and out["set"]["neutralization"] == "MARKET"
        rows = out["alpha_results"]
        assert "set" not in rows[0]
        assert rows[1]["set"] == {"nanHandling": "ON"} and rows[2]["set"] == {"pasteurization": "OFF"}
        assert all("location" not in r and "progress_url" not in r for r in rows)
        tsv = await call(s, "get_simulation", simulation_ids=sub["simulation_id"], format="tsv")
        lines = tsv["tsv"].split("\n")
        assert lines[0].split("\t")[:4] == ["index", "id", "ops", "sharpe"] and len(lines) == 4
        assert "nanHandling=ON" in lines[2] and "decay=4" in lines[2] and "alpha_results" not in tsv


async def test_cancelled_children_become_one_list(mcp_session, fake):
    fake.state.child_polls_needed = 1
    fake.state.cancel_others = True
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions=["rank(a)", "fail()", "rank(b)", "rank(c)"])
        out = await call(s, "get_simulation", simulation_ids=sub["simulation_id"], wait_seconds=10)
    assert out["cancelled"] == [0, 2, 3]
    assert [r["index"] for r in out["alpha_results"]] == [1] and out["errors"][0]["index"] == 1


async def test_running_time_replaces_progress_and_stuck_runs_are_flagged(mcp_session, fake, monkeypatch):
    fake.state.child_polls_needed = 1000
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions="rank(x)")
        run = await call(s, "get_simulation", simulation_ids=sub["simulation_id"])
        assert run["status"] == "RUNNING" and "progress" not in run and run["running_seconds"] >= 0
        assert "stale" not in run
        monkeypatch.setattr(pf, "STALE_SINGLE_SECONDS", 0.0)
        stuck = await call(s, "get_simulation", simulation_ids=sub["simulation_id"])
        assert stuck["stale"] is True and "cancel_simulation" in stuck["note"]
        assert f'resubmit=\\"{sub["simulation_id"]}\\"' in stuck["note"] or \
            f'resubmit="{sub["simulation_id"]}"' in stuck["note"]


async def test_a_batch_failed_without_reason_is_resubmitted_once(mcp_session, fake):
    fake.state.child_polls_needed = 1
    fake.state.glitch_batches = 1                 # the next multi fails every child, no message
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions=["rank(a)", "rank(b)"])
        first = await call(s, "get_simulation", simulation_ids=sub["simulation_id"], wait_seconds=10)
        assert first["status"] == "RETRIED" and first["retried_as"] != sub["simulation_id"]
        again = await call(s, "get_simulation", simulation_ids=sub["simulation_id"], wait_seconds=10)
        assert again["status"] == "COMPLETE" and again["retried_as"] == first["retried_as"]
        assert [r["expr"] for r in again["alpha_results"]] == ["rank(a)", "rank(b)"]
    assert len(posted(fake)) == 2


async def test_a_simulation_brain_forgot_is_recovered_from_the_alpha_list(mcp_session, fake):
    fake.state.child_polls_needed = 1
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions=["rank(a)", "rank(b)"], decay=3)
        sid = sub["simulation_id"]
        fake.state.forgotten.add(sid)             # BRAIN now answers 404 for it
        fake.state.listed = [dict(fake_alpha("Z1", "rank(b)", 3)), dict(fake_alpha("Z0", "rank(a)", 3))]
        out = await call(s, "get_simulation", simulation_ids=sid)
    assert out["recovered"] is True and out["status"] == "COMPLETE"
    assert [(r["index"], r["id"]) for r in out["alpha_results"]] == [(0, "Z0"), (1, "Z1")]


async def test_resubmit_sends_the_same_items_again(mcp_session, fake):
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", expressions="rank(x)", per_alpha_settings=[{"decay": 2}, {"decay": 7}])
        again = await call(s, "create_simulation", resubmit=sub["simulation_id"])
        assert again["status"] == "SUBMITTED" and again["resubmitted"] == sub["simulation_id"]
        missing = await call(s, "create_simulation", resubmit="NOPE")
        assert "no record" in missing["error"]
    first, second = posted(fake)
    assert first == second


async def test_tagged_results_are_logged_and_served(mcp_session, fake, tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(pf, "WATCH_INTERVAL", 0.05)
    fake.state.child_polls_needed = 1
    async with mcp_session() as s:
        bad = await call(s, "create_simulation", expressions="rank(x)", tag="../evil")
        assert "tag must be" in bad["error"] and not posted(fake)
        assert "pass tag as well" in (await call(s, "create_simulation", expressions="rank(x)", labels=["a"]))["error"]
        sub = await call(s, "create_simulation", expressions="rank(x)", tag="G-r3",
                         labels=["momentum", "reversal"], per_alpha_settings=[{"decay": 2}, {"decay": 7}])
        assert sub["status"] == "SUBMITTED"
        path = tmp_path / "G-r3.jsonl"
        for _ in range(200):                      # the background watcher writes it, nobody polls
            if path.exists():
                break
            await asyncio.sleep(0.02)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [(r["index"], r["label"], r["set"]["decay"]) for r in rows] == [(0, "momentum", 2), (1, "reversal", 7)]
        assert rows[0]["tag"] == "G-r3" and rows[0]["expr"] == "rank(x)" and rows[0]["id"]
        await call(s, "get_simulation", simulation_ids=sub["simulation_id"])
        assert len(path.read_text().splitlines()) == 2        # written once
        conc = await call(s, "create_simulation", expressions=["rank(a)", "rank(b)"], mode="concurrent",
                          tag="G-r3", labels=["a", "b"])
        for sid in conc["simulation_ids"]:
            await call(s, "get_simulation", simulation_ids=sid, wait_seconds=10)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert sorted((r["index"], r["label"]) for r in rows[2:]) == [(0, "a"), (1, "b")]

    from starlette.testclient import TestClient
    client = TestClient(pf.mcp.streamable_http_app())
    got = client.get("/results/G-r3.jsonl")
    assert got.status_code == 200 and got.headers["X-Lines"] == "4" and len(got.text.splitlines()) == 4
    assert len(client.get("/results/G-r3.jsonl?since=3").text.splitlines()) == 1
    tsv = client.get("/results/G-r3.jsonl?format=tsv").text.splitlines()
    assert tsv[0].startswith("ts\tsimulation_id\tlabel\tindex\tid") and len(tsv) == 5
    assert client.get("/results/nothing.jsonl").text == ""
    assert client.get("/results/..%2Fx.jsonl").status_code in (400, 404)


async def test_wait_is_one_budget_and_says_when_it_was_capped(mcp_session, fake):
    fake.state.child_polls_needed = 1000
    async with mcp_session() as s:
        a = await call(s, "create_simulation", expressions="rank(a)")
        b = await call(s, "create_simulation", expressions="rank(b)")
        t0 = time.monotonic()
        out = await call(s, "get_simulation", simulation_ids=[a["simulation_id"], b["simulation_id"]], wait_seconds=2)
        took = time.monotonic() - t0
        assert 1.5 <= took < 4 and out["waited_seconds"] >= 1 and "wait_capped_to" not in out
        capped = await call(s, "get_simulation", simulation_ids=a["simulation_id"], wait_seconds=0.1)
        assert "wait_capped_to" not in capped
    assert pf._capped(120) == {"wait_capped_to": 40} and pf._capped(30) == {}

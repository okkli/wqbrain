"""End-to-end MCP tests: real FastMCP server <-> in-memory client session."""

from __future__ import annotations

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import brain_client
import server

EXPECTED_TOOLS = {
    "brain_status", "create_simulation", "get_simulation", "cancel_simulation",
    "get_platform_setting_options", "preview_super_selection", "list_alphas", "get_alpha",
    "get_alpha_recordset", "check_alpha", "submit_alpha", "update_alpha", "get_alpha_performance",
    "get_datasets", "get_datafields", "get_operators", "get_activity", "get_leaderboard",
    "get_competitions", "get_events", "get_messages", "get_documentation",
    "search_forum_posts", "read_forum_post", "get_glossary_terms",
}


@pytest.fixture
def mcp_session(fake, monkeypatch):
    """Factory for an in-memory MCP session (the anyio task group must be entered and
    exited in the test's own task, so this can't be an async yield fixture)."""
    fresh = brain_client.BrainClient(fake.url)
    monkeypatch.setattr(server, "brain", fresh)
    yield lambda: create_connected_server_and_client_session(server.mcp._mcp_server)
    fresh._executor.shutdown(wait=False)


async def call(session, _tool, **args):
    res = await session.call_tool(_tool, args)
    text = res.content[0].text if res.content else ""
    if res.isError:
        return True, text
    return False, json.loads(text)


async def test_tool_surface(mcp_session):
    async with mcp_session() as session:
        tools = (await session.list_tools()).tools
        assert {t.name for t in tools} == EXPECTED_TOOLS
        by_name = {t.name: t for t in tools}
        assert by_name["submit_alpha"].annotations.destructiveHint is True
        assert by_name["cancel_simulation"].annotations.destructiveHint is True
        assert by_name["get_alpha"].annotations.readOnlyHint is True
        # enums are part of the schema, so clients can't send e.g. correlation_type='production'
        check_schema = by_name["check_alpha"].inputSchema["properties"]["correlations"]
        assert check_schema["items"]["enum"] == ["self", "prod", "power-pool"]
        assert sum(len(t.description or "") for t in tools) < 8000


async def test_errors_are_mcp_errors(mcp_session, fake):
    async with mcp_session() as session:
        is_error, text = await call(session, "get_alpha", alpha_id="../users/self")
        assert is_error and "invalid alpha id" in text
        is_error, text = await call(session, "get_alpha", alpha_id="MISSING")
        assert is_error and "HTTP 404" in text
        for url in (f"{fake.url}/cookies?x=worldquantbrain.com", "http://127.0.0.1:8762/cookies",
                    f"{fake.url}/simulations/S1/../../cookies"):
            is_error, text = await call(session, "get_simulation", simulation_ids=[url])
            assert is_error and "not a BRAIN simulation URL" in text
        assert fake.state.credd_calls == 1  # only the session bootstrap; the URLs were never fetched
        assert len(fake.state.calls("GET", "/cookies")) == 1


async def test_simulation_flow(mcp_session, fake):
    async with mcp_session() as session:
        is_error, sub = await call(session, "create_simulation", expressions=["rank(close)", "rank(open)"],
                                   region="USA", decay=6)
        assert not is_error and sub["status"] == "SUBMITTED" and sub["multi"]
        assert sub["settings_used"]["decay"] == 6 and sub["settings_used"]["neutralization"] == "SUBINDUSTRY"
        is_error, recent = await call(session, "get_simulation")
        assert recent["recent_simulations"][0]["simulation_id"] == sub["simulation_id"]
        is_error, state = await call(session, "get_simulation", simulation_ids=[sub["simulation_id"]], wait_seconds=15)
        sim = state["simulations"][0]
        assert sim["status"] == "COMPLETE" and sim["total_children"] == 2
        assert sim["children"][0]["alpha"]["is"]["sharpe"] == 1.4


async def test_create_simulation_validation(mcp_session):
    async with mcp_session() as session:
        is_error, text = await call(session, "create_simulation", expressions=[])
        assert is_error and "at least one" in text
        is_error, text = await call(session, "create_simulation", type="SUPER", combo="x")
        assert is_error and "combo and selection" in text
        is_error, text = await call(session, "create_simulation", expressions=["rank(x)"], decay=1000)
        assert is_error


async def test_submit_requires_confirm(mcp_session, fake):
    async with mcp_session() as session:
        is_error, dry = await call(session, "submit_alpha", alpha_id="A1", wait_seconds=10)
        assert not is_error and dry["dry_run"] and dry["failed"] == ["SELF_CORRELATION"]
        assert not fake.state.calls("POST", "/alphas/A1/submit")
        is_error, real = await call(session, "submit_alpha", alpha_id="A1", confirm=True, wait_seconds=10)
        assert not is_error and real["status"] == "SUBMITTED"
        assert len(fake.state.calls("POST", "/alphas/A1/submit")) == 1


async def test_update_alpha_tool_clears_with_empty_string(mcp_session, fake):
    async with mcp_session() as session:
        is_error, res = await call(session, "update_alpha", alpha_ids=["A1"], name="", color="")
        assert not is_error
        assert fake.state.calls("PATCH", "/alphas")[0]["body"] == [{"id": "A1", "color": None}]
        assert fake.state.calls("PATCH", "/alphas/A1")[0]["body"] == {"name": None}


async def test_read_tools(mcp_session, fake):
    async with mcp_session() as session:
        is_error, st = await call(session, "brain_status", overview=True)
        assert st["user_id"] == "U123" and st["alphas"]["active"] == 3 and st["messages"]["unread"] == 2
        is_error, ops = await call(session, "get_operators", category="time series")
        assert [o["name"] for o in ops["results"]] == ["ts_rank"] and "Cross Sectional" in ops["categories"]
        is_error, ds = await call(session, "get_datasets", search="assets", include_fields=True)
        assert ds["fields_total"] == 1
        is_error, df = await call(session, "get_datafields", dataset_id="ds1", limit=10)
        assert df["returned"] == 10
        is_error, vf = await call(session, "get_activity", kind="value-factor")
        assert vf["leaderboard"]["valueFactor"] == 0.8
        is_error, div = await call(session, "get_activity", kind="diversity-score",
                                   start_date="2026-01-01", end_date="2026-12-31")
        assert div["N"] == 250 and div["official"]["valueFactor"] == 0.8
        is_error, rs = await call(session, "get_alpha_recordset", alpha_id="A1", recordset="yearly-stats",
                                  wait_seconds=10, max_rows=0)
        assert rs["total_rows"] == 500 and len(rs["rows"]) == 500
        is_error, comp = await call(session, "get_competitions", competition_id="GAC2026", include_agreement=True)
        assert comp["competition"]["id"] == "GAC2026" and comp["agreement"] is None
        is_error, docs = await call(session, "get_documentation", page_id="P1")
        assert docs["content"][2] == {"type": "EQUATION", "value": "rank(close)"}


async def test_write_kill_switches(mcp_session, fake, monkeypatch):
    monkeypatch.setattr(server, "ALLOW_SUBMIT", False)
    async with mcp_session() as session:
        is_error, text = await call(session, "submit_alpha", alpha_id="A1", confirm=True)
        assert is_error and "WQMCP_ALLOW_SUBMIT=0" in text
        is_error, dry = await call(session, "submit_alpha", alpha_id="A1", wait_seconds=10)
        assert not is_error and dry["dry_run"]
    monkeypatch.setattr(server, "READ_ONLY", True)
    async with mcp_session() as session:
        for tool, args in (("create_simulation", {"expressions": ["rank(close)"]}),
                           ("cancel_simulation", {"simulation_id": "S1"}),
                           ("update_alpha", {"alpha_ids": ["A1"], "favorite": True})):
            is_error, text = await call(session, tool, **args)
            assert is_error and "WQMCP_READ_ONLY=1" in text, tool
    assert not fake.state.calls("POST", "/simulations") and not fake.state.calls("POST", "/alphas/A1/submit")



async def test_empty_bodies_are_wrapped(mcp_session, fake, monkeypatch):
    async def empty(*a, **k):
        return None
    async with mcp_session() as session:
        monkeypatch.setattr(server.brain, "consultant", empty)
        is_error, res = await call(session, "get_activity", kind="value-factor")
        assert not is_error and res == {"result": None}

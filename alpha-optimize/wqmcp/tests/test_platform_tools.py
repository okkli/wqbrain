"""platform_functions (main's 48-tool server) against the fake BRAIN, over a real
in-memory MCP session. Tool names/params are the public contract: only
additive changes are allowed."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import platform_functions as pf

MAIN_TOOLS = {
    "authenticate", "manage_config", "create_simulation", "check_simulation_progress",
    "create_raa_simulation", "get_raa_alpha", "get_alpha_details", "get_datasets", "get_datafields",
    "get_alpha_pnl", "get_user_alphas", "submit_alpha", "value_factor_trendScore", "get_events",
    "get_leaderboard", "get_operators", "run_selection", "get_user_profile", "get_documentations",
    "get_messages", "get_glossary_terms", "search_forum_posts", "read_forum_post",
    "get_alpha_yearly_stats", "check_correlation", "get_submission_check", "set_alpha_properties",
    "get_record_sets", "get_record_set_data", "get_user_activities", "get_pyramid_multipliers",
    "get_pyramid_alphas", "get_user_competitions", "get_competition_details",
    "get_competition_agreement", "get_platform_setting_options", "performance_comparison",
    "expand_nested_data", "get_documentation_page", "create_multi_simulation",
    "get_daily_and_quarterly_payment", "lookINTO_SimError_message", "prodmemo_sync",
    "prodmemo_sync_status", "prodmemo_check", "prodmemo_get", "prodmemo_stats", "prodmemo_manage",
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


# ------------------------------------------------------------- contract


async def test_tool_names_are_unchanged(mcp_session):
    async with mcp_session() as s:
        names = {t.name for t in (await s.list_tools()).tools}
    assert names == MAIN_TOOLS


def test_server_binds_loopback_by_default():
    assert pf.mcp.settings.host == "127.0.0.1" and pf.mcp.settings.port == 8761


# ------------------------------------------------------------- security


async def test_arbitrary_urls_and_path_tricks_are_refused(mcp_session, fake):
    async with mcp_session() as s:
        for url in (f"{fake.url}/cookies", "http://127.0.0.1:8762/cookies?x=worldquantbrain.com",
                    "https://evil.example/simulations/S1", f"{fake.url}/simulations/S1/../../cookies"):
            res = await call(s, "check_simulation_progress", progress_url=url)
            assert "not a BRAIN simulation URL" in res["error"]
        looked = await call(s, "lookINTO_SimError_message", locations=[f"{fake.url}/cookies"])
        assert "not a BRAIN simulation URL" in looked["results"][0]["error"]
        bad = await call(s, "get_alpha_details", alpha_id="../users/self")
        assert "invalid alpha id" in bad["error"]
        bad = await call(s, "set_alpha_properties", alpha_id="A1/../../users/self", name="x")
        assert "invalid alpha id" in bad["error"]
    assert not fake.state.calls("GET", "/cookies")[1:]  # only the session bootstrap
    assert not fake.state.calls("GET", "/users/self")


async def test_manage_config_never_echoes_secrets(mcp_session, fake, tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"credentials": {"email": "me@x", "password": "hunter2"}, "note": "ok"}))
    monkeypatch.setenv("MCP_CONFIG_FILE", str(cfg))
    async with mcp_session() as s:
        res = await call(s, "manage_config", action="get")
    assert res["config"]["credentials"]["password"] == "***" and res["config"]["note"] == "ok"
    assert "hunter2" not in json.dumps(res)


async def test_write_kill_switches(mcp_session, fake, monkeypatch):
    monkeypatch.setattr(pf, "ALLOW_SUBMIT", False)
    async with mcp_session() as s:
        res = await call(s, "submit_alpha", alpha_id="A1")
        assert "WQMCP_ALLOW_SUBMIT=0" in res["error"]
    monkeypatch.setattr(pf, "READ_ONLY", True)
    async with mcp_session() as s:
        for tool, args in (("create_simulation", {"regular": "rank(close)"}),
                           ("create_multi_simulation", {"alpha_expressions": ["rank(a)", "rank(b)"]}),
                           ("create_raa_simulation", {"regular": "rank(close)"}),
                           ("set_alpha_properties", {"alpha_id": "A1", "name": "x"})):
            res = await call(s, tool, **args)
            assert "WQMCP_READ_ONLY=1" in res["error"], tool
    assert not fake.state.calls("POST", r"/simulations|/alphas/.*")


# ----------------------------------------------------------- simulations


async def test_single_simulation_compact_flow(mcp_session, fake):
    async with mcp_session() as s:
        sub = await call(s, "create_simulation", regular="rank(close)")
        assert sub["status"] == "SUBMITTED"
        running = await call(s, "check_simulation_progress", progress_url=sub["progress_url"])
        assert running["status"] == "RUNNING"
        done = await call(s, "check_simulation_progress", progress_url=sub["simulation_id"], wait_seconds=10)
        assert done["status"] == "COMPLETE" and done["alpha"]["sharpe"] == 1.4


async def test_multi_simulation_parent_retry_after_and_busy_child(mcp_session, fake):
    fake.state.child_polls_needed = 1
    async with mcp_session() as s:
        sub = await call(s, "create_multi_simulation", alpha_expressions=["rank(a)", "rank(b)", "rank(c)"])
        fake.state.child_429_times = 3  # outlasts _request's two GET retries
        body = fake.state.calls("POST", "/simulations")[0]["body"]
        assert isinstance(body, list) and len(body) == 3
        first = await call(s, "check_simulation_progress", progress_url=sub["progress_url"])
        assert first["status"] == "RUNNING" and "children" not in first  # parent Retry-After honoured
        assert not fake.state.calls("GET", r"/simulations/S\d+C\d")
        second = await call(s, "check_simulation_progress", progress_url=sub["progress_url"])
        assert second["status"] == "RUNNING" and second["completed_children"] == 2  # busy child != done
        final = await call(s, "check_simulation_progress", progress_url=sub["progress_url"], wait_seconds=10)
        assert final["status"] == "COMPLETE" and len(final["alpha_results"]) == 3


async def test_multi_simulation_overrides_and_lint(mcp_session, fake):
    async with mcp_session() as s:
        bad = await call(s, "create_multi_simulation", alpha_expressions=["ts_backfill(x, 250)", "rank(x)"])
        assert "pre-check" in bad["error"] and not fake.state.calls("POST", "/simulations")
        ok = await call(s, "create_multi_simulation", alpha_expressions=["rank(x)"],
                        per_alpha_settings=[{"decay": 3}, {"neutralization": "MARKET"}])
        assert ok["status"] == "SUBMITTED"
        body = fake.state.calls("POST", "/simulations")[0]["body"]
        assert body[0]["settings"]["decay"] == 3 and body[1]["settings"]["neutralization"] == "MARKET"


async def test_raa_flow(mcp_session, fake):
    async with mcp_session() as s:
        sub = await call(s, "create_raa_simulation", regular="rank(close)")
        assert fake.state.calls("POST", "/simulations")[0]["body"]["settings"]["region"] == "ALL"
        done = await call(s, "check_simulation_progress", progress_url=sub["progress_url"], wait_seconds=10)
        assert done["status"] == "COMPLETE" and done["type"] == "REGION_AGNOSTIC"
        assert {c["region"] for c in done["children"]} == {"USA", "EUR", "ASI", "GLB"}


# ------------------------------------------------------------- submission


async def test_submit_follows_the_submission_to_its_verdict(mcp_session, fake):
    async with mcp_session() as s:
        res = await call(s, "submit_alpha", alpha_id="A1", wait_seconds=10)
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


async def test_record_set_polls_then_returns_data(mcp_session, fake):
    async with mcp_session() as s:
        res = await call(s, "get_record_set_data", alpha_id="A1", record_set_name="sharpe")
        assert len(res["records"]) == 500  # waited out the Retry-After instead of returning {}
        listed = await call(s, "get_record_sets", alpha_id="A1")
        assert [r["name"] for r in listed["results"]] == ["pnl", "yearly-stats"]


async def test_record_set_pending_is_explicit(client, fake):
    res = await client.get_record_set_data("A2", "turnover", max_wait=0)
    assert res["status"] == "PENDING" and res["retry_after_seconds"] >= 1


async def test_activities_profile_pyramids_performance(mcp_session, fake):
    async with mcp_session() as s:
        div = await call(s, "get_user_activities", user_id="self", grouping="region,delay")
        assert div["query"] == {"grouping": "region,delay"}
        assert fake.state.calls("GET", "/users/self/activities/diversity")
        other = await call(s, "get_user_activities", user_id="SOMEONE")
        assert "only available for the current user" in other["error"]
        prof = await call(s, "get_user_profile", user_id="U999")
        assert prof["geniusLevel"] == "GOLD" and fake.state.calls("GET", "/users/U999/profile")
        pyr = await call(s, "get_pyramid_alphas", start_date="2026-01-01", end_date="2026-02-01T00:00:00Z")
        assert pyr["query"] == {"startDate": "2026-01-01", "endDate": "2026-02-01"}
        perf = await call(s, "performance_comparison", alpha_id="A1", team_id="T1")
        assert perf["stats"]["after"]["sharpe"] == 1.6 and "team_id" in perf["note"]


async def test_value_factor_without_per_alpha_requests(mcp_session, fake):
    async with mcp_session() as s:
        res = await call(s, "value_factor_trendScore", start_date="2026-01-01", end_date="2026-12-31")
    assert res["N"] == 250 and res["A"] == 125 and res["P"] == 3 and res["P_max"] == 4
    assert res["complete"] and len(fake.state.calls("GET", "/users/self/alphas")) == 3
    assert not fake.state.calls("GET", r"/alphas/[^/]+")


async def test_paging_selection_payments_messages_leaderboard(mcp_session, fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    async with mcp_session() as s:
        await call(s, "get_datafields", dataset_id="ds1", limit=5, offset=10)
        q = fake.state.calls("GET", "/data-fields")[0]["query"]
        assert q["limit"] == "5" and q["offset"] == "10" and "type" not in q
        await call(s, "get_datasets", limit=5, offset=5)
        assert fake.state.calls("GET", "/data-sets")[0]["query"]["offset"] == "5"
        await call(s, "run_selection", selection="rank(sharpe)", region="CHN", limit=3)
        q = fake.state.calls("GET", "/simulations/super-selection")[0]["query"]
        assert q["settings.region"] == "CHN" and q["region"] == "CHN" and q["limit"] == "3"
        pay = await call(s, "get_daily_and_quarterly_payment")
        assert "records" in pay["base_payments"] and "error" not in pay["base_payments"]
        msgs = await call(s, "get_messages", limit=3)
        assert "base64" not in json.dumps(msgs) and not (tmp_path / "message_images").exists()
        lb = await call(s, "get_leaderboard")
        assert lb["results"][0]["user"] == "U123" and not fake.state.calls("GET", "/users/self")


async def test_accept_versions_are_opt_in(client, fake, monkeypatch):
    await client.get_user_alphas(stage="IS", limit=1)
    assert "version=" not in (fake.state.calls("GET", "/users/self/alphas")[0]["accept"] or "")
    monkeypatch.setattr(pf, "SEND_ACCEPT_VERSIONS", True)
    await client.get_user_alphas(stage="IS", limit=1)
    assert fake.state.calls("GET", "/users/self/alphas")[1]["accept"] == "application/json;version=4.0"


async def test_forum_tools_keep_their_output_shape(mcp_session, fake, monkeypatch):
    async def glossary():
        return {"terms": [{"term": "Alpha", "definition": "A signal"}], "count": 1}

    async def search(query, max_results=20, locale="zh-cn"):
        return {"results": [{"title": "t"}], "count": 1}

    monkeypatch.setattr(pf.forum_client, "get_glossary_terms", glossary)
    monkeypatch.setattr(pf.forum_client, "search_posts", search)
    async with mcp_session() as s:
        terms = await s.call_tool("get_glossary_terms", {})
        assert json.loads(terms.content[0].text)["term"] == "Alpha"  # list of terms, as before
        res = await call(s, "search_forum_posts", search_query="sharpe", password="ignored")
    assert res["success"] is True and res["total_found"] == 1


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

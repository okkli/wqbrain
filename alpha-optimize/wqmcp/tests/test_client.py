"""BrainClient behaviour against the fake BRAIN (catalog-modelled)."""

from __future__ import annotations

import asyncio
import time

import pytest

import brain_client
from brain_client import BrainAPIError, InvalidArgument, parse_setting_options, seg


# ------------------------------------------------------------------ helpers


def test_seg_rejects_path_tricks():
    assert seg("abc_123-X.y", "id") == "abc_123-X.y"
    for bad in ("", "../users/self", "a/b", "a?b", "a#b", "a b", ".hidden", "..", "x" * 200, None):
        with pytest.raises(InvalidArgument):
            seg(bad, "id")


def test_accept_versions_follow_catalog():
    h = brain_client.accept_header
    assert h("GET", "/users/self/alphas") == "application/json;version=4.0"
    assert h("GET", "/users/self/alphas/summary") == "application/json;version=4.0"
    assert h("GET", "/users/self/activities/base-payment") == "application/json;version=3.0"
    assert h("OPTIONS", "/simulations") == "application/json;version=3.0"
    assert h("POST", "/simulations") == "application/json;version=2.0"
    assert h("GET", "/alphas/abc") == "application/json;version=2.0"


def test_simulation_ref_validation(client, fake):
    assert client.simulation_id_from("S1") == "S1"
    assert client.simulation_id_from(f"{fake.url}/simulations/S1") == "S1"
    for bad in ("http://127.0.0.1:8762/cookies?x=worldquantbrain.com",
                "https://evil.example/simulations/S1",
                f"{fake.url}/simulations/S1?x=1",
                f"{fake.url}/users/self",
                "../users/self"):
        with pytest.raises(InvalidArgument):
            client.simulation_id_from(bad)


async def test_path_injection_never_reaches_brain(client, fake):
    with pytest.raises(InvalidArgument):
        await client.get_alpha("../users/self")
    with pytest.raises(InvalidArgument):
        await client.update_alpha(["A1/../../users/self"], {"name": "x"}, {})
    assert fake.state.requests == []


# -------------------------------------------------------------- simulation


async def test_single_simulation_uses_platform_defaults(client, fake):
    settings = await client.build_settings("REGULAR", {"region": "USA"})
    assert settings["decay"] == 4 and isinstance(settings["decay"], int)
    assert settings["neutralization"] == "SUBINDUSTRY"
    for key in ("selectionHandling", "selectionLimit", "componentActivation", "lookback"):
        assert key not in settings
    res = await client.create_simulations([{"type": "REGULAR", "settings": settings, "regular": "rank(close)"}])
    assert res["status"] == "SUBMITTED" and res["multi"] is False
    post = fake.state.calls("POST", "/simulations")[0]
    assert isinstance(post["body"], dict) and post["body"]["regular"] == "rank(close)"
    assert client.recent_simulations[0]["simulation_id"] == res["simulation_id"]

    first = await client.simulations([res["simulation_id"]], wait_seconds=0)
    assert first[0]["status"] == "RUNNING" and first[0]["progress"] == 0.5
    done = await client.simulations([res["simulation_id"]], wait_seconds=10)
    assert done[0]["status"] == "COMPLETE"
    assert done[0]["alpha"]["id"] == f"A-{res['simulation_id']}"
    assert "researchNotes" not in done[0]["alpha"]


async def test_multi_simulation_child_rate_limit_is_not_completion(client, fake):
    fake.state.child_429_once = True
    settings = await client.build_settings("REGULAR", {})
    items = [{"type": "REGULAR", "settings": settings, "regular": f"rank(x{i})"} for i in range(3)]
    res = await client.create_simulations(items)
    assert isinstance(fake.state.calls("POST", "/simulations")[0]["body"], list)
    state = (await client.simulations([res["simulation_id"]], wait_seconds=0))[0]
    assert state["multi"] and state["status"] == "RUNNING"
    child1 = [c for c in state["children"] if c["simulation_id"].endswith("C1")][0]
    assert child1["status"] == "UNKNOWN"
    final = (await client.simulations([res["simulation_id"]], wait_seconds=15))[0]
    assert final["status"] == "COMPLETE" and final["completed_children"] == 3
    assert all(c.get("alpha", {}).get("id") for c in final["children"])


async def test_simulation_error_message_is_surfaced(client, fake):
    settings = await client.build_settings("REGULAR", {})
    res = await client.create_simulations([{"type": "REGULAR", "settings": settings, "regular": "fail()"}])
    state = (await client.simulations([res["simulation_id"]], wait_seconds=10))[0]
    assert state["status"] == "ERROR"
    assert "unknown variable" in state["message"]


async def test_create_error_keeps_brain_reason(client, fake):
    settings = await client.build_settings("REGULAR", {})
    with pytest.raises(BrainAPIError) as exc:
        await client.create_simulations([{"type": "REGULAR", "settings": settings, "regular": "bad("}])
    assert exc.value.status == 400 and "Invalid expression" in str(exc.value)


async def test_simulation_slots_full(client, fake):
    fake.state.sim_slots_full = True
    settings = await client.build_settings("REGULAR", {})
    res = await client.create_simulations([{"type": "REGULAR", "settings": settings, "regular": "rank(close)"}])
    assert res["status"] == "RATE_LIMITED" and res["retry_after_seconds"] == 30
    # a slot-limit 429 on POST /simulations must not throttle every other call
    t0 = time.monotonic()
    await client.get_alpha("A1")
    assert time.monotonic() - t0 < 1


async def test_settings_validation(client, fake):
    with pytest.raises(InvalidArgument):
        await client.build_settings("REGULAR", {"decay": 1.5})
    with pytest.raises(InvalidArgument):
        await client.build_settings("REGULAR", {"language": "PYTHON"})
    py = await client.build_settings("REGULAR", {"language": "PYTHON", "lookback": 20})
    assert py["lookback"] == 20 and "nanHandling" not in py and "testPeriod" not in py
    sup = await client.build_settings("SUPER", {})
    assert sup["selectionHandling"] == "POSITIVE" and sup["componentActivation"] == "IS"
    with pytest.raises(InvalidArgument):
        await client.build_settings("REGULAR", {"truncation": 2})


async def test_cancel_and_setting_options(client, fake):
    assert (await client.cancel_simulation("S9"))["cancelled"]
    assert fake.state.calls("DELETE", "/simulations/S9")
    opts = await client.setting_options()
    combos = {(r["region"], r["delay"]) for r in opts["instrument_options"]}
    assert combos == {("USA", 1), ("USA", 0), ("CHN", 1)}
    assert opts["platform_defaults"]["decay"] == 4
    assert fake.state.calls("OPTIONS", "/simulations")[0]["accept"] == "application/json;version=3.0"


def test_parse_setting_options_tolerates_other_shapes():
    assert parse_setting_options({})["instrument_options"] == []
    assert parse_setting_options({"actions": {"POST": {"settings": {"type": "nested object"}}}})["instrument_options"] == []


async def test_super_selection_uses_catalog_param_names(client, fake):
    res = await client.super_selection("rank(sharpe)", instrument_type="EQUITY", region="CHN", delay=1,
                                       selection_limit=50, selection_handling="POSITIVE", limit=5)
    q = fake.state.calls("GET", "/simulations/super-selection")[0]["query"]
    assert q["settings.region"] == "CHN" and q["settings.delay"] == "1" and q["limit"] == "5"
    assert "region" not in q and res["results"][0]["id"] == "A900"


# ------------------------------------------------------------------- alphas


async def test_check_alpha_polls_and_reports_failures(client, fake):
    pending = await client.check_alpha("A1", wait_seconds=0)
    assert pending["status"] == "PENDING" and pending["retry_after_seconds"] == 1
    done = await client.check_alpha("A1", wait_seconds=10, correlations=["prod"])
    assert done["status"] == "DONE" and done["failed"] == ["SELF_CORRELATION"] and not done["all_passed"]
    assert done["self_correlation_max"] == 0.82
    corr = done["correlations"][0]
    assert corr["type"] == "prod" and corr["max"] == 0.65
    assert [r["id"] for r in corr["top"]] == ["X2", "X4", "X1"]


async def test_correlation_failure_does_not_hide_checks(client, fake, monkeypatch):
    async def broken(alpha_id, kind, wait_seconds, top=5):
        raise BrainAPIError(412, "GET", f"/alphas/{alpha_id}/correlations/{kind}", "not available")
    monkeypatch.setattr(client, "correlation", broken)
    res = await client.check_alpha("A1", wait_seconds=10, correlations=["power-pool"])
    assert res["status"] == "DONE" and res["correlations"][0]["status"] == "ERROR"


async def test_check_alpha_rejects_unknown_correlation(client, fake):
    with pytest.raises(InvalidArgument):
        await client.check_alpha("A1", 0, correlations=["production"])


async def test_submit_dry_run_does_not_post(client, fake):
    res = await client.submit("A1", confirm=False, wait_seconds=10)
    assert res["dry_run"] and res["status"] == "DONE"
    assert not fake.state.calls("POST", r"/alphas/A1/submit")


async def test_submit_polls_until_final_and_never_posts_twice(client, fake):
    first = await client.submit("A1", confirm=True, wait_seconds=0)
    assert first["status"] == "PENDING"
    second = await client.submit("A1", confirm=True, wait_seconds=10)
    assert second["status"] == "SUBMITTED" and second["all_passed"]
    assert len(fake.state.calls("POST", r"/alphas/A1/submit")) == 1


async def test_submit_rejected(client, fake):
    fake.state.submit_fail = True
    res = await client.submit("A2", confirm=True, wait_seconds=10)
    assert res["status"] == "REJECTED" and res["failed"] == ["PROD_CORRELATION"]


async def test_recordset_polling_and_truncation(client, fake):
    assert (await client.recordset("A1", None, 0, 300))["recordsets"] == ["pnl", "yearly-stats"]
    pending = await client.recordset("A1", "pnl", 0, 300)
    assert pending["status"] == "PENDING"
    done = await client.recordset("A1", "pnl", 10, 300)
    assert done["status"] == "DONE" and done["total_rows"] == 500 and len(done["rows"]) == 300
    assert done["columns"] == ["date", "pnl", "equal-weight-pnl"] and "truncated" in done


async def test_list_alphas_pagination_and_compaction(client, fake):
    res = await client.list_alphas(stage="OS", status=None, alpha_type=None, limit=20, offset=0,
                                   order="-dateCreated", created_after="2026-01-01", created_before=None,
                                   submitted_after=None, submitted_before=None, hidden=False, full=False)
    assert res["count"] == 250 and res["returned"] == 20 and res["next_offset"] == 20
    assert "researchNotes" not in res["results"][0]
    q = fake.state.calls("GET", "/users/self/alphas")[0]
    assert q["accept"] == "application/json;version=4.0"
    assert q["query"]["dateCreated>"] == "2026-01-01" and q["query"]["hidden"] == "false"
    with pytest.raises(InvalidArgument):
        await client.list_alphas(stage=None, status=None, alpha_type=None, limit=10, offset=0, order=None,
                                 created_after="yesterday", created_before=None, submitted_after=None,
                                 submitted_before=None, hidden=None, full=False)


async def test_update_alpha_bulk_and_single(client, fake):
    await client.update_alpha(["A1", "A2"], {}, {"favorite": True})
    body = fake.state.calls("PATCH", "/alphas")[0]["body"]
    assert body == [{"id": "A1", "favorite": True}, {"id": "A2", "favorite": True}]
    res = await client.update_alpha(["A1"], {"name": None, "tags": ["x"]}, {})
    assert fake.state.calls("PATCH", "/alphas/A1")[0]["body"] == {"name": None, "tags": ["x"]}
    assert res["alpha"]["id"] == "A1"
    with pytest.raises(InvalidArgument):
        await client.update_alpha(["A1", "A2"], {"name": "x"}, {})
    with pytest.raises(InvalidArgument):
        await client.update_alpha(["A1"], {}, {})


async def test_alpha_performance(client, fake):
    res = await client.alpha_performance("A1", None)
    assert res["stats"]["after"]["sharpe"] == 1.6
    assert res["yearlyStats"]["before"]["columns"] == ["year"]


# --------------------------------------------------------------- data etc.


async def test_datasets_and_fields(client, fake):
    ds = await client.datasets({"instrumentType": "EQUITY", "region": "USA", "delay": 1, "universe": "TOP3000",
                                "search": None}, limit=20, offset=20)
    assert ds["count"] == 45 and ds["next_offset"] == 40 and "researchPapers" not in ds["results"][0]
    assert len(ds["results"][0]["description"]) <= 301
    q = fake.state.calls("GET", "/data-sets")[0]["query"]
    assert "theme" not in q and "search" not in q and q["offset"] == "20"
    fields = await client.datafields({"instrumentType": "EQUITY", "region": "USA", "delay": 1,
                                      "universe": "TOP3000", "dataset.id": "ds1"}, limit=100, offset=0)
    assert fields["returned"] == 100 and fields["has_more"]
    both = await client.search_data("assets", 5)
    assert both["datasets"][0]["id"] == "fundamental6" and both["fields"][0]["dataset"] == "fundamental6"


async def test_activity_kinds(client, fake):
    div = await client.activity("diversity", grouping="region,delay", start_date=None, end_date=None,
                                since=None, max_rows=10)
    assert div["query"] == {"grouping": "region,delay"}
    with pytest.raises(InvalidArgument):
        await client.activity("diversity", grouping="region;drop", start_date=None, end_date=None,
                              since=None, max_rows=10)
    pay = await client.activity("base-payment", grouping=None, start_date=None, end_date=None,
                                since=None, max_rows=10)
    assert pay["records"]["total_rows"] == 100 and len(pay["records"]["rows"]) == 10
    assert fake.state.calls("GET", "/users/self/activities/base-payment")[0]["accept"].endswith("3.0")
    sims = await client.activity("simulations", grouping=None, start_date=None, end_date=None,
                                 since="2026-01-01", max_rows=5)
    assert fake.state.calls("GET", "/users/self/activities/simulations")[0]["query"] == {"date>": "2026-01-01"}
    assert len(sims["records"]["rows"]) == 5


async def test_diversity_score_paginates_without_n_plus_one(client, fake):
    res = await client.diversity_score("2026-01-01T00:00:00Z", "2026-12-31T00:00:00Z")
    assert res["N"] == 250 and res["A"] == 125 and res["P"] == 3 and res["P_max"] == 4
    assert res["S_P"] == 0.75 and res["complete"]
    assert len(fake.state.calls("GET", "/users/self/alphas")) == 3
    assert not fake.state.calls("GET", r"/alphas/[^/]+")


async def test_leaderboard_competitions_events_messages_docs(client, fake):
    lb = await client.leaderboard("leader", limit=5, offset=0, order=None, aggregate=None, user=None, mine=True)
    assert lb["results"][0]["user"] == "U123"
    assert not fake.state.calls("GET", "/users/self")  # no extra user lookup via getUser
    comps = await client.competitions(competition_id=None, scope="active", limit=10, offset=0,
                                      include_agreement=False)
    assert "faq" not in comps["results"][0]
    assert "endDate!<" in fake.state.calls("GET", "/competitions")[0]["query"]
    ev = await client.events(limit=5, offset=0, order=None, event_type=None, language=None, upcoming_only=True)
    assert ev["results"][0]["id"] == "E1" and fake.state.calls("GET", "/events")[0]["query"]["order"] == "start"
    msgs = await client.messages(limit=5, offset=0, unread_only=True, message_type=None, order=None)
    assert "base64" not in msgs["results"][0]["description"] and "[image removed]" in msgs["results"][0]["description"]
    assert fake.state.calls("GET", "/users/self/messages")[0]["query"]["read"] == "false"
    docs = await client.documentation(None, 10)
    assert docs["results"][0]["pages"] == [{"id": "P1", "title": "Page 1"}]
    page = await client.documentation("P1", 10)
    assert page["content"][0] == {"type": "HEADING", "text": "Intro", "level": "H2"}


# ------------------------------------------------------------ auth / limits


async def test_401_refreshes_cookie_once_for_concurrent_requests(client, fake):
    await client.get_alpha("A1")
    assert fake.state.credd_calls == 1
    fake.state.valid_token = "tok-2"
    fake.state.cookie_version = 2
    results = await asyncio.gather(*[client.get_alpha(f"A{i}") for i in range(8)])
    assert all(r["id"] for r in results)
    assert fake.state.credd_calls == 2  # one refresh, not one per thread


async def test_429_is_retried_then_surfaced(client, fake):
    fake.state.rate_limit_next_get = 1
    t0 = time.monotonic()
    assert (await client.get_alpha("A1"))["id"] == "A1"  # retried after Retry-After
    assert time.monotonic() - t0 >= 0.8
    fake.state.rate_limit_next_get = 5
    with pytest.raises(BrainAPIError) as exc:
        await client.get_alpha("A2")
    assert exc.value.status == 429 and exc.value.retry_after == 1
    assert len(fake.state.calls("GET", "/alphas/A2")) == 3  # GETs: 3 attempts, then give up


async def test_writes_are_not_retried(client, fake):
    fake.state.sim_slots_full = True
    settings = await client.build_settings("REGULAR", {})
    await client.create_simulations([{"type": "REGULAR", "settings": settings, "regular": "rank(close)"}])
    assert len(fake.state.calls("POST", "/simulations")) == 1


async def test_network_errors_are_typed(fake):
    c = brain_client.BrainClient(fake.url)
    await c.get_alpha("A1")  # bootstrap the session against the fake
    c.base_url = "http://127.0.0.1:9"  # nothing listens here
    with pytest.raises(brain_client.BrainNetworkError):
        await c.get_alpha("A1")
    c._executor.shutdown(wait=False)


async def test_diversity_score_refilters_window(client, fake):
    fake.state.alphas[0]["dateSubmitted"] = "2025-06-01T00:00:00Z"
    res = await client.diversity_score("2026-01-01", "2026-12-31")
    assert res["filtered_out"] == 1 and res["N"] == 249 and "outside the window" in res["note"]


async def test_user_id_is_required_for_user_scoped_calls(client, fake, monkeypatch):
    async def no_user(refresh=False):
        return {"authenticated": True, "user_id": None}
    monkeypatch.setattr(client, "status", no_user)
    with pytest.raises(brain_client.BrainError):
        await client.leaderboard("leader", limit=5, offset=0, order=None, aggregate=None, user=None, mine=True)


async def test_status_reports_user(client, fake):
    st = await client.status()
    assert st == {"authenticated": True, "user_id": "U123", "token_expiry": 1900000000,
                  "permissions": ["TUTORIAL"]}

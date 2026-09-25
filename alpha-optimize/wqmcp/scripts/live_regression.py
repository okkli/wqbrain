#!/usr/bin/env python3
"""Live regression for wqmcp v2 against the real BRAIN platform.

Drives every tool through a real (in-memory) MCP session, exactly as an MCP
client would, using the login that credd provides. It also probes the items
API_AUDIT.md marks UNVERIFIABLE (things the catalog could not settle).

Needs: CREDD_URL (+ CREDD_TOKEN if credd requires it) and network access to
api.worldquantbrain.com (and support.worldquantbrain.com for --forum).

    python scripts/live_regression.py                # read-only (default)
    python scripts/live_regression.py --writes       # + 2 simulations, metadata round-trip
    python scripts/live_regression.py --forum        # + forum tools (Playwright)
    python scripts/live_regression.py --selection "<SuperAlpha selection expr>"

Safety: never submits an alpha (submit_alpha is only called with confirm=False).
--writes creates one single and one 2-alpha multi simulation (cheap FASTEXPR
expressions) and edits the name/tags/favorite of the alpha they produce, then
restores them. Output: a console table and a JSON report (--report).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Verdict = Tuple[str, str]  # (PASS|FAIL|WARN|INFO|SKIP, detail)


class Run:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
        self.http: List[Dict[str, Any]] = []
        self.session = None

    async def tool(self, _tool: str, **args: Any) -> Tuple[bool, Any]:
        res = await self.session.call_tool(_tool, args)
        text = res.content[0].text if res.content else ""
        if res.isError:
            return True, text
        try:
            return False, json.loads(text)
        except ValueError:
            return False, text

    def note(self, label: str, verdict: str, detail: str, audit: str = "") -> None:
        self.results.append({"check": label, "verdict": verdict, "detail": detail[:600], "seconds": 0,
                             "audit": audit})
        print(f"{verdict:5} {'':7} {label}{f' [{audit}]' if audit else ''}  {detail[:300]}", flush=True)

    async def check(self, label: str, fn: Callable[[], Awaitable[Any]],
                    expect: Optional[Callable[[Any], Verdict]] = None, audit: str = "") -> Any:
        t0 = time.monotonic()
        value: Any = None
        try:
            value = await fn()
            verdict, detail = expect(value) if expect else ("PASS", "")
        except Exception as exc:  # the harness must keep going
            verdict, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        elapsed = round(time.monotonic() - t0, 1)
        self.results.append({"check": label, "verdict": verdict, "detail": detail[:400],
                             "seconds": elapsed, "audit": audit})
        tag = f" [{audit}]" if audit else ""
        print(f"{verdict:5} {elapsed:6.1f}s  {label}{tag}  {detail[:160]}", flush=True)
        return value


def ok_tool(pred: Callable[[Any], bool] = lambda v: True, why: str = "") -> Callable[[Any], Verdict]:
    """Expectation for a (is_error, payload) tool result."""
    def _check(res: Tuple[bool, Any]) -> Verdict:
        is_error, payload = res
        if is_error:
            return "FAIL", str(payload)
        try:
            good = pred(payload)
        except Exception as exc:
            return "FAIL", f"unexpected payload ({exc}): {json.dumps(payload, ensure_ascii=False)[:200]}"
        return ("PASS", why) if good else ("FAIL", f"{why}: {json.dumps(payload, ensure_ascii=False)[:200]}")
    return _check


def status_of(res: Tuple[bool, Any]) -> str:
    return "" if res[0] else str(res[1].get("status", ""))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--writes", action="store_true", help="create simulations and round-trip alpha metadata")
    ap.add_argument("--forum", action="store_true", help="exercise the Playwright forum tools")
    ap.add_argument("--selection", help="a valid SuperAlpha selection expression for the super-selection probe")
    ap.add_argument("--region", default="USA")
    ap.add_argument("--universe", default="TOP3000")
    ap.add_argument("--wait", type=float, default=120, help="wait_seconds for long-running BRAIN jobs")
    ap.add_argument("--report", default="live_regression_report.json")
    args = ap.parse_args()

    if not os.environ.get("CREDD_URL"):
        print("CREDD_URL is not set (credd's address, e.g. http://127.0.0.1:8762).", file=sys.stderr)
        return 2

    import server  # reads WQMCP_*/CREDD_* from the environment
    from mcp.shared.memory import create_connected_server_and_client_session

    run = Run()
    brain = server.brain
    original_request = brain.request

    async def traced(method: str, path: str, **kw: Any):
        resp = await original_request(method, path, **kw)
        run.http.append({"method": method.upper(), "path": path, "status": resp.status_code,
                         "retry_after": resp.headers.get("Retry-After"),
                         "accept": kw.get("accept") or "(default)"})
        return resp

    brain.request = traced  # poll()/call() look the method up on the instance
    today = dt.date.today()
    d90 = (today - dt.timedelta(days=90)).isoformat()
    d30 = (today - dt.timedelta(days=30)).isoformat()
    ctx: Dict[str, Any] = {}

    async with create_connected_server_and_client_session(server.mcp._mcp_server) as session:
        run.session = session
        T = run.tool

        # ---- 0. login ------------------------------------------------------
        cookies = await run.check("credd provides cookies", brain.cookie_list,
                                  lambda c: ("PASS", f"{len(c)} cookies") if c else ("FAIL", "no cookies"))
        if not cookies:
            return finish(run, args.report)
        st = await run.check("brain_status(overview)", lambda: T("brain_status", overview=True),
                             ok_tool(lambda p: p["authenticated"], "authenticated"), "AUTH-13/17")
        if not st or st[0] or not st[1].get("authenticated"):
            return finish(run, args.report)

        # ---- 1. settings & data ---------------------------------------------
        await run.check("get_platform_setting_options", lambda: T("get_platform_setting_options"),
                        ok_tool(lambda p: p["total_combinations"] > 0, "combos parsed"), "SIM-13/14")
        options_calls = [h for h in run.http if h["method"] == "OPTIONS"]
        run.note("OPTIONS /simulations Accept fallback used", "INFO", f"{len(options_calls)} OPTIONS request(s): {options_calls}", "SIM-13")
        ds = await run.check("get_datasets page 1",
                             lambda: T("get_datasets", region=args.region, delay=1, universe=args.universe, limit=5),
                             ok_tool(lambda p: p["returned"] > 0 and (not p["has_more"] or p["next_offset"] == 5),
                                     "results + next_offset"), "DATA-2")
        if ds and not ds[0] and ds[1]["results"]:
            ctx["dataset"] = ds[1]["results"][0]["id"]
            await run.check("get_datasets page 2 differs",
                            lambda: T("get_datasets", region=args.region, delay=1, universe=args.universe,
                                      limit=5, offset=5),
                            ok_tool(lambda p: not p["results"] or p["results"][0]["id"] != ctx["dataset"],
                                    "offset honoured"), "DATA-2")
            df = await run.check("get_datafields(dataset)",
                                 lambda: T("get_datafields", region=args.region, delay=1, universe=args.universe,
                                           dataset_id=ctx["dataset"], limit=5),
                                 ok_tool(lambda p: p["returned"] > 0, "fields"), "DATA-1")
            if df and not df[0] and df[1]["results"]:
                fid = df[1]["results"][0]["id"]
                await run.check("get_datafields(field_id)", lambda: T("get_datafields", field_id=fid),
                                ok_tool(lambda p: p.get("id") == fid, "detail"))
        await run.check("get_datasets(search, include_fields)",
                        lambda: T("get_datasets", search="close", include_fields=True),
                        ok_tool(lambda p: p["datasets_total"] + p["fields_total"] > 0, "hits"), "DATA-11")
        await run.check("get_datasets(min_coverage, order)",
                        lambda: T("get_datasets", region=args.region, delay=1, universe=args.universe,
                                  min_coverage=0.8, order="-valueScore", limit=3),
                        ok_tool(lambda p: all((d.get("coverage") or 1) >= 0.8 for d in p["results"]),
                                "coverage filter honoured"), "DATA-7")
        await run.check("get_operators", lambda: T("get_operators"),
                        ok_tool(lambda p: p["count"] > 20, "operators"), "DATA-8")

        # ---- 2. alphas ------------------------------------------------------
        for stage in ("IS", "OS"):
            res = await run.check(f"list_alphas(stage={stage})", lambda s=stage: T("list_alphas", stage=s, limit=5),
                                  ok_tool(lambda p: p["count"] is not None, "count present"), "ALPHA-9/10")
            if res and not res[0] and res[1]["results"]:
                ctx[f"alpha_{stage}"] = res[1]["results"][0]["id"]
                ctx[f"count_{stage}"] = res[1]["count"]
        alpha = ctx.get("alpha_IS") or ctx.get("alpha_OS")
        if "count_IS" in ctx:
            async def date_probe():
                far = (today + dt.timedelta(days=3650)).isoformat()
                _, a = await T("list_alphas", stage="IS", limit=1, created_after=far)
                _, b = await T("list_alphas", stage="IS", limit=1, hidden=True)
                return a, b
            await run.check("undocumented list filters honoured?", date_probe,
                            lambda ab: ("INFO", f"created_after=+10y -> count {ab[0].get('count')} "
                                                f"(unfiltered {ctx['count_IS']}); hidden=true -> count "
                                                f"{ab[1].get('count')}. A smaller count means the filter works."),
                            "ALPHA-12")
        if alpha:
            await run.check("get_alpha", lambda: T("get_alpha", alpha_id=alpha),
                            ok_tool(lambda p: p["id"] == alpha and "settings" in p, "summary"), "SIM-7")
            await run.check("get_alpha(full)", lambda: T("get_alpha", alpha_id=alpha, full=True),
                            ok_tool(lambda p: "is" in p, "raw object"), "ALPHA-10")
            await run.check("get_alpha_recordset(list)", lambda: T("get_alpha_recordset", alpha_id=alpha),
                            ok_tool(lambda p: bool(p.get("recordsets")), "names"), "ANLY-7")
            for rs in ("pnl", "yearly-stats"):
                await run.check(f"get_alpha_recordset({rs})",
                                lambda r=rs: T("get_alpha_recordset", alpha_id=alpha, recordset=r,
                                               wait_seconds=args.wait, max_rows=5),
                                ok_tool(lambda p: p["status"] in ("DONE", "PENDING", "EMPTY"), "status"), "ANLY-2")
            await run.check("check_alpha(+self corr)",
                            lambda: T("check_alpha", alpha_id=alpha, correlations=["self"], wait_seconds=args.wait),
                            ok_tool(lambda p: p["status"] in ("DONE", "PENDING"), "status"), "ALPHA-1")
            before = len([h for h in run.http if h["method"] == "POST"])
            await run.check("submit_alpha dry run (no POST)",
                            lambda: T("submit_alpha", alpha_id=alpha, wait_seconds=args.wait),
                            ok_tool(lambda p: p["dry_run"] and len([h for h in run.http if h["method"] == "POST"]) == before,
                                    "no POST sent"), "CRIT-2")
            await run.check("get_alpha_performance", lambda: T("get_alpha_performance", alpha_id=alpha,
                                                               wait_seconds=args.wait),
                            ok_tool(lambda p: isinstance(p, dict), "before/after"), "ANLY-6")
        else:
            run.note("alpha checks", "SKIP", "no alphas on this account", "")

        # ---- 3. activity & community ------------------------------------------
        for kind, extra in (("list", {}), ("diversity", {"grouping": "region,delay"}),
                            ("pyramid-alphas", {"start_date": d90, "end_date": today.isoformat()}),
                            ("pyramid-multipliers", {}), ("base-payment", {"max_rows": 5}),
                            ("other-payment", {"max_rows": 5}), ("simulations", {"since": d30, "max_rows": 5}),
                            ("submissions", {"since": d30, "max_rows": 5}), ("value-factor", {}),
                            ("diversity-score", {"start_date": d90, "end_date": today.isoformat()})):
            await run.check(f"get_activity({kind})", lambda k=kind, e=extra: T("get_activity", kind=k, **e),
                            ok_tool(), "AUTH-1/9/10")
        await run.check("get_leaderboard(mine)", lambda: T("get_leaderboard"), ok_tool(), "AUTH-11/19")
        await run.check("get_leaderboard(browse)", lambda: T("get_leaderboard", mine=False, limit=3),
                        ok_tool(lambda p: p["returned"] <= 3, "paged"))
        comps = await run.check("get_competitions(active)", lambda: T("get_competitions"), ok_tool(), "MISC-3/9")
        await run.check("get_competitions(mine)", lambda: T("get_competitions", scope="mine"), ok_tool(), "MISC-5")
        if comps and not comps[0] and comps[1]["results"]:
            cid = comps[1]["results"][0]["id"]
            await run.check("competition agreement endpoint exists?",
                            lambda: T("get_competitions", competition_id=cid, include_agreement=True),
                            lambda r: ("FAIL", str(r[1])) if r[0] else
                            ("INFO", "agreement endpoint returned data" if r[1].get("agreement") is not None
                             else "no /agreement (404): consider dropping include_agreement"), "MISC-4")
        await run.check("get_events", lambda: T("get_events", upcoming_only=False, limit=3), ok_tool(), "MISC-3")
        await run.check("get_messages", lambda: T("get_messages", limit=3), ok_tool(), "MISC-1/2")
        docs = await run.check("get_documentation", lambda: T("get_documentation", limit=5),
                               ok_tool(lambda p: p["returned"] > 0, "tutorials"), "MISC-8")
        if docs and not docs[0]:
            page = next((pg["id"] for t in docs[1]["results"] for pg in t["pages"]), None)
            if page:
                await run.check("get_documentation(page)", lambda: T("get_documentation", page_id=page),
                                ok_tool(lambda p: bool(p.get("content")), "blocks"))

        # ---- 4. probes for UNVERIFIABLE items ---------------------------------
        if args.selection:
            async def selection_probe():
                out = {}
                for label, params in (("settings.* (v2)", {"settings.region": "CHN"}), ("flat (v1)", {"region": "CHN"})):
                    resp = await brain.request("GET", "/simulations/super-selection",
                                               params={"selection": args.selection, "limit": 5, **params})
                    body = resp.json() if resp.status_code == 200 else {}
                    regions = {((a.get("settings") or {}).get("region")) for a in body.get("results") or []}
                    out[label] = f"HTTP {resp.status_code}, regions={sorted(r for r in regions if r)}"
                return out
            await run.check("super-selection param names", selection_probe,
                            lambda o: ("INFO", f"{o} - the style that yields only CHN is the one BRAIN honours"),
                            "SIM-1")
        else:
            run.note("super-selection param names", "SKIP", "pass --selection '<expr>' to probe", "SIM-1")

        # ---- 5. writes (opt-in) ------------------------------------------------
        if args.writes:
            single = await run.check("create_simulation(single)",
                                     lambda: T("create_simulation", expressions=["rank(-returns)"],
                                               region=args.region, universe=args.universe, delay=1),
                                     ok_tool(lambda p: p["status"] in ("SUBMITTED", "RATE_LIMITED"), "submitted"),
                                     "SIM-4/10")
            multi = await run.check("create_simulation(multi x2)",
                                    lambda: T("create_simulation", expressions=["rank(-returns)", "rank(volume)"],
                                              region=args.region, universe=args.universe, delay=1),
                                    ok_tool(lambda p: p["status"] in ("SUBMITTED", "RATE_LIMITED"), "submitted"),
                                    "SIM-9")
            ids = [r[1]["simulation_id"] for r in (single, multi) if r and not r[0] and r[1].get("simulation_id")]
            if ids:
                sims = await run.check("get_simulation(wait)",
                                       lambda: T("get_simulation", simulation_ids=ids, wait_seconds=min(args.wait * 2, 300)),
                                       ok_tool(lambda p: all(s["status"] not in ("ERROR",) for s in p["simulations"]),
                                               "finished or still running"), "SIM-3/5/6")
                finals = [h for h in run.http if h["path"].startswith("/simulations/") and h["method"] == "GET"
                          and h["status"] == 200]
                ra_values = sorted({str(h["retry_after"]) for h in finals})
                run.note("Retry-After values seen on /simulations polls", "INFO", f"{ra_values} ('None' = header absent on the finished response; "
                                              "'0' on a finished response would mean in_progress() must use ra>0)", "SIM-17")
                new_alpha = None
                if sims and not sims[0]:
                    for s in sims[1]["simulations"]:
                        new_alpha = new_alpha or s.get("alpha_id") or next(
                            (c.get("alpha_id") for c in s.get("children", []) if c.get("alpha_id")), None)
                if new_alpha:
                    async def metadata_roundtrip():
                        e1, _ = await T("update_alpha", alpha_ids=[new_alpha], name="wqmcp-live-test",
                                        tags=["wqmcp_live"], favorite=True)
                        _, raw = await T("get_alpha", alpha_id=new_alpha, full=True)
                        tags = raw.get("tags")
                        e2, _ = await T("update_alpha", alpha_ids=[new_alpha], name="", tags=[], favorite=False)
                        return e1, e2, raw.get("name"), tags, raw.get("favorite")
                    await run.check("update_alpha round-trip", metadata_roundtrip,
                                    lambda r: ("FAIL", f"errors: {r[0]}, {r[1]}") if r[0] or r[1] else
                                    ("PASS" if r[2] == "wqmcp-live-test" and r[4] else "WARN",
                                     f"name={r[2]!r} favorite={r[4]} tags as stored={r[3]!r}"), "ALPHA-13/14")
                    await run.check("check_alpha(new alpha, prod+self)",
                                    lambda: T("check_alpha", alpha_id=new_alpha, correlations=["self", "prod"],
                                              wait_seconds=args.wait),
                                    ok_tool(lambda p: p["status"] in ("DONE", "PENDING"), "status"), "ALPHA-5/7")
        else:
            run.note("write checks", "SKIP", "pass --writes", "SIM-17, ALPHA-13")

        # ---- 6. forum (opt-in) --------------------------------------------------
        if args.forum:
            await run.check("get_glossary_terms", lambda: T("get_glossary_terms"),
                            ok_tool(lambda p: p["count"] > 10, "terms"), "FORUM-11")
            found = await run.check("search_forum_posts", lambda: T("search_forum_posts", query="sharpe", max_results=5),
                                    ok_tool(lambda p: p["count"] > 0 and p.get("signed_in", True), "results"), "FORUM-1/4")
            if found and not found[0] and found[1]["results"]:
                url = found[1]["results"][0]["url"]
                await run.check("read_forum_post", lambda: T("read_forum_post", post=url, max_comments=10),
                                ok_tool(lambda p: bool(p["post"]["body"]), "body"), "FORUM-2/5")
        else:
            run.note("forum checks", "SKIP", "pass --forum", "FORUM-1")

    return finish(run, args.report)


def finish(run: Run, report: str) -> int:
    counts: Dict[str, int] = {}
    for r in run.results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    accept_versions = sorted({(h["method"], h["path"].split("/")[1], h["status"]) for h in run.http})
    Path(report).write_text(json.dumps({"summary": counts, "results": run.results,
                                        "http_status_by_endpoint": accept_versions,
                                        "http_calls": len(run.http)}, ensure_ascii=False, indent=1))
    print(f"\nsummary: {counts}   http calls: {len(run.http)}   report: {report}")
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""Live regression for the wqmcp server (platform_functions.py) against real BRAIN.

Calls the tools through a real in-memory MCP session, exactly as an MCP client
would, using the login credd provides, and probes what the API catalog could
not settle (see API_AUDIT.md: SIM-1, SIM-17, ALPHA-12/13, MISC-4, FORUM-1).

Needs CREDD_URL (+ CREDD_TOKEN if credd requires it) and network access to
api.worldquantbrain.com (support.worldquantbrain.com for --forum).

    python scripts/live_regression.py                # read-only (default)
    python scripts/live_regression.py --writes       # + QUICK simulations, metadata round-trip
    python scripts/live_regression.py --raa          # + one RAA simulation (uses 4 slots)
    python scripts/live_regression.py --forum        # + forum tools (Playwright)
    python scripts/live_regression.py --prodmemo     # + ProdMemo status (needs its database)
    python scripts/live_regression.py --selection "<SuperAlpha selection expr>"
    python scripts/live_regression.py --accept-versions   # same run with versioned Accept headers

Never submits an alpha. Exit code 1 if any check FAILs; report in --report.
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

    async def tool(self, _tool: str, **args: Any) -> Any:
        res = await self.session.call_tool(_tool, args)
        text = res.content[0].text if res.content else ""
        if res.isError:
            return {"error": text}
        try:
            return json.loads(text)
        except ValueError:
            return {"raw": text}

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
            verdict, detail = expect(value) if expect else ok()(value)
        except Exception as exc:  # keep going
            verdict, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        elapsed = round(time.monotonic() - t0, 1)
        self.results.append({"check": label, "verdict": verdict, "detail": detail[:400],
                             "seconds": elapsed, "audit": audit})
        print(f"{verdict:5} {elapsed:6.1f}s  {label}{f' [{audit}]' if audit else ''}  {detail[:160]}", flush=True)
        return value


def ok(pred: Callable[[Any], bool] = lambda v: True, why: str = "") -> Callable[[Any], Verdict]:
    """A tool result passes when it is not an {"error": ...} payload and pred holds."""
    def _check(payload: Any) -> Verdict:
        if isinstance(payload, dict) and payload.get("error") and "status" not in payload:
            return "FAIL", str(payload["error"])
        try:
            good = pred(payload)
        except Exception as exc:
            return "FAIL", f"unexpected payload ({exc}): {json.dumps(payload, ensure_ascii=False)[:200]}"
        return ("PASS", why) if good else ("FAIL", f"{why}: {json.dumps(payload, ensure_ascii=False)[:200]}")
    return _check


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--writes", action="store_true")
    ap.add_argument("--raa", action="store_true")
    ap.add_argument("--forum", action="store_true")
    ap.add_argument("--prodmemo", action="store_true")
    ap.add_argument("--accept-versions", action="store_true", help="send the catalog's versioned Accept headers")
    ap.add_argument("--selection")
    ap.add_argument("--region", default="USA")
    ap.add_argument("--universe", default="TOP3000")
    ap.add_argument("--wait", type=float, default=120)
    ap.add_argument("--report", default="live_regression_report.json")
    args = ap.parse_args()
    if not os.environ.get("CREDD_URL"):
        print("CREDD_URL is not set (credd's address, e.g. http://127.0.0.1:8762).", file=sys.stderr)
        return 2
    if args.accept_versions:
        os.environ["WQMCP_ACCEPT_VERSIONS"] = "1"

    import platform_functions as pf
    from mcp.shared.memory import create_connected_server_and_client_session

    run = Run()
    brain = pf.brain_client
    original = brain._request

    async def traced(method: str, url: str, **kw: Any):
        resp = await original(method, url, **kw)
        run.http.append({"method": method.upper(), "path": url.replace(brain.base_url, ""),
                         "status": resp.status_code, "retry_after": resp.headers.get("Retry-After")})
        return resp

    brain._request = traced
    today = dt.date.today()
    d90 = (today - dt.timedelta(days=90)).isoformat()
    ctx: Dict[str, Any] = {}

    async with create_connected_server_and_client_session(pf.mcp._mcp_server) as session:
        run.session = session
        T = run.tool
        run.note("versioned Accept headers", "INFO", "on" if pf.SEND_ACCEPT_VERSIONS else "off (default)", "AUTH-9")

        auth = await run.check("authenticate", lambda: T("authenticate"),
                               ok(lambda p: p.get("status") == "authenticated", "authenticated"), "AUTH-17")
        if not isinstance(auth, dict) or auth.get("status") != "authenticated":
            return finish(run, args.report)

        await run.check("get_platform_setting_options", lambda: T("get_platform_setting_options"),
                        ok(lambda p: p["total_combinations"] > 0, "combos parsed"), "SIM-13")
        ds = await run.check("get_datasets(limit=5)",
                             lambda: T("get_datasets", region=args.region, universe=args.universe, limit=5),
                             ok(lambda p: 0 < len(p["results"]) <= 5, "paged"), "DATA-2")
        if isinstance(ds, dict) and ds.get("results"):
            ctx["dataset"] = ds["results"][0]["id"]
            await run.check("get_datasets(offset=5) differs",
                            lambda: T("get_datasets", region=args.region, universe=args.universe, limit=5, offset=5),
                            ok(lambda p: not p["results"] or p["results"][0]["id"] != ctx["dataset"], "offset honoured"),
                            "DATA-2")
            await run.check("get_datafields(dataset, limit=5, offset=5)",
                            lambda: T("get_datafields", region=args.region, universe=args.universe,
                                      dataset_id=ctx["dataset"], limit=5, offset=5),
                            ok(lambda p: len(p["results"]) <= 5, "paged"), "DATA-1")
        await run.check("get_operators", lambda: T("get_operators"),
                        ok(lambda p: p["count"] > 20, "operators"), "DATA-9")

        for stage in ("IS", "OS"):
            res = await run.check(f"get_user_alphas(stage={stage})",
                                  lambda s=stage: T("get_user_alphas", stage=s, limit=5),
                                  ok(lambda p: "results" in p, "results"), "ALPHA-9")
            if isinstance(res, dict) and res.get("results"):
                ctx[stage] = res["results"][0]["id"]
                ctx[f"count_{stage}"] = res.get("count")
        if "count_IS" in ctx:
            far = (today + dt.timedelta(days=3650)).isoformat()
            async def probe():
                a = await T("get_user_alphas", stage="IS", limit=1, start_date=far)
                b = await T("get_user_alphas", stage="IS", limit=1, hidden=True)
                return a, b
            await run.check("undocumented list filters honoured?", probe,
                            lambda ab: ("INFO", f"start_date=+10y -> count {ab[0].get('count')} vs unfiltered "
                                                f"{ctx['count_IS']}; hidden=true -> {ab[1].get('count')} "
                                                "(a smaller count means the filter works)"), "ALPHA-12")
        alpha = ctx.get("IS") or ctx.get("OS")
        if alpha:
            await run.check("get_alpha_details", lambda: T("get_alpha_details", alpha_id=alpha),
                            ok(lambda p: p.get("id") == alpha, "alpha"))
            await run.check("get_record_sets", lambda: T("get_record_sets", alpha_id=alpha),
                            ok(lambda p: p.get("results") or p.get("status") == "PENDING", "list"), "ANLY-7")
            await run.check("get_record_set_data(pnl)", lambda: T("get_record_set_data", alpha_id=alpha,
                                                                  record_set_name="pnl"),
                            ok(lambda p: p.get("records") or p.get("status") == "PENDING", "data or PENDING"), "ANLY-2")
            await run.check("get_alpha_yearly_stats", lambda: T("get_alpha_yearly_stats", alpha_id=alpha),
                            ok(lambda p: isinstance(p, dict), "dict ({} = still computing)"), "ANLY-2")
            await run.check("check_correlation(both)",
                            lambda: T("check_correlation", alpha_id=alpha, max_wait=args.wait),
                            ok(lambda p: p["status"] in ("DONE", "PENDING", "ERROR"), "status"), "ALPHA-5")
            await run.check("get_submission_check", lambda: T("get_submission_check", alpha_id=alpha,
                                                              max_wait=args.wait),
                            ok(lambda p: p["status"] in ("DONE", "PENDING"), "status"), "ALPHA-1")
            await run.check("performance_comparison", lambda: T("performance_comparison", alpha_id=alpha),
                            ok(lambda p: isinstance(p, dict), "before/after"), "ANLY-6")

        for label, tool, kw in (
                ("get_user_activities(diversity)", "get_user_activities", {"user_id": "self", "grouping": "region,delay"}),
                ("get_pyramid_multipliers", "get_pyramid_multipliers", {}),
                ("get_pyramid_alphas", "get_pyramid_alphas", {"start_date": d90, "end_date": today.isoformat()}),
                ("get_daily_and_quarterly_payment", "get_daily_and_quarterly_payment", {}),
                ("value_factor_trendScore", "value_factor_trendScore", {"start_date": d90, "end_date": today.isoformat()}),
                ("get_leaderboard", "get_leaderboard", {}),
                ("get_user_profile(self)", "get_user_profile", {}),
                ("get_user_competitions", "get_user_competitions", {}),
                ("get_events", "get_events", {}),
                ("get_messages", "get_messages", {"limit": 3}),
                ("get_documentations", "get_documentations", {})):
            res = await run.check(label, lambda t=tool, k=kw: T(t, **k), ok(), "AUTH/MISC")
            if label == "get_daily_and_quarterly_payment" and isinstance(res, dict):
                errs = [k for k, v in res.items() if isinstance(v, dict) and v.get("error")]
                if errs:
                    run.note("payment halves", "WARN", f"errors in {errs}: {[res[k]['error'][:120] for k in errs]}", "AUTH-10")
            if label == "get_user_competitions" and isinstance(res, dict) and res.get("results"):
                ctx["competition"] = res["results"][0]["id"]
            if label == "get_documentations" and isinstance(res, dict):
                ctx["page"] = next((p["id"] for t in res.get("results") or [] for p in t.get("pages") or []), None)
        if ctx.get("competition"):
            await run.check("get_competition_agreement exists?",
                            lambda: T("get_competition_agreement", competition_id=ctx["competition"]),
                            lambda p: ("INFO", "agreement endpoint returned data" if not p.get("error")
                                       else f"no agreement: {p['error'][:120]}"), "MISC-4")
        if ctx.get("page"):
            await run.check("get_documentation_page", lambda: T("get_documentation_page", page_id=ctx["page"]),
                            ok(lambda p: p.get("content"), "content"))

        if args.selection:
            async def selection_probe():
                out = {}
                for label, params in (("settings.* only", {"settings.region": "CHN"}), ("flat only", {"region": "CHN"})):
                    resp = await original("get", f"{brain.base_url}/simulations/super-selection",
                                          params={"selection": args.selection, "limit": 5, **params})
                    body = resp.json() if resp.status_code == 200 else {}
                    regions = sorted({(a.get("settings") or {}).get("region") for a in body.get("results") or []} - {None})
                    out[label] = f"HTTP {resp.status_code}, regions={regions}"
                return out
            await run.check("super-selection param names", selection_probe,
                            lambda o: ("INFO", f"{o} — the style returning only CHN is honoured"), "SIM-1")

        if args.writes:
            single = await run.check("create_simulation(QUICK)",
                                     lambda: T("create_simulation", regular="rank(-returns)", region=args.region,
                                               universe=args.universe, decay=4, neutralization="SUBINDUSTRY",
                                               truncation=0.08, simulation_mode="QUICK"),
                                     ok(lambda p: p["status"] in ("SUBMITTED", "RATE_LIMITED"), "submitted"), "SIM-4")
            multi = await run.check("create_multi_simulation(per_alpha_settings)",
                                    lambda: T("create_multi_simulation", alpha_expressions=["rank(-returns)"],
                                              per_alpha_settings=[{"decay": 2}, {"decay": 6}], region=args.region,
                                              universe=args.universe, simulation_mode="QUICK"),
                                    ok(lambda p: p["status"] in ("SUBMITTED", "RATE_LIMITED"), "submitted"), "SIM-9")
            for label, sub in (("single", single), ("multi", multi)):
                if isinstance(sub, dict) and sub.get("progress_url"):
                    res = await run.check(f"check_simulation_progress({label})",
                                          lambda u=sub["progress_url"]: T("check_simulation_progress", progress_url=u,
                                                                          wait_seconds=min(args.wait, 120)),
                                          ok(lambda p: p["status"] in ("COMPLETE", "RUNNING", "WARNING"), "status"),
                                          "SIM-3/5/6")
                    if label == "single" and isinstance(res, dict) and (res.get("alpha") or {}).get("id"):
                        ctx["new_alpha"] = res["alpha"]["id"]
            ra = sorted({str(h["retry_after"]) for h in run.http if h["path"].startswith("/simulations/") and h["status"] == 200})
            run.note("Retry-After values on /simulations polls", "INFO",
                     f"{ra} ('None' = absent on finished responses; '0' would mean presence alone is wrong)", "SIM-17")
            if ctx.get("new_alpha"):
                async def roundtrip():
                    a = await T("set_alpha_properties", alpha_id=ctx["new_alpha"], name="wqmcp-live-test",
                                tags=["wqmcp_live"])
                    stored = await T("get_alpha_details", alpha_id=ctx["new_alpha"])
                    await T("set_alpha_properties", alpha_id=ctx["new_alpha"], name="", tags=[])
                    return a, stored
                await run.check("set_alpha_properties round-trip", roundtrip,
                                lambda r: ("FAIL", str(r[0])) if r[0].get("error") else
                                ("INFO", f"name={r[1].get('name')!r} tags as stored={r[1].get('tags')!r}"), "ALPHA-13")
        if args.raa:
            raa = await run.check("create_raa_simulation", lambda: T("create_raa_simulation", regular="rank(-returns)",
                                                                     simulation_mode="QUICK"),
                                  ok(lambda p: p["status"] in ("SUBMITTED", "RATE_LIMITED"), "submitted"))
            if isinstance(raa, dict) and raa.get("progress_url"):
                await run.check("check_simulation_progress(RAA)",
                                lambda: T("check_simulation_progress", progress_url=raa["progress_url"],
                                          wait_seconds=min(args.wait, 120)),
                                ok(lambda p: p["status"] in ("COMPLETE", "RUNNING"), "status"))
        if args.forum:
            await run.check("get_glossary_terms", lambda: T("get_glossary_terms"),
                            ok(lambda p: isinstance(p, list) and len(p) > 10, "terms"), "FORUM-11")
            found = await run.check("search_forum_posts", lambda: T("search_forum_posts", search_query="sharpe", max_results=5),
                                    ok(lambda p: p["count"] > 0 and p.get("signed_in", True), "results"), "FORUM-1/4")
            if isinstance(found, dict) and found.get("results"):
                await run.check("read_forum_post", lambda: T("read_forum_post", article_id=found["results"][0]["url"]),
                                ok(lambda p: bool(p["post"]["body"]), "body"), "FORUM-2/5")
        if args.prodmemo:
            await run.check("prodmemo_stats", lambda: T("prodmemo_stats"), ok())
            await run.check("prodmemo_sync_status", lambda: T("prodmemo_sync_status"), ok())

    return finish(run, args.report)


def finish(run: Run, report: str) -> int:
    counts: Dict[str, int] = {}
    for r in run.results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    Path(report).write_text(json.dumps({"summary": counts, "results": run.results, "http": run.http},
                                       ensure_ascii=False, indent=1))
    print(f"\nsummary: {counts}   http calls: {len(run.http)}   report: {report}")
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

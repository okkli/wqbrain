"""wqmcp — MCP server for the WorldQuant BRAIN platform.

Tools are thin: validation and HTTP live in brain_client.BrainClient, forum
scraping in forum_functions.ForumClient. A failing tool raises, so MCP clients
receive ``isError: true`` with a single readable message.

Run: ``python server.py`` (or the legacy ``python platform_functions.py``).
Environment: WQMCP_HOST (default 127.0.0.1), WQMCP_PORT (8761), WQMCP_TRANSPORT
(streamable-http | stdio | sse), WQMCP_ALLOWED_HOSTS, CREDD_URL, CREDD_TOKEN —
see README.md for the full list.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Annotated, Any, Dict, List, Literal, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field

from brain_client import ACTIVITY_KINDS, RECORDSET_TYPES, BrainClient, InvalidArgument

logger = logging.getLogger("wqmcp")

HOST = os.environ.get("WQMCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("WQMCP_PORT", "8761"))
# Kill switches: READ_ONLY blocks every write tool, ALLOW_SUBMIT=0 blocks real submissions.
READ_ONLY = os.environ.get("WQMCP_READ_ONLY", "0") == "1"
ALLOW_SUBMIT = os.environ.get("WQMCP_ALLOW_SUBMIT", "1") != "0"


def _require_writes(what: str) -> None:
    if READ_ONLY:
        raise InvalidArgument(f"{what} is disabled: this server runs with WQMCP_READ_ONLY=1")


def _transport_security() -> Optional[TransportSecuritySettings]:
    extra = [h.strip() for h in os.environ.get("WQMCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    if not extra:
        if HOST not in ("127.0.0.1", "localhost", "::1"):
            logger.warning("WQMCP_HOST=%s without WQMCP_ALLOWED_HOSTS: no DNS-rebinding protection and "
                           "no authentication. Put an authenticating reverse proxy in front and set "
                           "WQMCP_ALLOWED_HOSTS to the host names it forwards.", HOST)
        return None  # FastMCP's default: DNS-rebinding protection for loopback binds
    return TransportSecuritySettings(
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", *extra],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
                         *[f"https://{h}" for h in extra], *[f"http://{h}" for h in extra]],
    )


mcp = FastMCP(
    "brain-platform-mcp",
    instructions=(
        "WorldQuant BRAIN research tools. Typical loop: get_platform_setting_options -> "
        "get_datasets/get_datafields/get_operators -> create_simulation -> get_simulation "
        "(poll with wait_seconds) -> check_alpha -> submit_alpha(confirm=True). "
        "Long-running BRAIN jobs return status RUNNING/PENDING with retry_after_seconds; "
        "call the same tool again instead of waiting idle."
    ),
    host=HOST,
    port=PORT,
    transport_security=_transport_security(),
)

brain = BrainClient()
_forum = None


def _forum_client():
    global _forum
    if _forum is None:
        from forum_functions import ForumClient  # lazy: forum deps are optional at startup
        _forum = ForumClient(brain.cookie_list)
    return _forum


READ = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
WRITE_IDEMPOTENT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True,
                                   openWorldHint=True)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True)

Region = Annotated[Optional[str], Field(description="e.g. USA, CHN, EUR, ASI, GLB; valid combos: get_platform_setting_options")]
Delay = Annotated[Optional[int], Field(description="0 or 1")]
Universe = Annotated[Optional[str], Field(description="e.g. TOP3000")]
Wait = Annotated[float, Field(ge=0, le=300, description="Seconds to keep polling BRAIN before answering (0 = check once)")]
Limit = Annotated[int, Field(ge=1, le=100)]
Offset = Annotated[int, Field(ge=0)]


def _none_if_blank(value: Optional[str]) -> Optional[str]:
    return value if value not in ("",) else None


def _obj(value: Any) -> Dict[str, Any]:
    """Tools are typed -> Dict; wrap an empty (None) or list body from BRAIN."""
    return value if isinstance(value, dict) else {"result": value}


# ============================================================ account


@mcp.tool(annotations=READ)
async def brain_status(refresh: bool = False, overview: bool = False) -> Dict[str, Any]:
    """Check the BRAIN login (managed by credd) and optionally return an account overview.

    refresh=True force-pulls fresh cookies from credd first. overview=True adds alpha counts
    (unsubmitted/active/decommissioned), consultant summary and unread-message counts.
    """
    status = await brain.status(refresh=refresh)
    if overview and status.get("authenticated"):
        for key, fn in (("alphas", brain.alpha_summary_counts), ("consultant", brain.consultant_summary),
                        ("messages", brain.message_summary)):
            try:
                status[key] = await fn()
            except Exception as exc:  # overview is best effort
                status[key] = {"error": str(exc)}
    return status


# ========================================================== simulation


@mcp.tool(annotations=WRITE)
async def create_simulation(
    expressions: Annotated[Optional[List[str]], Field(description="REGULAR: 1-10 alpha expressions (2+ = one multi-simulation). For language=PYTHON each item is Python source.")] = None,
    type: Literal["REGULAR", "SUPER"] = "REGULAR",
    combo: Annotated[Optional[str], Field(description="SUPER only: combo expression")] = None,
    selection: Annotated[Optional[str], Field(description="SUPER only: selection expression")] = None,
    region: Region = None,
    universe: Universe = None,
    delay: Delay = None,
    decay: Annotated[Optional[int], Field(ge=0, le=512)] = None,
    neutralization: Annotated[Optional[str], Field(description="e.g. NONE, MARKET, SECTOR, INDUSTRY, SUBINDUSTRY")] = None,
    truncation: Annotated[Optional[float], Field(ge=0, le=1)] = None,
    pasteurization: Optional[Literal["ON", "OFF"]] = None,
    nan_handling: Optional[Literal["ON", "OFF"]] = None,
    unit_handling: Optional[Literal["VERIFY"]] = None,
    max_trade: Optional[Literal["ON", "OFF"]] = None,
    max_position: Optional[Literal["ON", "OFF"]] = None,
    test_period: Annotated[Optional[str], Field(description="ISO duration P0Y0M .. P6Y0M0D")] = None,
    language: Optional[Literal["FASTEXPR", "PYTHON"]] = None,
    visualization: Optional[bool] = None,
    lookback: Annotated[Optional[int], Field(ge=0, le=1024, description="Required for PYTHON")] = None,
    selection_handling: Optional[Literal["POSITIVE", "NON_ZERO", "NON_NAN"]] = None,
    selection_limit: Annotated[Optional[int], Field(ge=10, le=1000)] = None,
    component_activation: Optional[Literal["IS", "OS"]] = None,
    instrument_type: Literal["EQUITY"] = "EQUITY",
) -> Dict[str, Any]:
    """Submit a simulation (returns immediately; typically 1-10 min to finish).

    Settings you omit come from your saved BRAIN simulation defaults (the same ones the
    website uses); the settings actually sent are echoed in settings_used. Poll with
    get_simulation(simulation_ids=[simulation_id], wait_seconds=60). RATE_LIMITED means the
    account's concurrent simulation slots are full.
    """
    overrides = {
        "instrumentType": instrument_type, "region": region, "universe": universe, "delay": delay,
        "decay": decay, "neutralization": neutralization, "truncation": truncation,
        "pasteurization": pasteurization, "nanHandling": nan_handling, "unitHandling": unit_handling,
        "maxTrade": max_trade, "maxPosition": max_position, "testPeriod": test_period,
        "language": language, "visualization": visualization, "lookback": lookback,
        "selectionHandling": selection_handling, "selectionLimit": selection_limit,
        "componentActivation": component_activation,
    }
    _require_writes("create_simulation")
    settings = await brain.build_settings(type, overrides)
    if type == "SUPER":
        if not combo or not selection:
            raise InvalidArgument("SUPER simulations need both combo and selection")
        if expressions:
            raise InvalidArgument("expressions are for REGULAR simulations; SUPER uses combo + selection")
        items = [{"type": "SUPER", "settings": settings, "combo": combo, "selection": selection}]
    else:
        exprs = [e for e in (expressions or []) if isinstance(e, str) and e.strip()]
        if not exprs:
            raise InvalidArgument("expressions must contain at least one non-empty alpha expression")
        if len(exprs) > 10:
            raise InvalidArgument("at most 10 expressions per multi-simulation")
        items = [{"type": "REGULAR", "settings": settings, "regular": e} for e in exprs]
    result = await brain.create_simulations(items)
    result["settings_used"] = settings
    if result.get("status") == "SUBMITTED":
        result["next"] = f"get_simulation(simulation_ids=['{result['simulation_id']}'], wait_seconds=60)"
    return result


@mcp.tool(annotations=READ)
async def get_simulation(
    simulation_ids: Annotated[Optional[List[str]], Field(description="Simulation ids (or their https://api.worldquantbrain.com/simulations/<id> URLs). Omit to list simulations created by this server recently.")] = None,
    wait_seconds: Wait = 0,
    include_alpha: Annotated[bool, Field(description="When finished, include a compact alpha summary (IS metrics + checks)")] = True,
) -> Dict[str, Any]:
    """Status of one or more simulations (single or multi).

    RUNNING/UNKNOWN -> call again after retry_after_seconds (or pass wait_seconds to block).
    Finished: status COMPLETE/WARNING/ERROR/FAIL with alpha_id, BRAIN's message/detail, and the
    alpha summary. Multi-simulations list every child; a child that errored does not hide the others.
    """
    if not simulation_ids:
        recent = list(brain.recent_simulations)
        return {"recent_simulations": recent,
                "note": "Pass simulation_ids to check status." if recent else "No simulations created by this server yet."}
    if len(simulation_ids) > 20:
        raise InvalidArgument("at most 20 simulation ids per call")
    states = await brain.simulations(simulation_ids, wait_seconds, include_alpha)
    return {"simulations": states}


@mcp.tool(annotations=DESTRUCTIVE)
async def cancel_simulation(simulation_id: str) -> Dict[str, Any]:
    """Cancel (DELETE) a running simulation to free an account simulation slot."""
    _require_writes("cancel_simulation")
    return await brain.cancel_simulation(simulation_id)


@mcp.tool(annotations=READ)
async def get_platform_setting_options() -> Dict[str, Any]:
    """Valid simulation settings: instrument/region/delay combos with their universes and
    neutralizations, enum lists, plus your saved platform defaults (platform_defaults).
    Use it to fix invalid region/universe/delay combinations before simulating."""
    return await brain.setting_options()


@mcp.tool(annotations=READ)
async def preview_super_selection(
    selection: Annotated[str, Field(description="SuperAlpha selection expression")],
    region: Region = None,
    delay: Delay = None,
    selection_limit: Annotated[Optional[int], Field(ge=10, le=1000, description="Max number of alphas the selection may pick")] = None,
    selection_handling: Optional[Literal["POSITIVE", "NON_ZERO", "NON_NAN"]] = None,
    limit: Annotated[int, Field(ge=1, le=100, description="How many selected alphas to return")] = 20,
) -> Dict[str, Any]:
    """Preview which of your alphas a SuperAlpha selection expression would pick
    (GET /simulations/super-selection). Returns compact alpha summaries."""
    return await brain.super_selection(selection, instrument_type="EQUITY", region=region, delay=delay,
                                       selection_limit=selection_limit,
                                       selection_handling=selection_handling, limit=limit)


# ============================================================== alphas


@mcp.tool(annotations=READ)
async def list_alphas(
    stage: Optional[Literal["IS", "OS", "PROD"]] = None,
    status: Optional[Literal["UNSUBMITTED", "ACTIVE", "DECOMMISSIONED"]] = None,
    alpha_type: Optional[Literal["REGULAR", "SUPER", "RA_PARENT", "RA_CHILD"]] = None,
    limit: Limit = 20,
    offset: Offset = 0,
    order: Annotated[Optional[str], Field(description="-dateCreated (default) or dateCreated; other fields are unverified")] = "-dateCreated",
    created_after: Annotated[Optional[str], Field(description="ISO date/date-time (undocumented filter)")] = None,
    created_before: Optional[str] = None,
    submitted_after: Optional[str] = None,
    submitted_before: Optional[str] = None,
    hidden: Annotated[Optional[bool], Field(description="undocumented filter")] = None,
    full: Annotated[bool, Field(description="Return raw alpha objects instead of compact summaries")] = False,
) -> Dict[str, Any]:
    """List your alphas with filters and pagination (returns count, has_more, next_offset).

    Results are compact summaries (settings, code, IS metrics, failed/pending checks) unless
    full=True. The date filters and `hidden` are not in the documented API (best effort).
    """
    return await brain.list_alphas(stage=stage, status=status, alpha_type=alpha_type, limit=limit,
                                   offset=offset, order=_none_if_blank(order), created_after=created_after,
                                   created_before=created_before, submitted_after=submitted_after,
                                   submitted_before=submitted_before, hidden=hidden, full=full)


@mcp.tool(annotations=READ)
async def get_alpha(alpha_id: str, full: bool = False) -> Dict[str, Any]:
    """One alpha: compact summary (settings, code, IS metrics, checks, pyramids) or the raw
    object with full=True."""
    return await brain.get_alpha(alpha_id, full)


@mcp.tool(annotations=READ)
async def get_alpha_recordset(
    alpha_id: str,
    recordset: Annotated[Optional[str], Field(description=f"e.g. {', '.join(RECORDSET_TYPES)}; omit to list the recordsets available for this alpha")] = None,
    wait_seconds: Wait = 30,
    max_rows: Annotated[int, Field(ge=0, le=10000, description="Keep only the most recent rows (0 = all)")] = 300,
) -> Dict[str, Any]:
    """Time series / tables of an alpha (PnL, sharpe, turnover, daily-pnl, yearly-stats) as
    columns + rows. status PENDING means BRAIN is still computing it: call again."""
    return await brain.recordset(alpha_id, recordset, wait_seconds, max_rows)


@mcp.tool(annotations=READ)
async def check_alpha(
    alpha_id: str,
    correlations: Annotated[List[Literal["self", "prod", "power-pool"]],
                            Field(description="Also fetch these correlation reports (max/min + top correlated alphas)")] = [],
    wait_seconds: Wait = 60,
) -> Dict[str, Any]:
    """BRAIN's authoritative pre-submission checks (GET /alphas/{id}/check): every check with
    result PASS/FAIL/PENDING, limit and value, plus failed/pending lists and all_passed.
    status PENDING means the checks are still running: call again."""
    return await brain.check_alpha(alpha_id, wait_seconds, correlations)


@mcp.tool(annotations=DESTRUCTIVE)
async def submit_alpha(
    alpha_id: str,
    confirm: Annotated[bool, Field(description="False (default) = dry run: only run check_alpha. True = really submit.")] = False,
    wait_seconds: Wait = 60,
) -> Dict[str, Any]:
    """Submit an alpha to BRAIN. Irreversible, so the default is a dry run that returns the
    pre-submission checks. With confirm=True it submits and follows BRAIN's asynchronous
    submission until the final checks: SUBMITTED, REJECTED (see failed), or PENDING/BUSY
    (call again with confirm=True; it resumes polling and never submits twice)."""
    if confirm:
        _require_writes("submit_alpha")
        if not ALLOW_SUBMIT:
            raise InvalidArgument("real submissions are disabled (WQMCP_ALLOW_SUBMIT=0); "
                                  "confirm=False still runs the pre-submission checks")
    return await brain.submit(alpha_id, confirm, wait_seconds)


@mcp.tool(annotations=WRITE_IDEMPOTENT)
async def update_alpha(
    alpha_ids: Annotated[List[str], Field(min_length=1, max_length=100)],
    favorite: Optional[bool] = None,
    hidden: Optional[bool] = None,
    color: Annotated[Optional[str], Field(description="e.g. RED, GREEN; '' clears")] = None,
    name: Annotated[Optional[str], Field(description="single alpha only; '' clears")] = None,
    category: Annotated[Optional[str], Field(description="single alpha only; '' clears")] = None,
    tags: Annotated[Optional[List[str]], Field(description="single alpha only; [] clears")] = None,
    regular_desc: Annotated[Optional[str], Field(description="single alpha only")] = None,
    selection_desc: Annotated[Optional[str], Field(description="single alpha only (SUPER)")] = None,
    combo_desc: Annotated[Optional[str], Field(description="single alpha only (SUPER)")] = None,
    osmosis_points: Annotated[Optional[int], Field(ge=1, le=100000, description="single alpha only; not in the documented API")] = None,
) -> Dict[str, Any]:
    """Update alpha metadata. favorite/hidden/color apply to all alpha_ids in one bulk request;
    name/category/tags/descriptions/osmosis_points apply to a single alpha."""
    _require_writes("update_alpha")
    bulk: Dict[str, Any] = {}
    if favorite is not None:
        bulk["favorite"] = favorite
    if hidden is not None:
        bulk["hidden"] = hidden
    if color is not None:
        bulk["color"] = color or None
    fields: Dict[str, Any] = {}
    if name is not None:
        fields["name"] = name or None
    if category is not None:
        fields["category"] = category or None
    if tags is not None:
        fields["tags"] = tags
    for key, val in (("regular", regular_desc), ("selection", selection_desc), ("combo", combo_desc)):
        if val is not None:
            fields[key] = {"description": val}
    if osmosis_points is not None:
        fields["osmosisPoints"] = osmosis_points
    return await brain.update_alpha(alpha_ids, fields, bulk)


@mcp.tool(annotations=READ)
async def get_alpha_performance(alpha_id: str, competition_id: Optional[str] = None,
                                wait_seconds: Wait = 30) -> Dict[str, Any]:
    """How adding this alpha changes your portfolio: before/after stats (sharpe, fitness,
    turnover, returns, drawdown, margin), yearly stats and PnL. With competition_id, the
    competition's before/after view instead."""
    return _obj(await brain.alpha_performance(alpha_id, competition_id, wait_seconds))


# ================================================================ data


@mcp.tool(annotations=READ)
async def get_datasets(
    region: Region = "USA",
    delay: Delay = 1,
    universe: Universe = "TOP3000",
    search: Annotated[Optional[str], Field(description="Free-text search")] = None,
    include_fields: Annotated[bool, Field(description="With search: use /data-sets/search, which also returns matching fields across all regions")] = False,
    category: Annotated[Optional[str], Field(description="Category id, e.g. fundamental, analyst, model")] = None,
    subcategory: Optional[str] = None,
    theme: Optional[str] = None,
    min_coverage: Annotated[Optional[float], Field(ge=0, le=1, description="coverage lower bound (0-1)")] = None,
    min_value_score: Annotated[Optional[float], Field(ge=0)] = None,
    min_alpha_count: Annotated[Optional[int], Field(ge=0)] = None,
    max_alpha_count: Annotated[Optional[int], Field(ge=0, description="e.g. find under-used datasets")] = None,
    order: Annotated[Optional[str], Field(description="e.g. -valueScore, -alphaCount, coverage")] = None,
    limit: Annotated[int, Field(ge=1, le=50)] = 20,
    offset: Offset = 0,
) -> Dict[str, Any]:
    """Find datasets for a region/delay/universe (compact: id, name, coverage, valueScore,
    alphaCount, fieldCount, pyramidMultiplier, description). Page with offset/next_offset."""
    if include_fields:
        if not search:
            raise InvalidArgument("include_fields=True needs a search term")
        return await brain.search_data(search, limit)
    params = {"instrumentType": "EQUITY", "region": region, "delay": delay, "universe": universe,
              "search": search, "category": category, "subcategory": subcategory, "theme": theme,
              "coverage>": min_coverage, "valueScore>": min_value_score, "alphaCount>": min_alpha_count,
              "alphaCount<": max_alpha_count, "order": order}
    return await brain.datasets(params, limit, offset)


@mcp.tool(annotations=READ)
async def get_datafields(
    region: Region = "USA",
    delay: Delay = 1,
    universe: Universe = "TOP3000",
    dataset_id: Optional[str] = None,
    search: Optional[str] = None,
    field_type: Optional[Literal["MATRIX", "VECTOR", "GROUP", "UNIVERSE", "SYMBOL"]] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    theme: Optional[str] = None,
    min_coverage: Annotated[Optional[float], Field(ge=0, le=1, description="coverage lower bound (0-1)")] = None,
    min_alpha_count: Annotated[Optional[int], Field(ge=0)] = None,
    max_alpha_count: Annotated[Optional[int], Field(ge=0)] = None,
    order: Annotated[Optional[str], Field(description="e.g. -alphaCount, coverage, -userCount")] = None,
    limit: Limit = 50,
    offset: Offset = 0,
    field_id: Annotated[Optional[str], Field(description="Return one field's details (coverage per region/delay/universe) instead of a list")] = None,
) -> Dict[str, Any]:
    """Data fields usable in expressions, filtered by dataset/search/type, paged with
    offset/next_offset. field_id returns a single field's details."""
    if field_id:
        return _obj(await brain.datafield(field_id))
    params = {"instrumentType": "EQUITY", "region": region, "delay": delay, "universe": universe,
              "dataset.id": dataset_id, "search": search, "type": field_type, "category": category,
              "subcategory": subcategory, "theme": theme, "coverage>": min_coverage,
              "alphaCount>": min_alpha_count, "alphaCount<": max_alpha_count, "order": order}
    return await brain.datafields(params, limit, offset)


@mcp.tool(annotations=READ)
async def get_operators(
    category: Annotated[Optional[str], Field(description="e.g. Arithmetic, Time Series, Cross Sectional, Group")] = None,
    scope: Optional[Literal["REGULAR", "COMBO", "SELECTION"]] = None,
    name: Annotated[Optional[str], Field(description="Substring match on the operator name")] = None,
    detail: Annotated[bool, Field(description="Include full descriptions and documentation links")] = False,
) -> Dict[str, Any]:
    """Operators available in FASTEXPR, filterable by category/scope/name (cached for 1h)."""
    ops = await brain.operators()
    out = []
    for op in ops:
        if category and str(op.get("category", "")).lower() != category.lower():
            continue
        if scope and scope not in (op.get("scope") or []):
            continue
        if name and name.lower() not in str(op.get("name", "")).lower():
            continue
        if detail:
            out.append(op)
        else:
            out.append({"name": op.get("name"), "category": op.get("category"),
                        "definition": op.get("definition"), "scope": op.get("scope"),
                        "description": (op.get("description") or "")[:200]})
    categories = sorted({str(op.get("category")) for op in ops if op.get("category")})
    return {"count": len(out), "results": out, "categories": categories}


# ============================================================ activity


@mcp.tool(annotations=READ)
async def get_activity(
    kind: Literal["list", "diversity", "pyramid-alphas", "pyramid-multipliers", "base-payment",
                  "other-payment", "referrals", "simulations", "submissions", "value-factor",
                  "diversity-score"],
    grouping: Annotated[Optional[str], Field(description="diversity: comma list of dataCategory,region,delay")] = None,
    start_date: Annotated[Optional[str], Field(description="pyramid-alphas: YYYY-MM-DD; diversity-score: ISO date/date-time")] = None,
    end_date: Optional[str] = None,
    since: Annotated[Optional[str], Field(description="simulations/submissions: YYYY-MM-DD lower bound")] = None,
    max_rows: Annotated[int, Field(ge=0, le=5000, description="payments/simulations/submissions: keep the most recent rows")] = 60,
) -> Dict[str, Any]:
    """Your BRAIN activity and scores.

    diversity: alpha counts + data-diversity PASS/FAIL per region/delay/category.
    pyramid-alphas / pyramid-multipliers: pyramid distribution and current multipliers.
    base-payment / other-payment: payments (daily, quarterly, competitions, referrals).
    simulations / submissions: daily counts. value-factor: BRAIN's official valueFactor,
    weightFactor, mean correlations. diversity-score: client-side diversity estimate for
    alphas submitted between start_date and end_date (no per-alpha requests).
    """
    if kind == "value-factor":
        return _obj(await brain.consultant())
    if kind == "diversity-score":
        if not start_date or not end_date:
            raise InvalidArgument("diversity-score needs start_date and end_date")
        result = await brain.diversity_score(start_date, end_date)
        try:
            perf = await brain.consultant() or {}
            result["official"] = (perf.get("leaderboard") or {}) if isinstance(perf, dict) else perf
        except Exception as exc:
            result["official"] = {"error": str(exc)}
        return result
    assert kind in ACTIVITY_KINDS
    return _obj(await brain.activity(kind, grouping=grouping, start_date=start_date, end_date=end_date,
                                     since=since, max_rows=max_rows))


@mcp.tool(annotations=READ)
async def get_leaderboard(
    board: Literal["leader", "spc", "power-pool", "referral"] = "leader",
    mine: Annotated[bool, Field(description="Filter to your own row")] = True,
    user: Annotated[Optional[str], Field(description="Filter to this user id instead")] = None,
    order: Annotated[Optional[str], Field(description="e.g. -dailyOsmosisRank; fields depend on the board")] = None,
    aggregate: Optional[Literal["user", "university", "country"]] = None,
    limit: Limit = 10,
    offset: Offset = 0,
) -> Dict[str, Any]:
    """Consultant leaderboards (leader, spc, power-pool, referral) with paging and ordering.
    mine=True (default) returns your own row; set mine=False to browse the board."""
    return await brain.leaderboard(board, limit=limit, offset=offset, order=order, aggregate=aggregate,
                                   user=user, mine=mine and not user)


# ========================================================== community


@mcp.tool(annotations=READ)
async def get_competitions(
    competition_id: Annotated[Optional[str], Field(description="Return one competition's details")] = None,
    scope: Annotated[Literal["active", "all", "ended", "mine"], Field(description="List filter when no competition_id: active (running/upcoming), ended, all, or mine (joined)")] = "active",
    include_agreement: Annotated[bool, Field(description="With competition_id: also fetch the rules/agreement (best effort)")] = False,
    limit: Limit = 20,
    offset: Offset = 0,
) -> Dict[str, Any]:
    """Competitions: discover running ones, list the ones you joined, or read one's details."""
    return await brain.competitions(competition_id=competition_id, scope=scope, limit=limit,
                                    offset=offset, include_agreement=include_agreement)


@mcp.tool(annotations=READ)
async def get_events(
    upcoming_only: bool = True,
    event_type: Optional[Literal["ONLINE", "OFFLINE"]] = None,
    language: Optional[Literal["en", "zh-cn", "ru", "es", "vi", "ko", "th", "ro", "hu"]] = None,
    limit: Limit = 10,
    offset: Offset = 0,
) -> Dict[str, Any]:
    """BRAIN online/offline events (webinars, meetups). Competitions are in get_competitions."""
    return await brain.events(limit=limit, offset=offset, order=None, event_type=event_type,
                              language=language, upcoming_only=upcoming_only)


@mcp.tool(annotations=READ)
async def get_messages(
    unread_only: bool = False,
    message_type: Optional[Literal["ANNOUNCEMENT", "NOTIFICATION"]] = None,
    order: Optional[Literal["-dateCreated", "dateCreated"]] = None,
    limit: Limit = 10,
    offset: Offset = 0,
) -> Dict[str, Any]:
    """Your BRAIN announcements and notifications (embedded images are stripped)."""
    return await brain.messages(limit=limit, offset=offset, unread_only=unread_only,
                                message_type=message_type, order=order)


@mcp.tool(annotations=READ)
async def get_documentation(
    page_id: Annotated[Optional[str], Field(description="A page id from results[].pages[].id of the listing")] = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
) -> Dict[str, Any]:
    """Official BRAIN documentation. Without page_id: tutorials with their pages (ids + titles).
    With page_id: that page's content blocks (text, headings, formulas, example simulations)."""
    return _obj(await brain.documentation(page_id, limit))


# =============================================================== forum


@mcp.tool(annotations=READ)
async def search_forum_posts(
    query: str,
    max_results: Annotated[int, Field(ge=1, le=50)] = 20,
    locale: Annotated[str, Field(description="Support-site locale, e.g. zh-cn, en-us")] = "zh-cn",
) -> Dict[str, Any]:
    """Search the BRAIN community forum (support.worldquantbrain.com) via a headless browser.
    For indexed forum/tutorial content, the brain-rag MCP (rag_search) is faster."""
    return await _forum_client().search_posts(query, max_results=max_results, locale=locale)


@mcp.tool(annotations=READ)
async def read_forum_post(
    post: Annotated[str, Field(description="Post/article id (e.g. 32984819083415), 'posts/<id>', 'articles/<id>' or a support.worldquantbrain.com URL")],
    include_comments: bool = True,
    max_comments: Annotated[int, Field(ge=0, le=500)] = 30,
    locale: Annotated[str, Field(description="Used to build URLs from bare ids, e.g. zh-cn, en-us")] = "zh-cn",
) -> Dict[str, Any]:
    """Read one forum post or article (body keeps line breaks and code) with its comments."""
    return await _forum_client().read_post(post, include_comments=include_comments,
                                           max_comments=max_comments, locale=locale)


@mcp.tool(annotations=READ)
async def get_glossary_terms() -> Dict[str, Any]:
    """BRAIN glossary terms and definitions from the support site (cached for 24h)."""
    return await _forum_client().get_glossary_terms()


# ================================================================ main


def main() -> None:
    logging.basicConfig(level=os.environ.get("WQMCP_LOG_LEVEL", "INFO"), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    transport = os.environ.get("WQMCP_TRANSPORT", "streamable-http")
    if transport != "stdio":
        logger.info("wqmcp listening on http://%s:%s%s", HOST, PORT, mcp.settings.streamable_http_path)
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()

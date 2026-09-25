"""Regression tests for the review of main's new code (ProdMemo, RAA, lint, polling)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time

import psycopg2
import pytest

import platform_functions as pf
import prodmemo_calc
import prodmemo_db
import prodmemo_service


@pytest.fixture
def client(fake):
    c = pf.BrainApiClient()
    yield c
    c._executor.shutdown(wait=False)


# ------------------------------------------------------------------ DAO


def _dead_dao():
    # Nothing listens on port 1: connecting fails fast with OperationalError.
    return prodmemo_db.ProdMemoDao({"host": "127.0.0.1", "port": 1, "dbname": "x", "user": "x",
                                    "connect_timeout": 1})


def test_failed_connect_releases_the_lock():
    dao = _dead_dao()
    errors = []

    def attempt():
        try:
            with dao._cursor():
                pass
        except psycopg2.Error as e:
            errors.append(e)

    threads = [threading.Thread(target=attempt) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "a DAO call hung on a leaked lock"
    assert len(errors) == 4


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=None):
        if self.conn.dead:
            raise psycopg2.OperationalError("server closed the connection unexpectedly")

    def fetchone(self):
        return None

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, dead=False):
        self.dead, self.closed, self.commits = dead, 0, 0

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        if self.dead:
            raise psycopg2.InterfaceError("connection already closed")

    def close(self):
        self.closed = 1


def test_dead_cached_connection_is_replaced(monkeypatch):
    fresh = _FakeConn()
    monkeypatch.setattr(prodmemo_db.psycopg2, "connect", lambda **kw: fresh)
    dao = prodmemo_db.ProdMemoDao({})
    dao._conn = _FakeConn(dead=True)  # e.g. Postgres restarted overnight
    with dao._cursor() as cur:
        cur.execute("SELECT 1")
    assert dao._conn is fresh and fresh.commits == 1


def test_rollback_failure_does_not_mask_the_error(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(prodmemo_db.psycopg2, "connect", lambda **kw: conn)
    dao = prodmemo_db.ProdMemoDao({})
    with pytest.raises(ValueError, match="real problem"):
        with dao._cursor():
            conn.dead = True
            raise ValueError("real problem")
    assert dao._conn is None  # broken connection dropped, next call reconnects


def test_dsn_bounds_statements_and_detects_dead_peers():
    dsn = prodmemo_db.ProdMemoDao.from_env()._dsn
    assert "statement_timeout" in dsn["options"] and dsn["keepalives"] == 1


# -------------------------------------------------------------- service


class _StubDao:
    def __init__(self):
        self.meta = {}

    def ensure_schema(self):
        pass

    def set_sync_meta(self, patch):
        self.meta.update(patch)
        return self.meta

    def get_sync_meta(self):
        return dict(self.meta)


async def test_concurrent_start_sync_starts_one_run(monkeypatch):
    svc = prodmemo_service.ProdMemoService(dao=_StubDao(), fetcher=object())
    gate = asyncio.Event()

    async def long_run(mode):
        await gate.wait()

    monkeypatch.setattr(svc, "_run_sync", long_run)
    monkeypatch.setattr(svc, "ensure_ready", lambda: asyncio.sleep(0))
    a, b = await asyncio.gather(svc.start_sync(), svc.start_sync())
    assert sorted([a["started"], b["started"]]) == [False, True]
    assert svc.dao.meta["error"] is None  # run-scoped keys reset
    gate.set()
    await asyncio.sleep(0)


def test_pnl_errors_are_fatal_only_for_auth():
    class E(Exception):
        def __init__(self, status):
            self.status_code = status

    class CreddUnavailable(Exception):
        pass

    assert not prodmemo_service._is_fatal_for_pnl(E(404))
    assert not prodmemo_service._is_fatal_for_pnl(E(410))
    assert prodmemo_service._is_fatal_for_pnl(E(401))
    assert prodmemo_service._is_fatal(CreddUnavailable("credd down"))


# ----------------------------------------------------------------- calc


def test_js_number_str_matches_javascript():
    samples = [5e-05, 1e-07, 1.5e-07, 123.456, 1e21, 1.5e21, 1e-06, 0.1 + 0.2, 1e20, -3.25e-08, 6.0, 0.001]
    expected = ["0.00005", "1e-7", "1.5e-7", "123.456", "1e+21", "1.5e+21", "0.000001",
                "0.30000000000000004", "100000000000000000000", "-3.25e-8", "6", "0.001"]
    assert [prodmemo_calc.js_number_str(x) for x in samples] == expected
    try:
        out = subprocess.run(["node", "-e", f"console.log(JSON.stringify({json.dumps(samples)}.map(String)))"],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    if out.returncode == 0:
        assert json.loads(out.stdout) == expected


def test_null_schema_does_not_crash():
    # used to raise AttributeError: 'NoneType' object has no attribute 'get'
    prodmemo_calc.extract_platform_correlation_stats({"schema": None, "records": [["X", 0.3]]})


# ---------------------------------------------------------- platform side


async def test_correlation_throttle_is_pending_not_error(client, fake):
    fake.state.rate_limit_next_get = 10
    r = await client._poll_correlation("A1", "prod", max_wait=0)
    assert r["status"] == "PENDING" and r.get("busy") == "HTTP 429"


async def test_writeback_runs_in_background(client, fake, monkeypatch):
    calls = []

    async def slow_write(*args):
        calls.append(args)
        await asyncio.sleep(5)

    monkeypatch.setattr(pf.prodmemo_client, "record_platform_corr", slow_write)
    t0 = time.monotonic()
    res = await client.check_correlation("A1", "prod", max_wait=10)
    assert res["status"] == "DONE" and time.monotonic() - t0 < 4
    await asyncio.sleep(0.2)  # let the background write-back start
    assert calls and calls[0][:2] == ("A1", "prod")
    for task in list(client._background):
        task.cancel()


async def test_raa_with_unfinished_children_is_not_complete(client, fake):
    fake.state.raa_children_pending = True
    summary = await client.get_raa_alpha("RAPX")
    assert summary["children_pending"] == 4 and summary["children_without_fails"] == 0


@pytest.mark.parametrize("expr, ok", [
    ("bucket(rank(cap), range='0.1,1,0.1')", True),
    ("rank(x) # a note with (", True),
    ("ts_backfill(x, lookback=250)", True),
    ("ts_backfill(x, 250)", False),
    ("rank((x)", False),
])
def test_lint(expr, ok):
    assert (pf._lint_expression(expr) == []) is ok

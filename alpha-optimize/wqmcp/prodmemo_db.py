"""ProdMemo persistence layer (PostgreSQL 9.6).

Port of the browser extension's IndexedDB store (``prodMemoDb.js``) plus the
persistence-facing half of ``prodMemoService.js``. See
docs/PRODMEMO_IMPLEMENTATION.md §4.

Target server runs PostgreSQL 9.6, so: no generated columns (group_key is
written by the application), no ``CREATE INDEX IF NOT EXISTS`` (guarded via
pg_indexes), no ``INSERT ... ON CONFLICT`` limitations beyond 9.5+ (upsert is
fine). Only psycopg2 is required — no ORM, no connection pool: the MCP server is
a single low-concurrency process.
"""

import json
import os
import threading
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS prodmemo_alphas (
        id                  text PRIMARY KEY,
        name                text,
        type                text,
        status              text,
        stage               text,
        date_submitted      timestamptz,
        region              text,
        universe            text,
        delay               smallint,
        instrument_type     text,
        classifications     jsonb NOT NULL DEFAULT '[]',
        is_metrics          jsonb,
        submitted           boolean NOT NULL DEFAULT false,
        no_longer_submitted boolean NOT NULL DEFAULT false,
        group_key           text NOT NULL DEFAULT '',
        synced_at           timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prodmemo_pnls (
        alpha_id    text PRIMARY KEY,
        fingerprint text NOT NULL,
        last_date   date,
        point_count integer NOT NULL DEFAULT 0,
        fetched_at  timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prodmemo_pnl_points (
        alpha_id text NOT NULL REFERENCES prodmemo_pnls(alpha_id) ON DELETE CASCADE,
        date     date NOT NULL,
        value    double precision NOT NULL,
        PRIMARY KEY (alpha_id, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prodmemo_platform_corrs (
        alpha_id   text PRIMARY KEY,
        prod       jsonb,
        pool       jsonb,
        self       jsonb,
        updated_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prodmemo_local_corrs (
        alpha_id          text NOT NULL,
        corr_type         text NOT NULL
            CHECK (corr_type IN ('SELF','POOL','PROD_LOWER_BOUND')),
        group_key         text NOT NULL DEFAULT '',
        result            jsonb NOT NULL,
        algorithm_version integer NOT NULL,
        input_fingerprint text NOT NULL,
        calculated_at     timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (alpha_id, corr_type)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prodmemo_sync_meta (
        key   text PRIMARY KEY,
        value jsonb NOT NULL
    )
    """,
]

# 9.6 has no CREATE INDEX IF NOT EXISTS; create only when absent.
INDEX_STATEMENTS = [
    ('prodmemo_alphas_group_idx',
     "CREATE INDEX prodmemo_alphas_group_idx ON prodmemo_alphas (group_key) WHERE submitted"),
    ('prodmemo_alphas_submitted_date_idx',
     "CREATE INDEX prodmemo_alphas_submitted_date_idx "
     "ON prodmemo_alphas (date_submitted DESC) WHERE submitted"),
]

# Human-facing overview: one row per alpha. NOTE it cannot express `stale`
# (that needs a fingerprint recomputation in the application layer), so treat it
# as an inspection aid — prodmemo_get is the authoritative output.
STATUS_VIEW = """
CREATE OR REPLACE VIEW prodmemo_alpha_status AS
SELECT
    a.id                                        AS alpha_id,
    a.name, a.stage, a.status,
    a.group_key, a.date_submitted,
    a.submitted, a.no_longer_submitted,
    (p.alpha_id IS NOT NULL)                    AS has_pnl,
    p.last_date                                 AS pnl_last_date,
    p.point_count                               AS pnl_points,
    p.fetched_at                                AS pnl_fetched_at,
    (pc.prod ->> 'max')::float8                 AS platform_prod_max,
    (pc.pool ->> 'max')::float8                 AS platform_pool_max,
    (pc.self ->> 'max')::float8                 AS platform_self_max,
    (ls.result ->> 'available')::boolean        AS local_self_available,
    (ls.result ->> 'max')::float8               AS local_self_max,
    (lp.result ->> 'max')::float8               AS local_pool_max,
    (lb.result ->> 'max')::float8               AS local_prod_lower_bound,
    lb.result -> 'witness'                      AS prod_witness,
    ls.calculated_at                            AS local_calculated_at,
    ls.algorithm_version,
    ls.input_fingerprint                        AS self_fingerprint
FROM prodmemo_alphas a
LEFT JOIN prodmemo_pnls           p  ON p.alpha_id  = a.id
LEFT JOIN prodmemo_platform_corrs pc ON pc.alpha_id = a.id
LEFT JOIN prodmemo_local_corrs    ls ON ls.alpha_id = a.id AND ls.corr_type = 'SELF'
LEFT JOIN prodmemo_local_corrs    lp ON lp.alpha_id = a.id AND lp.corr_type = 'POOL'
LEFT JOIN prodmemo_local_corrs    lb ON lb.alpha_id = a.id AND lb.corr_type = 'PROD_LOWER_BOUND'
"""

CORR_TYPES = ('SELF', 'POOL', 'PROD_LOWER_BOUND')


def _iso(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


class ProdMemoDao:
    """Synchronous psycopg2 DAO. Call from an executor in async contexts."""

    def __init__(self, dsn_params):
        self._dsn = dsn_params
        self._conn = None
        self._lock = threading.RLock()
        self._schema_ready = False

    @classmethod
    def from_env(cls):
        return cls({
            'host': os.environ.get('PRODMEMO_PG_HOST', '127.0.0.1'),
            'port': int(os.environ.get('PRODMEMO_PG_PORT', '5432')),
            'dbname': os.environ.get('PRODMEMO_PG_DB', 'brain_serve'),
            'user': os.environ.get('PRODMEMO_PG_USER', 'postgres'),
            'password': os.environ.get('PRODMEMO_PG_PASSWORD', ''),
            'connect_timeout': int(os.environ.get('PRODMEMO_PG_CONNECT_TIMEOUT', '10')),
        })

    # --- connection -------------------------------------------------------

    def _connect(self):
        with self._lock:
            if self._conn is None or self._conn.closed:
                self._conn = psycopg2.connect(**self._dsn)
            return self._conn

    def _cursor(self):
        """Context manager yielding a cursor inside a transaction.

        Reconnects once if the cached connection died (server restart, idle
        timeout) so a long-lived MCP process recovers on its own.
        """
        dao = self

        class _Ctx:
            def __enter__(self):
                dao._lock.acquire()
                try:
                    self.conn = dao._connect()
                    self.cur = self.conn.cursor(
                        cursor_factory=psycopg2.extras.RealDictCursor)
                except psycopg2.Error:
                    dao._conn = None
                    self.conn = dao._connect()
                    self.cur = self.conn.cursor(
                        cursor_factory=psycopg2.extras.RealDictCursor)
                return self.cur

            def __exit__(self, exc_type, exc, tb):
                try:
                    if exc_type is None:
                        self.conn.commit()
                    else:
                        self.conn.rollback()
                    self.cur.close()
                finally:
                    dao._lock.release()
                return False

        return _Ctx()

    def ensure_schema(self):
        """Idempotent migration; safe to call on every startup."""
        if self._schema_ready:
            return
        with self._cursor() as cur:
            for statement in SCHEMA_STATEMENTS:
                cur.execute(statement)
            for index_name, statement in INDEX_STATEMENTS:
                cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s", (index_name,))
                if not cur.fetchone():
                    cur.execute(statement)
            cur.execute(STATUS_VIEW)
        self._schema_ready = True

    def ping(self):
        with self._cursor() as cur:
            cur.execute("SELECT version()")
            return cur.fetchone()['version']

    # --- reads ------------------------------------------------------------

    def light_snapshot(self):
        """Everything needed for fingerprinting WITHOUT loading PnL bodies.

        Equivalent to the extension's getProdMemoLightSnapshot (prodMemoDb.js:219):
        PnL is represented by {'alphaId', 'fingerprint'} stubs, which is all
        pnl_fingerprint() needs. Bodies come later via load_pnl_points().
        """
        with self._cursor() as cur:
            cur.execute("""
                SELECT id, name, type, status, stage, date_submitted, region, universe,
                       delay, instrument_type, classifications, is_metrics, submitted,
                       no_longer_submitted, group_key
                FROM prodmemo_alphas
            """)
            alphas = [self._row_to_alpha(row) for row in cur.fetchall()]

            cur.execute("SELECT alpha_id, fingerprint, last_date, point_count FROM prodmemo_pnls")
            pnl_rows = cur.fetchall()

            cur.execute("SELECT alpha_id, prod, pool, self, updated_at FROM prodmemo_platform_corrs")
            platform = {row['alpha_id']: {'prod': row['prod'], 'pool': row['pool'],
                                          'self': row['self'],
                                          'updatedAt': _iso(row['updated_at'])}
                        for row in cur.fetchall()}

            cur.execute("""
                SELECT alpha_id, corr_type, group_key, result, algorithm_version,
                       input_fingerprint, calculated_at
                FROM prodmemo_local_corrs
            """)
            local = [{'alphaId': row['alpha_id'], 'corrType': row['corr_type'],
                      'groupKey': row['group_key'], 'result': row['result'],
                      'algorithmVersion': row['algorithm_version'],
                      'inputFingerprint': row['input_fingerprint'],
                      'calculatedAt': _iso(row['calculated_at'])}
                     for row in cur.fetchall()]

            cur.execute("SELECT key, value FROM prodmemo_sync_meta")
            sync = {row['key']: row['value'] for row in cur.fetchall()}

        return {
            'alphas': alphas,
            'pnlStubs': {row['alpha_id']: {'alphaId': row['alpha_id'],
                                           'fingerprint': row['fingerprint'],
                                           'lastDate': _iso(row['last_date']),
                                           'pointCount': row['point_count']}
                         for row in pnl_rows},
            'platformCorrs': platform,
            'localCorrs': local,
            'sync': sync.get('submitted') or {},
        }

    @staticmethod
    def _row_to_alpha(row):
        """DB row -> the alpha shape prodmemo_calc expects (settings sub-dict)."""
        return {
            'id': row['id'],
            'name': row['name'],
            'type': row['type'],
            'status': row['status'],
            'stage': row['stage'],
            'dateSubmitted': _iso(row['date_submitted']),
            'settings': {
                'region': row['region'],
                'universe': row['universe'],
                'delay': row['delay'],
                'instrumentType': row['instrument_type'],
            },
            'classifications': row['classifications'] or [],
            'is': row['is_metrics'] or {},
            'submitted': row['submitted'],
            'noLongerSubmitted': row['no_longer_submitted'],
            'groupKey': row['group_key'],
        }

    def load_pnl_points(self, alpha_ids):
        """Load PnL bodies on demand -> {alpha_id: {'records': [[date, value], ...]}}."""
        ids = list({str(a) for a in alpha_ids if a})
        if not ids:
            return {}
        out = {}
        with self._cursor() as cur:
            cur.execute("""
                SELECT alpha_id, date, value FROM prodmemo_pnl_points
                WHERE alpha_id = ANY(%s) ORDER BY alpha_id, date
            """, (ids,))
            for row in cur.fetchall():
                out.setdefault(row['alpha_id'], {'alphaId': row['alpha_id'], 'records': []})
                out[row['alpha_id']]['records'].append(
                    [row['date'].strftime('%Y-%m-%d'), row['value']])
        return out

    def get_alpha(self, alpha_id):
        with self._cursor() as cur:
            cur.execute("""
                SELECT id, name, type, status, stage, date_submitted, region, universe,
                       delay, instrument_type, classifications, is_metrics, submitted,
                       no_longer_submitted, group_key
                FROM prodmemo_alphas WHERE id = %s
            """, (alpha_id,))
            row = cur.fetchone()
        return self._row_to_alpha(row) if row else None

    def get_sync_state(self):
        """Baseline for the sync engine (prodMemoService.js:588-603)."""
        with self._cursor() as cur:
            cur.execute("SELECT id FROM prodmemo_alphas WHERE submitted")
            alpha_ids = [row['id'] for row in cur.fetchall()]

            cur.execute("""
                SELECT a.id FROM prodmemo_alphas a
                LEFT JOIN prodmemo_pnls p ON p.alpha_id = a.id
                WHERE a.submitted AND p.alpha_id IS NULL
            """)
            missing_pnl = [row['id'] for row in cur.fetchall()]

            # Reference curves for the Prod lower bound: a known platform prod
            # value but missing metadata or PnL -> must be backfilled.
            cur.execute("""
                SELECT pc.alpha_id FROM prodmemo_platform_corrs pc
                LEFT JOIN prodmemo_alphas a ON a.id = pc.alpha_id
                LEFT JOIN prodmemo_pnls   p ON p.alpha_id = pc.alpha_id
                WHERE pc.prod IS NOT NULL
                  AND (pc.prod ->> 'max') IS NOT NULL
                  AND (a.id IS NULL OR p.alpha_id IS NULL)
            """)
            backfill = [row['alpha_id'] for row in cur.fetchall()]

            cur.execute("SELECT value FROM prodmemo_sync_meta WHERE key = 'submitted'")
            row = cur.fetchone()

        return {
            'alphaIds': alpha_ids,
            'missingPnlIds': missing_pnl,
            'backfillIds': backfill,
            'sync': (row['value'] if row else {}) or {},
        }

    def stats(self):
        with self._cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM prodmemo_alphas WHERE submitted")
            submitted = cur.fetchone()['n']
            cur.execute("SELECT count(*) AS n FROM prodmemo_pnls")
            pnl_count = cur.fetchone()['n']
            cur.execute("SELECT count(*) AS n FROM prodmemo_platform_corrs")
            platform_count = cur.fetchone()['n']
            cur.execute("SELECT count(*) AS n FROM prodmemo_local_corrs")
            local_count = cur.fetchone()['n']
            # valid_reference_count: platform prod.max + metadata + PnL all present
            cur.execute("""
                SELECT count(*) AS n FROM prodmemo_platform_corrs pc
                JOIN prodmemo_alphas a ON a.id = pc.alpha_id
                JOIN prodmemo_pnls   p ON p.alpha_id = pc.alpha_id
                WHERE (pc.prod ->> 'max') IS NOT NULL
            """)
            valid_reference = cur.fetchone()['n']
            cur.execute("SELECT value FROM prodmemo_sync_meta WHERE key = 'submitted'")
            row = cur.fetchone()
        return {
            'submitted_alpha_count': submitted,
            'pnl_count': pnl_count,
            'platform_corr_count': platform_count,
            'local_corr_count': local_count,
            'valid_reference_count': valid_reference,
            'sync': (row['value'] if row else {}) or {},
        }

    def alpha_status(self, alpha_id=None, group_key='', limit=100, offset=0):
        """Read the prodmemo_alpha_status view (§4.4)."""
        clauses, params = [], []
        if alpha_id:
            clauses.append("alpha_id = %s")
            params.append(alpha_id)
        if group_key:
            clauses.append("group_key = %s")
            params.append(group_key)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._cursor() as cur:
            cur.execute(f"SELECT count(*) AS n FROM prodmemo_alpha_status {where}", params)
            total = cur.fetchone()['n']
            cur.execute(
                f"""SELECT * FROM prodmemo_alpha_status {where}
                    ORDER BY date_submitted DESC NULLS LAST, alpha_id
                    LIMIT %s OFFSET %s""",
                params + [limit, offset])
            rows = [dict(row) for row in cur.fetchall()]
        for row in rows:
            for key, value in list(row.items()):
                if isinstance(value, datetime):
                    row[key] = _iso(value)
                elif hasattr(value, 'strftime'):
                    row[key] = value.strftime('%Y-%m-%d')
        return {'total': total, 'rows': rows}

    # --- writes -----------------------------------------------------------

    def save_alpha_batch(self, records):
        """Upsert alpha metadata.

        `submitted` is sticky (prodMemoService.js:386-397): once true it is never
        downgraded by a later backfill write that only knows the alpha as a
        reference curve.
        """
        if not records:
            return {'saved': 0}
        rows = [(
            r['id'], r.get('name'), r.get('type'), r.get('status'), r.get('stage'),
            r.get('dateSubmitted'), r.get('region'), r.get('universe'), r.get('delay'),
            r.get('instrumentType'), json.dumps(r.get('classifications') or []),
            json.dumps(r.get('is') or {}), bool(r.get('submitted')), r.get('groupKey') or '',
        ) for r in records if r.get('id')]
        if not rows:
            return {'saved': 0}
        with self._cursor() as cur:
            psycopg2.extras.execute_values(cur, """
                INSERT INTO prodmemo_alphas
                    (id, name, type, status, stage, date_submitted, region, universe,
                     delay, instrument_type, classifications, is_metrics, submitted,
                     group_key, synced_at)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    type = EXCLUDED.type,
                    status = EXCLUDED.status,
                    stage = EXCLUDED.stage,
                    date_submitted = EXCLUDED.date_submitted,
                    region = EXCLUDED.region,
                    universe = EXCLUDED.universe,
                    delay = EXCLUDED.delay,
                    instrument_type = EXCLUDED.instrument_type,
                    classifications = EXCLUDED.classifications,
                    is_metrics = EXCLUDED.is_metrics,
                    submitted = prodmemo_alphas.submitted OR EXCLUDED.submitted,
                    no_longer_submitted = false,
                    group_key = EXCLUDED.group_key,
                    synced_at = now()
            """, rows, template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())")
        return {'saved': len(rows)}

    def save_pnl(self, alpha_id, records, fingerprint):
        """Idempotent short-circuit on fingerprint (prodMemoService.js:399-417)."""
        with self._cursor() as cur:
            cur.execute("SELECT fingerprint FROM prodmemo_pnls WHERE alpha_id = %s", (alpha_id,))
            row = cur.fetchone()
            if row and row['fingerprint'] == fingerprint:
                return {'unchanged': True, 'points': 0}

            last_date = records[-1][0] if records else None
            cur.execute("""
                INSERT INTO prodmemo_pnls (alpha_id, fingerprint, last_date, point_count, fetched_at)
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (alpha_id) DO UPDATE SET
                    fingerprint = EXCLUDED.fingerprint,
                    last_date = EXCLUDED.last_date,
                    point_count = EXCLUDED.point_count,
                    fetched_at = now()
            """, (alpha_id, fingerprint, last_date, len(records)))
            cur.execute("DELETE FROM prodmemo_pnl_points WHERE alpha_id = %s", (alpha_id,))
            if records:
                psycopg2.extras.execute_values(cur, """
                    INSERT INTO prodmemo_pnl_points (alpha_id, date, value) VALUES %s
                """, [(alpha_id, date, value) for date, value in records])
        return {'unchanged': False, 'points': len(records)}

    def mark_submitted_snapshot(self, keep_ids):
        """Soft-delete reconciliation (prodMemoDb.js:261-276).

        Alphas no longer present remotely keep their rows and PnL — they remain
        valid reference curves for the Prod lower bound.
        """
        ids = list({str(a) for a in keep_ids if a})
        with self._cursor() as cur:
            if ids:
                cur.execute("""
                    UPDATE prodmemo_alphas
                    SET submitted = false, no_longer_submitted = true
                    WHERE submitted AND NOT (id = ANY(%s))
                """, (ids,))
            else:
                cur.execute("""
                    UPDATE prodmemo_alphas
                    SET submitted = false, no_longer_submitted = true
                    WHERE submitted
                """)
            return {'marked': cur.rowcount}

    def save_platform_corr(self, alpha_id, corr_type, stat):
        """Read-modify-write so the other two buckets survive (prodMemoService.js:160-173)."""
        column = {'prod': 'prod', 'pool': 'pool', 'self': 'self'}[corr_type]
        with self._cursor() as cur:
            cur.execute(f"""
                INSERT INTO prodmemo_platform_corrs (alpha_id, {column}, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (alpha_id) DO UPDATE SET
                    {column} = EXCLUDED.{column}, updated_at = now()
            """, (alpha_id, json.dumps(stat)))
        return {'saved': True}

    def save_local_corrs(self, rows):
        if not rows:
            return {'saved': 0}
        payload = [(r['alphaId'], r['corrType'], r.get('groupKey') or '',
                    json.dumps(r['result']), r['algorithmVersion'], r['inputFingerprint'])
                   for r in rows]
        with self._cursor() as cur:
            psycopg2.extras.execute_values(cur, """
                INSERT INTO prodmemo_local_corrs
                    (alpha_id, corr_type, group_key, result, algorithm_version,
                     input_fingerprint, calculated_at)
                VALUES %s
                ON CONFLICT (alpha_id, corr_type) DO UPDATE SET
                    group_key = EXCLUDED.group_key,
                    result = EXCLUDED.result,
                    algorithm_version = EXCLUDED.algorithm_version,
                    input_fingerprint = EXCLUDED.input_fingerprint,
                    calculated_at = now()
            """, payload, template="(%s,%s,%s,%s,%s,%s,now())")
        return {'saved': len(payload)}

    def set_sync_meta(self, patch):
        """Merge-patch the single 'submitted' row (single writer, no race)."""
        with self._cursor() as cur:
            cur.execute("SELECT value FROM prodmemo_sync_meta WHERE key = 'submitted'")
            row = cur.fetchone()
            value = dict(row['value'] or {}) if row else {}
            value.update(patch)
            cur.execute("""
                INSERT INTO prodmemo_sync_meta (key, value) VALUES ('submitted', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """, (json.dumps(value),))
        return value

    def get_sync_meta(self):
        with self._cursor() as cur:
            cur.execute("SELECT value FROM prodmemo_sync_meta WHERE key = 'submitted'")
            row = cur.fetchone()
        return (row['value'] if row else {}) or {}

    # --- destructive ------------------------------------------------------

    def delete_alpha_corrs(self, alpha_id):
        with self._cursor() as cur:
            cur.execute("DELETE FROM prodmemo_platform_corrs WHERE alpha_id = %s", (alpha_id,))
            platform_deleted = cur.rowcount
            cur.execute("DELETE FROM prodmemo_local_corrs WHERE alpha_id = %s", (alpha_id,))
            local_deleted = cur.rowcount
        return {'deleted': platform_deleted + local_deleted,
                'platform': platform_deleted, 'local': local_deleted}

    def clear_corrs(self):
        with self._cursor() as cur:
            cur.execute("DELETE FROM prodmemo_platform_corrs")
            platform = cur.rowcount
            cur.execute("DELETE FROM prodmemo_local_corrs")
            local = cur.rowcount
        return {'platform_cleared': platform, 'local_cleared': local}

    def clear_sync_data(self):
        """Drop alphas/PnL/sync state, keep corrs (local ones go stale naturally)."""
        with self._cursor() as cur:
            cur.execute("DELETE FROM prodmemo_pnl_points")
            cur.execute("DELETE FROM prodmemo_pnls")
            pnls = cur.rowcount
            cur.execute("DELETE FROM prodmemo_alphas")
            alphas = cur.rowcount
            cur.execute("DELETE FROM prodmemo_sync_meta")
        return {'alphas_cleared': alphas, 'pnls_cleared': pnls}

    def export_corrs(self):
        """schemaVersion 2 payload, byte-compatible with the extension's export."""
        with self._cursor() as cur:
            cur.execute("SELECT alpha_id, prod, pool, self FROM prodmemo_platform_corrs")
            platform = {row['alpha_id']: {'prod': row['prod'], 'pool': row['pool'],
                                          'self': row['self']}
                        for row in cur.fetchall()}
            cur.execute("""
                SELECT alpha_id, corr_type, group_key, result, algorithm_version,
                       input_fingerprint, calculated_at
                FROM prodmemo_local_corrs
            """)
            local = [{'alphaId': row['alpha_id'], 'corrType': row['corr_type'],
                      'groupKey': row['group_key'], 'result': row['result'],
                      'algorithmVersion': row['algorithm_version'],
                      'inputFingerprint': row['input_fingerprint'],
                      'calculatedAt': _iso(row['calculated_at'])}
                     for row in cur.fetchall()]
        return {'schemaVersion': 2, 'platformCorrs': platform, 'localCorrs': local}

"""ProdMemo orchestration: sync engine, fingerprint cache, decision facade.

Port of the browser extension's MAIN-world sync engine
(``prodmemoMain.js``) and the orchestration half of ``prodMemoService.js``.
See docs/PRODMEMO_IMPLEMENTATION.md §6.

Deliberately free of any ``mcp`` import so the test-suite runs on machines that
only have psycopg2 — platform_functions.py injects the BRAIN client at import
time (``prodmemo_client.fetcher = brain_client``).
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone

from prodmemo_calc import (
    PROD_MEMO_ALGORITHM_VERSION,
    alpha_group_key,
    calculate_local_correlation,
    calculate_prod_lower_bound,
    calculation_fingerprint,
    extract_platform_correlation_stats,
    finite_number,
    format_resolved_metric,
    normalize_correlation_type,
    normalize_pnl,
    pnl_fingerprint,
    resolve_preferred_correlation,
    select_candidate_alphas,
)
from prodmemo_db import CORR_TYPES, ProdMemoDao

logger = logging.getLogger(__name__)

ALPHA_PAGE_LIMIT = 100
ALPHA_WINDOW_LIMIT = 1000     # BRAIN caps offset at 1000; beyond that use a time cursor
PNL_BATCH_SIZE = 100
PNL_CONCURRENCY = 3
BACKFILL_CONCURRENCY = 2
RETRY_DELAYS = (1, 2, 4)
# "Never skip a page" still needs an end: after this long the sync fails loudly
# (cron exits non-zero) instead of retrying for days.
PAGE_RETRY_BUDGET_SECONDS = 1800
PROGRESS_EVERY = 25
PROD_THRESHOLD = 0.7
# IS alpha 的 PnL 是懒生成: 首次 GET 返回 204/空并触发平台生成, 实测 10-60s
# 后才有数据。_ensure_alpha_data 据此做有界等待重试, 免得 prodmemo_check
# 首查即 insufficient_data、全靠调用方手动再试。RETRY_DELAYS(1/2/4s) 对此太短。
PNL_PENDING_RETRIES = 4
PNL_PENDING_DELAY = 8
# Empirical PROD estimate from the local POOL max: prod ≈ a + b × pool. The
# default was fitted by hand on 4 platform-measured alphas (2026-09, all within
# ±0.05). Once PROD_EST_MIN_FIT alphas have both a platform Prod value and a
# local POOL result, the coefficients are refitted from them (least squares), so
# every measured Prod written back improves the estimate.
PROD_EST_DEFAULT = (0.428, 1.139)
PROD_EST_MIN_FIT = 8
CHECK_MANY_LIMIT = 20
CHECK_MANY_CONCURRENCY = 2


def fit_prod_estimate(pairs):
    """(a, b, n, resid_sd, source) for prod ≈ a + b·pool from (pool, prod) pairs;
    falls back to PROD_EST_DEFAULT with too few / degenerate points."""
    n = len(pairs)
    if n >= PROD_EST_MIN_FIT:
        mx = sum(x for x, _ in pairs) / n
        my = sum(y for _, y in pairs) / n
        sxx = sum((x - mx) ** 2 for x, _ in pairs)
        if sxx > 1e-9:
            b = sum((x - mx) * (y - my) for x, y in pairs) / sxx
            a = my - b * mx
            resid = sum((y - a - b * x) ** 2 for x, y in pairs)
            return a, b, n, (resid / max(n - 2, 1)) ** 0.5, 'fitted'
    a, b = PROD_EST_DEFAULT
    return a, b, n, None, 'default'


class SyncStopped(Exception):
    """Raised at checkpoints when a stop was requested."""


class FatalSyncError(Exception):
    """Auth failures (401/403) — retrying is pointless."""


def _status_of(exc):
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    return status if status is not None else getattr(exc, 'status_code', None)


def _is_fatal(exc):
    """Errors retrying cannot fix: auth failures, other client errors on list
    pages, and credd being unavailable (down / backoff / pending biometric)."""
    if type(exc).__name__ == 'CreddUnavailable':
        return True
    status = _status_of(exc)
    if status in (401, 403):
        return True
    if status is not None and 400 <= status < 500 and status not in (408, 429):
        return True
    return False


def _is_fatal_for_pnl(exc):
    """For one alpha's PnL only auth/credd problems are fatal: a 404/410 on a
    single alpha (e.g. no PnL for it) must not abort the whole sync."""
    return type(exc).__name__ == 'CreddUnavailable' or _status_of(exc) in (401, 403)


def _oldest_submitted_at(results):
    """JS oldestSubmittedAt (prodmemoMain.js:156-166)."""
    oldest, oldest_time = '', None
    for alpha in results:
        raw = (alpha or {}).get('dateSubmitted')
        parsed = _parse_iso(raw)
        if parsed is None:
            continue
        if oldest_time is None or parsed < oldest_time:
            oldest_time, oldest = parsed, raw
    return oldest


def _parse_iso(value):
    if not value:
        return None
    text = str(value).strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def next_exclusive_upper_bound(oldest):
    """JS nextExclusiveUpperBound (prodmemoMain.js:168-173).

    +1ms so records sharing the boundary timestamp get fetched again and
    de-duplicated by id — losing them would be silent data loss.
    """
    parsed = _parse_iso(oldest)
    if parsed is None:
        return ''
    return (parsed.astimezone(timezone.utc) + timedelta(milliseconds=1)) \
        .isoformat().replace('+00:00', 'Z')


def normalize_alpha_record(alpha, submitted=True):
    """BRAIN alpha payload -> the compact row shape used by prodmemo_db."""
    if not alpha or not alpha.get('id'):
        return None
    settings = alpha.get('settings') or {}
    is_metrics = alpha.get('is') or {}
    classifications = []
    for item in alpha.get('classifications') or []:
        if isinstance(item, dict):
            if item.get('id'):
                classifications.append({'id': item['id'], 'name': item.get('name')})
        elif item:
            classifications.append({'id': str(item), 'name': None})
    region = settings.get('region', alpha.get('region'))
    universe = settings.get('universe', alpha.get('universe'))
    delay = settings.get('delay', alpha.get('delay'))
    return {
        'id': alpha['id'],
        'name': alpha.get('name'),
        'type': alpha.get('type'),
        'status': alpha.get('status'),
        'stage': alpha.get('stage'),
        'dateSubmitted': alpha.get('dateSubmitted'),
        'region': region,
        'universe': universe,
        'delay': finite_number(delay),
        'instrumentType': settings.get('instrumentType', alpha.get('instrumentType')),
        'classifications': classifications,
        'is': {k: is_metrics.get(k) for k in
               ('sharpe', 'returns', 'turnover', 'fitness', 'margin')},
        'submitted': bool(submitted),
        'groupKey': alpha_group_key(
            {'settings': {'region': region, 'universe': universe, 'delay': delay}}),
    }


class ProdMemoService:
    def __init__(self, dao=None, fetcher=None):
        self.dao = dao or ProdMemoDao.from_env()
        self.fetcher = fetcher
        self._sync_task = None
        self._stop_event = asyncio.Event()
        self._start_lock = asyncio.Lock()
        self._schema_lock = asyncio.Lock()
        self._schema_ready = False

    # --- infrastructure ---------------------------------------------------

    async def _db(self, fn, *args, **kwargs):
        """Run a blocking DAO call off the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    async def ensure_ready(self):
        if self._schema_ready:
            return
        async with self._schema_lock:
            if not self._schema_ready:
                await self._db(self.dao.ensure_schema)
                self._schema_ready = True

    def _check_stopped(self):
        if self._stop_event.is_set():
            raise SyncStopped('同步已被请求停止。')

    async def _progress(self, **patch):
        patch.setdefault('updatedAt', datetime.now(timezone.utc).isoformat())
        await self._db(self.dao.set_sync_meta, patch)

    # --- BRAIN fetch helpers ---------------------------------------------

    async def _fetch_alpha_page(self, offset, limit=ALPHA_PAGE_LIMIT, before=''):
        kwargs = {
            'stage': 'OS',
            'limit': limit,
            'offset': offset,
            'order': '-dateSubmitted',
            'hidden': False,
        }
        if before:
            kwargs['submission_end_date'] = before
        payload = await self.fetcher.get_user_alphas(**kwargs)
        if not isinstance(payload, dict):
            raise RuntimeError(f'Unexpected alphas payload: {type(payload)}')
        return {'count': finite_number(payload.get('count')) or 0,
                'results': payload.get('results') or []}

    async def _fetch_alpha_page_persistent(self, offset, limit=ALPHA_PAGE_LIMIT, before=''):
        """Never skip a page: retry forever with capped backoff.

        Losing one page means permanently missing alphas, so blocking beats
        skipping (prodmemoMain.js:55-69 withPersistentRetries).
        """
        attempt = 0
        give_up_at = time.monotonic() + PAGE_RETRY_BUDGET_SECONDS
        while True:
            self._check_stopped()
            try:
                return await self._fetch_alpha_page(offset, limit, before)
            except SyncStopped:
                raise
            except Exception as exc:
                if _is_fatal(exc):
                    raise FatalSyncError(str(exc)) from exc
                if time.monotonic() > give_up_at:
                    raise FatalSyncError(
                        f'alpha page offset={offset} still failing after '
                        f'{PAGE_RETRY_BUDGET_SECONDS // 60} minutes: {exc}') from exc
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                attempt += 1
                logger.warning('alpha page offset=%s failed (attempt %s), retrying in %ss: %s',
                               offset, attempt, delay, exc)
                await self._progress(
                    phase='alphas-retry',
                    message=f'Alpha 分页 offset={offset} 第 {attempt} 次失败，{delay}s 后原地重试（不会跳过）。')
                await asyncio.sleep(delay)

    async def _fetch_pnl_once(self, alpha_id):
        """Returns (records, status). status: 'ok' | 'pending' | 'failed'."""
        try:
            payload = await self.fetcher.get_alpha_pnl(alpha_id)
        except Exception as exc:
            if _is_fatal_for_pnl(exc):
                raise FatalSyncError(str(exc)) from exc
            logger.warning('pnl fetch failed for %s: %s', alpha_id, exc)
            return None, 'failed'
        if not payload or not isinstance(payload, dict):
            return None, 'pending'       # 202/204/empty body -> still generating
        records = normalize_pnl(payload)
        if not records:
            return None, 'pending'
        return records, 'ok'

    # --- alpha pagination -------------------------------------------------

    async def _fetch_all_submitted_alphas(self):
        """Full pagination with time-window splitting (prodmemoMain.js:175-243)."""
        alpha_ids = set()
        before = ''
        window_number = 0
        pages_done = 0

        async def save_page(page, window_results):
            nonlocal pages_done
            submitted = [a for a in page['results']
                         if a.get('id') and a.get('dateSubmitted')
                         and a.get('status') != 'UNSUBMITTED']
            records = [r for r in (normalize_alpha_record(a, True) for a in submitted) if r]
            if records:
                await self._db(self.dao.save_alpha_batch, records)
            alpha_ids.update(r['id'] for r in records)
            window_results.extend(page['results'])
            pages_done += 1
            if pages_done % 5 == 0:
                await self._progress(phase='alphas', success=len(alpha_ids),
                                     message=f'Alpha 时间段 {window_number} · 已获取 {len(alpha_ids)}')

        while True:
            self._check_stopped()
            window_number += 1
            first = await self._fetch_alpha_page_persistent(0, ALPHA_PAGE_LIMIT, before)
            if not first['results']:
                break
            window_count = min(int(first['count']), ALPHA_WINDOW_LIMIT)
            window_results = []
            ids_before_window = len(alpha_ids)
            await save_page(first, window_results)
            offset = ALPHA_PAGE_LIMIT
            while offset < window_count:
                self._check_stopped()
                page = await self._fetch_alpha_page_persistent(offset, ALPHA_PAGE_LIMIT, before)
                await save_page(page, window_results)
                offset += ALPHA_PAGE_LIMIT

            if int(first['count']) <= ALPHA_WINDOW_LIMIT:
                break
            next_before = next_exclusive_upper_bound(_oldest_submitted_at(window_results))
            if not next_before or next_before == before or len(alpha_ids) == ids_before_window:
                # Refuse to continue rather than silently drop data.
                raise RuntimeError('Alpha 时间分页无法继续推进；为避免漏掉数据，已停止本次 Alpha 补齐。')
            before = next_before
            await self._progress(
                phase='alphas-window', success=len(alpha_ids),
                message=f'已获取 {len(alpha_ids)}，继续获取 {before} 之前提交的 Alpha。')

        unique_ids = sorted(alpha_ids)
        await self._db(self.dao.mark_submitted_snapshot, unique_ids)
        return unique_ids

    async def _fetch_missing_submitted_alphas(self, known_ids, missing_count):
        """Walk back from newest until the gap is filled (prodmemoMain.js:245-316)."""
        added, seen = set(), set(known_ids)
        before = ''
        while len(added) < missing_count:
            self._check_stopped()
            first = await self._fetch_alpha_page_persistent(0, ALPHA_PAGE_LIMIT, before)
            if not first['results']:
                break
            window_count = min(int(first['count']), ALPHA_WINDOW_LIMIT)
            window_results = []

            async def collect(page):
                fresh = [a for a in page['results']
                         if a.get('id') and a.get('dateSubmitted')
                         and a.get('status') != 'UNSUBMITTED' and a['id'] not in seen]
                records = [r for r in (normalize_alpha_record(a, True) for a in fresh) if r]
                if records:
                    await self._db(self.dao.save_alpha_batch, records)
                    for record in records:
                        seen.add(record['id'])
                        added.add(record['id'])
                window_results.extend(page['results'])

            await collect(first)
            offset = ALPHA_PAGE_LIMIT
            while offset < window_count and len(added) < missing_count:
                self._check_stopped()
                page = await self._fetch_alpha_page_persistent(offset, ALPHA_PAGE_LIMIT, before)
                await collect(page)
                await self._progress(
                    phase='incremental-alphas', current=len(added), total=missing_count,
                    message=f'已发现 {len(added)}/{missing_count}')
                offset += ALPHA_PAGE_LIMIT

            if len(added) >= missing_count or int(first['count']) <= ALPHA_WINDOW_LIMIT:
                break
            next_before = next_exclusive_upper_bound(_oldest_submitted_at(window_results))
            if not next_before or next_before == before:
                break
            before = next_before
        return sorted(added)

    # --- PnL --------------------------------------------------------------

    async def _fetch_pnls(self, alpha_ids, phase='pnl'):
        """Batch + concurrency-limited PnL fetch with batch-level retry rounds.

        'pending' (platform still generating) is not a failure: those ids go
        into the next round (prodmemoMain.js:381-423).
        """
        pending = list(dict.fromkeys(alpha_ids))
        total = len(pending)
        success, done = 0, 0
        semaphore = asyncio.Semaphore(PNL_CONCURRENCY)

        for round_index in range(len(RETRY_DELAYS) + 1):
            if not pending:
                break
            if round_index:
                await asyncio.sleep(RETRY_DELAYS[min(round_index - 1, len(RETRY_DELAYS) - 1)])
            next_pending = []

            async def worker(alpha_id):
                nonlocal success, done
                async with semaphore:
                    self._check_stopped()
                    records, status = await self._fetch_pnl_once(alpha_id)
                    if status == 'ok':
                        await self._db(self.dao.save_pnl, alpha_id, records,
                                       pnl_fingerprint({'records': records}))
                        success += 1
                    else:
                        next_pending.append(alpha_id)
                    done += 1
                    if done % PROGRESS_EVERY == 0:
                        await self._progress(phase=phase, current=done, total=total,
                                             success=success,
                                             message=f'PnL {done}/{total} · 成功 {success}')

            for batch_start in range(0, len(pending), PNL_BATCH_SIZE):
                self._check_stopped()
                batch = pending[batch_start:batch_start + PNL_BATCH_SIZE]
                await asyncio.gather(*(worker(a) for a in batch))
            pending = next_pending

        return {'success': success, 'failedIds': pending}

    async def _backfill_known_references(self):
        """Complete metadata+PnL for alphas known only through a platform Prod value.

        These become reference curves for the Prod lower bound
        (prodmemoMain.js:472-496).
        """
        state = await self._db(self.dao.get_sync_state)
        targets = state['backfillIds']
        if not targets:
            return {'requested': 0, 'failedIds': []}
        failed = []
        semaphore = asyncio.Semaphore(BACKFILL_CONCURRENCY)

        async def worker(alpha_id):
            async with semaphore:
                self._check_stopped()
                try:
                    await self._ensure_alpha_data(alpha_id, submitted=False)
                except SyncStopped:
                    raise
                except Exception as exc:
                    logger.warning('backfill failed for %s: %s', alpha_id, exc)
                    failed.append(alpha_id)

        await self._progress(phase='backfill', total=len(targets),
                             message=f'回填 {len(targets)} 条参考曲线的元数据与 PnL。')
        await asyncio.gather(*(worker(a) for a in targets))
        return {'requested': len(targets), 'failedIds': failed}

    async def _ensure_alpha_data(self, alpha_id, submitted=None):
        """Make sure one alpha has metadata + PnL locally."""
        existing = await self._db(self.dao.get_alpha, alpha_id)
        if not existing:
            detail = await self.fetcher.get_alpha_details(alpha_id)
            if isinstance(detail, dict) and detail.get('id'):
                record = normalize_alpha_record(
                    detail, submitted if submitted is not None else bool(detail.get('dateSubmitted')))
                if record:
                    await self._db(self.dao.save_alpha_batch, [record])
        snapshot_ids = await self._db(self.dao.load_pnl_points, [alpha_id])
        if alpha_id not in snapshot_ids:
            records, status = await self._fetch_pnl_once(alpha_id)
            for _ in range(PNL_PENDING_RETRIES if status == 'pending' else 0):
                await asyncio.sleep(PNL_PENDING_DELAY)
                records, status = await self._fetch_pnl_once(alpha_id)
                if status != 'pending':
                    break
            if status == 'ok':
                await self._db(self.dao.save_pnl, alpha_id, records,
                               pnl_fingerprint({'records': records}))
            return status
        return 'ok'

    # --- sync entry points ------------------------------------------------

    async def _run_full_sync(self):
        await self._progress(phase='alphas', status='running', mode='full',
                             message='开始全量同步 Submitted Alpha。')
        alpha_ids = await self._fetch_all_submitted_alphas()
        state = await self._db(self.dao.get_sync_state)
        pnl_result = await self._fetch_pnls(state['missingPnlIds'], phase='pnl')
        backfill = await self._backfill_known_references()
        final = await self._db(self.dao.get_sync_state)
        await self._progress(
            status='completed', phase='done', mode='full',
            fullSyncedAt=datetime.now(timezone.utc).isoformat(),
            alphaCount=len(final['alphaIds']),
            submittedPnlCount=len(final['alphaIds']) - len(final['missingPnlIds']),
            pnlSuccess=pnl_result['success'], failedIds=pnl_result['failedIds'],
            backfillFailedIds=backfill['failedIds'],
            message=f"全量同步完成：Alpha {len(alpha_ids)}，PnL 成功 {pnl_result['success']}")
        return {'mode': 'full', 'alphaCount': len(alpha_ids), 'pnl': pnl_result,
                'backfill': backfill}

    async def _run_incremental_sync(self):
        """Count-probe driven incremental sync (prodmemoMain.js:520-644)."""
        state = await self._db(self.dao.get_sync_state)
        await self._progress(phase='incremental-check', status='running', mode='incremental',
                             message='正在用小批量请求探测远端 Submitted Alpha 总数。')
        head = await self._fetch_alpha_page_persistent(0, 1)
        remote_count = int(head['count'])
        local_ids = set(state['alphaIds'])
        local_count = len(local_ids)

        alpha_action, added = 'skipped', []
        if local_count != remote_count:
            if local_count < remote_count:
                alpha_action = 'incremental'
                missing = remote_count - local_count
                await self._progress(phase='incremental-alphas', total=missing,
                                     message=f'Submitted Alpha 少 {missing} 个，从最新记录增量补齐。')
                added = await self._fetch_missing_submitted_alphas(local_ids, missing)
            else:
                alpha_action = 'reconciled'
                await self._progress(
                    phase='incremental-alphas-reconcile',
                    message=f'本地（{local_count}）多于远端（{remote_count}），开始校正快照。')
                await self._fetch_all_submitted_alphas()
            state = await self._db(self.dao.get_sync_state)
            if len(state['alphaIds']) != remote_count:
                alpha_action = 'reconciled'
                await self._progress(
                    phase='incremental-alphas-reconcile',
                    message=f"补齐后仍不一致（{len(state['alphaIds'])}/{remote_count}），完整校正。")
                refreshed = await self._fetch_all_submitted_alphas()
                added = [a for a in refreshed if a not in local_ids]
                state = await self._db(self.dao.get_sync_state)
        else:
            await self._progress(phase='incremental-alphas-skipped', success=local_count,
                                 message=f'Submitted Alpha 数量已一致（{remote_count}），跳过。')

        submitted_pnl_count = len(state['alphaIds']) - len(state['missingPnlIds'])
        pnl_targets = [] if submitted_pnl_count == remote_count \
            else list(dict.fromkeys(state['missingPnlIds']))
        pnl_result = await self._fetch_pnls(pnl_targets, phase='incremental-pnl') \
            if pnl_targets else {'success': 0, 'failedIds': []}
        backfill = await self._backfill_known_references()

        final = await self._db(self.dao.get_sync_state)
        final_missing = set(final['missingPnlIds'])
        pnl_result['failedIds'] = [a for a in pnl_result['failedIds'] if a in final_missing]
        pnl_result['success'] = len(pnl_targets) - len(pnl_result['failedIds'])
        final_pnl_count = len(final['alphaIds']) - len(final['missingPnlIds'])
        await self._progress(
            status='completed', phase='done', mode='incremental',
            incrementalSyncedAt=datetime.now(timezone.utc).isoformat(),
            remoteCount=remote_count, alphaCount=len(final['alphaIds']),
            submittedPnlCount=final_pnl_count, alphaAction=alpha_action, added=len(added),
            pnlRequested=len(pnl_targets), pnlSuccess=pnl_result['success'],
            failedIds=pnl_result['failedIds'], backfillFailedIds=backfill['failedIds'],
            message=(f'增量同步完成：远端 {remote_count} · 本地 Alpha {len(final["alphaIds"])} '
                     f'· PnL {final_pnl_count}'))
        return {'mode': 'incremental', 'remoteCount': remote_count,
                'alphaAction': alpha_action, 'added': len(added),
                'pnlRequested': len(pnl_targets), 'pnl': pnl_result, 'backfill': backfill}

    async def _run_sync(self, mode):
        try:
            if mode == 'full':
                return await self._run_full_sync()
            return await self._run_incremental_sync()
        except SyncStopped as exc:
            await self._progress(status='stopped', phase='stopped', message=str(exc))
        except FatalSyncError as exc:
            await self._progress(status='error', phase='error', error=str(exc),
                                 message=f'认证失败，同步终止：{exc}')
        except Exception as exc:  # noqa: BLE001 - surfaced through sync_meta
            logger.exception('prodmemo sync failed')
            await self._progress(status='error', phase='error', error=str(exc),
                                 message=f'同步失败：{exc}')
        return None

    async def start_sync(self, mode='incremental'):
        await self.ensure_ready()
        if mode == 'stop':
            if not self.is_running:
                return {'started': False, 'stopping': False, 'reason': 'not_running'}
            self._stop_event.set()
            return {'started': False, 'stopping': True}
        async with self._start_lock:  # two concurrent calls must not start two syncs
            if self.is_running:
                return {'started': False, 'reason': 'already_running',
                        'sync': await self._db(self.dao.get_sync_meta)}
            if self.fetcher is None:
                raise RuntimeError('ProdMemo fetcher 未注入（应由 platform_functions.py 设置）。')
            self._stop_event = asyncio.Event()
            # Reset run-scoped keys so a finished run never shows an old error.
            await self._progress(status='running', phase='starting', mode=mode,
                                 startedAt=datetime.now(timezone.utc).isoformat(),
                                 error=None, current=None, total=None, success=None,
                                 message='同步任务已启动。')
            self._sync_task = asyncio.create_task(self._run_sync(mode))
        return {'started': True, 'mode': mode}

    @property
    def is_running(self):
        return self._sync_task is not None and not self._sync_task.done()

    async def sync_status(self):
        await self.ensure_ready()
        meta = await self._db(self.dao.get_sync_meta)
        return {'running': self.is_running, **meta}

    # --- local computation ------------------------------------------------

    def _build_context(self, snapshot, target_alpha, corr_type):
        """Candidates (SELF/POOL) or reference curves (PROD_LOWER_BOUND).

        Mirrors buildCalculationContext (prodMemoService.js:233-257). References
        are intentionally NOT filtered by `submitted`: soft-deleted alphas remain
        valid reference curves as long as their PnL is around.
        """
        pnl_ids = set(snapshot['pnlStubs'])
        if corr_type in ('SELF', 'POOL'):
            candidates = select_candidate_alphas(
                snapshot['alphas'], pnl_ids, target_alpha, corr_type)
            return {'candidates': candidates, 'references': []}

        group_key = alpha_group_key(target_alpha)
        alpha_by_id = {a['id']: a for a in snapshot['alphas']}
        references = []
        for alpha_id, buckets in snapshot['platformCorrs'].items():
            prod = buckets.get('prod') or {}
            prod_max = finite_number(prod.get('max'))
            alpha = alpha_by_id.get(alpha_id)
            if prod_max is None or not alpha or alpha_id not in pnl_ids:
                continue
            if not group_key or alpha_group_key(alpha) != group_key:
                continue
            references.append({'alpha': alpha, 'platformProdMax': prod_max,
                               'platformUpdated': prod.get('updated') or 0})
        return {'candidates': [], 'references': references}

    @staticmethod
    def _stub_pnl_map(snapshot):
        """{'alphaId': {'fingerprint': ...}} — enough for fingerprinting only."""
        return dict(snapshot['pnlStubs'])

    async def calculate_local(self, alpha_id, force=False):
        """Three-tier cache: fingerprint from a light snapshot, load PnL bodies
        only for corr types that actually need recomputing
        (prodMemoService.js:426-516)."""
        await self.ensure_ready()
        snapshot = await self._db(self.dao.light_snapshot)
        target = next((a for a in snapshot['alphas'] if a['id'] == alpha_id), None)
        if not target:
            return {'available': False, 'reason': f'本地没有 Alpha {alpha_id} 的元数据，请先同步。'}
        if alpha_id not in snapshot['pnlStubs']:
            return {'available': False, 'reason': f'本地没有 Alpha {alpha_id} 的 PnL，请先同步。'}

        def describe():
            stub_map = self._stub_pnl_map(snapshot)
            existing = {(r['alphaId'], r['corrType']): r for r in snapshot['localCorrs']}
            out = []
            for corr_type in CORR_TYPES:
                context = self._build_context(snapshot, target, corr_type)
                fingerprint = calculation_fingerprint(
                    corr_type=corr_type, target_alpha=target, target_pnl=stub_map[alpha_id],
                    pnl_by_id=stub_map, candidates=context['candidates'],
                    references=context['references'])
                record = existing.get((alpha_id, corr_type))
                fresh = (not force and record
                         and record['algorithmVersion'] == PROD_MEMO_ALGORITHM_VERSION
                         and record['inputFingerprint'] == fingerprint)
                out.append({'corrType': corr_type, 'context': context,
                            'fingerprint': fingerprint, 'record': record,
                            'fresh': bool(fresh)})
            return out

        # Pure CPU work (hundreds of alphas x thousands of PnL points): keep it off
        # the event loop so other MCP clients are not stalled.
        descriptors = await asyncio.to_thread(describe)

        if all(d['fresh'] for d in descriptors):
            return {'available': True, 'reused': True, 'alphaId': alpha_id,
                    'results': {d['corrType']: d['record']['result'] for d in descriptors}}

        needed = {alpha_id}
        for descriptor in descriptors:
            if descriptor['fresh']:
                continue
            needed.update(a['id'] for a in descriptor['context']['candidates'])
            needed.update(r['alpha']['id'] for r in descriptor['context']['references'])
        pnl_by_id = await self._db(self.dao.load_pnl_points, sorted(needed))

        group_key = alpha_group_key(target)

        def compute():
            rows, results = [], {}
            for descriptor in descriptors:
                corr_type = descriptor['corrType']
                if descriptor['fresh']:
                    results[corr_type] = descriptor['record']['result']
                    continue
                if corr_type == 'PROD_LOWER_BOUND':
                    result = calculate_prod_lower_bound(
                        target_alpha=target, target_pnl=pnl_by_id.get(alpha_id),
                        references=descriptor['context']['references'], pnl_by_id=pnl_by_id)
                else:
                    result = calculate_local_correlation(
                        target_alpha=target, target_pnl=pnl_by_id.get(alpha_id),
                        candidates=descriptor['context']['candidates'],
                        pnl_by_id=pnl_by_id, corr_type=corr_type)
                results[corr_type] = result
                rows.append({'alphaId': alpha_id, 'corrType': corr_type, 'groupKey': group_key,
                             'result': result, 'algorithmVersion': PROD_MEMO_ALGORITHM_VERSION,
                             'inputFingerprint': descriptor['fingerprint']})
            return rows, results

        rows, results = await asyncio.to_thread(compute)
        if rows:
            await self._db(self.dao.save_local_corrs, rows)
        return {'available': True, 'reused': False, 'alphaId': alpha_id, 'results': results}

    def _decorate_local(self, snapshot, alpha_id, stub_map=None):
        """Attach a live `stale` flag by recomputing fingerprints
        (prodMemoService.js:272-278). Stale local values are never shown."""
        target = next((a for a in snapshot['alphas'] if a['id'] == alpha_id), None)
        stub_map = stub_map if stub_map is not None else self._stub_pnl_map(snapshot)
        existing = {(r['alphaId'], r['corrType']): r for r in snapshot['localCorrs']}
        decorated = {}
        for corr_type in CORR_TYPES:
            record = existing.get((alpha_id, corr_type))
            if not record:
                decorated[corr_type] = None
                continue
            stale = True
            if target and alpha_id in stub_map:
                context = self._build_context(snapshot, target, corr_type)
                fingerprint = calculation_fingerprint(
                    corr_type=corr_type, target_alpha=target, target_pnl=stub_map[alpha_id],
                    pnl_by_id=stub_map, candidates=context['candidates'],
                    references=context['references'])
                stale = (record['algorithmVersion'] != PROD_MEMO_ALGORITHM_VERSION
                         or fingerprint != record['inputFingerprint'])
            decorated[corr_type] = {**record, 'stale': stale}
        return decorated

    @staticmethod
    def _resolved(platform_buckets, local_decorated):
        """Ⓟ platform first, Ⓛ non-stale local fallback, ≥ for the lower bound."""
        buckets = platform_buckets or {}
        out = {}
        for key, corr_type, lower in (('self', 'SELF', False),
                                      ('pool', 'POOL', False),
                                      ('prod', 'PROD_LOWER_BOUND', True)):
            metric = resolve_preferred_correlation(
                buckets.get(key), local_decorated.get(corr_type), {'lowerBound': lower})
            out[key] = {**metric, 'display': format_resolved_metric(metric)} if metric else None
        return out

    @staticmethod
    def _calibration_pairs(snapshot):
        """(local POOL max, platform Prod max) for every alpha that has both."""
        pool_by_id = {}
        for rec in snapshot['localCorrs']:
            result = rec.get('result') or {}
            if rec.get('corrType') == 'POOL' and result.get('available'):
                value = finite_number(result.get('max'))
                if value is not None:
                    pool_by_id[rec['alphaId']] = value
        pairs = []
        for alpha_id, buckets in snapshot['platformCorrs'].items():
            prod = finite_number(((buckets or {}).get('prod') or {}).get('max'))
            if prod is not None and alpha_id in pool_by_id:
                pairs.append((pool_by_id[alpha_id], prod))
        return pairs

    def _prod_estimate(self, snapshot, pool_record):
        """prod_est from this alpha's fresh local POOL max, or None."""
        result = (pool_record or {}).get('result') or {}
        pool = finite_number(result.get('max')) if result.get('available') else None
        if pool is None or (pool_record or {}).get('stale'):
            return None
        a, b, n, resid_sd, source = fit_prod_estimate(self._calibration_pairs(snapshot))
        value = max(-1.0, min(1.0, a + b * pool))
        return {'value': round(value, 4), 'pool': pool, 'a': round(a, 4), 'b': round(b, 4),
                'calibration_points': n, 'resid_sd': None if resid_sd is None else round(resid_sd, 4),
                'source': source, 'over_threshold': value > PROD_THRESHOLD}

    async def record_platform_corr(self, alpha_id, corr_type, max_value, min_value=None,
                                   source='platform'):
        """Store an officially measured correlation. A Prod value turns the alpha
        into a reference curve / calibration point, so its metadata + PnL are
        fetched too (the flywheel)."""
        await self.ensure_ready()
        value = finite_number(max_value)
        if value is None:
            return False
        await self._db(self.dao.save_platform_corr, alpha_id, corr_type, {
            'max': value, 'min': finite_number(min_value),
            'updated': int(time.time() * 1000), 'source': source,
        })
        if corr_type == 'prod' and self.fetcher is not None:
            try:
                await self._ensure_alpha_data(alpha_id)
            except Exception as exc:
                logger.warning('reference backfill failed for %s: %s', alpha_id, exc)
        return True

    # --- public API used by MCP tools -------------------------------------

    async def check_many(self, alpha_ids, run_platform_check=False, verbose=False):
        """check() for up to CHECK_MANY_LIMIT alphas; compact rows unless verbose."""
        ids = list(dict.fromkeys(a.strip() for a in alpha_ids if a and a.strip()))
        if not ids:
            raise ValueError('alpha_ids is empty')
        if len(ids) > CHECK_MANY_LIMIT:
            raise ValueError(f'at most {CHECK_MANY_LIMIT} alphas per call')
        gate = asyncio.Semaphore(CHECK_MANY_CONCURRENCY)

        async def one(alpha_id):
            async with gate:
                try:
                    full = await self.check(alpha_id, run_platform_check)
                except Exception as exc:
                    return {'alpha_id': alpha_id, 'error': str(exc)}
            return full if verbose else self.compact_check(full)

        results = await asyncio.gather(*(one(a) for a in ids))
        return {'count': len(results), 'results': list(results)}

    @staticmethod
    def compact_check(full):
        """One short row out of a check() result."""
        if 'local' not in full:
            return full  # insufficient_data / error rows are already short
        local = full['local']
        platform_prod = finite_number((full['platform'].get('prod') or {}).get('max'))
        est = full.get('prod_est')
        return {
            'alpha_id': full['alpha_id'],
            'recommendation': full['recommendation'],
            'prod_est': est['value'] if est else None,
            'pool': (local.get('pool') or {}).get('max'),
            'self': (local.get('self') or {}).get('max'),
            'prod_lower_bound': (local.get('prod_lower_bound') or {}).get('max'),
            'platform_prod': platform_prod,
            'platform_status': full['platform_status'],
        }

    async def check(self, alpha_id, run_platform_check=False):
        """Decision facade: should this alpha's platform Prod check be skipped?"""
        await self.ensure_ready()
        if self.fetcher is None:
            raise RuntimeError('ProdMemo fetcher 未注入。')

        status = await self._ensure_alpha_data(alpha_id)
        if status != 'ok':
            return {'alpha_id': alpha_id, 'recommendation': 'insufficient_data',
                    'reason': f'PnL 尚未就绪（status={status}），请稍后重试或先执行同步。'}

        platform_status = 'not_requested'
        if run_platform_check:
            platform_status = await self._refresh_platform_corrs(alpha_id)

        await self.calculate_local(alpha_id)
        snapshot = await self._db(self.dao.light_snapshot)
        local = self._decorate_local(snapshot, alpha_id)
        platform = snapshot['platformCorrs'].get(alpha_id) or {}
        resolved = self._resolved(platform, local)

        lower = local.get('PROD_LOWER_BOUND')
        lower_result = (lower or {}).get('result') or {}
        lower_max = finite_number(lower_result.get('max')) if lower_result.get('available') else None
        lower_fresh = bool(lower and not lower['stale'])

        prod_est = self._prod_estimate(snapshot, local.get('POOL'))

        if lower_max is None or not lower_fresh:
            recommendation = 'insufficient_data' if lower_max is None and prod_est is None else 'check'
        elif lower_max > PROD_THRESHOLD:
            recommendation = 'skip'
        else:
            recommendation = 'check'

        return {
            'alpha_id': alpha_id,
            'recommendation': recommendation,
            'threshold': PROD_THRESHOLD,
            'local': {
                'self': self._public_local(local.get('SELF')),
                'pool': self._public_local(local.get('POOL')),
                'prod_lower_bound': self._public_local(local.get('PROD_LOWER_BOUND'),
                                                       include_witness=True),
            },
            'platform': {k: platform.get(k) for k in ('prod', 'pool', 'self')},
            'platform_status': platform_status,
            'prod_est': prod_est,
            'resolved': resolved,
            'assumption': ('本地下限基于"平台与本地同口径（四年窗口 / 前向填充差分 / Pearson）"假设，'
                           '为条件下限；口径不一致时不再严格成立。prod_est 是 pool→prod 的经验线性换算'
                           '（source=default 为手工 4 点标定，fitted 为按已测 Prod 回归），仅供排序参考。'),
        }

    @staticmethod
    def _public_local(record, include_witness=False):
        if not record:
            return None
        result = record.get('result') or {}
        out = {
            'available': bool(result.get('available')),
            'max': result.get('max'),
            'min': result.get('min'),
            'stale': record.get('stale'),
            'calculatedAt': record.get('calculatedAt'),
            'corrCount': result.get('corrCount'),
            'maxOverlapCount': result.get('maxOverlapCount'),
            'windowStart': result.get('windowStart'),
            'windowEnd': result.get('windowEnd'),
        }
        if not result.get('available'):
            out['reason'] = result.get('reason')
        if include_witness:
            out['witness'] = result.get('witness')
            out['referenceCount'] = result.get('referenceCount')
        return out

    async def _refresh_platform_corrs(self, alpha_id):
        """Query platform Prod/Self and persist. Empty payload = still computing.

        NOTE: deliberately bypasses brain_client.check_correlation, which
        collapses "not ready" and "failed" into the same empty/error shape.
        """
        statuses = []
        for corr_type, method in (('prod', 'get_production_correlation'),
                                  ('self', 'get_self_correlation')):
            fn = getattr(self.fetcher, method, None)
            if fn is None:
                continue
            try:
                payload = await fn(alpha_id)
            except Exception as exc:
                logger.warning('platform %s corr failed for %s: %s', corr_type, alpha_id, exc)
                statuses.append('failed')
                continue
            stats = extract_platform_correlation_stats(payload) if payload else None
            if not stats or stats.get('max') is None:
                statuses.append('pending')
                continue
            await self.record_platform_corr(alpha_id, corr_type, stats['max'], stats.get('min'))
            statuses.append('fresh')
        if not statuses:
            return 'not_requested'
        if 'fresh' in statuses:
            return 'fresh'
        return 'pending' if 'pending' in statuses else 'failed'

    async def get(self, alpha_id='', stale_only=False, above=0.0, group_key='', limit=100):
        """Single-alpha status card, or a filtered list."""
        await self.ensure_ready()
        snapshot = await self._db(self.dao.light_snapshot)
        stub_map = self._stub_pnl_map(snapshot)
        alpha_by_id = {a['id']: a for a in snapshot['alphas']}

        def card(target_id):
            alpha = alpha_by_id.get(target_id)
            local = self._decorate_local(snapshot, target_id, stub_map)
            platform = snapshot['platformCorrs'].get(target_id) or {}
            stub = stub_map.get(target_id) or {}
            return {
                'alpha_id': target_id,
                'name': (alpha or {}).get('name'),
                'stage': (alpha or {}).get('stage'),
                'group_key': alpha_group_key(alpha) if alpha else '',
                'date_submitted': (alpha or {}).get('dateSubmitted'),
                'submitted': (alpha or {}).get('submitted'),
                'no_longer_submitted': (alpha or {}).get('noLongerSubmitted'),
                'has_metadata': alpha is not None,
                'has_pnl': target_id in stub_map,
                'pnl_last_date': stub.get('lastDate'),
                'pnl_points': stub.get('pointCount'),
                'platform': {k: platform.get(k) for k in ('prod', 'pool', 'self')},
                'local': {
                    'self': self._public_local(local.get('SELF')),
                    'pool': self._public_local(local.get('POOL')),
                    'prod_lower_bound': self._public_local(local.get('PROD_LOWER_BOUND'),
                                                           include_witness=True),
                },
                'resolved': self._resolved(platform, local),
            }

        if alpha_id:
            if alpha_id not in alpha_by_id and alpha_id not in snapshot['platformCorrs']:
                return {'found': False, 'alpha_id': alpha_id,
                        'reason': '本地没有该 Alpha 的任何记录，请先同步或执行 prodmemo_check。'}
            return {'found': True, **await asyncio.to_thread(card, alpha_id)}
        return await asyncio.to_thread(self._list_cards, snapshot, alpha_by_id, card,
                                       stale_only, above, group_key, limit)

    @staticmethod
    def _list_cards(snapshot, alpha_by_id, card, stale_only, above, group_key, limit):
        candidate_ids = sorted(
            set(alpha_by_id) | set(snapshot['platformCorrs']),
            key=lambda i: (alpha_by_id.get(i, {}).get('dateSubmitted') or '', i),
            reverse=True)
        rows = []
        for candidate in candidate_ids:
            if group_key and alpha_group_key(alpha_by_id.get(candidate) or {}) != group_key:
                continue
            entry = card(candidate)
            if stale_only and not any(
                    (entry['local'][k] or {}).get('stale') for k in entry['local']):
                continue
            if above:
                maxima = [m['max'] for m in entry['resolved'].values() if m]
                if not maxima or max(maxima) < above:
                    continue
            rows.append(entry)
            if len(rows) >= limit:
                break
        return {'total': len(rows), 'rows': rows,
                'note': '列表按提交时间降序；resolved 为 Ⓟ 平台优先、Ⓛ 本地回退（过期本地值不展示）。'}

    async def stats(self):
        await self.ensure_ready()
        stats = await self._db(self.dao.stats)
        stats['running'] = self.is_running
        return stats

    async def manage(self, action, alpha_id='', data=''):
        await self.ensure_ready()
        if action == 'export':
            return await self._db(self.dao.export_corrs)
        if action == 'import':
            return await self._import_corrs(data)
        if action == 'clear_corrs':
            return await self._db(self.dao.clear_corrs)
        if action == 'clear_sync':
            return await self._db(self.dao.clear_sync_data)
        if action == 'delete':
            if not alpha_id:
                raise ValueError('delete 需要 alpha_id。')
            return await self._db(self.dao.delete_alpha_corrs, alpha_id)
        raise ValueError(f'未知的 ProdMemo 管理操作：{action}')

    async def _import_corrs(self, data):
        """Import the extension's export payload (schemaVersion 2, v1 tolerated).

        This is the migration path for platform Prod values captured by the
        browser extension — they cannot be re-derived server-side.
        """
        payload = json.loads(data) if isinstance(data, str) else data
        if not isinstance(payload, dict):
            raise ValueError('导入数据必须是 JSON 对象。')

        imported, skipped = 0, 0
        platform = payload.get('platformCorrs') or {}
        for raw_id, buckets in platform.items():
            alpha_id = str(raw_id or '').replace('WQP_ProdMemo_', '').strip()
            if not alpha_id or not isinstance(buckets, dict):
                skipped += 1
                continue
            # v1 legacy: {'timestamp': ..., 'result': {...}} meaning a prod value
            if 'result' in buckets and not any(k in buckets for k in ('prod', 'pool', 'self')):
                buckets = {'prod': {**(buckets.get('result') or {}),
                                    'updated': buckets.get('timestamp')}}
            for key, value in buckets.items():
                corr_type = normalize_correlation_type(key)
                if not isinstance(value, dict):
                    continue
                stats = extract_platform_correlation_stats(value)
                if not stats or stats.get('max') is None:
                    skipped += 1
                    continue
                await self._db(self.dao.save_platform_corr, alpha_id, corr_type, {
                    'max': stats['max'], 'min': stats.get('min'),
                    'updated': value.get('updated') or value.get('timestamp')
                    or int(time.time() * 1000),
                    'source': value.get('source') or 'import',
                })
                imported += 1

        rows = []
        for record in payload.get('localCorrs') or []:
            if not isinstance(record, dict):
                continue
            alpha_id = str(record.get('alphaId') or '').strip()
            corr_type = record.get('corrType')
            if not alpha_id or corr_type not in CORR_TYPES or not record.get('result'):
                skipped += 1
                continue
            rows.append({'alphaId': alpha_id, 'corrType': corr_type,
                         'groupKey': record.get('groupKey') or '',
                         'result': record['result'],
                         'algorithmVersion': record.get('algorithmVersion')
                         or PROD_MEMO_ALGORITHM_VERSION,
                         'inputFingerprint': record.get('inputFingerprint') or ''})
        if rows:
            await self._db(self.dao.save_local_corrs, rows)
            imported += len(rows)
        return {'imported': imported, 'skipped': skipped}


# Module-level singleton, mirroring forum_functions.forum_client.
# platform_functions.py injects the BRAIN client: prodmemo_client.fetcher = brain_client
prodmemo_client = ProdMemoService()

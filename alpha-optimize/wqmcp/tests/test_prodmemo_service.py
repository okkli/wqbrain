"""Sync-engine and orchestration tests.

Mirrors the scenarios in the extension's ``tests/prodMemoMainSync.test.mjs``:
the 1000-row window boundary, in-place retry of a failed page, and the three
incremental-probe branches. Network and DB are both stubbed — the point is to
assert on *request shapes and counts*, which is what makes "never skip a page"
and "probe then skip" verifiable.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prodmemo_calc import pnl_fingerprint  # noqa: E402
from prodmemo_service import (  # noqa: E402
    ProdMemoService,
    next_exclusive_upper_bound,
    normalize_alpha_record,
)


class StubDao:
    """In-memory stand-in with the same surface prodmemo_service uses."""

    def __init__(self):
        self.alphas = {}
        self.pnls = {}          # alpha_id -> {'fingerprint', 'records'}
        self.platform = {}
        self.local = {}
        self.meta = {}
        self.save_alpha_calls = []
        self.save_pnl_calls = []

    def ensure_schema(self):
        pass

    def save_alpha_batch(self, records):
        self.save_alpha_calls.append([r['id'] for r in records])
        for record in records:
            existing = self.alphas.get(record['id'])
            merged = dict(record)
            if existing:
                merged['submitted'] = existing.get('submitted') or record.get('submitted')
            merged['noLongerSubmitted'] = False
            self.alphas[record['id']] = merged
        return {'saved': len(records)}

    def save_pnl(self, alpha_id, records, fingerprint):
        self.save_pnl_calls.append(alpha_id)
        if self.pnls.get(alpha_id, {}).get('fingerprint') == fingerprint:
            return {'unchanged': True, 'points': 0}
        self.pnls[alpha_id] = {'fingerprint': fingerprint, 'records': records}
        return {'unchanged': False, 'points': len(records)}

    def mark_submitted_snapshot(self, keep_ids):
        keep = set(keep_ids)
        marked = 0
        for alpha_id, alpha in self.alphas.items():
            if alpha.get('submitted') and alpha_id not in keep:
                alpha['submitted'] = False
                alpha['noLongerSubmitted'] = True
                marked += 1
        return {'marked': marked}

    def save_platform_corr(self, alpha_id, corr_type, stat):
        self.platform.setdefault(alpha_id, {})[corr_type] = stat
        return {'saved': True}

    def save_local_corrs(self, rows):
        for row in rows:
            self.local[(row['alphaId'], row['corrType'])] = dict(row)
        return {'saved': len(rows)}

    def get_alpha(self, alpha_id):
        return self.alphas.get(alpha_id)

    def load_pnl_points(self, alpha_ids):
        return {a: {'alphaId': a, 'records': self.pnls[a]['records']}
                for a in alpha_ids if a in self.pnls}

    def light_snapshot(self):
        return {
            'alphas': [dict(a, settings={'region': a.get('region'),
                                         'universe': a.get('universe'),
                                         'delay': a.get('delay'),
                                         'instrumentType': a.get('instrumentType')})
                       for a in self.alphas.values()],
            'pnlStubs': {a: {'alphaId': a, 'fingerprint': p['fingerprint'],
                             'lastDate': p['records'][-1][0] if p['records'] else None,
                             'pointCount': len(p['records'])}
                         for a, p in self.pnls.items()},
            'platformCorrs': self.platform,
            'localCorrs': [dict(v) for v in self.local.values()],
            'sync': self.meta,
        }

    def get_sync_state(self):
        submitted = [a for a, v in self.alphas.items() if v.get('submitted')]
        return {
            'alphaIds': submitted,
            'missingPnlIds': [a for a in submitted if a not in self.pnls],
            'backfillIds': [a for a, b in self.platform.items()
                            if (b.get('prod') or {}).get('max') is not None
                            and (a not in self.alphas or a not in self.pnls)],
            'sync': self.meta,
        }

    def set_sync_meta(self, patch):
        self.meta.update(patch)
        return self.meta

    def get_sync_meta(self):
        return self.meta

    def stats(self):
        return {'submitted_alpha_count': sum(1 for v in self.alphas.values()
                                             if v.get('submitted')),
                'pnl_count': len(self.pnls), 'platform_corr_count': len(self.platform),
                'local_corr_count': len(self.local), 'valid_reference_count': 0,
                'sync': self.meta}


class StubFetcher:
    """Fake BRAIN API. Unknown request shapes raise, so unplanned traffic fails
    the test rather than passing silently."""

    def __init__(self, total=2, fail_offsets=(), window_limit=1000):
        self.total = total
        self.fail_offsets = dict.fromkeys(fail_offsets, 1)   # offset -> times left to fail
        self.window_limit = window_limit
        self.alpha_calls = []      # (offset, limit, before)
        self.pnl_calls = []
        self.detail_calls = []

    def _all(self):
        # newest first; dateSubmitted descends by one minute per index
        return [{
            'id': f'alpha-{i}',
            'dateSubmitted': f'2026-08-07T{23 - i // 60:02d}:{59 - i % 60:02d}:00+00:00',
            'status': 'ACTIVE', 'stage': 'OS', 'name': f'alpha-{i}',
            'settings': {'region': 'USA', 'universe': 'TOP3000', 'delay': 1,
                         'instrumentType': 'EQUITY'},
            'classifications': [{'id': 'REGULAR:REGULAR'}], 'is': {'sharpe': 1.0},
        } for i in range(self.total)]

    async def get_user_alphas(self, **kwargs):
        offset = kwargs.get('offset', 0)
        limit = kwargs.get('limit', 100)
        before = kwargs.get('submission_end_date') or ''
        self.alpha_calls.append((offset, limit, before))

        if self.fail_offsets.get(offset):
            self.fail_offsets[offset] -= 1
            raise ConnectionError('temporary connection failure')

        rows = self._all()
        if before:
            rows = [r for r in rows if r['dateSubmitted'] < before]
        count = len(rows)
        visible = rows[:self.window_limit]
        return {'count': count, 'results': visible[offset:offset + limit]}

    async def get_alpha_pnl(self, alpha_id):
        self.pnl_calls.append(alpha_id)
        return {'records': [['2026-01-01', 0.0], ['2026-01-02', 1.0], ['2026-01-03', 2.5]]}

    async def get_alpha_details(self, alpha_id):
        self.detail_calls.append(alpha_id)
        return next((a for a in self._all() if a['id'] == alpha_id), None)


def make_service(fetcher, dao=None):
    """Build the service and clear DAO call logs, so assertions about writes see
    only what the sync did — not what seed_alpha() wrote during setup."""
    service = ProdMemoService(dao=dao or StubDao(), fetcher=fetcher)
    service._schema_ready = True
    service.dao.save_alpha_calls.clear()
    service.dao.save_pnl_calls.clear()
    return service


def seed_alpha(dao, alpha_id, submitted=True, with_pnl=True, values=(0.0, 1.0, 2.5),
               classifications=None, stage='OS'):
    raw = {'id': alpha_id, 'name': alpha_id, 'stage': stage,
           'dateSubmitted': '2026-08-01T00:00:00+00:00', 'status': 'ACTIVE',
           'settings': {'region': 'USA', 'universe': 'TOP3000', 'delay': 1,
                        'instrumentType': 'EQUITY'},
           'classifications': classifications or [{'id': 'REGULAR:REGULAR'}],
           'is': {'sharpe': 1.0}}
    dao.save_alpha_batch([normalize_alpha_record(raw, submitted)])
    if with_pnl:
        records = [[f'2026-01-0{i + 1}', v] for i, v in enumerate(values)]
        dao.save_pnl(alpha_id, records, pnl_fingerprint({'records': records}))


def test_next_exclusive_upper_bound_adds_one_millisecond():
    assert next_exclusive_upper_bound('2026-08-07T12:00:00+00:00') == '2026-08-07T12:00:00.001000Z'
    assert next_exclusive_upper_bound('bad') == ''


@pytest.mark.asyncio
async def test_full_sync_crosses_the_1000_window_and_retries_a_failed_page():
    fetcher = StubFetcher(total=1500, fail_offsets=(300,))
    service = make_service(fetcher)
    result = await service._run_full_sync()

    saved = [i for call in service.dao.save_alpha_calls for i in call]
    assert len(set(saved)) == 1500, 'every alpha must survive the window split, de-duplicated'
    assert result['alphaCount'] == 1500

    unfiltered = [offset for offset, _limit, before in fetcher.alpha_calls if not before]
    assert max(unfiltered) == 900, 'must not request past the 1000-row window'
    assert any(before for _o, _l, before in fetcher.alpha_calls), \
        'must switch to a dateSubmitted< cursor for the second window'

    index = unfiltered.index(300)
    assert unfiltered[index:index + 2] == [300, 300], \
        'a failed page is retried in place, never skipped'


@pytest.mark.asyncio
async def test_incremental_probe_skips_when_counts_match():
    dao = StubDao()
    seed_alpha(dao, 'alpha-0')
    seed_alpha(dao, 'alpha-1')
    fetcher = StubFetcher(total=2)
    service = make_service(fetcher, dao)

    result = await service._run_incremental_sync()

    assert len(fetcher.alpha_calls) == 1, 'only the limit=1 count probe'
    assert fetcher.alpha_calls[0][1] == 1
    assert fetcher.pnl_calls == []
    assert dao.save_alpha_calls == []
    assert result['alphaAction'] == 'skipped'
    assert result['pnlRequested'] == 0


@pytest.mark.asyncio
async def test_incremental_probe_fetches_only_the_missing_pnl():
    dao = StubDao()
    seed_alpha(dao, 'alpha-0')
    seed_alpha(dao, 'alpha-1', with_pnl=False)
    fetcher = StubFetcher(total=2)
    service = make_service(fetcher, dao)

    result = await service._run_incremental_sync()

    assert len(fetcher.alpha_calls) == 1, 'alpha count matches -> no list fetch'
    assert fetcher.pnl_calls == ['alpha-1']
    assert dao.save_alpha_calls == []
    assert result['alphaAction'] == 'skipped'
    assert result['pnlRequested'] == 1


@pytest.mark.asyncio
async def test_incremental_probe_fetches_only_the_missing_alpha_then_its_pnl():
    dao = StubDao()
    seed_alpha(dao, 'alpha-1')
    fetcher = StubFetcher(total=2)
    service = make_service(fetcher, dao)

    result = await service._run_incremental_sync()

    saved = [i for call in dao.save_alpha_calls for i in call]
    assert saved == ['alpha-0'], 'only the missing alpha is written'
    assert fetcher.pnl_calls == ['alpha-0'], 'its PnL is cascaded'
    assert result['alphaAction'] == 'incremental'
    assert result['pnlRequested'] == 1


@pytest.mark.asyncio
async def test_incremental_reconciles_when_local_exceeds_remote():
    dao = StubDao()
    for i in range(4):
        seed_alpha(dao, f'alpha-{i}')
    fetcher = StubFetcher(total=2)
    service = make_service(fetcher, dao)

    result = await service._run_incremental_sync()

    assert result['alphaAction'] == 'reconciled'
    assert dao.alphas['alpha-3']['submitted'] is False
    assert dao.alphas['alpha-3']['noLongerSubmitted'] is True
    assert dao.alphas['alpha-0']['submitted'] is True
    # soft delete only: PnL is retained so it still works as a reference curve
    assert 'alpha-3' in dao.pnls


@pytest.mark.asyncio
async def test_local_calculation_caches_on_fingerprint_and_invalidates_on_change():
    dao = StubDao()
    seed_alpha(dao, 'target', values=(0.0, 1.0, 3.0, 6.0))
    seed_alpha(dao, 'peer', values=(0.0, 2.0, 6.0, 12.0))
    service = make_service(StubFetcher(total=2), dao)

    first = await service.calculate_local('target')
    assert first['reused'] is False
    assert first['results']['SELF']['available'] is True
    assert first['results']['SELF']['max'] == 1

    second = await service.calculate_local('target')
    assert second['reused'] is True, 'unchanged inputs must hit the fingerprint cache'

    seed_alpha(dao, 'peer', values=(0.0, 2.0, 7.0, 13.0))
    third = await service.calculate_local('target')
    assert third['reused'] is False, 'a changed candidate PnL must invalidate the cache'


@pytest.mark.asyncio
async def test_prod_lower_bound_drives_the_skip_recommendation():
    dao = StubDao()
    seed_alpha(dao, 'target', values=(0.0, 1.0, 3.0, 6.0))
    seed_alpha(dao, 'ref', values=(0.0, 2.0, 6.0, 12.0))
    # ref is perfectly correlated with target and platform says its Prod is 0.95,
    # so target's Prod is provably >= 0.95 > 0.7 -> skip the platform check.
    dao.save_platform_corr('ref', 'prod', {'max': 0.95, 'min': 0.1, 'updated': 1, 'source': 'platform'})
    service = make_service(StubFetcher(total=2), dao)

    result = await service.check('target')
    assert result['recommendation'] == 'skip'
    bound = result['local']['prod_lower_bound']
    assert bound['max'] == 0.95
    assert bound['witness']['alphaId'] == 'ref'
    assert bound['stale'] is False

    card = await service.get(alpha_id='target')
    assert card['found'] is True
    assert card['resolved']['prod']['display'].startswith('≥0.9500')
    assert card['resolved']['prod']['icon'] == 'Ⓛ'


@pytest.mark.asyncio
async def test_platform_value_wins_over_local_in_resolution():
    dao = StubDao()
    seed_alpha(dao, 'target', values=(0.0, 1.0, 3.0, 6.0))
    seed_alpha(dao, 'peer', values=(0.0, 2.0, 6.0, 12.0))
    service = make_service(StubFetcher(total=2), dao)
    await service.calculate_local('target')
    dao.save_platform_corr('target', 'self',
                           {'max': 0.42, 'min': -0.1, 'updated': 9, 'source': 'platform'})

    card = await service.get(alpha_id='target')
    assert card['resolved']['self']['source'] == 'platform'
    assert card['resolved']['self']['max'] == 0.42
    assert card['resolved']['self']['display'] == '0.4200 Ⓟ'


@pytest.mark.asyncio
async def test_import_accepts_extension_export_payloads():
    service = make_service(StubFetcher(total=1))
    payload = {
        'schemaVersion': 2,
        'platformCorrs': {
            'A1': {'prod': {'max': 0.62, 'min': -0.3, 'updated': 111, 'source': 'platform'},
                   'self': {'max': 0.55, 'min': -0.2, 'updated': 111}},
            # v1 legacy shape: bare {timestamp, result} means a prod value
            'WQP_ProdMemo_A2': {'timestamp': 222, 'result': {'max': 0.71, 'min': 0.0}},
        },
        'localCorrs': [{'alphaId': 'A1', 'corrType': 'SELF', 'groupKey': 'USA|TOP3000|D1',
                        'result': {'available': True, 'max': 0.6},
                        'algorithmVersion': 2, 'inputFingerprint': 'ff'}],
    }
    result = await service.manage('import', data=payload)
    assert result['imported'] == 4
    assert service.dao.platform['A1']['prod']['max'] == 0.62
    assert service.dao.platform['A2']['prod']['max'] == 0.71, 'legacy key prefix is stripped'
    assert ('A1', 'SELF') in service.dao.local


@pytest.mark.asyncio
async def test_stop_request_halts_a_running_sync():
    service = make_service(StubFetcher(total=2))
    assert (await service.start_sync('stop'))['reason'] == 'not_running'
    service._stop_event.set()
    with pytest.raises(Exception):
        service._check_stopped()


class LazyPnlFetcher(StubFetcher):
    """前 `pending_calls` 次 PnL GET 返回空 body(懒生成首查触发), 之后才有数据。"""

    def __init__(self, pending_calls=2):
        super().__init__()
        self.pending_calls = pending_calls

    async def get_alpha_pnl(self, alpha_id):
        self.pnl_calls.append(alpha_id)
        if len(self.pnl_calls) <= self.pending_calls:
            return {}
        return {'records': [['2026-01-01', 0.0], ['2026-01-02', 1.0], ['2026-01-03', 2.5]]}


@pytest.mark.asyncio
async def test_ensure_alpha_data_retries_lazy_pnl(monkeypatch):
    monkeypatch.setattr('prodmemo_service.PNL_PENDING_DELAY', 0)
    fetcher = LazyPnlFetcher(pending_calls=2)
    service = make_service(fetcher)
    assert await service._ensure_alpha_data('alpha-0') == 'ok'
    assert len(fetcher.pnl_calls) == 3, '两次空响应后第三次拿到数据'
    assert service.dao.save_pnl_calls == ['alpha-0']
    # PnL 入库后再次调用直接命中本地, 不再发请求
    assert await service._ensure_alpha_data('alpha-0') == 'ok'
    assert len(fetcher.pnl_calls) == 3


@pytest.mark.asyncio
async def test_ensure_alpha_data_bounds_pending_retries(monkeypatch):
    monkeypatch.setattr('prodmemo_service.PNL_PENDING_DELAY', 0)
    monkeypatch.setattr('prodmemo_service.PNL_PENDING_RETRIES', 3)
    fetcher = LazyPnlFetcher(pending_calls=99)
    service = make_service(fetcher)
    assert await service._ensure_alpha_data('alpha-0') == 'pending'
    assert len(fetcher.pnl_calls) == 4, '首次 + 至多 3 次重试后放弃'
    assert service.dao.save_pnl_calls == []


def test_fit_prod_estimate_defaults_until_enough_points_then_refits():
    from prodmemo_service import PROD_EST_DEFAULT, fit_prod_estimate
    a, b, n, _sd, source = fit_prod_estimate([(0.2, 0.65)] * 3)
    assert (a, b, source) == (*PROD_EST_DEFAULT, 'default') and n == 3

    pairs = [(x / 10, 0.3 + 2 * x / 10) for x in range(8)]
    a, b, n, sd, source = fit_prod_estimate(pairs)
    assert source == 'fitted' and n == 8
    assert abs(a - 0.3) < 1e-9 and abs(b - 2.0) < 1e-9 and sd < 1e-9


@pytest.mark.asyncio
async def test_check_reports_prod_est_from_local_pool_and_compacts():
    dao = StubDao()
    seed_alpha(dao, 'target', values=(0.0, 1.0, 3.0, 6.0))
    seed_alpha(dao, 'pp', values=(0.0, 2.0, 6.0, 12.0),
               classifications=[{'id': 'POWER_POOL:POWER_POOL_ELIGIBLE'}])
    service = make_service(StubFetcher(total=2), dao)

    full = await service.check('target')
    pool = full['local']['pool']['max']
    assert pool is not None
    est = full['prod_est']
    assert est['source'] == 'default'
    assert est['value'] == round(min(1.0, 0.428 + 1.139 * pool), 4)

    row = service.compact_check(full)
    assert row['prod_est'] == est['value'] and row['pool'] == pool
    assert set(row) == {'alpha_id', 'recommendation', 'prod_est', 'prod_est_confidence', 'pool',
                        'self', 'prod_lower_bound', 'platform_prod', 'platform_status'}
    assert est['confidence'] == 'normal' and 'note' not in est


@pytest.mark.asyncio
async def test_record_platform_corr_writes_back_and_check_many_batches():
    dao = StubDao()
    seed_alpha(dao, 'a1', values=(0.0, 1.0, 3.0, 6.0))
    seed_alpha(dao, 'a2', values=(0.0, 2.0, 6.0, 12.0))
    service = make_service(StubFetcher(total=2), dao)

    assert await service.record_platform_corr('a1', 'prod', 0.6333, source='platform_check')
    assert dao.platform['a1']['prod']['max'] == 0.6333
    assert not await service.record_platform_corr('a1', 'prod', None)

    out = await service.check_many(['a1', 'a2', 'a1', ' '])
    assert out['count'] == 2
    assert out['results'][0]['platform_prod'] == 0.6333
    with pytest.raises(ValueError):
        await service.check_many([])


def test_prod_estimate_says_when_the_pool_has_no_peer():
    service = make_service(StubFetcher(total=2), StubDao())
    snapshot = {'localCorrs': [], 'platformCorrs': {}}
    def estimate(pool):
        return service._prod_estimate(snapshot, {'result': {'available': True, 'max': pool}, 'stale': False})
    far = estimate(0.12)
    assert far['confidence'] == 'low' and far['over_threshold'] is None
    assert 'correlates only 0.12' in far['note'] and 'far HIGHER' in far['note']
    near = estimate(0.45)
    assert near['confidence'] == 'normal' and 'note' not in near and near['over_threshold'] is True

    pairs = {f'r{i}': 0.4 + i * 0.05 for i in range(10)}           # calibrated on pools from 0.4 up
    snapshot = {'localCorrs': [{'alphaId': k, 'corrType': 'POOL', 'result': {'available': True, 'max': v}}
                               for k, v in pairs.items()],
                'platformCorrs': {k: {'prod': {'max': min(0.99, 0.3 + v)}} for k, v in pairs.items()}}
    below = estimate(0.35)
    assert below['source'] == 'fitted' and below['confidence'] == 'low'
    assert 'below what the estimate was calibrated on' in below['note']
    row = service.compact_check({'alpha_id': 'x', 'recommendation': 'check', 'prod_est': below,
                                 'local': {}, 'platform': {}, 'platform_status': 'not_requested'})
    assert row['prod_est_confidence'] == 'low'

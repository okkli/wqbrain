"""Acceptance tests for prodmemo_calc.

Each test mirrors a case in the browser extension's
``tests/prodMemoCalculator.test.mjs`` (WebDataScope v1.5.0). Expected values are
copied verbatim from the JS assertions — see docs/PRODMEMO_IMPLEMENTATION.md
§5.10. If a value here changes, the Python port has diverged from the extension
and stored fingerprints/results stop being mutually recognisable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prodmemo_calc import (  # noqa: E402
    alpha_group_key,
    calculate_forward_filled_returns,
    calculate_local_correlation,
    calculate_prod_lower_bound,
    calculation_fingerprint,
    correlation_lower_bound,
    extract_platform_correlation_stats,
    normalize_pnl,
    pearson_correlation,
    resolve_preferred_correlation,
    rolling_window_start,
    select_candidate_alphas,
)


def settings(delay=1, universe='TOP3000', region='USA'):
    return {'region': region, 'universe': universe, 'delay': delay,
            'instrumentType': 'EQUITY'}


def alpha(alpha_id, **options):
    return {
        'id': alpha_id,
        'name': alpha_id,
        'settings': options.get('settings') or settings(),
        'stage': options.get('stage') or 'OS',
        'submitted': options.get('submitted', True),
        'classifications': options.get('classifications') or [{'id': 'REGULAR:REGULAR'}],
        'is': {},
    }


def pnl(alpha_id, values):
    return {'alphaId': alpha_id,
            'records': [[f'2026-01-0{i + 1}', v] for i, v in enumerate(values)]}


# T1 — JS test :46-67
def test_normalization_rolling_window_and_forward_fill():
    source = {'records': [
        ['2026-08-05T00:00:00Z', 4],
        ['bad', 2],
        ['2022-08-05', 1],
        ['2026-08-05', 5],
    ]}
    normalized = normalize_pnl(source)
    assert normalized == [('2022-08-05', 1), ('2026-08-05', 5)]
    assert rolling_window_start(normalized) == '2022-08-05'

    returns = calculate_forward_filled_returns(
        {'records': [['2026-01-01', 1], ['2026-01-03', 4]]},
        ['2026-01-01', '2026-01-02', '2026-01-03'],
        '2025-01-01',
        '2026-01-03',
    )
    assert list(returns.items()) == [('2026-01-02', 0), ('2026-01-03', 3)]


# T2 — JS test :69-80
def test_pearson_reports_overlap_and_preserves_sign():
    positive = pearson_correlation({'a': 1, 'b': 2, 'c': 3}, {'a': 2, 'b': 4, 'c': 6})
    negative = pearson_correlation({'a': 1, 'b': 2, 'c': 3}, {'a': -2, 'b': -4, 'c': -6})
    assert positive == {'value': 1, 'overlapCount': 3}
    assert negative == {'value': -1, 'overlapCount': 3}


# T3 — JS test :82-109
def test_candidate_pools_enforce_group_and_classifications():
    target = alpha('target')
    regular = alpha('regular')
    different_delay = alpha('delay', settings=settings(0))
    power_only = alpha('power', stage='IS',
                       classifications=[{'id': 'POWER_POOL:POWER_POOL_ELIGIBLE'}])
    power_regular = alpha('power-regular', classifications=[
        {'id': 'POWER_POOL:POWER_POOL_ELIGIBLE'},
        {'id': 'REGULAR:REGULAR'},
    ])
    not_submitted = alpha('draft', submitted=False)
    alphas = [target, regular, different_delay, power_only, power_regular, not_submitted]
    pnl_ids = {item['id'] for item in alphas}

    assert [a['id'] for a in select_candidate_alphas(alphas, pnl_ids, target, 'SELF')] == \
        ['regular', 'power', 'power-regular']
    assert [a['id'] for a in select_candidate_alphas(alphas, pnl_ids, target, 'POOL')] == \
        ['power', 'power-regular']
    assert alpha_group_key(different_delay) == 'USA|TOP3000|D0'


# T4 — JS test :111-132
def test_local_correlation_returns_signed_extrema_and_overlap():
    target = alpha('target')
    target_pnl = pnl('target', [0, 1, 3, 6])
    pnl_by_id = {
        'positive': pnl('positive', [0, 2, 6, 12]),
        'negative': pnl('negative', [0, -2, -6, -12]),
    }
    result = calculate_local_correlation(
        target_alpha=target,
        target_pnl=target_pnl,
        candidates=[alpha('positive'), alpha('negative')],
        pnl_by_id=pnl_by_id,
        corr_type='SELF',
    )
    assert result['available'] is True
    assert result['max'] == 1
    assert result['min'] == -1
    assert result['corrCount'] == 2
    assert result['records'][0]['overlapCount'] == 3


# T5 — JS test :134-159
def test_prod_lower_bound_formula_and_max_witness():
    assert abs(correlation_lower_bound(0.5, 0.8) - (-0.11961524227066308)) < 1e-12
    assert correlation_lower_bound(1, 0.75) == 0.75

    target_pnl = pnl('target', [0, 1, 3, 6])
    pnl_by_id = {
        'best': pnl('best', [0, 2, 6, 12]),
        'other': pnl('other', [0, -2, -6, -12]),
    }
    result = calculate_prod_lower_bound(
        target_alpha=alpha('target'),
        target_pnl=target_pnl,
        references=[
            {'alpha': alpha('best'), 'platformProdMax': 0.75},
            {'alpha': alpha('other'), 'platformProdMax': 0.9},
        ],
        pnl_by_id=pnl_by_id,
    )
    assert result['available'] is True
    # 0.9 belongs to the negatively correlated reference, which yields a weaker
    # bound — picking the largest platform value instead of the largest bound
    # would be wrong.
    assert result['max'] == 0.75
    assert result['witness']['alphaId'] == 'best'
    assert result['witness']['overlapCount'] == 3


# T6 — JS test :161-175
def test_target_with_platform_prod_is_its_own_exact_witness():
    target = alpha('target')
    target_pnl = pnl('target', [0, 1, 3, 6])
    result = calculate_prod_lower_bound(
        target_alpha=target,
        target_pnl=target_pnl,
        references=[{'alpha': target, 'platformProdMax': 0.82}],
        pnl_by_id={'target': target_pnl},
    )
    assert result['available'] is True
    assert result['max'] == 0.82
    assert result['witness']['alphaId'] == 'target'
    assert result['witness']['correlation'] == 1


# T7 — JS test :177-193
def test_fingerprints_change_with_pnl_or_known_prod_corr():
    target = alpha('target')
    reference = alpha('reference')
    target_pnl = pnl('target', [0, 1, 3, 6])
    pnl_by_id = {'reference': pnl('reference', [0, 2, 6, 12])}

    def make(platform_prod_max):
        return calculation_fingerprint(
            corr_type='PROD_LOWER_BOUND',
            target_alpha=target,
            target_pnl=target_pnl,
            pnl_by_id=pnl_by_id,
            references=[{'alpha': reference, 'platformProdMax': platform_prod_max,
                         'platformUpdated': 1}],
        )

    assert make(0.7) != make(0.8)
    before = make(0.8)
    pnl_by_id['reference'] = pnl('reference', [0, 2, 7, 13])
    assert before != make(0.8)


# T8 — JS test :195-201
def test_platform_extraction_preserves_signs_and_schemas():
    assert extract_platform_correlation_stats({'max': 0.8, 'min': -0.4}) == \
        {'max': 0.8, 'min': -0.4}
    assert extract_platform_correlation_stats({
        'schema': {'properties': [{'name': 'id'}, {'name': 'correlation'}]},
        'records': [['a', -0.2], ['b', 0.7]],
    }) == {'max': 0.7, 'min': -0.2}


# T9 — JS test :203-226
def test_resolution_prefers_platform_and_rejects_stale_local():
    local = {'stale': False, 'calculatedAt': 2,
             'result': {'available': True, 'max': 0.65, 'min': -0.2}}
    assert resolve_preferred_correlation({'max': 0.72, 'min': -0.4, 'updated': 1}, local) == {
        'max': 0.72, 'min': -0.4, 'updated': 1,
        'source': 'platform', 'icon': 'Ⓟ', 'lowerBound': False,
    }
    assert resolve_preferred_correlation(None, local, {'lowerBound': True}) == {
        'max': 0.65, 'min': -0.2, 'updated': 2,
        'source': 'local', 'icon': 'Ⓛ', 'lowerBound': True,
    }
    assert resolve_preferred_correlation(None, {**local, 'stale': True}) is None

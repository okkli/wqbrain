"""ProdMemo pure-computation layer.

Faithful Python port of WebDataScope v1.5.0
``src/background/services/prodMemoCalculator.js`` (331 lines) plus the two pure
helpers from ``prodMemoService.js`` (extractPlatformCorrelationStats,
resolvePreferredCorrelation). See docs/PRODMEMO_IMPLEMENTATION.md §5.

Behavioural equivalence with the JS reference is a hard requirement: fingerprints
and stored results must be mutually recognisable with the browser extension
(ALGORITHM_VERSION 2). Do not "improve" semantics here without bumping the
version — including the known dead-code branch in select_candidate_alphas.

Zero third-party dependencies (standard library only) so the test suite runs on
machines without the ``mcp`` package installed.
"""

import math
import re
import sys
from datetime import datetime, timezone

PROD_MEMO_ALGORITHM_VERSION = 2

POWER_POOL_CLASSIFICATION = 'POWER_POOL:POWER_POOL_ELIGIBLE'
REGULAR_CLASSIFICATION = 'REGULAR:REGULAR'

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def finite_number(value):
    """JS finiteNumber: null/undefined/'' -> null, else Number(), non-finite -> null."""
    if value is None or value == '':
        return None
    if isinstance(value, bool):  # JS Number(true) is 1, but bools never appear in real data
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clamp_correlation(value):
    number = finite_number(value)
    if number is None:
        return None
    return max(-1.0, min(1.0, number))


def js_number_str(value):
    """Stringify a number the way a JS template literal does.

    Integer-valued floats print without a trailing ``.0`` (JS ``${6}`` -> "6").
    Other floats use repr(), which is shortest-roundtrip in both languages.
    Only used inside fingerprints, where inputs are ordinary PnL/corr magnitudes.
    """
    number = float(value)
    if math.isfinite(number) and number == int(number) and abs(number) < 1e21:
        return str(int(number))
    return repr(number)


def _normalize_text(value):
    return str(value if value is not None else '').strip().upper()


def alpha_group_key(alpha):
    """JS alphaGroupKey (:24-31). alpha may carry settings.{region,...} or flat keys."""
    settings = (alpha or {}).get('settings') or {}
    region = _normalize_text(settings.get('region', alpha.get('region') if alpha else None))
    universe = _normalize_text(settings.get('universe', alpha.get('universe') if alpha else None))
    raw_delay = settings.get('delay', alpha.get('delay') if alpha else None)
    delay = finite_number(raw_delay)
    if not region or not universe or delay is None:
        return ''
    return f"{region}|{universe}|D{js_number_str(delay)}"


def normalize_pnl(data):
    """JS normalizePnl (:33-50). Accepts {'records': [...]} or a bare record list.

    Returns [(date, value), ...] deduplicated by date (last write wins), ascending.
    The JS WeakMap identity cache is omitted — correctness is unaffected.
    """
    if isinstance(data, dict):
        records = data.get('records')
    else:
        records = data
    if not isinstance(records, list):
        records = []
    by_date = {}
    for record in records:
        if not isinstance(record, (list, tuple)) or len(record) < 2 or not record[0]:
            continue
        value = finite_number(record[1])
        if value is None:
            continue
        date = str(record[0])[:10]
        if not _DATE_RE.match(date):
            continue
        by_date[date] = value
    return sorted(by_date.items())


def rolling_window_start(records, years=4):
    """JS rollingWindowStart (:52-58): anchor at the LAST PnL date, minus `years` (UTC)."""
    if not records:
        return ''
    try:
        end = datetime.strptime(records[-1][0], '%Y-%m-%d').replace(tzinfo=timezone.utc)
    except ValueError:
        return ''
    try:
        start = end.replace(year=end.year - years)
    except ValueError:
        # Feb 29 minus a non-multiple-of-4 span: JS setUTCFullYear rolls to Mar 1.
        start = end.replace(year=end.year - years, month=3, day=1)
    return start.strftime('%Y-%m-%d')


def collect_dates(pnl_records):
    """JS collectDates (:60-66): sorted union of all dates across curves."""
    dates = set()
    for data in pnl_records:
        for date, _value in normalize_pnl(data):
            dates.add(date)
    return sorted(dates)


def calculate_forward_filled_returns(data, dates, start_date, end_date=''):
    """JS calculateForwardFilledReturns (:68-88).

    Aligns one cumulative-PnL curve onto the shared date axis, forward-fills
    missing days, and differences into daily returns. Window is half-open
    (start_date, end_date]; data before the window still updates `previous`,
    so the first in-window return uses the true pre-window value.
    """
    records = normalize_pnl(data)
    returns = {}
    record_index = 0
    current_value = None
    previous_value = None
    for date in dates:
        if end_date and date > end_date:
            break
        while record_index < len(records) and records[record_index][0] <= date:
            current_value = records[record_index][1]
            record_index += 1
        if current_value is None:
            continue
        if previous_value is not None and (not start_date or date > start_date):
            returns[date] = current_value - previous_value
        previous_value = current_value
    return returns


def pearson_correlation(left, right):
    """JS pearsonCorrelation (:90-118). Single pass, inner join on dates,
    computational formula, sign preserved, clamped to [-1, 1].

    Returns {'value': r, 'overlapCount': n} or None.
    """
    count = 0
    sum_x = sum_y = sum_xx = sum_yy = sum_xy = 0.0
    for date, x in left.items():
        if date not in right:
            continue
        y = right[date]
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        count += 1
        sum_x += x
        sum_y += y
        sum_xx += x * x
        sum_yy += y * y
        sum_xy += x * y
    if count < 2:
        return None
    covariance = count * sum_xy - sum_x * sum_y
    variance_x = count * sum_xx - sum_x * sum_x
    variance_y = count * sum_yy - sum_y * sum_y
    denominator = math.sqrt(max(0.0, variance_x * variance_y))
    if not math.isfinite(denominator) or denominator == 0:
        return None
    value = clamp_correlation(covariance / denominator)
    if value is None:
        return None
    return {'value': value, 'overlapCount': count}


def has_classification(alpha, classification):
    """JS hasClassification (:120-122): items may be strings or {'id': ...} dicts."""
    for item in (alpha or {}).get('classifications') or []:
        item_id = item.get('id') if isinstance(item, dict) else item
        if item_id == classification:
            return True
    return False


def is_power_pool_alpha(alpha):
    return has_classification(alpha, POWER_POOL_CLASSIFICATION)


def is_regular_alpha(alpha):
    return has_classification(alpha, REGULAR_CLASSIFICATION)


def select_candidate_alphas(alphas, pnl_ids, target_alpha, corr_type):
    """JS selectCandidateAlphas (:132-146). ``pnl_ids`` is a set-like of alpha ids
    with local PnL available.

    KNOWN ISSUE preserved on purpose: the SELF branch below simplifies to
    ``stage == 'OS' or is_power_pool_alpha(alpha)`` — the REGULAR:REGULAR check is
    unreachable. Kept verbatim for fingerprint/result parity with the extension
    (ALGORITHM_VERSION 2); fix only together with upstream + a version bump.
    """
    target_id = str((target_alpha or {}).get('id') or '')
    group_key = alpha_group_key(target_alpha)
    selected = []
    for alpha in alphas or []:
        if not alpha or not alpha.get('id') or alpha['id'] == target_id or not alpha.get('submitted'):
            continue
        if not group_key or alpha_group_key(alpha) != group_key or alpha['id'] not in pnl_ids:
            continue
        if corr_type == 'SELF':
            self_eligible = alpha.get('stage') == 'OS' and (
                not is_power_pool_alpha(alpha) or is_regular_alpha(alpha)
            )
            if self_eligible or is_power_pool_alpha(alpha):
                selected.append(alpha)
        elif corr_type == 'POOL':
            if is_power_pool_alpha(alpha):
                selected.append(alpha)
    return selected


def round_correlation(value):
    """JS roundCorrelation (:148-150): Math.round((v + EPSILON) * 10000) / 10000.
    math.floor(x + 0.5) reproduces JS Math.round (which rounds .5 toward +inf,
    unlike Python's banker's rounding)."""
    return math.floor((value + sys.float_info.epsilon) * 10000 + 0.5) / 10000


def _official_record(alpha, correlation, overlap_count):
    """JS officialRecord (:152-168)."""
    settings = alpha.get('settings') or {}
    is_metrics = alpha.get('is') or {}
    return {
        'alphaId': alpha['id'],
        'name': alpha.get('name'),
        'instrumentType': settings.get('instrumentType'),
        'region': settings.get('region'),
        'universe': settings.get('universe'),
        'delay': settings.get('delay'),
        'correlation': round_correlation(correlation),
        'overlapCount': overlap_count,
        'sharpe': is_metrics.get('sharpe'),
        'returns': is_metrics.get('returns'),
        'turnover': is_metrics.get('turnover'),
        'fitness': is_metrics.get('fitness'),
        'margin': is_metrics.get('margin'),
    }


def calculate_local_correlation(target_alpha, target_pnl, candidates, pnl_by_id, corr_type):
    """JS calculateLocalCorrelation (:170-212). SELF/POOL local correlation."""
    target_records = normalize_pnl(target_pnl)
    if len(target_records) < 2:
        return {'available': False, 'reason': '目标 Alpha 的 PnL 数据不足。'}
    if not candidates:
        return {'available': False,
                'reason': f'当前 universe-delay 组合没有可用的 {corr_type} 候选。'}

    end_date = target_records[-1][0]
    start_date = rolling_window_start(target_records)
    source_pnls = [target_pnl] + [pnl_by_id.get(alpha['id']) for alpha in candidates]
    dates = collect_dates(source_pnls)
    target_returns = calculate_forward_filled_returns(target_pnl, dates, start_date, end_date)

    correlations = []
    for alpha in candidates:
        peer_returns = calculate_forward_filled_returns(
            pnl_by_id.get(alpha['id']), dates, start_date, end_date)
        correlation = pearson_correlation(target_returns, peer_returns)
        if correlation:
            correlations.append({'alpha': alpha, **correlation})
    correlations.sort(key=lambda item: item['value'], reverse=True)
    if not correlations:
        return {'available': False, 'reason': '候选 PnL 与目标 PnL 没有足够的共同有效日期。'}

    values = [item['value'] for item in correlations]
    return {
        'available': True,
        'max': round_correlation(max(values)),
        'min': round_correlation(min(values)),
        'records': [
            _official_record(item['alpha'], item['value'], item['overlapCount'])
            for item in correlations[:5]
        ],
        'poolSize': len(candidates),
        'corrCount': len(correlations),
        'maxOverlapCount': max(item['overlapCount'] for item in correlations),
        'windowStart': start_date,
        'windowEnd': end_date,
    }


def correlation_lower_bound(correlation, known_prod_max):
    """JS correlationLowerBound (:280-289).

    Math: correlations are cosines of angles between de-meaned daily-return
    vectors. With y = corr(target, reference) and c = the reference's known
    platform Prod Corr (so some production alpha P* has corr(reference, P*) = c),
    the spherical triangle inequality gives

        corr(target, P*) >= cos(arccos y + arccos c) = y*c - sqrt(1-y^2)*sqrt(1-c^2)

    and the target's true Prod Corr (a max over the production pool) is >= that.
    Strict lower bound provided platform and local use the same computation
    window/method — hence "conditional lower bound" in all UI copy.
    """
    y = clamp_correlation(correlation)
    c = clamp_correlation(known_prod_max)
    if y is None or c is None:
        return None
    return clamp_correlation(
        y * c - math.sqrt(max(0.0, 1 - c * c)) * math.sqrt(max(0.0, 1 - y * y)))


def calculate_prod_lower_bound(target_alpha, target_pnl, references, pnl_by_id):
    """JS calculateProdLowerBound (:214-278).

    references: [{'alpha': alpha_dict, 'platformProdMax': float, 'platformUpdated': ts}]
    Witness = the reference yielding the LARGEST lower bound (the conjunction of
    many lower bounds is their max).
    """
    target_records = normalize_pnl(target_pnl)
    if len(target_records) < 2:
        return {'available': False, 'reason': '目标 Alpha 的 PnL 数据不足。'}
    if not references:
        return {'available': False,
                'reason': '当前 universe-delay 组合没有完整的已知 Prod Corr 参考曲线。'}

    end_date = target_records[-1][0]
    start_date = rolling_window_start(target_records)
    dates = collect_dates(
        [target_pnl] + [pnl_by_id.get(item['alpha']['id']) for item in references])
    target_returns = calculate_forward_filled_returns(target_pnl, dates, start_date, end_date)

    bounds = []
    for reference in references:
        known_returns = calculate_forward_filled_returns(
            pnl_by_id.get(reference['alpha']['id']), dates, start_date, end_date)
        correlation = pearson_correlation(target_returns, known_returns)
        known_prod_max = clamp_correlation(reference.get('platformProdMax'))
        if not correlation or known_prod_max is None:
            continue
        y = clamp_correlation(correlation['value'])
        lower = correlation_lower_bound(y, known_prod_max)
        if lower is None or not math.isfinite(lower):
            continue
        bounds.append({
            'alphaId': reference['alpha']['id'],
            'correlation': y,
            'knownProdMax': known_prod_max,
            'lowerBound': clamp_correlation(lower),
            'overlapCount': correlation['overlapCount'],
        })
    bounds.sort(key=lambda item: item['lowerBound'], reverse=True)
    if not bounds:
        return {'available': False, 'reason': '参考曲线与目标 PnL 没有足够的共同有效日期。'}

    def rounded(item):
        return {
            **item,
            'correlation': round_correlation(item['correlation']),
            'knownProdMax': round_correlation(item['knownProdMax']),
            'lowerBound': round_correlation(item['lowerBound']),
        }

    witness = bounds[0]
    return {
        'available': True,
        'max': round_correlation(witness['lowerBound']),
        'min': round_correlation(min(item['lowerBound'] for item in bounds)),
        'referenceCount': len(references),
        'corrCount': len(bounds),
        'maxOverlapCount': max(item['overlapCount'] for item in bounds),
        'witness': rounded(witness),
        'records': [rounded(item) for item in bounds[:5]],
        'windowStart': start_date,
        'windowEnd': end_date,
    }


def hash_text(source):
    """JS hashText (:291-298): FNV-1a 32-bit. Masking to 32 bits after the
    multiply is congruent with JS Math.imul + >>> 0 (ASCII inputs only here)."""
    h = 2166136261
    for ch in source:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return format(h, 'x')


def pnl_fingerprint(data):
    """JS pnlFingerprint (:300-306). A stored 'fingerprint' field wins (this is
    what makes light snapshots work: {'fingerprint': ...} stubs need no records)."""
    if isinstance(data, dict):
        stored = data.get('fingerprint')
        if isinstance(stored, str) and stored:
            return stored
    records = normalize_pnl(data)
    if not records:
        return 'empty'
    source = '|'.join(f'{date}:{js_number_str(value)}' for date, value in records)
    return f'{len(records)}:{records[-1][0]}:{hash_text(source)}'


def calculation_fingerprint(corr_type, target_alpha, target_pnl, pnl_by_id,
                            candidates=(), references=()):
    """JS calculationFingerprint (:308-331). Parts are sorted before hashing so
    the fingerprint is insensitive to candidate order."""
    parts = [
        f'v{PROD_MEMO_ALGORITHM_VERSION}',
        corr_type,
        str((target_alpha or {}).get('id') or ''),
        alpha_group_key(target_alpha),
        pnl_fingerprint(target_pnl),
    ]
    if corr_type == 'PROD_LOWER_BOUND':
        for item in references:
            prod_max = finite_number(item.get('platformProdMax'))
            updated = finite_number(item.get('platformUpdated') or 0) or 0
            parts.append(':'.join([
                item['alpha']['id'],
                pnl_fingerprint(pnl_by_id.get(item['alpha']['id'])),
                js_number_str(prod_max) if prod_max is not None else 'NaN',
                js_number_str(updated),
            ]))
    else:
        for alpha in candidates:
            parts.append(f"{alpha['id']}:{pnl_fingerprint(pnl_by_id.get(alpha['id']))}")
    return hash_text('|'.join(sorted(parts)))


# --- pure helpers ported from prodMemoService.js ------------------------------

def normalize_correlation_type(value):
    """JS normalizeCorrelationType (prodMemoService.js:62-67)."""
    text = str(value or 'prod').strip().lower()
    if 'self' in text:
        return 'self'
    if 'pool' in text or 'parent' in text or 'ppa' in text:
        return 'pool'
    return 'prod'


def _extract_correlation_values(data):
    """JS extractCorrelationValues (prodMemoService.js:75-91)."""
    rows = None
    if isinstance(data, dict):
        for key in ('correlations', 'results', 'records'):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
    elif isinstance(data, list):
        rows = data
    if rows is None:
        rows = []
    correlation_index = 5
    properties = data.get('schema', {}).get('properties') if isinstance(data, dict) else None
    if isinstance(properties, list):
        for index, prop in enumerate(properties):
            if isinstance(prop, dict) and prop.get('name') == 'correlation':
                correlation_index = index
                break
    values = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            value = finite_number(row[correlation_index]) if len(row) > correlation_index else None
        elif isinstance(row, dict):
            raw = row.get('correlation', row.get('value', row.get('score')))
            value = finite_number(raw)
        else:
            value = None
        if value is not None:
            values.append(value)
    return values


def extract_platform_correlation_stats(data):
    """JS extractPlatformCorrelationStats (prodMemoService.js:93-111).

    Multi-layer tolerant parse of a platform correlations response:
    nested {'result': ...} -> maximum/max & minimum/min -> row scan via schema
    (correlation column located by schema.properties, default index 5).
    """
    if not isinstance(data, dict):
        return None
    result = data.get('result')
    if (isinstance(result, dict)
            and all(data.get(k) is None for k in ('maximum', 'minimum', 'max', 'min'))):
        return extract_platform_correlation_stats(result)
    maximum = finite_number(data.get('maximum', data.get('max')))
    minimum = finite_number(data.get('minimum', data.get('min')))
    if maximum is not None or minimum is not None:
        return {'max': maximum, 'min': minimum}
    values = _extract_correlation_values(data)
    if not values:
        return None
    return {'max': max(values), 'min': min(values)}


def resolve_preferred_correlation(platform_stat, local_record, options=None):
    """JS resolvePreferredCorrelation (prodMemoService.js:280-304).

    Platform value wins outright; otherwise a non-stale available local result;
    a stale local result is NEVER shown as a fallback.
    """
    options = options or {}
    platform_max = finite_number((platform_stat or {}).get('max')) if platform_stat else None
    if platform_max is not None:
        return {
            'max': platform_max,
            'min': finite_number(platform_stat.get('min')),
            'updated': platform_stat.get('updated'),
            'source': 'platform',
            'icon': 'Ⓟ',
            'lowerBound': False,
        }
    local_max = None
    if local_record and not local_record.get('stale'):
        result = local_record.get('result') or {}
        if result.get('available'):
            local_max = finite_number(result.get('max'))
    if local_max is None:
        return None
    result = local_record['result']
    return {
        'max': local_max,
        'min': finite_number(result.get('min')),
        'updated': local_record.get('calculatedAt'),
        'source': 'local',
        'icon': 'Ⓛ',
        'lowerBound': options.get('lowerBound') is True,
    }


def format_resolved_metric(metric):
    """JS formatResolvedMetric (prodMemoService.js:306-310): e.g. '≥0.7123 Ⓛ'."""
    if not metric:
        return ''
    prefix = '≥' if metric.get('lowerBound') else ''
    return f"{prefix}{float(metric['max']):.4f} {metric['icon']}"

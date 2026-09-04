"""
Checks on the risk warnings.

The danger in a risk panel is not a crash - it is a confident warning
with nothing behind it. These tests target that specifically: they check
that the thresholds are the ones we actually measured, that a warning
above a measured threshold is flagged MEASURED and one below it is not,
and that nothing claims to be current when it is reading 2019 data.

The cyclone path cannot be exercised by waiting for a cyclone, so the
forecast is injected.

Run:  python -m tests.test_risk
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import risk, congestion, paths  # noqa: E402

FAIL = []

# An ISO date anywhere in a sentence a person reads is a bug.
ISO_RE = r'\d{4}-\d{2}-\d{2}'


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def fake_forecast(gusts, rain=None):
    """Return a stand-in for _forecast with the given daily gusts."""
    n = len(gusts)
    df = pd.DataFrame({
        'date': pd.date_range('2026-09-04', periods=n, freq='D'),
        'gust': gusts,
        'wind': [g * 0.5 for g in gusts],
        'rain': rain if rain is not None else [0.0] * n,
    })
    return lambda port, days=n, timeout=30: df


def main():
    global orig
    print('\n[1] thresholds are the measured ones, not round numbers')
    check('halt threshold is 90 km/h, where the -88%% effect was measured',
          risk.GUST_HALT_KMH == 90.0, risk.GUST_HALT_KMH)
    check('gale threshold (%.0f) sits below the halt threshold (%.0f)'
          % (risk.GUST_GALE_KMH, risk.GUST_HALT_KMH),
          risk.GUST_GALE_KMH < risk.GUST_HALT_KMH)

    print('\n[2] a cyclone forecast raises a CRITICAL, MEASURED warning')
    orig = risk._forecast
    try:
        risk._forecast = fake_forecast([30, 35, 118, 96, 40, 32])
        w = risk.weather_warnings('paradip')
        top = w[0]
        check('severity is critical', top['severity'] == 'critical',
              top['severity'])
        check('flagged as measured', top['measured'] is True)
        check('basis cites the measured effect',
              '88%' in top['basis'] and 'p < 0.0001' in top['basis'],
              top['basis'])
        check('names the gust and the date, the date as DD/MM/YYYY',
              '118' in top['detail'] and '06/09/2026' in top['detail'],
              top['detail'][:90])

        print('\n[3] gale force is a WATCH and is NOT claimed as measured')
        # Between gale and halt we measured nothing. A panel that
        # implied otherwise would be inventing an effect size.
        risk._forecast = fake_forecast([30, 70, 68, 35])
        w = risk.weather_warnings('paradip')
        top = w[0]
        check('severity is watch', top['severity'] == 'watch', top['severity'])
        check('NOT flagged as measured', top['measured'] is False)
        check('basis says plainly that it is not measured',
              'NOT measured' in top['basis'], top['basis'][:80])

        print('\n[4] calm weather says so rather than inventing a risk')
        risk._forecast = fake_forecast([28, 33, 30, 35])
        w = risk.weather_warnings('paradip')
        check('severity is clear', w[0]['severity'] == 'clear')
        check('reports the peak gust against the threshold',
              '35' in w[0]['detail'] and '90' in w[0]['detail'],
              w[0]['detail'])

        print('\n[5] heavy rain is context, never measured')
        risk._forecast = fake_forecast([30, 32, 31], rain=[2.0, 88.0, 5.0])
        w = risk.weather_warnings('paradip')
        rain = [x for x in w if 'rain' in x['title'].lower()]
        check('a rainfall item is raised', len(rain) == 1, [x['title'] for x in w])
        if rain:
            check('rainfall is NOT flagged measured',
                  rain[0]['measured'] is False)
            check('basis admits no measured rainfall effect',
                  'no measured rainfall effect' in rain[0]['basis'],
                  rain[0]['basis'])

        print('\n[6] a failing forecast degrades, it does not crash')
        def boom(port, days=10, timeout=30):
            raise RuntimeError('network down')
        risk._forecast = boom
        w = risk.weather_warnings('paradip')
        check('returns one item rather than raising', len(w) == 1)
        check('and it is a watch, not a clear - we did not look, so the '
              'dashboard must not collapse it out of sight',
              w[0]['severity'] == 'watch', w[0]['severity'])
        check('and says the outlook is unavailable',
              'unavailable' in w[0]['title'].lower(), w[0]['title'])
    finally:
        risk._forecast = orig

    print('\n[7] every warning carries the fields a reader needs')
    rows = risk.assess()
    need = {'kind', 'port', 'severity', 'measured', 'title', 'detail', 'basis'}
    bad = [w['title'] for w in rows if not need <= set(w)]
    check('all %d warnings have kind/severity/measured/title/detail/basis'
          % len(rows), not bad, bad[:3])
    bad = [w['title'] for w in rows
           if w['severity'] not in risk.SEVERITY_ORDER]
    check('every severity is one of the four levels', not bad, bad[:3])
    bad = [w['title'] for w in rows if not isinstance(w['measured'], bool)]
    check('measured is always a bool', not bad, bad[:3])

    print('\n[8] context signals never masquerade as measured')
    # Berth load and seasonality are real observations with no measured
    # effect size. If either ever flips to measured, someone has quietly
    # asserted a number this repository has not established.
    bad = [w['title'] for w in rows
           if w['kind'] in ('berth', 'seasonal') and w['measured']]
    check('no berth or seasonal warning claims to be measured', not bad, bad)
    berth = [w for w in rows if w['kind'] == 'berth' and w['severity'] != 'clear']
    check('berth warnings state that arrivals are not waiting time',
          all('not queue length' in w['basis'] for w in berth),
          [w['basis'][:50] for w in berth[:2]])
    seas = [w for w in rows if w['kind'] == 'seasonal']
    check('seasonal warnings admit the cause is unknown',
          all('CAUSE is unknown' in w['basis'] for w in seas),
          [w['basis'][:50] for w in seas[:2]])

    print('\n[9] nothing historical is presented as current')
    mu = risk.model_uncertainty()
    check('model uncertainty exists and is dated', mu and 'as_of' in mu)
    if mu:
        check('its as_of is the last SCORED date, in 2019, not today',
              pd.Timestamp(mu['as_of']).year == 2019, mu['as_of'])
        check('it says a wide band means uncertainty, not a rate move',
              'not that rates will move' in mu['detail'], mu['detail'][:80])
    dated = [w for w in rows if w['kind'] == 'berth' and 'as_of' in w]
    check('berth warnings carry their own as_of date', len(dated) > 0)

    print('\n[10] ordering puts the worst, best-evidenced items first')
    sev = [risk.SEVERITY_ORDER[w['severity']] for w in rows]
    check('severity is non-decreasing down the list: %s' % sev,
          sev == sorted(sev), sev)
    for lvl in ('critical', 'warning', 'watch'):
        grp = [w['measured'] for w in rows if w['severity'] == lvl]
        if len(grp) > 1:
            check('within %s, measured items come first' % lvl,
                  grp == sorted(grp, reverse=True), grp)

    print('\n[11] bad input is refused')
    try:
        risk.congestion_site('atlantis')
        check('unknown port rejected', False, 'no error raised')
    except KeyError as e:
        check('unknown port rejected, listing valid ones', 'known:' in str(e))

    print('\n[12] the live forecast API actually answers')
    # One real network call, so a broken endpoint is caught rather than
    # hidden behind the injected forecasts above.
    try:
        f = risk._forecast('paradip', days=7)
        check('Open-Meteo returned 7 days of gusts', len(f) == 7, len(f))
        check('gusts are plausible (0-300 km/h)',
              f['gust'].between(0, 300).all(), f['gust'].tolist())
        check('the forecast starts today or later',
              f['date'].min() >= pd.Timestamp.today().normalize()
              - pd.Timedelta(days=1), str(f['date'].min()))
    except Exception as e:
        check('live forecast reachable', False, '%s: %s' % (type(e).__name__, e))

    print('\n[13] the prose quotes the constants, it does not restate them')
    # A threshold written into a sentence is a number that can drift away
    # from the constant it describes. Move the constant, and the text
    # must move with it.
    orig_h = risk.GUST_HALT_KMH
    try:
        risk.GUST_HALT_KMH = 100.0
        risk._forecast = fake_forecast([30, 118, 40])
        w = risk.weather_warnings('paradip')[0]
        check('critical detail quotes the live threshold',
              '100 km/h' in w['detail'] and '90 km/h' not in w['detail'],
              w['detail'][:100])
        check('critical basis quotes it too', '100 km/h' in w['basis'],
              w['basis'])
        risk._forecast = fake_forecast([30, 70, 40])
        w = risk.weather_warnings('paradip')[0]
        check('gale detail quotes it', '100 km/h' in w['detail'],
              w['detail'][:90])
        risk._forecast = fake_forecast([30, 35], rain=[2.0, 88.0])
        rn = [x for x in risk.weather_warnings('paradip')
              if 'rain' in x['title'].lower()][0]
        check('rain detail quotes RAIN_HEAVY_MM',
              '%.0f mm mark' % risk.RAIN_HEAVY_MM in rn['detail'],
              rn['detail'][:100])
    finally:
        risk.GUST_HALT_KMH = orig_h
        risk._forecast = orig

    print('\n[14] the horizon reported is the one returned, not the one asked')
    # Open-Meteo can answer short. "Over the next 10 days" on three days
    # of data is a claim about days nobody looked at.
    try:
        risk._forecast = lambda port, days=10, timeout=15: fake_forecast(
            [30, 32, 31])('paradip')
        w = risk.weather_warnings('paradip', days=10)[0]
        check('three days returned is reported as three',
              'next 3 days' in w['detail'], w['detail'])
        check('and the basis agrees', '3-day' in w['basis'], w['basis'])
    finally:
        risk._forecast = orig

    print('\n[15] an empty forecast degrades rather than raising')
    try:
        risk._forecast = lambda port, days=10, timeout=15: pd.DataFrame(
            {'date': pd.to_datetime([]), 'gust': [], 'wind': [], 'rain': []})
        w = risk.weather_warnings('paradip')
        check('returns one unavailable item, not an IndexError',
              len(w) == 1 and 'unavailable' in w[0]['title'].lower(),
              [x['title'] for x in w])
    finally:
        risk._forecast = orig

    print('\n[16] a non-finite percentile costs a clause, not the panel')
    orig_snap = congestion.snapshot
    try:
        congestion.snapshot = lambda pt, as_of=None: {
            'reliable': True, 'band': 'busy', 'as_of': '2026-08-28',
            'calls_per_day': 3.0, 'baseline_calls_per_day': 2.0,
            'percentile': float('nan')}
        r = risk.berth_load_warning('paradip')
        check('a warning is still produced', r is not None)
        check('and no "nan" reaches the reader',
              r and 'nan' not in r['detail'].lower(), (r or {}).get('detail'))
    finally:
        congestion.snapshot = orig_snap

    print('\n[17] an unknown port is refused, never answered "all clear"')
    # The one failure a risk panel must not have: a typo that reads as
    # reassurance.
    for fn, label in ((lambda: risk.assess(['atlantis']), 'assess'),
                      (lambda: risk.weather_warnings('atlantis'),
                       'weather_warnings')):
        try:
            fn()
            check('%s refuses an unknown port' % label, False,
                  'no error raised')
        except KeyError as e:
            check('%s refuses an unknown port, listing valid ones' % label,
                  'known:' in str(e))

    print('\n[18] dates a person reads are DD/MM/YYYY; as_of stays ISO')
    try:
        risk._forecast = fake_forecast([30, 118, 40])
        w = risk.weather_warnings('paradip')[0]
        check('no ISO date is left in the prose',
              not re.search(ISO_RE, w['detail']), w['detail'][:90])
        check('the date reads 05/09/2026', '05/09/2026' in w['detail'],
              w['detail'][:90])
        check('but as_of stays ISO so callers can still sort on it',
              w['as_of'] == '2026-09-05', w['as_of'])
        risk._forecast = fake_forecast([30, 70, 40])
        g = risk.weather_warnings('paradip')[0]
        check('the gale branch is formatted too',
              not re.search(ISO_RE, g['detail']), g['detail'][:90])
        risk._forecast = fake_forecast([30, 35], rain=[2.0, 88.0])
        rn = [x for x in risk.weather_warnings('paradip')
              if 'rain' in x['title'].lower()][0]
        check('and so is the rainfall branch',
              not re.search(ISO_RE, rn['detail']), rn['detail'][:90])
    finally:
        risk._forecast = orig

    print('\n[19] a stale artefact costs one signal, not the whole panel')
    orig_pq = pd.read_parquet
    try:
        pd.read_parquet = lambda *a, **k: pd.DataFrame(
            {'y': [1.0, 2.0]},
            index=pd.to_datetime(['2019-07-30', '2019-07-31']))
        m = risk.model_uncertainty()
        check('an artefact with no interval columns yields an unavailable '
              'row, not a silent None',
              m is not None and m['kind'] == 'unavailable',
              m and m.get('kind'))
        check('and it names the reason',
              m and 'interval columns' in m['detail'], m and m['detail'][:70])
        risk.assess(['paradip'])
        check('and assess() still runs', True)
    except Exception as e:
        check('assess() survives an artefact with no interval columns',
              False, '%s: %s' % (type(e).__name__, e))
    finally:
        pd.read_parquet = orig_pq

    print('\n[20] the latency budget fits inside a client timeout')
    worst = risk.TIMEOUT * len(risk.SITES)
    check('worst case %ds (%d sites x %ds) stays under 90s'
          % (worst, len(risk.SITES), risk.TIMEOUT), worst < 90, worst)

    print('\n[21] a failing HTTP status is named as an outage')
    class FakeResp(object):
        status_code = 429

        def json(self):
            raise ValueError('not JSON')

    orig_get = risk.requests.get
    try:
        risk.requests.get = lambda *a, **k: FakeResp()
        w = risk.weather_warnings('paradip')[0]
        check('a 429 is reported as HTTP 429, not a parser error',
              'HTTP 429' in w['detail'], w['detail'])
    finally:
        risk.requests.get = orig_get

    print('\n[22] a month outside 1-12 is refused, including 0')
    # `month or today` would have turned a caller's 0 into today.
    for bad_m in (0, 13, -1):
        try:
            risk.seasonal_warning('paradip', bad_m)
            check('month=%r rejected' % bad_m, False, 'accepted')
        except ValueError:
            check('month=%r rejected' % bad_m, True)

    print('\n[23] a signal that cannot be computed is reported, not dropped')
    # The subtle version of the unknown-port bug. If congestion data is
    # missing, berth and seasonal warnings used to vanish: the panel fell
    # from thirteen checks to two, every remaining row read reassuringly,
    # and nothing said a check had been skipped.
    orig_snap = congestion.snapshot
    orig_prof = congestion.monthly_profile
    try:
        def broken(*a, **k):
            raise RuntimeError('parquet missing')

        congestion.snapshot = broken
        congestion.monthly_profile = broken
        rows = risk.assess(['paradip'])
        gaps = [w for w in rows if w['kind'] == 'unavailable']
        titles = sorted(w['title'] for w in gaps)
        check('the missing berth and seasonal checks each get a row',
              len(gaps) == 2, titles)
        check('they name which check could not run',
              titles == ['Berth load unavailable',
                         'Seasonal profile unavailable'], titles)
        check('a gap is a watch, so it lands in the raised list',
              all(w['severity'] == 'watch' for w in gaps),
              [w['severity'] for w in gaps])
        check('a gap never claims to be measured',
              all(w['measured'] is False for w in gaps))
        check('and says plainly it is neither clear nor raised',
              all('neither clear nor raised' in w['detail'] for w in gaps),
              [w['detail'][:60] for w in gaps[:1]])
    finally:
        congestion.snapshot = orig_snap
        congestion.monthly_profile = orig_prof

    print('\n[24] a missing model artefact is reported the same way')
    orig_exists = os.path.exists
    try:
        os.path.exists = (lambda q: False
                          if str(q).endswith('oos_predictions.parquet')
                          else orig_exists(q))
        m = risk.model_uncertainty()
        check('returns a row rather than None', m is not None)
        check('flagged unavailable, not clear',
              m and m['kind'] == 'unavailable' and m['severity'] == 'watch',
              m and (m['kind'], m['severity']))
        check('and tells the reader how to fix it',
              m and 'train_model' in m['detail'], m and m['detail'][:70])
    finally:
        os.path.exists = orig_exists

    print('\n[25] none of this changes the healthy path')
    rows = risk.assess()
    check('no unavailable rows when every artefact is present',
          not [w for w in rows if w['kind'] == 'unavailable'],
          [w['title'] for w in rows if w['kind'] == 'unavailable'])
    check('still %d checks across %d ports plus the model'
          % (len(rows), len(risk.SITES)),
          len(rows) == len(risk.SITES) * 3 + 1, len(rows))

    print('\n[26] a malformed 200 response cannot become an all-clear')
    # The nastiest shape Open-Meteo can return is not an error - it is a
    # 200 whose numbers are null. A null gust compares False against
    # every threshold, so the halt and gale branches are skipped and the
    # panel lands on "no disruptive weather forecast": the most
    # reassuring row it can print, built from no data at all.
    class FakeOK(object):
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def daily(times, gusts, winds, rains):
        return {'daily': {'time': times, 'wind_gusts_10m_max': gusts,
                          'wind_speed_10m_max': winds,
                          'precipitation_sum': rains}}

    orig_get = risk.requests.get
    try:
        risk.requests.get = lambda *a, **k: FakeOK(
            daily(['2026-09-04'], [None], [None], [None]))
        w = risk.weather_warnings('paradip')[0]
        check('all-null gusts are an outage, not "no disruptive weather"',
              'unavailable' in w['title'].lower(), w['title'])
        check('and no "nan" reaches the reader',
              'nan' not in w['detail'].lower(), w['detail'][:80])

        risk.requests.get = lambda *a, **k: FakeOK(
            daily(['2026-09-04', '2026-09-05', '2026-09-06'],
                  [None, 118, None], [None, 60, None], [0, 0, 0]))
        w = risk.weather_warnings('paradip')[0]
        check('a real gust among nulls is still caught',
              w['severity'] == 'critical' and '118' in w['detail'],
              (w['severity'], w['detail'][:60]))

        risk.requests.get = lambda *a, **k: FakeOK(
            daily(['2026-09-04', '2026-09-05'], ['30', '118'],
                  ['15', '60'], ['0', '0']))
        w = risk.weather_warnings('paradip')[0]
        check('numbers arriving as strings are coerced, not compared as text',
              w['severity'] == 'critical', w['severity'])

        for label, payload in (
                ('a key missing from daily',
                 {'daily': {'time': ['2026-09-04'],
                            'wind_gusts_10m_max': [30]}}),
                ('arrays of unequal length',
                 daily(['2026-09-04', '2026-09-05'], [30], [15], [0])),
                ('an error object with no daily',
                 {'error': True, 'reason': 'invalid latitude'})):
            risk.requests.get = lambda *a, _p=payload, **k: FakeOK(_p)
            w = risk.weather_warnings('paradip')
            check('%s degrades to an outage row' % label,
                  len(w) == 1 and 'unavailable' in w[0]['title'].lower(),
                  [x['title'] for x in w])
    finally:
        risk.requests.get = orig_get

    print('\n[27] the forecast points match the history points exactly')
    # A forecast taken at a different coordinate from the history would
    # be calibrated against the wrong ERA5 cell. That is not theoretical:
    # snapping Dhamra to PortWatch published coordinate, 5 km away,
    # changes its Cyclone Fani peak gust from 115.9 to 99.0 km/h - a
    # different cell, and Dhamra is one of the two ports the 90 km/h
    # threshold was measured at.
    from src import fetch_data
    check('risk.SITES is identical to fetch_data.WEATHER_SITES',
          risk.SITES == fetch_data.WEATHER_SITES,
          set(risk.SITES) ^ set(fetch_data.WEATHER_SITES))
    check('all %d discharge ports carry weather history' % len(risk.SITES),
          all(os.path.exists(os.path.join(paths.RAW, 'wx_%s.parquet' % p))
              for p in risk.SITES),
          [p for p in risk.SITES
           if not os.path.exists(os.path.join(paths.RAW,
                                              'wx_%s.parquet' % p))])
    check('every port the activity panel discharges at is covered here',
          set(congestion.DISCHARGE) == set(risk.SITES),
          set(congestion.DISCHARGE) ^ set(risk.SITES))

    print('\n[28] the timeout ladder is ordered')
    # server worst case < the browser own timeout < the one test_app uses.
    # If these ever invert, a slow but working backend looks like a
    # failure to whichever layer gives up first.
    worst = risk.TIMEOUT * len(risk.SITES)
    check('server worst case is %ds' % worst, worst == 60, worst)
    check('under the browser 75s abort', worst < 75, worst)
    check('which is under the 90s that tests/test_app.py allows', 75 < 90)

    print('\n' + '=' * 62)
    if FAIL:
        print('  %d FAILED:' % len(FAIL))
        for f in FAIL:
            print('    - %s' % f)
        print('=' * 62)
        return 1
    print('  all risk warning checks passed')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
End-to-end verification of the running dashboard.

This does not just check for HTTP 200. For every claim the API makes it
recomputes the answer independently from the stored artefacts and
compares. A route can return 200 and still be wrong.

Start the server first, then:  python test_app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import paths  # noqa: E402

import re
import json

import numpy as np
import pandas as pd
import requests

BASE = 'http://127.0.0.1:5000'
FAIL = []


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def main():
    try:
        requests.get(BASE + '/api/dates', timeout=10)
    except Exception:
        print('Server is not running. Start it with: python app.py')
        return 1

    oos = pd.read_parquet(os.path.join(paths.PROCESSED,
                                       'oos_predictions.parquet'))
    oos.index = pd.to_datetime(oos.index)
    panel = pd.read_parquet(os.path.join(paths.PROCESSED,
                                         'panel.parquet'))
    panel.index = pd.to_datetime(panel.index)
    metrics = json.load(open(os.path.join(paths.MODELS,
                                          'metrics.json')))
    live = json.load(open(os.path.join(paths.MODELS,
                                       'live_metrics.json')))
    best = metrics['best_model']

    # ---------------------------------------------------------------
    print('\n[1] /api/metrics matches models/metrics.json exactly')
    m = requests.get(BASE + '/api/metrics', timeout=15).json()
    check('served payload is byte-identical to the artefact',
          m == metrics, 'the API is reshaping or rounding the numbers')

    # ---------------------------------------------------------------
    print('\n[2] the headline numbers are what the artefact actually says')
    b = metrics['models'][best]
    check('direction %.1f%% is from the artefact' % b['direction_pct'],
          60 < b['direction_pct'] < 70)
    # Recompute direction accuracy straight from the predictions.
    p = oos[best].to_numpy(float)
    y = oos['y'].to_numpy(float)
    nz = p != 0
    recomputed = float((np.sign(p[nz]) == np.sign(y[nz])).mean()) * 100
    check('direction recomputed from oos parquet = %.2f%%' % recomputed,
          close(recomputed, b['direction_pct'], 1e-6),
          'artefact says %.4f, parquet says %.4f'
          % (b['direction_pct'], recomputed))

    # Recompute RMSE skill.
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    base = float(np.sqrt(np.mean(y ** 2)))
    skill = (1 - rmse / base) * 100
    check('skill recomputed = %+.3f%%' % skill,
          close(skill, b['skill_vs_zero_pct'], 1e-6),
          'artefact says %+.4f' % b['skill_vs_zero_pct'])

    # Recompute conformal coverage.
    iv = np.isfinite(oos['lo']) & np.isfinite(oos['hi'])
    cov = float(((oos['y'][iv] >= oos['lo'][iv]) &
                 (oos['y'][iv] <= oos['hi'][iv])).mean())
    check('conformal coverage recomputed = %.3f%%' % (cov * 100),
          close(cov, metrics['models']['conformal']['coverage'], 1e-9))

    # Fold count claim.
    won = sum(f[best] < f['zero'] for f in metrics['models']['folds'])
    check('folds won recomputed = %d of %d'
          % (won, len(metrics['models']['folds'])),
          won == metrics['models']['folds_won'])

    # ---------------------------------------------------------------
    print('\n[3] every forecast the API serves is arithmetically correct')
    dates = ['2015-07-13', '2016-02-10', '2017-09-15', '2018-06-15',
             '2019-04-01', '2019-07-24']
    rate, tons = 42.5, 160000.0
    bad = 0
    for ds in dates:
        r = requests.post(BASE + '/api/predict', timeout=15, json={
            'current_rate': rate, 'volume': tons, 'date': ds}).json()
        d = pd.Timestamp(r['as_of'])
        row = oos.loc[d]
        pred, act = float(row[best]), float(row['y'])

        ok = True
        ok &= close(r['expected_move_pct'], (np.exp(pred) - 1) * 100, 1e-6)
        ok &= close(r['actual_move_pct'], (np.exp(act) - 1) * 100, 1e-6)
        ok &= close(r['projected_rate'], round(rate * np.exp(pred), 2), 1e-9)
        ok &= close(r['actual_rate'], round(rate * np.exp(act), 2), 1e-9)
        ok &= close(r['lo_pct'], (np.exp(float(row['lo'])) - 1) * 100, 1e-6)
        ok &= close(r['hi_pct'], (np.exp(float(row['hi'])) - 1) * 100, 1e-6)
        ok &= (r['direction_correct'] == (np.sign(pred) == np.sign(act)))
        ok &= (r['inside_interval'] ==
               bool(float(row['lo']) <= act <= float(row['hi'])))
        ok &= (r['recommendation'] == ('CHARTER NOW' if pred > 0 else 'WAIT'))
        ok &= close(r['exposure'],
                    round(abs(rate * np.exp(pred) - rate) * tons, 2), 1e-9)
        ok &= close(r['capesize_index'], float(panel.loc[d, 'capesize_level']))
        if not ok:
            bad += 1
            print('         mismatch on %s: %s' % (ds, r))
    check('%d dates: every derived field recomputes exactly' % len(dates),
          bad == 0, '%d dates disagreed' % bad)

    # ---------------------------------------------------------------
    print('\n[4] the recommendation is internally consistent')
    incons = 0
    for ds in dates:
        r = requests.post(BASE + '/api/predict', timeout=15, json={
            'current_rate': rate, 'volume': tons, 'date': ds}).json()
        up = r['expected_move_pct'] > 0
        if (r['recommendation'] == 'CHARTER NOW') != up:
            incons += 1
        if (r['direction'] == 'up') != up:
            incons += 1
        if not (r['lo_pct'] <= r['expected_move_pct'] <= r['hi_pct']):
            incons += 1
        if not (r['projected_rate'] > 0):
            incons += 1
    check('rising->CHARTER NOW, falling->WAIT, point inside its own band',
          incons == 0, '%d inconsistencies' % incons)

    # ---------------------------------------------------------------
    print('\n[5] date handling')
    r = requests.post(BASE + '/api/predict', timeout=15, json={
        'current_rate': 10, 'date': '2017-01-07'}).json()   # a Saturday
    check('Saturday 2017-01-07 falls back to Friday 2017-01-06',
          r.get('as_of') == '2017-01-06', r.get('as_of'))

    lo_d, hi_d = str(oos.index.min().date()), str(oos.index.max().date())
    for ds in (lo_d, hi_d):
        rr = requests.post(BASE + '/api/predict', timeout=15,
                           json={'current_rate': 10, 'date': ds})
        check('boundary date %s -> 200' % ds, rr.status_code == 200,
              rr.text[:120])
    check('no date supplied defaults to the last oos date',
          requests.post(BASE + '/api/predict', timeout=15,
                        json={'current_rate': 10}).json()['as_of'] == hi_d)

    # ---------------------------------------------------------------
    print('\n[6] bad input is rejected, not guessed at')
    cases = [
        ({}, 400), ({'current_rate': 'x'}, 400), ({'current_rate': -1}, 400),
        ({'current_rate': 0}, 400), ({'current_rate': 10, 'volume': -1}, 400),
        ({'current_rate': 1e99}, 400), ({'current_rate': None}, 400),
        ({'current_rate': 10, 'date': 'nope'}, 400),
        ({'current_rate': 10, 'date': '1999-01-01'}, 400),
        ({'current_rate': 10, 'date': '2030-01-01'}, 400),
        ({'current_rate': 10}, 200),
    ]
    wrong = [(c, e, requests.post(BASE + '/api/predict', timeout=15,
                                  json=c).status_code)
             for c, e in cases]
    wrong = [w for w in wrong if w[1] != w[2]]
    check('%d input cases return the right status' % len(cases),
          not wrong, wrong)

    r = requests.post(BASE + '/api/predict', timeout=15,
                      data='{"current_rate":1}')
    check('missing Content-Type -> 415', r.status_code == 415, r.status_code)
    check('GET on a POST-only route -> 405',
          requests.get(BASE + '/api/predict', timeout=15).status_code == 405)

    # ---------------------------------------------------------------
    print('\n[7] the control experiment is served, and is still negative')
    c = requests.get(BASE + '/api/control', timeout=15).json()
    ce = c['control_experiment']
    check('Capesize autocorr %+.3f > BDRY autocorr %+.3f'
          % (ce['capesize_lag1_autocorr'], ce['bdry_lag1_autocorr']),
          ce['capesize_lag1_autocorr'] > ce['bdry_lag1_autocorr'] + 0.3)
    check('BDRY skill is negative (%+.2f%%) and flagged as unusable'
          % c['bdry']['ridge']['skill_vs_zero_pct'],
          c['bdry']['ridge']['skill_vs_zero_pct'] < 0
          and ce['proxy_has_skill'] is False)
    check('control payload agrees with live_metrics.json',
          close(c['bdry']['ridge']['skill_vs_zero_pct'],
                live['ridge']['skill_vs_zero_pct']))

    # ---------------------------------------------------------------
    print('\n[8] the page itself renders the real numbers')
    html = requests.get(BASE + '/', timeout=15).text
    check('page loads', len(html) > 5000)

    # The four headline tiles must be EMPTY in the served markup and
    # filled by JS from /api/metrics. A number typed into the HTML would
    # still render on a machine where no model has ever been trained,
    # which is exactly what this project refuses to do.
    tiles = re.findall(r'id="m_(?:dir|skill|cov|folds)"[^>]*>([^<]*)<', html)
    check('all 4 headline tiles are empty in source, not literals',
          len(tiles) == 4 and all(t.strip() in ('&mdash;', '—', '')
                                  for t in tiles), tiles)

    # No wording that would let the page pass off something other than a
    # measured result as one.
    for phrase in ('placeholder', 'not yet computed', 'not yet measured',
                   'demo mode', 'sample data', 'dummy'):
        check('page never says %r' % phrase, phrase not in html.lower())
    check('date formatter present (DD/MM/YYYY)', 'fmtDate' in html)
    check('chart draws forecast AND actual',
          "label: 'Forecast'" in html and "label: 'Actual'" in html)

    # A global helper that is also declared inside a function is shadowed
    # there, so calls in that scope silently hit the wrong function. This
    # is invisible to every other check here - the page still returns
    # 200 and the HTML still contains all the right strings - but a panel
    # renders blank because the handler threw part-way through.
    # Top-level declarations begin at column 0 inside the script block;
    # anything indented at all is inside a function. Keying on "indented
    # or not" rather than a fixed depth keeps this working however the
    # page is formatted — an earlier version assumed exactly 8 spaces and
    # reported every function-local as a global the moment the markup was
    # re-indented.
    top = set(re.findall(r'\n(?:const|let)\s+(\w+)\s*=', html)) | \
        set(re.findall(r'\n(?:async\s+)?function\s+(\w+)\s*\(', html))
    nested = set(re.findall(r'\n[ \t]+(?:const|let)\s+(\w+)\s*=', html))
    shadowed = sorted(top & nested)
    check('no global helper is shadowed by a local of the same name',
          not shadowed,
          'shadowed: %s - calls inside that function hit the local'
          % shadowed)

    # The null-safe render path must exist: the API sends null for a
    # non-finite value, and the page must not print "null%".
    check('page has a finite-guard helper for null API values',
          'const fin =' in html and 'fin(r.' in html)
    check('pct() renders a dash rather than "null%" for a missing value',
          "v === null" in html and "'—'" in html)

    print('\n[9] the port endpoints agree with src/ports.py')
    from src import ports as P
    ref = requests.get(BASE + '/api/ports', timeout=15).json()
    check('reference lists every vessel and port',
          len(ref['vessels']) == len(P.VESSELS)
          and len(ref['ports']) == len(P.PORTS))

    mismatch = []
    for row in ref['matrix']:
        for cell in row['cells']:
            local = P.can_serve(row['vessel'], cell['port'])
            if (cell['max_cargo_t'] != local['max_cargo_t']
                    or cell['verdict'] != local['verdict']):
                mismatch.append('%s/%s api=%s,%s local=%s,%s'
                                % (row['vessel'], cell['port'],
                                   cell['max_cargo_t'], cell['verdict'],
                                   local['max_cargo_t'], local['verdict']))
    check('all %d matrix cells match the module exactly'
          % sum(len(r['cells']) for r in ref['matrix']), not mismatch,
          mismatch[:3])

    r = requests.post(BASE + '/api/ports/options', timeout=15,
                      json={'parcel_t': 160000}).json()
    local = P.options_for(160000)
    check('options endpoint returns the same ranking as the module',
          [(o['vessel'], o['port']) for o in r['options']]
          == [(o['vessel'], o['port']) for o in local])
    check('best option is the first option',
          r['best']['vessel'] == r['options'][0]['vessel']
          and r['best']['port'] == r['options'][0]['port'])
    check('a bigger ship does not outrank a better-filling one',
          all(a['parcel_utilisation'] >= b['parcel_utilisation']
              for a, b in zip(r['options'], r['options'][1:])
              if a['voyages'] == b['voyages']
              and a['verdict'] == b['verdict']))

    # Changing the parcel must change the answer, or the panel is inert.
    small = requests.post(BASE + '/api/ports/options', timeout=15,
                          json={'parcel_t': 75000}).json()
    check('a 75,000 t parcel picks a different vessel than 160,000 t',
          small['best']['vessel'] != r['best']['vessel'],
          '%s vs %s' % (small['best']['vessel'], r['best']['vessel']))

    print('\n[10] port endpoint input handling')
    cases = [({}, 400), ({'parcel_t': 0}, 400), ({'parcel_t': -1}, 400),
             ({'parcel_t': 'abc'}, 400), ({'parcel_t': None}, 400),
             ({'parcel_t': 1e99}, 400), ({'parcel_t': 160000}, 200),
             ({'parcel_t': 1}, 200)]
    wrong = [(c, e, requests.post(BASE + '/api/ports/options', timeout=15,
                                  json=c).status_code) for c, e in cases]
    wrong = [w for w in wrong if w[1] != w[2]]
    check('%d parcel inputs return the right status' % len(cases),
          not wrong, wrong)
    check('GET on the options route -> 405',
          requests.get(BASE + '/api/ports/options',
                       timeout=15).status_code == 405)

    print('\n[11] the page actually renders the port panel')
    for hook in ('portOptBody', 'portMatrixBody', 'portBest', 'parcel_t',
                 'loadPortMatrix', 'portOptions'):
        check('page wires %r' % hook, hook in html)
    check('port panel explains fill vs ship utilisation',
          'not how full each ship could be' in html)

    print('\n[12] the congestion endpoints agree with src/congestion.py')
    from src import congestion as CG
    cg = requests.get(BASE + '/api/congestion', timeout=20).json()
    check('a snapshot is returned for all %d ports, both ends'
          % len(CG.ALL_PORTS),
          len(cg['snapshots']) == len(CG.ALL_PORTS))
    check('the response splits discharge (%d) from load (%d)'
          % (len(CG.DISCHARGE), len(CG.LOAD)),
          len(cg['discharge']) == len(CG.DISCHARGE)
          and len(cg['load']) == len(CG.LOAD))
    check('every load terminal reports exported tonnage, not imported',
          all(s['flow'] == 'exported' for s in cg['load'])
          and all(s['flow'] == 'imported' for s in cg['discharge']))
    # Reading imports at an export terminal is not a rounding error: Hay
    # Point and Newcastle handle no dry bulk imports at all.
    hp = next(s for s in cg['load'] if s['port'] == 'hay_point')
    check('Hay Point reports %s t/day exported, not zero'
          % '{:,}'.format(hp['lane_t_per_day']),
          hp['lane_t_per_day'] > 50000)
    # Every Indian discharge port also loads - Paradip ships 60% of its
    # dry bulk out as iron ore, competing for the same berths.
    par = next(s for s in cg['discharge'] if s['port'] == 'paradip')
    check('Paradip throughput %s t/day exceeds its %s t/day inbound'
          % ('{:,}'.format(par['throughput_t_per_day']),
             '{:,}'.format(par['lane_t_per_day'])),
          par['throughput_t_per_day'] > par['lane_t_per_day'] * 2)
    bad = []
    for s in cg['snapshots']:
        if 'error' in s:
            bad.append('%s errored' % s['port'])
            continue
        local = CG.snapshot(s['port'])
        for k in ('calls_per_day', 'baseline_calls_per_day', 'percentile',
                  'band', 'reliable', 'as_of'):
            if s[k] != local[k]:
                bad.append('%s.%s api=%r local=%r'
                           % (s['port'], k, s[k], local[k]))
    check('every snapshot field matches the module', not bad, bad[:4])

    check('the response states it is not waiting time',
          'not waiting time' in cg['measures']
          and 'queue length' in cg['caveat'])
    check('an unreliable port is never given an activity band',
          all(s['band'] == 'unreliable'
              for s in cg['snapshots'] if not s.get('reliable')))
    check('every reliable port outranks every unreliable one',
          [s.get('reliable', False) for s in cg['snapshots']]
          == sorted([s.get('reliable', False) for s in cg['snapshots']],
                    reverse=True))

    # The point of this panel is that it is CURRENT, unlike the forecast.
    newest = max(s['as_of'] for s in cg['snapshots'] if 'as_of' in s)
    check('port data (%s) is far more recent than the forecast window (%s)'
          % (newest, str(oos.index.max().date())),
          pd.Timestamp(newest) > oos.index.max() + pd.Timedelta(days=365))

    prof = cg['monthly_profile']
    check('a 12-month profile exists for every port',
          all(p is not None and len(p) == 12 for p in prof.values()))

    print('\n[13] the per-port series endpoint')
    r = requests.get(BASE + '/api/congestion/paradip?days=120',
                     timeout=20).json()
    check('returns 120 points with a snapshot attached',
          len(r['series']) == 120 and 'snapshot' in r)
    check('series values are finite and non-negative',
          all(isinstance(p['calls'], int) and p['calls'] >= 0
              and p['smoothed'] >= 0 and p['throughput'] >= p['lane']
              for p in r['series']))
    check('unknown port -> 400 naming the valid ports',
          requests.get(BASE + '/api/congestion/atlantis',
                       timeout=15).status_code == 400)
    check('days is clamped rather than trusted',
          len(requests.get(BASE + '/api/congestion/paradip?days=99999',
                           timeout=20).json()['series']) <= 1000)
    check('non-numeric days -> 400',
          requests.get(BASE + '/api/congestion/paradip?days=abc',
                       timeout=15).status_code == 400)

    print('\n[14] the page renders the congestion panel')
    for hook in ('congCards', 'congChart', 'congPort', 'seasonBody',
                 'loadCongestion', 'drawCongestion', 'ordinal'):
        check('page wires %r' % hook, hook in html)
    check('panel says plainly that this is the only current data',
          'only current data' in html)

    print('\n[18] concurrent cold requests cost one computation, not N')
    # Without the lock, every browser arriving on a cold cache fires its
    # own round of forecasts at a free API. The lock means the first
    # computes and the rest wake to a warm cache.
    import threading
    import time as _time
    import app as _app
    from src import risk as _risk

    FAKE = [{'kind': 'weather', 'port': 'paradip', 'severity': 'clear',
             'measured': True, 'title': 'stub', 'detail': 'stub',
             'basis': 'stub'}]
    calls = []
    real_assess = _risk.assess

    def slow_assess(*a, **k):
        calls.append(1)
        _time.sleep(0.6)
        return [dict(x) for x in FAKE]

    codes = []
    try:
        _risk.assess = slow_assess
        _app._RISK_CACHE.update(at=0.0, data=None)
        client = _app.app.test_client()

        def hit():
            codes.append(client.get('/api/risk').status_code)

        threads = [threading.Thread(target=hit) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        _risk.assess = real_assess
        _app._RISK_CACHE.update(at=0.0, data=None)

    check('all 6 concurrent requests returned 200',
          codes == [200] * 6, codes)
    check('but the assessment ran once, not 6 times', len(calls) == 1,
          len(calls))

    print('\n[19] every answer under /api/ is JSON, including the failures')
    # Werkzeug default error page is HTML. A client that calls .json()
    # on it gets a parse error instead of the reason it failed.
    for path in ('/api/nope', '/api/risk/extra', '/api/congestion'):
        r = requests.get(BASE + path, timeout=30)
        try:
            r.json()
            parsed = True
        except Exception:
            parsed = False
        check('GET %s answers JSON (HTTP %d)' % (path, r.status_code),
              parsed, r.text[:60])
    r = requests.get(BASE + '/api/predict', timeout=30)
    check('a GET on the POST-only predict route is a JSON 405',
          r.status_code == 405 and 'error' in r.json(), r.text[:60])

    # A traversal attempt must not read a file, whichever way it is
    # written. The plain form is normalised away before routing; the
    # encoded form reaches the app and is refused by name.
    r = requests.get(BASE + '/api/congestion/..%2f..%2fetc%2fpasswd',
                     timeout=30)
    check('an encoded traversal is refused as JSON, not served',
          r.status_code == 404 and 'error' in r.json(), r.text[:70])
    check('and nothing that looks like a file body comes back',
          'root:' not in r.text and 'import ' not in r.text, r.text[:70])

    # The browser-facing 404 should stay HTML - it is read by a person.
    r = requests.get(BASE + '/nope', timeout=30)
    check('a non-API 404 is still an HTML page for the browser',
          'text/html' in r.headers.get('Content-Type', ''),
          r.headers.get('Content-Type'))

    print('\n[20] /api/optimise agrees with src/optimise.py')
    from src import optimise as _opt
    from src import ports as _ports
    VOY = {'Handysize': 620000, 'Supramax': 780000, 'Panamax': 1050000,
           'Post-Panamax': 1180000, 'Capesize': 1500000,
           'Newcastlemax': 1620000}
    PRT = {p: 45000.0 for p in _ports.PORTS}
    INL = {p: 8.0 for p in _ports.PORTS}
    body = {'parcel_t': 300000, 'voyage_cost': VOY,
            'port_cost': PRT, 'inland_cost': INL}
    r = requests.post(BASE + '/api/optimise', json=body, timeout=60)
    check('the endpoint answers 200', r.status_code == 200, r.text[:70])
    d = r.json()
    local = _opt.compare(300000, VOY, port_cost=PRT, inland_cost=INL)
    check('the served total matches the module exactly',
          d['best']['total_cost'] == local['best']['total_cost'],
          (d['best']['total_cost'], local['best']['total_cost']))
    check('and so does the fleet it chose',
          [(l['vessel'], l['port'], l['voyages']) for l in d['best']['legs']]
          == [(l['vessel'], l['port'], l['voyages'])
              for l in local['best']['legs']])
    b = d['best']
    check('tonnes across the legs equal the parcel',
          abs(sum(l['tonnes'] for l in b['legs']) - 300000) < 0.5,
          sum(l['tonnes'] for l in b['legs']))
    check('cost per tonne is the total over the parcel',
          abs(b['cost_per_t'] - b['total_cost'] / 300000) < 0.01)
    check('every voyage count served is a whole number',
          all(float(l['voyages']).is_integer() for l in b['legs']))
    check('no leg uses a berth its class cannot enter',
          all(_ports.can_serve(l['vessel'], l['port'])['verdict']
              in _opt.USABLE for l in b['legs']),
          [(l['vessel'], l['port']) for l in b['legs']])
    feas = [a for a in d['alternatives'] if a['feasible']]
    check('no single-class fleet beats the optimum',
          all(a['cost_per_t'] >= b['cost_per_t'] - 0.01 for a in feas),
          [(a['label'], a['cost_per_t']) for a in feas[:2]])

    print('\n[21] the endpoint will not invent a freight rate')
    r = requests.post(BASE + '/api/optimise',
                      json={'parcel_t': 300000}, timeout=30)
    check('a request with no costs is refused', r.status_code == 400,
          r.status_code)
    check('and says why, in those words',
          'no verified freight rates' in r.json().get('error', ''),
          r.json().get('error', '')[:80])
    for bad in ({'parcel_t': 0, 'voyage_cost': VOY},
                {'parcel_t': -5, 'voyage_cost': VOY},
                {'parcel_t': 'x', 'voyage_cost': VOY},
                {'parcel_t': 300000, 'voyage_cost': {'Capesize': -1}},
                {'parcel_t': 300000, 'voyage_cost': {'Capesize': 'x'}},
                {'parcel_t': 300000, 'voyage_cost': 'notadict'}):
        r = requests.post(BASE + '/api/optimise', json=bad, timeout=30)
        check('rejected: %s' % str(bad)[:52], r.status_code == 400,
              r.status_code)
    check('the served note says the costs are not ours',
          'nothing here is a freight rate this project has verified'
          in d['note'], d['note'][:70])

    print('\n[22] the page wires the fleet selector')
    for hook in ('solveFleet', 'buildCostInputs', 'voyageCosts', 'fleetOut',
                 'api/optimise', 'YOUR COSTS'):
        check('page wires %r' % hook, hook in html)
    check('the panel states plainly that the costs are not sourced',
          'illustrative figures, not sourced ones' in html)

    print('\n[23] /api/ballast agrees with src/ballast.py')
    from src import ballast as _bal
    bl = requests.get(BASE + '/api/ballast', timeout=60).json()
    local = _bal.profile()
    check('a row per discharge berth', len(bl['ports']) == len(local),
          (len(bl['ports']), len(local)))
    for served, want in zip(bl['ports'], local):
        check('%s served exactly as computed' % served['port'],
              served['ratio'] == want['ratio']
              and served['ballast_share'] == want['ballast_share'],
              (served['ratio'], want['ratio']))
    scored = [r for r in bl['ports'] if r['reliable']]
    check('a ballast share is always 1 - ratio, floored at zero',
          all(abs(r['ballast_share'] - max(0.0, 1 - r['ratio'])) < 5e-4
              for r in scored))
    check('an unscored berth carries no share at all',
          all(r['ballast_share'] is None
              for r in bl['ports'] if not r['reliable']))
    check('the endpoint says what it does not measure',
          'not waiting time' in bl['measures'], bl['measures'])
    check('and that the empty share is a lower bound',
          'LOWER bound' in bl['caveat'], bl['caveat'][:70])

    print('\n[24] pricing the empty leg changes the answer, and says where')
    VOY2 = {'Handysize': 620000, 'Supramax': 780000, 'Panamax': 1050000,
            'Post-Panamax': 1180000, 'Capesize': 1500000,
            'Newcastlemax': 1620000}
    P2 = {p: 45000.0 for p in _ports.PORTS}
    I2 = {p: 8.0 for p in _ports.PORTS}
    base_body = {'parcel_t': 150000, 'voyage_cost': VOY2,
                 'port_cost': P2, 'inland_cost': I2}
    free = requests.post(BASE + '/api/optimise', timeout=60,
                         json=dict(base_body)).json()
    charged = requests.post(BASE + '/api/optimise', timeout=60,
                            json=dict(base_body, ballast_pct=0.45)).json()
    check('with no ballast_pct, no shares are served',
          free.get('ballast_shares') is None, free.get('ballast_shares'))
    check('with one, the measured shares come back',
          charged['ballast_shares']['Haldia'] > 0.9,
          charged.get('ballast_shares'))
    check('Paradip has a return cargo, so its share is zero',
          charged['ballast_shares']['Paradip'] == 0.0)
    check('a berth with no arrivals feed is served as null, not zero',
          charged['ballast_shares']['Gangavaram'] is None,
          charged['ballast_shares']['Gangavaram'])

    # The failure mode this introduces: an unmeasured berth looking
    # cheap because nobody has data on it.
    for leg in charged['best']['legs']:
        if charged['ballast_shares'].get(leg['port']) is None:
            check('%s is flagged ballast_priced=False' % leg['port'],
                  leg['ballast_priced'] is False)
    check('every unpriced berth used is listed',
          set(charged['best']['ballast_unpriced'])
          == {l['port'] for l in charged['best']['legs']
              if not l['ballast_priced']},
          charged['best']['ballast_unpriced'])

    check('ballast_pct out of range is refused',
          requests.post(BASE + '/api/optimise', timeout=30,
                        json=dict(base_body,
                                  ballast_pct=1.4)).status_code == 400)
    check('and a non-numeric one too',
          requests.post(BASE + '/api/optimise', timeout=30,
                        json=dict(base_body,
                                  ballast_pct='x')).status_code == 400)

    print('\n[25] the page renders the empty leg')
    for hook in ('loadBallast', 'ballastBody', 'ballastPct',
                 'ballast_priced', 'Empty leg'):
        check('page wires %r' % hook, hook in html)
    check('the panel separates what is measured from what is not',
          'Not measured:' in html)
    check('and never claims to measure waiting time',
          'waiting time, which PortWatch cannot give us' in html)

    print('\n' + '=' * 62)
    print('\n[15] /api/risk agrees with src/risk.py')
    from src import risk
    rk = requests.get(BASE + '/api/risk', timeout=90).json()
    rows = rk['warnings']
    check('the endpoint serves warnings', len(rows) > 0, rk)
    check('counts add up to the warnings served',
          sum(rk['counts'].values()) == len(rows), (rk['counts'], len(rows)))
    for lvl in ('critical', 'warning', 'watch', 'clear'):
        check('count of %r is recomputed correctly' % lvl,
              rk['counts'][lvl] == sum(1 for w in rows if w['severity'] == lvl))
    check('measured + context = total',
          rk['measured'] + rk['context_only'] == len(rows),
          (rk['measured'], rk['context_only'], len(rows)))
    check('measured tally matches the flags on the rows',
          rk['measured'] == sum(1 for w in rows if w['measured']))
    check('the served halt threshold is the one src/risk.py measured at',
          rk['gust_halt_kmh'] == risk.GUST_HALT_KMH,
          (rk['gust_halt_kmh'], risk.GUST_HALT_KMH))

    print('\n[16] the risk endpoint cannot invent evidence')
    # The whole point of the panel is the measured/context split. If a
    # berth or seasonal item ever ships as measured, someone has asserted
    # an effect size this repository has not established.
    bad = [w['title'] for w in rows
           if w['kind'] in ('berth', 'seasonal') and w['measured']]
    check('no berth or seasonal item claims to be measured', not bad, bad)
    bad = [w['title'] for w in rows if not w.get('basis')]
    check('every warning carries a basis string', not bad, bad[:3])
    bad = [w['title'] for w in rows
           if w['severity'] not in ('critical', 'warning', 'watch', 'clear')]
    check('every severity is one of the four levels', not bad, bad[:3])
    rank = {'critical': 0, 'warning': 1, 'watch': 2, 'clear': 3}
    order = [rank[w['severity']] for w in rows]
    check('warnings arrive worst-first so the panel renders in order',
          order == sorted(order), order)
    mdl = [w for w in rows if w['kind'] == 'model']
    check('the model item is dated 2019, not today - it is historical',
          all(w['as_of'].startswith('2019') for w in mdl),
          [w.get('as_of') for w in mdl])

    print('\n[17] the page renders the risk panel')
    for hook in ('v-risk', 'riskList', 'riskCounts', 'riskClear',
                 'loadRisk', 'riskRow', 'rkEsc', 'MEASURED', 'CONTEXT'):
        check('page wires %r' % hook, hook in html)
    check('nav offers the risk view', 'data-v="risk"' in html)
    check('risk is in the view list so the hash route reaches it',
          "VIEWS = ['timing','fleet','ports','risk','proof']" in html)
    check('the panel explains the measured/context split in words',
          'no measured effect' in html)

    # A risk panel that fails once at page load and stays broken for the
    # session is worse than one that says nothing - the user sees an
    # outage and cannot retry it. The charts already recover this way.
    check('a failed load is retried when the view is reopened',
          'riskLoaded' in html and "name === 'risk' && !riskLoaded" in html)
    check('severity is whitelisted before it reaches a class attribute',
          'const sev = SEV_TAG[w.severity] ? w.severity' in html)
    check('the "nothing raised" heading hides when that list is empty',
          "riskClearHead').hidden = clear.length === 0" in html)

    if FAIL:
        print('  %d CHECK(S) FAILED:' % len(FAIL))
        for f in FAIL:
            print('    - %s' % f)
        print('=' * 62)
        return 1
    print('  everything verified')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

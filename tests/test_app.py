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
    top = set(re.findall(r'\n {8}(?:const|let)\s+(\w+)\s*=', html)) | \
        set(re.findall(r'\n {8}function\s+(\w+)\s*\(', html))
    nested = set(re.findall(r'\n {12,}(?:const|let)\s+(\w+)\s*=', html))
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

    print('\n' + '=' * 62)
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

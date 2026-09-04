"""
Checks on the evaluation protocol itself.

This suite exists because of a hole a mutation test found: inverting the
purge gap so the training window OVERLAPPED the test window - the
textbook walk-forward leak this project's README leads with - left every
other suite in the repository green. `test_leakage.py` guards the panel,
not the fold construction, and nothing imported `train_model` at all.

So the two things checked here are the two nothing else checks:

  1. the folds are built correctly, asserted directly on the boundaries
  2. the headline figures are compared against their TARGETS, not merely
     against what `metrics.json` happens to say

The second matters because every existing check of coverage and skill is
a consistency check: it recomputes the number from the artefact and
compares it to the artefact's own summary. If the model produced 50%
coverage and honestly wrote 50% into metrics.json, all of those pass.

Run:  python -m tests.test_walkforward
"""

import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import train_model as tm, paths  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def main():
    panel = os.path.join(paths.PROCESSED, 'panel.parquet')
    metrics = os.path.join(paths.MODELS, 'metrics.json')
    oos = os.path.join(paths.PROCESSED, 'oos_predictions.parquet')
    for p in (panel, metrics, oos):
        if not os.path.exists(p):
            print('  missing %s - run python -m src.train_model first' % p)
            return 1
    df = pd.read_parquet(panel)
    m = json.load(open(metrics, encoding='utf-8'))
    o = pd.read_parquet(oos)
    o.index = pd.to_datetime(o.index)
    n = len(df)

    print('\n[1] the purge gap is real, on both boundaries, in every fold')
    fs = tm.folds(n)
    check('folds() returns %d folds, matching N_FOLDS' % len(fs),
          len(fs) == tm.N_FOLDS, (len(fs), tm.N_FOLDS))
    for k, te_lo, te_hi, tr_hi, cal_lo, fit_hi in fs:
        # The target at row i is built from row i+HORIZON. A training
        # block ending at tr_hi therefore reaches to tr_hi+HORIZON-1, so
        # the test block must not start before that.
        check('fold %d: training ends %d rows before the test block starts'
              % (k, te_lo - tr_hi), te_lo - tr_hi >= tm.HORIZON,
              (tr_hi, te_lo, tm.HORIZON))
        check('fold %d: the fit block ends %d rows before calibration'
              % (k, cal_lo - fit_hi), cal_lo - fit_hi >= tm.HORIZON,
              (fit_hi, cal_lo))
        check('fold %d: fit and calibration do not overlap' % k,
              fit_hi <= cal_lo, (fit_hi, cal_lo))
        check('fold %d: calibration ends before the test block' % k,
              tr_hi <= te_lo, (tr_hi, te_lo))
        check('fold %d: the test block is non-empty and inside the panel'
              % k, 0 < te_lo < te_hi <= n, (te_lo, te_hi, n))

    print('\n[2] folds move forward and never reuse a scored row')
    check('test blocks are strictly increasing',
          all(fs[i][1] < fs[i + 1][1] for i in range(len(fs) - 1)),
          [f[1] for f in fs])
    seen = set()
    overlap = []
    for _, te_lo, te_hi, _, _, _ in fs:
        rows = set(range(te_lo, te_hi))
        overlap += sorted(rows & seen)[:3]
        seen |= rows
    check('no row is scored by two folds', not overlap, overlap[:5])
    check('the training window expands, never shrinks',
          all(fs[i][5] <= fs[i + 1][5] for i in range(len(fs) - 1)),
          [f[5] for f in fs])

    print('\n[3] conformal_quantile implements the finite-sample formula')
    rng = np.random.default_rng(0)
    for size in (25, 100, 404, 1000):
        r = np.abs(rng.normal(size=size))
        got = tm.conformal_quantile(r, tm.ALPHA)
        k = math.ceil((size + 1) * (1 - tm.ALPHA))
        want = float(np.quantile(r, k / size, method='higher'))
        check('n=%d: matches ceil((n+1)(1-a))/n taken from above' % size,
              got == want, (got, want))
        plain = float(np.quantile(r, 1 - tm.ALPHA))
        check('n=%d: and is never narrower than the plain quantile' % size,
              got >= plain, (got, plain))
    check('an empty calibration block cannot bound anything',
          tm.conformal_quantile([], tm.ALPHA) == float('inf'))
    check('too few points to bound at this level falls back to the max',
          tm.conformal_quantile([1.0, 2.0], 0.01) == 2.0)

    print('\n[4] the headline figures are checked against their TARGETS')
    # Not against metrics.json - against what the numbers must BE for the
    # claims in the README to stand.
    cov = float(((o['lo'] <= o['y']) & (o['y'] <= o['hi'])).mean())
    check('realised coverage %.1f%% is at or above the 80%% target'
          % (cov * 100), cov >= 1 - tm.ALPHA, cov)
    check('and is not absurdly wide either (under 90%%)', cov < 0.90, cov)
    rmse = {k: float(np.sqrt(np.mean((o['y'] - o[k]) ** 2)))
            for k in ('zero', 'momentum', 'ridge', 'lgbm')}
    skill = (1 - rmse['ridge'] / rmse['zero']) * 100
    check('ridge beats assume-no-change (skill %+.2f%% > 0)' % skill,
          skill > 0, skill)
    check('and beats momentum too', rmse['ridge'] < rmse['momentum'],
          (rmse['ridge'], rmse['momentum']))
    da = float((np.sign(o['ridge']) == np.sign(o['y'])).mean())
    check('direction %.1f%% is better than a coin flip' % (da * 100),
          da > 0.5, da)
    check('but is NOT implausibly high - over 75%% on a 5-day freight '
          'return would mean a leak, not a model', da < 0.75, da)

    print('\n[5] the local scaling earns its extra width')
    c = m['models']['conformal'] if 'conformal' in m.get('models', {}) \
        else m.get('conformal', {})
    check('metrics.json records the plain-conformal control',
          'plain_coverage' in c, sorted(c))
    if 'plain_coverage' in c:
        check('local (%.1f%%) covers better than plain (%.1f%%)'
              % (c['coverage'] * 100, c['plain_coverage'] * 100),
              c['coverage'] > c['plain_coverage'],
              (c['coverage'], c['plain_coverage']))
        check('and it costs width, which the docs must not hide '
              '(%.3f vs %.3f)' % (c['mean_width'], c['plain_mean_width']),
              c['mean_width'] > c['plain_mean_width'])

    print('\n[6] the significance test uses the EFFECTIVE sample')
    # Overlapping 5-day targets mean 1,010 rows are not 1,010
    # observations. Testing on the row count would inflate significance
    # about fivefold.
    check('metrics.json reports n_effective', 'n_effective' in m, sorted(m))
    if 'n_effective' in m and 'n_scored' in m:
        check('n_effective (%d) is n_scored (%d) divided by the horizon'
              % (m['n_effective'], m['n_scored']),
              m['n_effective'] == m['n_scored'] // m['horizon_days'],
              (m['n_effective'], m['n_scored'], m['horizon_days']))
        check('and is far smaller than the row count, as it must be',
              m['n_effective'] < m['n_scored'] / 2)

    print('\n[7] the artefact and the summary still agree')
    # The consistency check the other suites already do - kept here so
    # this file fails loudly if the two drift apart.
    for k in ('zero', 'momentum', 'ridge', 'lgbm'):
        check('%s RMSE matches metrics.json' % k,
              abs(rmse[k] - m['models'][k]['rmse']) < 5e-5,
              (rmse[k], m['models'][k]['rmse']))
    check('scored-row count matches', len(o) == m['n_scored'],
          (len(o), m['n_scored']))

    print('\n[8] the edge is concentrated in the moves that matter')
    # The README and METHOD.md publish this table. It came from a
    # research script the first time and three of its four rows were
    # wrong against the production model, so it is asserted here now.
    mag = o['y'].abs()
    band = []
    for lo_q, hi_q in ((0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)):
        a, b = mag.quantile(lo_q), mag.quantile(hi_q)
        q = (mag >= a) & (mag <= b)
        d = float((np.sign(o['ridge'][q]) == np.sign(o['y'][q])).mean())
        up = float((o['y'][q] > 0).mean())
        band.append((d * 100, (d - max(up, 1 - up)) * 100, int(q.sum())))
    for i, (d, lift, n) in enumerate(band):
        print('       quartile %d: n=%d  direction %.1f%%  lift %+.1fpp'
              % (i + 1, n, d, lift))
    check('direction rises monotonically with the size of the move: %s'
          % ' '.join('%.1f' % b[0] for b in band),
          all(band[i][0] <= band[i + 1][0] + 0.5 for i in range(3)),
          [b[0] for b in band])
    check('the largest quartile beats the smallest by at least 10 points '
          '(%.1f vs %.1f)' % (band[3][0], band[0][0]),
          band[3][0] - band[0][0] >= 10, (band[3][0], band[0][0]))
    check('the smallest quartile adds nothing, and the docs say so '
          '(%.1f%%, lift %+.1fpp)' % (band[0][0], band[0][1]),
          band[0][1] < 2.0, band[0][1])
    check('the largest quartile lift is real, not a constant-guess artefact '
          '(%+.1fpp)' % band[3][1], band[3][1] > 10, band[3][1])
    check('every quartile holds a quarter of the scored rows',
          all(abs(b[2] - len(o) / 4) <= 2 for b in band),
          [b[2] for b in band])

    print('\n[9] the confidence tiering cannot see the future')
    # This is the whole claim. "The strongest half of weeks" is only an
    # honest thing to publish if a week can be placed in that half using
    # information available when the call is made. A quantile over the
    # full test period would need next year's predictions to rank this
    # one, and the tier table would be a look-ahead artefact.
    rng2 = np.random.default_rng(7)
    n_t, burn = 900, 250
    pa = rng2.normal(size=n_t)
    ya = rng2.normal(size=n_t)

    # Truncation invariance is the sharpest statement of it: scoring a
    # shorter history must give the SAME answer for the rows both runs
    # share. If it does not, later rows are informing earlier ones.
    for share in (0.75, 0.5, 0.25):
        full = tm.tier_mask(pa, share, burn)
        part = tm.tier_mask(pa[:600], share, burn)
        check('share %.0f%%: truncating the history leaves earlier '
              'decisions untouched' % (share * 100),
              np.array_equal(full[:600], part),
              int(np.sum(full[:600] != part)))

    # And the direct version: rewrite the tail with enormous values.
    # A whole-period quantile would move the threshold and de-select
    # rows in the first half; a causal one cannot.
    pb = pa.copy()
    pb[600:] = pa[600:] * 50 + 100
    for share in (0.75, 0.5, 0.25):
        check('share %.0f%%: a huge move in 2026 does not un-call a week '
              'in 2022' % (share * 100),
              np.array_equal(tm.tier_mask(pa, share, burn)[:600],
                             tm.tier_mask(pb, share, burn)[:600]))

    check('no row inside the burn-in is ever selected',
          not tm.tier_mask(pa, 0.75, burn)[:burn].any())
    # The boundary needs its own branch and therefore its own check.
    # Left to the general path, "the strongest 100%" compares each week
    # against the running MINIMUM, so a week weaker than anything before
    # it fails a >= test and drops out - one row in 250. Never reached
    # by the published 75/50/25, but the name has to mean what it says.
    check('share=1.0 selects every eligible week, with none lost to a '
          'new running minimum',
          int(tm.tier_mask(pa, 1.0, burn).sum()) == n_t - burn,
          (int(tm.tier_mask(pa, 1.0, burn).sum()), n_t - burn))
    check('and share=1.0 still respects the burn-in',
          not tm.tier_mask(pa, 1.0, burn)[:burn].any())
    m75 = tm.tier_mask(pa, 0.75, burn)
    m50 = tm.tier_mask(pa, 0.50, burn)
    m25 = tm.tier_mask(pa, 0.25, burn)
    check('the tiers nest: strongest 25% is inside 50% is inside 75%',
          bool((m25 <= m50).all() and (m50 <= m75).all()),
          (int(m25.sum()), int(m50.sum()), int(m75.sum())))
    for mk, want in ((m75, 0.75), (m50, 0.50), (m25, 0.25)):
        got = mk.sum() / (n_t - burn)
        check('a %.0f%% tier actually selects about %.0f%% of eligible '
              'weeks (%.1f%%)' % (want * 100, want * 100, got * 100),
              abs(got - want) < 0.05, got)

    # Negative control. Tiering a model with no edge must not create
    # one - if slicing by |prediction| lifted random noise above 50%,
    # the lift on the real model would be an artefact of the slicing.
    noise = tm.confidence_tiers(pa, ya, min_history=burn)
    check('tiering pure noise stays near a coin flip: %s'
          % ' '.join('%.1f%%' % (t['direction'] * 100)
                     for t in noise['tiers']),
          all(0.40 < t['direction'] < 0.60 for t in noise['tiers']),
          [t['direction'] for t in noise['tiers']])
    check('too short a history returns None rather than a tier table',
          tm.confidence_tiers(pa[:100], ya[:100]) is None)

    print('\n[10] the published tier table matches the predictions file')
    conf = m['models'].get('confidence')
    check('metrics.json carries the confidence tiers', bool(conf),
          sorted(m['models']))
    if conf:
        yv = o['y'].to_numpy(dtype=float)
        pv = o[conf['model']].to_numpy(dtype=float)
        check('it is tiered on the model the page headlines (%s)'
              % conf['model'], conf['model'] == m['best_model'],
              (conf['model'], m['best_model']))
        elig = np.zeros(len(o), dtype=bool)
        elig[conf['min_history']:] = True
        check('eligible weeks = scored rows minus the burn-in (%d)'
              % conf['n_eligible'],
              conf['n_eligible'] == len(o) - conf['min_history'],
              (conf['n_eligible'], len(o), conf['min_history']))
        hit = np.sign(pv) == np.sign(yv)
        check('the "always" row recomputes from the predictions file',
              abs(float(hit[elig].mean()) - conf['all_direction']) < 5e-9,
              (float(hit[elig].mean()), conf['all_direction']))
        # The baseline must be the eligible rows, not all of them -
        # otherwise the lift is partly the burn-in being dropped.
        check('and is NOT simply the headline over every scored row',
              elig.sum() < len(o))
        for t in conf['tiers']:
            sel = tm.tier_mask(pv, t['share'], conf['min_history'])
            check('the %.0f%% tier recomputes: %d weeks, %.1f%%'
                  % (t['share'] * 100, t['n'], t['direction'] * 100),
                  int(sel.sum()) == t['n']
                  and abs(float(hit[sel].mean()) - t['direction']) < 5e-9,
                  (int(sel.sum()), t['n']))
            check('the %.0f%% tier is tested on independent windows, not '
                  'rows (%d vs %d)' % (t['share'] * 100, t['n_effective'],
                                       t['n']),
                  t['n_effective'] == max(1, t['n'] // m['horizon_days']),
                  (t['n_effective'], t['n']))
            check('the %.0f%% tier null is the best CONSTANT call, never '
                  'below a coin (%.3f)' % (t['share'] * 100, t['null_rate']),
                  t['null_rate'] >= 0.5, t['null_rate'])
        half = [t for t in conf['tiers'] if abs(t['share'] - 0.5) < 1e-9]
        check('a strongest-half tier is published', len(half) == 1)
        if half:
            h = half[0]
            check('the claim the page makes holds: the strongest half '
                  '(%.1f%%) beats every week (%.1f%%)'
                  % (h['direction'] * 100, conf['all_direction'] * 100),
                  h['direction'] > conf['all_direction'],
                  (h['direction'], conf['all_direction']))
            check('and that lift is significant after adjusting for the '
                  'tiers tested (p=%.5f)' % h['p_value'],
                  h['significant'] and h['p_value'] < 0.05, h['p_value'])
            check('but is not implausibly high either - over 85% on a '
                  '5-day freight return would mean a leak',
                  h['direction'] < 0.85, h['direction'])

    print('\n[11] the per-year breakdown is not a flattering slice')
    g = m['models'].get('regimes')
    check('metrics.json carries the per-year regimes', bool(g),
          sorted(m['models']))
    if g:
        yv = o['y'].to_numpy(dtype=float)
        pv = o['ridge'].to_numpy(dtype=float)
        zv = o['zero'].to_numpy(dtype=float)
        hit = np.sign(pv) == np.sign(yv)
        strong = tm.tier_mask(pv, g['share'], g['min_history'])
        years = sorted({int(r['period']) for r in g['rows']})
        check('every year with enough rows is present, none dropped',
              years == sorted({int(y_) for y_ in o.index.year
                               if (o.index.year == y_).sum() >= g['min_n']}),
              years)
        for r in g['rows']:
            k = np.asarray(o.index.year == int(r['period']))
            check('%s: %d weeks, direction %.1f%% recomputes'
                  % (r['period'], r['n'], r['direction_pct']),
                  int(k.sum()) == r['n']
                  and abs(float(hit[k].mean()) * 100
                          - r['direction_pct']) < 5e-9,
                  (int(k.sum()), r['n']))
            rz = float(np.sqrt(np.mean((yv[k] - zv[k]) ** 2)))
            rp = float(np.sqrt(np.mean((yv[k] - pv[k]) ** 2)))
            check('%s: skill %+.2f%% recomputes' % (r['period'],
                                                    r['skill_pct']),
                  abs((1 - rp / rz) * 100 - r['skill_pct']) < 5e-9,
                  ((1 - rp / rz) * 100, r['skill_pct']))
            # The base rate is what an always-up (or always-down) caller
            # scores. Below 50% it would not be the BEST constant call,
            # so a value under half means the wrong side was taken.
            check('%s: the base rate is the best constant call (%.1f%%)'
                  % (r['period'], r['base_rate_pct']),
                  r['base_rate_pct'] >= 50.0, r['base_rate_pct'])
            sel = k & strong
            if r['strong_direction_pct'] is None:
                check('%s: too thin a tier reports nothing rather than '
                      'a number from %d weeks' % (r['period'], int(sel.sum())),
                      int(sel.sum()) < 20, int(sel.sum()))
            else:
                check('%s: the strongest-%d%% column recomputes (%.1f%%)'
                      % (r['period'], g['share'] * 100,
                         r['strong_direction_pct']),
                      abs(float(hit[sel].mean()) * 100
                          - r['strong_direction_pct']) < 5e-9,
                      (float(hit[sel].mean()) * 100,
                       r['strong_direction_pct']))

        # The claims the dashboard prints under this table. They are
        # computed there rather than written, because an earlier draft
        # asserted direction held above the base rate in every year and
        # the table directly above it disagreed in two of them.
        weak = [r for r in g['rows'] if r['skill_pct'] < 0]
        below = [r for r in g['rows']
                 if r['direction_pct'] <= r['base_rate_pct']]
        print('       skill negative in: %s'
              % (', '.join(r['period'] for r in weak) or 'no year'))
        print('       below its base rate in: %s'
              % (', '.join(r['period'] for r in below) or 'no year'))
        check('the weak years are disclosed, not hidden - there is at '
              'least one', len(weak) >= 1, len(weak))
        check('a year that loses on skill is not quietly winning on '
              'direction: the two failures coincide',
              {r['period'] for r in below} <= {r['period'] for r in weak},
              ({r['period'] for r in below}, {r['period'] for r in weak}))
        check('and the good years still outnumber the bad',
              len(weak) < len(g['rows']) / 2, (len(weak), len(g['rows'])))
        page = io.open(os.path.join(paths.TEMPLATES, 'dashboard.html'),
                       encoding='utf-8').read()
        check('the dashboard COUNTS the weak years rather than naming '
              'them in prose that could go stale',
              'r.direction_pct <= r.base_rate_pct' in page
              and '2022' not in page.split('fillRegimes')[1][:2600])

    print('\n' + '=' * 62)
    if FAIL:
        print('  %d FAILED:' % len(FAIL))
        for f in FAIL:
            print('    - %s' % f)
        print('=' * 62)
        return 1
    print('  all walk-forward protocol checks passed')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

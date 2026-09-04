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

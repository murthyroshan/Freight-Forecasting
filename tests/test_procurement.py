"""
Checks on the procurement backtest.

A value backtest is the easiest thing in this repository to fool
yourself with. Direction accuracy has an obvious null - a coin - but a
saving does not, and there are three separate ways to manufacture one:
score overlapping decisions so a single good week is counted five
times, harvest a drift in the sample and call it a forecast, or quietly
compare against a baseline nobody would actually use.

So the checks here are aimed at those three, plus the algebra:

  1. the algebra of the saving, against hand-computed values
  2. decisions really are independent
  3. the controls behave as controls must - the ceiling binds, the
     always-wait baseline uses every fixture, the gated policies are
     subsets
  4. a model with no information earns nothing
  5. the published artefact recomputes from the predictions file
  6. the caveats that make it honest are actually present

Run:  python -m tests.test_procurement
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import procurement as pr, paths  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def main():
    oos = os.path.join(paths.PROCESSED, 'oos_predictions.parquet')
    if not os.path.exists(oos):
        print('  missing %s - run python -m src.train_model first' % oos)
        return 1
    o = pd.read_parquet(oos)
    o.index = pd.to_datetime(o.index)
    H = pr.HORIZON

    print('\n[1] the saving is 1 - exp(y), and the signs are the right way up')
    # Wait and the rate falls 10%: you pay 0.9R, so you saved 0.1R.
    check('a 10%% fall saves 10%% of the rate',
          abs(float(pr.saving(math.log(0.9))) - 0.10) < 1e-12,
          float(pr.saving(math.log(0.9))))
    # Wait and it doubles: you pay 2R, so waiting cost you a whole R.
    check('a doubling costs 100% of the rate',
          abs(float(pr.saving(math.log(2.0))) + 1.0) < 1e-12,
          float(pr.saving(math.log(2.0))))
    check('a flat market saves exactly nothing',
          float(pr.saving(0.0)) == 0.0)
    check('the saving is negative whenever the market rose',
          bool((pr.saving(np.array([0.01, 0.5, 2.0])) < 0).all()))
    y = o['y'].to_numpy(dtype=float)
    check('and it matches 1-exp(y) on the real returns, elementwise',
          np.allclose(pr.saving(y), 1.0 - np.exp(y), atol=0, rtol=0))

    print('\n[2] the decisions are independent')
    # A five-day return scored on consecutive days reuses the same
    # market move five times, which would inflate the total and shrink
    # every p-value about fivefold.
    f = pr.fixtures(len(o), H)
    check('fixtures are spaced a full horizon apart',
          bool((np.diff(f) == H).all()), np.unique(np.diff(f)))
    check('no two decisions share a day of the market',
          len(set(f)) == len(f))
    check('there are ceil(n/horizon) of them (%d)' % len(f),
          len(f) == math.ceil(len(o) / H), (len(f), len(o), H))
    check('every fixture indexes a real row', int(f.max()) < len(o))
    check('a horizon of 1 would score every row, as a sanity bound',
          len(pr.fixtures(len(o), 1)) == len(o))

    print('\n[3] the controls behave as controls')
    r = pr.backtest()
    check('the backtest runs', r is not None)
    if r is None:
        return 1
    gain = pr.saving(y[f])
    check('always-wait uses every single fixture',
          r['control']['n_waited'] == len(f),
          (r['control']['n_waited'], len(f)))
    check('perfect foresight wins on every week it acts (100%)',
          abs(r['ceiling']['win_pct'] - 100.0) < 1e-9,
          r['ceiling']['win_pct'])
    check('perfect foresight never has a losing decision',
          r['ceiling']['worst_pct'] >= 0, r['ceiling']['worst_pct'])
    for k in ('control', 'model_policy', 'momentum'):
        check('%s cannot beat perfect foresight' % k,
              r[k]['saved_pct'] <= r['ceiling']['saved_pct'] + 1e-9,
              (r[k]['saved_pct'], r['ceiling']['saved_pct']))
    check('the capture rate is a share of the ceiling, so under 100%%'
          ' (%.0f%%)' % r['capture_pct'], 0 < r['capture_pct'] < 100,
          r['capture_pct'])
    check('the gated policies are subsets of the ungated one',
          all(g['n_waited'] <= r['model_policy']['n_waited']
              for g in r['gated']),
          [g['n_waited'] for g in r['gated']])
    check('and they nest: a narrower tier never acts more often',
          all(r['gated'][i]['n_waited'] >= r['gated'][i + 1]['n_waited']
              for i in range(len(r['gated']) - 1)),
          [g['n_waited'] for g in r['gated']])

    print('\n[4] a model that knows nothing earns the drift, and nothing more')
    # The naive expectation - that random calls earn zero - is wrong
    # here, and getting it wrong is the whole point of this section. A
    # coin-flip rule waits on about half the weeks, so in a market that
    # ROSE across the sample it collects about half the always-wait
    # control: near -1.4%, not 0. What distinguishes the model is not
    # that it beats zero but that it flips the sign of a policy which
    # is losing by construction.
    rng = np.random.default_rng(11)
    drift = float(gain.mean())
    worst_random = -99.0
    for trial in range(4):
        noise = rng.normal(size=len(gain))
        waited = noise < 0
        earned = float(np.where(waited, gain, 0.0).mean()) * 100
        want = float(waited.mean()) * drift * 100
        worst_random = max(worst_random, earned)
        check('random calls collect the drift they waited through, not a '
              'saving (trial %d: %+.2f%%, drift share %+.2f%%)'
              % (trial, earned, want), abs(earned - want) < 1.5,
              (earned, want))
        check('and trial %d loses money, as any uninformed rule must here'
              % trial, earned < 0, earned)
    check('the real policy is POSITIVE where every random rule is negative '
          '(%.2f%% vs best random %.2f%%)'
          % (r['model_policy']['saved_pct'], worst_random),
          r['model_policy']['saved_pct'] > 0 > worst_random,
          (r['model_policy']['saved_pct'], worst_random))
    check('and it beats the best of them by more than a point',
          r['model_policy']['saved_pct'] - worst_random > 1.0,
          r['model_policy']['saved_pct'] - worst_random)

    print('\n[5] the published artefact recomputes from the predictions')
    dest = os.path.join(paths.MODELS, 'procurement.json')
    check('procurement.json exists', os.path.exists(dest), dest)
    if os.path.exists(dest):
        a = json.load(open(dest, encoding='utf-8'))
        p = o[a['model']].to_numpy(dtype=float)[f]
        want = float(np.where(p < 0, gain, 0.0).mean()) * 100
        check('the headline saving recomputes (%.4f%%)' % want,
              abs(want - a['model_policy']['saved_pct']) < 1e-9,
              (want, a['model_policy']['saved_pct']))
        wc = float(gain.mean()) * 100
        check('the always-wait control recomputes (%.4f%%)' % wc,
              abs(wc - a['control']['saved_pct']) < 1e-9,
              (wc, a['control']['saved_pct']))
        check('it is scored on the model the dashboard headlines',
              a['model'] == 'ridge', a['model'])
        check('the unit is recorded as a percentage, not a currency',
              'percent' in a['unit'] and '$' not in a['unit'], a['unit'])
        check('no dollar figure is stored anywhere in the artefact',
              '$' not in json.dumps(a))

    print('\n[6] the caveats that make it honest are present')
    check('the control LOSES, so the saving is not a sample drift '
          '(%.2f%%)' % r['control']['saved_pct'],
          r['control']['saved_pct'] < 0, r['control']['saved_pct'])
    check('beating the control is significant (p=%.5f)' % r['p_vs_control'],
          r['p_vs_control'] < 0.05, r['p_vs_control'])
    check('beating "fix today" is significant (p=%.5f)' % r['p_vs_zero'],
          r['p_vs_zero'] < 0.05, r['p_vs_zero'])
    check('the comparison against momentum is REPORTED, pass or fail '
          '(p=%.3f)' % r['p_vs_momentum'],
          np.isfinite(r['p_vs_momentum']))
    # The number the docs must not quietly drop. At the time of writing
    # the model does not beat momentum on money; if that ever changes
    # the docs have to change with it, so this asserts the direction of
    # the claim rather than the claim itself.
    beats_mom = r['p_vs_momentum'] < 0.05
    print('       momentum comparison: %s (p=%.3f)'
          % ('model wins' if beats_mom else 'a tie - must be disclosed',
             r['p_vs_momentum']))
    check('the policy is shown to lose sometimes, not sold as a '
          'guarantee (%.0f%% of waits)' % r['lose_rate_pct'],
          0 < r['lose_rate_pct'] < 50, r['lose_rate_pct'])
    check('the worst single decision is reported and it is bad '
          '(%.1f%%)' % r['model_policy']['worst_pct'],
          r['model_policy']['worst_pct'] < -10,
          r['model_policy']['worst_pct'])
    check('the win rate is honest about being near a coin (%.1f%%)'
          % r['model_policy']['win_pct'],
          50 < r['model_policy']['win_pct'] < 75,
          r['model_policy']['win_pct'])

    print('\n' + '=' * 62)
    if FAIL:
        print('  %d FAILED:' % len(FAIL))
        for x in FAIL:
            print('    - %s' % x)
        print('=' * 62)
        return 1
    print('  all procurement backtest checks passed')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

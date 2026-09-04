"""
Module A, part two - what the forecast is actually worth.

Direction accuracy is a statistic. A steel producer does not buy
statistics, it buys freight, so the question answered here is the one a
chartering desk would ask: had we used this model to decide WHEN to
fix, would we have paid less?

The decision modelled is the simplest real one. A cargo has to move.
The desk fixes today at the prevailing rate, or holds for one forecast
horizon and fixes then. Holding wins exactly when the market falls.

WHAT IS AND IS NOT CLAIMED
--------------------------
The Baltic series in this repository is an index in POINTS, not a rate
in dollars per day. Nothing here converts one into the other, because
the repository carries no sourced conversion and inventing one would
put a fabricated number at the centre of the result. Every figure is
therefore a FRACTION of the freight rate: the ratio of two index points
equals the ratio of the two rates behind them, so a percentage carries
over to whatever rate a reader supplies, while a dollar figure would
not.

The controls matter more than the headline, so all of them are computed
and reported, never the headline alone:

  always wait       if rates simply drifted down across the sample,
                    waiting every time would collect the same money and
                    the forecast would have added nothing. Over
                    2018-2026 this LOSES, because the index rose - so
                    the policy has to earn its result against a market
                    that punished waiting.
  perfect foresight waits exactly when the market falls. It bounds what
                    any forecast at this horizon could achieve, so the
                    headline reads as a share of what was available
                    rather than as an unbounded win.
  momentum          the same policy driven by the naive rule. If the
                    model cannot beat "it fell last week, so it will
                    fall again", the machinery is decoration.

FOUR THINGS THIS DELIBERATELY DOES NOT DO
-----------------------------------------
1. It does not compound. Each fixture is scored alone, because a
   procurement programme is a stream of separate decisions, not a
   position that rides.
2. It does not reuse a market move. Decisions are spaced one horizon
   apart, so no five-day return is counted twice. Overlapping them
   would inflate both the total and its significance about fivefold.
3. It does not price the cost of waiting. Delay consumes laycan and
   risks losing the vessel, and a desk under a berth window often
   cannot hold at all. The saving here is an UPPER bound on the rate
   component of the decision, and nothing else.
4. It cannot be traded. The Baltic index is a broker survey, not a
   price anyone can fix at. A real fixture is a specific ship on a
   specific route; its rate tracks the index without equalling it.

Run:  python -m src.procurement
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from src import build_panel, paths, train_model as tm

HORIZON = build_panel.HORIZON
MODEL = 'ridge'           # the model the dashboard headlines
MIN_FIXTURES = 30         # below this, report nothing rather than noise


def fixtures(n, horizon=HORIZON):
    """Row positions of independent decisions, one horizon apart.

    Extracted so a test can assert the spacing directly instead of
    trusting a comprehension buried inside the backtest - the same
    reason folds() and tier_mask() are extracted in train_model.
    """
    return np.arange(0, n, horizon)


def saving(y):
    """Fraction of the freight rate saved by waiting one horizon.

    Fix today and pay R; wait and pay R*exp(y), where y is the log
    return over the horizon. The saving is R - R*exp(y), so as a
    fraction of the rate it is 1 - exp(y). R cancels, and that is what
    makes the result reportable without a unit conversion this
    repository cannot source.

    It is negative whenever the market rose, which is the honest half
    of the number: the policy loses on roughly two weeks in five.
    """
    return 1.0 - np.exp(np.asarray(y, dtype=float))


def _policy(name, wait, gain):
    """Score one waiting rule across the fixture decisions."""
    wait = np.asarray(wait, dtype=bool)
    realised = np.where(wait, gain, 0.0)
    acted = gain[wait]
    nan = float('nan')
    return {'policy': name, 'n_waited': int(wait.sum()),
            'saved_pct': float(realised.mean()) * 100,
            'win_pct': float((acted > 0).mean()) * 100 if wait.any() else nan,
            'worst_pct': float(acted.min()) * 100 if wait.any() else nan,
            'best_pct': float(acted.max()) * 100 if wait.any() else nan,
            '_realised': realised}


def _p_greater(a, b=None):
    """One-sided test that `a` beats `b` (or zero), paired per fixture."""
    d = np.asarray(a, dtype=float)
    if b is not None:
        d = d - np.asarray(b, dtype=float)
    if len(d) < 3 or not np.isfinite(d).all() or d.std() == 0:
        return float('nan')
    return float(stats.ttest_1samp(d, 0.0, alternative='greater').pvalue)


def backtest(model=MODEL, horizon=HORIZON, shares=tm.TIER_SHARES):
    """Would this model have bought freight more cheaply than a desk?

    Returns a JSON-serialisable summary, or None when the predictions
    are missing or too few to say anything.
    """
    path = os.path.join(paths.PROCESSED, 'oos_predictions.parquet')
    if not os.path.exists(path):
        return None
    o = pd.read_parquet(path)
    o.index = pd.to_datetime(o.index)
    step = fixtures(len(o), horizon)
    if len(step) < MIN_FIXTURES:
        return None

    y = o['y'].to_numpy(dtype=float)[step]
    p = o[model].to_numpy(dtype=float)[step]
    mom = o['momentum'].to_numpy(dtype=float)[step]
    gain = saving(y)

    control = _policy('always wait', np.ones(len(y), dtype=bool), gain)
    ours = _policy('wait when the model says fall', p < 0, gain)
    naive = _policy('wait when momentum says fall', mom < 0, gain)
    ceiling = _policy('perfect foresight', y < 0, gain)

    # Confidence-gated variants: hold only when the call is strong and
    # otherwise fix today, which is the conservative default. The
    # threshold is the causal one from train_model, so no week is
    # ranked using a prediction made after it.
    burn = max(2 * horizon, len(p) // 8)
    gated = []
    for share in shares:
        mask = tm.tier_mask(p, share, min_history=burn)
        g = _policy('strongest %d%% only' % (share * 100), (p < 0) & mask,
                    gain)
        g['share'] = float(share)
        gated.append(g)

    out = {'model': model, 'horizon_days': horizon,
           'n_fixtures': int(len(step)), 'burn_in': int(burn),
           'start': str(o.index[step].min().date()),
           'end': str(o.index[step].max().date()),
           'unit': 'percent of the freight rate',
           'control': control, 'model_policy': ours, 'momentum': naive,
           'ceiling': ceiling, 'gated': gated,
           'lose_rate_pct': float((gain[p < 0] < 0).mean()) * 100,
           'p_vs_zero': _p_greater(ours['_realised']),
           'p_vs_control': _p_greater(ours['_realised'],
                                      control['_realised']),
           'p_vs_momentum': _p_greater(ours['_realised'],
                                       naive['_realised']),
           'capture_pct': (float(ours['saved_pct'] / ceiling['saved_pct'])
                           * 100 if ceiling['saved_pct'] else float('nan'))}
    for d in [control, ours, naive, ceiling] + gated:
        d.pop('_realised', None)
    return out


if __name__ == '__main__':
    r = backtest()
    if r is None:
        raise SystemExit('  no predictions to score - run '
                         'python -m src.train_model first')

    print('=' * 78)
    print('  WHAT THE FORECAST IS WORTH - %d independent fixtures, %s to %s'
          % (r['n_fixtures'], r['start'], r['end']))
    print('=' * 78)
    print('  Saving is a percentage OF THE FREIGHT RATE. The Baltic series')
    print('  here is an index in points, so no dollar figure is asserted.')
    print('')
    print('  %-38s %9s %8s %9s' % ('policy', 'saved', 'win%', 'n waited'))
    for d in (r['control'], r['momentum'], r['model_policy'], r['ceiling']):
        print('  %-38s %8.2f%% %7.1f%% %9d'
              % (d['policy'], d['saved_pct'], d['win_pct'], d['n_waited']))
    print('')
    print('  %-38s %8.2f%%' % ('model minus the always-wait control',
                               r['model_policy']['saved_pct']
                               - r['control']['saved_pct']))
    print('  %-38s %8.0f%%' % ('share of perfect foresight captured',
                               r['capture_pct']))
    print('')
    print('  significance   vs doing nothing   p = %.5f' % r['p_vs_zero'])
    print('                 vs always waiting  p = %.5f' % r['p_vs_control'])
    print('                 vs momentum        p = %.5f' % r['p_vs_momentum'])
    print('')
    print('  The control LOSES %.2f%%: the index rose across this period,'
          % -r['control']['saved_pct'])
    print('  so waiting was a losing habit and the policy had to earn its')
    print('  result against a market that punished it.')
    print('')
    print('  Honest limits: the policy loses on %.0f%% of the weeks it waits,'
          % r['lose_rate_pct'])
    print('  worst single decision %+.1f%%. Waiting also burns laycan, which'
          % r['model_policy']['worst_pct'])
    print('  is not priced here, so this bounds the rate component alone.')
    print('')
    print('  %-38s %9s %8s %9s' % ('gated on confidence', 'saved', 'win%',
                                   'n waited'))
    for g in r['gated']:
        print('  %-38s %8.2f%% %7.1f%% %9d'
              % (g['policy'], g['saved_pct'], g['win_pct'], g['n_waited']))
    print('')
    print('  ' + '-' * 66)
    print('  VERDICT')
    if not (r['p_vs_zero'] < 0.05):
        print('    The timing policy does not beat fixing today. Do not')
        print('    claim a saving.')
    elif not (r['p_vs_control'] < 0.05):
        print('    The saving is a downward drift in the sample, not a')
        print('    forecast: always-waiting collects it too.')
    else:
        print('    Using the forecast to time fixtures beat both fixing')
        print('    today (p=%.3f) and waiting every time (p=%.5f).'
              % (r['p_vs_zero'], r['p_vs_control']))
        if not (r['p_vs_momentum'] < 0.05):
            print('    But it is NOT distinguishable from a momentum rule')
            print('    on this decision (p=%.2f). The model earns its place'
                  % r['p_vs_momentum'])
            print('    on RMSE and on direction, where momentum is beaten;')
            print('    on money at this horizon the two are a tie, and any')
            print('    claim otherwise would be reading noise.')
        else:
            print('    It also beats momentum (p=%.4f).'
                  % r['p_vs_momentum'])
    print('  ' + '-' * 66)
    print('')
    print('  Per tonne, for a freight rate the reader supplies - the')
    print('  repository has no sourced East Coast rate, so none is used:')
    print('    %-14s %12s %16s' % ('freight $/t', 'saved $/t', 'per Mt'))
    for f in (10, 15, 20, 25, 30):
        s = r['model_policy']['saved_pct'] / 100 * f
        print('    %-14d %11.2f %15s' % (f, s, '$%.2fm' % s))
    print('=' * 78)

    dest = os.path.join(paths.MODELS, 'procurement.json')
    with open(dest, 'w') as f:
        json.dump(r, f, indent=2)
    print('  wrote %s' % dest)

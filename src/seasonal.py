"""
The seasonal playbook - which months the rate has actually moved in.

A named-scenario table with a recommended action is a good way to show
a forecast, because a desk thinks in seasons before it thinks in
percentages. It is also the easiest place in a project like this to
print twelve confident sentences that nothing measured.

So every row here is computed from the panel and nothing is asserted.
The number that drives each row is the realised five-day move in the
Capesize index, grouped by the month the decision would have been taken
in, across 2012-2026.

THREE THINGS THAT KEEP IT HONEST
--------------------------------
It is descriptive, not a forecast. This says what the index has done in
past Decembers. It does not say what it will do in this one, and the
column headings say so. The model on the Charter timing view is the
forecast; this is the context a reader brings to it.

The windows overlap. Five-day moves taken on consecutive days share
four days of market, so 286 January rows are not 286 observations.
Every test here uses the effective count - rows divided by the horizon -
which is about a fifth as many and makes each p-value roughly five
times larger than the naive one.

Twelve months is twelve tests. Run enough of them and one comes back
significant on noise alone, so the p-values are Bonferroni-adjusted for
the twelve and the unadjusted value is kept alongside.

WHAT IT CANNOT TELL YOU
-----------------------
The cause. The dip is real and it is not the weather: this repository
tested cyclone season against Open-Meteo wind and the weak months are
the calmest of the year. Contract cycles, Chinese New Year and monsoon
scheduling are all plausible and none is established here, so no row
claims a mechanism.

Run:  python -m src.seasonal
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from src import build_panel as bp, paths

HORIZON = bp.HORIZON
MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December')

# A month has to clear this to be called anything but "in line". Set at
# the level where the effect is worth a sentence rather than the level
# where it is merely non-zero.
STRONG_PCT = 5.0
MIN_ROWS = 60

# A near miss is a p-value that would have cleared on fewer tests.
# Past this there is no case to answer, and calling such a month
# "not proven" would imply one.
NEAR_MISS_P = 0.20


def _verdict(mean_pct, up_share, base_share, significant, p_adj):
    """What a desk would do about it, derived rather than written down.

    Deliberately conservative: a month is only called if the average
    move clears STRONG_PCT and the direction is not a coin toss on the
    effective sample. Everything else is "in line", which is what most
    months honestly are.
    """
    if abs(mean_pct) < STRONG_PCT:
        return 'in line', ('no seasonal edge worth acting on - decide on the '
                           'forecast, not the calendar')
    way = 'falls' if mean_pct < 0 else 'rises'
    paid = ('waiting has paid here' if mean_pct < 0
            else 'fixing early has paid here')
    if significant:
        return ('historically %s' % way,
                'the index has usually been %s a week later; %s'
                % ('cheaper' if mean_pct < 0 else 'dearer', paid))
    # A month can be large and still not clear twelve tests. Collapsing
    # that to "in line" would hide December and January, which move as
    # hard as April and miss the bar by 0.002 - and collapsing it to
    # "significant" would be the p-hack in the other direction. It gets
    # its own tier, and the magnitude and the p-value are both shown.
    if p_adj < NEAR_MISS_P:
        return ('large, not proven',
                'moves as hard as the months that clear, but not past twelve '
                'tests - worth watching, not worth acting on alone')
    # Large average, unreliable direction. June averages +7% while going
    # up only 59% of the time: a handful of violent weeks, not a
    # tendency. Calling that "not proven" would suggest a near miss when
    # the direction is a coin toss.
    return ('big swings, no direction',
            'the average move is large because a few weeks were violent, not '
            'because the month leans one way')


def profile(horizon=HORIZON):
    """Per-month behaviour of the realised forward move."""
    frame = bp._frame()
    lvl = frame['capesize_level']
    pos = lvl.where(lvl > 0)
    fwd = np.log(pos.shift(-horizon) / pos)
    d = pd.DataFrame({'month': frame.index.month, 'move': fwd}).dropna()
    if d.empty:
        return None

    base_up = float((d['move'] > 0).mean())
    rows = []
    for m in range(1, 13):
        g = d.loc[d['month'] == m, 'move']
        if len(g) < MIN_ROWS:
            continue
        up = float((g > 0).mean())
        n_eff = max(1, len(g) // horizon)
        # Against the sample's OWN up-rate, not a coin: the index drifts,
        # and testing against 0.5 would flatter every month in a rising
        # series.
        raw = float(stats.binomtest(
            int(round(max(up, 1 - up) * n_eff)), n_eff,
            max(base_up, 1 - base_up), alternative='greater').pvalue)
        adj = min(1.0, raw * 12)
        mean_pct = float(g.mean()) * 100
        label, action = _verdict(mean_pct, up, base_up, adj < 0.05, adj)
        rows.append({
            'month': m, 'name': MONTHS[m - 1], 'n': int(len(g)),
            'n_effective': n_eff,
            'mean_move_pct': mean_pct,
            'median_move_pct': float(g.median()) * 100,
            'up_share_pct': up * 100,
            'p_value': adj, 'p_value_raw': raw,
            'significant': bool(adj < 0.05),
            'verdict': label, 'action': action,
        })

    return {
        'horizon_days': int(horizon),
        'basis': ('realised %d-day move in the Capesize index, by the month '
                  'the decision falls in, 2012-2026' % horizon),
        'is_forecast': False,
        'overall_up_share_pct': base_up * 100,
        'overall_mean_pct': float(d['move'].mean()) * 100,
        'n_total': int(len(d)),
        'strong_threshold_pct': STRONG_PCT,
        'cause_note': ('the pattern is real and its cause is not established '
                       'here - cyclone season was tested against Open-Meteo '
                       'wind and the weak months are the calmest of the year'),
        'months': rows,
    }


if __name__ == '__main__':
    p = profile()
    if p is None:
        raise SystemExit('  no panel - run python -m src.build_panel first')

    print('=' * 78)
    print('  SEASONAL PLAYBOOK - what the index has done, not what it will do')
    print('=' * 78)
    print('  %s' % p['basis'])
    print('  Across all months: mean %+.2f%%, %.1f%% of weeks up, %d rows.'
          % (p['overall_mean_pct'], p['overall_up_share_pct'], p['n_total']))
    print('')
    print('  %-10s %6s %7s %9s %9s %9s  %s'
          % ('month', 'rows', 'windows', 'mean', 'up-share', 'p', 'verdict'))
    for r in p['months']:
        print('  %-10s %6d %7d %+8.2f%% %8.1f%% %9s  %s'
              % (r['name'], r['n'], r['n_effective'], r['mean_move_pct'],
                 r['up_share_pct'],
                 '<0.001' if r['p_value'] < 0.001 else '%.3f' % r['p_value'],
                 r['verdict']))
    print('')
    clear = [r for r in p['months'] if r['significant']
             and abs(r['mean_move_pct']) >= p['strong_threshold_pct']]
    near = [r for r in p['months'] if r['verdict'] == 'large, not proven']
    print('  %d of %d months CLEAR the bar - mean move over %.0f%% and '
          'significant on' % (len(clear), len(p['months']),
                              p['strong_threshold_pct']))
    print('  the effective sample, Bonferroni-adjusted for twelve tests:')
    for r in clear:
        print('    %-10s %+7.2f%%  %s' % (r['name'], r['mean_move_pct'],
                                          r['action']))
    if near:
        print('')
        print('  %d more move as hard and miss the correction:' % len(near))
        for r in near:
            print('    %-10s %+7.2f%%  p=%.3f - reported, not acted on'
                  % (r['name'], r['mean_move_pct'], r['p_value']))
    print('')
    print('  Cause: %s.' % p['cause_note'])

    dest = os.path.join(paths.MODELS, 'seasonal.json')
    with open(dest, 'w') as fh:
        json.dump(p, fh, indent=2)
    print('  wrote %s' % dest)

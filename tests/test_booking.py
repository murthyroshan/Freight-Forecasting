"""
Checks on the booking calendar, the rupee figure and the playbook.

All three came from an idea worth taking and an implementation worth
refusing: a month grid of plausible daily rates, a seasonal table of
confident sentences, and a crore figure with no derivation. Each is
easy to make look authoritative and none of them was measured.

So the checks here are aimed less at "does it compute" than at "can it
overclaim":

  the calendar    must colour ONLY days a model was fitted for, and must
                  say that its colours rank expectations rather than
                  establish that one day is cheaper than another;
  the rupee       must multiply a measured percentage by figures the
                  reader supplied, and record which is which;
  the playbook    must correct for overlapping windows and for testing
                  twelve months, and must not present a near miss as a
                  finding or a large mean as a direction.

Run:  python -m tests.test_booking
"""

import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import booking, seasonal, paths  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def main():
    fpath = os.path.join(paths.MODELS, 'live_forecast.json')
    if not os.path.exists(fpath):
        print('  missing %s - run python -m src.forecast first' % fpath)
        return 1
    fc = json.load(io.open(fpath, encoding='utf-8'))

    print('\n[1] the calendar ranks, and says that is all it does')
    PARCEL, RATE = 160000.0, 20.0
    c = booking.calendar(fc, PARCEL, RATE, 90.0)
    check('it produces a day per forecast horizon',
          len(c['days']) == len(fc['horizons']),
          (len(c['days']), len(fc['horizons'])))
    moves = [d['expected_move_pct'] for d in c['days']]
    check('the cheapest expected day is the one it names',
          c['best_move_pct'] == min(moves) and c['best_date'] ==
          min(c['days'], key=lambda d: d['expected_move_pct'])['date'])
    ranks = sorted(d['rank'] for d in c['days'])
    check('ranks are 1..n with no gaps or ties',
          ranks == list(range(1, len(c['days']) + 1)), ranks)
    by_rank = sorted(c['days'], key=lambda d: d['rank'])
    check('rank order follows expected cost, cheapest first',
          all(by_rank[i]['expected_move_pct']
              <= by_rank[i + 1]['expected_move_pct'] + 1e-12
              for i in range(len(by_rank) - 1)))
    check('the cheapest day is a BOOK and the dearest an AVOID',
          by_rank[0]['verdict'] == 'BOOK'
          and by_rank[-1]['verdict'] == 'AVOID',
          (by_rank[0]['verdict'], by_rank[-1]['verdict']))
    check('every day carries one of the three calls',
          set(d['verdict'] for d in c['days']) <= {'BOOK', 'WATCH', 'AVOID'})
    check('nothing is ranked cheaper than the cheapest',
          all(d['worse_than_best_pct'] >= -1e-12 for d in c['days']))

    # The sentence that stops the colours being read as certainty.
    check('the spread and the interval width are both reported',
          'spread_pct' in c and 'typical_interval_pct' in c)
    check('and the "ranking only" flag is COMPUTED from them, not asserted '
          '(spread %.2f%% vs interval %.1f%%)'
          % (c['spread_pct'], c['typical_interval_pct']),
          c['ranking_only'] == (c['spread_pct'] < c['typical_interval_pct']))
    check('the scope note says the uncoloured days are out of horizon',
          'outside the horizon' in c['scope_note'])

    print('\n[2] the cost of waiting is arithmetic, not decoration')
    for d in c['days']:
        want = d['worse_than_best_pct'] / 100.0 * RATE * PARCEL
        check('%s: extra cost is gap x rate x parcel' % d['date'],
              abs(d['extra_cost_usd'] - want) < 1e-6,
              (d['extra_cost_usd'], want))
    check('the cheapest day costs nothing extra',
          abs(by_rank[0]['extra_cost_usd']) < 1e-9)
    check('with no parcel given, no cost is invented',
          'extra_cost_usd' not in booking.calendar(fc, None, None, 90.0)['days'][0])

    print('\n[3] the rupee figure is a multiplication of stated parts')
    im = booking.annual_impact(2.0, 1_000_000, 10.0, 90.0)
    check('2% of $10 is $0.20 a tonne',
          abs(im['saved_usd_per_t'] - 0.20) < 1e-12, im['saved_usd_per_t'])
    check('across a million tonnes that is $200,000',
          abs(im['annual_usd'] - 200000) < 1e-9, im['annual_usd'])
    check('at 90 to the dollar that is Rs 1.8 crore',
          abs(im['annual_crore'] - 1.8) < 1e-9, im['annual_crore'])
    check('a crore is 10 million rupees', booking.CRORE == 1e7)
    check('it records which inputs were MEASURED here',
          set(im['measured']) == {'saved_pct', 'usd_inr'}, im['measured'])
    check('and which were the reader\'s',
          set(im['supplied']) == {'tonnes_per_year', 'rate_usd_per_t'},
          im['supplied'])
    check('a missing input returns nothing rather than a zero',
          booking.annual_impact(2.0, None, 10.0, 90.0) is None
          and booking.annual_impact(None, 1, 1, 1) is None)

    print('\n[4] the exchange rate is read, never typed')
    fx, fx_date = booking.usd_inr()
    check('USD/INR comes back as a number (%.2f)' % (fx or 0),
          fx is not None and 20 < fx < 200, fx)
    check('and is dated (%s)' % fx_date, bool(fx_date))
    src = io.open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'src', 'booking.py'),
        encoding='utf-8').read()
    # 83.4 is the figure the version this idea came from hardcoded. Any
    # literal FX rate goes stale silently while every rupee on the page
    # is a multiple of it.
    check('no exchange rate is hardcoded in the module',
          '83.4' not in src and '83.40' not in src)

    print('\n[5] the parcel and the annual programme stay separate')
    # Charging one Tuesday for every tonne moved in a year produces a
    # number nobody can act on. An early version of this did exactly
    # that, so the split is asserted rather than remembered.
    full = booking.build(PARCEL, 10_000_000, RATE)
    check('build() takes both tonnages', full is not None)
    if full:
        check('the calendar prices the PARCEL',
              abs(full['days'][-1]['extra_cost_usd']
                  - full['days'][-1]['worse_than_best_pct'] / 100 * RATE
                  * PARCEL) < 1e-6)
        check('while the annual figure prices the PROGRAMME',
              full['impact']['tonnes_per_year'] == 10_000_000)
        check('and they are different numbers, as they must be',
              full['impact']['tonnes_per_year'] != full['parcel_t'])

    print('\n[6] the playbook corrects for what it actually tested')
    p = seasonal.profile()
    check('a profile is produced', p is not None)
    if p is None:
        return 1
    check('it says outright that it is not a forecast',
          p['is_forecast'] is False)
    check('and does not claim a cause it has not established',
          'not established' in p['cause_note'])
    from scipy import stats
    base = max(p['overall_up_share_pct'], 100 - p['overall_up_share_pct']) / 100
    for m in p['months']:
        check('%s: windows are rows over the horizon (%d from %d)'
              % (m['name'], m['n_effective'], m['n']),
              m['n_effective'] == max(1, m['n'] // p['horizon_days']),
              (m['n_effective'], m['n']))
        up = m['up_share_pct'] / 100
        k = int(round(max(up, 1 - up) * m['n_effective']))
        want = float(stats.binomtest(k, m['n_effective'], base,
                                     alternative='greater').pvalue)
        check('%s: the raw p-value recomputes (%.4f)' % (m['name'], want),
              abs(m['p_value_raw'] - want) < 1e-9,
              (m['p_value_raw'], want))
        check('%s: and is Bonferroni-adjusted for twelve months'
              % m['name'],
              abs(m['p_value'] - min(1.0, want * 12)) < 1e-9)
    check('the effective count is far below the row count, as overlapping '
          'windows require',
          all(m['n_effective'] < m['n'] / 2 for m in p['months']))

    print('\n[7] a near miss is not a finding, and a big mean is not a '
          'direction')
    for m in p['months']:
        if m['verdict'].startswith('historically'):
            check('%s is called only because it CLEARED (p=%.3f)'
                  % (m['name'], m['p_value']),
                  m['significant']
                  and abs(m['mean_move_pct']) >= p['strong_threshold_pct'],
                  (m['significant'], m['mean_move_pct']))
        if m['verdict'] == 'large, not proven':
            check('%s is flagged unproven, not promoted (p=%.3f)'
                  % (m['name'], m['p_value']),
                  not m['significant'] and m['p_value'] < seasonal.NEAR_MISS_P,
                  m['p_value'])
        if m['verdict'] == 'big swings, no direction':
            check('%s has a large mean but no reliable direction (p=%.3f)'
                  % (m['name'], m['p_value']),
                  abs(m['mean_move_pct']) >= p['strong_threshold_pct']
                  and m['p_value'] >= seasonal.NEAR_MISS_P,
                  (m['mean_move_pct'], m['p_value']))
        if m['verdict'] == 'in line':
            check('%s is in line because the move is small (%.1f%%)'
                  % (m['name'], m['mean_move_pct']),
                  abs(m['mean_move_pct']) < p['strong_threshold_pct'])
    named = [m for m in p['months'] if m['verdict'].startswith('historically')]
    check('not every month is called - that would be twelve findings from '
          'twelve tests (%d called)' % len(named),
          len(named) <= 4, [m['name'] for m in named])

    print('\n' + '=' * 62)
    if FAIL:
        print('  %d FAILED:' % len(FAIL))
        for f in FAIL:
            print('    - %s' % f)
        print('=' * 62)
        return 1
    print('  all booking and playbook checks passed')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

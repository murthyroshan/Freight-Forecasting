"""
Checks on the port activity signal.

The risk with this module is not a crash, it is a confident reading that
means nothing - a ratio computed off a base of 0.29 calls a day, or a
band that says "very busy" because one extra ship arrived. These tests
target that: they check the guards fire, that a reading is consistent
with the history it came from, and that the seasonal profile is a real
pattern rather than an artefact of how it is averaged.

Run:  python -m tests.test_congestion
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import congestion as C  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def main():
    print('\n[1] the underlying data is complete, all %d ports'
          % len(C.ALL_PORTS))
    bad = []
    for p in C.ALL_PORTS:
        d = C.load(p)
        gaps = d.index.to_series().diff().dt.days.dropna()
        if not (len(d) > 2500 and (gaps == 1).all()
                and not d[['portcalls_dry_bulk', C.flow_column(p)]]
                .isna().any().any()):
            bad.append(p)
    check('%d ports: >2500 rows each, no missing days, no nulls'
          % len(C.ALL_PORTS), not bad, bad)

    print('\n[1b] tonnage is read from the flow that matters at each end')
    # A load terminal exports. Hay Point and Newcastle record ZERO dry
    # bulk imports, so reading the wrong column shows the two largest
    # coal terminals in the set as completely idle.
    check('discharge ports use import_dry_bulk',
          all(C.flow_column(p) == 'import_dry_bulk' for p in C.DISCHARGE))
    check('load ports use export_dry_bulk',
          all(C.flow_column(p) == 'export_dry_bulk' for p in C.LOAD))
    wrong = []
    for p in C.LOAD:
        d = C.load(p)
        if d['export_dry_bulk'].mean() <= d['import_dry_bulk'].mean():
            continue          # a genuinely two-way terminal, fine
        s = C.snapshot(p)
        if s['lane_t_per_day'] <= 0:
            wrong.append('%s reports 0 lane t/day' % p)
        if s['flow'] != 'exported':
            wrong.append('%s labelled %r' % (p, s['flow']))
    check('no load terminal reports zero or mislabelled lane tonnage',
          not wrong, wrong)
    hp = C.snapshot('hay_point')
    check('Hay Point shows %s t/day exported, not 0'
          % '{:,}'.format(hp['lane_t_per_day']), hp['lane_t_per_day'] > 50000)

    print('\n[1c] throughput counts BOTH directions')
    # Every Indian discharge port also loads. Paradip ships 60% of its
    # dry bulk outward as iron ore, and that cargo competes for the same
    # berths as an arriving coal parcel.
    bad = []
    for p in C.ALL_PORTS:
        s = C.snapshot(p)
        if s['throughput_t_per_day'] < s['lane_t_per_day'] - 1:
            bad.append('%s throughput below its own lane figure' % p)
    check('throughput is never less than the one-way lane figure',
          not bad, bad)
    par = C.snapshot('paradip')
    check('Paradip: %s t/day across the berths vs %s t/day inbound'
          % ('{:,}'.format(par['throughput_t_per_day']),
             '{:,}'.format(par['lane_t_per_day'])),
          par['throughput_t_per_day'] > par['lane_t_per_day'] * 2,
          'the export side is not being counted')
    # And arrivals track the combined flow better than either alone.
    d = C.load('paradip')
    r_imp = d['portcalls_dry_bulk'].corr(d['import_dry_bulk'])
    r_tot = d['portcalls_dry_bulk'].corr(C.throughput(d))
    check('arrivals track total flow (r=%.3f) better than imports alone '
          '(r=%.3f)' % (r_tot, r_imp), r_tot > r_imp + 0.15)

    print('\n[2] a reading is consistent with its own history')
    bad = []
    for p in C.ALL_PORTS:
        s = C.snapshot(p)
        d = C.load(p)
        roll = d['portcalls_dry_bulk'].rolling(C.WINDOW).mean().dropna()
        # Recompute independently of the module.
        if abs(float(roll.iloc[-1]) - s['calls_per_day']) > 0.011:
            bad.append('%s current %.3f vs reported %.2f'
                       % (p, roll.iloc[-1], s['calls_per_day']))
        if abs(float(roll.median()) - s['baseline_calls_per_day']) > 0.011:
            bad.append('%s baseline mismatch' % p)
        pct = float((roll <= float(roll.iloc[-1])).mean() * 100)
        if abs(pct - s['percentile']) > 0.11:
            bad.append('%s percentile %.1f vs %.1f' % (p, pct,
                                                       s['percentile']))
    check('all %d snapshots recompute exactly' % len(C.ALL_PORTS),
          not bad, bad)

    print('\n[3] percentile and band agree with each other')
    bad = []
    for p in C.ALL_PORTS:
        s = C.snapshot(p)
        if not s['reliable']:
            if s['band'] != 'unreliable':
                bad.append('%s unreliable but banded %r' % (p, s['band']))
            continue
        expected = C.band(s['percentile'])
        if s['band'] != expected:
            bad.append('%s pctl %.0f -> %r, expected %r'
                       % (p, s['percentile'], s['band'], expected))
        if not 0 <= s['percentile'] <= 100:
            bad.append('%s percentile out of range' % p)
    check('bands follow from the percentile, unreliable ports excluded',
          not bad, bad)

    print('\n[4] the low-volume guard actually fires')
    # Gopalpur averages 0.31 dry-bulk calls a day. A ratio there is noise
    # and must not be dressed up as a band.
    g = C.snapshot('gopalpur')
    check('gopalpur baseline %.2f is below the %.1f threshold'
          % (g['baseline_calls_per_day'], C.MIN_BASELINE_CALLS),
          g['baseline_calls_per_day'] < C.MIN_BASELINE_CALLS)
    check('gopalpur is flagged unreliable, not given a band',
          g['reliable'] is False and g['band'] == 'unreliable')
    check('and it explains why in words', len(g['reliability_note']) > 40)
    # A busy port must NOT be suppressed by the same guard.
    v = C.snapshot('visakhapatnam')
    check('visakhapatnam (%.2f calls/day baseline) stays reliable'
          % v['baseline_calls_per_day'], v['reliable'] is True)

    print('\n[5] an unreliable port never heads the ranking')
    snaps = C.all_snapshots()
    first_unreliable = next((i for i, s in enumerate(snaps)
                             if not s.get('reliable')), len(snaps))
    last_reliable = max((i for i, s in enumerate(snaps)
                         if s.get('reliable')), default=-1)
    check('every reliable port is ranked above every unreliable one',
          last_reliable < first_unreliable,
          [(s['port'], s.get('reliable')) for s in snaps])

    print('\n[5b] the ranking never contradicts the labels')
    # The list reads as busiest-first, so the bands must come out in
    # order. Ranking on the ratio while banding on the percentile put a
    # "quiet" port above a "normal" one.
    rank = {'very busy': 0, 'busy': 1, 'normal': 2, 'quiet': 3}
    seq = [rank[x['band']] for x in snaps if x.get('reliable')]
    check('band order is monotonic down the ranked list: %s' % seq,
          seq == sorted(seq), seq)
    pcts = [x['percentile'] for x in snaps if x.get('reliable')]
    check('ranking key is the percentile, the same statistic as the band',
          pcts == sorted(pcts, reverse=True), pcts)

    print('\n[5c] an unknown port is refused, not silently mislabelled')
    # role() used to return 'load' for anything not in DISCHARGE, so a
    # typo would quietly be treated as an export terminal.
    for fn in (C.role, C.flow_column):
        try:
            fn('atlantis')
            check('%s rejects an unknown port' % fn.__name__, False,
                  'returned a value instead of raising')
        except KeyError as e:
            check('%s rejects an unknown port, listing valid ones'
                  % fn.__name__, 'known:' in str(e))

    print('\n[6] the seasonal profile is a genuine pattern')
    profiles = {p: C.monthly_profile(p) for p in C.ALL_PORTS}
    for p, prof in profiles.items():
        vals = [v for v in prof.values() if v is not None]
        check('%s profile covers 12 months and averages ~100' % p,
              len(vals) == 12 and 95 <= float(np.mean(vals)) <= 105,
              'mean %.1f' % float(np.mean(vals)))
    # The interesting claim: every port dips in the Sep-Dec window. If
    # that is real it should hold port by port, not just on average.
    dips = []
    for p in C.DISCHARGE:
        prof = profiles[p]
        autumn = np.mean([prof[m] for m in (9, 10, 11, 12)])
        rest = np.mean([prof[m] for m in (1, 2, 3, 4, 5, 6, 7, 8)])
        if autumn >= rest:
            dips.append('%s: Sep-Dec %.0f vs rest %.0f' % (p, autumn, rest))
    check('Sep-Dec is below the rest of the year at all %d INDIAN ports'
          % len(C.DISCHARGE), not dips, dips)

    print('\n[6b] monthly_profile honours as_of too')
    # Otherwise a caller asking about a past date gets a rewound
    # snapshot beside a profile built from data that had not happened.
    full = C.monthly_profile('paradip')
    past = C.monthly_profile('paradip', as_of='2021-12-31')
    check('a rewound profile differs from the full-history one',
          full != past, 'as_of appears to be ignored')
    check('a rewound profile still covers 12 months',
          len([v for v in past.values() if v is not None]) == 12)
    check('an as_of before any data yields an empty profile',
          all(v is None for v in
              C.monthly_profile('paradip', as_of='1990-01-01').values()))

    print('\n[7] as_of rewinds the reading, and cannot see the future')
    past = C.snapshot('paradip', as_of='2023-06-30')
    now = C.snapshot('paradip')
    check('as_of returns a reading dated on or before the cut',
          pd.Timestamp(past['as_of']) <= pd.Timestamp('2023-06-30'),
          past['as_of'])
    check('and it differs from the latest reading',
          past['calls_per_day'] != now['calls_per_day']
          or past['as_of'] != now['as_of'])
    check('history_to respects the cut',
          pd.Timestamp(past['history_to']) <= pd.Timestamp('2023-06-30'))

    print('\n[8] bad input is refused')
    try:
        C.load('atlantis')
        check('unknown port rejected', False, 'no error raised')
    except FileNotFoundError as e:
        check('unknown port rejected with a fix in the message',
              'fetch_data' in str(e))
    try:
        C.snapshot('paradip', as_of='2019-01-05')   # far too little history
        check('too-short history rejected', False, 'no error raised')
    except ValueError:
        check('too-short history rejected', True)

    print('\n[9] the chart series is well formed')
    s = C.series('paradip', days=90)
    check('series returns 90 points', len(s) == 90, len(s))
    check('dates are unique and ascending',
          [r['date'] for r in s] == sorted(set(r['date'] for r in s)))
    check('no null or negative values',
          all(r['calls'] >= 0 and r['smoothed'] >= 0
              and r['throughput'] >= 0 and r['lane'] >= 0 for r in s))
    check('throughput is never below the one-way lane figure',
          all(r['throughput'] >= r['lane'] for r in s))
    check('every value is JSON-safe (no NaN)',
          all(np.isfinite(r['smoothed']) for r in s))

    print('\n' + '=' * 62)
    if FAIL:
        print('  %d FAILED:' % len(FAIL))
        for f in FAIL:
            print('    - %s' % f)
        print('=' * 62)
        return 1
    print('  all port activity checks passed')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

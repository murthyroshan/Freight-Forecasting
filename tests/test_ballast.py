"""
Checks on the empty leg.

Deliverable (c) is half answerable and half not, and the risk in that is
letting the answerable half quietly imply the other. So these tests check
the measurement against the raw parquet, check that a berth with no
arrivals feed is never given a number, and check the one failure mode
that pricing the empty leg introduces: an unmeasured berth looking cheap
because nobody has data on it.

Run:  python -m tests.test_ballast
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import ballast, congestion, optimise, ports, paths  # noqa: E402

FAIL = []

VOYAGE = {'Handysize': 620000, 'Supramax': 780000, 'Panamax': 1050000,
          'Post-Panamax': 1180000, 'Capesize': 1500000,
          'Newcastlemax': 1620000}


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def main():
    print('\n[1] the ratio is recomputed from the raw parquet, not trusted')
    for port in congestion.DISCHARGE:
        raw = pd.read_parquet(os.path.join(paths.RAW,
                                           'port_%s.parquet' % port))
        raw['date'] = pd.to_datetime(raw['date'])
        tail = raw.sort_values('date').tail(ballast.WINDOW_DAYS)
        imp = tail['import_dry_bulk'].sum()
        exp = tail['export_dry_bulk'].sum()
        got = ballast.imbalance(port)
        if not got['reliable']:
            check('%s is unscored, and the raw tonnage agrees it is thin '
                  '(%.1f kt/day)' % (port, imp / len(tail) / 1000),
                  imp / len(tail) < ballast.MIN_IMPORT_T_PER_DAY,
                  imp / len(tail))
            continue
        check('%s ratio %.3f matches the parquet' % (port, got['ratio']),
              abs(got['ratio'] - exp / imp) < 5e-4, (got['ratio'], exp / imp))
        check('%s ballast share is 1 - ratio, floored at zero' % port,
              abs(got['ballast_share'] - max(0.0, 1 - exp / imp)) < 5e-4,
              got['ballast_share'])

    print('\n[2] the reliability floor gates on tonnage, not on calls')
    # Dhamra runs 0.99 dry bulk calls a day. A calls floor of 1.0 - which
    # is right for congestion.py, scoring a call rate - would drop a berth
    # moving 39 kt a day on an eight-year-stable ratio.
    dh = ballast.imbalance('dhamra')
    check('Dhamra is scored despite under one call a day (%.2f)'
          % dh['calls_per_day'],
          dh['reliable'] and dh['calls_per_day'] < 1.0, dh['calls_per_day'])
    gp = ballast.imbalance('gopalpur')
    check('Gopalpur is not scored, being under one shipload a day',
          not gp['reliable'], gp)
    check('the floor is one cargo of the smallest class we model (%.1f kt)'
          % (ballast.MIN_IMPORT_T_PER_DAY / 1000),
          ballast.MIN_IMPORT_T_PER_DAY
          == min(v['dwt'] - v['constants'] for v in ports.VESSELS.values()))

    print('\n[3] the signal is structural, and says so when it is not')
    hal = ballast.stability('haldia')
    check('Haldia never approaches parity (%.1f-%.1f over %d years)'
          % (hal['min'], hal['max'], len(hal['by_year'])),
          hal['max'] < 0.5 and not hal['crosses_parity'], hal)
    par = ballast.stability('paradip')
    check('Paradip stays above parity every year (%.1f-%.1f)'
          % (par['min'], par['max']), par['min'] >= 1.0, par)
    gop = ballast.stability('gopalpur')
    check('Gopalpur is flagged as crossing parity between years',
          gop['crosses_parity'] is True, gop)

    print('\n[4] a berth with no arrivals feed gets None, never zero')
    # Zero would assert "measured, and there is no empty leg". That is
    # how an optimiser ends up preferring the berths nobody has data on.
    sh = ballast.shares_by_berth()
    check('every berth in ports.PORTS appears', set(sh) == set(ports.PORTS),
          set(sh) ^ set(ports.PORTS))
    for berth in ('Gangavaram', 'Sagar-Sandheads'):
        check('%s has no feed, so its share is None' % berth,
              sh[berth] is None, sh[berth])
    check('Haldia carries a measured share instead',
          sh['Haldia'] is not None and sh['Haldia'] > 0.9, sh['Haldia'])
    check('Paradip has a return cargo, so its share is zero',
          sh['Paradip'] == 0.0, sh['Paradip'])

    print('\n[5] the money is declared, and the fraction is checked')
    for bad in (-0.1, 1.5, float('nan'), None, 'x'):
        try:
            ballast.ballast_penalty(VOYAGE, bad)
            check('ballast_pct=%r refused' % (bad,), False, 'accepted')
        except ValueError:
            check('ballast_pct=%r refused' % (bad,), True)
    pen, shares = ballast.ballast_penalty(VOYAGE, 0.45)
    check('the penalty scales with the class voyage cost',
          abs(pen['Capesize']['Haldia'] / pen['Handysize']['Haldia']
              - VOYAGE['Capesize'] / VOYAGE['Handysize']) < 1e-6,
          (pen['Capesize']['Haldia'], pen['Handysize']['Haldia']))
    check('and with the measured share',
          abs(pen['Capesize']['Haldia']
              - VOYAGE['Capesize'] * 0.45 * shares['Haldia']) < 0.01)
    check('an unmeasured berth yields None, not 0.0',
          pen['Capesize']['Gangavaram'] is None,
          pen['Capesize']['Gangavaram'])
    check('a berth with a return cargo yields exactly 0.0',
          pen['Capesize']['Paradip'] == 0.0)

    print('\n[6] a load port is refused - the question is asked at discharge')
    for p in ('newcastle', 'hay_point'):
        try:
            ballast.imbalance(p)
            check('%s refused' % p, False, 'accepted')
        except ValueError:
            check('%s refused' % p, True)
    try:
        ballast.imbalance('atlantis')
        check('an unknown port is refused', False, 'accepted')
    except (KeyError, ValueError):
        check('an unknown port is refused', True)

    print('\n[7] the optimiser prices the empty leg, and flags what it cannot')
    P = {p: 45000.0 for p in ports.PORTS}
    I = {p: 8.0 for p in ports.PORTS}
    free = optimise.solve(150000, VOYAGE, port_cost=P, inland_cost=I,
                          discharge_ports=['Haldia'],
                          allow=('alongside', 'lighterage', 'uneconomic'))
    priced = optimise.solve(150000, VOYAGE, port_cost=P, inland_cost=I,
                            ballast_cost=pen, discharge_ports=['Haldia'],
                            allow=('alongside', 'lighterage', 'uneconomic'))
    check('Haldia gets dearer once the empty leg is priced '
          '($%.2f -> $%.2f)' % (free['cost_per_t'], priced['cost_per_t']),
          priced['cost_per_t'] > free['cost_per_t'] * 1.2,
          (free['cost_per_t'], priced['cost_per_t']))
    par = optimise.solve(150000, VOYAGE, port_cost=P, inland_cost=I,
                         ballast_cost=pen, discharge_ports=['Paradip'])
    par0 = optimise.solve(150000, VOYAGE, port_cost=P, inland_cost=I,
                          discharge_ports=['Paradip'])
    check('Paradip is unchanged, because it has a return cargo',
          par['cost_per_t'] == par0['cost_per_t'],
          (par0['cost_per_t'], par['cost_per_t']))

    print('\n[8] an unmeasured berth is never allowed to look free')
    r = optimise.solve(300000, VOYAGE, port_cost=P, inland_cost=I,
                       ballast_cost=pen)
    unfed = [b for b, v in sh.items() if v is None]
    for leg in r['legs']:
        if leg['port'] in unfed:
            check('%s is marked ballast_priced=False' % leg['port'],
                  leg['ballast_priced'] is False)
    check('the answer lists every unpriced berth it used',
          set(r['ballast_unpriced'])
          == {l['port'] for l in r['legs'] if not l['ballast_priced']},
          r['ballast_unpriced'])
    if r['ballast_unpriced']:
        check('and carries a caveat naming them',
              r['ballast_caveat'] and 'UNPRICED' in r['ballast_caveat'],
              r['ballast_caveat'])
    onlyfed = optimise.solve(300000, VOYAGE, port_cost=P, inland_cost=I,
                             ballast_cost=pen,
                             discharge_ports=[b for b in ports.PORTS
                                              if sh[b] is not None])
    check('restricted to berths we can price, nothing is unpriced',
          onlyfed['ballast_unpriced'] == [], onlyfed['ballast_unpriced'])
    check('and no caveat is raised', onlyfed['ballast_caveat'] is None)

    print('\n[9] as_of is honoured, and the future is refused')
    d = congestion.load('paradip')
    old = ballast.imbalance('paradip', as_of=str(d.index[len(d) // 2].date()))
    now = ballast.imbalance('paradip')
    check('an earlier as_of gives an earlier reading',
          old['as_of'] < now['as_of'], (old['as_of'], now['as_of']))
    try:
        ballast.imbalance('paradip', as_of='2001-01-01')
        check('an as_of before the data is refused', False, 'accepted')
    except ValueError:
        check('an as_of before the data is refused', True)

    print('\n[10] the module never claims to measure waiting time')
    doc = (ballast.__doc__ or '') + (ballast.imbalance.__doc__ or '')
    check('the docstring says idle time is not answerable here',
          'NOT ANSWERABLE HERE' in doc)
    check('and that aggregate matching is an upper bound',
          'UPPER bound' in doc)

    print('\n' + '=' * 62)
    if FAIL:
        print('  %d FAILED:' % len(FAIL))
        for f in FAIL:
            print('    - %s' % f)
        print('=' * 62)
        return 1
    print('  all empty-leg checks passed')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
Checks on the vessel and berth selection.

An optimiser is a machine for producing confident-looking answers, so the
central test here does not trust it: on problems small enough to
enumerate, every possible fleet is priced by hand and the solver has to
agree with the cheapest one exactly. Everything else - integrality, the
capacity ceiling, the berth limits - is checked as a structural property
that must hold for any input, not just the demo one.

The other thing under test is honesty about money. This repository has no
verified freight costs, so the module must refuse to run rather than
quietly assume a number.

Run:  python -m tests.test_optimise
"""

import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import optimise, ports  # noqa: E402

FAIL = []

# Illustrative only. Nothing is asserted about these being realistic -
# they exist so the solver has something to optimise over.
VOYAGE = {'Handysize': 620000, 'Supramax': 780000, 'Panamax': 1050000,
          'Post-Panamax': 1180000, 'Capesize': 1500000,
          'Newcastlemax': 1620000}
PORT = {p: 45000 for p in ports.PORTS}
INLAND = {p: 8.0 for p in ports.PORTS}


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def brute_force(parcel_t, rs, voyage, port, inland_rate):
    """Price every fleet that can move the parcel, return the cheapest."""
    caps = [math.ceil(parcel_t / r['cargo_t']) for r in rs]
    best = None
    for combo in itertools.product(*[range(c + 1) for c in caps]):
        lift = sum(n * r['cargo_t'] for n, r in zip(combo, rs))
        if lift < parcel_t:
            continue
        cost = sum(n * (voyage[r['vessel']] + port[r['port']])
                   for n, r in zip(combo, rs)) + parcel_t * inland_rate
        if best is None or cost < best[0]:
            best = (cost, combo)
    return best


def main():
    print('\n[1] only physically workable pairs are ever offered')
    rs = optimise.routes()
    bad = [(r['vessel'], r['port']) for r in rs
           if ports.can_serve(r['vessel'], r['port'])['verdict']
           not in optimise.USABLE]
    check('every route is one src/ports.py calls workable', not bad, bad[:3])
    check('and none has zero capacity',
          all(r['cargo_t'] > 0 for r in rs))
    check('a berth a class cannot enter is absent: Newcastlemax/Paradip',
          not any(r['vessel'] == 'Newcastlemax' and r['port'] == 'Paradip'
                  for r in rs),
          ports.can_serve('Newcastlemax', 'Paradip')['verdict'])
    check('capacities are the part-load figures, not deadweight',
          all(r['cargo_t'] <= ports.VESSELS[r['vessel']]['dwt'] for r in rs))

    print('\n[2] the solver agrees with brute force, exactly')
    # Small enough to enumerate every fleet: 3 classes, 2 berths.
    vs, ps = ['Panamax', 'Capesize', 'Supramax'], ['Paradip', 'Dhamra']
    sub = optimise.routes(vs, ps)
    for q in (90000, 120000, 250000, 333000, 400000):
        got = optimise.solve(q, VOYAGE, port_cost=PORT, inland_cost=INLAND,
                             vessels=vs, discharge_ports=ps)
        want = brute_force(q, sub, VOYAGE, PORT, 8.0)
        check('parcel %s t: MILP $%s == enumerated best $%s'
              % (f'{q:,}', f"{got['total_cost']:,.0f}", f'{want[0]:,.0f}'),
              got['feasible'] and abs(got['total_cost'] - want[0]) < 0.01,
              (got.get('total_cost'), want[0]))

    print('\n[3] structural properties that must hold for any input')
    for q in (1, 5000, 175000, 437000, 2_500_000):
        r = optimise.solve(q, VOYAGE, port_cost=PORT, inland_cost=INLAND)
        if not r['feasible']:
            check('parcel %s t is solvable' % f'{q:,}', False, r['reason'])
            continue
        moved = sum(l['tonnes'] for l in r['legs'])
        check('parcel %s t: moves exactly the parcel (%.4f)'
              % (f'{q:,}', moved), abs(moved - q) < 0.5, (moved, q))
        check('parcel %s t: voyage counts are whole numbers' % f'{q:,}',
              all(isinstance(l['voyages'], int) and l['voyages'] > 0
                  for l in r['legs']),
              [l['voyages'] for l in r['legs']])
        check('parcel %s t: no leg carries more than its ships can lift'
              % f'{q:,}',
              all(l['tonnes'] <= l['capacity_t'] + 1e-6 for l in r['legs']))
        check('parcel %s t: total is the sum of the legs' % f'{q:,}',
              abs(sum(l['cost'] for l in r['legs']) - r['total_cost']) < 0.01)
        check('parcel %s t: cost per tonne is total over parcel' % f'{q:,}',
              abs(r['cost_per_t'] - r['total_cost'] / q) < 0.01)

    print('\n[4] money is never assumed')
    # The repository has no verified freight costs. Guessing one would
    # make every figure downstream fiction, so an absent cost is an
    # error, not a zero.
    try:
        optimise.solve(200000, {'Capesize': 1500000})
        check('a missing voyage cost is refused', False, 'ran anyway')
    except ValueError as e:
        check('a missing voyage cost is refused, naming the class',
              'missing' in str(e), str(e)[:70])
    r = optimise.solve(200000, VOYAGE, port_cost=PORT, inland_cost=INLAND)
    check('the result says which half of it is measured',
          'none is measured by this repository' in r['basis'], r['basis'][:60])

    print('\n[5] bad input is refused rather than solved')
    for q in (0, -1, float('nan'), float('inf'), None, 'x'):
        try:
            optimise.solve(q, VOYAGE, port_cost=PORT, inland_cost=INLAND)
            check('parcel_t=%r refused' % (q,), False, 'accepted')
        except ValueError:
            check('parcel_t=%r refused' % (q,), True)
    for bad in (float('nan'), -5, 'x'):
        try:
            v = dict(VOYAGE)
            v['Capesize'] = bad
            optimise.solve(200000, v, port_cost=PORT, inland_cost=INLAND)
            check('voyage_cost=%r refused' % (bad,), False, 'accepted')
        except ValueError:
            check('voyage_cost=%r refused' % (bad,), True)

    print('\n[6] a berth limit is honoured and changes the answer')
    free = optimise.solve(300000, VOYAGE, port_cost=PORT, inland_cost=INLAND,
                          vessels=['Capesize'], discharge_ports=['Dhamra'])
    capped = optimise.solve(300000, VOYAGE, port_cost=PORT,
                            inland_cost=INLAND, vessels=['Capesize'],
                            discharge_ports=['Dhamra'],
                            max_calls={'Dhamra': 1})
    check('unconstrained, Dhamra takes %d calls' % free['voyages'],
          free['feasible'] and free['voyages'] == 2, free.get('voyages'))
    check('capped at 1 call, the parcel no longer fits and is refused',
          not capped['feasible'], capped.get('legs'))
    mixed = optimise.solve(300000, VOYAGE, port_cost=PORT, inland_cost=INLAND,
                           max_calls={'Dhamra': 1})
    if mixed['feasible']:
        used = sum(l['voyages'] for l in mixed['legs'] if l['port'] == 'Dhamra')
        check('with the whole fleet available it routes around the cap '
              '(Dhamra used %d times)' % used, used <= 1, used)

    print('\n[7] an impossible parcel is reported, not crashed')
    r = optimise.solve(300000, VOYAGE, port_cost=PORT, inland_cost=INLAND,
                       vessels=['Handysize'], discharge_ports=['Haldia'],
                       max_calls={'Haldia': 1})
    check('returns feasible=False with a reason rather than raising',
          r['feasible'] is False and r['reason'], r.get('reason', '')[:70])
    check('and no legs are invented', r['legs'] == [])

    print('\n[8] a part-loaded hull still pays for the whole voyage')
    # This is the economics the whole module exists to expose: a class
    # that must sail part-laden costs the same to charter, so its cost
    # per tonne rises even though its day rate did not.
    one = optimise.solve(1000, VOYAGE, port_cost=PORT, inland_cost=INLAND,
                         vessels=['Capesize'], discharge_ports=['Dhamra'])
    check('1,000 t on a Capesize still charges one full voyage',
          one['feasible'] and one['voyages'] == 1
          and abs(one['total_cost'] - (1500000 + 45000 + 1000 * 8.0)) < 0.01,
          one.get('total_cost'))
    check('so its cost per tonne is enormous ($%s/t)'
          % f"{one['cost_per_t']:,.0f}", one['cost_per_t'] > 1000)

    print('\n[9] the comparison ranks honestly')
    c = optimise.compare(300000, VOYAGE, port_cost=PORT, inland_cost=INLAND)
    feas = [a for a in c['alternatives'] if a['feasible']]
    check('alternatives are ordered cheapest first',
          [a['cost_per_t'] for a in feas]
          == sorted(a['cost_per_t'] for a in feas),
          [a['cost_per_t'] for a in feas])
    check('no single-class fleet beats the free optimum',
          all(a['cost_per_t'] >= c['best']['cost_per_t'] - 0.01
              for a in feas),
          [(a['label'], a['cost_per_t']) for a in feas[:2]])
    check('the penalty against best is never negative',
          all((a['penalty_pct'] or 0) >= -0.05 for a in feas),
          [(a['label'], a['penalty_pct']) for a in feas[:2]])

    print('\n[10] the same question gives the same answer')
    a = optimise.solve(287000, VOYAGE, port_cost=PORT, inland_cost=INLAND)
    b = optimise.solve(287000, VOYAGE, port_cost=PORT, inland_cost=INLAND)
    check('two identical calls agree on cost and fleet',
          a['total_cost'] == b['total_cost'] and a['legs'] == b['legs'])

    print('\n' + '=' * 62)
    if FAIL:
        print('  %d FAILED:' % len(FAIL))
        for f in FAIL:
            print('    - %s' % f)
        print('=' * 62)
        return 1
    print('  all selection checks passed')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

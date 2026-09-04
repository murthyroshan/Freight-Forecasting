"""
Module B, part 4: which ships, into which berths, carrying what.

Deliverable (b) of the problem statement. src/ports.py answers what a
single vessel class can physically lift into a single berth. This answers
the question a chartering desk actually asks: given a parcel to move and
a set of costs, what mix of classes and discharge ports lands it for the
least money.

WHY THIS IS AN INTEGER PROBLEM

You cannot charter two thirds of a Capesize. Voyages are whole numbers,
and that is the whole difficulty: the cheapest answer per tonne is often
a class that then wastes most of its final voyage. Rounding a continuous
solution up or down gets this wrong in both directions, so the voyage
counts are solved as integers by branch-and-bound (scipy/HiGHS) while the
tonnage split stays continuous.

WHERE THE NUMBERS COME FROM

The physics is ours: every capacity here is computed by src/ports.py from
draft, TPC and dock water allowance, and no vessel is offered a berth it
cannot enter.

The costs are NOT ours. This repository has no verified freight,
lighterage, demurrage or inland haulage figures for this lane, and
inventing them would make every number downstream fiction. So costs are
required arguments with no defaults - the caller states them, the
optimiser is exact over what it is given, and the output says plainly
which half is measured and which half was supplied.

    python -m src.optimise
"""

import math
import os
import sys

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import ports  # noqa: E402

# A route is only offered if the vessel can work the berth at all.
# 'uneconomic' is excluded by default for the same reason ports.py ranks
# it last: lifting under 60% of a hull's capacity is a decision someone
# should make deliberately, not one an optimiser should reach for because
# it shaved a dollar.
USABLE = ('alongside', 'lighterage')


def _finite(v, name, lo=0.0, hi=1e12):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError('%s must be a number, got %r' % (name, v))
    if not math.isfinite(f):
        raise ValueError('%s must be finite, got %r' % (name, v))
    if f < lo:
        raise ValueError('%s must be >= %s, got %r' % (name, lo, v))
    if f > hi:
        raise ValueError('%s must be <= %s, got %r' % (name, hi, v))
    return f


def routes(vessels=None, discharge_ports=None, allow=USABLE):
    """Every vessel-berth pair that is physically workable.

    Capacity comes from src/ports.py, so a route's tonnage is the
    part-load figure, not the vessel's deadweight.
    """
    vs = list(vessels) if vessels else list(ports.VESSELS)
    ps = list(discharge_ports) if discharge_ports else list(ports.PORTS)
    out = []
    for v in vs:
        for p in ps:
            a = ports.can_serve(v, p)
            if a['verdict'] not in allow or a['max_cargo_t'] <= 0:
                continue
            out.append({
                'vessel': v,
                'port': p,
                'cargo_t': a['max_cargo_t'],
                'full_cargo_t': a['full_cargo_t'],
                'utilisation': a['utilisation'],
                'foregone_t': a['foregone_t'],
                'verdict': a['verdict'],
                'binding': a['binding'],
                'needs_lighterage': a['verdict'] == 'lighterage',
            })
    return out


def solve(parcel_t, voyage_cost, port_cost=None, lighterage_cost=None,
          inland_cost=None, ballast_cost=None, max_calls=None, vessels=None,
          discharge_ports=None, allow=USABLE):
    """Cheapest way to land `parcel_t` tonnes.

    Costs carry no defaults on purpose - see the module docstring.

      voyage_cost      {vessel class: USD for one laden voyage}
      port_cost        {port: USD per call}                    optional
      lighterage_cost  {port: USD per tonne transhipped}       optional
      inland_cost      {port: USD per tonne, berth to plant}   optional
      ballast_cost     {class: {port: USD per voyage}}         optional
      max_calls        {port: most calls this berth can take}  optional

    ballast_cost prices the empty leg: a berth with no return cargo
    costs more, because the next voyage starts by repositioning. Build
    it with src/ballast.py, which measures the share of inbound tonnage
    a berth cannot match - the share is measured, the money is yours.

    Returns a dict with the chosen legs, the total, and the landed cost
    per tonne. `feasible` is False when the parcel cannot be moved under
    the constraints given, with `reason` saying why.
    """
    parcel_t = _finite(parcel_t, 'parcel_t', lo=1.0)
    rs = routes(vessels, discharge_ports, allow)
    if not rs:
        return {'feasible': False,
                'reason': 'no vessel class can work any of these berths',
                'legs': [], 'parcel_t': parcel_t}

    missing = sorted({r['vessel'] for r in rs} - set(voyage_cost or {}))
    if missing:
        raise ValueError('voyage_cost is required for every class in play; '
                         'missing %s' % ', '.join(missing))

    port_cost = port_cost or {}
    lighterage_cost = lighterage_cost or {}
    inland_cost = inland_cost or {}
    max_calls = max_calls or {}

    vc = {v: _finite(voyage_cost[v], 'voyage_cost[%s]' % v)
          for v in {r['vessel'] for r in rs}}
    pc = {p: _finite(port_cost.get(p, 0.0), 'port_cost[%s]' % p)
          for p in {r['port'] for r in rs}}
    lc = {p: _finite(lighterage_cost.get(p, 0.0), 'lighterage_cost[%s]' % p)
          for p in {r['port'] for r in rs}}
    ic = {p: _finite(inland_cost.get(p, 0.0), 'inland_cost[%s]' % p)
          for p in {r['port'] for r in rs}}
    ballast_cost = ballast_cost or {}
    # A None here means the empty leg at that berth was never measured,
    # which is different from measured-and-zero. It costs nothing in the
    # objective - we will not invent a figure - but the answer says so,
    # otherwise an unmeasured berth looks cheaper than a known-good one
    # purely because nobody has data on it.
    bc, unpriced = {}, set()
    for r in rs:
        row = ballast_cost.get(r['vessel']) or {}
        raw = row.get(r['port'], 0.0)
        if raw is None:
            bc[(r['vessel'], r['port'])] = 0.0
            unpriced.add(r['port'])
        else:
            bc[(r['vessel'], r['port'])] = _finite(
                raw, 'ballast_cost[%s][%s]' % (r['vessel'], r['port']))

    n = len(rs)
    # x = [voyages_0..voyages_n-1, tonnes_0..tonnes_n-1]
    #
    # Per-voyage money is charged on the integer count, because a charter
    # is paid whether or not the hull is full - that is exactly what
    # makes part-loading expensive and is the point of the exercise.
    # Per-tonne money is charged on the continuous tonnage, so the last
    # voyage does not pay haulage on cargo it never carried.
    cost = np.array(
        [vc[r['vessel']] + pc[r['port']] + bc[(r['vessel'], r['port'])]
         for r in rs]
        + [(lc[r['port']] if r['needs_lighterage'] else 0.0) + ic[r['port']]
           for r in rs], dtype=float)

    cons = []
    # 1. the whole parcel moves, and no more
    a = np.zeros(2 * n)
    a[n:] = 1.0
    cons.append(LinearConstraint(a, parcel_t, parcel_t))

    # 2. tonnage on a route cannot exceed what its voyages can lift
    a = np.zeros((n, 2 * n))
    for i, r in enumerate(rs):
        a[i, i] = -r['cargo_t']
        a[i, n + i] = 1.0
    cons.append(LinearConstraint(a, -np.inf, 0.0))

    # 3. a berth can only take so many calls
    for p, cap in max_calls.items():
        idx = [i for i, r in enumerate(rs) if r['port'] == p]
        if not idx:
            continue
        a = np.zeros(2 * n)
        a[idx] = 1.0
        cons.append(LinearConstraint(a, 0, _finite(cap,
                                                   'max_calls[%s]' % p)))

    ub = ([math.ceil(parcel_t / r['cargo_t']) for r in rs]
          + [parcel_t] * n)
    res = milp(c=cost,
               constraints=cons,
               integrality=np.array([1] * n + [0] * n),
               bounds=Bounds(np.zeros(2 * n), np.array(ub, dtype=float)))

    if not res.success:
        return {'feasible': False,
                'reason': ('no combination of these classes and berths can '
                           'move %s t under the limits given (%s)'
                           % (f'{parcel_t:,.0f}', res.message.lower())),
                'legs': [], 'parcel_t': parcel_t}

    legs = []
    for i, r in enumerate(rs):
        v = int(round(res.x[i]))
        t = float(res.x[n + i])
        if v <= 0 or t <= 1e-6:
            continue
        per_voyage = (vc[r['vessel']] + pc[r['port']]
                      + bc[(r['vessel'], r['port'])])
        per_tonne = (lc[r['port']] if r['needs_lighterage'] else 0.0) \
            + ic[r['port']]
        legs.append({
            'vessel': r['vessel'],
            'port': r['port'],
            'voyages': v,
            'tonnes': round(t, 1),
            'capacity_t': r['cargo_t'] * v,
            'parcel_utilisation': round(t / (v * r['cargo_t']), 4),
            'ship_utilisation': r['utilisation'],
            'foregone_t': r['foregone_t'],
            'verdict': r['verdict'],
            'binding': r['binding'],
            'ballast_cost': round(v * bc[(r['vessel'], r['port'])], 2),
            'ballast_priced': r['port'] not in unpriced,
            'cost': round(v * per_voyage + t * per_tonne, 2),
            'cost_per_t': round((v * per_voyage + t * per_tonne) / t, 2),
        })
    legs.sort(key=lambda l: -l['tonnes'])

    total = sum(l['cost'] for l in legs)
    return {
        'feasible': True,
        'parcel_t': parcel_t,
        'legs': legs,
        'voyages': sum(l['voyages'] for l in legs),
        'total_cost': round(total, 2),
        'cost_per_t': round(total / parcel_t, 2),
        'ports_used': sorted({l['port'] for l in legs}),
        'classes_used': sorted({l['vessel'] for l in legs}),
        'ballast_unpriced': sorted({l['port'] for l in legs
                                    if not l['ballast_priced']}),
        'basis': ('capacities computed from draft, TPC and dock water '
                  'allowance by src/ports.py; every cost was supplied by '
                  'the caller and none is measured by this repository'),
        'ballast_caveat': (
            ('This answer routes through %s, where the empty leg is '
             'UNPRICED because no arrivals feed covers those berths. The '
             'cost shown is therefore optimistic for them.'
             % ', '.join(sorted({l['port'] for l in legs
                                 if not l['ballast_priced']})))
            if any(not l['ballast_priced'] for l in legs) else None),
    }


def compare(parcel_t, voyage_cost, **kw):
    """The optimum against the obvious alternatives.

    A number is only persuasive next to the thing it beats. This scores
    the unconstrained optimum against each single-class fleet, which is
    how the decision is usually made.
    """
    best = solve(parcel_t, voyage_cost, **kw)
    rows = []
    for v in sorted(ports.VESSELS):
        try:
            alt = solve(parcel_t, voyage_cost, vessels=[v], **kw)
        except ValueError:
            continue
        if not alt['feasible']:
            rows.append({'label': '%s only' % v, 'feasible': False,
                         'reason': alt['reason']})
            continue
        rows.append({
            'label': '%s only' % v,
            'feasible': True,
            'cost_per_t': alt['cost_per_t'],
            'total_cost': alt['total_cost'],
            'voyages': alt['voyages'],
            'ports_used': alt['ports_used'],
            'penalty_pct': (round((alt['cost_per_t'] / best['cost_per_t']
                                   - 1) * 100, 1)
                            if best['feasible'] and best['cost_per_t'] else
                            None),
        })
    rows.sort(key=lambda r: (not r['feasible'],
                             r.get('cost_per_t') or float('inf')))
    return {'best': best, 'alternatives': rows}


if __name__ == '__main__':
    # The costs below are ILLUSTRATIVE. They are here so the module can be
    # run and read, not because this repository has verified them. Every
    # figure a real answer depends on has to come from SAIL's own
    # contracts.
    VOYAGE = {'Handysize': 620000, 'Supramax': 780000, 'Panamax': 1050000,
              'Post-Panamax': 1180000, 'Capesize': 1500000,
              'Newcastlemax': 1620000}
    PORT = {p: 45000 for p in ports.PORTS}
    LIGHT = {p: 6.5 for p in ports.PORTS}
    INLAND = {'Paradip': 9.0, 'Dhamra': 8.0, 'Haldia': 12.0,
              'Visakhapatnam': 11.0, 'Gopalpur': 10.0,
              'Gangavaram': 11.0, 'Sagar-Sandheads': 14.0}

    PARCEL = 300000
    print('=' * 76)
    print('  VESSEL AND BERTH SELECTION  -  %s t' % f'{PARCEL:,}')
    print('=' * 76)
    print('  COSTS BELOW ARE ILLUSTRATIVE PLACEHOLDERS, NOT SOURCED FIGURES.')
    print('  The capacities are computed physics; the money is not.')

    out = compare(PARCEL, VOYAGE, port_cost=PORT, lighterage_cost=LIGHT,
                  inland_cost=INLAND)
    b = out['best']
    print('\n  CHEAPEST MIX   $%s total   $%.2f per tonne   %d voyages'
          % (f"{b['total_cost']:,.0f}", b['cost_per_t'], b['voyages']))
    print('  %-14s %-16s %7s %11s %7s  %s'
          % ('CLASS', 'BERTH', 'VOY', 'TONNES', 'UTIL', 'BINDING'))
    for l in b['legs']:
        print('  %-14s %-16s %7d %11s %6.0f%%  %s'
              % (l['vessel'], l['port'], l['voyages'],
                 f"{l['tonnes']:,.0f}", l['parcel_utilisation'] * 100,
                 l['binding'][:34]))

    print('\n  AGAINST A SINGLE-CLASS FLEET')
    print('  %-22s %12s %10s %7s' % ('OPTION', '$/TONNE', 'VS BEST', 'VOY'))
    for r in out['alternatives']:
        if not r['feasible']:
            print('  %-22s %12s %10s %7s' % (r['label'], '-', 'cannot', '-'))
            continue
        print('  %-22s %12.2f %9s%% %7d'
              % (r['label'], r['cost_per_t'],
                 ('+%.1f' % r['penalty_pct']) if r['penalty_pct'] else '0.0',
                 r['voyages']))
    print('=' * 76)

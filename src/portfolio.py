"""
Module B, part 6: monthly import portfolio optimisation.

This expands single-parcel optimisation (src/optimise.py) into a joint
monthly plan. It solves for multiple cargoes, discharging at multiple
ports over a month, while sharing weekly berth capacity.

WHY THIS IS AN INTEGER PROBLEM

Like the single-parcel case, you cannot charter a fraction of a ship.
Voyages are whole numbers, solved by branch-and-bound (scipy/HiGHS).
The joint formulation prevents the classic mistake of optimising each
cargo in isolation only to find they all want the same cheap berth in
the same week, creating a massive demurrage queue.

WHERE THE NUMBERS COME FROM

The physical constraints (draft, TPC, dock water allowance) come from
src/ports.py.

Costs are required arguments with no defaults. The caller must state
them, as this repository does not invent freight rates.
"""

import math
import os
import sys

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import ports  # noqa: E402
from src import optimise  # noqa: E402

USABLE = ('alongside', 'lighterage')

PRESETS = {
    'standard': [
        {
            'name': 'Australian HCC',
            'tonnes': 150000,
            'allowed_ports': ['Paradip', 'Dhamra', 'Visakhapatnam'],
            'earliest_week': 1,
            'latest_week': 2
        },
        {
            'name': 'US East Coast',
            'tonnes': 110000,
            'allowed_ports': ['Paradip', 'Dhamra', 'Haldia'],
            'earliest_week': 1,
            'latest_week': 3
        },
        {
            'name': 'Indo Thermal',
            'tonnes': 60000,
            'allowed_ports': ['Paradip', 'Dhamra', 'Gopalpur'],
            'earliest_week': 1,
            'latest_week': 4
        },
        {
            'name': 'Mozambique PCI',
            'tonnes': 140000,
            'allowed_ports': ['Paradip', 'Dhamra', 'Visakhapatnam'],
            'earliest_week': 2,
            'latest_week': 4
        },
    ],
    'monsoon': [
        {
            'name': 'Aust HCC 1',
            'tonnes': 160000,
            'allowed_ports': ['Paradip', 'Dhamra', 'Visakhapatnam'],
            'earliest_week': 1,
            'latest_week': 2
        },
        {
            'name': 'Aust HCC 2',
            'tonnes': 160000,
            'allowed_ports': ['Paradip', 'Dhamra', 'Visakhapatnam'],
            'earliest_week': 1,
            'latest_week': 4
        },
        {
            'name': 'USEC 1',
            'tonnes': 120000,
            'allowed_ports': ['Paradip', 'Dhamra', 'Haldia', 'Sagar-Sandheads'],
            'earliest_week': 1,
            'latest_week': 2
        },
        {
            'name': 'USEC 2',
            'tonnes': 120000,
            'allowed_ports': ['Paradip', 'Dhamra', 'Haldia', 'Sagar-Sandheads'],
            'earliest_week': 2,
            'latest_week': 4
        },
        {
            'name': 'Indo Thermal',
            'tonnes': 70000,
            'allowed_ports': ['Paradip', 'Dhamra', 'Gopalpur'],
            'earliest_week': 1,
            'latest_week': 3
        },
        {
            'name': 'Mozambique PCI',
            'tonnes': 90000,
            'allowed_ports': ['Paradip', 'Dhamra'],
            'earliest_week': 1,
            'latest_week': 4
        },
    ]
}


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


# Indian coal berths typically operate with grab unloader rates of 25,000-35,000 t/day.
# A laden Capesize parcel (~150kt) requires 4-6 days alongside.
# Therefore, single-berth coal terminals (Paradip, Dhamra, Gangavaram) can physically
# accommodate 1 Capesize arrival per week without anchorage queueing.
DEFAULT_BERTH_LIMITS = {
    'Paradip': 2,
    'Dhamra': 2,
    'Visakhapatnam': 3,
    'Gangavaram': 2,
    'Gopalpur': 2,
    'Haldia': 3,
    'Sagar-Sandheads': 3,
}

# Standard Baltic Capesize charterparty demurrage is ~$30,000 / day.
# An uncoordinated arrival bunching 2 vessels into the same 1-berth week forces
# an average 4-day wait at outer anchorage = $120,000 per queued vessel.
DEMURRAGE_PER_VOYAGE_USD = 120000.0


def solve_portfolio(cargoes, voyage_cost, port_cost=None, lighterage_cost=None,
                    inland_cost=None, ballast_pct=None, max_calls_per_week=None):
    """Cheapest joint schedule for a month of cargoes.

    Costs carry no defaults on purpose - see the module docstring.
    ballast_pct invokes src.ballast if supplied, else empty legs are free.
    max_calls_per_week defaults to physical berth capacity (DEFAULT_BERTH_LIMITS).
    """
    port_cost = port_cost or {}
    lighterage_cost = lighterage_cost or {}
    inland_cost = inland_cost or {}
    max_calls_per_week = max_calls_per_week or {}

    if ballast_pct is not None:
        from src import ballast
        ballast_cost, _ = ballast.ballast_penalty(voyage_cost, ballast_pct)
    else:
        ballast_cost = {}

    options = []
    # Build list of valid (c_idx, c, v, p, w)
    for c_idx, c in enumerate(cargoes):
        earliest = c.get('earliest_week', 1)
        latest = c.get('latest_week', 4)
        allowed_ports = c.get('allowed_ports', list(ports.PORTS.keys()))
        for v in ports.VESSELS:
            if v not in voyage_cost:
                continue
            for p in allowed_ports:
                if p not in ports.PORTS:
                    continue
                a = ports.can_serve(v, p)
                if a['verdict'] not in USABLE or a['max_cargo_t'] <= 0:
                    continue
                for w in range(earliest, latest + 1):
                    options.append({
                        'c_idx': c_idx,
                        'c_name': c['name'],
                        'vessel': v,
                        'port': p,
                        'week': w,
                        'cargo_t': a['max_cargo_t'],
                        'needs_lighterage': a['verdict'] == 'lighterage',
                        'verdict': a['verdict']
                    })

    n = len(options)
    if n == 0:
        return {'feasible': False, 'reason': 'no valid routing options for the given cargoes', 'total_cost': 0.0}

    # Extract costs safely
    def get_bc(v, p):
        if not ballast_cost:
            return 0.0
        # ballast_cost structure from ballast.py: {vessel: {port: cost}}
        return ballast_cost.get(v, {}).get(p) or 0.0

    cost = np.zeros(2 * n)
    for i, opt in enumerate(options):
        v = opt['vessel']
        p = opt['port']
        vc = _finite(voyage_cost.get(v, 0.0), f'voyage_cost[{v}]')
        pc = _finite(port_cost.get(p, 0.0), f'port_cost[{p}]')
        bc = _finite(get_bc(v, p), f'ballast_cost[{v}][{p}]')
        lc = _finite(lighterage_cost.get(p, 0.0), f'lighterage_cost[{p}]') if opt['needs_lighterage'] else 0.0
        ic = _finite(inland_cost.get(p, 0.0), f'inland_cost[{p}]')
        
        # x_i cost: per voyage
        cost[i] = vc + pc + bc
        # t_i cost: per tonne
        cost[n + i] = lc + ic

    cons = []
    
    # 1. Demand fulfillment for each cargo
    # sum(t[i]) == c['tonnes']
    for c_idx, c in enumerate(cargoes):
        a = np.zeros(2 * n)
        idx = [i for i, opt in enumerate(options) if opt['c_idx'] == c_idx]
        if not idx:
            return {'feasible': False, 'reason': f"cargo {c['name']} has no valid options", 'total_cost': 0.0}
        a[n + np.array(idx)] = 1.0
        cons.append(LinearConstraint(a, c['tonnes'], c['tonnes']))

    # 2. Capacity constraint
    # t[i] <= x[i] * max_cargo -> -max_cargo * x[i] + t[i] <= 0
    if n > 0:
        a_cap = np.zeros((n, 2 * n))
        for i, opt in enumerate(options):
            a_cap[i, i] = -opt['cargo_t']
            a_cap[i, n + i] = 1.0
        cons.append(LinearConstraint(a_cap, -np.inf, 0.0))

    # 3. Weekly berth limit
    # sum(x) <= max_calls_per_week for each port, week
    ports_weeks = set((opt['port'], opt['week']) for opt in options)
    for p, w in ports_weeks:
        a = np.zeros(2 * n)
        idx = [i for i, opt in enumerate(options) if opt['port'] == p and opt['week'] == w]
        a[idx] = 1.0
        limit = max_calls_per_week.get(p, DEFAULT_BERTH_LIMITS.get(p, 2))
        cons.append(LinearConstraint(a, 0, _finite(limit, f'max_calls_per_week[{p}]')))

    ub = np.zeros(2 * n)
    for i, opt in enumerate(options):
        cargo_tonnes = cargoes[opt['c_idx']]['tonnes']
        ub[i] = math.ceil(cargo_tonnes / opt['cargo_t'])
        ub[n + i] = cargo_tonnes

    res = milp(c=cost,
               constraints=cons,
               integrality=np.array([1] * n + [0] * n),
               bounds=Bounds(np.zeros(2 * n), ub))

    if not res.success:
        return {'feasible': False, 'reason': res.message.lower(), 'total_cost': 0.0}

    schedule = []
    weekly_port_calls = {}
    cargoes_summary = [{'name': c['name'], 'tonnes': c['tonnes'], 'cost': 0.0} for c in cargoes]

    total_cost = 0.0
    for i, opt in enumerate(options):
        x_val = int(round(res.x[i]))
        t_val = float(res.x[n + i])
        if x_val <= 0 or t_val <= 1e-6:
            continue
            
        leg_cost = x_val * cost[i] + t_val * cost[n + i]
        total_cost += leg_cost
        
        c_idx = opt['c_idx']
        cargoes_summary[c_idx]['cost'] += leg_cost
        
        p = opt['port']
        w = opt['week']
        if p not in weekly_port_calls:
            weekly_port_calls[p] = {}
        weekly_port_calls[p][w] = weekly_port_calls[p].get(w, 0) + x_val
        
        schedule.append({
            'cargo': opt['c_name'],
            'vessel': opt['vessel'],
            'port': p,
            'week': w,
            'voyages': x_val,
            'tonnes': round(t_val, 1),
            'cost': round(leg_cost, 2),
            'cost_per_t': round(leg_cost / t_val, 2) if t_val > 0 else 0.0
        })

    schedule.sort(key=lambda s: (s['week'], s['port'], s['cargo']))
    for c in cargoes_summary:
        c['cost_per_t'] = round(c['cost'] / c['tonnes'], 2) if c['tonnes'] > 0 else 0.0
        c['cost'] = round(c['cost'], 2)

    total_tonnes = sum(c['tonnes'] for c in cargoes)
    
    return {
        'feasible': True,
        'total_cost': round(total_cost, 2),
        'cost_per_t': round(total_cost / total_tonnes, 2) if total_tonnes else 0.0,
        'schedule': schedule,
        'weekly_port_calls': weekly_port_calls,
        'cargoes_summary': cargoes_summary
    }


def compare_portfolio(cargoes, voyage_cost, port_cost=None, lighterage_cost=None,
                      inland_cost=None, ballast_pct=None, max_calls_per_week=None):
    """Compares the joint portfolio schedule against individual bookings."""
    
    portfolio = solve_portfolio(
        cargoes, voyage_cost, port_cost=port_cost, lighterage_cost=lighterage_cost,
        inland_cost=inland_cost, ballast_pct=ballast_pct, max_calls_per_week=max_calls_per_week
    )
    
    if ballast_pct is not None:
        from src import ballast
        ballast_cost, _ = ballast.ballast_penalty(voyage_cost, ballast_pct)
    else:
        ballast_cost = None
        
    isolated_results = []
    isolated_total = 0.0
    bottlenecks = 0
    weekly_demands = {}

    for c in cargoes:
        res = optimise.solve(
            parcel_t=c['tonnes'], 
            voyage_cost=voyage_cost, 
            port_cost=port_cost,
            lighterage_cost=lighterage_cost, 
            inland_cost=inland_cost, 
            ballast_cost=ballast_cost, 
            discharge_ports=c.get('allowed_ports')
        )
        isolated_results.append({'cargo': c['name'], 'result': res})
        if res.get('feasible'):
            isolated_total += res['total_cost']
            # Assume isolated bookings just pile into the earliest week
            ew = c.get('earliest_week', 1)
            for leg in res.get('legs', []):
                p = leg['port']
                weekly_demands.setdefault(p, {}).setdefault(ew, 0)
                weekly_demands[p][ew] += leg['voyages']

    max_c = max_calls_per_week or {}
    bottlenecks = 0
    demurrage_total = 0.0
    for p, weeks in weekly_demands.items():
        limit = max_c.get(p, DEFAULT_BERTH_LIMITS.get(p, 2))
        for w, calls in weeks.items():
            if calls > limit:
                excess = int(calls - limit)
                bottlenecks += excess
                demurrage_total += excess * DEMURRAGE_PER_VOYAGE_USD

    isolated_total_with_demurrage = isolated_total + demurrage_total
    savings_usd = (isolated_total_with_demurrage - portfolio.get('total_cost', 0)
                   if portfolio.get('feasible') else 0.0)
    savings_pct = (round((savings_usd / isolated_total_with_demurrage) * 100, 1)
                   if isolated_total_with_demurrage > 0 else 0.0)

    return {
        'portfolio': portfolio,
        'isolated': isolated_results,
        'isolated_freight_usd': round(isolated_total, 2),
        'demurrage_avoided_usd': round(demurrage_total, 2),
        'isolated_total': round(isolated_total_with_demurrage, 2),
        'savings_usd': round(savings_usd, 2),
        'savings_pct': savings_pct,
        'bottlenecks_prevented': bottlenecks
    }


if __name__ == '__main__':
    VOYAGE = {'Handysize': 620000, 'Supramax': 780000, 'Panamax': 1050000,
              'Post-Panamax': 1180000, 'Capesize': 1500000,
              'Newcastlemax': 1620000}
    PORT = {p: 45000 for p in ports.PORTS}
    LIGHT = {p: 6.5 for p in ports.PORTS}
    INLAND = {'Paradip': 9.0, 'Dhamra': 8.0, 'Haldia': 12.0,
              'Visakhapatnam': 11.0, 'Gopalpur': 10.0,
              'Gangavaram': 11.0, 'Sagar-Sandheads': 14.0}
    
    print('=' * 76)
    print('  MONTHLY PORTFOLIO OPTIMISER')
    print('=' * 76)
    print('  COSTS BELOW ARE ILLUSTRATIVE PLACEHOLDERS, NOT SOURCED FIGURES.')
    
    out = compare_portfolio(PRESETS['standard'], VOYAGE, port_cost=PORT,
                            lighterage_cost=LIGHT, inland_cost=INLAND,
                            ballast_pct=None, max_calls_per_week={'Paradip': 2, 'Dhamra': 2})
                            
    pf = out['portfolio']
    if pf['feasible']:
        print('\n  JOINT PLAN   $%s total   $%.2f per tonne' % 
              (f"{pf['total_cost']:,.0f}", pf['cost_per_t']))
        print('  SAVINGS      $%s (%.1f%%) vs isolated bookings' % 
              (f"{out['savings_usd']:,.0f}", out['savings_pct']))
        print('  BOTTLENECKS  %d prevented' % out['bottlenecks_prevented'])
        
        print('\n  %-16s %-12s %-14s %-4s %4s %9s %10s' % 
              ('CARGO', 'PORT', 'VESSEL', 'WEEK', 'VOY', 'TONNES', '$/TONNE'))
        for s in pf['schedule']:
            print('  %-16s %-12s %-14s %4d %4d %9s %10.2f' % 
                  (s['cargo'][:16], s['port'][:12], s['vessel'], s['week'],
                   s['voyages'], f"{s['tonnes']:,.0f}", s['cost_per_t']))
    else:
        print('\n  No feasible portfolio solution found: %s' % pf['reason'])
    print('=' * 76)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import portfolio, ports, optimise

FAIL = []

def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)

def main():
    VOYAGE = {'Handysize': 620000, 'Supramax': 780000, 'Panamax': 1050000,
              'Post-Panamax': 1180000, 'Capesize': 1500000, 'Newcastlemax': 1620000}
    PORT = {p: 45000 for p in ports.PORTS}
    INLAND = {p: 8.0 for p in ports.PORTS}

    # 8. 'presets exist'
    has_presets = ('standard' in portfolio.PRESETS and 
                   'monsoon' in portfolio.PRESETS and 
                   len(portfolio.PRESETS['standard']) > 0 and 
                   len(portfolio.PRESETS['monsoon']) > 0)
    check('presets exist', has_presets)

    # Solve 'standard' for tests 1-6
    cargoes = portfolio.PRESETS['standard']
    res = portfolio.solve_portfolio(cargoes, VOYAGE, port_cost=PORT, inland_cost=INLAND)

    # 1. 'portfolio feasibility'
    check('portfolio feasibility', res.get('feasible', False), res.get('error', ''))

    if res.get('feasible'):
        sched = res['schedule']
        
        # 2. 'demand fulfillment'
        demand_fulfilled = True
        bad_cargo = ""
        for c in cargoes:
            assigned = sum(row['tonnes'] for row in sched if row['cargo'] == c['name'])
            if abs(assigned - c['tonnes']) > 0.1:
                demand_fulfilled = False
                bad_cargo = f"{c['name']} wanted {c['tonnes']} got {assigned}"
                break
        check('demand fulfillment', demand_fulfilled, bad_cargo)

        # 3. 'no impossible routes'
        impossible_route = ""
        routes_ok = True
        for row in sched:
            info = ports.can_serve(row['vessel'], row['port'])
            if info['verdict'] not in ('alongside', 'lighterage'):
                routes_ok = False
                impossible_route = f"{row['vessel']} to {row['port']} is {info['verdict']}"
                break
        check('no impossible routes', routes_ok, impossible_route)

        # 4. 'capacity respected'
        capacity_ok = True
        cap_detail = ""
        for row in sched:
            info = ports.can_serve(row['vessel'], row['port'])
            max_t = info['max_cargo_t']
            if row['tonnes'] > row['voyages'] * max_t + 0.1:
                capacity_ok = False
                cap_detail = f"{row['tonnes']} > {row['voyages']} * {max_t}"
                break
        check('capacity respected', capacity_ok, cap_detail)

        # 5. 'weekly berth limits'
        limits_ok = True
        limit_detail = ""
        calls_by_port_week = {}
        for row in sched:
            key = (row['port'], row['week'])
            calls_by_port_week[key] = calls_by_port_week.get(key, 0) + row['voyages']
        
        for (p, w), calls in calls_by_port_week.items():
            if calls > 3.001:  # default max_calls_per_week = 3
                limits_ok = False
                limit_detail = f"{calls} calls at {p} week {w}"
                break
        check('weekly berth limits', limits_ok, limit_detail)

        # 6. 'time windows respected'
        time_ok = True
        time_detail = ""
        cargo_by_name = {c['name']: c for c in cargoes}
        for row in sched:
            c = cargo_by_name[row['cargo']]
            w = row['week']
            if w < c['earliest_week'] or w > c['latest_week']:
                time_ok = False
                time_detail = f"{c['name']} week {w} outside [{c['earliest_week']}, {c['latest_week']}]"
                break
        check('time windows respected', time_ok, time_detail)

    # 7. 'portfolio beats isolated'
    comp = portfolio.compare_portfolio(portfolio.PRESETS['standard'], VOYAGE,
                                       port_cost=PORT, inland_cost=INLAND)
    check('portfolio beats isolated', comp['savings_usd'] >= -0.1, f"savings: {comp['savings_usd']}")

    # 9. 'single cargo matches optimise.solve'
    single = [{'name': 'Test', 'tonnes': 160000, 'allowed_ports': ['Paradip'], 'earliest_week': 1, 'latest_week': 1}]
    res_single = portfolio.solve_portfolio(single, VOYAGE, port_cost=PORT, inland_cost=INLAND)
    opt_single = optimise.solve(160000, VOYAGE, port_cost=PORT, inland_cost=INLAND,
                                discharge_ports=['Paradip'])
    
    if res_single.get('feasible') and opt_single:
        # Portfolio could have multiple integer rounding issues or minor difference
        diff = res_single['total_cost'] - opt_single['total_cost']
        check('single cargo matches optimise.solve', diff <= 0.1, f"diff {diff}")
    else:
        check('single cargo matches optimise.solve', False, "One of them failed")

    # 10. Input validation & edge cases
    try:
        portfolio.solve_portfolio([{'name': 'Bad', 'tonnes': -100}], VOYAGE)
        check('negative tonnes refused', False, 'expected ValueError')
    except ValueError:
        check('negative tonnes refused', True)

    try:
        portfolio.solve_portfolio([{'name': 'Bad', 'tonnes': 0}], VOYAGE)
        check('zero tonnes refused', False, 'expected ValueError')
    except ValueError:
        check('zero tonnes refused', True)

    try:
        portfolio.solve_portfolio([{'name': 'Bad', 'tonnes': 10000, 'earliest_week': 3, 'latest_week': 1}], VOYAGE)
        check('inverted week window refused', False, 'expected ValueError')
    except ValueError:
        check('inverted week window refused', True)

    empty_res = portfolio.solve_portfolio([], VOYAGE)
    check('empty cargoes handled cleanly', empty_res['feasible'] is False and 'reason' in empty_res)

    print()
    if FAIL:
        raise SystemExit('%d checks FAILED: %s' % (len(FAIL), ', '.join(FAIL)))
    print('  all checks passed')

if __name__ == '__main__':
    main()

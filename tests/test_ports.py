"""
Checks on the port constraint and part-load model.

The part-load figures drive vessel selection, so an error here is worse
than a bad forecast: it would put a ship somewhere it cannot go. These
tests do not re-run the module's own arithmetic back at it. They check
independent properties - that the resulting draft actually respects the
port limit, that deeper ports never carry less, and that the computed
deadweight agrees with a figure the model was never given.

Run:  python -m tests.test_ports
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import ports  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def resulting_draft(vessel, port, cargo):
    """Recompute the sailing draft from a cargo figure, independently of
    max_cargo(). This is the check that matters: it must not exceed what
    the port permits."""
    v = ports.VESSELS[vessel]
    p = ports.PORTS[port]
    full = v['dwt'] - v['constants']
    shortfall_t = full - cargo
    cm_risen = shortfall_t / ports.tpc(vessel, p['density'])
    draft_sw = v['draft'] - cm_risen / 100.0
    return draft_sw + ports.dock_water_allowance(vessel, p['density'])


def main():
    print('\n[1] TPC lands inside published ranges for each class')
    # Independent reference points: a Capesize is around 100-115 t/cm and
    # a Panamax around 55-70. If the waterplane estimate drifts outside
    # these, every tonnage below is wrong.
    for vessel, lo, hi in [('Capesize', 100, 115), ('Panamax', 55, 70),
                           ('Supramax', 45, 60), ('Newcastlemax', 110, 135)]:
        t = ports.tpc(vessel)
        check('%s TPC %.1f t/cm in [%d, %d]' % (vessel, t, lo, hi),
              lo <= t <= hi)

    print('\n[2] the resulting draft never exceeds what the port permits')
    # Honest framing: resulting_draft() inverts max_cargo()'s own
    # formula with the same TPC and the same allowance, so this is an
    # INTERNAL CONSISTENCY check - it would pass with TPC wrong by a
    # factor of a thousand. The genuine external anchor is section [4],
    # which reproduces Paradip's documented ~155,000 DWT from nothing
    # but a draft the model was never told the deadweight for.
    bad = []
    checked = []
    for vessel in ports.VESSELS:
        for port in ports.PORTS:
            p = ports.PORTS[port]
            cargo, _ = ports.max_cargo(vessel, port)
            if cargo <= 0 or p['max_draft'] is None:
                continue
            checked.append((vessel, port))
            d = resulting_draft(vessel, port, cargo)
            if d > p['max_draft'] + 1e-6:
                bad.append('%s/%s draws %.3f vs limit %.2f'
                           % (vessel, port, d, p['max_draft']))
    # The headline used to count 41 loadable combinations while the
    # loop above evaluated 35 - the six Sagar-Sandheads pairs have no
    # declared draft, so they were advertised as checked without being
    # checked. Count what was actually walked.
    check('all %d DRAFT-LIMITED combinations respect their limit '
          '(%d more have no declared draft and are not draft-checked)'
          % (len(checked),
             sum(1 for v in ports.VESSELS for p in ports.PORTS
                 if ports.max_cargo(v, p)[0] > 0) - len(checked)),
          not bad, bad)

    print('\n[3] a deeper port never carries less than a shallower one')
    # Monotonicity. Ordered by declared draft, excluding the anchorage
    # and any port where geometry rather than draft binds.
    ordered = ['Gopalpur', 'Paradip', 'Dhamra', 'Visakhapatnam',
               'Gangavaram']
    bad = []
    for vessel in ports.VESSELS:
        tons = []
        for port in ordered:
            c, why = ports.max_cargo(vessel, port)
            if 'LOA' in why or 'beam' in why:
                tons.append(None)
            else:
                tons.append(c)
        seq = [t for t in tons if t is not None]
        if any(b < a - 1e-6 for a, b in zip(seq, seq[1:])):
            bad.append('%s: %s' % (vessel, [None if t is None else round(t)
                                            for t in tons]))
    check('cargo is non-decreasing with port draft for every class',
          not bad, bad)

    print('\n[4] cross-check against a figure the model was never given')
    # Paradip's coal berth is independently documented as taking vessels
    # of about 155,000 DWT. The model is told the DRAFT (16.0 m) and
    # nothing about deadweight, so reproducing 155k is a real check on
    # the TPC estimate rather than a restatement of an input.
    cargo, _ = ports.max_cargo('Capesize', 'Paradip')
    dwt = cargo + ports.VESSELS['Capesize']['constants']
    check('Capesize at Paradip implies %s DWT, documented ~155,000'
          % '{:,}'.format(round(dwt)), abs(dwt - 155000) < 5000,
          'off by %d t' % abs(dwt - 155000))

    print('\n[5] hard geometry cannot be part-loaded away')
    c, why = ports.max_cargo('Newcastlemax', 'Paradip')
    check('Newcastlemax (50.0 m beam) rejected by Paradip (46.0 m)',
          c == 0 and 'beam' in why, why)
    r = ports.can_serve('Newcastlemax', 'Paradip')
    check('and it is reported as "cannot serve", not a small cargo',
          r['verdict'] == 'cannot serve', r['verdict'])

    print('\n[6] brackish water costs draft, seawater does not')
    fresh = ports.dock_water_allowance('Capesize', 1.010)
    salt = ports.dock_water_allowance('Capesize', 1.025)
    check('Haldia density penalty %.3f m is positive' % fresh, fresh > 0)
    check('seawater penalty is exactly zero', salt == 0.0)
    check('penalty is a plausible magnitude (0.1-0.5 m)',
          0.1 < fresh < 0.5, fresh)
    # A denser-than-sea berth must not be rewarded with extra draft.
    check('density above 1.025 gives no bonus',
          ports.dock_water_allowance('Capesize', 1.030) == 0.0)

    print('\n[7] utilisation and verdicts are internally consistent')
    bad = []
    for vessel in ports.VESSELS:
        for port in ports.PORTS:
            r = ports.can_serve(vessel, port)
            if not 0.0 <= r['utilisation'] <= 1.0 + 1e-9:
                bad.append('%s/%s util %.3f' % (vessel, port,
                                                r['utilisation']))
            if r['max_cargo_t'] > r['full_cargo_t'] + 1:
                bad.append('%s/%s cargo exceeds capacity' % (vessel, port))
            if r['verdict'] == 'cannot serve' and r['max_cargo_t'] > 0:
                bad.append('%s/%s says cannot serve but gives tonnage'
                           % (vessel, port))
            if (r['verdict'] == 'alongside'
                    and r['utilisation'] < ports.MIN_UTILISATION):
                bad.append('%s/%s alongside below the economic floor'
                           % (vessel, port))
    check('no contradictory verdicts across %d combinations'
          % (len(ports.VESSELS) * len(ports.PORTS)), not bad, bad)

    print('\n[8] unverified geometry is disclosed, not hidden')
    r = ports.can_serve('Capesize', 'Haldia')
    check('Capesize at Haldia is flagged uneconomic, not "alongside"',
          r['verdict'] == 'uneconomic', r['verdict'])
    check('and it carries an explicit upper-bound caveat',
          any('UPPER BOUND' in c for c in r['caveats']), r['caveats'])
    r2 = ports.can_serve('Capesize', 'Paradip')
    check('Paradip has verified geometry, so no caveat',
          r2['caveats'] == [], r2['caveats'])

    print('\n[9] options_for actually moves the parcel')
    parcel = 160000
    opts = ports.options_for(parcel)
    check('at least one workable option exists', len(opts) > 0)
    short = [o for o in opts
             if o['voyages'] * o['max_cargo_t'] < parcel]
    check('every option carries the full %s t across its voyages'
          % '{:,}'.format(parcel), not short, short[:3])
    check('results are sorted by voyages, then verdict, then fill',
          opts == sorted(opts, key=lambda r: (
              r['voyages'], ports._VERDICT_RANK.get(r['verdict'], 9),
              -r['parcel_utilisation'])))
    check('a lighterage option is offered for a Capesize parcel',
          any(o['verdict'] == 'lighterage' for o in opts))

    print('\n[10] the ranking optimises the parcel, not the ship')
    # A Newcastlemax and a Capesize both load to 100% of their OWN
    # capacity, so ranking on vessel utilisation calls them equal. For a
    # 160,000 t parcel the Newcastlemax leaves 44,200 t of chartered
    # capacity empty. The bigger ship must rank lower.
    cape = next(o for o in opts if o['vessel'] == 'Capesize'
                and o['port'] == 'Gangavaram')
    ncmax = next(o for o in opts if o['vessel'] == 'Newcastlemax'
                 and o['port'] == 'Gangavaram')
    check('Capesize fills %.0f%% vs Newcastlemax %.0f%% on this parcel'
          % (cape['parcel_utilisation'] * 100,
             ncmax['parcel_utilisation'] * 100),
          cape['parcel_utilisation'] > ncmax['parcel_utilisation'])
    check('and the Capesize is ranked above the Newcastlemax',
          opts.index(cape) < opts.index(ncmax))
    bad = [o for o in opts if not 0 < o['parcel_utilisation'] <= 1.0 + 1e-9]
    check('parcel fill is always in (0, 1]', not bad, bad[:3])

    print('\n[11] uneconomic options are demoted, and can be excluded')
    une = [o for o in opts if o['verdict'] == 'uneconomic']
    als = [o for o in opts if o['verdict'] == 'alongside']
    if une and als:
        check('no uneconomic option outranks an equal-voyage alongside one',
              all(opts.index(u) > opts.index(a) for u in une for a in als
                  if a['voyages'] == u['voyages']))
    check('include_uneconomic=False removes them',
          not any(o['verdict'] == 'uneconomic'
                  for o in ports.options_for(parcel,
                                             include_uneconomic=False)))

    print('\n[12] bad input is rejected rather than answered')
    for bad_parcel in (0, -5, -0.001):
        try:
            ports.options_for(bad_parcel)
            check('parcel_t=%r rejected' % bad_parcel, False,
                  'returned options for a non-positive parcel')
        except ValueError:
            check('parcel_t=%r rejected' % bad_parcel, True)
    for fn, args in [(ports.max_cargo, ('Nope', 'Paradip')),
                     (ports.max_cargo, ('Capesize', 'Atlantis'))]:
        try:
            fn(*args)
            check('%s rejected' % (args,), False, 'no error raised')
        except KeyError as e:
            check('%s rejected with a listing of valid names' % (args,),
                  'known:' in str(e), str(e)[:70])
    for rho in (0.0, -1.0):
        try:
            ports.tpc('Capesize', rho)
            check('density %r rejected' % rho, False,
                  'negative or zero density accepted')
        except ValueError:
            check('density %r rejected' % rho, True)

    print('\n[13] a reported capacity is never above the true one')
    # Rounding a capacity UP is an instruction to overload. Every
    # reported tonnage must be at or below the exact figure.
    over = []
    for vessel in ports.VESSELS:
        for port in ports.PORTS:
            exact, _ = ports.max_cargo(vessel, port)
            shown = ports.can_serve(vessel, port)['max_cargo_t']
            if shown > exact + 1e-9:
                over.append('%s/%s shows %d vs %.2f'
                            % (vessel, port, shown, exact))
    check('no combination over-reports its capacity', not over, over)

    print('\na non-finite density is refused, not quietly propagated')
    # `density <= 0` is False for NaN, so a bare sign test lets a NaN
    # through and it comes out the far end as a NaN TPC, a NaN allowance
    # and finally a NaN tonnage on screen.
    for bad in (0.0, -1.0, float('nan'), float('inf'), float('-inf'),
                None, 'x'):
        for fn, nm in ((ports.tpc, 'tpc'),
                       (ports.dock_water_allowance, 'dock_water_allowance')):
            try:
                v = fn('Capesize', bad)
                check('%s(density=%r) refused' % (nm, bad), False,
                      'returned %r' % (v,))
            except (ValueError, TypeError):
                check('%s(density=%r) refused' % (nm, bad), True)
    check('and a normal density still works: tpc = %.2f'
          % ports.tpc('Capesize'), abs(ports.tpc('Capesize') - 109.91) < 0.01)

    print('\nthe bridge to the lowercase modules is complete both ways')
    # ports.py names ports for display; congestion.py and risk.py key
    # them in lowercase. The MILP has to join the two, and a join that
    # guesses at case fails silently on whichever spelling differs.
    from src import congestion
    missing = [p for p in ports.PORTS if p not in ports.PORTWATCH_KEY]
    check('every port in PORTS has an entry in PORTWATCH_KEY',
          not missing, missing)
    extra = [p for p in ports.PORTWATCH_KEY if p not in ports.PORTS]
    check('and PORTWATCH_KEY invents no ports of its own', not extra, extra)
    mapped = set(v for v in ports.PORTWATCH_KEY.values() if v)
    check('every discharge port congestion.py tracks is reachable from here',
          mapped == set(congestion.DISCHARGE),
          mapped ^ set(congestion.DISCHARGE))
    unfed = sorted(p for p, v in ports.PORTWATCH_KEY.items() if v is None)
    check('berths with no PortWatch feed map to None rather than being '
          'left out: %s' % unfed,
          all(ports.portwatch_key(p) is None for p in unfed))
    try:
        ports.portwatch_key('Atlantis')
        check('portwatch_key refuses an unknown port', False, 'accepted')
    except KeyError:
        check('portwatch_key refuses an unknown port', True)

    print('\n' + '=' * 62)
    if FAIL:
        print('  %d FAILED:' % len(FAIL))
        for f in FAIL:
            print('    - %s' % f)
        print('=' * 62)
        return 1
    print('  all port constraint checks passed')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

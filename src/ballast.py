"""
Module B, part 5: the empty leg.

Deliverable (c) of the problem statement is "idle / deadhead management".
It is two different questions, and this repository can answer exactly one
of them.

  IDLE      how long a ship waits at anchorage before it gets a berth.
            NOT ANSWERABLE HERE. PortWatch reports arrivals, not arrivals
            and departures, so there is no dwell time to compute and any
            figure would be invented. src/congestion.py reports berth
            load as a leading indicator of competition and says plainly
            that it is not waiting time.

  DEADHEAD  whether a ship that discharges here can pick up a cargo for
            the return leg, or has to sail empty. THIS IS MEASURABLE,
            and it falls straight out of data already fetched.

WHY THE EMPTY LEG IS THE BUYER'S PROBLEM

A vessel with no return cargo sails in ballast - tanks flooded for
stability, earning nothing, still burning fuel and paying crew. The owner
knows before quoting whether the discharge port offers a backhaul, and
prices the empty leg into the rate. So the berth chosen changes the
freight offered, before any negotiation happens.

WHAT IS MEASURED, AND WHAT IS NOT

Measured: the share of inbound dry bulk tonnage a port cannot match with
outbound dry bulk tonnage. At Haldia that is over 90% and has been every
year since 2019; at Paradip it is zero, because more dry bulk leaves than
arrives.

Not measured: whether a particular ship can take a particular export
cargo. Hold cleanliness, parcel size and timing all intervene, so
aggregate matching is an UPPER bound on backhaul availability - and the
ballast share below is therefore a LOWER bound on empty sailing. It is
also silent on what an empty leg costs, which is why the money is a
declared input everywhere it is used.

    python -m src.ballast
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import congestion, ports  # noqa: E402

# Two years of arrivals. Long enough that a single quarter cannot swing
# the ratio, short enough to reflect the berth as it trades now.
WINDOW_DAYS = 730

# A reliability floor, but on TONNAGE rather than calls.
#
# src/congestion.py gates on calls per day because it scores a call rate
# against its own median, and that ratio is unstable near zero. This
# module ratios tonnage, so the calls floor is the wrong instrument:
# Dhamra runs 0.99 dry bulk calls a day and would fail it, while moving
# 39 kt a day on a ratio that has sat between 0.3 and 0.7 for eight
# straight years. That is not noise.
#
# What does break a tonnage ratio is lumpiness. Below roughly one
# shipload a day the ratio stops being a rate and becomes an accident of
# which few vessels happened to call. So the floor is one full cargo of
# the smallest class this project models, read from the fleet definition
# rather than typed in, so it tracks that fleet if it changes.
MIN_IMPORT_T_PER_DAY = min(v['dwt'] - v['constants']
                           for v in ports.VESSELS.values())

# A port that ships out at least as much dry bulk as it takes in can, in
# aggregate, offer every arriving vessel a return cargo. Below that, some
# share must leave empty. The cut is at parity because that is where the
# arithmetic changes sign, not because a threshold was chosen to make the
# answer look tidy.
PARITY = 1.0


def _smallest_class():
    return min(ports.VESSELS,
               key=lambda v: ports.VESSELS[v]['dwt']
               - ports.VESSELS[v]['constants'])


def _series(port, as_of=None, window=WINDOW_DAYS):
    d = congestion.load(port)
    if as_of is not None:
        cut = pd.Timestamp(as_of)
        if cut < d.index.min():
            raise ValueError('as_of %s precedes this port\'s data, which '
                             'starts %s' % (cut.date(), d.index.min().date()))
        d = d[d.index <= cut]
    return d.tail(window)


def imbalance(port, as_of=None, window=WINDOW_DAYS):
    """How much of what arrives at this berth can find a way home.

    ballast_share is the fraction of inbound dry bulk tonnage with no
    outbound dry bulk tonnage to match it. Zero means a return cargo
    exists for everything that arrives; 0.94 means roughly nine ships in
    ten leave empty.
    """
    if congestion.role(port) != 'discharge':
        raise ValueError('%r is a load port; the empty leg question is '
                         'asked at the discharge end' % (port,))
    d = _series(port, as_of, window)
    days = max(len(d), 1)
    imp = float(d['import_dry_bulk'].sum())
    exp = float(d['export_dry_bulk'].sum())
    calls = float(d['portcalls_dry_bulk'].sum()) / days

    out = {
        'port': port,
        'as_of': str(d.index.max().date()) if len(d) else None,
        'days': int(days),
        'import_kt_per_day': round(imp / days / 1000, 2),
        'export_kt_per_day': round(exp / days / 1000, 2),
        'calls_per_day': round(calls, 2),
    }

    per_day = imp / days
    if per_day < MIN_IMPORT_T_PER_DAY or imp <= 0:
        out.update({
            'reliable': False,
            'ratio': None,
            'ballast_share': None,
            'verdict': 'not scored',
            'note': ('%.1f kt/day inbound is under one %s cargo a day '
                     '(%.1f kt), so the ratio is lumpy rather than a rate'
                     % (per_day / 1000, _smallest_class(),
                        MIN_IMPORT_T_PER_DAY / 1000)),
        })
        return out

    ratio = exp / imp
    share = max(0.0, 1.0 - ratio)
    st = stability(port, as_of)
    out.update({
        'reliable': True,
        'ratio': round(ratio, 3),
        'ballast_share': round(share, 4),
        'verdict': ('return cargo available' if ratio >= PARITY else
                    'most ships leave empty' if share >= 0.5 else
                    'partial backhaul'),
        # Scored is not the same as stable. A berth whose annual ratio
        # crosses parity says one thing some years and the opposite in
        # others, and should not be priced as though it were settled.
        'stable': not st['crosses_parity'],
        'note': ('%.0f%% of inbound dry bulk tonnage has no outbound dry '
                 'bulk to match it%s'
                 % (share * 100,
                    '' if not st['crosses_parity'] else
                    ' - but this berth crosses parity between years '
                    '(%.1f to %.1f), so treat it as unsettled'
                    % (st['min'], st['max']))),
    })
    return out


def stability(port, as_of=None):
    """The same ratio year by year.

    A structural property of a berth is worth acting on; a ratio that
    wanders is not. This is what separates the two.
    """
    d = congestion.load(port)
    if as_of is not None:
        d = d[d.index <= pd.Timestamp(as_of)]
    by = {}
    for y, g in d.groupby(d.index.year):
        imp = float(g['import_dry_bulk'].sum())
        if imp <= 0:
            continue
        by[int(y)] = round(float(g['export_dry_bulk'].sum()) / imp, 2)
    if not by:
        return {'by_year': {}, 'min': None, 'max': None, 'crosses_parity': None}
    vals = list(by.values())
    return {
        'by_year': by,
        'min': min(vals),
        'max': max(vals),
        # A berth whose ratio never crosses parity is telling you the same
        # thing every year, which is the only kind worth pricing.
        'crosses_parity': min(vals) < PARITY <= max(vals),
    }


def profile(ports=None, as_of=None, window=WINDOW_DAYS):
    """Every discharge berth, worst empty leg first."""
    ps = list(ports) if ports else list(congestion.DISCHARGE)
    out = []
    for p in ps:
        row = imbalance(p, as_of, window)
        row['stability'] = stability(p, as_of)
        out.append(row)
    out.sort(key=lambda r: (r['ballast_share'] is None,
                            -(r['ballast_share'] or 0)))
    return out


def shares_by_berth(as_of=None, window=WINDOW_DAYS):
    """Ballast share keyed the way src/ports.py names its berths.

    congestion.py and this module key ports in lowercase; ports.py uses
    the display spelling. Crossing that gap by guessing at case is
    exactly the silent-miss this project added ports.PORTWATCH_KEY to
    prevent, so the bridge is used rather than reimplemented.

    A berth with no PortWatch feed - Gangavaram, Sagar-Sandheads - gets
    None, not zero. Zero would assert a return cargo we never measured.
    """
    scored = {r['port']: r['ballast_share'] for r in profile(None, as_of, window)}
    out = {}
    for display in ports.PORTS:
        key = ports.portwatch_key(display)
        out[display] = scored.get(key) if key else None
    return out


def ballast_penalty(voyage_cost, ballast_pct, as_of=None,
                    window=WINDOW_DAYS):
    """Extra cost per voyage, by class and berth, for the empty leg.

    `ballast_pct` is what an empty repositioning leg costs as a fraction
    of a laden voyage - a number this repository does not know and will
    not invent, so the caller supplies it. Bigger ships cost more to
    reposition, which the class's own voyage cost already carries, so the
    penalty scales with it.

        penalty[v][p] = voyage_cost[v] * ballast_pct * ballast_share[p]

    A berth with a return cargo adds nothing. An unscored berth also adds
    nothing, because guessing there would be inventing a number.
    """
    try:
        pct = float(ballast_pct)
    except (TypeError, ValueError):
        raise ValueError('ballast_pct must be a number, got %r'
                         % (ballast_pct,))
    if not 0.0 <= pct <= 1.0:
        raise ValueError('ballast_pct is a fraction of a laden voyage and '
                         'must be between 0 and 1, got %r' % (ballast_pct,))

    shares = shares_by_berth(as_of, window)
    # None propagates. A berth we hold no arrivals for must not be handed
    # a zero penalty, because zero reads as "measured, and there is no
    # empty leg" - which is how an optimiser ends up preferring exactly
    # the berths nobody has data on.
    return ({v: {p: (None if s is None else round(float(c) * pct * s, 2))
                 for p, s in shares.items()}
             for v, c in voyage_cost.items()},
            shares)


if __name__ == '__main__':
    rows = profile()
    print('=' * 78)
    print('  THE EMPTY LEG  -  can a ship discharging here find a cargo home?')
    print('=' * 78)
    print('  %-15s %9s %9s %8s %9s  %s'
          % ('BERTH', 'IMP kt/d', 'EXP kt/d', 'RATIO', 'EMPTY', 'READS AS'))
    for r in rows:
        if not r['reliable']:
            print('  %-15s %9.1f %9.1f %8s %9s  %s'
                  % (r['port'].replace('_', ' ').title(),
                     r['import_kt_per_day'], r['export_kt_per_day'],
                     '-', '-', 'not scored'))
            continue
        print('  %-15s %9.1f %9.1f %8.2f %8.0f%%  %s'
              % (r['port'].replace('_', ' ').title(),
                 r['import_kt_per_day'], r['export_kt_per_day'],
                 r['ratio'], r['ballast_share'] * 100, r['verdict']))

    print('\n  IS IT STRUCTURAL, OR JUST THIS YEAR?')
    for r in rows:
        st = r['stability']
        if not st['by_year']:
            continue
        span = '%.1f - %.1f' % (st['min'], st['max'])
        verdict = ('wanders across parity' if st['crosses_parity']
                   else 'same answer every year')
        print('  %-15s %-13s %s' % (r['port'].replace('_', ' ').title(),
                                    span, verdict))

    print('\n' + '-' * 78)
    print('  Measured: tonnage in against tonnage out. Not measured: whether')
    print('  a given ship can take a given export cargo, or what an empty')
    print('  leg costs. Aggregate matching is an upper bound on backhaul, so')
    print('  the empty share above is a LOWER bound on ballast sailing.')
    print('=' * 78)

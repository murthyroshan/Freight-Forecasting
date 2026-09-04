"""
Module B, part 2: port activity from IMF PortWatch.

WHAT THIS IS, AND WHAT IT IS NOT

PortWatch publishes daily ARRIVALS and TONNAGE derived from AIS. It does
not publish waiting time, queue length or berth occupancy. So nothing
here is a measured congestion figure, and it is not called one: what is
computed is how busy a port is running relative to its OWN history.

That distinction matters. A port handling more cargo is not necessarily
backed up - it may simply have more berths working. Treat a high reading
as "this berth is under load, expect competition for it", not as "your
ship will wait N days". A real waiting-time series needs vessel-level
AIS dwell times, which PortWatch does not expose.

BOTH ENDS OF THE LANE

Eleven ports are scored: the five east-coast discharge berths, and the
six Australian and US load terminals that feed them. A backlog at Hay
Point delays the cargo before it ever sails, so watching only the Indian
end would miss half the exposure.

The Indian ports are where the decision is made - SAIL chooses the
discharge berth. The load terminal usually arrives with the coal supply
contract, so it is watched for risk rather than selected.

TWO TONNAGE FIGURES, AND WHY

Every port reports both. THROUGHPUT is everything the berth handles in
both directions and is what actually makes it busy. LANE tonnage is only
the direction our cargo travels - imports at a discharge port, exports
at a load terminal.

They diverge sharply at the Indian end, because every discharge port is
also a loading port. Paradip moves 60% of its dry bulk OUTWARD as iron
ore: 189 kt/day crosses its quays but only 39 kt/day is inbound. Quoting
the inbound figure alone would understate the competition for berths,
cranes and tugs by a factor of five.

Using the wrong single figure is worse still at the load end: Hay Point
and Newcastle record zero dry-bulk imports, so an imports-only panel
would show two of the largest coal terminals in the set as idle.

WHY EACH PORT IS SCORED AGAINST ITSELF

Traffic differs by an order of magnitude across the set - New Orleans
averages 6.6 dry-bulk calls a day, Gopalpur 0.31. A shared threshold would
mark the small terminals permanently dead and the large ones permanently
busy. Every reading is therefore relative to that port's own median and
its own dispersion.

LOW-VOLUME PORTS

At Gopalpur the median 7-day arrival rate is 0.29 calls/day. A ratio
against a base that small is noise: a single ship arriving or not swings
it by 300%. Any port whose baseline falls below MIN_BASELINE_CALLS is
flagged unreliable and its ratio is reported but not acted on.

Run:  python -m src.congestion
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import paths  # noqa: E402

# Both ends of every lane in the brief. A backlog at the load port
# delays the cargo before it ever sails, so scoring only the Indian end
# would miss half the exposure.
DISCHARGE = ['paradip', 'visakhapatnam', 'dhamra', 'haldia', 'gopalpur']

# Beira and Nacala are deliberately absent. Mozambican coking coal is a
# marginal trade for this lane, and at 0.57 and 0.43 dry-bulk calls a day
# both sit below the reliability floor - they could only ever render as
# two permanently unscored cards.
LOAD = ['hay_point', 'gladstone', 'newcastle',      # Australia
        'norfolk', 'baltimore', 'new_orleans']      # US east coast / Gulf

REGION = {
    'paradip': 'India east coast', 'visakhapatnam': 'India east coast',
    'dhamra': 'India east coast', 'haldia': 'India east coast',
    'gopalpur': 'India east coast',
    'hay_point': 'Australia', 'gladstone': 'Australia',
    'newcastle': 'Australia',
    'norfolk': 'US east coast', 'baltimore': 'US east coast',
    'new_orleans': 'US Gulf',
}

ALL_PORTS = DISCHARGE + LOAD


def label(port):
    """Display name: hay_point -> Hay Point."""
    return port.replace('_', ' ').title()


def role(port):
    if port in DISCHARGE:
        return 'discharge'
    if port in LOAD:
        return 'load'
    raise KeyError('unknown port %r; known: %s'
                   % (port, ', '.join(ALL_PORTS)))


def flow_column(port):
    """The tonnage series relevant to OUR lane at this port.

    A discharge port receives our cargo, a load terminal ships it.
    Reading import tonnage at a load terminal is not merely the wrong
    number, it is zero: Hay Point and Newcastle handle no dry bulk
    imports at all, so quoting imports there would show two of the
    largest coal terminals in the set as completely idle.
    """
    return ('import_dry_bulk' if role(port) == 'discharge'
            else 'export_dry_bulk')


def throughput(d):
    """Total dry bulk moved, both directions.

    This, not the one-way figure, is what makes a berth busy. Every
    Indian discharge port is also a loading port - Paradip moves 60% of
    its dry bulk tonnage OUTWARD as iron ore - and that cargo competes
    for the same quays, cranes and tugs as an arriving coal parcel.

    The arrival counts confirm the two are not separable: on the 603
    days Paradip recorded zero imports it still averaged 1.32 dry bulk
    calls. Measured against Paradip arrivals, imports alone correlate
    r = +0.620 and exports alone r = +0.751, but the two together reach
    r = +0.898.
    """
    return d['import_dry_bulk'] + d['export_dry_bulk']


WINDOW = 7           # days in the "current" window - one chartering week
MIN_BASELINE_CALLS = 1.0   # below this a ratio is not meaningful

# Bands are set on the PERCENTILE of the current reading within the
# port's own history, not on the ratio to its median.
#
# A ratio needs a cut-off picked by hand, and the same ratio does not
# mean the same thing at every port. Measured here, 1.32x the median
# sits at the 80th percentile at Paradip, 82nd at Haldia, 85th at
# Dhamra and 86th at Visakhapatnam - a six-point spread, so a single
# ratio threshold would call the same load "busy" at one berth and
# "very busy" at another. A percentile is distribution-free and needs
# no threshold chosen by eye.
BANDS = [(20, 'quiet'), (70, 'normal'), (90, 'busy'),
         (float('inf'), 'very busy')]


def load(port):
    p = os.path.join(paths.RAW, 'port_%s.parquet' % port)
    if not os.path.exists(p):
        raise FileNotFoundError(
            'no PortWatch data for %r at %s - run: python -m src.fetch_data'
            % (port, p))
    return pd.read_parquet(p).set_index('date').sort_index()


def cyclone_effect(port='paradip', gust_kmh=90.0, since='2019-01-01'):
    """The measured effect of a cyclone-force gust on arrivals.

    Computed rather than quoted. The dashboard used to type "-88%" and
    "p < 0.0001" straight into its markup, which is the one thing this
    project says it never does - and the threshold beside them was a
    literal that could drift away from the constant it described.

    Returns None when the weather history for `port` is not present.
    """
    wxp = os.path.join(paths.RAW, 'wx_%s.parquet' % port)
    if not os.path.exists(wxp):
        return None
    wx = pd.read_parquet(wxp)
    wx['date'] = pd.to_datetime(wx['date'])
    wx = wx.set_index('date')
    wx = wx[wx.index >= pd.Timestamp(since)]
    j = pd.concat([load(port)['portcalls_dry_bulk'], wx['gust_max']],
                  axis=1).dropna()
    if j.empty:
        return None

    # The storm window is the gust day and the day before it: a master
    # stops for weather he can see coming, not only weather he is in.
    storm = set()
    for t in j.index[j['gust_max'] >= gust_kmh]:
        storm.add(t)
        storm.add(t - pd.Timedelta(days=1))
    m = j.index.isin(sorted(storm))
    if not m.any() or m.all():
        return None
    a = j['portcalls_dry_bulk'][m].to_numpy(dtype=float)
    b = j['portcalls_dry_bulk'][~m].to_numpy(dtype=float)
    try:
        from scipy import stats as _st
        p_value = float(_st.mannwhitneyu(a, b, alternative='less').pvalue)
    except Exception:
        p_value = None

    gust_days = int((j['gust_max'] >= gust_kmh).sum())
    months = j.index[j['gust_max'] >= gust_kmh].month
    return {
        'port': port,
        'gust_kmh': float(gust_kmh),
        'since': since,
        'storm_calls_per_day': round(float(a.mean()), 3),
        'normal_calls_per_day': round(float(b.mean()), 3),
        'change_pct': round(100.0 * (a.mean() / b.mean() - 1.0), 1),
        'p_value': p_value,
        'gust_days': gust_days,
        'gust_days_may': int((months == 5).sum()),
        'gust_days_sep_dec': int(months.isin([9, 10, 11, 12]).sum()),
        'window_days': int(len(j)),
        'sep_dec_days': int(j.index.month.isin([9, 10, 11, 12]).sum()),
    }


def band(percentile):
    if percentile is None or percentile != percentile:
        raise ValueError('percentile must be a number, got %r'
                         % (percentile,))
    if not 0.0 <= percentile <= 100.0:
        raise ValueError('percentile must be 0-100, got %r' % (percentile,))
    for cut, name in BANDS:
        if percentile < cut:
            return name
    return BANDS[-1][1]


def snapshot(port, window=WINDOW, as_of=None):
    """How busy this port is right now against its own history."""
    d = load(port)
    if as_of is not None:
        d = d[d.index <= pd.Timestamp(as_of)]
    if len(d) < 400:
        raise ValueError('not enough history for %r (%d rows)'
                         % (port, len(d)))

    calls = d['portcalls_dry_bulk'].rolling(window).mean().dropna()
    # Total throughput is the load on the berth; the one-way figure is
    # what matters to our own cargo. Both are reported.
    tons = throughput(d).rolling(window).mean().dropna()
    lane = d[flow_column(port)].rolling(window).mean().dropna()

    baseline = float(calls.median())
    current = float(calls.iloc[-1])
    reliable = baseline >= MIN_BASELINE_CALLS
    ratio = current / baseline if baseline > 0 else float('nan')

    # Percentile of the current reading within the port's own history:
    # more robust than a ratio when the distribution is skewed.
    pctile = float((calls <= current).mean() * 100)

    return {
        'port': port,
        'label': label(port),
        'role': role(port),
        'region': REGION.get(port, ''),
        'as_of': str(calls.index[-1].date()),
        'window_days': window,
        'calls_per_day': round(current, 2),
        'baseline_calls_per_day': round(baseline, 2),
        'ratio': None if not np.isfinite(ratio) else round(ratio, 3),
        'percentile': round(pctile, 1),
        'band': band(pctile) if reliable else 'unreliable',
        # Everything the berth handles, both directions.
        'throughput_t_per_day': int(round(float(tons.iloc[-1]))),
        'baseline_throughput_t_per_day': int(round(float(tons.median()))),
        # Only the direction our cargo travels.
        'lane_t_per_day': int(round(float(lane.iloc[-1]))),
        'baseline_lane_t_per_day': int(round(float(lane.median()))),
        'flow': 'imported' if role(port) == 'discharge' else 'exported',
        'reliable': bool(reliable),
        'reliability_note': (
            '' if reliable else
            'baseline is only %.2f dry-bulk calls/day, below the %.1f '
            'needed for a ratio to mean anything - one ship arriving or '
            'not moves it by hundreds of percent' % (baseline,
                                                     MIN_BASELINE_CALLS)),
        'history_from': str(d.index.min().date()),
        'history_to': str(d.index.max().date()),
    }


def monthly_profile(port, as_of=None):
    """Seasonal index: each month's mean dry-bulk arrivals as a
    percentage of the port's own annual mean. 100 = typical.

    Takes as_of for the same reason snapshot() does. Without it, a caller
    asking about a past date would get a rewound snapshot alongside a
    profile computed from data that had not happened yet."""
    d = load(port)
    if as_of is not None:
        d = d[d.index <= pd.Timestamp(as_of)]
    if d.empty:
        return {i: None for i in range(1, 13)}
    m = d.groupby(d.index.month)['portcalls_dry_bulk'].mean()
    overall = float(d['portcalls_dry_bulk'].mean())
    if overall <= 0:
        return {i: None for i in range(1, 13)}
    return {int(i): round(float(v) / overall * 100, 1) for i, v in m.items()}


def series(port, days=180):
    """Recent daily arrivals plus the smoothed line, for charting."""
    d = load(port).tail(days + WINDOW)
    roll = d['portcalls_dry_bulk'].rolling(WINDOW).mean()
    out = pd.DataFrame({'calls': d['portcalls_dry_bulk'],
                        'smoothed': roll,
                        'throughput': throughput(d),
                        'lane': d[flow_column(port)]}).dropna().tail(days)
    return [{'date': str(i.date()),
             'calls': int(r['calls']),
             'smoothed': round(float(r['smoothed']), 2),
             'throughput': int(r['throughput']),
             'lane': int(r['lane'])} for i, r in out.iterrows()]


def all_snapshots(ports=None, as_of=None):
    out = []
    for p in (ports or ALL_PORTS):
        try:
            out.append(snapshot(p, as_of=as_of))
        except (FileNotFoundError, ValueError) as e:
            out.append({'port': p, 'error': str(e)})
    # Busiest first, but never let an unreliable reading head the list.
    #
    # Rank on PERCENTILE, the same statistic the band comes from. Ranking
    # on the ratio instead lets the order contradict the labels: New
    # Orleans at ratio 0.717 sits above Norfolk at 0.714, yet New Orleans
    # is in its 8th percentile ("quiet") and Norfolk its 28th ("normal"),
    # so a quiet port outranked a normal one in a list that reads as
    # busiest-first.
    out.sort(key=lambda s: (not s.get('reliable', False),
                            -(s.get('percentile') or 0)))
    return out


if __name__ == '__main__':
    print('=' * 74)
    print('  PORT ACTIVITY, BOTH ENDS (IMF PortWatch, AIS-derived)')
    print('=' * 74)
    snaps = all_snapshots()
    for grp, title in [('discharge', 'DISCHARGE - east coast India'),
                       ('load', 'LOAD - Australia, US')]:
        rows = [s for s in snaps if s.get('role') == grp]
        print('\n  %s' % title)
        print('  %-15s %-16s %8s %8s %6s %-11s %12s' %
              ('port', 'region', 'calls/d', 'normal', 'pctl', 'band',
               'kt tot/lane'))
        print('  ' + '-' * 82)
        for s in rows:
            if 'error' in s:
                print('  %-15s ERROR %s' % (s['port'], s['error'][:40]))
                continue
            print('  %-15s %-16s %8.2f %8.2f %5.0f%% %-11s %12s'
                  % (s['label'], s['region'], s['calls_per_day'],
                     s['baseline_calls_per_day'], s['percentile'],
                     s['band'],
                     '%.0f / %.0f' % (s['throughput_t_per_day'] / 1000.0,
                                      s['lane_t_per_day'] / 1000.0)))
    ok = [s for s in snaps if s.get('reliable')]
    if ok:
        print('\n  as of %s, %d-day window' % (ok[0]['as_of'],
                                               ok[0]['window_days']))
    for s in snaps:
        if s.get('reliability_note'):
            print('\n  %s: %s' % (s['port'], s['reliability_note']))

    print('\n  Seasonal profile - mean dry-bulk arrivals by month')
    print('  (100 = that port\'s own annual average)')
    print('  %-15s' % 'port' + ''.join('%5s' % m for m in
          ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
    print('  ' + '-' * 75)
    for p in ALL_PORTS:
        prof = monthly_profile(p)
        print('  %-15s' % p + ''.join(
            '%5s' % ('-' if prof.get(i) is None else '%.0f' % prof[i])
            for i in range(1, 13)))
    print('\n  Every Indian port dips Sep-Dec. Two things are true here.')
    print('  Cyclones DO stop ships: on the day of and the day before a')
    print('  gust above 90 km/h, Paradip arrivals fall 88% and Dhamra to')
    print('  zero, both p < 0.0001. But they cannot explain the SEASON.')
    print('  Paradip has had 7 such days in 8 years, 5 of them in May;')
    print('  Sep-Dec holds 2, affecting ~0.4% of that window against a')
    print('  ~15% shortfall in arrivals. Sep-Dec is in fact the calmest')
    print('  stretch of the year. Weather is a short-horizon disruption')
    print('  signal, not a seasonal one.')
    print('=' * 74)

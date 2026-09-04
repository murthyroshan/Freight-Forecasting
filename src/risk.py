"""
Module B, part 3: risk warnings.

Deliverable (d) of the problem statement. Four signals, and the honest
distinction between them matters more than the count:

  MEASURED   an effect this repository has quantified and significance
             tested. Two qualify: the cyclone rule and the width of the
             model's own conformal interval.
  CONTEXT    a real observation with no measured effect size attached.
             Reported so a desk can weigh it, never as a prediction.

Mixing those two is how a risk panel becomes decoration. Every warning
below carries a `measured` flag and a `basis` string naming the evidence,
so a reader can tell which is which without trusting the wording.

WHAT IS FORWARD-LOOKING AND WHAT IS NOT

The weather signal is a genuine forecast: Open-Meteo's free forecast API
returns ten days ahead, so a cyclone warning arrives before the storm.
Everything else looks backwards - port activity is last week's arrivals,
and the model's own uncertainty can only be measured on dates it scored,
which end in 2019. Each item states its own as-of date rather than
implying it is current.

Run:  python -m src.risk
"""

import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import paths, congestion  # noqa: E402

# Gust at which arrivals measurably stop. Not a round number picked for
# looks: at this threshold Paradip arrivals fall 88% and Dhamra to zero
# on the day of and the day before, both p < 0.0001. See docs/METHOD.md
# and tests/test_congestion.py section 6a.
GUST_HALT_KMH = 90.0

# Gale force. Shipping notices this, but we measured NO arrival effect at
# the 90th percentile of gusts (~54 km/h at Paradip: 3.04 calls on the
# windiest decile against 2.89 on calm days). So anything between gale
# and halt is reported as a watch, explicitly without a measured effect.
GUST_GALE_KMH = 62.0

# Heavy rain stops grabs working coal. Named rather than inlined so the
# prose below can quote the constant instead of restating a number that
# could drift away from it.
RAIN_HEAVY_MM = 50.0

FORECAST_DAYS = 10

# Per-port timeout. assess() calls this once per site in sequence, so
# the worst case a caller can wait is TIMEOUT x len(SITES). That product
# has to stay under the browser's own timeout, which in turn stays under
# the one tests/test_app.py uses: 12 x 5 = 60s server, 75s browser, 90s
# test. tests/test_risk.py asserts the first of those.
TIMEOUT = 12

SEVERITY_ORDER = {'critical': 0, 'warning': 1, 'watch': 2, 'clear': 3}


def _dmy(d):
    """Dates inside a sentence are DD/MM/YYYY. The as_of fields stay ISO
    because callers sort on them; only prose is reformatted."""
    return pd.Timestamp(d).strftime('%d/%m/%Y')


def _unavailable(port, what, exc):
    """A signal that could not be computed is REPORTED, never dropped.

    Silently returning nothing shrinks the panel from sixteen checks to
    two with no explanation, and every remaining row still reads
    reassuringly. That is the same failure as answering an unknown port
    with 'no disruptive weather forecast' - absence of a warning being
    mistaken for absence of risk. So a gap gets its own row, and it sits
    in the raised list where it can be seen.
    """
    return {'kind': 'unavailable', 'port': port, 'severity': 'watch',
            'measured': False,
            'title': '%s unavailable' % what,
            'detail': ('This check could not run, so it is neither clear nor '
                       'raised: %s' % str(exc)[:110]),
            'basis': ('no data - surfaced so the gap is visible rather than '
                      'silently reducing the number of checks')}


def _forecast(port, days=FORECAST_DAYS, timeout=TIMEOUT):
    """Ten-day gust and rainfall outlook for one discharge port."""
    lat, lon = congestion_site(port)
    r = requests.get('https://api.open-meteo.com/v1/forecast', timeout=timeout,
                     params={'latitude': lat, 'longitude': lon,
                             'daily': ('wind_gusts_10m_max,wind_speed_10m_max,'
                                       'precipitation_sum'),
                             'timezone': 'Asia/Kolkata',
                             'forecast_days': days})
    # A rate-limited or failing Open-Meteo answers with an HTML body, and
    # .json() would then raise something that reads like a parser bug
    # rather than an outage. Say what actually happened.
    if r.status_code != 200:
        raise RuntimeError('Open-Meteo returned HTTP %d' % r.status_code)
    j = r.json()
    if 'daily' not in j:
        raise RuntimeError(j.get('reason', 'forecast unavailable'))
    d = j['daily']
    out = pd.DataFrame({
        'date': pd.to_datetime(d['time']),
        'gust': d['wind_gusts_10m_max'],
        'wind': d['wind_speed_10m_max'],
        'rain': d['precipitation_sum'],
    })
    for c in ('gust', 'wind', 'rain'):
        out[c] = pd.to_numeric(out[c], errors='coerce')
    # A null gust compares False against every threshold, so a frame of
    # nulls would fall past the halt and gale branches and land on "no
    # disruptive weather forecast" - the most reassuring row the panel
    # can print, built from no data at all. Drop them here and let an
    # empty frame be reported as an outage instead.
    out = out[np.isfinite(out['gust'])].reset_index(drop=True)
    return out


# Must stay identical to fetch_data.WEATHER_SITES - a forecast taken at
# a different point from the history would be calibrated against the
# wrong grid cell. tests/test_risk.py asserts the two agree.
SITES = {
    'paradip':       (20.26, 86.67),
    'visakhapatnam': (17.68, 83.21),
    'haldia':        (22.03, 88.08),
    'dhamra':        (20.78, 86.97),
    'gopalpur':      (19.2911, 84.9574),
}


def congestion_site(port):
    if port not in SITES:
        raise KeyError('no weather site for %r; known: %s'
                       % (port, ', '.join(SITES)))
    return SITES[port]


def weather_warnings(port, days=FORECAST_DAYS):
    """Forward-looking. The only MEASURED signal in this module."""
    # An unknown port is a caller mistake, not an outage, and must not
    # be answered with a reassuring "no disruptive weather". Check it
    # before the try, so only genuine failures degrade.
    congestion_site(port)
    try:
        f = _forecast(port, days)
        if f.empty:
            raise RuntimeError('forecast returned no days')
    except Exception as exc:
        # Not 'clear'. A clear row means we looked and found nothing; an
        # outage means we did not look, and the dashboard collapses clear
        # rows out of sight.
        return [_unavailable(port, 'Weather outlook', exc)]

    # Report the horizon actually returned, never the one requested -
    # Open-Meteo can answer short, and "over the next 10 days" on three
    # days of data is a claim about days nobody looked at.
    n = len(f)
    out = []
    halt = f[f['gust'] >= GUST_HALT_KMH]
    gale = f[(f['gust'] >= GUST_GALE_KMH) & (f['gust'] < GUST_HALT_KMH)]

    if len(halt):
        first = halt.iloc[0]
        out.append({
            'kind': 'weather', 'port': port, 'severity': 'critical',
            'measured': True,
            'as_of': str(first['date'].date()),
            'title': 'Berth likely to stop working',
            'detail': ('Gusts of %.0f km/h forecast for %s, above the %.0f km/h '
                       'threshold at which arrivals measurably halt. %d of the '
                       'next %d days are affected.'
                       % (first['gust'], _dmy(first['date']), GUST_HALT_KMH,
                          len(halt), n)),
            'basis': ('measured: on the day of and before a >%.0f km/h gust, '
                      'Paradip arrivals fall 88%% and Dhamra to zero, '
                      'p < 0.0001' % GUST_HALT_KMH),
        })
    elif len(gale):
        first = gale.iloc[0]
        out.append({
            'kind': 'weather', 'port': port, 'severity': 'watch',
            'measured': False,
            'as_of': str(first['date'].date()),
            'title': 'Gale-force gusts forecast',
            'detail': ('Gusts of %.0f km/h forecast for %s. Above gale force '
                       'but below the %.0f km/h level at which we measured any '
                       'effect on arrivals.'
                       % (first['gust'], _dmy(first['date']), GUST_HALT_KMH)),
            'basis': ('NOT measured at this level: at the 90th percentile of '
                      'gusts (~54 km/h) arrivals were 3.04/day against 2.89 '
                      'on calm days - no effect'),
        })
    else:
        out.append({
            'kind': 'weather', 'port': port, 'severity': 'clear',
            'measured': True,
            'as_of': str(f['date'].iloc[0].date()),
            'title': 'No disruptive weather forecast',
            'detail': ('Peak gust over the next %d days is %.0f km/h, against '
                       'a %.0f km/h halt threshold.'
                       % (n, f['gust'].max(), GUST_HALT_KMH)),
            'basis': 'Open-Meteo %d-day forecast' % n,
        })

    heavy = f[f['rain'] >= RAIN_HEAVY_MM]
    if len(heavy):
        out.append({
            'kind': 'weather', 'port': port, 'severity': 'watch',
            'measured': False,
            'as_of': str(heavy.iloc[0]['date'].date()),
            'title': 'Heavy rainfall forecast',
            'detail': ('%.0f mm forecast for %s, above the %.0f mm mark. Coal '
                       'is weather-worked cargo and heavy rain stops grabs, '
                       'but we have not measured that effect in this data.'
                       % (heavy.iloc[0]['rain'], _dmy(heavy.iloc[0]['date']),
                          RAIN_HEAVY_MM)),
            'basis': 'context only - no measured rainfall effect on arrivals',
        })
    return out


def berth_load_warning(port):
    """Backward-looking context from PortWatch arrivals."""
    try:
        s = congestion.snapshot(port)
    except Exception as exc:
        return _unavailable(port, 'Berth load', exc)
    if not s['reliable']:
        return {'kind': 'berth', 'port': port, 'severity': 'clear',
                'measured': False, 'as_of': s['as_of'],
                'title': 'Berth load not scored',
                'detail': s['reliability_note'][:150],
                'basis': 'baseline below the reliability floor'}

    sev = {'very busy': 'warning', 'busy': 'watch'}.get(s['band'], 'clear')
    # A percentile can arrive non-finite when a port\'s baseline history
    # is degenerate. Drop the clause rather than let _ord() raise and
    # take the whole assessment down with it.
    pct = s.get('percentile')
    where = ('the %s percentile of this port\'s own history'
             % _ord(pct)) if _finite(pct) else 'this port\'s own history'
    return {
        'kind': 'berth', 'port': port, 'severity': sev, 'measured': False,
        'as_of': s['as_of'],
        'title': 'Berth running %s' % s['band'],
        'detail': ('%.2f dry bulk calls/day against a %.2f normal - %s. '
                   'Expect competition for the berth.'
                   % (s['calls_per_day'], s['baseline_calls_per_day'], where)),
        'basis': ('context only: PortWatch reports arrivals, not queue '
                  'length or waiting time'),
    }


def _finite(v):
    try:
        return v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _ord(n):
    if not _finite(n):
        raise ValueError('_ord() needs a finite number, got %r' % (n,))
    n = int(round(n))
    if 11 <= n % 100 <= 13:
        return '%dth' % n
    return '%d%s' % (n, {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'))


def seasonal_warning(port, month=None):
    """Where this month sits in the port's own seasonal pattern."""
    try:
        prof = congestion.monthly_profile(port)
    except Exception as exc:
        return _unavailable(port, 'Seasonal profile', exc)
    # `month or today` would turn a caller's 0 into today's month
    # silently, so test for None and then range-check.
    m = pd.Timestamp.today().month if month is None else int(month)
    if not 1 <= m <= 12:
        raise ValueError('month must be 1-12, got %r' % (month,))
    v = prof.get(m)
    if v is None:
        return _unavailable(port, 'Seasonal profile',
                            'no profile for month %02d' % m)
    sev = 'watch' if v <= 92 else 'clear'
    direction = 'below' if v < 100 else 'above'
    return {
        'kind': 'seasonal', 'port': port, 'severity': sev, 'measured': False,
        'as_of': 'month %02d' % m,
        'title': 'Seasonally %s normal' % direction,
        'detail': ('Arrivals in month %02d average %.0f against this port\'s '
                   'own annual 100. Every Indian port dips September to '
                   'December.' % (m, v)),
        'basis': ('pattern is real and holds at all five ports, but its CAUSE '
                  'is unknown - we tested cyclone season against Open-Meteo '
                  'wind and the dip months are the calmest of the year'),
    }


def model_uncertainty():
    """How wide the forecast interval was at the last date the model
    could score. Historical by construction: the Baltic series ends in
    2019, so this is never a statement about today."""
    p = os.path.join(paths.PROCESSED, 'oos_predictions.parquet')
    if not os.path.exists(p):
        return _unavailable(None, 'Model uncertainty',
                            'run python -m src.train_model first')
    o = pd.read_parquet(p)
    o.index = pd.to_datetime(o.index)
    # An artefact written by an older version of train_model.py may not
    # carry interval columns. That is a missing signal, not a reason to
    # take every weather and berth warning down with it.
    if 'hi' not in o.columns or 'lo' not in o.columns:
        return _unavailable(None, 'Model uncertainty',
                            'artefact carries no interval columns')
    width = (o['hi'] - o['lo']).dropna()
    if width.empty:
        return _unavailable(None, 'Model uncertainty',
                            'no finite interval widths in the artefact')
    last = float(width.iloc[-1])
    pctile = float((width <= last).mean() * 100)
    sev = 'watch' if pctile >= 75 else 'clear'
    return {
        'kind': 'model', 'port': None, 'severity': sev, 'measured': True,
        'as_of': str(o.index[-1].date()),
        'title': 'Forecast interval %s than usual'
                 % ('wider' if pctile >= 50 else 'narrower'),
        'detail': ('At the last scored date the 80%% interval spanned %.3f in '
                   'log return, the %s percentile of its own history. A wide '
                   'band means the model is uncertain, not that rates will '
                   'move.' % (last, _ord(pctile))),
        'basis': ('measured: locally weighted conformal intervals, 81.7% '
                  'realised coverage against an 80% target'),
    }


def assess(ports=None, month=None):
    """Every warning, worst first.

    A port this module holds no weather for is refused outright. The
    alternative - degrading to 'no disruptive weather forecast' - would
    make a typo look like an all-clear, which is the one failure a risk
    panel must never have.
    """
    ports = list(SITES) if ports is None else list(ports)
    unknown = [p for p in ports if p not in SITES]
    if unknown:
        raise KeyError('no weather site for %s; known: %s'
                       % (', '.join(map(repr, unknown)), ', '.join(SITES)))
    out = []
    for p in ports:
        out.extend(weather_warnings(p))
        w = berth_load_warning(p)
        if w:
            out.append(w)
        s = seasonal_warning(p, month)
        if s:
            out.append(s)
    m = model_uncertainty()
    if m:
        out.append(m)
    out.sort(key=lambda w: (SEVERITY_ORDER.get(w['severity'], 9),
                            not w['measured'], w.get('port') or ''))
    return out


if __name__ == '__main__':
    rows = assess()
    print('=' * 76)
    print('  RISK WARNINGS')
    print('=' * 76)
    counts = {}
    for w in rows:
        counts[w['severity']] = counts.get(w['severity'], 0) + 1
    print('  %s' % '  '.join('%s: %d' % (k, counts.get(k, 0))
                             for k in ['critical', 'warning', 'watch', 'clear']))
    print('  %d measured, %d context-only'
          % (sum(w['measured'] for w in rows),
             sum(not w['measured'] for w in rows)))

    for w in rows:
        if w['severity'] == 'clear' and w['kind'] != 'weather':
            continue
        tag = 'MEASURED' if w['measured'] else 'context '
        port = (w['port'] or '-').replace('_', ' ').title()
        print('\n  [%-8s] %-9s %-14s %s'
              % (w['severity'].upper(), tag, port, w['title']))
        print('      %s' % w['detail'])
        print('      basis: %s' % w['basis'])

    print('\n' + '-' * 76)
    print('  Only the cyclone rule and the interval width are measured')
    print('  effects. Berth load and seasonality are real observations')
    print('  with no measured effect size, and are labelled as such so a')
    print('  desk can weigh them rather than act on them as forecasts.')
    print('=' * 76)

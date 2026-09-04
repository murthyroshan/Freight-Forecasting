"""
Data fetcher for the SAIL freight forecasting model.

Every source here has been verified to return real rows. Nothing is
simulated, and nothing is filled in when a fetch fails - a failed source
prints FAIL and leaves no file, so a downstream script cannot silently
train on a gap.

Sources, all free and keyless:
  Baltic indices   Mendeley 10.17632/t76ckh2ygg (CC BY 4.0), 2012-2019
  Market data      Yahoo Finance
  Port calls       IMF PortWatch (AIS-derived, IMF + Oxford)
  Weather          Open-Meteo archive

Run:  python fetch_data.py           (skips anything already cached)
      python fetch_data.py --force   (refetch everything)

LICENCE NOTE. The Baltic series past 2019-07-31 comes from a public
mirror with no stated licence. It is fetched at runtime and never
committed - data/ is gitignored - so nothing proprietary is
redistributed here. Production would need a Baltic Exchange
licence.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import paths  # noqa: E402

import time
import hashlib
import warnings

import pandas as pd
import requests

warnings.filterwarnings('ignore')

RAW = paths.RAW
START = '2012-01-01'
END = '2026-09-04'
FORCE = '--force' in sys.argv

# Mendeley: dataset t76ckh2ygg, the only file in it.
MENDELEY_URL = (
    'https://data.mendeley.com/public-files/datasets/t76ckh2ygg/files/'
    '9824e3e8-9e88-40e0-a27d-d8cc5e913f4b/file_downloaded'
)
MENDELEY_SHA256 = ('0689b716f21a10d9510bb6f1739a18c296c1efb8a8c5e4450'
                   '788e480a15791d4')

# Yahoo tickers. ^BDI and EURN are deliberately absent: ^BDI has never
# existed as a Yahoo symbol and EURN is delisted. Both return an empty
# frame, which is why every fetch below reports its row count instead of
# assuming success.
TICKERS = {
    'BDRY':     'bdry',      # dry bulk freight futures ETF - the live proxy
    # Brent, not WTI: marine bunker prices are benchmarked to Brent, and
    # the two are redundant here anyway (level r 0.984, return r 0.889).
    'BZ=F':     'brent',     # bunker cost driver
    'INR=X':    'usdinr',    # SAIL pays in INR, charters in USD
    '^GSPC':    'sp500',     # global risk appetite
    'HG=F':     'copper',    # industrial demand proxy
    'DX-Y.NYB': 'dxy',       # commodities are dollar-denominated
    'SBLK':     'sblk',      # dry bulk owners - they trade on the rate
    'DSX':      'dsx',
    'NMM':      'nmm',
    # GNK deliberately omitted: it correlates 0.43-0.53 with the three
    # owners above and its history starts 18 months later, so a fourth
    # name in the average buys nothing at ~394 effective observations.
    'BHP':      'bhp',       # miners - the cargo side
    'RIO':      'rio',
    'WHC.AX':   'whc',       # Whitehaven, Australian coal
}

# IMF PortWatch port ids, verified present in Daily_Ports_Data.
# Gangavaram and Sagar-Sandheads are NOT in PortWatch and are handled
# separately in the port-constraint table.
PORTS = {
    # East coast India - where the coal is discharged
    'port883':  'paradip',
    'port1367': 'visakhapatnam',
    'port442':  'haldia',
    'port290':  'dhamra',
    'port2299': 'gopalpur',
    # Load ports - Australia
    'port458':  'hay_point',
    'port398':  'gladstone',
    'port816':  'newcastle',
    # Load ports - US east coast
    'port826':  'norfolk',
    'port103':  'baltimore',
    'port812':  'new_orleans',
    # Mozambique (Beira port137, Nacala port784) is deliberately not
    # fetched: a marginal coking coal trade for this lane, and both sit
    # below the reliability floor in src/congestion.py.
}

PAGE = 1000   # IMF PortWatch maxRecordCount; asking for more is ignored

PORTWATCH = ('https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/'
             'services/Daily_Ports_Data/FeatureServer/0/query')

# Discharge ports, for the monsoon / weather signal.
# All five east coast discharge ports.
#
# The first four coordinates are kept exactly as they were first fetched,
# NOT snapped to PortWatch's published lat/lon, because the 90 km/h halt
# threshold in src/risk.py was measured on this history. The shift is
# only 3-5 km and for Paradip, Visakhapatnam and Haldia it lands in the
# same ERA5 cell - identical readings. Dhamra does not: its peak gust in
# the Cyclone Fani window reads 115.9 km/h here against 99.0 at
# PortWatch's point, a different cell. Moving it would quietly move the
# evidence, so it stays put.
#
# Gopalpur has no such history to preserve, so it takes PortWatch's own
# coordinate from the ports database.
WEATHER_SITES = {
    'paradip':       (20.26, 86.67),
    'visakhapatnam': (17.68, 83.21),
    'haldia':        (22.03, 88.08),
    'dhamra':        (20.78, 86.97),
    'gopalpur':      (19.2911, 84.9574),
}


def out(name):
    return os.path.join(RAW, name + '.parquet')


def cached(name):
    p = out(name)
    if os.path.exists(p) and not FORCE:
        n = len(pd.read_parquet(p))
        print('  SKIP %-16s cached, %d rows' % (name, n))
        return True
    return False


def report(name, df, datecol='date'):
    """Single place that prints a row count, so every number in the
    pitch traces back to a line this script actually printed."""
    df.to_parquet(out(name), index=False)
    lo, hi = df[datecol].min(), df[datecol].max()
    print('  OK   %-16s %5d rows  %s -> %s' %
          (name, len(df), str(lo)[:10], str(hi)[:10]))


# ---------------------------------------------------------------- baltic
def fetch_baltic():
    print('\n[1/5] Baltic indices (Mendeley, CC BY 4.0)')
    if cached('baltic_indices'):
        return
    xls = os.path.join(RAW, 'mendeley_bdi.xls')
    if not os.path.exists(xls) or FORCE:
        r = requests.get(MENDELEY_URL, timeout=120)
        r.raise_for_status()
        open(xls, 'wb').write(r.content)
    got = hashlib.sha256(open(xls, 'rb').read()).hexdigest()
    if got != MENDELEY_SHA256:
        print('  FAIL checksum mismatch - refusing to use this file')
        print('       expected %s' % MENDELEY_SHA256)
        print('       got      %s' % got)
        return
    df = pd.ExcelFile(xls).parse('MASTER EXCEL SHEET BDI ')
    df.columns = [c.strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    # CTI is 48% missing and is a clean-tanker index anyway; DTI is dirty
    # tanker, kept only as a crude "is shipping broadly hot" control.
    df = (df.dropna(subset=['date'])
            .drop(columns=['Date', 'CTI'])
            .sort_values('date')
            .reset_index(drop=True))
    df = df.rename(columns={'HSI': 'handysize', 'SI': 'supramax',
                            'PI': 'panamax', 'CI': 'capesize',
                            'DTI': 'dirty_tanker'})
    df = df[['date', 'capesize', 'panamax', 'supramax',
             'handysize', 'dirty_tanker']]
    report('baltic_indices', df)


# ------------------------------------------------- baltic, past 2019
# The licensed Mendeley copy stops on 2019-07-31. East Money, a public
# Chinese financial portal, mirrors the same Baltic Exchange indices
# through a keyless JSON API, which is what carries the series to today.
#
# LICENCE. The Baltic Exchange indices are proprietary and East Money
# states no licence for its mirror, so this project fetches at runtime
# and redistributes nothing - data/ is gitignored and no index value is
# committed. Production would need a Baltic Exchange licence.
#
# It is validated rather than trusted: fetch_baltic_extension() refuses
# to write anything unless the mirror reproduces the licensed copy on
# the overlapping days (see MIN_CORRELATION / MAX_P99_REL_GAP below).
EASTMONEY = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
EASTMONEY_IDS = {'capesize': 'EMI00107666',
                 'panamax': 'EMI00107665',
                 'supramax': 'EMI00107667'}

# HOW THE MIRROR IS VALIDATED, and why not by exact matching.
#
# Byte-identical agreement over 2012-2019 is 99.5% for Capesize and
# 99.2% for Panamax, but only 42.9% for Supramax. That is not a
# different index - the level correlation is 0.9992 and the median
# disagreement is 0.22% - it is noisier early revisions. Gating on the
# exact-match RATE would therefore have left Supramax about three points
# above its floor, and because a single failing series aborts the whole
# fetch, one bad year of revisions would have silently killed the
# Capesize extension too: the series the model actually needs.
#
# So agreement is measured in VALUE, which is what "the same series"
# actually means. Measured: correlation 1.0000 / 1.0000 / 0.9992, and
# 99th-percentile relative gaps of 0.000% / 0.000% / 3.367%.
MIN_CORRELATION = 0.99
MAX_P99_REL_GAP = 0.05

# Whatever the rest of the history looks like, the two sources must
# agree EXACTLY on the day the splice happens, or the halves sit on
# different bases and the join puts a step in the middle of the target.
JOIN_TOLERANCE = 1e-9


def _eastmoney(indicator, timeout=45):
    """Every daily print for one indicator, oldest first."""
    rows = []
    for page in range(1, 25):
        r = requests.get(EASTMONEY, timeout=timeout, params={
            'reportName': 'RPT_INDUSTRY_INDEX',
            'columns': 'REPORT_DATE,INDICATOR_VALUE',
            'filter': '(INDICATOR_ID="%s")' % indicator,
            'sortColumns': 'REPORT_DATE', 'sortTypes': '-1',
            'pageSize': '500', 'pageNumber': str(page),
            'source': 'WEB', 'client': 'WEB'})
        if r.status_code != 200:
            raise RuntimeError('east money returned HTTP %d' % r.status_code)
        page_rows = (r.json().get('result') or {}).get('data')
        if not page_rows:
            break
        rows += page_rows
    if not rows:
        raise RuntimeError('east money returned no rows for %s' % indicator)
    d = pd.DataFrame(rows)
    d['date'] = pd.to_datetime(d['REPORT_DATE']).dt.normalize()
    d['value'] = pd.to_numeric(d['INDICATOR_VALUE'])
    return (d[['date', 'value']].drop_duplicates('date')
            .sort_values('date').reset_index(drop=True))


def fetch_baltic_extension():
    print('\n[2/5] Baltic extension past 2019 (East Money mirror)')
    if cached('baltic_extension'):
        return
    base = os.path.join(RAW, 'baltic_indices.parquet')
    if not os.path.exists(base):
        print('  SKIP  fetch the licensed Mendeley copy first')
        return
    lic = pd.read_parquet(base)
    lic['date'] = pd.to_datetime(lic['date']).dt.normalize()
    cut = lic['date'].max()

    out = {}
    for col, indicator in EASTMONEY_IDS.items():
        try:
            feed = _eastmoney(indicator)
        except Exception as exc:
            print('  FAIL %-10s %s' % (col, exc))
            return
        # Validate on the overlap BEFORE keeping anything. A mirror that
        # cannot reproduce the licensed copy is a different series, and
        # splicing it would put a break in the middle of the target.
        j = lic[['date', col]].merge(
            feed.rename(columns={'value': 'feed'}), on='date', how='inner')
        if j.empty:
            print('  FAIL %-10s no overlapping days to validate against' % col)
            return
        gap = ((j[col] - j['feed']).abs()
               / j[col].abs().clip(lower=1.0))
        corr = float(j[col].corr(j['feed']))
        p99 = float(gap.quantile(0.99))
        exact = float(((j[col] - j['feed']).abs() < JOIN_TOLERANCE).mean())
        if not (corr == corr) or corr < MIN_CORRELATION:
            print('  FAIL %-10s correlates %.4f with the licensed copy - '
                  'that is a different series' % (col, corr))
            return
        if p99 > MAX_P99_REL_GAP:
            print('  FAIL %-10s disagrees by %.2f%% at the 99th percentile'
                  % (col, p99 * 100))
            return
        # And the join itself must be continuous: the last licensed day
        # has to match, or the two halves are on different bases.
        edge = j[j['date'] == cut]
        if edge.empty or abs(float(edge[col].iloc[0])
                             - float(edge['feed'].iloc[0])) > JOIN_TOLERANCE:
            print('  FAIL %-10s the two sources disagree on %s, the join day'
                  % (col, cut.date()))
            return
        print('  ok   %-10s corr %.4f  p99 gap %.2f%%  (%.1f%% of %d days '
              'byte-identical)' % (col, corr, p99 * 100, exact * 100, len(j)))
        out[col] = feed.set_index('date')['value']

    ext = pd.DataFrame(out)
    ext = ext[ext.index > cut].dropna(how='all').reset_index()
    if ext.empty:
        print('  FAIL the mirror carries nothing past %s' % cut.date())
        return
    report('baltic_extension', ext)


# ---------------------------------------------------------------- yahoo
def fetch_yahoo():
    print('\n[3/5] Market data (Yahoo Finance)')
    import yfinance as yf
    for tk, name in TICKERS.items():
        if cached(name):
            continue
        try:
            d = yf.download(tk, start=START, end=END,
                            progress=False, auto_adjust=True)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if d.empty:
                print('  FAIL %-16s %s returned no rows' % (name, tk))
                continue
            d = d[['Close', 'Volume']].reset_index()
            d.columns = ['date', name, name + '_vol']
            report(name, d)
        except Exception as e:
            print('  FAIL %-16s %s' % (name, e))


# ------------------------------------------------------------ portwatch
def fetch_portwatch():
    print('\n[4/5] Port calls (IMF PortWatch)')
    fields = ('date,portid,portname,portcalls,portcalls_dry_bulk,'
              'import,export,import_dry_bulk,export_dry_bulk')
    for pid, name in PORTS.items():
        key = 'port_' + name
        if cached(key):
            continue
        rows, offset = [], 0
        try:
            while True:
                r = requests.get(PORTWATCH, timeout=90, params={
                    'where': "portid='%s'" % pid,
                    'outFields': fields,
                    'returnGeometry': 'false',
                    'f': 'json',
                    'resultOffset': offset,
                    'resultRecordCount': PAGE,
                })
                j = r.json()
                if 'error' in j:
                    print('  FAIL %-16s %s' % (key, j['error'].get('message')))
                    rows = []
                    break
                feats = j.get('features', [])
                rows += [f['attributes'] for f in feats]
                # The server caps a page at its own maxRecordCount, which is
                # below anything we ask for, so "short page means done" is
                # wrong - it silently truncated every port at 1000 rows.
                # ArcGIS flags a truncated page explicitly; trust that.
                if not j.get('exceededTransferLimit', False) or not feats:
                    break
                offset += len(feats)
                time.sleep(0.3)
            if not rows:
                print('  FAIL %-16s no rows for %s' % (key, pid))
                continue
            d = pd.DataFrame(rows)
            # PortWatch returns epoch milliseconds, UTC.
            d['date'] = pd.to_datetime(d['date'], unit='ms').dt.normalize()
            d = d.sort_values('date').reset_index(drop=True)
            report(key, d)
        except Exception as e:
            print('  FAIL %-16s %s' % (key, e))


# -------------------------------------------------------------- weather
def fetch_weather():
    print('\n[5/5] Weather at discharge ports (Open-Meteo)')
    for name, (lat, lon) in WEATHER_SITES.items():
        key = 'wx_' + name
        if cached(key):
            continue
        try:
            r = requests.get('https://archive-api.open-meteo.com/v1/archive',
                             timeout=90, params={
                                 'latitude': lat, 'longitude': lon,
                                 'start_date': START,
                                 'end_date': '2026-09-01',
                                 'daily': ('precipitation_sum,'
                                           'wind_speed_10m_max,'
                                           'wind_gusts_10m_max'),
                                 'timezone': 'Asia/Kolkata',
                             })
            j = r.json()
            if 'daily' not in j:
                print('  FAIL %-16s %s' % (key, j.get('reason', r.status_code)))
                continue
            d = pd.DataFrame(j['daily'])
            d.columns = ['date', 'rain_mm', 'wind_max', 'gust_max']
            d['date'] = pd.to_datetime(d['date'])
            report(key, d)
        except Exception as e:
            print('  FAIL %-16s %s' % (key, e))


if __name__ == '__main__':
    os.makedirs(RAW, exist_ok=True)
    print('=' * 64)
    print('  FETCHING RAW DATA' + ('  (--force)' if FORCE else ''))
    print('=' * 64)
    fetch_baltic()
    fetch_baltic_extension()
    fetch_yahoo()
    fetch_portwatch()
    fetch_weather()
    print('\n' + '=' * 64)
    files = sorted(f for f in os.listdir(RAW) if f.endswith('.parquet'))
    print('  %d parquet files in %s' % (len(files), RAW))
    print('=' * 64)

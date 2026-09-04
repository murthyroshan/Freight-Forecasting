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
    # name in the average buys nothing at ~202 effective observations.
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
WEATHER_SITES = {
    'paradip':       (20.26, 86.67),
    'visakhapatnam': (17.68, 83.21),
    'haldia':        (22.03, 88.08),
    'dhamra':        (20.78, 86.97),
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
    print('\n[1/4] Baltic indices (Mendeley, CC BY 4.0)')
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


# ---------------------------------------------------------------- yahoo
def fetch_yahoo():
    print('\n[2/4] Market data (Yahoo Finance)')
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
    print('\n[3/4] Port calls (IMF PortWatch)')
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
    print('\n[4/4] Weather at discharge ports (Open-Meteo)')
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
    fetch_yahoo()
    fetch_portwatch()
    fetch_weather()
    print('\n' + '=' * 64)
    files = sorted(f for f in os.listdir(RAW) if f.endswith('.parquet'))
    print('  %d parquet files in %s' % (len(files), RAW))
    print('=' * 64)

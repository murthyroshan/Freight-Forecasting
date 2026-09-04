"""
Builds the modelling panel for Module A (Capesize rate direction).

Design decisions, and why:

  Target is the 5-business-day FORWARD RETURN of the Capesize index,
  not the level. Capesize ranges 92 -> 4438 in this sample with a 53%
  coefficient of variation; a level model scores well by predicting
  "about the same as yesterday" and is useless for a timing decision.
  A forward return also makes leakage structural rather than a thing
  we have to remember - the target is strictly in the future, so any
  feature built from data up to and including t is legitimate.

  Capesize specifically, because SAIL lifts coking coal from Hay Point,
  Gladstone and US east coast in 150-180k t parcels. That is Capesize
  work. It is also the only index whose returns track BDRY strongly
  enough (weekly r = +0.671) for the live proxy to be defensible.

  No PortWatch features. PortWatch starts 2019-01-01 and the Baltic
  series ends 2019-07-31 - 211 calendar days of overlap. Congestion
  belongs to Module B, which is deterministic.

Run:  python build_panel.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import paths  # noqa: E402

import numpy as np
import pandas as pd

RAW = paths.RAW
PROC = paths.PROCESSED

HORIZON = 5      # business days ahead - one chartering week


def load(name):
    return pd.read_parquet(os.path.join(RAW, name + '.parquet'))


def ret(s, n):
    """n-period LOG return, ending at t. Uses no future data.

    Log for the same reason as the target: it keeps a move off a small
    base on the same scale as a move off a large one, so 2019 does not
    dominate every fitted coefficient."""
    return np.log(s / s.shift(n))


def build():
    bal = load('baltic_indices').set_index('date').sort_index()

    # Market series, aligned onto Baltic's trading calendar. Baltic
    # publishes on London business days; Yahoo series have their own
    # holidays, so reindex-then-ffill. ffill only ever carries a stale
    # PAST value forward, never a future one.
    mkt = {}
    for n in ['brent', 'copper', 'dxy', 'sp500', 'usdinr',
              'sblk', 'dsx', 'nmm', 'bhp', 'rio', 'whc']:
        s = load(n).set_index('date')[n].sort_index()
        mkt[n] = s.reindex(bal.index, method='ffill')
    mkt = pd.DataFrame(mkt, index=bal.index)

    df = pd.DataFrame(index=bal.index)
    cape = bal['capesize']

    # --- target ------------------------------------------------------
    # LOG return from close of t to close of t+5. Strictly future.
    #
    # Log rather than simple: Capesize bottomed at 92 on 2019-04-02 in
    # the wake of the Vale Brumadinho dam collapse, which wiped out
    # Brazilian iron ore exports. Those prints are real, not corrupt, so
    # they stay - but a simple return off a denominator of 92 gives
    # +311% and skew +3.1 / kurtosis +25.6, and RMSE would then be
    # decided by about four days in 2019. Logs give skew +0.5 and
    # kurtosis +3.2 over the identical rows.
    df['y'] = np.log(cape.shift(-HORIZON) / cape)

    # --- tier 1: the target's own dynamics ---------------------------
    # Freight is strongly autocorrelated and strongly mean-reverting;
    # these carry most of the signal that exists.
    df['cape_ret_1'] = ret(cape, 1)
    df['cape_ret_5'] = ret(cape, 5)
    df['cape_ret_21'] = ret(cape, 21)
    # Distance from a 63-day mean, in that window's own std units.
    m63 = cape.rolling(63).mean()
    s63 = cape.rolling(63).std()
    df['cape_z_63'] = (cape - m63) / s63
    df['cape_vol_21'] = ret(cape, 1).rolling(21).std()

    # --- tier 2: fleet structure -------------------------------------
    # Capesize/Panamax is the classic tightness spread: when Capes are
    # scarce, cargo splits down into Panamaxes and the ratio compresses.
    df['cape_pmx_ratio'] = np.log(cape / bal['panamax'])
    df['pmx_ret_5'] = ret(bal['panamax'], 5)
    df['smx_ret_5'] = ret(bal['supramax'], 5)

    # --- tier 3: cost side -------------------------------------------
    df['brent_ret_5'] = ret(mkt['brent'], 5)
    df['brent_ret_21'] = ret(mkt['brent'], 21)

    # --- tier 4: demand side -----------------------------------------
    df['copper_ret_21'] = ret(mkt['copper'], 21)
    df['dxy_ret_21'] = ret(mkt['dxy'], 21)
    df['sp500_ret_21'] = ret(mkt['sp500'], 21)
    # Miners ship the cargo; their tape is a cheap proxy for volumes.
    miners = pd.concat([ret(mkt[c], 21) for c in ['bhp', 'rio', 'whc']],
                       axis=1).mean(axis=1)
    df['miners_ret_21'] = miners

    # --- tier 5: owners price the rate forward ------------------------
    owners = pd.concat([ret(mkt[c], 5) for c in ['sblk', 'dsx', 'nmm']],
                       axis=1).mean(axis=1)
    df['owners_ret_5'] = owners

    # --- tier 6: seasonality -----------------------------------------
    # Sine/cosine rather than a month integer: December and January are
    # adjacent, 12 and 1 are not.
    doy = df.index.dayofyear
    df['seas_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['seas_cos'] = np.cos(2 * np.pi * doy / 365.25)

    # Keep the raw level for reporting/plots only - dropped before fit.
    df['capesize_level'] = cape

    # Warm-up rows (63-day window) and the final HORIZON rows (no target
    # yet) are dropped rather than imputed.
    df = df.dropna()
    return df


if __name__ == '__main__':
    os.makedirs(PROC, exist_ok=True)
    df = build()
    feats = [c for c in df.columns if c not in ('y', 'capesize_level')]
    df.to_parquet(os.path.join(PROC, 'panel.parquet'))

    print('=' * 64)
    print('  PANEL BUILT')
    print('=' * 64)
    print('  rows      %d' % len(df))
    print('  range     %s -> %s' % (df.index.min().date(), df.index.max().date()))
    print('  features  %d' % len(feats))
    print('  horizon   %d business days' % HORIZON)
    # Overlapping windows mean rows are not independent. State the
    # honest effective count rather than quoting the raw row count.
    print('  effective %d independent windows (rows / horizon)'
          % (len(df) // HORIZON))
    print('\n  target y = %d-day forward Capesize return' % HORIZON)
    print('    mean   %+.4f' % df['y'].mean())
    print('    std     %.4f' % df['y'].std())
    print('    min    %+.4f' % df['y'].min())
    print('    max    %+.4f' % df['y'].max())
    print('    up-weeks %.1f%%' % ((df['y'] > 0).mean() * 100))
    print('    skew   %+.2f   kurtosis %+.2f' % (df['y'].skew(), df['y'].kurt()))

    print('\n  feature correlation with y:')
    for c in feats:
        print('    %-18s %+.3f' % (c, df[c].corr(df['y'])))
    print('=' * 64)

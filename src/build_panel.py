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

  The Capesize series is SPLICED. The licensed Mendeley copy runs to
  2019-07-31; past that the series continues from a public mirror of
  the same Baltic indices, which fetch_data.py validates against the
  licensed copy before writing. See baltic() below.

  No PortWatch features. PortWatch starts 2019-01-01, and although the
  spliced Baltic series now reaches today, congestion belongs to
  Module B, which is deterministic and needs no fitting.

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
    # Same guard as the target: a non-positive level has no log return.
    s = s.where(s > 0) if (s <= 0).any() else s
    return np.log(s / s.shift(n))


def baltic(licensed_only=False):
    """The Capesize/Panamax/Supramax series, spliced.

    `licensed_only` stops at the Mendeley copy and never appends the
    mirror. It exists because the two halves support two different
    models: the licensed years are cleaner and score better, and for
    the internal demo the replay uses that model where it can and the
    extended one after. Passing the flag is the only way to get the
    short series - there is no silent fallback that could produce one
    by accident.

    The licensed Mendeley copy runs to 2019-07-31 and is used unchanged
    up to that date. Past it the series continues from the East Money
    mirror, which src/fetch_data.py refuses to write unless it
    reproduces the licensed copy on the overlapping days.

    The join only ever APPENDS - no licensed value is overwritten - so
    the two halves cannot disagree about a day they both cover.
    """
    lic = load('baltic_indices').set_index('date').sort_index()
    cols = ['capesize', 'panamax', 'supramax']
    if licensed_only:
        return lic[cols]
    # A missing extension must be LOUD. Falling back silently gives a
    # 2019-only panel that looks perfectly normal - same columns, no
    # error - and every number downstream would quietly describe a
    # different model from the one the documentation describes.
    ext_path = os.path.join(RAW, 'baltic_extension.parquet')
    if not os.path.exists(ext_path):
        print('  WARNING no baltic_extension.parquet - the panel will stop '
              'at %s' % lic.index.max().date())
        print('          Run: python -m src.fetch_data')
        return lic[cols]
    ext = pd.read_parquet(ext_path).set_index('date').sort_index()
    ext = ext[ext.index > lic.index.max()]
    if ext.empty:
        print('  WARNING baltic_extension.parquet carries nothing past %s - '
              'the panel will stop there' % lic.index.max().date())
        return lic[cols]
    out = pd.concat([lic[cols], ext[cols]]).sort_index()
    # A splice that introduces a jump would put a break in the middle of
    # the target. Measured: the largest move within a week of the join is
    # 3.4%, against a 3.2% typical daily move - i.e. nothing unusual.
    if out.index.duplicated().any():
        raise ValueError('the splice produced duplicate dates')
    return out


def _frame(licensed_only=False):
    """Every column, before anything is dropped.

    Split out of build() so a forecast for a day that has no outcome yet
    runs literally the same feature code as training, rather than a copy
    of it that can drift. build() is this plus the dropna it always did.
    """
    bal = baltic(licensed_only)

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
    # The index went NEGATIVE for the first time in early 2020 - 44
    # sessions between 2020-01-31 and 2020-05-14, bottoming at -372,
    # because the Capesize basis is a timecharter equivalent and a TCE
    # can go below zero when the market collapses. A log return is
    # undefined there, so those days are excluded rather than patched.
    #
    # It costs 12 rows of 3,296, and every alternative target we tested
    # on the extended series scored WORSE: simple returns -2.3%, an
    # offset log -2.5%, and a volatility-standardised difference -21.9%,
    # against +5.5% for the log return kept here.
    pos = cape.where(cape > 0)
    df['y'] = np.log(pos.shift(-HORIZON) / pos)

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
    df['cape_pmx_ratio'] = np.log(pos / bal['panamax'].where(bal['panamax'] > 0))
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

    return df


FEATURE_EXCLUDE = ('y', 'capesize_level')


def feature_names(df):
    """The model's inputs: every column that is not the target or the
    level kept for plotting. Defined once so the live path and the
    training path cannot disagree about what the features are."""
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def build(licensed_only=False):
    """The panel used for fitting and scoring.

    Warm-up rows (the 63-day windows) and the final HORIZON rows (no
    target yet) are dropped rather than imputed.
    """
    return _frame(licensed_only).dropna()


def features_asof(as_of=None):
    """Feature rows for the days that have no outcome yet.

    The target at row t is built from t+HORIZON, so the last few trading
    days always have complete features and no y. build() drops them,
    correctly - they cannot be scored - and that is exactly why nothing
    in this project has ever forecast forward: the only rows that could
    be were the ones being thrown away.

    "No outcome yet" is defined as strictly AFTER the last row that has
    one, never as "y is missing". A non-positive Capesize print also
    produces a missing y, and those sit in the middle of history; a
    forecast built on one would be presented as tomorrow's while
    describing 2020.

    A row whose features are not all present is left out rather than
    imputed. A stale market feed is a reason to say the forecast cannot
    be made, not to quietly fill in yesterday's dollar and call the
    answer current.

    Returns an empty frame - not an error - when the data is too old to
    have a forward window, so a caller can report staleness itself.
    """
    f = _frame()
    have = f['y'].notna()
    if not have.any():
        return f.iloc[0:0]
    fwd = f[f.index > f.index[have][-1]]
    fwd = fwd[fwd[feature_names(f)].notna().all(axis=1)]
    if as_of is not None:
        fwd = fwd[fwd.index <= pd.Timestamp(as_of)]
    return fwd


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
    # Say which half of the series this panel actually used, so a silent
    # fallback to the licensed-only window is visible in the output.
    lic_end = pd.to_datetime(load('baltic_indices')['date']).max()
    if df.index.max() > lic_end:
        print('  source    licensed to %s, then extended to %s'
              % (lic_end.date(), df.index.max().date()))
    else:
        print('  source    LICENSED ONLY - no extension was applied')
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

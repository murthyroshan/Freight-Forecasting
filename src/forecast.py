"""
Module A, part three - the forecast for the days that have not happened.

Everything else in this project scores the past. The panel drops the
last few trading days because their target needs prices that do not
exist yet, which is correct for measuring a method and is exactly why
nothing here has ever answered the only question a charterer actually
asks: what about next week?

Those dropped rows are the answer. They have complete features and no
outcome - which is what a forecast is.

A CURVE, NOT A POINT
--------------------
The evaluated model looks five business days ahead, which is one
calendar week. Asking it about seven or ten days means asking a
different model, so one is fitted per horizon and each is validated on
its own folds with its own purge gap. A ten-day target purged by five
would leak half of itself across the fold boundary.

They do not perform equally, and the curve is more honest than a single
number would be: the near horizons are easy because freight is
autocorrelated, and the edge decays as the horizon lengthens. Publishing
all of them lets a reader see the decay instead of being handed the most
flattering point.

WHAT MAKES THIS THE RISKIEST NUMBER ON THE PAGE
-----------------------------------------------
Every other figure this repository publishes can be checked against an
outcome that has already happened. A forward forecast cannot, which
makes it the one place where a confident presentation is not disciplined
by anything. Three things are therefore attached to it and none is
optional:

  the interval    conformal, from residuals on a calibration block the
                  model never fitted, so the width is earned rather than
                  assumed;
  the strength    where this week's call sits in the distribution of
                  every call the model has made. A weak signal that
                  happens to be positive should not read like a strong
                  one;
  the record      how the model has actually done this year. In a year
                  it is losing in, a forward call needs to say so
                  beside itself, not three screens away.

Run:  python -m src.forecast
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src import build_panel as bp, paths, train_model as tm

# Every trading day out to a calendar month. 5 is the evaluated horizon
# and one week; 21 trading days is about a month. Each gets its own
# model and its own folds - the booking calendar needs a call for every
# day it colours, and interpolating between horizons it never fitted
# would be inventing the days in between, which is the whole problem
# with the version this idea came from.
#
# Longer is not worse here, which is counter-intuitive and worth saying:
# 21 days scores better than 5 (62.9% against 61.7%) because freight
# cycles are slow and a month of trend is easier to call than a week of
# noise. It stops at a month because the honest sample does: 21-day
# windows leave about 93 independent observations, and 42-day windows
# only 46, which is too few to publish however good it looks.
HORIZONS = tuple(range(1, 22))

MODEL = 'ridge'          # the model the rest of the dashboard reports
ALPHA = tm.ALPHA         # -> 80% interval, same as everywhere else
MIN_ROWS = 400           # below this, refuse rather than answer thinly


def target(level, horizon):
    """The log return over `horizon` trading days.

    Non-positive levels are masked, not clipped. Capesize is a
    timecharter equivalent and printed below zero for 44 sessions in
    2020; a ratio through one of those is not a return.
    """
    pos = level.where(level > 0)
    return np.log(pos.shift(-horizon) / pos)


def _fit_predict(X, y, x_new, vol, vol_new, horizon):
    """Fit, calibrate and predict once, mirroring the evaluation split.

    The calibration block is carved off the end of history and purged
    from the fitting block by `horizon` rows, exactly as folds() does.
    Without the purge the last targets the model fits on are built from
    prices inside the block used to size its interval, and the interval
    comes out too narrow.
    """
    n = len(y)
    cal_lo = int(n * 0.85)
    fit_hi = cal_lo - horizon
    if fit_hi < MIN_ROWS or n - cal_lo < 30:
        return None

    sc = StandardScaler().fit(X[:fit_hi])
    rg = Ridge(alpha=10.0).fit(sc.transform(X[:fit_hi]), y[:fit_hi])
    point = float(rg.predict(sc.transform(x_new.reshape(1, -1)))[0])

    # Locally weighted split conformal, the same variant train_model
    # ships: residuals scaled by a volatility estimate known at t, then
    # rescaled at the point being predicted.
    cal_pred = rg.predict(sc.transform(X[cal_lo:]))
    v_cal = np.clip(vol[cal_lo:], 1e-4, None)
    v_new = float(np.clip(vol_new, 1e-4, None))
    q = tm.conformal_quantile(np.abs(y[cal_lo:] - cal_pred) / v_cal, ALPHA)
    half = q * v_new
    return point, half, int(fit_hi), int(n - cal_lo)


def _validate(X, y, horizon):
    """Walk-forward scoring at this horizon, on its own purge gap.

    Returns the out-of-sample predictions as well as the summary,
    because the strength of today's call is measured against them.
    """
    n = len(y)
    pred = np.full(n, np.nan)
    for _, te_lo, te_hi, tr_hi, cal_lo, fit_hi in tm.folds(n, horizon):
        if fit_hi < MIN_ROWS:
            continue
        sl = slice(te_lo, te_hi)
        sc = StandardScaler().fit(X[:fit_hi])
        pred[sl] = Ridge(alpha=10.0).fit(
            sc.transform(X[:fit_hi]), y[:fit_hi]).predict(sc.transform(X[sl]))
    m = np.isfinite(pred)
    if m.sum() < 100:
        return None, None
    yy, pp = y[m], pred[m]
    rz = float(np.sqrt(np.mean(yy ** 2)))
    rp = float(np.sqrt(np.mean((yy - pp) ** 2)))
    up = float((yy > 0).mean())
    return pred, {
        'n_scored': int(m.sum()),
        'n_effective': int(m.sum() // horizon),
        'direction_pct': float((np.sign(pp) == np.sign(yy)).mean()) * 100,
        'base_rate_pct': float(max(up, 1 - up)) * 100,
        'skill_pct': (1 - rp / rz) * 100 if rz else float('nan'),
    }


def _trading_days_ahead(index, as_of, horizon):
    """The date `horizon` trading days after `as_of`.

    Baltic publishes on London business days. The future half of that
    calendar is not knowable here, so weekends are stepped over and
    holidays are not - which is stated in the payload rather than
    quietly rounded away.
    """
    d = pd.Timestamp(as_of)
    step = 0
    while step < horizon:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            step += 1
    return d


def build(horizons=HORIZONS):
    """The forward curve, or None when there is nothing to forecast from."""
    frame = bp._frame()
    fwd = bp.features_asof()
    if not len(fwd):
        return None
    feats = bp.feature_names(frame)
    as_of = fwd.index[-1]
    x_new = fwd[feats].to_numpy(dtype=float)[-1]
    vol_new = float(fwd['cape_vol_21'].iloc[-1])
    level = float(fwd['capesize_level'].iloc[-1])

    rows = []
    for h in horizons:
        yh = target(frame['capesize_level'], h)
        keep = frame[feats].notna().all(axis=1) & yh.notna()
        if int(keep.sum()) < MIN_ROWS:
            continue
        X = frame.loc[keep, feats].to_numpy(dtype=float)
        y = yh[keep].to_numpy(dtype=float)
        vol = frame.loc[keep, 'cape_vol_21'].to_numpy(dtype=float)

        oos, stats = _validate(X, y, h)
        got = _fit_predict(X, y, x_new, vol, vol_new, h)
        if got is None or stats is None:
            continue
        point, half, n_fit, n_cal = got

        # Where does today's call sit among every call this model has
        # made? A weak signal that happens to be positive must not read
        # like a strong one, and the comparison is against out-of-sample
        # predictions only - never against the fitted training values,
        # which are always more confident than they deserve.
        past = np.abs(oos[np.isfinite(oos)])
        strength = float((past <= abs(point)).mean()) * 100

        rows.append({
            'horizon_days': int(h),
            'target_date': str(_trading_days_ahead(frame.index, as_of,
                                                   h).date()),
            'expected_move_pct': (np.exp(point) - 1) * 100,
            'lo_pct': (np.exp(point - half) - 1) * 100,
            'hi_pct': (np.exp(point + half) - 1) * 100,
            'direction': 'up' if point > 0 else 'down' if point < 0 else 'flat',
            'strength_pct': strength,
            'stronger_than_usual': bool(strength >= 50.0),
            'n_fitted': n_fit, 'n_calibrated': n_cal,
            'validation': stats,
        })

    if not rows:
        return None

    # The current year's record, sat next to the call rather than three
    # screens away. A forward number in a year the model is losing in
    # has to carry that with it.
    metrics_path = os.path.join(paths.MODELS, 'metrics.json')
    recent = None
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding='utf-8') as fh:
            reg = json.load(fh).get('models', {}).get('regimes', {})
        for r in reg.get('rows', []):
            if r['period'] == str(as_of.year):
                recent = r
                break

    stale = int((pd.Timestamp.now().normalize() - as_of).days)
    return {
        'as_of': str(as_of.date()),
        'generated': pd.Timestamp.now('UTC').isoformat(),
        'data_age_days': stale,
        'capesize_index': level,
        'model': MODEL,
        'interval_pct': int(round((1 - ALPHA) * 100)),
        'unit': 'percent change in the Capesize index',
        'calendar_note': ('target dates step over weekends but not '
                          'exchange holidays, so a date may land one '
                          'session late'),
        'basis': ('fitted only on days before the forecast is made; the '
                  'interval is conformal, calibrated on a block the '
                  'model never fitted and purged from it by the horizon'),
        'current_year': recent,
        'horizons': rows,
    }


if __name__ == '__main__':
    out = build()
    if out is None:
        raise SystemExit('  nothing to forecast from - run '
                         'python -m src.fetch_data && python -m src.build_panel')

    print('=' * 76)
    print('  THE WEEK AHEAD - as of %s, Capesize %s'
          % (out['as_of'], format(int(out['capesize_index']), ',')))
    print('=' * 76)
    if out['data_age_days'] >= 2:
        print('  STALE: the newest price is %d days old. Re-run '
              'python -m src.fetch_data' % out['data_age_days'])
    elif out['data_age_days'] == 1:
        print('  (newest close is yesterday, which is normal before '
              'today publishes)')
    print('  %-8s %-12s %9s %20s %9s' %
          ('ahead', 'target', 'move',
           '%d%% interval' % round((1 - ALPHA) * 100), 'strength'))
    for r in out['horizons']:
        print('  %-8s %-12s %+8.2f%% %9.1f%% to %+6.1f%% %8.0f%%'
              % ('%dd' % r['horizon_days'], r['target_date'],
                 r['expected_move_pct'], r['lo_pct'], r['hi_pct'],
                 r['strength_pct']))
    print('')
    print('  how each horizon has actually scored, walk-forward:')
    print('  %-8s %8s %10s %9s %9s'
          % ('ahead', 'scored', 'direction', 'base rate', 'skill%'))
    for r in out['horizons']:
        v = r['validation']
        print('  %-8s %8d %9.1f%% %8.1f%% %+8.2f'
              % ('%dd' % r['horizon_days'], v['n_scored'],
                 v['direction_pct'], v['base_rate_pct'], v['skill_pct']))
    cy = out['current_year']
    if cy:
        print('')
        print('  %s so far: direction %.1f%% against a %.1f%% base rate, '
              'skill %+.2f%%' % (cy['period'], cy['direction_pct'],
                                 cy['base_rate_pct'], cy['skill_pct']))
        if cy['skill_pct'] < 0 or cy['direction_pct'] <= cy['base_rate_pct']:
            print('  This is a year the model is NOT beating. The call above')
            print('  is published with that beside it, not in spite of it.')

    dest = os.path.join(paths.MODELS, 'live_forecast.json')
    with open(dest, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('')
    print('  wrote %s' % dest)

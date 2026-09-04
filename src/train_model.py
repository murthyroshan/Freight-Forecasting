"""
Module A: Capesize 5-day-ahead direction, evaluated honestly.

Evaluation protocol, and why each piece is there:

  Expanding-window walk-forward. A random train/test split would let the
  model see 2019 while predicting 2015. Every score below is strictly
  out-of-sample in time.

  A PURGE GAP of HORIZON days between train and test. The target at t
  spans t..t+5, so without a gap the last few training rows overlap the
  first test row's window and the model is scored on data it partly saw.

  Baselines first. "Accuracy" on a return series is a meaningless
  number - you can score 97% on a series that never moves much. What
  matters is whether we beat the alternatives a chartering desk already
  has for free:
      zero      - assume next week's rate equals this week's
      momentum  - the single strongest feature, fitted by OLS
  A model that cannot beat both of those is not worth deploying, and we
  report it as such rather than quoting an impressive-looking RMSE.

  Conformal intervals. Split-conformal residual quantiles give an
  interval with finite-sample coverage guarantees under exchangeability,
  which is far more defensible than a model's own predict_proba.

Run:  python train_model.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import paths  # noqa: E402

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src import build_panel

warnings.filterwarnings('ignore')

PROC = paths.PROCESSED
MODELS = paths.MODELS
HORIZON = build_panel.HORIZON
N_FOLDS = 8
ALPHA = 0.20          # -> 80% prediction interval


CONF_ON = 'ridge'     # intervals wrap whichever model actually wins


def conformal_quantile(residuals, alpha):
    """The split-conformal quantile, with its finite-sample correction.

    Plain `np.quantile(r, 1 - alpha)` is NOT the conformal quantile. The
    guarantee requires the ceil((n+1)(1-alpha))/n order statistic taken
    from above, which for a calibration block of n is strictly wider. The
    difference is small at n=100 and vanishes as n grows, but without it
    the coverage guarantee is asserted rather than held - measured on the
    exchangeable random-walk null, the uncorrected version sits at 78.0%
    against a nominal 80%.
    """
    r = np.asarray(residuals, dtype=float)
    n = r.size
    if n == 0:
        return float('inf')
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:                      # too few points to bound at this level
        return float(np.max(r))
    return float(np.quantile(r, k / n, method='higher'))


def folds(n):
    """The fold boundaries, as (k, te_lo, te_hi, tr_hi, cal_lo, fit_hi).

    Split out of walk_forward so the purge gap is something a test can
    assert on directly. It could not be before: the boundaries were
    local variables, and inverting the gap so the training window
    overlapped the test window left every test in this repository green.

      fit  [0 : fit_hi)          model fitted here
      gap  HORIZON rows          purge
      cal  [cal_lo : tr_hi)      conformal calibration
      gap  HORIZON rows          purge
      test [te_lo : te_hi)       scored here

    Both gaps are HORIZON wide because the target at row i is a function
    of row i+HORIZON, so without them the last targets of one block are
    computed from rows inside the next.
    """
    start = int(n * 0.40)                     # first fold trains on 40%
    fold_size = (n - start) // N_FOLDS
    out = []
    for k in range(N_FOLDS):
        te_lo = start + k * fold_size
        te_hi = n if k == N_FOLDS - 1 else te_lo + fold_size
        tr_hi = te_lo - HORIZON               # the purge gap
        if tr_hi < 100:
            continue
        # Carve a calibration tail off the training block for conformal.
        cal_lo = int(tr_hi * 0.85)
        fit_hi = cal_lo - HORIZON
        out.append((k, te_lo, te_hi, tr_hi, cal_lo, fit_hi))
    return out


def walk_forward(df, feats, conformal='local'):
    """Expanding-window folds with a purge gap. Returns per-row
    out-of-sample predictions for every model.

    `conformal` selects the interval method: 'local' scales residuals by
    cape_vol_21 (what ships), 'plain' uses one global quantile. The
    second exists so the claim that local weighting helps is a number
    this repository computes, on these exact folds, rather than a figure
    written into a comment where it can drift.
    """
    if conformal not in ('local', 'plain'):
        raise ValueError("conformal must be 'local' or 'plain', got %r"
                         % (conformal,))
    n = len(df)
    X = df[feats].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)

    preds = {k: np.full(n, np.nan) for k in
             ('zero', 'momentum', 'ridge', 'lgbm')}
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    fold_id = np.full(n, -1)
    importance = np.zeros(len(feats))
    mom_i = feats.index('cape_ret_1')
    vol_i = feats.index('cape_vol_21')
    fold_rows = []

    for k, te_lo, te_hi, tr_hi, cal_lo, fit_hi in folds(n):

        Xtr, ytr = X[:fit_hi], y[:fit_hi]
        Xcal, ycal = X[cal_lo:tr_hi], y[cal_lo:tr_hi]
        Xte = X[te_lo:te_hi]
        sl = slice(te_lo, te_hi)
        fold_id[sl] = k

        # --- baseline 1: no change ---------------------------------
        preds['zero'][sl] = 0.0

        # --- baseline 2: momentum, one feature, OLS ----------------
        mom = LinearRegression().fit(Xtr[:, [mom_i]], ytr)
        preds['momentum'][sl] = mom.predict(Xte[:, [mom_i]])

        # --- ridge over all features -------------------------------
        sc = StandardScaler().fit(Xtr)
        rg = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr)
        preds['ridge'][sl] = rg.predict(sc.transform(Xte))
        ridge_cal = rg.predict(sc.transform(Xcal)) if len(Xcal) else None

        # --- lightgbm ----------------------------------------------
        # Deliberately tiny. There are ~656 independent windows here,
        # not 3,284 rows; a normal-sized GBM memorises this instantly.
        gb = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.02, num_leaves=4,
            max_depth=3, min_child_samples=40, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.6,
            reg_lambda=5.0, verbose=-1, random_state=42)
        gb.fit(Xtr, ytr)
        preds['lgbm'][sl] = gb.predict(Xte)
        importance += gb.feature_importances_
        best_cal = ridge_cal if CONF_ON == 'ridge' else gb.predict(Xcal)

        # --- locally adaptive split conformal -----------------------
        # Plain split conformal on this model gives 81.0% coverage for a
        # nominal 80% (computed below, not asserted here). Freight vol
        # clusters hard, so one global residual
        # quantile is too narrow in stressed regimes and too wide in calm
        # ones. Scale each residual by a volatility estimate known at t
        # (cape_vol_21, a trailing window), then rescale on the test
        # side - the Lei/Romano locally-weighted variant.
        if len(Xcal) > 20:
            if conformal == 'local':
                v_cal = np.clip(X[cal_lo:tr_hi, vol_i], 1e-4, None)
                v_te = np.clip(X[te_lo:te_hi, vol_i], 1e-4, None)
            else:
                v_cal = np.ones(tr_hi - cal_lo)
                v_te = np.ones(te_hi - te_lo)
            resid = np.abs(ycal - best_cal) / v_cal
            q = conformal_quantile(resid, ALPHA)
            lo[sl] = preds[CONF_ON][sl] - q * v_te
            hi[sl] = preds[CONF_ON][sl] + q * v_te

        # per-fold scoring, so a single lucky fold cannot carry the
        # aggregate without it being visible
        fold_rows.append({
            'fold': k, 'n': te_hi - te_lo,
            'start': str(df.index[te_lo].date()),
            'end': str(df.index[te_hi - 1].date()),
            **{mk: float(np.sqrt(np.mean((y[sl] - preds[mk][sl]) ** 2)))
               for mk in ('zero', 'momentum', 'ridge', 'lgbm')}})

    return preds, lo, hi, fold_id, importance, fold_rows


def score(y, p):
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    # Direction is only meaningful where the model actually commits.
    nz = p != 0
    da = float((np.sign(p[nz]) == np.sign(y[nz])).mean()) if nz.any() else float('nan')
    return rmse, mae, da


if __name__ == '__main__':
    os.makedirs(MODELS, exist_ok=True)
    df = build_panel.build()
    feats = [c for c in df.columns if c not in ('y', 'capesize_level')]

    preds, lo, hi, fold_id, imp, fold_rows = walk_forward(df, feats)
    m = fold_id >= 0
    y = df['y'].to_numpy(dtype=float)[m]

    print('=' * 70)
    print('  MODULE A - CAPESIZE %d-DAY DIRECTION, WALK-FORWARD' % HORIZON)
    print('=' * 70)
    print('  panel      %d rows, %s -> %s'
          % (len(df), df.index.min().date(), df.index.max().date()))
    print('  scored     %d out-of-sample rows across %d folds'
          % (m.sum(), N_FOLDS))
    print('  effective  ~%d independent windows (scored / horizon)'
          % (m.sum() // HORIZON))
    print('  purge gap  %d days between train and test' % HORIZON)

    base_rmse = score(y, preds['zero'][m])[0]
    print('\n  %-10s %8s %8s %8s %8s' %
          ('model', 'RMSE', 'MAE', 'skill%', 'dir%'))
    print('  ' + '-' * 46)
    results = {}
    for k in ('zero', 'momentum', 'ridge', 'lgbm'):
        r, a, d = score(y, preds[k][m])
        skill = (1 - r / base_rmse) * 100
        results[k] = {'rmse': r, 'mae': a, 'skill_vs_zero_pct': skill,
                      'direction_pct': None if np.isnan(d) else d * 100}
        print('  %-10s %8.4f %8.4f %+8.2f %8s' %
              (k, r, a, skill, '-' if np.isnan(d) else '%.1f' % (d * 100)))

    # Conformal coverage: does the 80% interval actually contain 80%?
    hasiv = m & np.isfinite(lo)
    if hasiv.any():
        yy = df['y'].to_numpy(dtype=float)[hasiv]
        cov = float(((yy >= lo[hasiv]) & (yy <= hi[hasiv])).mean())
        width = float(np.mean(hi[hasiv] - lo[hasiv]))
        print('\n  conformal interval  target %d%%  actual %.1f%%  '
              'mean width %.3f' % ((1 - ALPHA) * 100, cov * 100, width))
        results['conformal'] = {'target': 1 - ALPHA, 'coverage': cov,
                                'mean_width': width, 'n': int(hasiv.sum()),
                                'wraps': CONF_ON, 'method':
                                'locally-weighted split conformal'}

        # The control for the interval method. Same folds, same model,
        # same corrected quantile - only the local scaling removed - so
        # the difference is attributable to the scaling and to nothing
        # else. Written to the artefact so the README can quote a number
        # a script here printed.
        _, plo, phi, _, _, _ = walk_forward(df, feats, conformal="plain")
        pv = m & np.isfinite(plo)
        if pv.any():
            yp = df['y'].to_numpy(dtype=float)[pv]
            pcov = float(((yp >= plo[pv]) & (yp <= phi[pv])).mean())
            pwid = float(np.mean(phi[pv] - plo[pv]))
            results['conformal']['plain_coverage'] = pcov
            results['conformal']['plain_mean_width'] = pwid
            print('  plain (no local scaling)  actual %.1f%%  '
                  'mean width %.3f' % (pcov * 100, pwid))

    # Is the direction edge real, or 202 coin flips that went our way?
    # Test on the EFFECTIVE count, not the row count - overlapping
    # windows would otherwise inflate significance about 5x.
    from scipy import stats as _st
    n_eff = int(m.sum() // HORIZON)
    # The model tested is chosen by its OUT-OF-SAMPLE direction score and
    # then tested on that same score. Three candidates, so the p-value is
    # Bonferroni-adjusted; without it this is not a clean out-of-sample p.
    candidates = ('momentum', 'ridge', 'lgbm')
    p_best = preds[max(candidates,
                       key=lambda k: results[k]['direction_pct'])][m]
    da = float((np.sign(p_best) == np.sign(y)).mean())
    k_eff = int(round(da * n_eff))
    # The null is the realised up-rate, not 0.5. An "always up" predictor
    # already scores the base rate, so testing against a coin flatters
    # any model on a series that drifts.
    base = float((y > 0).mean())
    raw = float(_st.binomtest(k_eff, n_eff, base,
                              alternative='greater').pvalue)
    pval = min(1.0, raw * len(candidates))
    print('\n  direction %.1f%% on ~%d independent windows -> p = %.4f %s'
          % (da * 100, n_eff, pval,
             '(significant)' if pval < 0.05 else '(NOT significant)'))
    results['direction_test'] = {'rate': da, 'n_effective': n_eff,
                                 'base_rate': base,
                                 'p_value': pval, 'p_value_raw': raw,
                                 'n_candidates': len(candidates),
                                 'significant': bool(pval < 0.05)}

    print('\n  per-fold RMSE (is the win consistent?):')
    print('    %-4s %-11s %-11s %7s %7s %7s %7s'
          % ('fold', 'start', 'end', 'zero', 'mom', 'ridge', 'lgbm'))
    wins = 0
    for fr in fold_rows:
        better = fr['ridge'] < fr['zero']
        wins += better
        print('    %-4d %-11s %-11s %7.4f %7.4f %7.4f %7.4f  %s'
              % (fr['fold'], fr['start'], fr['end'], fr['zero'],
                 fr['momentum'], fr['ridge'], fr['lgbm'],
                 'ok' if better else 'LOSES'))
    print('    ridge beats no-change in %d of %d folds'
          % (wins, len(fold_rows)))
    results['folds'] = fold_rows
    results['folds_won'] = wins

    print('\n  LightGBM feature importance (summed over folds):')
    order = np.argsort(imp)[::-1]
    for i in order[:8]:
        print('    %-18s %5.0f' % (feats[i], imp[i]))

    best = min(('momentum', 'ridge', 'lgbm'),
               key=lambda k: results[k]['rmse'])
    beats_zero = results[best]['rmse'] < base_rmse
    beats_mom = results[best]['rmse'] < results['momentum']['rmse']

    print('\n  ' + '-' * 66)
    print('  VERDICT')
    if not beats_zero:
        print('    Nothing beats assuming no change. Do not ship a forecast.')
    elif best == 'momentum' or not beats_mom:
        print('    Momentum alone is the best model here. The extra')
        print('    features do not earn their place yet.')
    else:
        print('    %s beats both the no-change and momentum baselines.'
              % best)
    print('  ' + '-' * 66)
    print('=' * 70)

    # Persist so the dashboard can serve real numbers instead of
    # hard-coded ones. Anything the API shows must come from here.
    out = {'horizon_days': HORIZON, 'n_scored': int(m.sum()),
           'n_effective': int(m.sum() // HORIZON), 'n_folds': N_FOLDS,
           'panel_rows': len(df), 'features': feats,
           'train_start': str(df.index.min().date()),
           'train_end': str(df.index.max().date()),
           'target': 'capesize_%dd_log_return' % HORIZON,
           'best_model': best, 'beats_zero': bool(beats_zero),
           'beats_momentum': bool(beats_mom), 'models': results}
    with open(os.path.join(MODELS, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('  wrote %s' % os.path.join(MODELS, 'metrics.json'))

    oos = pd.DataFrame({'y': df['y'][m]}, index=df.index[m])
    for k in preds:
        oos[k] = preds[k][m]
    oos['lo'], oos['hi'] = lo[m], hi[m]
    oos.to_parquet(os.path.join(PROC, 'oos_predictions.parquet'))
    print('  wrote %s' % os.path.join(PROC, 'oos_predictions.parquet'))

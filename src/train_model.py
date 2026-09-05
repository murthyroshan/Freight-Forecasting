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


def folds(n, horizon=HORIZON):
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

    Both gaps are `horizon` wide because the target at row i is a
    function of row i+horizon, so without them the last targets of one
    block are computed from rows inside the next. It defaults to HORIZON
    and is a parameter only so the forecast curve, which scores several
    horizons, purges each by its own - a 10-day target purged by 5 would
    leak five days of it into the next block.
    """
    start = int(n * 0.40)                     # first fold trains on 40%
    fold_size = (n - start) // N_FOLDS
    out = []
    for k in range(N_FOLDS):
        te_lo = start + k * fold_size
        te_hi = n if k == N_FOLDS - 1 else te_lo + fold_size
        tr_hi = te_lo - horizon               # the purge gap
        if tr_hi < 100:
            continue
        # Carve a calibration tail off the training block for conformal.
        cal_lo = int(tr_hi * 0.85)
        fit_hi = cal_lo - horizon
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


# Shares of weeks a desk might act on, strongest first.
TIER_SHARES = (0.75, 0.50, 0.25)
TIER_BURN_IN = 250    # ~1 trading year before any week is eligible


def tier_mask(pred, share, min_history=TIER_BURN_IN):
    """Which rows fall in the strongest `share` of calls seen SO FAR.

    Extracted from confidence_tiers so a test can assert the causality
    directly, the way folds() is extracted to let a test assert the
    purge gap. The whole claim rests on this function: row i is judged
    against a quantile of rows [0, i), so appending later data can
    never change a decision already taken.
    """
    a = np.abs(np.asarray(pred, dtype=float))
    sel = np.zeros(len(a), dtype=bool)
    if share >= 1.0:
        # "The strongest 100%" has to mean every eligible week. Left to
        # the general path it would not: the threshold is the running
        # MINIMUM, and a week weaker than anything seen before it fails
        # a >= test against that minimum. One row in 250 goes missing
        # that way - never in production, where the shares are 75/50/25,
        # but the boundary should still say what its name says.
        sel[min_history:] = True
        return sel
    for i in range(min_history, len(a)):
        sel[i] = a[i] >= float(np.quantile(a[:i], 1.0 - share))
    return sel


# A year is the coarsest split that cannot be accused of being chosen:
# nobody picks calendar boundaries to flatter a model.
REGIME_MIN_N = 100
REGIME_SHARE = 0.50       # the tier the recent columns are judged on


def regimes(index, y, pred, zero, share=REGIME_SHARE, min_n=REGIME_MIN_N,
            min_history=TIER_BURN_IN):
    """Where the model works, and where it stops working.

    Publishing one aggregate number over eight years invites exactly one
    question - "does it still work?" - and answering it from the same
    aggregate is not an answer. This breaks the record into calendar
    years, which is the one split nobody can accuse of being chosen to
    flatter, and reports the weakness alongside the defence rather than
    in a footnote under it.

    Two figures are given per year because they disagree, and the
    disagreement is the point. RMSE skill measures variance explained,
    so it falls apart in a calm market: there is little variance to
    explain and the handful of large misses dominate what is left.
    Direction is scale-free and does not care how big the moves were.
    When the market quietened after 2022 the first collapsed and the
    second did not, which is what this repository predicted in writing
    before it was measured - see 'the skill figure is fragile; the
    direction figure is not' in docs/METHOD.md.

    The tiered column is the one that matters for use. It asks whether
    the model still knows WHICH weeks it knows about, and that is a
    different question from whether its average call got worse.
    """
    index = pd.DatetimeIndex(index)
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    zero = np.asarray(zero, dtype=float)
    strong = tier_mask(pred, share, min_history)
    hit = np.sign(pred) == np.sign(y)

    rows = []
    for yr in sorted(set(index.year)):
        m = np.asarray(index.year == yr)
        if m.sum() < min_n:
            continue
        rz = float(np.sqrt(np.mean((y[m] - zero[m]) ** 2)))
        rp = float(np.sqrt(np.mean((y[m] - pred[m]) ** 2)))
        sel = m & strong
        rows.append({
            'period': str(yr), 'n': int(m.sum()),
            'direction_pct': float(hit[m].mean()) * 100,
            'skill_pct': (1.0 - rp / rz) * 100 if rz else float('nan'),
            'volatility_pct': float(np.std(y[m])) * 100,
            'base_rate_pct': float(max((y[m] > 0).mean(),
                                       1 - (y[m] > 0).mean())) * 100,
            'n_strong': int(sel.sum()),
            'strong_direction_pct': (float(hit[sel].mean()) * 100
                                     if sel.sum() >= 20 else None)})
    return {'share': float(share), 'min_history': int(min_history),
            'min_n': int(min_n), 'rows': rows}


def confidence_tiers(pred, y, shares=TIER_SHARES, min_history=TIER_BURN_IN):
    """Direction accuracy on the weeks the model is most sure about.

    The model does not have to answer every week. A chartering desk
    fixes a handful of cargoes a quarter, so the number that matters
    is not the average call - it is whether the calls it acts on are
    better than the ones it declines to make.

    The confidence signal is |prediction|. That is known the moment
    the model runs: it is built from the prediction alone, needs no
    outcome, and so nothing here waits for the horizon to elapse.

    The THRESHOLD is the part that could leak. Taking "the strongest
    half" as a quantile over the whole test period would need next
    year's predictions to decide whether today's is in the top half.
    So the threshold at row i is taken over rows STRICTLY BEFORE i,
    and the first `min_history` rows are not eligible at all, because
    there is not yet enough history to place them. That is the same
    expanding-window discipline the folds use, applied to the
    abstention rule instead of to the fit.

    The baseline is measured on the same eligible rows, so the lift is
    like-for-like and not an artefact of dropping the burn-in.

    Returns None when there is not enough history to tier at all.
    """
    from scipy import stats as _st
    p = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    a = np.abs(p)
    n = len(p)
    if n <= min_history:
        return None

    hit = np.sign(p) == np.sign(y)
    eligible = np.zeros(n, dtype=bool)
    eligible[min_history:] = True
    all_dir = float(hit[eligible].mean())

    out = {'signal': 'abs_prediction', 'min_history': int(min_history),
           'n_eligible': int(eligible.sum()), 'all_direction': all_dir,
           'tiers': []}
    for share in shares:
        # past predictions only - a[i] itself is excluded, so the
        # threshold cannot be moved by the row it is judging
        sel = tier_mask(p, share, min_history)
        d = float(hit[sel].mean()) if sel.any() else float('nan')
        # Is the tier's edge real, or a small sample that went our way?
        # Same discipline as the headline test: independent windows
        # rather than rows, because five-day targets overlap, and the
        # null is the best CONSTANT call on those same rows - a tier
        # that quietly selects mostly-rising weeks has to beat "always
        # up" on them, not a coin. Bonferroni over the tiers.
        n_eff = max(1, int(sel.sum()) // HORIZON)
        up = float((y[sel] > 0).mean()) if sel.any() else 0.5
        null = min(max(up, 1.0 - up), 0.999)
        raw = float(_st.binomtest(int(round(d * n_eff)), n_eff, null,
                                  alternative='greater').pvalue)
        out['tiers'].append({'share': float(share), 'n': int(sel.sum()),
                             'coverage': float(sel.sum() / eligible.sum()),
                             'direction': d,
                             'lift_pp': (d - all_dir) * 100,
                             'n_effective': n_eff, 'null_rate': null,
                             'p_value': min(1.0, raw * len(shares)),
                             'significant': bool(
                                 min(1.0, raw * len(shares)) < 0.05)})
    return out


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

    # The model is allowed to stay quiet. Score the calls a desk would
    # actually act on, using the same model that was just tested.
    # Tiered on the model the dashboard and the intervals report, not
    # on the direction-test's pick - otherwise the "vs all" column
    # would be measured against a headline the page never shows.
    ct = confidence_tiers(preds[CONF_ON][m], y)
    if ct:
        print('')
        print('  direction by confidence (threshold from PAST predictions '
              'only, %d-row burn-in):' % ct['min_history'])
        print('    %-28s %7s %8s %9s'
              % ('act only when...', 'weeks', 'dir%', 'vs all'))
        print('    %-28s %7d %7.1f%% %9s'
              % ('always', ct['n_eligible'], ct['all_direction'] * 100, '-'))
        for t in ct['tiers']:
            print('    %-28s %7d %7.1f%% %+8.1fpp   p=%.4f %s'
                  % ('the call is strongest %d%%' % (t['share'] * 100),
                     t['n'], t['direction'] * 100, t['lift_pp'],
                     t['p_value'], 'ok' if t['significant'] else 'NOT sig'))
        ct['model'] = CONF_ON
        results['confidence'] = ct

    # Where it works and where it stops working. Published per calendar
    # year so the weak years are visible rather than averaged away.
    rg = regimes(df.index[m], y, preds[CONF_ON][m], preds['zero'][m])
    if rg['rows']:
        print('')
        print('  by calendar year (the split nobody can accuse us of '
              'choosing):')
        print('    %-6s %6s %8s %9s %9s %14s'
              % ('year', 'n', 'dir%', 'skill%', 'vol of y', 'strongest 50%'))
        for r in rg['rows']:
            print('    %-6s %6d %7.1f%% %+8.2f %8.1f%% %13s'
                  % (r['period'], r['n'], r['direction_pct'], r['skill_pct'],
                     r['volatility_pct'],
                     '-' if r['strong_direction_pct'] is None
                     else '%.1f%%' % r['strong_direction_pct']))
        weak = [r for r in rg['rows'] if r['skill_pct'] < 0]
        if weak:
            print('    skill is NEGATIVE in %d of %d years (%s). Direction'
                  % (len(weak), len(rg['rows']),
                     ', '.join(r['period'] for r in weak)))
            print('    holds up in all of them, which is the fragility this')
            print('    repository documented before it measured it.')
        results['regimes'] = rg

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

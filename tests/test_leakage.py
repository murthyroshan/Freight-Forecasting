"""
Leakage guard rail for the panel.

A headline accuracy figure is worthless if the target leaked into a
feature. The classic version of this, and the one these tests are built
around, is a feature list containing

    Coal_Freight_Ratio = Coal_Price / (Freight_Rate + 1)

while the target IS Freight_Rate. Neither Coal_Price nor the ratio looks
like the target on its own, which is why it survives review - but the
pair inverts to it exactly, so the score measures division rather than
forecasting. The same applies to .rolling(7).mean() with no shift, which
on a level target includes the value being predicted.

These tests exist so that class of mistake cannot get in silently. Test 5 is the important one: it does not pattern-match on
feature names, it perturbs the future and checks the past did not move.

Run:  python test_leakage.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import paths  # noqa: E402

import numpy as np
import pandas as pd

from src import build_panel

FAILURES = []


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '  <- ' + detail))
    if not ok:
        FAILURES.append(name)


def main():
    df = build_panel.build()
    feats = [c for c in df.columns if c not in ('y', 'capesize_level')]

    print('\n[1] no feature is a near-copy of the target')
    worst, worst_c = 0.0, None
    for c in feats:
        r = abs(df[c].corr(df['y']))
        if r > worst:
            worst, worst_c = r, c
    check('max |corr| with y is %.3f (%s), threshold 0.95'
          % (worst, worst_c), worst < 0.95,
          'a feature nearly reproduces the target')

    print('\n[2] the panel has no NaN or inf reaching the model')
    sub = df[feats + ['y']]
    check('no NaN', not sub.isna().any().any())
    check('no inf', np.isfinite(sub.to_numpy()).all())

    print('\n[3] target is strictly in the future')
    # y at t must equal the log change from t to t+HORIZON, recomputed
    # here independently of build_panel.
    cape = df['capesize_level']
    manual = np.log(cape.shift(-build_panel.HORIZON) / cape)
    aligned = manual.dropna()
    common = aligned.index.intersection(df.index)
    check('y matches an independent recomputation on %d rows' % len(common),
          np.allclose(df.loc[common, 'y'], aligned.loc[common], atol=1e-12))

    print('\n[4] a target-leaking feature pair is shown to be fatal')
    # Listing BOTH Coal_Price and Coal_Price/(Freight_Rate+1) as features
    # while predicting Freight_Rate is the textbook version of this.
    # Neither column alone looks like the target, which is why it passes
    # review - but the PAIR inverts to the target exactly. A real series
    # is used as the numerator so this is not an artefact of a synthetic
    # ramp.
    lvl = df['capesize_level'].astype(float)
    coal = df['capesize_level'].rolling(40).mean().bfill() * 0.37  # stand-in
    leaky = coal / (lvl + 1.0)
    recovered = coal / leaky - 1.0
    err = float(np.max(np.abs(recovered - lvl)))
    check('the two features invert to the level target, max err = %.2e' % err,
          err < 1e-6,
          'could not reproduce the leak - test 4 is not testing anything')

    # Now the part that matters: our panel is immune by construction,
    # because the target is a forward return, not a level.
    r_ours = abs(leaky.corr(df['y']))
    check('the same leaky ratio vs OUR forward-return target, |r| = %.4f'
          % r_ours, r_ours < 0.30,
          'the return target is not protecting us the way it should')

    print('\n[5] causality: perturbing the future cannot move the past')
    # Corrupt every raw input after a cut date, rebuild, and require the
    # features before the cut to be bit-identical. This catches any
    # centred window, any negative shift, any full-sample scaler -
    # without needing to know the feature's name.
    cut = df.index[len(df) // 2]
    base = build_panel.build()

    orig_load = build_panel.load
    rng = np.random.default_rng(0)

    def poisoned(name):
        d = orig_load(name)
        if 'date' not in d.columns:
            return d
        m = (d['date'] > cut).to_numpy()
        for c in d.columns:
            if c != 'date' and pd.api.types.is_numeric_dtype(d[c]):
                # Cast first: the Baltic columns are int64 and pandas
                # refuses to write floats back into them.
                d[c] = d[c].astype('float64')
                d.loc[m, c] = d.loc[m, c] * rng.uniform(3.0, 9.0, m.sum())
        return d

    build_panel.load = poisoned
    try:
        after = build_panel.build()
    finally:
        build_panel.load = orig_load

    idx = base.index[base.index <= cut].intersection(after.index)
    bad = []
    for c in feats:
        if not np.allclose(base.loc[idx, c], after.loc[idx, c],
                           rtol=1e-9, atol=1e-12):
            n = (~np.isclose(base.loc[idx, c], after.loc[idx, c],
                             rtol=1e-9, atol=1e-12)).sum()
            bad.append('%s (%d rows)' % (c, n))
    check('all %d features unchanged before %s on %d rows'
          % (len(feats), cut.date(), len(idx)),
          not bad, 'future data leaked into: ' + ', '.join(bad))

    # Sanity: the poison must actually have done something after the cut,
    # otherwise test 5 passes vacuously.
    post = base.index[base.index > cut].intersection(after.index)
    moved = sum(not np.allclose(base.loc[post, c], after.loc[post, c],
                                rtol=1e-9, atol=1e-12) for c in feats)
    check('poison did perturb %d/%d features after the cut'
          % (moved, len(feats)), moved > 0,
          'test 5 passed vacuously - the poison had no effect')

    print('\n' + '=' * 60)
    if FAILURES:
        print('  %d FAILED: %s' % (len(FAILURES), ', '.join(FAILURES)))
        print('=' * 60)
        return 1
    print('  all leakage checks passed')
    print('=' * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())

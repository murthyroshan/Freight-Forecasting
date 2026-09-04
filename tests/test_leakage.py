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
from src import paths, train_model  # noqa: E402

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
    # Recomputed on the SOURCE series, not on the panel.
    #
    # The panel is dropna'd, and since the Capesize index went negative
    # in early 2020 - it is a timecharter equivalent, and a TCE can fall
    # below zero - that now removes rows from the MIDDLE. shift(-5) on
    # the gapped panel index would step over the hole and compare a
    # different pair of days. Stepping on the source index is what the
    # target actually means.
    bal = build_panel.baltic()
    cape = bal['capesize'].where(bal['capesize'] > 0)
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

    print('\n[7] the Baltic splice appends, and never rewrites history')
    # The licensed Mendeley copy runs to 2019-07-31; past it the series
    # continues from a public mirror. Two things must hold or the target
    # has a seam in the middle of it: the licensed half must come
    # through untouched, and the join must not introduce a jump.
    lic = pd.read_parquet(os.path.join(paths.RAW, 'baltic_indices.parquet'))
    lic['date'] = pd.to_datetime(lic['date'])
    lic = lic.set_index('date').sort_index()
    cut = lic.index.max()
    bal = build_panel.baltic()

    for col in ('capesize', 'panamax', 'supramax'):
        same = (bal[col].reindex(lic.index) == lic[col]) | lic[col].isna()
        check('%s: all %d licensed days come through the splice unchanged'
              % (col, len(lic)), bool(same.all()),
              [str(d.date()) for d in lic.index[~same]][:3])

    check('no date appears twice', not bal.index.duplicated().any(),
          [str(d.date()) for d in bal.index[bal.index.duplicated()]][:3])
    check('the series is in date order',
          bool(bal.index.is_monotonic_increasing))
    check('the licensed half is not extended backwards',
          bal.index.min() == lic.index.min(),
          (str(bal.index.min().date()), str(lic.index.min().date())))

    ext_path = os.path.join(paths.RAW, 'baltic_extension.parquet')
    if os.path.exists(ext_path):
        ext = pd.read_parquet(ext_path)
        ext['date'] = pd.to_datetime(ext['date'])
        check('every appended row is strictly AFTER the licensed copy ends '
              '(%s)' % str(cut.date()), bool((ext['date'] > cut).all()),
              [str(d.date()) for d in ext['date'][ext['date'] <= cut]][:3])
        check('the splice reaches past the licensed copy (%d rows added)'
              % len(ext), bal.index.max() > cut,
              (str(bal.index.max().date()), str(cut.date())))

        # A seam would show up as an outsized move at the join. Compare
        # the join week against the series' own typical daily move.
        moves = bal['capesize'].pct_change().abs()
        near = moves[(moves.index > cut - pd.Timedelta(days=7))
                     & (moves.index < cut + pd.Timedelta(days=7))]
        typical = float(moves.median())
        check('the join is smooth: largest move within a week of it is '
              '%.1f%%, against a %.1f%% typical daily move'
              % (float(near.max()) * 100, typical * 100),
              float(near.max()) < typical * 6,
              (float(near.max()), typical))

    print('\n[8] the target survives the days the index went negative')
    # The Capesize basis is a timecharter equivalent and went below zero
    # for 44 sessions in early 2020. A log return is undefined there.
    lvl = df['capesize_level']
    check('no non-positive level survives into the panel',
          bool((lvl > 0).all()), float(lvl.min()))
    check('and no target value is non-finite',
          bool(np.isfinite(df['y']).all()),
          int((~np.isfinite(df['y'])).sum()))
    src = build_panel.baltic()['capesize']
    n_bad = int((src <= 0).sum())
    check('the source really does contain %d non-positive days, so the '
          'guard is doing something' % n_bad, n_bad > 0, n_bad)

    print('\n[9] dropping those rows did not narrow the purge gap')
    # The panel is dropna'd, so removing rows from the MIDDLE leaves
    # gaps. train_model purges by POSITION, so a gap makes the purge
    # span more source days, never fewer - conservative, not leaky.
    loc = {t: i for i, t in enumerate(src.index)}
    spans = [loc[df.index[i + train_model.HORIZON]] - loc[df.index[i]]
             for i in range(len(df) - train_model.HORIZON)]
    check('a %d-row purge never spans fewer than %d source days '
          '(min %d, max %d)'
          % (train_model.HORIZON, train_model.HORIZON, min(spans),
             max(spans)),
          min(spans) >= train_model.HORIZON, min(spans))

    print('\n[10] the mirror is validated, not trusted')
    # fetch_data refuses to write the extension unless the mirror
    # reproduces the licensed copy. Exercised by feeding it deliberately
    # wrong series - if any of these were accepted, a corrupted or
    # rebased feed would be spliced into the middle of the target.
    import shutil
    from src import fetch_data

    ext_path = os.path.join(paths.RAW, 'baltic_extension.parquet')
    backup = ext_path + '.testbak'
    real_fetch = fetch_data._eastmoney
    had = os.path.exists(ext_path)
    if had:
        shutil.copy2(ext_path, backup)

    def corrupted(kind):
        def f(indicator, timeout=45):
            d = real_fetch(indicator, timeout).copy()
            if kind == 'shuffled':
                d['value'] = d['value'].sample(frac=1,
                                               random_state=0).values
            elif kind == 'scaled':
                d['value'] = d['value'] * 1.5
            elif kind == 'drifted':
                d['value'] = d['value'] + np.linspace(0, 400, len(d))
            elif kind == 'joinbroken':
                cut = pd.to_datetime(
                    pd.read_parquet(os.path.join(
                        paths.RAW, 'baltic_indices.parquet'))['date']).max()
                d.loc[d['date'] == cut, 'value'] += 1.0
            return d
        return f

    try:
        for kind, why in (('shuffled', 'a series with the right values in '
                                       'the wrong order'),
                          ('scaled', 'a series on a different scale'),
                          ('drifted', 'a series that drifts away'),
                          ('joinbroken', 'a series that disagrees on the '
                                         'join day')):
            if os.path.exists(ext_path):
                os.remove(ext_path)
            fetch_data._eastmoney = corrupted(kind)
            try:
                fetch_data.fetch_baltic_extension()
            except Exception:
                pass
            check('%s is refused' % why, not os.path.exists(ext_path),
                  'the fetcher wrote an extension it should have rejected')
    finally:
        fetch_data._eastmoney = real_fetch
        if os.path.exists(ext_path):
            os.remove(ext_path)
        if had:
            shutil.move(backup, ext_path)

    check('and the real extension is back in place after the test',
          os.path.exists(ext_path) == had)
    check('the validation gates on VALUE, not on exact matching',
          hasattr(fetch_data, 'MIN_CORRELATION')
          and hasattr(fetch_data, 'MAX_P99_REL_GAP')
          and not hasattr(fetch_data, 'MIN_AGREEMENT'))

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

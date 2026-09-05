"""
Checks on the forward-looking path.

Everything else in this repository is auditable against an outcome that
has already happened. A forecast for next week is the one number on the
dashboard nobody can check, which makes the machinery behind it worth
more scrutiny than the parts that can be graded.

This file starts at the beginning of that machinery: the feature rows
for days that have no outcome yet. Two failures matter here and neither
would announce itself.

  1. Picking up the wrong rows. "No outcome yet" is not the same as "y
     is missing" - a non-positive Capesize print also produces a missing
     y, and 51 rows in the middle of history have one. A forecast built
     on one of those would be presented as next week's while describing
     early 2020.

  2. Imputing a hole. A market series that failed to update leaves a gap
     in the newest row. Filling it forward would produce a confident
     forecast built partly on yesterday's dollar, and nothing on screen
     would say so.

Run:  python -m tests.test_forecast
"""

import io
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import build_panel as bp, forecast as fc, paths, \
    train_model as tm  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def main():
    panel_path = os.path.join(paths.PROCESSED, 'panel.parquet')
    if not os.path.exists(panel_path):
        print('  missing %s - run python -m src.build_panel first'
              % panel_path)
        return 1

    print('\n[1] splitting build() did not change build()')
    # _frame() was extracted so the live path runs the same feature code
    # as training instead of a copy that can drift. The whole refactor is
    # worthless if it moved a single number in the panel.
    committed = pd.read_parquet(panel_path)
    built = bp.build()
    check('same number of rows (%d)' % len(built),
          len(built) == len(committed), (len(built), len(committed)))
    check('same dates, in the same order', built.index.equals(committed.index))
    check('same columns, in the same order',
          list(built.columns) == list(committed.columns))
    check('and every value is identical',
          np.allclose(built[committed.columns].to_numpy(dtype=float),
                      committed.to_numpy(dtype=float), equal_nan=True))
    check('build() is still _frame() with the same rows dropped',
          len(bp._frame()) > len(built), (len(bp._frame()), len(built)))

    print('\n[2] the feature list is defined once, not twice')
    # train_model builds its list with the same comprehension. If the two
    # ever disagree the live forecast would be fitted on one set of
    # columns and predicted on another.
    feats = bp.feature_names(built)
    check('feature_names() matches how train_model selects columns',
          feats == [c for c in built.columns
                    if c not in ('y', 'capesize_level')], feats[:3])
    check('the target is not a feature', 'y' not in feats)
    check('nor is the level kept for plotting',
          'capesize_level' not in feats)
    check('and there are some', len(feats) >= 10, len(feats))

    print('\n[3] the forward window is the days with no outcome YET')
    fwd = bp.features_asof()
    check('there is a forward window at all (%d rows)' % len(fwd),
          len(fwd) > 0, len(fwd))
    if not len(fwd):
        print('  data is stale - rebuild before judging the rest')
        return 1 if FAIL else 0
    check('every row sits after the last scored day (%s)'
          % built.index.max().date(),
          bool((fwd.index > built.index.max()).all()),
          (fwd.index.min(), built.index.max()))
    check('none of them is already in the panel',
          not (set(fwd.index) & set(built.index)))
    check('it is at most one horizon long (%d <= %d)'
          % (len(fwd), bp.HORIZON), len(fwd) <= bp.HORIZON, len(fwd))
    check('the rows are in date order', fwd.index.is_monotonic_increasing)

    print('\n[4] "no outcome yet" is not "y is missing"')
    # The one that would not announce itself. 51 rows in the middle of
    # history have no y because Capesize printed non-positive and the log
    # target masks it; a forecast built on one would be dated next week
    # and describing 2020.
    frame = bp._frame()
    no_y = frame[frame['y'].isna()]
    mid = no_y[no_y.index < fwd.index.min()]
    check('there ARE mid-history rows with no target (%d of them)'
          % len(mid), len(mid) > 0, len(mid))
    check('and not one of them is in the forward window',
          not (set(mid.index) & set(fwd.index)),
          sorted(set(mid.index) & set(fwd.index))[:3])
    nonpos = frame[frame['capesize_level'] <= 0]
    check('the non-positive Capesize days (%d) are excluded too'
          % len(nonpos), not (set(nonpos.index) & set(fwd.index)))

    print('\n[5] features are complete, or the row does not come back')
    check('no feature is missing in any forward row',
          bool(fwd[feats].notna().all().all()),
          fwd[feats].isna().sum().to_dict())
    check('every one of them is finite, not merely non-null',
          bool(np.isfinite(fwd[feats].to_numpy(dtype=float)).all()))
    check('the target is absent for all of them, as it must be',
          bool(fwd['y'].isna().all()))
    check('the columns match training exactly',
          bp.feature_names(fwd) == feats)

    # A stale market feed must stop the forecast, not be filled in.
    real_frame = bp._frame

    def holed():
        g = real_frame()
        g.loc[g.index[-1], feats[0]] = np.nan
        return g

    try:
        bp._frame = holed
        punched = bp.features_asof()
        check('a hole in the newest row drops that row',
              len(punched) == len(fwd) - 1, (len(punched), len(fwd)))
        check('and it is not filled forward from yesterday',
              fwd.index[-1] not in punched.index)
    finally:
        bp._frame = real_frame

    print('\n[6] as_of asks the question from an earlier day')
    for n in range(len(fwd)):
        d = fwd.index[n]
        got = bp.features_asof(d)
        check('as_of %s returns the %d row(s) up to it'
              % (d.date(), n + 1), len(got) == n + 1, len(got))
    check('as_of on the last scored day returns nothing',
          len(bp.features_asof(built.index.max())) == 0)
    check('as_of years ago returns nothing, not history',
          len(bp.features_asof('2020-01-01')) == 0)

    print('\n[7] stale data reports itself instead of raising')
    def truncated():
        g = real_frame()
        return g[g.index <= g[g['y'].notna()].index[-1]]

    try:
        bp._frame = truncated
        empty = bp.features_asof()
        check('no forward window gives an empty frame, not an exception',
              isinstance(empty, pd.DataFrame) and len(empty) == 0)
        check('and it still carries the right columns, so a caller can '
              'inspect it', bp.feature_names(empty) == feats)
    finally:
        bp._frame = real_frame

    print('\n[8] the forward curve exists and is strictly valid JSON')
    fpath = os.path.join(paths.MODELS, 'live_forecast.json')
    if not os.path.exists(fpath):
        check('live_forecast.json exists - run python -m src.forecast',
              False, fpath)
        print('\n' + '=' * 62)
        return 1
    raw = io.open(fpath, encoding='utf-8').read()
    try:
        # NaN and Infinity are legal to Python and rejected by every
        # browser's JSON.parse. An interval that came out non-finite
        # would take the whole page down rather than one card.
        json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(
            ValueError('non-finite constant %r' % c)))
        ok, why = True, ''
    except ValueError as exc:
        ok, why = False, str(exc)
    check('it parses under strict JSON', ok, why)
    fx = json.loads(raw)
    hs = fx['horizons']
    check('every requested horizon is present',
          [h['horizon_days'] for h in hs] == list(fc.HORIZONS),
          [h['horizon_days'] for h in hs])
    check('it records how old the newest price is', 'data_age_days' in fx)
    check('and says the target dates skip weekends but not holidays',
          'holiday' in fx.get('calendar_note', ''))

    print('\n[9] the interval brackets the call and widens with distance')
    for h in hs:
        check('%2dd: low < expected < high' % h['horizon_days'],
              h['lo_pct'] < h['expected_move_pct'] < h['hi_pct'],
              (h['lo_pct'], h['expected_move_pct'], h['hi_pct']))
        check('%2dd: both ends are finite' % h['horizon_days'],
              np.isfinite([h['lo_pct'], h['hi_pct']]).all())
    widths = [h['hi_pct'] - h['lo_pct'] for h in hs]
    check('the further out, the wider - never the reverse: %s'
          % ' '.join('%.0f' % w for w in widths),
          all(widths[i] <= widths[i + 1] for i in range(len(widths) - 1)),
          widths)
    check('the calibration block is never fitted on, and is sized for it',
          all(h['n_calibrated'] >= 30 for h in hs),
          [h['n_calibrated'] for h in hs])
    check('and each model saw a real amount of history',
          all(h['n_fitted'] >= fc.MIN_ROWS for h in hs),
          [h['n_fitted'] for h in hs])

    print('\n[10] a weak call cannot be dressed as a strong one')
    for h in hs:
        check('%2dd: strength is a percentile, so within 0-100'
              % h['horizon_days'], 0 <= h['strength_pct'] <= 100,
              h['strength_pct'])
        sign = ('up' if h['expected_move_pct'] > 0
                else 'down' if h['expected_move_pct'] < 0 else 'flat')
        check('%2dd: the word matches the sign' % h['horizon_days'],
              h['direction'] == sign, (h['direction'], sign))
        check('%2dd: the strong/weak flag matches the number'
              % h['horizon_days'],
              h['stronger_than_usual'] == (h['strength_pct'] >= 50))

    print('\n[11] the target dates are real trading days, in order')
    as_of = pd.Timestamp(fx['as_of'])
    for h in hs:
        t = pd.Timestamp(h['target_date'])
        check('%2dd: lands after the data it was made from'
              % h['horizon_days'], t > as_of, (t.date(), as_of.date()))
        check('%2dd: is a weekday (%s)' % (h['horizon_days'], t.day_name()),
              t.weekday() < 5)
    check('they increase with the horizon',
          all(pd.Timestamp(hs[i]['target_date'])
              < pd.Timestamp(hs[i + 1]['target_date'])
              for i in range(len(hs) - 1)))

    print('\n[12] the curve reproduces the model the rest of the page shows')
    mpath = os.path.join(paths.MODELS, 'metrics.json')
    if os.path.exists(mpath):
        m = json.load(io.open(mpath, encoding='utf-8'))
        five = [h for h in hs if h['horizon_days'] == m['horizon_days']]
        check('the published horizon is on the curve', len(five) == 1)
        if five:
            v = five[0]['validation']
            # This is the cross-check that matters: forecast.py scores
            # its own folds, independently of train_model, and must land
            # on the same numbers. If it does not, one of them is wrong.
            check('its direction matches metrics.json exactly (%.4f%%)'
                  % v['direction_pct'],
                  abs(v['direction_pct']
                      - m['models']['ridge']['direction_pct']) < 1e-9,
                  (v['direction_pct'], m['models']['ridge']['direction_pct']))
            check('its skill matches too (%+.4f%%)' % v['skill_pct'],
                  abs(v['skill_pct']
                      - m['models']['ridge']['skill_vs_zero_pct']) < 1e-9)
            check('and it scored the same rows (%d)' % v['n_scored'],
                  v['n_scored'] == m['n_scored'])

    print('\n[13] the edge decays with distance, and is honest about it')
    dirs = [(h['horizon_days'], h['validation']['direction_pct']) for h in hs]
    print('       ' + '  '.join('%dd=%.1f%%' % d for d in dirs))
    check('one day ahead is the easiest', dirs[0][1] == max(d[1] for d in dirs),
          dirs)
    check('every horizon still beats its OWN base rate, not a coin',
          all(h['validation']['direction_pct'] > h['validation']['base_rate_pct']
              for h in hs),
          [(h['horizon_days'], round(h['validation']['direction_pct'], 1),
            round(h['validation']['base_rate_pct'], 1)) for h in hs])
    check('each horizon is purged by its own length, not by 5',
          all(f[1] - f[3] == 10 for f in tm.folds(3000, 10)),
          [f[1] - f[3] for f in tm.folds(3000, 10)][:3])
    check('and the default purge is unchanged',
          tm.folds(3000) == tm.folds(3000, tm.HORIZON))

    print('\n[14] it refuses rather than inventing an answer')
    lv = pd.Series([100.0, -5.0, 120.0, 130.0],
                   index=pd.date_range('2020-01-01', periods=4))
    t1 = fc.target(lv, 1)
    check('a return computed THROUGH a negative print is NaN',
          bool(np.isnan(t1.iloc[0]) and np.isnan(t1.iloc[1])), list(t1))
    real = bp.features_asof
    try:
        bp.features_asof = lambda as_of=None: bp._frame().iloc[0:0]
        check('with no forward window, build() returns None rather than '
              'forecasting from the last scored day', fc.build() is None)
    finally:
        bp.features_asof = real
    thin = fc.build(horizons=(1, 5000))
    check('a horizon with too little data is dropped, not filled in',
          thin is not None
          and [h['horizon_days'] for h in thin['horizons']] == [1],
          None if thin is None else [h['horizon_days']
                                     for h in thin['horizons']])

    print('\n[15] the repository does not claim a limit it has removed')
    # live_model.py was written when the Baltic history stopped in 2019
    # and said, in its own docstring, that the main model "cannot
    # forecast today". src/forecast.py is that claim being false. A
    # stale statement about our own capability is exactly the kind this
    # project refuses to leave standing anywhere else.
    lm = io.open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'src', 'live_model.py'),
        encoding='utf-8').read()
    check('live_model.py no longer says the main model cannot forecast '
          'today', 'cannot forecast today' not in lm)
    check('it points at the module that does',
          'src/forecast.py' in lm)
    check('and it still explains what it IS for - the traded control',
          'falsification' in lm.lower() and 'BDRY' in lm)

    print('\n' + '=' * 62)
    if FAIL:
        print('  %d FAILED:' % len(FAIL))
        for f in FAIL:
            print('    - %s' % f)
        print('=' * 62)
        return 1
    print('  all forward-path checks passed')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

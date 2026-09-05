"""
The licensed-years model, kept alongside the extended one.

Two models, two periods, one screen. The Mendeley copy of the Baltic
indices runs to 2019-07-31 and is the cleaner half of the record: a
model fitted on it alone scores 64.9% direction against the extended
model's 61.7%. Past that date it cannot be asked anything, because the
data stops.

So the replay uses the licensed model where the licensed data reaches,
and the extended model after it. That buys three extra years of
history on the Charter timing view - back to July 2015 instead of
February 2018.

WHY THIS IS THE COMPROMISE AND NOT THE ANSWER
---------------------------------------------
Two models behind one chart is a seam, and a seam is exactly the sort
of thing that produces a number nobody can defend. The honest version
is one model over one period with one accuracy figure; this is a
deliberate step away from that, taken for a demo, and it is worth being
precise about the cost:

  the headline cannot be blended. 64.9% belongs to the licensed model
  over 2015-2019 and 61.7% to the extended one over 2018-2026. There is
  no combined figure and this module does not compute one - averaging
  them would produce a number describing neither.

  the seam is labelled, not hidden. Every replayed call carries the
  name of the model that made it and that model's own accuracy, so a
  reader is never shown a prediction without being told which of the
  two produced it.

  the overlap goes to the licensed model. Both cover 2018-02 to
  2019-07. Picking per date by whichever scored better there would be
  choosing the answer after seeing it, so the rule is fixed in advance
  and stated: licensed while it reaches, extended after.

NEITHER MODEL REACHES 2012
--------------------------
A walk-forward model cannot score the block it first trained on. The
licensed model's first 672 days are that block, so its replay starts in
July 2015 - the same reason the extended one starts in February 2018,
not a property of the splice.

Run:  python -m src.licensed_model
"""

import json
import os

import numpy as np
import pandas as pd

from src import build_panel as bp, paths, train_model as tm

OOS_PATH = os.path.join(paths.PROCESSED, 'oos_licensed.parquet')
METRICS_PATH = os.path.join(paths.MODELS, 'metrics_licensed.json')


def run():
    """Walk-forward the licensed-only panel, exactly as train_model does.

    Same folds, same purge gap, same conformal method - the only thing
    that differs is the source series, so the two models' figures are
    comparable even though they are never combined.
    """
    df = bp.build(licensed_only=True)
    feats = [c for c in df.columns if c not in ('y', 'capesize_level')]
    preds, lo, hi, fold_id, _imp, fold_rows = tm.walk_forward(df, feats)
    m = fold_id >= 0
    if not m.any():
        return None, None

    y = df['y'].to_numpy(dtype=float)[m]
    out = pd.DataFrame({'y': y}, index=df.index[m])
    for k in preds:
        out[k] = preds[k][m]
    out['lo'], out['hi'] = lo[m], hi[m]

    def score(p):
        rz = float(np.sqrt(np.mean(y ** 2)))
        rp = float(np.sqrt(np.mean((y - p) ** 2)))
        return {'rmse': rp,
                'skill_vs_zero_pct': (1 - rp / rz) * 100 if rz else float('nan'),
                'direction_pct': float((np.sign(p) == np.sign(y)).mean()) * 100}

    up = float((y > 0).mean())
    meta = {
        'model': tm.CONF_ON,
        'source': 'licensed Mendeley Baltic copy only, no extension',
        'panel_rows': int(len(df)),
        'panel_start': str(df.index.min().date()),
        'panel_end': str(df.index.max().date()),
        'n_scored': int(m.sum()),
        'n_effective': int(m.sum() // tm.HORIZON),
        'horizon_days': tm.HORIZON,
        'scored_start': str(out.index.min().date()),
        'scored_end': str(out.index.max().date()),
        'base_rate_pct': float(max(up, 1 - up)) * 100,
        'models': {k: score(out[k].to_numpy(dtype=float))
                   for k in ('zero', 'momentum', 'ridge', 'lgbm')},
        'coverage': float(((out['lo'] <= out['y'])
                           & (out['y'] <= out['hi'])).mean()),
        'note': ('scored on its own folds; this figure describes this model '
                 'over this period only and is never averaged with the '
                 'extended model'),
        'why_not_2012': ('a walk-forward model cannot score the block it '
                         'first trained on - the first %d days are that '
                         'block' % int((df.index < out.index.min()).sum())),
    }
    return out, meta


if __name__ == '__main__':
    oos, meta = run()
    if oos is None:
        raise SystemExit('  nothing scored - run python -m src.build_panel')

    print('=' * 74)
    print('  LICENSED-YEARS MODEL - the cleaner half of the record')
    print('=' * 74)
    print('  panel   %d rows, %s -> %s'
          % (meta['panel_rows'], meta['panel_start'], meta['panel_end']))
    print('  scored  %d rows, %s -> %s  (~%d independent windows)'
          % (meta['n_scored'], meta['scored_start'], meta['scored_end'],
             meta['n_effective']))
    print('')
    print('  %-10s %9s %10s' % ('model', 'RMSE', 'direction'))
    for k in ('zero', 'momentum', 'ridge', 'lgbm'):
        s = meta['models'][k]
        print('  %-10s %9.4f %9.1f%%' % (k, s['rmse'], s['direction_pct']))
    r = meta['models']['ridge']
    print('')
    print('  ridge: %.1f%% direction against a %.1f%% base rate, skill %+.2f%%'
          % (r['direction_pct'], meta['base_rate_pct'],
             r['skill_vs_zero_pct']))
    print('  interval coverage %.1f%%' % (meta['coverage'] * 100))
    print('')
    print('  This figure describes THIS model over THIS period. It is not')
    print('  averaged with the extended model and there is no combined')
    print('  number - one would describe neither.')
    print('  %s.' % meta['why_not_2012'])

    os.makedirs(paths.PROCESSED, exist_ok=True)
    oos.to_parquet(OOS_PATH)
    with open(METRICS_PATH, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print('')
    print('  wrote %s' % OOS_PATH)
    print('  wrote %s' % METRICS_PATH)

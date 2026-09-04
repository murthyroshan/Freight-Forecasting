"""
SAIL freight forecasting dashboard.

Every number this app serves is read from an artefact produced by a
script in this repo:

    models/metrics.json              <- train_model.py
    models/live_metrics.json         <- live_model.py
    data/processed/oos_predictions   <- train_model.py

Nothing is hard-coded and nothing is invented. If an artefact is
missing the app says so rather than substituting a plausible number: a
dashboard that always renders something is indistinguishable from one
that is making it up.

The forecasts served here are genuine WALK-FORWARD OUT-OF-SAMPLE
predictions: for any date you ask about, the model that produced the
number was fitted only on data before that date, with a 5-day purge
gap. That is why the app can also show you what actually happened.

Run:  python app.py      ->  http://127.0.0.1:5000
"""

import os
import json
import sys

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import paths  # noqa: E402

app = Flask(__name__)

METRICS_PATH = os.path.join(paths.MODELS, 'metrics.json')
LIVE_PATH = os.path.join(paths.MODELS, 'live_metrics.json')
OOS_PATH = os.path.join(paths.PROCESSED, 'oos_predictions.parquet')

MISSING = ('Artefact %s is missing. Run:  python -m src.fetch_data && '
           'python -m src.build_panel && python -m src.train_model')


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


METRICS = _load_json(METRICS_PATH)
LIVE = _load_json(LIVE_PATH)
OOS = pd.read_parquet(OOS_PATH) if os.path.exists(OOS_PATH) else None
if OOS is not None:
    OOS.index = pd.to_datetime(OOS.index)

# The panel carries the Capesize level, which we need to turn a log
# return into a displayable rate path.
PANEL_PATH = os.path.join(paths.PROCESSED, 'panel.parquet')
PANEL = pd.read_parquet(PANEL_PATH) if os.path.exists(PANEL_PATH) else None
if PANEL is not None:
    PANEL.index = pd.to_datetime(PANEL.index)

READY = METRICS is not None and OOS is not None and PANEL is not None


DASHBOARD = os.path.join(paths.TEMPLATES, 'dashboard.html')


@app.route('/')
def home():
    """The page is static - it fetches every number from the API below,
    so there is no server-side templating and nothing can be injected
    into it at render time. Read per request so UI edits show up on
    reload without restarting the server."""
    with open(DASHBOARD, encoding='utf-8') as f:
        return Response(f.read(), mimetype='text/html')

@app.route('/api/metrics')
def api_metrics():
    """Validated performance. Straight from train_model.py's output."""
    if METRICS is None:
        return jsonify({'error': MISSING % METRICS_PATH}), 503
    return jsonify(METRICS)


@app.route('/api/control')
def api_control():
    """The control experiment: we tested whether a free traded proxy
    could stand in for the licensed Baltic feed, and it cannot. Shown
    in the UI because a model that is never falsified is not evidence."""
    if LIVE is None:
        return jsonify({'error': MISSING % LIVE_PATH}), 503
    return jsonify({
        'bdry': {k: LIVE[k] for k in ('zero', 'momentum', 'ridge')},
        'control_experiment': LIVE.get('control_experiment'),
        'n_effective': LIVE.get('n_effective'),
    })


@app.route('/api/dates')
def api_dates():
    """The window over which genuine out-of-sample forecasts exist."""
    if not READY:
        return jsonify({'error': MISSING % OOS_PATH}), 503
    return jsonify({'min': str(OOS.index.min().date()),
                    'max': str(OOS.index.max().date()),
                    'n': int(len(OOS))})


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """A real walk-forward forecast for a chosen date.

    The model behind this number never saw the outcome it is predicting,
    so we can show the forecast and the realised move side by side.
    """
    if not READY:
        return jsonify({'error': MISSING % OOS_PATH}), 503

    data = request.json or {}

    def _num(name, required=True, default=None, lo=None, hi=None):
        if name not in data or data[name] is None:
            if required:
                raise ValueError("'%s' is required" % name)
            return default
        try:
            v = float(data[name])
        except (TypeError, ValueError):
            raise ValueError("'%s' must be a number, got %r" % (name, data[name]))
        if v != v or v in (float('inf'), float('-inf')):
            raise ValueError("'%s' must be finite" % name)
        if lo is not None and v < lo:
            raise ValueError("'%s' must be >= %s" % (name, lo))
        if hi is not None and v > hi:
            raise ValueError("'%s' must be <= %s" % (name, hi))
        return v

    try:
        current_rate = _num('current_rate', lo=0.01, hi=1e6)
        volume = _num('volume', required=False, default=0.0, lo=0.0, hi=1e9)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    # Which out-of-sample date are we standing on?
    raw = data.get('date')
    if raw:
        try:
            when = pd.Timestamp(str(raw)).normalize()
        except Exception:
            return jsonify({'error': "'date' must be YYYY-MM-DD, got %r"
                            % raw}), 400
    else:
        when = OOS.index.max()

    if when < OOS.index.min() or when > OOS.index.max():
        return jsonify({'error':
                        'No out-of-sample forecast for %s. Available range '
                        'is %s to %s.' % (when.date(), OOS.index.min().date(),
                                          OOS.index.max().date())}), 400

    # Nearest trading day at or before the request - the desk asks on a
    # Sunday, the index published on the Friday.
    idx = OOS.index[OOS.index <= when]
    if len(idx) == 0:
        return jsonify({'error': 'No trading day on or before %s'
                        % when.date()}), 400
    d = idx[-1]
    row = OOS.loc[d]

    best = METRICS.get('best_model', 'ridge')
    pred = float(row[best])
    actual = float(row['y'])
    lo_r, hi_r = float(row['lo']), float(row['hi'])
    horizon = int(METRICS.get('horizon_days', 5))
    level = float(PANEL.loc[d, 'capesize_level'])

    def pct(logret):
        return (float(np.exp(logret)) - 1.0) * 100.0

    # The model forecasts the INDEX. We apply that percentage to the
    # desk's own negotiated $/ton, because dry bulk charter parties are
    # benchmarked to the Baltic assessment - that assumption is stated
    # in the response rather than hidden.
    projected = current_rate * float(np.exp(pred))
    realised = current_rate * float(np.exp(actual))

    rising = pred > 0
    # Rates rising -> fix now. Rates falling -> hold and re-approach.
    recommendation = 'CHARTER NOW' if rising else 'WAIT'
    delta_per_ton = projected - current_rate
    exposure = abs(delta_per_ton) * volume

    return jsonify({
        'as_of': str(d.date()),
        'requested': str(when.date()),
        'horizon_days': horizon,
        'capesize_index': level,

        'expected_move_pct': pct(pred),
        'lo_pct': pct(lo_r),
        'hi_pct': pct(hi_r),
        'interval_pct': int(round(
            METRICS.get('models', {}).get('conformal', {})
            .get('target', 0.8) * 100)),

        'current_rate': current_rate,
        'projected_rate': round(projected, 2),
        'recommendation': recommendation,
        # Named "exposure", not "savings": it is the size of the move at
        # risk on this parcel, not money banked.
        'exposure': round(exposure, 2),
        'direction': 'up' if rising else 'down',

        # Ground truth. The model never saw this.
        'actual_move_pct': pct(actual),
        'actual_rate': round(realised, 2),
        'direction_correct': bool(np.sign(pred) == np.sign(actual)),
        'inside_interval': bool(lo_r <= actual <= hi_r),

        'model': best,
        'is_placeholder': False,
        'basis': ('walk-forward out-of-sample; fitted only on data before '
                  'this date with a %d-day purge gap' % horizon),
        'assumption': ('the desk rate moves with the Capesize index, which '
                       'is what dry bulk charter parties are benchmarked to'),
        'skill_vs_naive': METRICS.get('models', {}).get(best, {})
                                 .get('skill_vs_zero_pct'),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Never default to True: the Werkzeug debugger is remote code execution.
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('1', 'true', 'yes')
    # Bind to loopback unless explicitly told otherwise.
    host = os.environ.get('HOST', '127.0.0.1')
    print("")
    print("=" * 60)
    print("  SAIL FREIGHT FORECASTING DASHBOARD")
    print("=" * 60)
    print("  Running on:  http://%s:%d" % (host, port))
    if READY:
        b = METRICS['best_model']
        m = METRICS['models'][b]
        print("  Model:       %s, %d-day Capesize direction"
              % (b, METRICS['horizon_days']))
        print("  Validated:   %.1f%% direction, %+.1f%% skill vs no-change"
              % (m['direction_pct'], m['skill_vs_zero_pct']))
        print("  Out-of-sample: %s to %s (%d rows, ~%d independent)"
              % (str(OOS.index.min().date()), str(OOS.index.max().date()),
                 METRICS['n_scored'], METRICS['n_effective']))
    else:
        print("  Model:       NOT BUILT - run  python -m src.fetch_data")
    print("  Debug mode:  %s" % debug)
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print("")
    app.run(host=host, port=port, debug=debug)

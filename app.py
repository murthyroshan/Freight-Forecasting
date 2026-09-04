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
import math
import sys

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import paths, ports, congestion  # noqa: E402

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


def _finite(value):
    """Return a JSON-safe float, or None.

    JSON has no NaN or Infinity. Flask will happily serialise a bare
    NaN, which Python's json.loads accepts but every browser's
    JSON.parse rejects - so the response is HTTP 200 with a body the
    page cannot read, and the dashboard fails silently.

    Nothing in the current artefacts is non-finite, but the conformal
    bounds are NaN for any fold whose calibration block is too small, so
    a change to N_FOLDS could reintroduce it. Send null instead and let
    the page decide how to show a missing bound.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


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


@app.route('/api/ports')
def api_ports():
    """Reference data for Module B: what each class is, what each berth
    permits, and the resulting cargo matrix. No input needed - this is
    physics and published port limits, not a forecast."""
    matrix = []
    for vessel in ports.VESSELS:
        row = {'vessel': vessel, 'cells': []}
        for port in ports.PORTS:
            r = ports.can_serve(vessel, port)
            row['cells'].append({
                'port': port,
                'max_cargo_t': r['max_cargo_t'],
                'utilisation': _finite(r['utilisation']),
                'verdict': r['verdict'],
                'binding': r['binding'],
                'foregone_t': r['foregone_t'],
                'caveats': r['caveats'],
            })
        matrix.append(row)

    return jsonify({
        'vessels': [{
            'name': n,
            'dwt': v['dwt'],
            'draft_m': v['draft'],
            'loa_m': v['loa'],
            'beam_m': v['beam'],
            'tpc': _finite(round(ports.tpc(n), 1)),
            'cargo_capacity_t': v['dwt'] - v['constants'],
        } for n, v in ports.VESSELS.items()],
        'ports': [{
            'name': n,
            'max_draft_m': p['max_draft'],
            'max_loa_m': p['max_loa'],
            'max_beam_m': p['max_beam'],
            'density': p['density'],
            'lighterage': p['lighterage'],
            'geometry_verified': p['geometry_verified'],
            'note': p['note'],
        } for n, p in ports.PORTS.items()],
        'matrix': matrix,
        'min_utilisation': ports.MIN_UTILISATION,
    })


@app.route('/api/ports/options', methods=['POST'])
def api_port_options():
    """Ranked ways to move a given parcel.

    Ranked by voyages, then verdict, then how much of the capacity you
    charter the cargo actually fills - not by how full each ship can be,
    which would call a Capesize and a Newcastlemax equally good and quietly
    recommend the more expensive one.
    """
    data = request.json or {}
    raw = data.get('parcel_t')
    if raw is None:
        return jsonify({'error': "'parcel_t' is required"}), 400
    try:
        parcel = float(raw)
    except (TypeError, ValueError):
        return jsonify({'error': "'parcel_t' must be a number, got %r"
                        % (raw,)}), 400
    if not math.isfinite(parcel):
        return jsonify({'error': "'parcel_t' must be finite"}), 400
    if parcel <= 0:
        return jsonify({'error': "'parcel_t' must be greater than 0"}), 400
    if parcel > 5e6:
        return jsonify({'error': "'parcel_t' must be <= 5000000"}), 400

    try:
        opts = ports.options_for(parcel)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    for o in opts:
        o['utilisation'] = _finite(o['utilisation'])
        o['parcel_utilisation'] = _finite(o['parcel_utilisation'])

    return jsonify({
        'parcel_t': parcel,
        'options': opts,
        'best': opts[0] if opts else None,
        'n': len(opts),
    })


@app.route('/api/congestion')
def api_congestion():
    """How busy each port is running against its own history, from IMF
    PortWatch AIS data - both the east-coast discharge berths and the
    load terminals that feed them.

    This is the only part of the system with CURRENT data - the Baltic
    series ends 2019-07-31, PortWatch runs to last week. It is arrivals
    and tonnage, not waiting time, and the response says so.
    """
    try:
        snaps = congestion.all_snapshots()
    except Exception as exc:
        return jsonify({'error': 'port activity unavailable: %s' % exc}), 503

    profiles = {}
    for p in congestion.ALL_PORTS:
        try:
            profiles[p] = congestion.monthly_profile(p)
        except Exception:
            profiles[p] = None

    return jsonify({
        'snapshots': snaps,
        'discharge': [s for s in snaps if s.get('role') == 'discharge'],
        'load': [s for s in snaps if s.get('role') == 'load'],
        'monthly_profile': profiles,
        'window_days': congestion.WINDOW,
        'bands': [{'below_percentile': None if c == float('inf') else c,
                   'label': n} for c, n in congestion.BANDS],
        'measures': 'arrivals and tonnage, not waiting time',
        'caveat': ('PortWatch reports vessel arrivals derived from AIS. '
                   'It does not publish queue length or berth occupancy, '
                   'so a high reading means the berth is under load and '
                   'you should expect competition for it - not that your '
                   'ship will wait a specific number of days.'),
    })


@app.route('/api/congestion/<port>')
def api_congestion_port(port):
    """Daily arrivals for one port, for charting."""
    if port not in congestion.ALL_PORTS:
        return jsonify({'error': 'unknown port %r; known: %s'
                        % (port, ', '.join(congestion.ALL_PORTS))}), 400
    try:
        days = int(request.args.get('days', 180))
    except (TypeError, ValueError):
        return jsonify({'error': "'days' must be an integer"}), 400
    days = max(30, min(days, 1000))
    try:
        return jsonify({'port': port, 'days': days,
                        'series': congestion.series(port, days=days),
                        'snapshot': congestion.snapshot(port)})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 503


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
        'capesize_index': _finite(level),

        'expected_move_pct': _finite(pct(pred)),
        'lo_pct': _finite(pct(lo_r)),
        'hi_pct': _finite(pct(hi_r)),
        'interval_pct': int(round(
            METRICS.get('models', {}).get('conformal', {})
            .get('target', 0.8) * 100)),

        'current_rate': _finite(current_rate),
        'projected_rate': _finite(round(projected, 2)),
        'recommendation': recommendation,
        # Named "exposure", not "savings": it is the size of the move at
        # risk on this parcel, not money banked.
        'exposure': _finite(round(exposure, 2)),
        'direction': 'up' if rising else 'down',

        # Ground truth. The model never saw this.
        'actual_move_pct': _finite(pct(actual)),
        'actual_rate': _finite(round(realised, 2)),
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

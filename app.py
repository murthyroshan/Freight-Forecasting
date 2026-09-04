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
import threading
import time

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response

# .env.example tells the reader to copy it to .env and edit. That only
# works if something loads the file: python-dotenv was in
# requirements.txt but imported nowhere, so PORT, HOST and FLASK_DEBUG
# were silently ignored unless exported by hand.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:                     # optional; env vars still work
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import (paths, ports, congestion, risk, optimise,  # noqa: E402
                 ballast)

app = Flask(__name__)

METRICS_PATH = os.path.join(paths.MODELS, 'metrics.json')
LIVE_PATH = os.path.join(paths.MODELS, 'live_metrics.json')
OOS_PATH = os.path.join(paths.PROCESSED, 'oos_predictions.parquet')

MISSING = ('Artefact %s is missing. Run:  python -m src.fetch_data && '
           'python -m src.build_panel && python -m src.train_model')

# The control experiment comes from a different script, and pointing the
# reader at train_model would have them run the wrong one and see no
# change.
MISSING_LIVE = ('Artefact %s is missing. Run:  python -m src.fetch_data && '
                'python -m src.live_model')


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


def _number(value, name, lo=None, hi=None):
    """Parse a JSON value as a finite number, or raise ValueError.

    Booleans are refused before float() sees them: `float(True)` is 1.0,
    so `parcel_t: true` would otherwise be answered, with a straight
    face, as a one-tonne parcel costing $620,000 a tonne.
    """
    if isinstance(value, bool):
        raise ValueError("'%s' must be a number, not a boolean" % name)
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("'%s' must be a number, got %r" % (name, value))
    if not math.isfinite(v):
        raise ValueError("'%s' must be finite, got %r" % (name, value))
    if lo is not None and v < lo:
        raise ValueError("'%s' must be >= %s, got %r" % (name, lo, value))
    if hi is not None and v > hi:
        raise ValueError("'%s' must be <= %s, got %r" % (name, hi, value))
    return v


def _json_safe(obj):
    """_finite, applied through a whole nested payload.

    _finite above takes one number. Handing it a dict returns None and
    silently blanks the entire response, which is a worse failure than
    the one it exists to prevent, so walk the structure instead.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    # A Python int is always finite and always JSON-safe. Sending it
    # through _finite() would return a float, turning 1010 into 1010.0 -
    # which is the same number but not the same payload, and tests/
    # test_app.py compares the served bytes against the artefact.
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float) or hasattr(obj, 'dtype'):
        return _finite(obj)
    return obj


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


# Werkzeug answers an unrouted path with an HTML error page. For a
# browser that is right; for /api/* it means a client calling .json() on
# the response gets a parse error instead of the reason. Keep every
# answer under /api/ in the same format as the successful ones.
def _body():
    """The JSON body as a dict, or a ready-made 400.

    `request.json or {}` only rescues a FALSY body - null, {}, [], 0, "".
    A truthy non-dict such as [1,2,3] sails through and raises
    AttributeError on the first .get(), which the caller sees as a 500.
    """
    # Keep the three failures distinct: a body sent as the wrong media
    # type is 415, a body that is not valid JSON is 400, and a body that
    # parses to something other than an object is 400 with a different
    # reason. Collapsing them loses information the caller needs.
    if request.content_length and not request.is_json:
        return None, (jsonify({
            'error': 'send Content-Type: application/json, got %r'
                     % (request.content_type or 'nothing')}), 415)
    data = request.get_json(silent=True)
    if data is None:
        if request.content_length:
            return None, (jsonify({'error': 'body is not valid JSON'}), 400)
        return {}, None
    if not isinstance(data, dict):
        return None, (jsonify({'error': 'request body must be a JSON '
                               'object, got %s' % type(data).__name__}), 400)
    return data, None


def _api_error(exc, code, message):
    if request.path.startswith('/api/'):
        return jsonify({'error': message, 'path': request.path}), code
    return exc


@app.errorhandler(404)
def _not_found(exc):
    return _api_error(exc, 404, 'no such endpoint')


@app.errorhandler(405)
def _bad_method(exc):
    return _api_error(exc, 405, 'method not allowed for this endpoint')


@app.errorhandler(400)
def _bad_request(exc):
    # Werkzeug raises this itself for an unparseable JSON body, before
    # any view runs. The page puts `await r.json()` inside its network
    # try block, so an HTML body here surfaces to the user as "backend
    # unreachable" while the server is up and answering.
    return _api_error(exc, 400, 'malformed request body')


@app.errorhandler(415)
def _bad_media_type(exc):
    return _api_error(exc, 415, 'send Content-Type: application/json')


@app.errorhandler(500)
def _server_error(exc):
    return _api_error(exc, 500, 'internal error')


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
    return jsonify(_json_safe(METRICS))


@app.route('/api/control')
def api_control():
    """The control experiment: we tested whether a free traded proxy
    could stand in for the licensed Baltic feed, and it cannot. Shown
    in the UI because a model that is never falsified is not evidence."""
    if LIVE is None:
        return jsonify({'error': MISSING_LIVE % LIVE_PATH}), 503
    return jsonify(_json_safe({
        'bdry': {k: LIVE[k] for k in ('zero', 'momentum', 'ridge')},
        'control_experiment': LIVE.get('control_experiment'),
        'n_effective': LIVE.get('n_effective'),
    }))


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
    data, err = _body()
    if err:
        return err
    if data.get('parcel_t') is None:
        return jsonify({'error': "'parcel_t' is required"}), 400
    try:
        parcel = _number(data['parcel_t'], 'parcel_t')
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
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


@app.route('/api/ballast')
def api_ballast():
    """Deliverable (c), the half that is measurable.

    Idle time is not here and cannot be: PortWatch reports arrivals, not
    arrivals and departures, so there is no dwell to compute. What is
    here is the empty leg - the share of inbound dry bulk tonnage a berth
    cannot match with outbound.
    """
    try:
        rows = ballast.profile()
    except Exception as exc:
        return jsonify({'error': 'ballast profile unavailable: %s'
                        % exc}), 503
    return jsonify(_json_safe({
        'ports': rows,
        'window_days': ballast.WINDOW_DAYS,
        'min_import_t_per_day': ballast.MIN_IMPORT_T_PER_DAY,
        'parity': ballast.PARITY,
        'measures': 'tonnage in against tonnage out, not waiting time',
        'caveat': ('Aggregate matching is an upper bound on backhaul - a '
                   'given ship may not be able to take a given export '
                   'cargo - so the empty share is a LOWER bound on ballast '
                   'sailing. Idle and demurrage are not measured here at '
                   'all: PortWatch publishes arrivals, not departures.'),
    }))


@app.route('/api/optimise', methods=['POST'])
def api_optimise():
    """Deliverable (b). Which classes, into which berths, at what split.

    Every cost here arrives in the request body. That is deliberate: this
    repository has no verified freight, lighterage or haulage figures for
    this lane, so it asks rather than assumes, and the answer says so.
    """
    data, err = _body()
    if err:
        return err

    def _money(obj, name):
        if obj is None:
            return None
        if not isinstance(obj, dict):
            raise ValueError('%s must be an object keyed by name' % name)
        out = {}
        for k, v in obj.items():
            if v is None or v == '':
                continue
            out[k] = _number(v, '%s[%s]' % (name, k), lo=0.0)
        return out

    try:
        if data.get('parcel_t') is None:
            raise ValueError("'parcel_t' is required")
        parcel = _number(data['parcel_t'], 'parcel_t', lo=1.0)
        voyage = _money(data.get('voyage_cost'), 'voyage_cost') or {}
        if not voyage:
            raise ValueError("'voyage_cost' is required - this project has "
                             'no verified freight rates and will not assume '
                             'one')
        kw = dict(
            port_cost=_money(data.get('port_cost'), 'port_cost'),
            lighterage_cost=_money(data.get('lighterage_cost'),
                                   'lighterage_cost'),
            inland_cost=_money(data.get('inland_cost'), 'inland_cost'),
            max_calls=_money(data.get('max_calls'), 'max_calls'),
        )
        # The empty leg. The share is measured from arrivals; what a
        # repositioning voyage costs is not, so it arrives as a fraction
        # of a laden voyage that the caller states.
        # Not named `pct`: that is a module-level helper here, and
        # shadowing it would make every later call in this function hit a
        # float instead of the function.
        ballast_pct = data.get('ballast_pct')
        shares = None
        if ballast_pct not in (None, ''):
            penalty, shares = ballast.ballast_penalty(voyage, ballast_pct)
            kw['ballast_cost'] = penalty
    except (TypeError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400

    try:
        out = optimise.compare(parcel, voyage, **kw)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:                       # solver blew up
        return jsonify({'error': 'selection failed: %s' % exc}), 503

    out['classes'] = sorted(ports.VESSELS)
    out['ports'] = sorted(ports.PORTS)
    out['ballast_shares'] = shares
    out['ballast_pct'] = (float(ballast_pct)
                          if ballast_pct not in (None, '') else None)
    out['note'] = ('Capacities are computed from draft, TPC and dock water '
                   'allowance, and the empty-leg share from PortWatch '
                   'tonnage. Costs are yours - nothing here is a freight '
                   'rate this project has verified.')
    return jsonify(_json_safe(out))


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
        # Computed, not typed. The page used to hardcode "-88%" and
        # "p < 0.0001" beside a literal "90 km/h", which is exactly the
        # thing this project says it never does.
        'cyclone_effect': congestion.cyclone_effect(),
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


# risk.assess() makes one live call to Open-Meteo per port. A short
# cache keeps a page refresh from hammering a free API without making the
# warnings meaningfully stale - a ten-day gust forecast does not change
# by the minute.
#
# The lock matters because the miss is slow: without it, N browsers
# arriving together on a cold cache each fire their own round of
# forecasts. Holding it means the first computes and the rest wake to a
# warm cache, so the free API sees one round of calls rather than N.
_RISK_CACHE = {'at': 0.0, 'data': None}
_RISK_TTL = 900          # seconds
_RISK_LOCK = threading.Lock()


@app.route('/api/risk')
def api_risk():
    """Deliverable (d). Every warning states whether its effect was
    MEASURED by this repository or is context with no measured effect
    size - the distinction matters more than the warning count."""
    def fresh():
        return (_RISK_CACHE['data'] is not None
                and time.time() - _RISK_CACHE['at'] < _RISK_TTL)

    if fresh():
        payload = _RISK_CACHE['data']
    else:
        with _RISK_LOCK:
            # Re-check inside the lock: whoever held it may already have
            # done the work while this request was waiting.
            if fresh():
                return jsonify(_RISK_CACHE['data'])
            try:
                rows = risk.assess()
            except Exception as exc:
                return jsonify({'error': 'risk assessment unavailable: %s'
                                % exc}), 503
            payload = {
                'warnings': rows,
                'counts': {k: sum(1 for w in rows if w['severity'] == k)
                           for k in ('critical', 'warning', 'watch', 'clear')},
                'measured': sum(1 for w in rows if w['measured']),
                'context_only': sum(1 for w in rows if not w['measured']),
                'ports': sorted(risk.SITES),
                'gust_halt_kmh': risk.GUST_HALT_KMH,
                'gust_gale_kmh': risk.GUST_GALE_KMH,
                'forecast_days': risk.FORECAST_DAYS,
                'note': ('Only the cyclone rule and the interval width are '
                         'measured effects. Berth load and seasonality are '
                         'real observations with no measured effect size, '
                         'and are labelled so a desk can weigh them rather '
                         'than act on them as forecasts.'),
            }
            _RISK_CACHE.update(at=time.time(), data=payload)
    return jsonify(payload)


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

    data, err = _body()
    if err:
        return err

    def _num(name, required=True, default=None, lo=None, hi=None):
        if data.get(name) is None:
            if required:
                raise ValueError("'%s' is required" % name)
            return default
        return _number(data[name], name, lo, hi)

    try:
        current_rate = _num('current_rate', lo=0.01, hi=1e6)
        volume = _num('volume', required=False, default=0.0, lo=0.0, hi=1e9)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    # Which out-of-sample date are we standing on?
    raw = data.get('date')
    if raw is not None and raw != '':
        try:
            when = pd.Timestamp(str(raw))
            # A browser sends new Date().toISOString(), which is
            # tz-aware. Comparing that against this tz-naive index
            # raises two lines below - outside any handler, as a 500.
            if when.tzinfo is not None:
                when = when.tz_convert(None)
            when = when.normalize()
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

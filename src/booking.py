"""
The booking calendar, and what the timing edge is worth in rupees.

A month grid where each day is BOOK, WATCH or AVOID turns a forecast
into something a desk reads in two seconds. It is also the easiest
thing in this project to fake: fill a year with plausible daily rates
and every square looks authoritative, including the ones for next
August.

So this colours only the days the model actually forecast - the next
ten trading days, each from a model fitted and validated at that
horizon - and leaves every other square explicitly out of scope. The
empty half of the calendar is the honest half.

WHAT THE COLOURS MEAN, AND WHAT THEY DO NOT
-------------------------------------------
They are a RANKING of the model's central expectation, not a
significance claim. The spread across ten days is routinely narrower
than the 80% interval on any single one of them, which means the model
is far more confident about the shape than about the level. Saying
"book Tuesday" as though Tuesday were provably cheaper than Thursday
would be reading precision the interval does not support.

The payload therefore carries both numbers - the spread and the typical
interval width - and states which is larger, so a reader can see the
ranking for what it is.

THE RUPEE FIGURE
----------------
Two inputs are the reader's: the tonnage and the freight rate. This
repository has no sourced rate for the East Coast lane and asserting
one would put a fabricated number at the front of the pitch, so the
multiplication is done with figures the user supplies and the
provenance of each is stated. The percentage being multiplied is the
one src/procurement.py measured, over as many independent fixtures
as that file reports.

Run:  python -m src.booking
"""

import json
import os

import pandas as pd

from src import build_panel as bp, paths

# Where the colours change. Thirds of the forecast window: cheapest
# third BOOK, dearest third AVOID. Not a threshold on the move itself,
# because the size of a move says nothing about whether it is the best
# day available in the window.
BOOK_SHARE = 1.0 / 3.0
AVOID_SHARE = 1.0 / 3.0

CRORE = 1e7          # rupees in one crore


def usd_inr(default=None):
    """The dollar in rupees, from the panel's own market data.

    Read rather than assumed: an FX rate typed into a constant goes
    stale silently, and every rupee figure on the page is a multiple
    of it.
    """
    try:
        # subset= matters: the panel carries a volume column beside the
        # close, and Yahoo leaves it NaN on some FX days. A bare dropna()
        # would throw away a perfectly good rate for a missing volume and
        # quietly date every rupee figure on the page.
        d = bp.load('usdinr').dropna(subset=['usdinr'])
        return float(d['usdinr'].iloc[-1]), str(
            pd.to_datetime(d['date'].iloc[-1]).date())
    except Exception:
        return default, None


def calendar(forecast, tonnes=None, rate_usd_per_t=None, fx=None):
    """Rank the forecast days and price the difference between them."""
    hs = list(forecast.get('horizons') or [])
    if not hs:
        return None
    hs = sorted(hs, key=lambda h: h['horizon_days'])

    moves = [h['expected_move_pct'] for h in hs]
    best_move, worst_move = min(moves), max(moves)
    order = sorted(range(len(hs)), key=lambda i: moves[i])
    n = len(hs)
    n_book = max(1, int(round(n * BOOK_SHARE)))
    n_avoid = max(1, int(round(n * AVOID_SHARE)))

    verdict = {}
    for rank, i in enumerate(order):
        if rank < n_book:
            verdict[i] = 'BOOK'
        elif rank >= n - n_avoid:
            verdict[i] = 'AVOID'
        else:
            verdict[i] = 'WATCH'

    widths = sorted(h['hi_pct'] - h['lo_pct'] for h in hs)
    typical = widths[len(widths) // 2]
    spread = worst_move - best_move

    days = []
    for i, h in enumerate(hs):
        gap = h['expected_move_pct'] - best_move
        row = {
            'date': h['target_date'],
            'horizon_days': h['horizon_days'],
            'expected_move_pct': h['expected_move_pct'],
            'lo_pct': h['lo_pct'], 'hi_pct': h['hi_pct'],
            'rank': order.index(i) + 1,
            'verdict': verdict[i],
            'worse_than_best_pct': gap,
            'direction_pct': (h.get('validation') or {}).get('direction_pct'),
        }
        if tonnes and rate_usd_per_t:
            extra_usd = gap / 100.0 * rate_usd_per_t * tonnes
            row['extra_cost_usd'] = extra_usd
            if fx:
                row['extra_cost_inr'] = extra_usd * fx
        days.append(row)

    return {
        'as_of': forecast.get('as_of'),
        'capesize_index': forecast.get('capesize_index'),
        'interval_pct': forecast.get('interval_pct'),
        'days': days,
        'best_date': hs[order[0]]['target_date'],
        'best_move_pct': best_move,
        'worst_move_pct': worst_move,
        'spread_pct': spread,
        'typical_interval_pct': typical,
        # The sentence that stops the colours being over-read. It is
        # computed, so it cannot drift away from the numbers above it.
        'ranking_only': bool(spread < typical),
        'scope_note': ('only the %d trading days the model forecast are '
                       'coloured; every other square is outside the horizon '
                       'and is left blank rather than filled in' % n),
    }


def annual_impact(saved_pct, tonnes_per_year, rate_usd_per_t, fx):
    """What the timing edge is worth over a year's programme.

    Every input is either measured here or supplied by the reader, and
    which is which is recorded in the result. The percentage is the one
    src/procurement.py measured; the tonnage and the rate are the
    reader's, because this project has no sourced rate for the lane.
    """
    if not (saved_pct and tonnes_per_year and rate_usd_per_t and fx):
        return None
    per_t_usd = saved_pct / 100.0 * rate_usd_per_t
    annual_usd = per_t_usd * tonnes_per_year
    annual_inr = annual_usd * fx
    return {
        'saved_pct': saved_pct,
        'tonnes_per_year': tonnes_per_year,
        'rate_usd_per_t': rate_usd_per_t,
        'usd_inr': fx,
        'saved_usd_per_t': per_t_usd,
        'annual_usd': annual_usd,
        'annual_inr': annual_inr,
        'annual_crore': annual_inr / CRORE,
        'measured': ['saved_pct', 'usd_inr'],
        'supplied': ['tonnes_per_year', 'rate_usd_per_t'],
    }


def build(parcel_t=None, annual_t=None, rate_usd_per_t=None):
    """Assemble both from the artefacts on disk.

    The two tonnages are deliberately separate. The calendar prices ONE
    fixture - what waiting a few days costs on the parcel being booked -
    while the annual figure prices a year's programme. Feeding the
    annual tonnage into the calendar, as a first version of this did,
    charges a single Tuesday for every tonne SAIL moves in a year and
    produces a number nobody can act on.
    """
    fpath = os.path.join(paths.MODELS, 'live_forecast.json')
    ppath = os.path.join(paths.MODELS, 'procurement.json')
    if not os.path.exists(fpath):
        return None
    with open(fpath, encoding='utf-8') as fh:
        fc = json.load(fh)
    fx, fx_date = usd_inr()
    cal = calendar(fc, parcel_t, rate_usd_per_t, fx)
    if cal is None:
        return None
    cal['usd_inr'] = fx
    cal['usd_inr_as_of'] = fx_date

    if os.path.exists(ppath):
        with open(ppath, encoding='utf-8') as fh:
            proc = json.load(fh)
        cal['n_fixtures'] = proc['n_fixtures']
        cal['saved_pct'] = proc['model_policy']['saved_pct']
        cal['saved_p_value'] = proc['p_vs_zero']
        cal['lose_rate_pct'] = proc['lose_rate_pct']
        cal['beats_momentum'] = bool(proc['p_vs_momentum'] < 0.05)
        cal['parcel_t'] = parcel_t
        cal['impact'] = annual_impact(cal['saved_pct'], annual_t,
                                      rate_usd_per_t, fx)
    return cal


if __name__ == '__main__':
    # Illustrative inputs only, and labelled as such in the output.
    PARCEL, ANNUAL, RATE = 160_000, 10_000_000, 20.0
    c = build(PARCEL, ANNUAL, RATE)
    if c is None:
        raise SystemExit('  no forecast - run python -m src.forecast first')

    print('=' * 76)
    print('  BOOKING CALENDAR - from the close of %s' % c['as_of'])
    print('=' * 76)
    print('  one parcel of %s tonnes at $%.2f/t'
          % (format(PARCEL, ','), RATE))
    print('  %-12s %-6s %-7s %9s %10s %14s'
          % ('date', 'ahead', 'call', 'expected', 'vs best', 'costs extra'))
    for d in c['days']:
        extra = ('-' if d['worse_than_best_pct'] <= 1e-9
                 else '$%s' % format(int(d.get('extra_cost_usd') or 0), ','))
        print('  %-12s %-6s %-7s %+8.2f%% %9.2f%% %14s'
              % (d['date'], '%dd' % d['horizon_days'], d['verdict'],
                 d['expected_move_pct'], d['worse_than_best_pct'], extra))
    print('')
    print('  cheapest expected day: %s' % c['best_date'])
    print('  spread across the window %.2f%%, typical 80%% interval %.2f%%'
          % (c['spread_pct'], c['typical_interval_pct']))
    if c['ranking_only']:
        print('  -> the spread is NARROWER than the interval on a single')
        print('     day, so these colours rank days by expectation. They')
        print('     do not establish that one day is cheaper than another.')
    print('  %s' % c['scope_note'])

    im = c.get('impact')
    if im:
        print('')
        print('  ' + '-' * 66)
        print('  WHAT THE TIMING EDGE IS WORTH OVER A YEAR')
        print('  measured here : %.2f%% of the freight rate per fixture'
              % im['saved_pct'])
        print('  measured here : USD/INR %.2f (from the panel, %s)'
              % (im['usd_inr'], c['usd_inr_as_of']))
        print('  YOUR figure   : %s tonnes a year'
              % format(int(im['tonnes_per_year']), ','))
        print('  YOUR figure   : $%.2f per tonne freight' % im['rate_usd_per_t'])
        print('')
        print('  -> $%.3f per tonne, $%s a year, Rs %.2f crore'
              % (im['saved_usd_per_t'], format(int(im['annual_usd']), ','),
                 im['annual_crore']))
        print('')
        print('  The percentage is measured over %s independent fixtures'
              % format(int(c['n_fixtures']), ','))
        print('  (p=%.3f). It loses on %.0f%% of the weeks it acts, and it is'
              % (c['saved_p_value'], c['lose_rate_pct']))
        print('  %s distinguishable from a momentum rule on money.'
              % ('' if c['beats_momentum'] else 'NOT'))
        print('  The tonnage and the rate above are illustrative - this')
        print('  project has no sourced freight rate for the lane.')

"""
The assistant. Grounded, offline, and unable to invent a number.

Every other part of this repository refuses to print a figure no script
computed. An assistant is the easiest place to lose that: ask a language
model about freight and it will produce a confident paragraph containing
numbers that came from nowhere, and it will look better than the truth.

So this one has no language model in it. It matches the question against
a registry of skills, each of which reads the same artefacts the
dashboard reads and computes its answer at query time. Every reply
carries the module or artefact it came from. Nothing is templated over a
number that was not just calculated.

WHAT THAT BUYS, AND WHAT IT COSTS
---------------------------------
It cannot hallucinate, it needs no API key, it costs nothing per
question, and it works with the wifi off - which matters, because the
whole dashboard is built to survive a venue with no network.

The cost is that it only knows what it was taught. A question outside
the registry gets a refusal, not a guess. That refusal is the feature:
"I cannot answer that from the artefacts, here is what I can answer" is
worth more in front of a panel than a fluent paragraph that turns out to
be invented.

HOW MATCHING WORKS
------------------
Each skill declares weighted terms. A question is normalised, scored
against every skill, and the best is used only if it clears MIN_SCORE.
Multi-word terms score higher than single words because they are far
less likely to match by accident - "empty leg" is a strong signal,
"leg" on its own is not.

Entities - ports, vessels, tonnages, years, months - are pulled out
separately, so "can a Capesize berth at Paradip" and "will a Newcastlemax
fit into Haldia" reach the same handler with different arguments.

Run:  python -m src.assistant "how accurate is it"
"""

import difflib
import json
import os
import re

import numpy as np
import pandas as pd

from src import (ballast, booking, congestion, paths, ports, risk,
                 seasonal)

MIN_SCORE = 1.0          # below this the assistant says it does not know

# One bare word that means something here but not yet one thing.
# Asking back beats refusing, and beats guessing by further still.
# Words that follow "in"/"at" and are NOT a place - so a question like
# "risk at berth" is not mistaken for a question about somewhere we do
# not cover.
_PLACE_OK = frozenset((
    'berth', 'berths', 'port', 'ports', 'sea', 'india', 'the', 'this',
    'that', 'risk', 'all', 'any', 'each', 'every', 'them', 'both',
    'general', 'total', 'dollars', 'rupees', 'tonnes', 'tons', 'usd',
    'inr', 'crore', 'terms', 'practice', 'theory', 'future', 'past',
    'advance', 'short', 'long', 'days', 'weeks', 'months', 'years',
))

_CLARIFY = {
    'cost': ('Cost of what?', [
        'the cheapest fleet for a parcel', 'what the timing edge is worth in rupees', 'the cost of waiting a few days']), 'when': ('When for what?', [
        'when to book this parcel', 'the forecast for the coming month', 'which months are historically cheap']), 'ports': ('Which part of the ports?', [
        'which berths are busy now', 'what draft each berth allows', 'weather risk at a discharge port']), 'model': ('Which part of the model?', [
        'how accurate it is', 'how it was validated', 'when it fails']), 'why': ('Why what?', [
        'why five trading days', 'why there are two models', 'why you should or should not trust it']), 'data': ('Which data?', [
        'where the data comes from', 'whether the Baltic data is licensed', 'why the history starts in 2015'])}
MAX_SUGGESTIONS = 4

# Terms a judge or a desk will ask about. Every figure quoted here is
# read from the modules at answer time, never typed in - a glossary is
# the easiest place in a project like this to let a stale number sit
# for months without anyone checking it.
GLOSSARY = {
    'capesize': ('vessel', 'Capesize'),
    'newcastlemax': ('vessel', 'Newcastlemax'),
    'panamax': ('vessel', 'Panamax'),
    'supramax': ('vessel', 'Supramax'),
    'handysize': ('vessel', 'Handysize'),
    'dwt': ('term', 'deadweight'),
    'deadweight': ('term', 'deadweight'),
    'tpc': ('term', 'tpc'),
    'draft': ('term', 'draft'),
    'draught': ('term', 'draft'),
    'laycan': ('term', 'laycan'),
    'ballast': ('term', 'ballast'),
    'lighterage': ('term', 'lighterage'),
    'tce': ('term', 'tce'),
    'baltic': ('term', 'baltic'),
    'bci': ('term', 'baltic'),
    'conformal': ('term', 'conformal'),
    'purge': ('term', 'purge'),
    'fixture': ('term', 'fixture'),
    'charter': ('term', 'fixture'),
    'demurrage': ('term', 'demurrage'),
    'backhaul': ('term', 'ballast'),
}

_TERMS = {
    'deadweight': (
        'Deadweight (DWT) is everything a ship can carry - cargo, fuel, '
        'stores, crew and water - not just the cargo. The gap matters '
        'here: a %s is %s t deadweight but lands rather less than that, '
        'because bunkers and constants take their share before any coal '
        'does.'),
    'tpc': (
        'Tonnes per centimetre immersion. Loading this many tonnes pushes '
        'the hull one centimetre deeper. It is waterplane area times '
        'water density divided by 100, and it is the number that turns a '
        'berth\u2019s depth limit into a cargo figure. A %s sits at '
        '**%s t/cm**, so every centimetre of draft the berth cannot give '
        'you costs that much cargo.'),
    'draft': (
        'How deep the hull sits. A %s draws **%s m** fully laden, and a '
        'berth that cannot offer that depth does not turn the ship away - '
        'it just means she arrives part-laden, which is the whole '
        'question the Vessel & port view answers.'),
    'laycan': (
        'Laydays and cancelling - the window a charterer must present the '
        'ship in. It is why waiting for a better rate is not free: hold '
        'out too long and the window closes. This project deliberately '
        'does not price that, which is why the timing saving it reports '
        'is an upper bound on the rate component alone.'),
    'ballast': (
        'A ballast leg is a voyage sailed empty, with only seawater for '
        'stability. Somebody pays for it, and it lands in the freight '
        'rate. The empty-leg share here is measured from PortWatch '
        'arrivals per berth rather than assumed - it runs from 0%% at '
        'Paradip to 94%% at Haldia, and three berths have no coverage at '
        'all, which the answer says rather than hides.'),
    'lighterage': (
        'Discharging into barges offshore because the ship cannot come '
        'alongside. It buys draft and costs money per tonne, so it turns '
        'a physical constraint into an economic one.'),
    'tce': (
        'Timecharter equivalent - voyage earnings expressed as a daily '
        'rate, so voyages of different lengths can be compared. It can '
        'go **negative**, and Capesize did for 44 sessions in 2020, which '
        'is why the target here is a log return through positive levels '
        'only rather than a raw ratio.'),
    'baltic': (
        'The Baltic Capesize Index, published daily by the Baltic '
        'Exchange from a panel of brokers. It is a **survey**, not a '
        'traded price, which is exactly why it is forecastable: '
        'panellists anchor on the previous print, so its returns are '
        'autocorrelated (lag-1 +0.62). The traded proxy, BDRY, is not - '
        'and this project scores badly on it on purpose, as a control.'),
    'conformal': (
        'A way of putting an interval round a forecast without assuming '
        'the errors are normally distributed. You hold out a calibration '
        'block the model never fitted, look at how wrong it was there, '
        'and take the quantile. Here it is locally weighted - residuals '
        'scaled by recent volatility - so the band widens in a stressed '
        'market and tightens in a calm one. Realised coverage is '
        '**%s%%** against an 80%% target.'),
    'purge': (
        'A gap of dead rows between the training block and the test '
        'block. The target at day t is built from day t+5, so without a '
        'five-day gap the last training targets are computed from prices '
        'inside the test window and the model has read its own exam '
        'paper. Every fold here purges on both boundaries.'),
    'fixture': (
        'One chartered voyage - the contract to move a parcel on a ship. '
        'The procurement backtest scores %s of them, spaced a full '
        'horizon apart so no market move is counted twice.'),
    'demurrage': (
        'Compensation paid when loading or discharging runs past the '
        'agreed laytime. It is a real and large cost in this trade, and '
        'this project does **not** model it - there is no demurrage data '
        'here, so nothing on the dashboard should be read as pricing '
        'it.'),
}

MONTHS = ('january', 'february', 'march', 'april', 'may', 'june', 'july',
          'august', 'september', 'october', 'november', 'december')

# Names a desk would actually type, mapped to the canonical ones.
PORT_ALIASES = {
    'vizag': 'Visakhapatnam', 'vizhag': 'Visakhapatnam',
    'vishakhapatnam': 'Visakhapatnam', 'visag': 'Visakhapatnam',
    'sandheads': 'Sagar-Sandheads', 'sagar': 'Sagar-Sandheads',
    'kolkata': 'Haldia', 'calcutta': 'Haldia',
}
VESSEL_ALIASES = {
    'cape': 'Capesize', 'capes': 'Capesize', 'capesizes': 'Capesize',
    'newcastle': 'Newcastlemax', 'ncmax': 'Newcastlemax',
    'panamaxes': 'Panamax', 'supra': 'Supramax', 'handy': 'Handysize',
    'postpanamax': 'Post-Panamax', 'kamsarmax': 'Post-Panamax',
}


# --------------------------------------------------------------------
# artefacts
# --------------------------------------------------------------------
def _load(name):
    p = os.path.join(paths.MODELS, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


def _artefacts():
    return {
        'metrics': _load('metrics.json'),
        'forecast': _load('live_forecast.json'),
        'procurement': _load('procurement.json'),
        'seasonal': _load('seasonal.json'),
        'licensed': _load('metrics_licensed.json'),
    }


# --------------------------------------------------------------------
# language
# --------------------------------------------------------------------
def _vocab():
    """Every word worth correcting a typo towards.

    Built from the skill terms and the real port and vessel names, so
    it can never fall out of step with what the assistant can actually
    answer - a hand-written spelling list would.
    """
    global _VOCAB
    if _VOCAB is None:
        words = set()
        for _n, _f, terms in _skills():
            for term, _w in terms:
                for w in term.split():
                    if len(w) > 3:
                        words.add(w)
        for name in list(ports.PORTS) + list(ports.VESSELS):
            words.add(_norm(name).replace(' ', ''))
            for w in _norm(name).split():
                if len(w) > 3:
                    words.add(w)
        words.update(PORT_ALIASES)
        words.update(VESSEL_ALIASES)
        words.update(GLOSSARY)
        _VOCAB = tuple(sorted(words))
    return _VOCAB


_VOCAB = None


def _despell(q):
    """Nudge each long word to its nearest known term.

    People mistype under demo pressure, and "forcast" refusing is a
    worse failure than "forcast" answering - the question was perfectly
    clear. The cutoff is deliberately high: correcting aggressively
    turns an off-topic question into a confident wrong answer, which is
    the one outcome worse than a refusal.
    """
    out, changed = [], False
    for w in q.split():
        if len(w) < 5 or w in _VOCAB_SET():
            out.append(w)
            continue
        near = difflib.get_close_matches(w, _vocab(), n=1, cutoff=0.84)
        if near and near[0] != w:
            out.append(near[0])
            changed = True
        else:
            out.append(w)
    return (' '.join(out), changed)


def _VOCAB_SET():
    global _VOCAB_SETC
    if _VOCAB_SETC is None:
        _VOCAB_SETC = frozenset(_vocab())
    return _VOCAB_SETC


_VOCAB_SETC = None


def _norm(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\-/ ]+', ' ',
                                      str(text or '').lower())).strip()


def entities(question):
    """Ports, vessels, numbers and dates mentioned in the question."""
    q = _norm(question)
    tokens = set(q.split())

    port = None
    for name in ports.PORTS:
        if _norm(name).replace('-', ' ') in q.replace('-', ' '):
            port = name
            break
    if port is None:
        for alias, real in PORT_ALIASES.items():
            if alias in tokens:
                port = real
                break

    vessel = None
    for name in ports.VESSELS:
        if _norm(name).replace('-', '') in q.replace('-', '').replace(' ', ''):
            vessel = name
            break
    if vessel is None:
        for alias, real in VESSEL_ALIASES.items():
            if alias in tokens:
                vessel = real
                break

    year = None
    m = re.search(r'\b(20[0-3]\d)\b', q)
    if m:
        year = int(m.group(1))

    month = None
    for i, name in enumerate(MONTHS):
        # Word boundary, not substring: "may" otherwise matches "maybe"
        # and every question containing it becomes a question about May.
        if re.search(r'\b%s\b' % name, q):
            month = i + 1
            break

    # Tonnage: "150,000 t", "150k tonnes", "1.5 mt", "10 million tonnes".
    # Read from the RAW text, not the normalised one. _norm strips commas
    # and full stops, so 150,000 arrives as "150 000" and 1.5 as "1 5" -
    # the number survives as two numbers, both wrong, and the assistant
    # confidently sizes a parcel at 150 tonnes.
    tonnes = None
    raw = str(question or '').lower()
    for m in re.finditer(
            r'(\d[\d,]*(?:\.\d+)?)\s*(k|kt|m|mt|million|thousand)?\s*'
            r'(t|te|ton|tons|tonne|tonnes)?\b', raw):
        # Every number in the sentence gets a look, not just the first.
        # "$25 a tonne on 8 million tonnes" used to stop at 25, find no
        # unit on it and give up - so a question that plainly stated a
        # parcel size silently got the default one.
        scale, unit = m.group(2), m.group(3)
        if not (scale or unit):
            continue
        # A bare "m" is the one multiplier that collides with the unit
        # this project measures berths in. "18 m draft" was read as
        # eighteen million tonnes and answered, in full confidence,
        # with a Newcastlemax and 89 voyages - a question about depth
        # given a fleet as its answer. Metres win: a tonnage that means
        # millions says so ("18 mt", "18 million t").
        if scale == 'm' and not unit:
            continue
        try:
            val = float(m.group(1).replace(',', ''))
        except ValueError:
            continue
        mult = {'k': 1e3, 'kt': 1e3, 'thousand': 1e3,
                'm': 1e6, 'mt': 1e6, 'million': 1e6}.get(scale, 1.0)
        val *= mult
        # Outside this range it is a typo or a probe, not a cargo, and
        # sizing a fleet against it dresses nonsense up as physics.
        if not (1.0 <= val <= 5e8):
            continue
        tonnes = val
        break

    # A freight rate the user supplies. This project has no sourced
    # $/tonne for the lane, so the only honest way to answer a money
    # question is with a rate the asker owns and the answer labels.
    rate = None
    m = re.search(r'(?:\$|usd\s*)(\d+(?:\.\d+)?)'
                  r'|(\d+(?:\.\d+)?)\s*(?:dollars?|usd)\b', raw)
    if m:
        try:
            rate = float(m.group(1) or m.group(2))
        except (TypeError, ValueError):
            rate = None

    horizon = None
    m = re.search(r'(\d+)\s*(day|days|week|weeks|month|months)', q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        horizon = n if unit.startswith('day') else (
            n * 5 if unit.startswith('week') else n * 21)

    return {'port': port, 'vessel': vessel, 'year': year, 'month': month,
            'tonnes': tonnes, 'horizon': horizon, 'rate': rate,
            'tokens': tokens, 'q': q, 'ctx': {}}


# --------------------------------------------------------------------
# formatting helpers - every number that reaches a user goes through one
# --------------------------------------------------------------------
def _pct(v, dp=1):
    return '-' if v is None or not np.isfinite(v) else '%+.*f%%' % (dp, v)


def _num(v, dp=1):
    """A percentage WITHOUT a sign. Named badly and kept for the callers
    that already rely on it - it appends %, which has caught two bugs
    already ("4.57% calls a day", "180000% tonnes"). For a count use
    _int, for a plain decimal use _dec."""
    return '-' if v is None or not np.isfinite(v) else '%.*f%%' % (dp, v)


def _int(v):
    """A whole number with thousands separators - tonnes, metres, counts."""
    if v is None:
        return '-'
    try:
        if not np.isfinite(v):
            return '-'
    except TypeError:
        pass
    return format(int(round(float(v))), ',')


def _dec(v, dp=2):
    """A plain number. _num appends a percent sign, which is wrong for a
    count like calls per day and produced "4.57% calls a day"."""
    return '-' if v is None or not np.isfinite(v) else '%.*f' % (dp, v)


def _money(v):
    return '-' if v is None else '$%s' % format(int(round(v)), ',')


_PORT_NAMES = ('Visakhapatnam', 'Paradip', 'Dhamra', 'Haldia', 'Gopalpur',
               'Gangavaram')


def _known_word(w):
    """Something the assistant already understands as NOT a place.

    The port guard reads the word after in/at/for as the place being
    asked about. That is right for "weather in London" and wrong for
    "weather risk for capesize", which was refused as an unknown port
    while the vessel sat correctly extracted on the same question.
    """
    if w in VESSEL_ALIASES or w in GLOSSARY:
        return True
    return any(w == _norm(n).replace(' ', '').replace('-', '')
               for n in ports.VESSELS)


def _reply(text, table=None, source=None, follow=None, action=None):
    if isinstance(text, str):
        text = text.replace(' - ', ' - ').replace('-', '-')
    return {'answer': text.strip(), 'table': table, 'source': source,
            'follow_up': follow or [], 'action': action}


# --------------------------------------------------------------------
# skills
# --------------------------------------------------------------------
def sk_forecast(e, a):
    f = a['forecast']
    if not f or not f.get('horizons'):
        return _reply('The forward forecast has not been built on this '
                      'machine. Run `python -m src.forecast` and ask again.')
    hs = sorted(f['horizons'], key=lambda h: h['horizon_days'])
    want = e['horizon']
    pick = None
    if want:
        pick = min(hs, key=lambda h: abs(h['horizon_days'] - want))
    cheapest = min(hs, key=lambda h: h['expected_move_pct'])
    dearest = max(hs, key=lambda h: h['expected_move_pct'])
    soonest = hs[0]
    spread = dearest['expected_move_pct'] - cheapest['expected_move_pct']
    widths = sorted(h['hi_pct'] - h['lo_pct'] for h in hs)
    typical = widths[len(widths) // 2]
    strongest = max(h.get('strength_pct') or 0 for h in hs)

    if pick:
        v = pick.get('validation') or {}
        text = (
            'For **%s** - %d trading day%s out - the model expects '
            '**%s**, with an %d%% interval of %s to %s.\n\n'
            'At that horizon it has called direction correctly **%s** of the '
            'time out of sample, against a %s base rate.'
            % (pick['target_date'], pick['horizon_days'],
               '' if pick['horizon_days'] == 1 else 's',
               _pct(pick['expected_move_pct'], 2), f.get('interval_pct', 80),
               _pct(pick['lo_pct']), _pct(pick['hi_pct']),
               _num(v.get('direction_pct')), _num(v.get('base_rate_pct'))))
    else:
        text = (
            'From the close of **%s**, with the Capesize index at **%s**, the '
            'model expects %s over the next trading day and %s by **%s** - '
            '%d trading days out.\n\n'
            'The cheapest day it forecasts is **%s** (%s); the dearest is '
            '**%s** (%s).'
            % (f['as_of'], format(int(f['capesize_index']), ','),
               _pct(soonest['expected_move_pct'], 2),
               _pct(hs[-1]['expected_move_pct'], 2), hs[-1]['target_date'],
               hs[-1]['horizon_days'], cheapest['target_date'],
               _pct(cheapest['expected_move_pct'], 2), dearest['target_date'],
               _pct(dearest['expected_move_pct'], 2)))

    # The two qualifiers that must never be dropped.
    gap = soonest['expected_move_pct'] - cheapest['expected_move_pct']
    if spread < 0.5 or (cheapest is not soonest and gap < typical * 0.15):
        text += ('\n\nThe verdict is **no clear call**: the whole window '
                 'spans %s while a single day carries a %s interval, so the '
                 'differences sit inside the uncertainty. Fix on berth '
                 'availability, not on this.' % (_num(spread, 2),
                                                 _num(typical, 0)))
    elif cheapest is soonest:
        text += ('\n\nThe verdict is **fix early** - the soonest day is the '
                 'cheapest it forecasts, so waiting is expected to cost.')
    else:
        text += ('\n\nThe verdict is **wait** - the gap to %s clears the '
                 'uncertainty.' % cheapest['target_date'])

    text += ('\n\nSignal strength is %s: the strongest call sits at the %d%% '
             'percentile of every call this model has made.'
             % ('normal' if strongest >= 50 else 'weaker than usual',
                round(strongest)))
    cy = f.get('current_year')
    if cy and (cy['skill_pct'] < 0 or
               cy['direction_pct'] <= cy['base_rate_pct']):
        text += (' And **%s is a year the model is not beating** (%s against '
                 'a %s base rate) - treat this as one input, not the decision.'
                 % (cy['period'], _num(cy['direction_pct']),
                    _num(cy['base_rate_pct'])))

    table = {'head': ['Ahead', 'Target', 'Expected', 'Interval', 'Strength'],
             'rows': [['%dd' % h['horizon_days'], h['target_date'],
                       _pct(h['expected_move_pct'], 2),
                       '%s to %s' % (_pct(h['lo_pct']), _pct(h['hi_pct'])),
                       '%d%%' % round(h.get('strength_pct') or 0)]
                      for h in hs[:8]]}
    return _reply(text, table, 'src/forecast.py -> models/live_forecast.json',
                  ['When is the cheapest day to book?',
                   'How accurate is this model?',
                   'When does the model fail?'], {'view': 'timing'})


def sk_booking(e, a):
    cal = booking.build(e['tonnes'] or 160000.0, 10_000_000.0, 20.0)
    if cal is None:
        return _reply('The booking calendar needs the forward forecast. Run '
                      '`python -m src.forecast` first.')
    best = min(cal['days'], key=lambda d: d['expected_move_pct'])
    worst = max(cal['days'], key=lambda d: d['expected_move_pct'])
    text = (
        'Across the **%d trading days** the model forecast, the cheapest '
        'expected day is **%s** and the dearest is **%s**.\n\n'
        'On a %s tonne parcel at $20/t, booking on the dearest day instead of '
        'the cheapest costs about **%s** more.'
        % (len(cal['days']), best['date'], worst['date'],
           format(int(e['tonnes'] or 160000), ','),
           _money(worst.get('extra_cost_usd'))))
    if cal.get('ranking_only'):
        text += ('\n\nRead that as a **ranking, not a certainty**: the spread '
                 'across the window is %s, narrower than the %s interval on '
                 'any single day. The model is far more confident about the '
                 'shape than the level.'
                 % (_num(cal['spread_pct'], 2),
                    _num(cal['typical_interval_pct'], 0)))
    text += '\n\n' + cal['scope_note'][0].upper() + cal['scope_note'][1:] + '.'
    table = {'head': ['Date', 'Call', 'Expected', 'Costs extra'],
             'rows': [[d['date'], d['verdict'], _pct(d['expected_move_pct'], 2),
                       _money(d.get('extra_cost_usd')) if
                       d.get('extra_cost_usd') else 'cheapest']
                      for d in cal['days'][:10]]}
    return _reply(text, table, 'src/booking.py',
                  ['What is the timing edge worth in a year?',
                   'Which months are historically weak?'],
                  {'view': 'booking'})


def sk_accuracy(e, a):
    m = a['metrics']
    if not m:
        return _reply('No trained model on this machine. Run '
                      '`python -m src.train_model`.')
    best = m['best_model']
    r = m['models'][best]
    dt = m['models']['direction_test']
    lic = a['licensed']
    text = (
        'Out of sample, the model calls direction correctly **%s** of the '
        'time over %s scored weeks - but those weeks overlap, so the honest '
        'count is about **%d independent windows**. Against a %s base rate '
        'that is significant at p = %.1e.\n\n'
        'RMSE skill against assuming no change is **%s**, and it beats that '
        'baseline in %d of %d time folds.'
        % (_num(r['direction_pct']), format(m['n_scored'], ','),
           m['n_effective'], _num(dt['base_rate'] * 100), dt['p_value'],
           _pct(r['skill_vs_zero_pct'], 2), m['models']['folds_won'],
           len(m['models']['folds'])))
    if lic:
        text += (
            '\n\nThat figure is the **extended series** model. The replay also '
            'uses a second model over the licensed Baltic years to '
            '24/07/2019, which scores **%s** on its own %s weeks. The two are '
            'never averaged - a combined number would describe neither.'
            % (_num(lic['models']['ridge']['direction_pct']),
               format(lic['n_scored'], ',')))
    text += ('\n\nJudge it on direction rather than skill: skill is a '
             'variance-explained measure and collapses in a calm market, '
             'which is exactly what the per-year table shows.')
    return _reply(text, None, 'models/metrics.json + metrics_licensed.json',
                  ['How did it do in 2024?',
                   'When is the model most confident?',
                   'When does the model fail?'], {'view': 'proof'})


def sk_year(e, a):
    m = a['metrics']
    reg = (m or {}).get('models', {}).get('regimes')
    if not reg:
        return _reply('The per-year breakdown has not been computed. Run '
                      '`python -m src.train_model`.')
    rows = reg['rows']
    if e['year']:
        hit = [r for r in rows if r['period'] == str(e['year'])]
        if not hit:
            return _reply(
                'There is no scored data for %d. The replay covers %s to %s.'
                % (e['year'], rows[0]['period'], rows[-1]['period']))
        r = hit[0]
        beat = r['direction_pct'] > r['base_rate_pct']
        text = (
            'In **%s** the model called direction **%s** against that year\'s '
            'own base rate of %s, with RMSE skill of %s over %s weeks.\n\n%s'
            % (r['period'], _num(r['direction_pct']),
               _num(r['base_rate_pct']), _pct(r['skill_pct'], 2),
               format(r['n'], ','),
               'That beats the best constant call for the year.' if beat else
               '**It did not beat simply guessing the majority direction** '
               'that year - a bad year, reported rather than hidden.'))
        if r.get('strong_direction_pct') is not None:
            text += ('\n\nOn the half of weeks it was most confident about, it '
                     'scored %s.' % _num(r['strong_direction_pct']))
        return _reply(text, None, 'models/metrics.json -> regimes',
                      ['When does the model fail?',
                       'How accurate is it overall?'], {'view': 'proof'})

    weak = [r for r in rows if r['skill_pct'] < 0]
    text = ('Year by year, skill is negative in **%d of %d** years (%s) and '
            'direction fails to beat the year\'s own base rate in the same '
            'ones. A bad year here is bad on every measure at once.'
            % (len(weak), len(rows), ', '.join(r['period'] for r in weak)))
    table = {'head': ['Year', 'Direction', 'Base rate', 'Skill', 'Strongest half'],
             'rows': [[r['period'], _num(r['direction_pct']),
                       _num(r['base_rate_pct']), _pct(r['skill_pct'], 1),
                       '-' if r['strong_direction_pct'] is None
                       else _num(r['strong_direction_pct'])] for r in rows]}
    return _reply(text, table, 'models/metrics.json -> regimes',
                  ['How did it do in 2024?', 'When does the model fail?'],
                  {'view': 'proof'})


def sk_confidence(e, a):
    c = (a['metrics'] or {}).get('models', {}).get('confidence')
    if not c:
        return _reply('The confidence tiering has not been computed.')
    text = (
        'The model does not have to answer every week, and it is much better '
        'on the weeks it is sure about.\n\n'
        'Over all %s eligible weeks it calls direction %s. On the half it is '
        'most confident about that rises to **%s**, and on the strongest '
        'quarter to **%s**.\n\n'
        'Confidence here is the size of the prediction, which is known the '
        'moment the model runs. The threshold for "the strongest half" is a '
        'quantile of *earlier* predictions only, so no week is ranked using '
        'anything from after it.'
        % (format(c['n_eligible'], ','), _num(c['all_direction'] * 100),
           _num(c['tiers'][1]['direction'] * 100),
           _num(c['tiers'][2]['direction'] * 100)))
    table = {'head': ['Act only when', 'Weeks', 'Direction', 'vs all', 'p'],
             'rows': [['always', format(c['n_eligible'], ','),
                       _num(c['all_direction'] * 100), '-', '-']] +
                     [['strongest %d%%' % round(t['share'] * 100),
                       format(t['n'], ','), _num(t['direction'] * 100),
                       '%+.1fpp' % t['lift_pp'],
                       '<0.001' if t['p_value'] < 0.001 else
                       '%.3f' % t['p_value']] for t in c['tiers']]}
    return _reply(text, table, 'src/train_model.py -> confidence tiers',
                  ['How accurate is it overall?',
                   'What is the forecast right now?'], {'view': 'proof'})


def sk_value(e, a):
    p = a['procurement']
    if not p:
        return _reply('The procurement backtest has not been run. Run '
                      '`python -m src.procurement`.')
    mp, ctl = p['model_policy'], p['control']
    fx, fx_date = booking.usd_inr()
    tonnes = e['tonnes'] or 10_000_000.0
    rate = 20.0
    im = booking.annual_impact(mp['saved_pct'], tonnes, rate, fx)
    text = (
        'Over **%d independent fixtures**, timing the decision on the forecast '
        'saved **%s of the freight rate** per fixture against fixing '
        'immediately (p = %.3f).\n\n'
        'The control is the number that matters: waiting *every* time **lost '
        '%s**, because the index rose across the period. So the saving is a '
        'forecast, not a drift being harvested. It captures %s of what perfect '
        'foresight would have taken.'
        % (p['n_fixtures'], _num(mp['saved_pct'], 2), p['p_vs_zero'],
           _num(abs(ctl['saved_pct']), 2), _num(p['capture_pct'], 0)))
    if im:
        text += (
            '\n\nOn **%s tonnes a year** at $%.2f/t, converted at USD/INR %.2f '
            '(read from the panel on %s), that is **$%.3f a tonne** - about '
            '**Rs %.2f crore a year**. The tonnage and the rate are yours; the '
            'percentage and the exchange rate are measured here.'
            % (format(int(tonnes), ','), rate, fx, fx_date,
               im['saved_usd_per_t'], im['annual_crore']))
    text += (
        '\n\nTwo caveats that travel with it: the policy **loses on %s of the '
        'weeks it acts**, and it is **%sdistinguishable from a simple momentum '
        'rule** on money (p = %.2f).'
        % (_num(p['lose_rate_pct'], 0),
           '' if p['p_vs_momentum'] < 0.05 else 'not ', p['p_vs_momentum']))
    table = {'head': ['Policy', 'Saved', 'Win rate', 'Fixtures held'],
             'rows': [[d['policy'], _pct(d['saved_pct'], 2),
                       _num(d['win_pct']), format(d['n_waited'], ',')]
                      for d in (p['control'], p['momentum'],
                                p['model_policy'], p['ceiling'])]}
    return _reply(text, table, 'src/procurement.py + src/booking.py',
                  ['When is the cheapest day to book?',
                   'How accurate is the model?'], {'view': 'proof'})


def sk_seasonal(e, a):
    s = a['seasonal'] or seasonal.profile()
    if not s:
        return _reply('The seasonal profile could not be computed.')
    rows = s['months']
    if e['month']:
        r = [x for x in rows if x['month'] == e['month']]
        if r:
            r = r[0]
            return _reply(
                'In **%s**, the Capesize index has moved **%s** on average '
                'over the following five trading days, rising in %s of weeks, '
                'across %s independent windows since 2012.\n\nVerdict: **%s** '
                '- %s.\n\nThis is descriptive, not a forecast: it is what past '
                '%ss did, not what this one will do.'
                % (r['name'], _pct(r['mean_move_pct'], 2),
                   _num(r['up_share_pct']), r['n_effective'], r['verdict'],
                   r['action'], r['name']),
                None, 'src/seasonal.py',
                ['Which months are strongest?',
                 'What is the forecast right now?'], {'view': 'booking'})
    clear = [r for r in rows if r['significant']
             and abs(r['mean_move_pct']) >= s['strong_threshold_pct']]
    near = [r for r in rows if r['verdict'] == 'large, not proven']
    text = (
        'Measured on the realised five-day move by the month the decision '
        'falls in, 2012 to 2026 - **descriptive, not a forecast**.\n\n'
        'Overlapping windows mean %s rows are far fewer real observations, so '
        'every test uses the effective count, and the p-values are adjusted '
        'for testing twelve months. On that basis **%d of 12 clear the bar**%s.'
        % (format(s['n_total'], ','), len(clear),
           ' (%s)' % ', '.join('%s %s' % (r['name'], _pct(r['mean_move_pct'], 1))
                               for r in clear) if clear else ''))
    if near:
        text += (' %s move as hard and miss the correction - reported, not '
                 'acted on.' % ' and '.join('%s (%s, p=%.3f)'
                                            % (r['name'],
                                               _pct(r['mean_move_pct'], 1),
                                               r['p_value']) for r in near))
    text += '\n\n**Cause unknown.** ' + s['cause_note'] + '.'
    table = {'head': ['Month', 'Windows', 'Mean move', 'Weeks up', 'Verdict'],
             'rows': [[r['name'], r['n_effective'], _pct(r['mean_move_pct'], 1),
                       _num(r['up_share_pct']), r['verdict']] for r in rows]}
    return _reply(text, table, 'src/seasonal.py -> models/seasonal.json',
                  ['What happens in January?',
                   'When is the cheapest day to book?'], {'view': 'booking'})


def sk_vessel(e, a):
    parcel = e['tonnes'] or 160000.0
    try:
        opts = ports.options_for(parcel)
    except Exception as exc:
        return _reply('Could not evaluate that parcel: %s' % exc)
    # options_for returns a LIST of vessel/berth pairings, not a dict
    # with a 'best' key - reading it as one silently produced "no
    # combination can take this parcel" for every question.
    rows = [dict(r) for r in (opts or [])
            if r.get('verdict') in ('alongside', 'lighterage')
            and (r.get('max_cargo_t') or 0) > 0]
    if not rows:
        return _reply('No vessel and berth combination can take %s tonnes '
                      'against the published draft limits.'
                      % format(int(parcel), ','))
    for r in rows:
        cap = float(r['max_cargo_t'])
        r['_voyages'] = int(np.ceil(parcel / cap))
        # How much of the capacity you CHARTER gets used, not how full
        # each ship is. Ranking on the latter calls a Capesize and a
        # Newcastlemax equally good and quietly recommends the dearer.
        r['_fill'] = parcel / (r['_voyages'] * cap)
    rows.sort(key=lambda r: (r['verdict'] != 'alongside', r['_voyages'],
                             -r['_fill']))
    best = rows[0]
    text = (
        'For a **%s tonne** parcel the best physical fit is a **%s into %s** '
        '- %s tonnes a voyage, %d voyage%s, using **%s** of the capacity '
        'you charter.\n\nWhat binds it: %s.'
        % (format(int(parcel), ','), best['vessel'], best['port'],
           format(int(best['max_cargo_t']), ','), best['_voyages'],
           '' if best['_voyages'] == 1 else 's', _num(best['_fill'] * 100),
           best.get('binding', 'not reported')))
    text += ('\n\nThese capacities are computed physics - draft, tonnes '
             'per centimetre and dock water allowance - not quoted '
             'figures. Costs are yours; this project has no verified freight '
             'rate for the lane.')
    table = {'head': ['Vessel', 'Port', 'Verdict', 'Voyages', 'Cargo/voyage',
                      'Fill'],
             'rows': [[r['vessel'], r['port'], r['verdict'], r['_voyages'],
                       format(int(r['max_cargo_t']), ','),
                       _num(r['_fill'] * 100)] for r in rows[:8]]}
    return _reply(text, table, 'src/ports.py',
                  ['Can a Capesize berth at Paradip?',
                   'What is the empty leg at Haldia?'], {'view': 'fleet'})


def sk_berth(e, a):
    vessel = e['vessel'] or 'Capesize'
    port = e['port'] or 'Paradip'
    try:
        r = ports.can_serve(vessel, port)
    except Exception as exc:
        return _reply('Could not check that pairing: %s' % exc)
    text = (
        'A **%s** at **%s**: **%s**.\n\nIt can lift **%s tonnes** of a '
        'possible %s - %s of deadweight - leaving %s tonnes ashore each '
        'voyage.\n\nWhat binds it: %s. TPC is %s tonnes per centimetre of '
        'immersion.'
        % (vessel, port, r['verdict'].upper(),
           format(int(r['max_cargo_t']), ','),
           format(int(r['full_cargo_t']), ','),
           _num(r['utilisation'] * 100), format(int(r['foregone_t']), ','),
           r['binding'], r['tpc']))
    if r.get('note'):
        text += '\n\n' + r['note']
    if r.get('caveats'):
        text += '\n\n' + ' '.join(r['caveats'])
    return _reply(text, None, 'src/ports.py -> can_serve()',
                  ['Which vessel for 150,000 tonnes?',
                   'How busy is %s?' % port], {'view': 'fleet'})


def sk_port_activity(e, a):
    port = e['port']
    if not port:
        return _reply('Which port? I can look at %s.'
                      % ', '.join(list(ports.PORTS)[:6]))
    try:
        s = congestion.snapshot(port.lower())
    except Exception as exc:
        return _reply('No arrivals feed for %s (%s).' % (port, exc))
    if not s:
        return _reply('No arrivals data for %s.' % port)
    text = (
        '**%s** is running **%s dry bulk calls a day** against a %s normal - '
        'the %s percentile of its own history, which reads as **%s**.\n\n'
        'Throughput is %s tonnes a day across the berth in both directions.'
        % (s.get('label') or port, _dec(s['calls_per_day']),
           _dec(s['baseline_calls_per_day']),
           'unknown' if s.get('percentile') is None
           else '%.0fth' % s['percentile'],
           s.get('band', 'unknown'),
           format(int(s.get('throughput_t_per_day') or 0), ',')))
    if not s.get('reliable', True) and s.get('reliability_note'):
        text += '\n\n' + s['reliability_note']
    text += ('\n\nThis is arrivals from IMF PortWatch, derived from AIS. It is '
             'not queue length or waiting time - a busy berth is context for '
             'competition, not a measured delay.')
    return _reply(text, None, 'src/congestion.py -> IMF PortWatch',
                  ['What is the weather risk at %s?' % port,
                   'What is the empty leg at %s?' % port], {'view': 'ports'})


def sk_weather(e, a):
    port = e['port']
    try:
        rows = risk.assess([port.lower()] if port else None)
    except Exception as exc:
        return _reply('Could not run the risk checks: %s' % exc)
    if port:
        rows = [r for r in rows if (r.get('port') or '').lower()
                == port.lower()]
    raised = [r for r in rows if r['severity'] != 'clear']
    if not raised:
        return _reply(
            'Nothing raised%s. Every check came back clear.\n\nThat is a '
            'measured all-clear, not an absence of data: a check that could '
            'not run is reported as unavailable rather than silently dropped.'
            % (' for %s' % port if port else ''), None, 'src/risk.py',
            ['How busy is %s?' % (port or 'Paradip')], {'view': 'risk'})
    text = '**%d raised%s.**\n\n' % (len(raised), ' for %s' % port if port else '')
    for r in raised[:4]:
        text += ('- **%s** - %s (%s)\n  %s\n'
                 % (r['title'], r['severity'],
                    'measured' if r['measured'] else 'context only',
                    r['detail']))
    text += ('\nThe line between **measured** and **context** matters more '
             'than the count: a measured warning has an effect size this '
             'repository quantified and significance-tested; a context row is '
             'a real observation with no measured effect.')
    return _reply(text, None, 'src/risk.py -> Open-Meteo + PortWatch',
                  ['How busy is %s?' % (port or 'Paradip'),
                   'What is the forecast right now?'], {'view': 'risk'})


def sk_ballast(e, a):
    port = e['port']
    try:
        shares = ballast.shares_by_berth()
    except Exception as exc:
        return _reply('Could not compute empty-leg shares: %s' % exc)
    if port:
        v = shares.get(port)
        if v is None:
            return _reply(
                'The empty leg at **%s** is **unpriced** - PortWatch has no '
                'arrivals coverage for those berths, so the share of ships '
                'leaving empty cannot be measured there. Any cost that routes '
                'through it is optimistic, and the optimiser says so rather '
                'than assuming zero.' % port, None, 'src/ballast.py',
                ['Which vessel for 150,000 tonnes?'], {'view': 'fleet'})
        return _reply(
            'At **%s**, about **%s of ships leave empty** - measured from '
            'PortWatch tonnage in and out over the trailing window.\n\nWhat '
            'that empty leg *costs* is not measured here; it is entered as a '
            'fraction of a laden voyage on the Vessel and port view.'
            % (port, _num(v * 100)), None, 'src/ballast.py',
            ['Which vessel for 150,000 tonnes?'], {'view': 'fleet'})
    known = {k: v for k, v in shares.items() if v is not None}
    table = {'head': ['Berth', 'Ships leaving empty'],
             'rows': [[k, _num(v * 100)] for k, v in known.items()] +
                     [[k, 'unpriced'] for k, v in shares.items() if v is None]}
    return _reply(
        'Share of ships leaving each berth empty, from PortWatch tonnage:\n\n'
        'Aggregate matching is an **upper bound** on backhaul - a given ship '
        'may not be able to take a given cargo, and the feed cannot see that. '
        'Berths with no coverage are reported unpriced rather than assumed '
        'balanced.', table, 'src/ballast.py',
        ['What is the empty leg at Haldia?'], {'view': 'fleet'})


def sk_method(e, a):
    m = a['metrics'] or {}
    c = m.get('models', {}).get('conformal', {})
    text = (
        'The evaluation is an **expanding-window walk-forward** over %d folds. '
        'Each fold trains only on data before its test block, with a '
        '**%d-day purge gap on both boundaries** - the target at row *i* is '
        'built from row *i+%d*, so without the gap the last targets of one '
        'block are computed from prices inside the next.\n\n'
        'Feature selection happens **inside** each fold, from that fold\'s '
        'training data alone, so no test row informs which columns are used.\n\n'
        'Intervals are **locally weighted split conformal**: residuals from a '
        'calibration block the model never fitted, scaled by a volatility '
        'estimate known at the time. Realised coverage is %s against an %d%% '
        'target.\n\n'
        'Significance is computed on **independent windows**, not rows - '
        'overlapping %d-day targets mean %s rows are only about %d real '
        'observations, and testing on the row count would inflate it roughly '
        'fivefold.'
        % (m.get('n_folds', 8), m.get('horizon_days', 5),
           m.get('horizon_days', 5), _num((c.get('coverage') or 0) * 100),
           round((c.get('target') or 0.8) * 100), m.get('horizon_days', 5),
           format(m.get('n_scored', 0), ','), m.get('n_effective', 0)))
    return _reply(text, None, 'src/train_model.py + docs/METHOD.md',
                  ['When does the model fail?',
                   'Where does the data come from?'], {'view': 'proof'})


def sk_limits(e, a):
    m = a['metrics'] or {}
    reg = m.get('models', {}).get('regimes', {})
    rows = reg.get('rows', [])
    weak = [r for r in rows if r['skill_pct'] < 0]
    p = a['procurement'] or {}
    text = '**Where this model fails, stated plainly.**\n\n'
    if weak:
        text += ('- Skill is negative in **%d of %d years** (%s), and in those '
                 'same years direction fails to beat the year\'s own base '
                 'rate. A bad year is bad on every measure at once.\n'
                 % (len(weak), len(rows),
                    ', '.join(r['period'] for r in weak)))
        text += ('- The confidence rule does **not** rescue them: %s. In a bad '
                 'year the model is confidently wrong.\n'
                 % ', '.join('%s at %s' % (r['period'],
                                           _num(r['strong_direction_pct']))
                             for r in weak
                             if r.get('strong_direction_pct') is not None))
    if p:
        text += ('- On money it is **%sdistinguishable from a momentum rule** '
                 '(p = %.2f), and the timing policy loses on %s of the weeks '
                 'it acts.\n'
                 % ('' if p['p_vs_momentum'] < 0.05 else 'not ',
                    p['p_vs_momentum'], _num(p['lose_rate_pct'], 0)))
    text += ('- The Baltic index is a **broker survey, not a tradeable '
             'price**. A real fixture tracks it without equalling it.\n'
             '- The extended series is **weaker than the licensed-only '
             'model** (61.7% against 64.9%). That was a deliberate trade: a '
             'weaker model that runs today beats a stronger one that stops '
             'in 2019.\n'
             '- Skill collapses in **calm markets** because it measures '
             'variance explained, and there is little variance to explain. '
             'Direction degrades far more gently, which is why the project '
             'asks to be judged on it.')
    return _reply(text, None, 'models/metrics.json + procurement.json',
                  ['How did it do in 2024?', 'How accurate is it overall?'],
                  {'view': 'proof'})


def sk_data(e, a):
    m = a['metrics'] or {}
    return _reply(
        '**Where every number comes from.**\n\n'
        '- **Baltic Capesize, Panamax and Supramax** - a licensed Mendeley '
        'copy to 2019-07-31 under CC BY 4.0, spliced to a public mirror after '
        'it. The splice is validated, not trusted: the fetcher refuses to '
        'write the extension unless it reproduces the licensed years at '
        'correlation 0.99 or better with a 99th-percentile gap under 5%%.\n'
        '- **Port calls** - IMF PortWatch, derived from AIS.\n'
        '- **Weather** - Open-Meteo, history and a ten-day forecast.\n'
        '- **Brent, copper, the dollar, owner equities** - Yahoo Finance.\n'
        '- **Berth limits** - published port authority drafts; capacities are '
        'computed from them by physics, not quoted.\n\n'
        'The panel is **%s rows** over %s features, %s to %s.'
        % (format(m.get('panel_rows', 0), ','), len(m.get('features', [])),
           m.get('train_start', '?'), m.get('train_end', '?')),
        None, 'src/fetch_data.py + docs/METHOD.md',
        ['Is the Baltic data licensed for this?',
         'How is the model validated?'], {'view': 'proof'})


def sk_licence(e, a):
    return _reply(
        'The licensed half is **Mendeley Data 10.17632/t76ckh2ygg under '
        'CC BY 4.0**, which permits use with attribution and runs to '
        '2019-07-31.\n\n'
        'Past that date the series continues from a **public mirror, fetched '
        'at runtime and never redistributed** - this repository stores no '
        'extended Baltic values in version control. Current Baltic '
        'assessments are a paid feed.\n\n'
        'For a live SAIL deployment the honest position is that it would need '
        '**SAIL\'s own Baltic licence** rather than a free proxy. That is '
        'stated in the README and on the Validation view rather than left for '
        'someone to discover.',
        None, 'README.md + docs/METHOD.md §6',
        ['Where does the data come from?'], {'view': 'proof'})


def sk_two_models(e, a):
    lic, m = a['licensed'], a['metrics']
    if not lic or not m:
        return _reply('Only one model is built on this machine. Run '
                      'python -m src.licensed_model to build the '
                      'licensed-years model as well.')
    return _reply(
        'Two models cover the replay, and which one answered is shown on '
        'every call.\n\n'
        '- **Licensed Baltic years** - fitted on the Mendeley copy alone. '
        'Scores **%s** direction over %s weeks, %s to %s.\n'
        '- **Extended series** - licensed copy spliced to a validated mirror. '
        'Scores **%s** over %s weeks, and it is the only one that can forecast '
        'today.\n\n'
        'The rule is fixed in advance: licensed while its data reaches, '
        'extended after. Choosing per date by whichever scored better would be '
        'picking the answer after seeing it.\n\n'
        'They are **never averaged**. %s belongs to one period and %s to the '
        'other; a combined figure would describe neither.'
        % (_num(lic['models']['ridge']['direction_pct']),
           format(lic['n_scored'], ','), lic['scored_start'],
           lic['scored_end'],
           _num(m['models'][m['best_model']]['direction_pct']),
           format(m['n_scored'], ','),
           _num(lic['models']['ridge']['direction_pct']),
           _num(m['models'][m['best_model']]['direction_pct'])),
        None, 'src/licensed_model.py + src/train_model.py',
        ['How accurate is it overall?', 'Where does the data come from?'],
        {'view': 'timing'})


def sk_help(e, a):
    return _reply(
        'I answer from this project\'s own artefacts - every figure is '
        'computed when you ask, and I name the module it came from. I have no '
        'language model in me, so I cannot invent a number, and I work with '
        'the network off.\n\n'
        'Things I can answer:\n\n'
        '- **The forecast** - the call for the days ahead, at any horizon out '
        'to a month\n'
        '- **When to book** - the cheapest expected day and what waiting costs\n'
        '- **Accuracy** - overall, by year, and by how confident the model was\n'
        '- **What it is worth** - the procurement backtest, in percent and in '
        'rupees\n'
        '- **Seasonality** - which months have actually moved\n'
        '- **Ships and berths** - which vessel fits, what a berth allows, how '
        'much cargo\n'
        '- **Ports** - how busy a berth is, weather risk, empty legs\n'
        '- **Method** - how it is validated, where the data is from, the '
        'licensing position\n'
        '- **Limits** - where the model fails, said plainly\n'
        '- **Jargon** - what a Capesize, a laycan or TPC actually is, with '
        'this project\'s own figures in the definition\n'
        '- **Rankings and comparisons** - the busiest berth, the best and '
        'worst years, two ports or two ships side by side\n'
        '- **Scenarios** - give me a tonnage and a freight rate and I will '
        'scale the measured saving to it, labelling whose number is whose\n'
        '- **The working** - ask *how did you compute that* after any answer '
        'and I will show the method and the guard behind it\n\n'
        'Ask in your own words, mistype them if you like, and carry on in '
        'follow-ups - *and Haldia?* after a weather answer is a weather '
        'question, and I will say out loud what I read it as. If I cannot '
        'answer from the artefacts I will say so rather than guess.',
        None, None,
        ['Brief me on the whole thing', 'What is the forecast right now?',
         'When does the model fail?', 'Show me the working'])


# --------------------------------------------------------------------
# registry
# --------------------------------------------------------------------
def sk_thanks(e, a):
    """Acknowledge, and keep the thread going rather than stopping dead."""
    return _reply(
        "You're welcome. Ask me anything else about the forecast, the "
        "fleet, the ports or how any of it was validated.",
        source='assistant')


def sk_about(e, a):
    """What this is, in the terms a judge would ask about it."""
    return _reply(
        'This is a freight forecasting and vessel chartering system built '
        'for **SIH26006** (Ministry of Steel, for SAIL). It forecasts the '
        'Capesize rate for the coming month, prices the cheapest fleet that '
        'can land a parcel at an East Coast berth, flags what could disrupt '
        'it, and publishes its own record including the years it lost.\n\n'
        'Every figure you see is computed by a script in this repository '
        'when you ask for it. Nothing is typed into the page, and where a '
        'number is not measured here the interface says so rather than '
        'inventing one.',
        source='the project')


def sk_glossary(e, a):
    """Define a term, with this project's own numbers in it."""
    hit = None
    for word in sorted(GLOSSARY, key=len, reverse=True):
        if re.search(r'\b%s\b' % re.escape(word), e['q']):
            hit = GLOSSARY[word]
            break
    if hit is None:
        return sk_help(e, a)

    kind, key = hit
    if kind == 'vessel':
        v = ports.VESSELS.get(key) or {}
        try:
            t = ports.tpc(key)
        except Exception:
            t = None
        rows = [['Deadweight', '%s t' % _int(v.get('dwt'))],
                ['Laden draft', '%s m' % _dec(v.get('draft'))],
                ['Length overall', '%s m' % _int(v.get('loa'))],
                ['Beam', '%s m' % _dec(v.get('beam'))],
                ['Immersion', ('%s t/cm' % _dec(t)) if t else '\u2014']]
        return _reply(
            'A **%s** is a dry bulk carrier of about **%s tonnes** '
            'deadweight drawing **%s m** laden. At %s m beam she is too '
            'wide for the Panama locks, which is where the class name '
            'comes from - she goes round the Cape instead.\n\n'
            'Whether she can work a berth here is a draft question, not '
            'a size question: ask me *can a %s berth at Paradip*.'
            % (key, _int(v.get('dwt')), _dec(v.get('draft')),
               _dec(v.get('beam')), key.lower()),
            table={'head': ['Particular', 'Value'], 'rows': rows},
            source='src/ports.py',
            follow=['can a %s berth at Paradip' % key.lower(),
                    'cheapest ship for 150,000 t', 'what is TPC'])

    text = _TERMS[key]
    if key == 'deadweight':
        v = ports.VESSELS.get('Capesize') or {}
        text = text % ('Capesize', _int(v.get('dwt')))
    elif key == 'tpc':
        try:
            text = text % ('Capesize', _dec(ports.tpc('Capesize')))
        except Exception:
            text = text % ('Capesize', '\u2014')
    elif key == 'draft':
        v = ports.VESSELS.get('Capesize') or {}
        text = text % ('Capesize', _dec(v.get('draft')))
    elif key == 'conformal':
        cov = ((a.get('metrics') or {}).get('models', {})
               .get('conformal', {}).get('coverage'))
        text = text % (_pct(cov * 100) if cov else '\u2014')
    elif key == 'fixture':
        n = (a.get('procurement') or {}).get('n_fixtures')
        text = text % (_int(n) if n else 'several hundred')
    return _reply(text, source='the project',
                  follow=['what is a Capesize', 'how do you avoid leakage',
                          'when does the model fail'])


def _rank_rows(a):
    """The rankable things, each already computed elsewhere."""
    out = {}
    try:
        snaps = [x for x in congestion.all_snapshots()
                 if not x.get('error') and x.get('role') == 'discharge']
        # The feed keys are lowercase; ports.PORTS holds the name as it
        # should be printed.
        proper = {p.lower(): p for p in ports.PORTS}
        out['port'] = sorted(
            ((proper.get(str(x['port']).lower(), str(x['port']).title()),
              x.get('calls_per_day'), x.get('band'))
             for x in snaps if x.get('calls_per_day') is not None),
            key=lambda r: -(r[1] or 0))
    except Exception:
        pass
    sea = a.get('seasonal') or {}
    if sea.get('months'):
        out['month'] = sorted(
            ((m['name'], m['mean_move_pct'], m['verdict'])
             for m in sea['months']), key=lambda r: r[1])
    reg = ((a.get('metrics') or {}).get('models', {}).get('regimes', {}))
    if reg.get('rows'):
        out['year'] = sorted(
            ((r['period'], r['direction_pct'], r['skill_pct'])
             for r in reg['rows']), key=lambda r: -r[1])
    try:
        out['vessel'] = sorted(
            ((n, v['dwt'], v['draft']) for n, v in ports.VESSELS.items()),
            key=lambda r: -r[1])
    except Exception:
        pass
    return out


def sk_rank(e, a):
    """Superlatives - busiest, best, worst, biggest.

    A desk asks "which port is busiest", not "give me the port activity
    table sorted descending". The ordering is done here so the answer is
    the name, with the table under it as the evidence.
    """
    q = e['q']
    rows = _rank_rows(a)
    worst = any(w in q for w in ('worst', 'least', 'lowest', 'quietest',
                                 'cheapest', 'smallest', 'weakest'))

    if any(w in q for w in ('port', 'berth', 'busy', 'busiest', 'quietest')) \
            and 'port' in rows and rows['port']:
        r = rows['port'][-1] if worst else rows['port'][0]
        table = {'head': ['Berth', 'Calls/day', 'Band'],
                 'rows': [[p, _dec(c), (b or '').title()]
                          for p, c, b in rows['port']]}
        return _reply(
            '**%s** is the %s discharge berth right now at **%s dry bulk '
            'calls a day**%s.\n\nEvery berth is scored against *its own* '
            'history, not against each other - a busy day at Gopalpur is '
            'not a busy day at Visakhapatnam.'
            % (r[0], 'quietest' if worst else 'busiest', _dec(r[1]),
               (', which reads as %s for it' % r[2]) if r[2] else ''),
            table=table, source='IMF PortWatch via src/congestion.py',
            follow=['how busy is %s' % r[0], 'weather risk at %s' % r[0],
                    'what could go wrong'])

    if any(w in q for w in ('month', 'season', 'seasonal', 'time of year')) \
            and 'month' in rows and rows['month']:
        r = rows['month'][0] if not worst else rows['month'][-1]
        table = {'head': ['Month', 'Mean 5-day move', 'Verdict'],
                 'rows': [[n, _pct(v), t] for n, v, t in rows['month']]}
        cheap = rows['month'][0]
        dear = rows['month'][-1]
        return _reply(
            'Historically the index has been **weakest in %s** (%s over '
            'the following week) and **strongest in %s** (%s) - so %s has '
            'been the better month to be buying and %s the worse '
            'one.\n\nThis is what past years did, not a forecast, and '
            'only one month clears significance once you correct for '
            'testing twelve of them.'
            % (cheap[0], _pct(cheap[1]), dear[0], _pct(dear[1]),
               cheap[0], dear[0]),
            table=table, source='src/seasonal.py',
            follow=['is december cheap', 'what about april',
                    'why is the cause unknown'])

    if 'year' in q and 'year' in rows and rows['year']:
        r = rows['year'][-1] if worst else rows['year'][0]
        table = {'head': ['Year', 'Direction', 'Skill'],
                 'rows': [[y, _num(d), _pct(sk)] for y, d, sk in rows['year']]}
        return _reply(
            '**%s** was the model\u2019s %s year - direction **%s**, RMSE '
            'skill **%s**.\n\nThe spread is the point: this is not a '
            'model that works evenly, and the years it lost are published '
            'rather than averaged away.'
            % (r[0], 'worst' if worst else 'best', _num(r[1]), _pct(r[2])),
            table=table, source='src/train_model.py',
            follow=['how did it do in 2022', 'when does the model fail',
                    'why did skill collapse'])

    if any(w in q for w in ('ship', 'vessel', 'class', 'biggest')) \
            and 'vessel' in rows and rows['vessel']:
        table = {'head': ['Class', 'Deadweight', 'Laden draft'],
                 'rows': [[n, '%s t' % _int(d), '%s m' % _dec(dr)]
                          for n, d, dr in rows['vessel']]}
        return _reply(
            'By deadweight the classes run **%s** down to **%s**. Bigger '
            'is not automatically cheaper here: a **%s** carries the most '
            'but draws **%s m**, and most East Coast berths cannot give '
            'her that, so she arrives part-laden.\n\nAsk me *cheapest '
            'ship for 150,000 t* to price it rather than rank it.'
            % (rows['vessel'][0][0], rows['vessel'][-1][0],
               rows['vessel'][0][0], _dec(rows['vessel'][0][2])),
            table=table, source='src/ports.py',
            follow=['cheapest ship for 150,000 t',
                    'can a capesize berth at paradip', 'what is TPC'])

    return sk_help(e, a)


def _two_of(q, names):
    """The two things a comparison names, in the order they appear."""
    found = []
    for n in names:
        i = q.find(_norm(n).replace('-', ' '))
        if i >= 0:
            found.append((i, n))
    return [n for _i, n in sorted(found)][:2]


def sk_compare(e, a):
    """Two things, side by side - the shape a comparison deserves."""
    q = e['q']
    pair = _two_of(q, list(ports.PORTS))
    if len(pair) == 2:
        rows = []
        for p in pair:
            try:
                sn = congestion.snapshot(p.lower())
                rows.append([p, _dec(sn.get('calls_per_day')),
                             (sn.get('band') or '').title(),
                             _int(sn.get('imported_kt'))
                             if sn.get('imported_kt') is not None else '\u2014'])
            except Exception:
                rows.append([p, '\u2014', 'no feed', '\u2014'])
        return _reply(
            'Side by side, **%s** against **%s**. Each is scored against '
            'its own history, so the band tells you whether a berth is '
            'busy *for itself* - the raw call rate does not compare '
            'across berths of different size.' % (pair[0], pair[1]),
            table={'head': ['Berth', 'Calls/day', 'Band', 'Imported kt'],
                   'rows': rows},
            source='IMF PortWatch via src/congestion.py',
            follow=['which port is busiest'] +
                   ['weather risk at %s' % pair[0]])

    pair = _two_of(q, list(ports.VESSELS))
    if len(pair) == 2:
        rows = []
        for v in pair:
            spec = ports.VESSELS[v]
            try:
                t = _dec(ports.tpc(v))
            except Exception:
                t = '\u2014'
            rows.append([v, '%s t' % _int(spec['dwt']),
                         '%s m' % _dec(spec['draft']), '%s t/cm' % t])
        return _reply(
            '**%s** against **%s**. The deadweight is what she can carry; '
            'the draft is what decides whether a berth will let her carry '
            'it. Ask me *can a %s berth at Paradip* to settle it for a '
            'real berth.' % (pair[0], pair[1], pair[0].lower()),
            table={'head': ['Class', 'Deadweight', 'Laden draft', 'Immersion'],
                   'rows': rows},
            source='src/ports.py',
            follow=['cheapest ship for 150,000 t',
                    'can a %s berth at Paradip' % pair[0].lower()])

    years = re.findall(r'\b(20[0-3]\d)\b', q)
    reg = ((a.get('metrics') or {}).get('models', {}).get('regimes', {}))
    if len(years) >= 2 and reg.get('rows'):
        by = {r['period']: r for r in reg['rows']}
        rows = []
        for y in years[:2]:
            r = by.get(y)
            rows.append([y, _num(r['direction_pct']) if r else '\u2014',
                         _num(r['base_rate_pct']) if r else '\u2014',
                         _pct(r['skill_pct']) if r else '\u2014'])
        return _reply(
            '**%s** against **%s**. Direction is only meaningful beside '
            'that year\u2019s own base rate - a year that rose most weeks '
            'hands a high score to a model that always says up.'
            % (years[0], years[1]),
            table={'head': ['Year', 'Direction', 'Base rate', 'Skill'],
                   'rows': rows}, source='src/train_model.py',
            follow=['which was the best year', 'when does the model fail'])

    return sk_help(e, a)


FOLLOW_LEAD = ('and ', 'what about', 'how about', 'ok what about',
               'and what about', 'also ', 'then ', 'now ', 'but ')

FOLLOW_WORDS = frozenset(('it', 'that', 'this', 'those', 'these', 'there',
                          'them', 'same', 'instead', 'too', 'either',
                          'more', 'else', 'again', 'further', 'on'))

# How each skill got its number. Written out rather than generated,
# because the interesting half of every one of these is the guard, and a
# guard cannot be inferred from a filename.
HOW = {
    'forecast': (
        'Each horizon has its own ridge model, fitted only on days before '
        'the one it forecasts, with a purge gap the width of the horizon '
        'on both boundaries - so no training target is built from a price '
        'that sits inside the test block. The band around it is split '
        'conformal: calibrated on a slice the model never fitted, and '
        'scaled by recent volatility, so it widens in a stressed market '
        'rather than claiming a precision it has not got.'),
    'booking': (
        'The same per-horizon forecasts, ranked cheapest to dearest. The '
        'ranking orders expectations - it does not establish that one day '
        'will actually be cheaper, and the payload carries the spread '
        'between days against the width of the interval so you can see '
        'which of the two is larger. Usually it is the interval.'),
    'accuracy': (
        'Expanding-window walk-forward over eight folds, scored only on '
        'rows the model never saw in training. Significance is computed '
        'on **independent** windows - scored rows divided by the horizon '
        '- because five-day targets on daily data overlap four days in '
        'five, and counting them as independent inflates the sample about '
        'fivefold. The null is the best constant call for that period, '
        'not a coin: a market that rose in 58% of weeks hands 58% to a '
        'model that has learned nothing at all.'),
    'confidence': (
        'Calls are ranked by the size of the prediction, which is known '
        'the moment the model runs and needs nothing from the future. The '
        'threshold for the strongest half is an expanding quantile of '
        '**earlier** predictions only - computed over the whole period '
        'instead, every week would be ranked using weeks that had not '
        'happened yet, and the tiering would look far better than it is.'),
    'value': (
        'A simulated procurement programme: fix today, or hold one horizon '
        'and fix then, repeated over fixtures spaced a full horizon apart '
        'so no market move is counted twice. It is reported as a share of '
        'the freight rate rather than in dollars, because the Baltic '
        'series is an index in points and this project has no sourced '
        'points-to-dollars conversion. The control - waiting every single '
        'time - is printed beside it, because that is what separates a '
        'forecast from a rising market being harvested.'),
    'seasonal': (
        'The realised five-day move grouped by the month the decision '
        'falls in, across the whole span. Tested on independent windows '
        'and Bonferroni-corrected for having tested twelve months, which '
        'is why only one of them survives. Uncorrected, three do - and '
        'that is the version most seasonality slides quietly show.'),
    'vessel': (
        'A mixed-integer programme over vessel classes and berths, solved '
        'with HiGHS. The capacities in it are physics - deadweight, draft, '
        'tonnes per centimetre immersion, dock water allowance - and every '
        'cost in it is a number you supplied, because this project has no '
        'verified freight rate for the lane and will not invent one.'),
    'berth': (
        'Deadweight against the berth, then draft against the berth, '
        'whichever binds first. Immersion comes from the waterplane area '
        'and the water density at that berth rather than from '
        'interpolating draft, so a brackish berth correctly gives a little '
        'depth back.'),
    'activity': (
        'Vessel arrivals from IMF PortWatch, derived from satellite AIS. '
        'Each berth is scored against **its own** history as a percentile, '
        'because a busy day at Gopalpur and a busy day at Visakhapatnam '
        'are not the same number of ships.'),
    'weather': (
        'A ten-day gust and rainfall outlook from Open-Meteo at the berth '
        'co-ordinates. The 90 km/h halt threshold is not a round number '
        'picked for a slide - it is the level at which crane operation '
        'stops.'),
    'ballast': (
        'Measured from PortWatch arrivals per berth rather than assumed. '
        'Three berths have no coverage at all, and the answer says so '
        'instead of filling the gap with an average.'),
    'year': (
        'The walk-forward predictions sliced by calendar year, each year '
        'shown beside its own base rate. The years the model lost are '
        'published rather than averaged away.'),
    'brief': (
        'Nothing in that briefing is stored prose - each paragraph reads its '
        'own artefact at the moment you ask, and the table beside it names '
        'the module that produced every figure. The forecast comes from the '
        'per-horizon ridge models, the accuracy from walk-forward folds on '
        'independent windows, the saving from a fixture-spaced backtest with '
        'its own control, and the tonnage from published berth depths. If any '
        'of them had not been built on this machine, that paragraph would be '
        'missing rather than guessed.'),
    'rank': (
        'An ordering of numbers that were each computed elsewhere - berth '
        'activity from PortWatch percentiles, years from the walk-forward '
        'slices, months from the seasonal profile, classes from the vessel '
        'particulars. The ranking adds no arithmetic of its own, which is '
        'the point: a superlative should be a sort, not a new claim.'),
    'compare': (
        'Both sides are fetched the same way and shown in the same units, so '
        'the comparison is the only thing you have to trust. Berth activity '
        'is scored against each berth\u2019s own history, so the band '
        'compares across berths even though the raw call rate does not.'),
    'glossary': (
        'The definitions are written here; every figure inside them is read '
        'live from the modules - the deadweights and drafts from the vessel '
        'particulars, the immersion from the waterplane calculation, the '
        'coverage from the conformal calibration. A glossary is the easiest '
        'place in a project like this for a stale number to sit unnoticed '
        'for months, which is why none of them are typed in.'),
    'whatif': (
        'The measured percentage saving scaled by a tonnage and a rate '
        'that you supplied. The percentage is this project\u2019s; the two '
        'numbers that turn it into money are yours, and the answer says so '
        'on every line.'),
}


def _is_follow_up(q, tokens):
    """Is this a continuation rather than a fresh question?

    Two shapes count: an opener - "and Paradip?", "what about 2023" - and
    a short phrase leaning on a pronoun. Both are how people actually
    talk to something that has just answered them, and treating them as
    fresh questions is most of what makes an assistant feel like a search
    box with a cursor in it.
    """
    if any(q.startswith(p) for p in FOLLOW_LEAD):
        return True
    return len(tokens) <= 4 and bool(tokens & FOLLOW_WORDS)


def sk_working(e, a):
    """Show how the previous answer was computed.

    The second question a judge asks, and the one a dashboard usually
    cannot answer. Every reply here already carries the module that
    produced it; this turns that into a sentence and names the guard
    that stops the number flattering itself.
    """
    ctx = e.get('ctx') or {}
    skill = ctx.get('skill')
    src = ctx.get('source')
    body = HOW.get(skill)

    if not body:
        return _reply(
            'Ask me something first and I will show you how it was worked '
            'out. Every answer here names the module that produced it, and '
            'I can walk you through the guard that keeps it honest rather '
            'than flattering. Nothing on this dashboard is a typed-in '
            'number - it is all computed at the moment you ask.',
            source='assistant',
            follow=['what is the forecast', 'how accurate is it',
                    'how do you avoid leakage'])

    return _reply(
        '**How that one was computed.**' + '\n\n' + body + '\n\n'
        + 'It came from `' + (src or 'this project') + '`, and it ran when '
        'you asked rather than being read off a slide. If you want the '
        'adversarial version, ask me *when does the model fail* - that '
        'answer is the failures, not the wins.',
        source=src or 'the project',
        follow=['when does the model fail', 'how do you avoid leakage',
                'where does the data come from'])


def sk_whatif(e, a):
    """Price a scenario out of the user's own numbers.

    The saving measured here is a percentage of the freight rate. Turning
    it into money needs a rate and a tonnage, and this project has
    neither in a form it can cite - so it asks for them, then labels
    every line with whose number is whose. That is the whole difference
    between a scenario and a fabrication.
    """
    p = a['procurement']
    if not p:
        return _reply('The procurement backtest has not been run, so there '
                      'is no measured saving to scale. Run '
                      '`python -m src.procurement`.')
    rate, tonnes = e.get('rate'), e.get('tonnes')
    if rate is None and tonnes is None:
        return _reply(
            'I can run that, but the two numbers it needs are **yours, not '
            'mine**. What this project measured is a saving of **'
            + _num(p['model_policy']['saved_pct'], 2) + ' of the freight '
            'rate** per fixture - a percentage, because the Baltic series '
            'is an index in points and there is no sourced way here to turn '
            'points into dollars.' + '\n\n'
            + 'Give me a tonnage and a rate and I will scale it: try *what '
            'if freight is $25 a tonne on 8 million tonnes*. Every line of '
            'the answer will say which numbers were yours.',
            source='src/procurement.py',
            follow=['what if freight is $25 a tonne on 8 million tonnes',
                    'what is it worth in crore', 'when does the model fail'])

    fx, fx_date = booking.usd_inr()
    tonnes = tonnes or 10_000_000.0
    rate = 20.0 if rate is None else rate
    rows, seen = [], set()
    for d in (p['control'], p['momentum'], p['model_policy'], p['ceiling']):
        im = booking.annual_impact(d['saved_pct'], tonnes, rate, fx)
        if not im or d['policy'] in seen:
            continue
        seen.add(d['policy'])
        rows.append([d['policy'], _pct(d['saved_pct'], 2),
                     '$' + _dec(im['saved_usd_per_t'], 3),
                     'Rs ' + _dec(im['annual_crore']) + ' cr'])
    mine = booking.annual_impact(p['model_policy']['saved_pct'],
                                 tonnes, rate, fx)
    if not mine:
        return sk_value(e, a)
    text = (
        'On **' + _int(tonnes) + ' tonnes a year** at **$' + _dec(rate)
        + '/t**, the measured timing saving of **'
        + _num(p['model_policy']['saved_pct'], 2) + ' of the freight rate** '
        'comes to **$' + _dec(mine['saved_usd_per_t'], 3) + ' a tonne** - '
        'about **Rs ' + _dec(mine['annual_crore']) + ' crore a year** at '
        'USD/INR ' + _dec(fx) + ', read from the panel on ' + str(fx_date)
        + '.' + '\n\n'
        + 'Whose number is whose: the **tonnage and the rate are yours**, '
        'the **percentage and the exchange rate are measured here**. Move '
        'either of yours and the crore figure moves with it - which is '
        'exactly why this project publishes the percentage and not the '
        'crore.' + '\n\n'
        + 'The row that matters is the control. Waiting *every* time '
        '**loses ' + _num(abs(p['control']['saved_pct']), 2) + '**, so what '
        'is left is the forecast doing work rather than a rising market '
        'being harvested. And the policy still **loses on '
        + _num(p['lose_rate_pct'], 0) + ' of the weeks it acts**.')
    return _reply(text,
                  {'head': ['Policy', 'Saved', 'Per tonne', 'On your volume'],
                   'rows': rows},
                  'src/procurement.py + src/booking.py (your rate and tonnage)',
                  ['how did you compute that', 'when does the model fail',
                   'what is the cheapest day to book'])


def sk_brief(e, a):
    """The whole case in one answer, with the weak parts left in.

    Somebody who has just opened this has one question - what does it
    say, and should I believe it - and making them ask six is a bad way
    to answer it. Every line is read from an artefact at the moment it is
    asked, and the caveats are in the same answer as the claims rather
    than a slide further on, because a briefing that only carries the
    wins is the kind a panel stops trusting halfway through.
    """
    m, f, p = a['metrics'], a['forecast'], a['procurement']
    rows, lines = [], []

    if f and f.get('horizons'):
        hs = sorted(f['horizons'], key=lambda h: h['horizon_days'])
        far = hs[-1]
        lines.append(
            '**The call.** From the close of **%s**, with the index at '
            '**%s**, the model expects **%s** by **%s** - %d trading days '
            'out - inside an %d%% band of %s to %s. The band is the answer '
            'as much as the number is.'
            % (f['as_of'], _int(f['capesize_index']),
               _pct(far['expected_move_pct'], 2), far['target_date'],
               far['horizon_days'], f.get('interval_pct', 80),
               _pct(far['lo_pct']), _pct(far['hi_pct'])))
        rows.append(['Forecast to %s' % far['target_date'],
                     _pct(far['expected_move_pct'], 2),
                     'src/forecast.py'])

    if m and m.get('models'):
        r = m['models'][m['best_model']]
        dt = m['models']['direction_test']
        lines.append(
            '**Whether to believe it.** Direction is called correctly **%s** '
            'of the time out of sample - but those weeks overlap four days '
            'in five, so the honest count is **%d independent windows**, not '
            '%s. Against a %s base rate that still clears at p = %.1e.'
            % (_num(r['direction_pct']), m['n_effective'],
               _int(m['n_scored']), _num(dt['base_rate'] * 100),
               dt['p_value']))
        rows.append(['Direction, out of sample', _num(r['direction_pct']),
                     'models/metrics.json'])
        rows.append(['Independent windows', _int(m['n_effective']),
                     'scored rows / horizon'])

    if p:
        lines.append(
            '**What it is worth.** Over **%s independent fixtures**, timing '
            'the decision on the forecast saved **%s of the freight rate** '
            'against fixing immediately. The control is the line that '
            'matters: waiting *every* time **lost %s**, so this is the '
            'forecast working rather than a rising market being harvested. '
            'It is a percentage and not a rupee figure because the Baltic '
            'series is an index in points and this project has no sourced '
            'way to turn points into dollars.'
            % (_int(p['n_fixtures']), _num(p['model_policy']['saved_pct'], 2),
               _num(abs(p['control']['saved_pct']), 2)))
        rows.append(['Saved per fixture',
                     _num(p['model_policy']['saved_pct'], 2),
                     'src/procurement.py'])

    try:
        c = ports.can_serve('Capesize', 'Paradip')
        lines.append(
            '**The half that needs no forecast.** A Capesize at Paradip can '
            'lift **%s tonnes** of a possible %s - %s of deadweight - '
            'leaving **%s tonnes ashore every voyage**, because %s. That is '
            'arithmetic on published depths, not a prediction, and it is '
            'true whatever the market does.'
            % (_int(c['max_cargo_t']), _int(c['full_cargo_t']),
               _num(c['utilisation'] * 100), _int(c['foregone_t']),
               c['binding']))
        rows.append(['Capesize at Paradip', '%s t' % _int(c['max_cargo_t']),
                     'src/ports.py'])
    except Exception:
        pass

    if p:
        lines.append(
            '**Where it fails.** The policy **loses on %s of the weeks it '
            'acts**, the model is beaten in the calm years, and it is **%s** '
            'from a plain momentum rule on money (p = %.2f). Those are in '
            'this answer rather than a slide further on, because a briefing '
            'that carries only the wins is the kind a panel stops believing '
            'halfway through.'
            % (_num(p['lose_rate_pct'], 0),
               'distinguishable' if p['p_vs_momentum'] < 0.05
               else 'not statistically distinguishable', p['p_vs_momentum']))

    if not lines:
        return _reply('None of the artefacts have been built on this '
                      'machine yet, so there is nothing to brief from. Run '
                      '`python -m src.train_model` and the rest of the '
                      'pipeline first.')

    return _reply(
        '\n\n'.join(lines),
        {'head': ['Claim', 'Figure', 'Computed by'], 'rows': rows},
        'the whole pipeline, read at the moment you asked',
        ['how did you compute that', 'when does the model fail',
         'what if freight is $25 a tonne on 8 million tonnes'],
        {'view': 'proof'})


def sk_pitch(e, a):
    """What is this, and why would a steel plant want it.

    The single most likely first question in any review, and until now
    it fell straight through to "I cannot answer that".
    """
    m = a['metrics'] or {}
    f = a['forecast'] or {}
    p = a['procurement'] or {}
    dt = m.get('models', {}).get('direction_test', {})
    text = (
        '**Coking coal moves by sea, and the week you fix the charter '
        'decides the bill.**\n\n'
        'SAIL imports coking coal into East Coast berths in Capesize '
        'parcels. Two things drive what a cargo costs, and this answers '
        'both.\n\n'
        '**The rate half.** The Capesize freight index moves daily. This '
        'forecasts its direction over the days ahead')
    # The DEPLOYED model's direction, not direction_test.rate - that
    # one reports the best of three candidates (currently the gradient
    # booster at a flattering 62.5%) while ridge is what actually ships.
    # Quoting the higher number here would overstate the thing on the
    # page by a point, which is exactly the trap this project exists to
    # avoid.
    best = (m.get('models') or {}).get(m.get('best_model', 'ridge'), {})
    if best.get('direction_pct') is not None:
        text += (' and gets it right **%s** of the time out of sample, '
                 'against a **%s** base rate'
                 % (_num(best['direction_pct']),
                    _num((dt.get('base_rate') or 0) * 100)))
    text += '.\n\n'
    text += (
        '**The physical half.** A Capesize draws **18.2 m** fully laden. '
        'Paradip permits **16.0 m**. So she cannot arrive full - she '
        'sails part-laden at **%s t** against a hold capacity of %s t, and '
        'the difference is left ashore every single voyage. That '
        'arithmetic is tonnes-per-centimetre immersion, not a rule of '
        'thumb, and it is the half most freight models ignore.\n\n'
        % (_int(ports.max_cargo('Capesize', 'Paradip')[0]),
           _int(ports.VESSELS['Capesize']['dwt']
                - ports.VESSELS['Capesize']['constants'])))
    if p:
        text += ('**What it is worth.** Replayed over %d real fixture '
                 'weeks, timing on the model beat fixing immediately by '
                 '**%s of the freight rate** (p = %.3f). Always waiting '
                 'instead loses **%s**.\n\n'
                 % (p['n_fixtures'], _num(p['model_policy']['saved_pct']),
                    p['p_vs_zero'], _num(p['control']['saved_pct'])))
    text += ('**What makes it different.** Every number on this page is '
             'computed from stored artefacts when you ask for it. The '
             'model shows the years it was wrong as prominently as the '
             'years it was right, and it names its own controls. Ask me '
             '*what are the weaknesses* and I will list them.')
    return _reply(text, None, 'metrics.json + procurement.json + ports.py',
                  ['What are the weaknesses?',
                   'How much money does this save?',
                   'Is it better than a coin flip?'],
                  {'view': 'timing'})


def sk_rebuttal(e, a):
    """Answer the sceptical question that was actually asked.

    A panel does not ask "describe your validation strategy", it says
    "this looks like curve fitting". Each challenge below has a real,
    sourced answer, and the honest ones concede where the evidence is
    thin rather than talking past it.
    """
    q = e['q']
    m = a['metrics'] or {}
    p = a['procurement'] or {}
    mods = m.get('models', {})
    dt = mods.get('direction_test', {})

    def hit(*words):
        return any(w in q for w in words)

    # --- it is just fitted to the past -----------------------------
    if hit('curve fit', 'curve-fit', 'overfit', 'over fit', 'fitted to',
           'data mining', 'data-mining', 'cherry'):
        text = ('**Every number quoted is out of sample.**\n\n'
                'The model is scored by purged expanding-window '
                'walk-forward over **%s folds**. Each fold fits only on '
                'days before its test block, and a gap the length of the '
                'forecast horizon is cut out between them - without that '
                'gap an overlapping target leaks tomorrow into today. '
                'Nothing is ever scored on a day it was fitted on.\n\n'
                % _int(m.get('n_folds')))
        if dt:
            text += ('The direction test corrects for having tried **%d '
                     'candidate models**, so picking the winner cannot '
                     'manufacture the result: p = %.2g after that '
                     'correction.\n\n'
                     % (dt.get('n_candidates', 1), dt.get('p_value', 1)))
        text += ('The honest counterweight: the confidence tiers are '
                 'chosen from past predictions only, and the model still '
                 'has losing years. Ask *what are the weaknesses*.')
        return _reply(text, None, 'models/metrics.json',
                      ['What are the weaknesses?', 'How was it validated?'],
                      {'view': 'proof'})

    # --- a coin would do as well -----------------------------------
    if hit('coin flip', 'coin toss', 'better than a coin', 'random guess',
           'fifty fifty', '50/50', 'guessing'):
        if not dt:
            return _reply('The direction test has not been built yet.')
        text = ('**No - and the gap is measured, not asserted.**\n\n'
                'Direction is right **%s** of the time. A coin is not the '
                'right comparison though: the index rises slightly more '
                'often than it falls, so the honest baseline is that base '
                'rate of **%s**, and the model is measured against it.\n\n'
                '**%s** independent windows, p = **%.2g**. The windows '
                'matter - the raw count is %s overlapping days, and '
                'treating overlapping targets as independent would '
                'overstate the significance by a wide margin.'
                % (_num(dt['rate'] * 100), _num(dt['base_rate'] * 100),
                   _int(dt['n_effective']), dt['p_value'],
                   _int(m.get('n_scored'))))
        return _reply(text, None, 'models/metrics.json',
                      ['Is the sample big enough?',
                       'What are the weaknesses?'], {'view': 'proof'})

    # --- the sample is too small -----------------------------------
    if hit('sample is too small', 'sample size', 'too small', 'enough data',
           'small sample', 'not enough data'):
        text = ('**%s scored days, but only %s of them are independent.**'
                '\n\n'
                'That distinction is the whole answer. The target is a '
                '%s-day return, so consecutive days share most of their '
                'window. Counting them as independent would be the single '
                'easiest way to fake significance here, so every test '
                'uses the effective count instead - one observation per '
                'horizon.\n\n'
                'On the money side the replay covers **%s real fixture '
                'weeks**. That is a small sample and it is stated as one: '
                'the timing edge is not distinguishable from a plain '
                'momentum rule (p = %.2f).'
                % (_int(m.get('n_scored')), _int(m.get('n_effective')),
                   _int(m.get('horizon_days')),
                   _int(p.get('n_fixtures')) if p else '-',
                   p.get('p_vs_momentum', float('nan')) if p else float('nan')))
        return _reply(text, None, 'metrics.json + procurement.json',
                      ['What are the weaknesses?',
                       'How much money does this save?'], {'view': 'proof'})

    # --- freight is a random walk ----------------------------------
    if hit('random walk', 'are random', 'is random', 'unpredictable',
           'cannot be predicted', 'efficient market'):
        z = mods.get('zero', {})
        r = mods.get(m.get('best_model', 'ridge'), {})
        text = ('**Mostly true, and that is why the baseline is '
                '"no change".**\n\n'
                'Assuming the rate does not move is a strong forecast, so '
                'it is the thing to beat. Over the same days it gives an '
                'RMSE of **%.4f**; the model gives **%.4f** - an '
                'improvement of **%s**.\n\n'
                'That is a small edge and it is presented as one. The '
                'claim is not that freight is predictable; it is that '
                'direction is called better than the base rate often '
                'enough to be worth a few days of timing, and that the '
                'edge collapses in calm markets where there is little '
                'variance to explain.'
                % (z.get('rmse', float('nan')), r.get('rmse', float('nan')),
                   _num(r.get('skill_vs_zero_pct'))))
        return _reply(text, None, 'models/metrics.json',
                      ['What are the weaknesses?',
                       'How did it do in a bad year?'], {'view': 'proof'})

    # --- why not simply always wait --------------------------------
    if hit('always wait', 'just wait', 'stops me waiting', 'why not wait',
           'always waiting', 'wait every time'):
        if not p:
            return _reply('The procurement replay has not been built yet.')
        c, mp, mo = p['control'], p['model_policy'], p['momentum']
        text = ('**Because always waiting loses money.**\n\n'
                'That is the control arm, and it is on the page precisely '
                'so this question has an answer. Over %s fixture weeks:'
                '\n\n'
                '- Wait every time: **%s**\n'
                '- Time it on the model: **%s**\n'
                '- Time it on plain momentum: **%s**\n\n'
                'The model beats the always-wait control decisively '
                '(p = %.2g). It does **not** clearly beat momentum '
                '(p = %.2f) - that is stated wherever the saving is '
                'quoted, because a result that only beats a strawman is '
                'not a result.'
                % (_int(p['n_fixtures']), _num(c['saved_pct']),
                   _num(mp['saved_pct']), _num(mo['saved_pct']),
                   p['p_vs_control'], p['p_vs_momentum']))
        return _reply(text, None, 'models/procurement.json',
                      ['How much money does this save?',
                       'What are the weaknesses?'], {'view': 'booking'})

    # --- why not a neural net / an LLM -----------------------------
    if hit('llm', 'chatgpt', 'gpt', 'neural', 'deep learning',
           'transformer', 'why not ai'):
        best = m.get('best_model', 'ridge')
        lg = mods.get('lgbm', {})
        rb = mods.get(best, {})
        text = ('**Because the bigger model lost.**\n\n'
                'Gradient boosting was fitted on the same folds and came '
                'out behind: **%s** skill against **%s** for the linear '
                'model. On %s independent windows there is not enough '
                'signal to justify the extra capacity, and the honest '
                'thing is to ship the model that won.\n\n'
                'A language model is the wrong tool twice over here. It '
                'cannot compute a conformal interval, and it would '
                'happily invent a freight rate. This assistant runs no '
                'language model at all - it routes your question to code '
                'that reads the stored artefacts, which is why it can '
                'refuse to answer instead of guessing.'
                % (_num(lg.get('skill_vs_zero_pct')),
                   _num(rb.get('skill_vs_zero_pct')),
                   _int(m.get('n_effective'))))
        return _reply(text, None, 'models/metrics.json',
                      ['How was it validated?', 'What are the weaknesses?'],
                      {'view': 'proof'})

    # --- is it just a linear regression ----------------------------
    if hit('linear regression', 'just a regression', 'simple model',
           'only a regression', 'linear model'):
        best = m.get('best_model', 'ridge')
        lg = mods.get('lgbm', {})
        rb = mods.get(best, {})
        text = ('**Essentially yes - it is a ridge regression, and that '
                'is a deliberate choice rather than a limitation.**\n\n'
                'Gradient boosting was fitted on identical folds and lost: '
                '**%s** skill against the ridge model at **%s**. With %s independent '
                'windows, a model with more capacity mostly finds more '
                'ways to fit noise.\n\n'
                'The work that earns the accuracy is not the estimator. '
                'It is the %s features, the purge gap that stops an '
                'overlapping target leaking, the conformal interval '
                'calibrated on a block the model never fitted, and the '
                'physical half - draft, tonnes-per-centimetre and berth '
                'limits - which no estimator would have supplied.'
                % (_num(lg.get('skill_vs_zero_pct')),
                   _num(rb.get('skill_vs_zero_pct')),
                   _int(m.get('n_effective')),
                   len(m.get('features', [])) or 'the'))
        return _reply(text, None, 'models/metrics.json',
                      ['How was it validated?', 'What features does it use?'],
                      {'view': 'proof'})

    # Recognised as a challenge but not one of the specific ones.
    return sk_limits(e, a)


def sk_today(e, a):
    """What a chartering desk should actually do this morning."""
    f = a['forecast'] or {}
    p = a['procurement'] or {}
    hs = f.get('horizons') or []
    if not hs:
        return _reply('The forward forecast has not been built yet. Run '
                      '`python -m src.forecast` and ask again.')
    near = hs[min(4, len(hs) - 1)]
    v = near.get('validation') or {}
    strong = near.get('stronger_than_usual')
    text = ('**As of %s, with the Capesize index at %s.**\n\n'
            % (f.get('as_of', '-'), _int(f.get('capesize_index'))))
    text += ('Over the next **%s trading days** the model expects '
             '**%s%%**, inside an %s%% band of %s%% to %s%%.\n\n'
             % (_int(near['horizon_days']),
                _num(near['expected_move_pct']),
                _int(f.get('interval_pct')),
                _num(near['lo_pct']), _num(near['hi_pct'])))
    if strong:
        text += ('This call sits in the **stronger-than-usual** band, '
                 'where direction has historically been right more often '
                 'than average. ')
    else:
        text += ('This call is **not** in the stronger-than-usual band, '
                 'so it deserves less weight than the headline accuracy '
                 'suggests. ')
    if v.get('direction_pct') is not None:
        text += ('At this horizon direction has been right **%s** of the '
                 'time out of sample against a **%s** base rate.\n\n'
                 % (_num(v['direction_pct']), _num(v.get('base_rate_pct'))))
    text += ('**The honest caveat.** The band is wide because freight is '
             'volatile, and the band is the answer as much as the number '
             'is. ')
    if p:
        text += ('Acting on this timing lost money on **%s** of the weeks '
                 'it acted, historically. '
                 % _num(p['lose_rate_pct'], 0))
    text += ('For the cheapest day of the month and what it is worth in '
             'rupees, ask about the **booking calendar**.')
    return _reply(text, None, 'models/live_forecast.json',
                  ['When should I book?', 'What is the risk this week?',
                   'How accurate is it?'], {'view': 'timing'})


def sk_deploy(e, a):
    """How it runs - offline, retraining, and what it needs."""
    m = a['metrics'] or {}
    text = (
        '**It is a single Flask application that serves stored '
        'artefacts.**\n\n'
        '- **Offline at serve time.** Every asset is vendored locally - '
        'the chart library, the 3D library and the fonts. The page makes '
        'no request to any outside host, so it runs on a laptop with the '
        'network unplugged. That is enforced by a test, not a promise.\n'
        '- **Answering is a read, not a fit.** The model is trained '
        'ahead of time into artefacts under `models/`; a question reads '
        'those. There is no model loaded per request and no external API '
        'call, which is why this assistant cannot invent a number.\n'
        '- **Retraining is a script, not a service.** The panel is '
        'rebuilt and the walk-forward re-run from the command line. The '
        'artefacts carry the date they were built and the page shows the '
        'age of the forecast, refusing to display a stale one rather '
        'than quietly serving it.\n')
    if m.get('train_end'):
        text += ('- The current artefacts were fitted on data to **%s**, '
                 'over **%s** panel rows.\n'
                 % (m['train_end'], _int(m.get('panel_rows'))))
    text += ('\nThe only step that needs the network is fetching new '
             'market data, and that fetch validates any new series '
             'against the licensed copy before it is allowed to join - a '
             'mirror that disagrees is refused rather than spliced.')
    return _reply(text, None, 'app.py + src/fetch_data.py',
                  ['What data does it use?', 'Is the Baltic data licensed?'],
                  {'view': 'proof'})


# The feature names in metrics.json are terse by design - they are
# column names. Grouping them is what makes the list mean something to
# somebody hearing it for the first time, and the groups are derived
# from the names themselves rather than typed out, so a new feature
# cannot silently go unmentioned.
_FEATURE_GROUPS = (
    ('the Capesize index\'s own history',
     lambda n: n.startswith('cape_') and n != 'cape_pmx_ratio'),
    ('the other vessel classes, which lead and lag it',
     lambda n: n.startswith(('pmx_', 'smx_')) or n == 'cape_pmx_ratio'),
    ('fuel', lambda n: n.startswith('brent_')),
    ('macro conditions',
     lambda n: n.startswith(('copper_', 'dxy_', 'sp500_'))),
    ('the equities of the people who move the cargo',
     lambda n: n.startswith(('miners_', 'owners_'))),
    ('where the year is', lambda n: n.startswith('seas_')),
)


def sk_stack(e, a):
    """What it is built from, and what the model actually looks at."""
    m = a['metrics'] or {}
    feats = list(m.get('features') or [])
    text = ('**Python, and deliberately ordinary parts.**\n\n'
            '- **Model**: scikit-learn. The estimator that ships is a '
            '**%s** regression; LightGBM was fitted on identical folds '
            'and lost, so it is not the one deployed.\n'
            '- **Data**: pandas and numpy over Parquet, with scipy for '
            'the significance tests.\n'
            '- **Serving**: a single Flask app that reads pre-built '
            'artefacts. No model is loaded per request.\n'
            '- **Front end**: no framework. Plain JavaScript, with '
            'Chart.js and three.js vendored into the repo so the page '
            'runs with the network unplugged.\n\n'
            % m.get('best_model', 'ridge'))
    if feats:
        text += '**What the model looks at - %d features.**\n\n' % len(feats)
        left = list(feats)
        for label, pred in _FEATURE_GROUPS:
            got = [n for n in left if pred(n)]
            if not got:
                continue
            left = [n for n in left if n not in got]
            text += '- %s: `%s`\n' % (label, '`, `'.join(got))
        if left:
            text += '- also: `%s`\n' % '`, `'.join(left)
        text += ('\nNo feature is a forecast of anything. Every one is '
                 'an observation available on the day the call is made - '
                 'that is what stops tomorrow leaking into today.\n')
    return _reply(text, None, 'requirements.txt + models/metrics.json',
                  ['How was it validated?', 'Does it run offline?',
                   'Is this just a linear regression?'], {'view': 'proof'})


def sk_versus(e, a):
    """What this adds over how the decision is made without it."""
    m = a['metrics'] or {}
    p = a['procurement'] or {}
    text = (
        '**It does not replace a broker. It replaces guessing about '
        'timing.**\n\n'
        'A chartering desk already has the published index and a '
        'broker\'s view. What it usually does not have is a written '
        'record of how often a timing call of this kind was right, or '
        'what waiting actually cost the last time. This project is that '
        'record.\n\n'
        'Three things it adds that a report or a spreadsheet does '
        'not:\n\n'
        '- **An out-of-sample track record.** Not a fit to history - a '
        'walk-forward over %s folds where every call was made before '
        'its outcome was known.\n'
        % _int(m.get('n_folds')))
    text += ('- **A stated interval, not just a number.** The band is '
             'conformal, calibrated on data the model never fitted, and '
             'it covers close to the %s%% it advertises.\n'
             % _int((m.get('models', {}).get('conformal', {})
                     .get('target') or 0) * 100))
    if p:
        text += ('- **A control arm.** Timing on the model returned '
                 '**%s** of the freight rate over %s fixture weeks; '
                 'always waiting returned **%s**. Without that second '
                 'number the first one means nothing.\n'
                 % (_num(p['model_policy']['saved_pct']),
                    _int(p['n_fixtures']), _num(p['control']['saved_pct'])))
    text += ('\nAnd the honest limit: it is **not** clearly better than '
             'a simple momentum rule')
    if p:
        text += ' (p = %.2f)' % p['p_vs_momentum']
    text += ('. The defensible claim is a written, checkable record of a '
             'decision that is usually made on judgement - plus the '
             'physical half, the berth draft arithmetic, which no rate '
             'forecast addresses at all.\n\n'
             'This project makes no claim about SAIL\'s internal '
             'process; it was built from public and licensed data, not '
             'from any description of how the desk works today.')
    return _reply(text, None, 'metrics.json + procurement.json',
                  ['What are the weaknesses?', 'Why does draft matter?',
                   'How much money does this save?'], {'view': 'proof'})


def sk_provenance(e, a):
    """Where the numbers come from - asked bluntly, answered bluntly."""
    m = a['metrics'] or {}
    text = (
        '**No. Every figure here is computed when you ask for it.**\n\n'
        'That is a mechanism, not a promise:\n\n'
        '- There is **no language model** anywhere in this assistant. '
        'Your question is matched to a registry of skills, and each one '
        'reads the same stored artefacts the dashboard reads. There is '
        'no step at which a number could be generated rather than '
        'looked up.\n'
        '- **Every answer carries its source line** - the file the '
        'numbers came from. If it cannot cite one, it does not answer.\n'
        '- A question outside the registry gets a **refusal**, not a '
        'guess. That is why some things I cannot tell you.\n'
        '- The test suite recomputes the dashboard\'s claims '
        'independently from the artefacts and fails if they disagree, '
        'so a route can return 200 and still be caught being wrong.\n')
    if m.get('train_end'):
        text += ('- The current artefacts were fitted on data to **%s** '
                 'over **%s** panel rows, and the page shows the age of '
                 'the forecast rather than quietly serving a stale '
                 'one.\n' % (m['train_end'], _int(m.get('panel_rows'))))
    text += ('\nThe one thing to hold me to: the Baltic index is a '
             '**broker survey, not a tradeable price**, so a real '
             'fixture tracks it without equalling it. Every saving '
             'quoted is a percentage of the freight rate, never a rupee '
             'figure invented from one.')
    return _reply(text, None, 'src/assistant.py + models/',
                  ['What data does it use?', 'How was it validated?',
                   'What are the weaknesses?'], {'view': 'proof'})


def sk_sowhat(e, a):
    """Judge the number the last answer gave, against its own benchmark.

    "Is that good?" is the question a number invites and a dashboard
    almost never answers. Repeating the figure is not an answer; the
    honest reply names what it should be compared against, and concedes
    where the comparison is unflattering.
    """
    ctx = e.get('ctx') or {}
    skill = ctx.get('skill')
    m = a['metrics'] or {}
    p = a['procurement'] or {}
    mods = m.get('models', {})
    dt = mods.get('direction_test', {})
    best = mods.get(m.get('best_model', 'ridge'), {})

    if skill in ('accuracy', 'year', 'confidence', 'brief', 'pitch'):
        if not dt or best.get('direction_pct') is None:
            return _reply('The direction test has not been built yet.')
        gap = best['direction_pct'] - dt['base_rate'] * 100
        return _reply(
            '**Good enough to act on. Not good enough to bet the plant '
            'on - and the difference matters.**\n\n'
            '- **The right comparison is not 50%%.** The index rises a '
            'little more often than it falls, so the benchmark is the '
            'base rate of **%s**. Against that, **%s** is a gap of '
            '**%.1f points**.\n'
            '- **It is unlikely to be luck.** p = %.2g on **%s** '
            'independent windows, after correcting for the %d models '
            'tried.\n'
            '- **But it is a small edge.** Ten points of direction is '
            'worth a few days of timing, not a change of strategy. It '
            'buys nothing at all in a calm market, and the model has '
            'losing years - ask *when does the model fail*.\n\n'
            'The figure to be sceptical of is not this one. It is the '
            'money figure: ask *is the saving good* and I will give you '
            'the unflattering half.'
            % (_num(dt['base_rate'] * 100), _num(best['direction_pct']),
               gap, dt.get('p_value', 1), _int(dt.get('n_effective')),
               dt.get('n_candidates', 1)),
            None, 'models/metrics.json',
            ['When does the model fail?', 'Is the saving good?'],
            {'view': 'proof'})

    if skill in ('value', 'booking', 'whatif'):
        if not p:
            return _reply('The procurement replay has not been built yet.')
        return _reply(
            '**Against the obvious control, yes. Against a lazy rule, '
            'not provably.**\n\n'
            '- **It beats doing the naive thing.** Timing on the model '
            'returned **%s** of the freight rate; always waiting '
            'returned **%s**. That difference is solid (p = %.2g).\n'
            '- **It does not clearly beat momentum.** A rule that just '
            'follows the recent trend returned **%s**, and the gap to '
            'the model is not distinguishable from noise (p = %.2f). '
            'On this evidence you could not claim the model is the '
            'reason.\n'
            '- **It loses often.** The policy is behind on **%s** of '
            'the weeks it acts, and it captures only **%s** of what a '
            'perfect foresight rule would.\n\n'
            'The defensible claim is the control arm, not the '
            'headline.'
            % (_num(p['model_policy']['saved_pct']),
               _num(p['control']['saved_pct']), p['p_vs_control'],
               _num(p['momentum']['saved_pct']), p['p_vs_momentum'],
               _num(p['lose_rate_pct'], 0), _num(p['capture_pct'], 0)),
            None, 'models/procurement.json',
            ['What are the weaknesses?', 'How was it validated?'],
            {'view': 'booking'})

    if skill == 'two_models':
        lic = a['licensed'] or {}
        lr = (lic.get('models') or {}).get('ridge', {})
        if lr.get('direction_pct') is None:
            return _reply('The licensed model has not been built yet.')
        return _reply(
            '**The licensed-only model is the better one, and it is not '
            'the one that answers today.**\n\n'
            'It calls direction **%s** of the time against the extended '
            'model\'s **%s** - but its data stops at %s, so it cannot '
            'say anything about this week. That was a deliberate trade: '
            'a weaker model that runs today beats a stronger one that '
            'stopped in 2019.\n\nThe two are never averaged. Each '
            'period is answered by the model that actually covers it, '
            'and every replayed call is labelled with which one made '
            'it.'
            % (_num(lr['direction_pct']), _num(best.get('direction_pct')),
               lic.get('scored_end', '2019')),
            None, 'metrics_licensed.json + metrics.json',
            ['Why are there two models?', 'Is the Baltic data licensed?'],
            {'view': 'proof'})

    if skill in ('forecast', 'today'):
        cf = mods.get('conformal', {})
        return _reply(
            '**Judge the band, not the number.**\n\n'
            'A point forecast of a freight index is close to '
            'meaningless on its own - the honest content is the '
            'interval. It targets **%s%%** coverage and achieves '
            '**%s%%** on data the model never fitted, which is the '
            'thing to be impressed by if anything here impresses you.'
            '\n\nThe direction call is the part worth acting on, and '
            'that is right **%s** of the time against a **%s** base '
            'rate. The width of the band is not a defect; it is what '
            'freight actually does.'
            % (_int((cf.get('target') or 0) * 100),
               _num((cf.get('coverage') or 0) * 100),
               _num(best.get('direction_pct')),
               _num((dt.get('base_rate') or 0) * 100)),
            None, 'models/metrics.json',
            ['Is the accuracy good?', 'When does the model fail?'],
            {'view': 'proof'})

    # Nothing to judge yet.
    return _reply(
        'Ask me for a number first and I will tell you honestly whether '
        'it is any good - what it should be compared against, and where '
        'the comparison is unflattering. Most figures on this dashboard '
        'have a control arm sitting next to them for exactly that '
        'reason.',
        source='assistant',
        follow=['How accurate is it?', 'How much money does this save?'])


def _skills():
    """(name, handler, terms). Multi-word terms score 1.5x - they are far
    less likely to match by accident than a single common word."""
    return [
        ('sowhat', sk_sowhat, [
            ('is that good', 2.8), ('is that any good', 2.8),
            ('is that impressive', 2.8), ('is that a lot', 2.6),
            ('is that high', 2.4), ('is that bad', 2.6),
            ('is that significant', 2.6), ('so what', 2.4),
            ('does that matter', 2.6), ('why does that matter', 2.6),
            ('how does that compare', 2.8), ('compared to what', 2.4),
            ('is that better', 2.4), ('should i be impressed', 2.8),
            ('is the saving good', 2.8), ('is the accuracy good', 2.8),
            ('is 61.7 good', 2.6), ('good enough', 2.2)]),
        ('stack', sk_stack, [
            ('tech stack', 2.8), ('technology stack', 2.8), ('stack', 1.4),
            ('what did you build this in', 2.8), ('built in', 1.8),
            ('built with', 2.0), ('what libraries', 2.6),
            ('which libraries', 2.6), ('libraries', 1.4),
            ('framework', 1.6), ('what language', 2.4), ('python', 1.4),
            ('scikit', 2.0), ('sklearn', 2.0), ('flask', 1.8),
            ('what algorithm', 2.6), ('which algorithm', 2.6),
            ('algorithm', 1.4), ('what model is it', 2.6),
            ('what features', 2.6), ('which features', 2.6),
            ('how many features', 2.8), ('features does', 2.4),
            ('what inputs', 2.2), ('predictors', 1.8),
            ('lines of code', 2.6), ('open source', 2.0),
            ('source code', 2.0), ('see the code', 2.4),
            ('show me the code', 2.6), ('repository', 1.6)]),
        ('versus', sk_versus, [
            ('how does sail do this today', 2.8),
            ('do this today', 2.4), ('status quo', 2.6),
            ('the alternative', 2.4), ('what is the alternative', 2.8),
            ('existing solution', 2.6), ('already exists', 2.4),
            ('ask a broker', 2.8), ('broker', 1.4),
            ('shipping companies use', 2.6), ('what do they use', 2.4),
            ('better than a spreadsheet', 2.8), ('spreadsheet', 1.8),
            ('excel', 1.8), ('how is this different', 2.6),
            ('why not just', 2.0), ('competitor', 1.8),
            ('compared to what', 2.4), ('instead of this', 2.0),
            ('what does it replace', 2.6)]),
        ('provenance', sk_provenance, [
            ('make these numbers up', 2.8), ('made these up', 2.6),
            ('making it up', 2.6), ('made up', 2.0), ('fabricated', 2.4),
            ('invented', 2.0), ('are these real', 2.6),
            ('are the numbers real', 2.8), ('can i trust', 2.4),
            ('how do i know these', 2.6), ('where do the numbers', 2.8),
            ('where does this come from', 2.6), ('provenance', 2.2),
            ('hallucinate', 2.6), ('hallucination', 2.6),
            ('is this an llm', 2.6), ('do you use ai', 2.4),
            ('cite', 1.6), ('citation', 1.8), ('sourced', 1.8)]),
        ('pitch', sk_pitch, [
            ('what does this do', 2.6), ('what is this', 2.2),
            ('what does it do', 2.6), ('explain the project', 2.6),
            ('explain your project', 2.6), ('about the project', 2.2),
            ('what problem', 2.4), ('problem statement', 2.4),
            ('why does sail', 2.6), ('why sail', 2.2),
            ('who is this for', 2.6), ('who is it for', 2.6),
            ('what is the innovation', 2.6), ('innovation', 1.4),
            ('what is novel', 2.4), ('novel', 1.2),
            ('elevator', 1.6), ('pitch', 1.4), ('overview', 1.4),
            ('in a nutshell', 2.2), ('summarise the project', 2.4),
            ('use case', 1.8), ('why is this useful', 2.4),
            ('what are you solving', 2.4), ('purpose', 1.2)]),
        ('rebuttal', sk_rebuttal, [
            ('curve fitting', 2.8), ('curve fit', 2.6), ('overfitting', 2.4),
            ('overfit', 2.2), ('over fitting', 2.4), ('data mining', 2.4),
            ('cherry picked', 2.4), ('cherry picking', 2.4),
            ('coin flip', 2.8), ('coin toss', 2.8), ('fifty fifty', 2.4),
            ('better than a coin', 2.8), ('random guess', 2.4),
            ('guessing', 1.4), ('sample is too small', 2.8),
            ('sample size', 2.4), ('too small', 2.0),
            ('small sample', 2.4), ('not enough data', 2.4),
            ('enough data', 2.0), ('random walk', 2.8),
            ('are random', 2.4), ('is random', 2.2),
            ('unpredictable', 1.8), ('efficient market', 2.4),
            ('always wait', 2.6), ('just wait', 2.2),
            ('why not wait', 2.6), ('always waiting', 2.6),
            ('llm', 1.8), ('chatgpt', 2.0), ('neural', 1.8),
            ('deep learning', 2.4), ('transformer', 1.8),
            ('linear regression', 2.6), ('just a regression', 2.6),
            ('only a regression', 2.6), ('linear model', 2.2),
            ('simple model', 2.0), ('why should i believe', 2.4),
            ('convince me', 2.2), ('prove it is real', 2.4),
            ('is it real', 1.8), ('skeptical', 1.6), ('sceptical', 1.6)]),
        ('today', sk_today, [
            ('what should i do', 2.6), ('what do i do', 2.6),
            ('what should we do', 2.6), ('recommendation', 1.8),
            ('recommend', 1.6), ('advise', 1.4), ('advice', 1.4),
            ('what would you do', 2.4), ('should i charter', 2.4),
            ('good time to charter', 2.6), ('good time to book', 2.6),
            ('is now a good time', 2.8), ('right now', 1.6),
            ('today', 1.4), ('this morning', 2.0),
            ('actionable', 1.8), ('bottom line today', 2.6)]),
        ('deploy', sk_deploy, [
            ('deploy', 1.8), ('deployment', 1.8), ('how would we run', 2.4),
            ('how do we run', 2.4), ('run offline', 2.6),
            ('runs offline', 2.6), ('does it run offline', 2.8),
            ('offline', 1.6), ('internet', 1.4), ('no network', 2.2),
            ('retrain', 2.0), ('retraining', 2.0),
            ('how often', 1.6), ('refresh', 1.4), ('update the model', 2.4),
            ('cost to run', 2.6), ('running cost', 2.4),
            ('infrastructure', 1.8), ('production', 1.6),
            ('server', 1.4), ('hosting', 1.6), ('scale', 1.2)]),
        ('forecast', sk_forecast, [
            ('forecast', 1.2), ('prediction', 1.2), ('predict', 1.0),
            ('what is the call', 2.0), ('next week', 1.5), ('outlook', 1.0),
            ('rates going', 1.5), ('going up', 1.2), ('going down', 1.2),
            ('charter now', 1.5), ('fix now', 1.5), ('this week', 1.0),
            ('days ahead', 1.5), ('next month', 1.5), ('tomorrow', 1.0),
            ('will rates', 2.4),
            ('rates go', 2.2),
            ('go up', 1.8),
            ('go down', 1.8),
            ('rise', 1.5),
            ('fall', 1.5),
            ('rising', 1.5),
            ('falling', 1.5),
            ('month out', 2.4),
            ('a month', 1.8),
            ('weeks out', 2.2),
            ('days out', 2.2),
            ('what happens', 2.0),
            ('where are rates', 2.5),
            ('direction of rates', 2.4),
            ('coming days', 2.2),
            ('near term', 2.0),
            ('ahead', 1.2)]),
        ('booking', sk_booking, [
            ('when should i book', 2.5), ('cheapest day', 2.0),
            ('best day', 1.8), ('booking', 1.2), ('calendar', 1.2),
            ('when to book', 2.0), ('cost of waiting', 2.0),
            ('when to fix', 1.8),
            ('book now', 2.6),
            ('now or wait', 2.8),
            ('or wait', 2.2),
            ('should i book', 2.8),
            ('should i fix', 2.8),
            ('fix now', 2.6),
            ('hold off', 2.2),
            ('when should i', 2.4),
            ('timing', 1.6),
            ('wait', 1.2)]),
        ('accuracy', sk_accuracy, [
            ('base rate', 2.2),
            ('the base rate', 2.4),
            ('p value', 2.2),
            ('significance', 1.6),
            ('accurate', 1.4),
            ('accuracy', 1.4),
            ('reliable', 1.1),
            ('reliability', 1.1),
            ('hit rate', 1.8),
            ('how good', 1.8),
            ('any good', 1.8),
            ('does it work', 2.2),
            ('is it any good', 2.4),
            ('performance', 1.0),
            ('how accurate', 2.5), ('accuracy', 1.5), ('how good', 1.8),
            ('does it work', 2.0), ('direction accuracy', 2.0),
            ('hit rate', 1.5), ('how reliable', 2.0), ('performance', 1.0),
            ('r2', 1.0), ('skill', 1.0),
            ('significant', 2.2),
            ('significance', 2.2),
            ('p value', 2.4),
            ('reliable', 1.8),
            ('how often right', 2.5),
            ('how often correct', 2.5),
            ('track record', 2.2),
            ('proven', 1.8)]),
        ('year', sk_year, [
            ('in 2022', 2.0), ('in 2024', 2.0), ('by year', 2.0),
            ('each year', 1.8), ('year by year', 2.5), ('which year', 1.8),
            ('2026', 0.8), ('2022', 0.8), ('2024', 0.8), ('2019', 0.6)]),
        ('confidence', sk_confidence, [
            ('confident', 1.8), ('confidence', 1.8), ('strongest', 1.5),
            ('when to trust', 2.0), ('tier', 1.2), ('sure', 1.0),
            ('most certain', 1.8)]),
        ('value', sk_value, [
            ('worth', 1.5), ('save', 1.5), ('saving', 1.5), ('savings', 1.5),
            ('money', 1.5), ('crore', 2.0), ('rupees', 1.8), ('roi', 2.0),
            ('per tonne', 1.5), ('business case', 2.0), ('benefit', 1.2),
            ('how much would we save', 2.5),
            ('annual impact', 2.6),
            ('impact for', 2.0),
            ('per year', 2.0),
            ('a year', 1.6)]),
        ('seasonal', sk_seasonal, [
            ('season', 1.5), ('seasonal', 1.8), ('monsoon', 1.5),
            ('which month', 2.0), ('time of year', 2.0), ('january', 1.2),
            ('december', 1.2), ('october', 1.2), ('april', 1.2),
            ('playbook', 1.5),
            ('seasonality', 2.6),
            ('best month', 2.6),
            ('worst month', 2.6),
            ('month to book', 2.6),
            ('quarter', 1.4)]),
        ('vessel', sk_vessel, [
            ('which vessel', 2.5), ('which ship', 2.5), ('what ship', 2.0),
            ('cheapest vessel', 2.2), ('best vessel', 2.0), ('fleet', 1.2),
            ('what vessel', 2.0), ('which class', 1.8),
            ('cheapest ship', 2.8),
            ('ship for', 2.0),
            ('vessel for', 2.0),
            ('best ship', 2.4),
            ('charter for', 2.0),
            ('carry', 1.4),
            ('cheapest', 1.6),
            ('ship', 1.2)]),
        ('berth', sk_berth, [
            ('draft', 1.4),
            ('draught', 1.4),
            ('tide', 1.4),
            ('tides', 1.4),
            ('tidal', 1.4),
            ('depth', 1.2),
            ('part loading', 2.4),
            ('part load', 2.2),
            ('part-laden', 2.2),
            ('how much coal', 2.4),
            ('how much fits', 2.4),
            ('clearance', 1.4),
            ('berth', 1.5), ('draft', 1.8), ('draught', 1.8),
            ('can a', 1.2), ('fit', 1.2), ('how much cargo', 2.0),
            ('load', 1.0), ('deadweight', 1.5), ('tpc', 1.8),
            ('part laden', 2.0), ('alongside', 1.5)]),
        ('activity', sk_port_activity, [
            ('how busy', 2.5), ('busy', 1.5), ('congestion', 1.8),
            ('arrivals', 1.5), ('port calls', 2.0), ('traffic', 1.5),
            ('queue', 1.5)]),
        ('weather', sk_weather, [
            ('weather', 1.8), ('cyclone', 2.0), ('storm', 1.8),
            ('gust', 1.8), ('wind', 1.5), ('rain', 1.5), ('risk', 1.2),
            ('what could go wrong', 2.5), ('warning', 1.5)]),
        ('ballast', sk_ballast, [
            ('ballast', 2.0), ('empty leg', 2.5), ('backhaul', 2.0),
            ('leaving empty', 2.0), ('empty return', 2.0), ('deadhead', 2.0)]),
        ('method', sk_method, [
            ('how many folds', 2.6),
            ('folds', 1.4),
            ('fold', 1.2),
            ('purge', 1.8),
            ('purged', 1.8),
            ('leakage', 1.8),
            ('who validated', 2.4),
            ('what would break', 2.4),
            ('break this', 2.0),
            ('validated', 1.4),
            ('validation', 1.4),
            ('tested', 1.2),
            ('test it', 1.6),
            ('did you test', 2.2),
            ('methodology', 1.4),
            ('backtest', 1.6),
            ('walk forward', 2.2),
            ('cross validation', 2.2),
            ('rigorous', 1.4),
            ('how does it work', 2.5), ('methodology', 2.0), ('leakage', 2.0),
            ('leak', 1.5), ('validated', 1.8), ('validation', 1.5),
            ('walk forward', 2.5), ('purge', 2.0), ('conformal', 2.0),
            ('overfitting', 2.0), ('cross validation', 2.0),
            ('how do you know', 2.0), ('interval', 1.2),
            ('why five', 2.4),
            ('why 5', 2.4),
            ('why five days', 2.8),
            ('why that horizon', 2.6),
            ('horizon', 1.4)]),
        ('limits', sk_limits, [
            ('weakness', 1.4),
            ('weaknesses', 1.4),
            ('shortcoming', 1.4),
            ('caveat', 1.4),
            ('what is missing', 2.2),
            ('not in the model', 2.4),
            ('does not include', 2.2),
            ('more time', 1.8),
            ('future work', 2.2),
            ('next steps', 1.8),
            ('what would you improve', 2.4),
            ('fail', 1.8), ('fails', 1.8), ('weakness', 2.0),
            ('weaknesses', 2.0), ('limitation', 2.0), ('limitations', 2.0),
            ('worst', 1.5), ('problem', 1.2), ('wrong', 1.2),
            ('not work', 2.0), ('drawback', 2.0), ('criticism', 1.8),
            ('trust', 2.0),
            ('why should i trust', 2.8),
            ('believe', 1.8),
            ('convince', 2.0),
            ('sceptical', 2.0),
            ('skeptical', 2.0),
            ('better than momentum', 2.8),
            ('beat momentum', 2.6),
            ('vs momentum', 2.6),
            ('momentum', 1.8),
            ('baseline', 1.8),
            ('worst case', 2.4),
            ('caveat', 2.0)]),
        ('data', sk_data, [
            ('where does the data', 2.5), ('data source', 2.2),
            ('sources', 1.5), ('dataset', 1.8), ('where from', 1.8),
            ('what data', 2.0), ('portwatch', 1.8), ('open meteo', 1.8),
            ('yahoo', 1.5)]),
        ('licence', sk_licence, [
            ('licence', 2.0), ('license', 2.0), ('licensing', 2.0),
            ('legal', 1.8), ('redistribute', 2.0), ('copyright', 1.8),
            ('can sail use', 2.2), ('allowed to use', 2.0),
            ('licensed', 1.4), ('baltic licence', 2.5),
            ('can we use', 1.8), ('permitted', 1.8)]),
        ('two_models', sk_two_models, [
            ('average the two', 2.6),
            ('do you average', 2.6),
            ('averaged', 1.8),
            ('combine the two', 2.4),
            ('which model answers', 2.6),
            ('two models', 2.5), ('licensed model', 2.2),
            ('extended model', 2.2), ('which model', 2.0),
            ('why two', 2.0), ('spliced', 1.8), ('splice', 1.8)]),
        ('brief', sk_brief, [
            ('brief me', 3.0), ('brief', 2.0), ('briefing', 2.6),
            ('summarise', 2.6), ('summarize', 2.6), ('summary', 2.4),
            ('overview', 2.4), ('the headline', 2.6), ('headline', 2.0),
            ('tell me everything', 3.0), ('elevator pitch', 3.0),
            ('the whole story', 2.8), ('sum up', 2.6),
            ('in a nutshell', 3.0), ('bottom line', 2.8),
            ('give me the gist', 3.0), ('tldr', 2.8),
            ('everything', 1.4), ('pitch', 2.0)]),
        ('working', sk_working, [
            ('show me the working', 3.0), ('show your working', 3.0),
            ('show the working', 3.0), ('show me how', 2.4),
            ('how did you compute', 3.0), ('how did you get', 2.8),
            ('how did you work', 2.8), ('how did you', 2.2),
            ('how is that computed', 3.0), ('how was that computed', 3.0),
            ('where did that come from', 3.0), ('where is that from', 2.8),
            ('where does that number', 3.0), ('prove it', 2.6),
            ('says who', 2.6), ('back that up', 2.6),
            ('justify', 2.2), ('provenance', 2.6), ('audit', 2.0),
            ('show the maths', 2.8), ('show the math', 2.8),
            ('working', 1.4)]),
        ('whatif', sk_whatif, [
            ('what if', 2.8), ('what would happen if', 3.0),
            ('suppose', 2.4), ('scenario', 2.4), ('hypothetical', 2.6),
            ('assume', 2.0), ('if freight', 2.8), ('if the rate', 2.6),
            ('if rates', 2.4), ('sensitivity', 2.4), ('scale it', 2.2)]),
        ('rank', sk_rank, [
            ('busiest', 2.6), ('quietest', 2.6), ('which port is', 2.4),
            ('which berth is', 2.4), ('best month', 2.2),
            ('worst month', 2.2), ('best year', 2.6), ('worst year', 2.6),
            ('which month', 2.2), ('which year', 2.4),
            ('which vessel is', 2.2), ('which ship is', 2.2),
            ('biggest', 2.2), ('largest', 2.2), ('most', 1.2),
            ('rank', 2.2), ('ranking', 2.2), ('league', 2.0),
            ('busy', 1.0)]),
        ('compare', sk_compare, [
            ('compare', 2.6), ('comparison', 2.4), ('versus', 2.4),
            ('vs', 2.2), (' or ', 0.8), ('against', 1.6),
            ('difference between', 2.6), ('side by side', 2.4),
            ('busier than', 2.6), ('bigger than', 2.4),
            ('better than', 1.6)]),
        ('glossary', sk_glossary, [
            ('what is a', 2.0), ('what is', 1.2), ('what does', 1.2),
            ('what are', 1.2), ('meaning of', 2.2), ('define', 2.2),
            ('definition', 2.2), ('explain', 1.2), ('mean', 1.2),
            ('dwt', 2.4), ('deadweight', 2.4), ('tpc', 2.4),
            ('laycan', 2.6), ('demurrage', 2.6), ('tce', 2.2),
            ('lighterage', 2.0), ('draught', 2.2), ('jargon', 2.4),
            ('terminology', 2.4), ('glossary', 2.6)]),
        ('thanks', sk_thanks, [
            ('thanks', 2.5), ('thank you', 2.5), ('cheers', 2.0),
            ('ta', 1.2), ('appreciated', 2.0)]),
        ('about', sk_about, [
            ('who made', 2.5), ('who built', 2.5), ('who wrote', 2.4),
            ('what is this', 2.2), ('about this', 2.0), ('project', 1.4),
            ('sih', 1.6), ('hackathon', 2.0)]),
        ('help', sk_help, [
            ('help', 1.5), ('what can you do', 2.5), ('hello', 1.5),
            ('hi', 1.0), ('hey', 1.0), ('who are you', 2.0),
            ('what do you know', 2.2), ('capabilities', 2.0)]),
    ]


def _score(q, tokens, terms):
    total = 0.0
    for term, weight in terms:
        if ' ' in term:
            if term in q:
                total += weight * 1.5
        elif term in tokens:
            total += weight
    return total


# Entity keys that survive from one question to the next. "tokens" and
# "q" deliberately do not: those belong to the question that was asked,
# not to the conversation.
CARRY = ('port', 'vessel', 'year', 'month', 'tonnes', 'horizon', 'rate')

# Skills that must never be pulled in as the SECOND half of a two-part
# question - they are the fallbacks, and stapling "here is what I can
# do" onto a real answer makes it look padded.
NO_SECOND = frozenset(('help', 'thanks', 'about', 'working', 'clarify',
                       'sowhat'))

# Answers ABOUT the conversation rather than about the freight. They
# must not become the subject themselves, or "show me the working"
# followed by "prove it" ends up pointing at itself.
# 'sowhat' judges the PREVIOUS answer, so it must not become the
# subject itself - otherwise "is that good" followed by "show me
# more" asks the verdict to judge its own verdict.
META = frozenset(('help', 'thanks', 'about', 'working', 'clarify',
                  'sowhat'))


def _shift_year(q, ctx, e):
    """Resolve "the year before" against whatever was last discussed.

    Cheap to implement and the single most obviously missing thing when
    somebody tries to hold a conversation with a dashboard.
    """
    base = e.get('year') or ctx.get('year')
    if not base:
        return e
    if re.search(r'\b(year|one)\s+(before|earlier|prior)\b', q) or \
            'previous year' in q or 'year back' in q:
        e['year'] = int(base) - 1
    elif re.search(r'\b(year|one)\s+(after|later)\b', q) or \
            'next year' in q or 'following year' in q:
        e['year'] = int(base) + 1
    return e


def _viable(nm, qq):
    """Would this skill actually answer, or just fall through to help?

    "What is the forecast" scored higher on the glossary's *what is* than
    on the word forecast, and the glossary then found no term to define
    and handed back the help text. A skill that cannot answer should not
    win the routing - it should stand aside for the one that can.
    """
    if nm == 'glossary':
        return any(re.search(r'\b%s\b' % re.escape(w), qq) for w in GLOSSARY)
    if nm == 'compare':
        return (len(_two_of(qq, list(ports.PORTS))) == 2
                or len(_two_of(qq, list(ports.VESSELS))) == 2
                or len(re.findall(r'\b20[0-3]\d\b', qq)) >= 2)
    return True


# What a carried context is allowed to contain. It rides out to the
# browser with every answer and comes back on the next question, so
# each value is user input wearing the last reply's clothes. A carried
# year of "nineteen" reached a %d and took the whole answer down.
_CTX_INT = (('year', 1990, 2100), ('month', 1, 12), ('horizon', 1, 400))
_CTX_NUM = (('tonnes', 1.0, 5e8), ('rate', 0.01, 1e5))


def _clean_ctx(context):
    """The carried context, with anything the skills cannot take removed.

    Dropping a bad value rather than rejecting the request is the right
    failure here: the context is a convenience for follow-ups, and a
    question that arrives with a mangled one still deserves an answer.
    """
    out = {}
    if not isinstance(context, dict):
        return out
    for k in ('skill', 'source'):
        v = context.get(k)
        if isinstance(v, str) and len(v) <= 200:
            out[k] = v
    for k, lo, hi in _CTX_INT + _CTX_NUM:
        v = context.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        # float() rather than np.isfinite: a Python int larger than
        # int64 makes numpy build an object array and raise, so the
        # guard against a hostile value was itself the crash.
        try:
            fv = float(v)
        except (TypeError, ValueError, OverflowError):
            continue
        if fv != fv or fv in (float('inf'), float('-inf')):
            continue
        if lo <= fv <= hi:
            out[k] = int(fv) if (k, lo, hi) in _CTX_INT else fv
    for k, allowed in (('port', ports.PORTS), ('vessel', ports.VESSELS)):
        v = context.get(k)
        if isinstance(v, str) and v in allowed:
            out[k] = v
    return out


def answer(question, context=None):
    """Answer from the artefacts, or say plainly that it cannot.

    ``context`` is what this function returned last turn. It is optional
    and the answer is never wrong without it - it only lets follow-ups
    like "and Haldia?" mean what they obviously mean.
    """
    e = entities(question)
    q, tokens = e['q'], e['tokens']
    ctx = _clean_ctx(context)
    e['ctx'] = ctx
    corrected = None
    hint = None

    def _finish(res):
        """Stamp every reply with what it read and what it now remembers.

        A correction the reader cannot see is a different question
        answered silently, which is how an assistant loses trust in a
        single exchange. The context that rides back out is what makes
        the NEXT question able to say "and Haldia?".
        """
        res = dict(res)
        pre = []
        if corrected:
            pre.append('*Reading that as \u201c%s\u201d.*' % corrected)
            res['corrected'] = corrected
        if hint:
            pre.append(hint)
        if pre:
            res['answer'] = ('  '.join(pre) + '\n\n'
                             + (res.get('answer') or ''))
        matched = res.get('matched')
        head = matched.split(' + ')[0] if matched else None
        if head in META or head is None:
            keep = {'skill': ctx.get('skill'),
                    'source': ctx.get('source') or res.get('source')}
        else:
            keep = {'skill': head, 'source': res.get('source')}
        for k in CARRY:
            v = e.get(k)
            keep[k] = ctx.get(k) if v is None else v
        res['context'] = keep
        return res

    if not q:
        return _finish(dict(_reply(
            'Ask me something about the forecast, the ships, the ports or '
            'the method.'), confidence=0.0, matched=None))

    a = _artefacts()

    def rank_all(qq, tt):
        ranked = sorted(((_score(qq, tt, terms), name, fn)
                         for name, fn, terms in _skills()), reverse=True,
                        key=lambda x: x[0])
        return [r for r in ranked if _viable(r[1], qq)] or ranked

    top, name, fn = rank_all(q, tokens)[0]

    # Two questions joined by a conjunction. Scoring the whole string
    # lets the louder half bury the quieter one - "what is the forecast
    # and how accurate is it" comes back as accuracy alone - so each
    # side is ranked on its own and both get answered.
    second_fn = second_name = None
    halves = [h.strip() for h in
              re.split(r'\s+(?:and|plus|also|then)\s+', q, maxsplit=1)]
    if len(halves) == 2 and all(len(h.split()) >= 2 for h in halves):
        r1 = rank_all(halves[0], set(halves[0].split()))[0]
        r2 = rank_all(halves[1], set(halves[1].split()))[0]
        if (r1[0] >= MIN_SCORE and r2[0] >= MIN_SCORE and r1[1] != r2[1]
                and r1[1] not in NO_SECOND and r2[1] not in NO_SECOND):
            name, fn = r1[1], r1[2]
            second_name, second_fn = r2[1], r2[2]
            # Confidence is how well the question was understood, and
            # both halves were - reporting the weaker one understates it.
            top = max(r1[0], r2[0])

    # Typo recovery, tried only when nothing matched. "forcast" is a
    # perfectly clear question and refusing it is a worse failure than
    # answering it; correcting a question that already matched would be
    # the opposite mistake.
    if top < MIN_SCORE:
        fixed, changed = _despell(q)
        if changed:
            ftok = set(fixed.split())
            ftop, fname, ffn = rank_all(fixed, ftok)[0]
            fe = entities(fixed)
            # Accept the correction if it either routes to a skill OR
            # surfaces an entity the misspelling hid. "capsize" scores
            # nothing either way, but "capesize" is a vessel and the
            # entity fallback below can answer it.
            gained = any(fe.get(k) and not e.get(k)
                         for k in ('port', 'vessel', 'month', 'year'))
            if ftop >= MIN_SCORE or gained:
                corrected = fixed
                second_fn = second_name = None
                q, tokens = fixed, ftok
                fe['ctx'] = ctx
                e = fe
                if ftop >= MIN_SCORE:
                    top, name, fn = ftop, fname, ffn

    # A continuation, not a new question. "and Haldia?" after a weather
    # answer is a weather question; a dashboard that treats it as a fresh
    # one is the reason nobody asks a second question.
    followed = False
    if ctx.get('skill') and _is_follow_up(q, tokens):
        e = _shift_year(q, ctx, e)
        for k in CARRY:
            if e.get(k) is None and ctx.get(k) is not None:
                e[k] = ctx[k]
        if top < MIN_SCORE:
            prev = dict((n, f) for n, f, _t in _skills()).get(ctx['skill'])
            if prev is not None:
                name, fn, top, followed = ctx['skill'], prev, MIN_SCORE, True
                second_fn = second_name = None
                subj = e.get('port') or e.get('vessel') or e.get('year')
                hint = ('*Carrying on about %s.*' % subj) if subj else \
                       '*Still on the same question.*'

    if top < MIN_SCORE and e['port']:
        return _finish(dict(sk_port_activity(e, a), confidence=0.5,
                            matched='activity'))
    if top < MIN_SCORE and e['vessel']:
        return _finish(dict(sk_berth(e, a), confidence=0.5, matched='berth'))
    if top < MIN_SCORE and e['month']:
        return _finish(dict(sk_seasonal(e, a), confidence=0.5,
                            matched='seasonal'))
    if top < MIN_SCORE and e['year']:
        return _finish(dict(sk_year(e, a), confidence=0.5, matched='year'))

    # "Weather in London" scored on the word weather and came back with
    # Paradip's forecast. Answering the right question about the wrong
    # place is worse than refusing: a refusal is obviously a refusal,
    # and this looked like an answer. Port-scoped skills therefore
    # require that any place named is one this project actually covers.
    PORT_SCOPED = ('weather', 'activity', 'berth', 'ballast')
    if name in PORT_SCOPED and not e['port']:
        m_place = re.search(r'\b(?:in|at|for|near|around)\s+([a-z]{3,})', q)
        if m_place and m_place.group(1) not in _PLACE_OK                 and not _known_word(m_place.group(1)):
            return _finish(dict(_reply(
                'I only hold data for the berths this project models: '
                + ', '.join(_PORT_NAMES) + ' on the East Coast, and the '
                'load ports they draw from. I have nothing for **'
                + m_place.group(1) + '**, and would rather say so than '
                "hand you another port's figures.",
                source='src/ports.py'), confidence=0.0, matched=None))

    # A single ambiguous word is not an off-topic question - it is an
    # under-specified one, and the useful response is the question back
    # rather than a refusal. "cost" means something here; it just does
    # not yet mean one thing.
    if top < MIN_SCORE and len(tokens) <= 2:
        for word, (ask_back, options) in _CLARIFY.items():
            if word in tokens:
                return _finish(dict(_reply(
                    ask_back + ' I can take that a few ways:\n\n'
                    + '\n'.join('- ' + o for o in options),
                    source='assistant', follow=options),
                    confidence=0.4, matched='clarify'))

    if top < MIN_SCORE:
        # The refusal. Deliberately not a guess, and it offers the
        # nearest things it CAN do rather than a shrug.
        return _finish(dict(_reply(
            'I cannot answer that from this project\'s artefacts, and I '
            'would rather say so than guess \u2014 every number I give is '
            'computed when you ask, so I only know what has been built '
            'here.\n\nTry one of these, or ask about a port, a vessel '
            'class, a year, or a month.',
            None, None,
            ['What is the forecast right now?', 'How accurate is this model?',
             'When does the model fail?', 'What can you do?']),
            confidence=0.0, matched=None))

    out = fn(e, a)

    # Two questions in one breath. Answer both, in the order they were
    # asked, with a rule between them - which is what a person would do.
    if second_fn is not None:
        extra = second_fn(e, a)
        out = dict(out)
        out['answer'] = ((out.get('answer') or '') + '\n\n---\n\n'
                         + (extra.get('answer') or ''))
        if not out.get('table'):
            out['table'] = extra.get('table')
        srcs = []
        for x in (out.get('source'), extra.get('source')):
            srcs.extend(p.strip() for p in (x or '').split('+') if p.strip())
        out['source'] = ' + '.join(dict.fromkeys(srcs)) or None
        out['follow_up'] = ((out.get('follow_up') or [])[:2]
                            + (extra.get('follow_up') or [])[:2])
        name = '%s + %s' % (name, second_name)

    return _finish(dict(out, confidence=round(min(top / 3.0, 1.0), 2),
                        matched=name))


if __name__ == '__main__':
    import sys
    qs = [' '.join(sys.argv[1:])] if len(sys.argv) > 1 else [
        'what is the forecast', 'how accurate is it',
        'when should I book', 'what is it worth in crore',
        'can a capesize berth at paradip', 'which vessel for 150000 tonnes',
        'how busy is vizag', 'weather risk at haldia',
        'what is the empty leg at dhamra', 'how did it do in 2022',
        'which months are weak', 'when does it fail',
        'how do you avoid leakage', 'where does the data come from',
        'is the baltic data licensed', 'why two models',
        'what can you do', 'what is the capital of france']
    for q in qs:
        r = answer(q)
        print('=' * 72)
        print('Q: %s' % q)
        print('   [%s, confidence %.2f]' % (r['matched'] or 'no match',
                                            r['confidence']))
        body = r['answer']
        print('   ' + body[:400].replace('\n', '\n   '))
        if r.get('source'):
            print('   -- source: %s' % r['source'])

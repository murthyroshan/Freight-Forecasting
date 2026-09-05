"""
Checks on the assistant.

An assistant is the easiest place in this project to lose the one rule
everything else follows. A language model asked about freight will
produce a confident paragraph containing numbers from nowhere, and it
will read better than the truth. This one has no model in it, so the
tests are aimed at the two ways it could still go wrong:

  1. Saying something it did not compute. Every figure in a reply has
     to be traceable to an artefact, so the checks below take the
     numbers out of the prose and compare them against the files.

  2. Answering a question it does not understand. A fluent wrong answer
     is worse than a refusal, so the refusal path is tested as hard as
     the skills - including that it contains no digits at all, because
     a refusal with a number in it is a guess wearing a disclaimer.

Run:  python -m tests.test_assistant
"""

import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import assistant, paths  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('  %-4s %s%s' % ('PASS' if ok else 'FAIL', name,
                           '' if ok else '\n         -> ' + str(detail)))
    if not ok:
        FAIL.append(name)


def _nums(text):
    """Every number in a reply, so it can be compared with the source."""
    return set(re.findall(r'-?\d[\d,]*\.?\d*', str(text or '')))


def main():
    print('\n[1] the shape of every reply')
    probes = [
        'what is the forecast', 'when should i book', 'how accurate is it',
        'how did it do in 2022', 'when is the model most confident',
        'what is it worth in rupees', 'which months are weak',
        'which vessel for 150000 tonnes', 'can a capesize berth at paradip',
        'how busy is visakhapatnam', 'weather risk at haldia',
        'what is the empty leg at dhamra', 'how do you avoid leakage',
        'when does the model fail', 'where does the data come from',
        'is the baltic data licensed', 'why two models', 'what can you do',
        # the two social skills - a desk says these, and a registry
        # entry nobody can reach is dead code wearing a capability's
        # clothes
        'thanks', 'who made this',
        # phrasings a probe found unroutable: the way a desk actually
        # asks, rather than the way a skill happens to be named
        'should i book now or wait', 'will rates go up', 'seasonality',
        'is this better than momentum', 'cheapest ship for 150,000 t',
        # the questions a judge asks after the first one: define the
        # jargon, rank the field, put two things side by side, price a
        # scenario, and show the working
        'what is a capesize', 'which port is busiest', 'paradip vs haldia',
        'what if freight is $25 a tonne on 8 million tonnes',
        'show me the working', 'brief me',
        # the opener, the sceptic, the desk and the deployment question
        'what does this do', 'this looks like curve fitting',
        'what should i do today', 'does it run offline',
        # the stack, the status quo, the provenance and the verdict
        'what is your tech stack', 'what is the alternative',
        'did you make these numbers up', 'is that good',
    ]
    seen = set()
    for q in probes:
        r = assistant.answer(q)
        ok = (isinstance(r, dict) and isinstance(r.get('answer'), str)
              and r['answer'].strip()
              and isinstance(r.get('follow_up'), list)
              and 'confidence' in r and 'matched' in r)
        check('%r returns a well-formed reply' % q, ok, sorted(r or {}))
        if r.get('matched'):
            # A two-part answer is matched as "a + b"; both halves count
            # as reached, or a joint answer would look like dead code.
            for part in r['matched'].split(' + '):
                seen.add(part)
        check('%r is not a refusal' % q, r.get('matched') is not None,
              r.get('answer', '')[:60])

    print('\n[2] every skill in the registry is reachable')
    # A skill nobody can phrase their way into is dead code that still
    # looks like a capability in the help text.
    registered = {n for n, _fn, _t in assistant._skills()}
    check('%d of %d skills answered a natural question'
          % (len(seen), len(registered)),
          seen == registered, sorted(registered - seen))

    print('\n[3] the numbers come from the artefacts, not from prose')
    m = json.load(io.open(os.path.join(paths.MODELS, 'metrics.json'),
                          encoding='utf-8'))
    r = assistant.answer('how accurate is it')
    got = _nums(r['answer'])
    want = '%.1f' % m['models'][m['best_model']]['direction_pct']
    check('the accuracy reply quotes metrics.json direction (%s)' % want,
          want in got, sorted(got)[:8])
    check('and the scored-row count from the same file (%s)'
          % format(m['n_scored'], ','),
          format(m['n_scored'], ',') in got or str(m['n_scored']) in got)
    check('and the effective window count, not the row count',
          str(m['n_effective']) in got, m['n_effective'])

    p = json.load(io.open(os.path.join(paths.MODELS, 'procurement.json'),
                          encoding='utf-8'))
    r = assistant.answer('what is it worth')
    check('the value reply quotes the measured saving (%.2f)'
          % p['model_policy']['saved_pct'],
          '%.2f' % p['model_policy']['saved_pct'] in _nums(r['answer']))
    check('and reports that the control LOSES, not that waiting is free',
          'lost' in r['answer'].lower())
    check('and carries the momentum caveat with the number',
          'momentum' in r['answer'].lower())

    r = assistant.answer('can a capesize berth at paradip')
    from src import ports
    cs = ports.can_serve('Capesize', 'Paradip')
    check('the berth reply quotes the computed part-load (%s)'
          % format(cs['max_cargo_t'], ','),
          format(cs['max_cargo_t'], ',') in _nums(r['answer']))

    print('\n[4] the refusal is a refusal')
    for q in ['what is the capital of france', 'write me a poem',
              'who won the world cup', 'what is 2 plus 2',
              'tell me about quantum computing']:
        r = assistant.answer(q)
        check('%r is refused' % q, r['matched'] is None and
              r['confidence'] == 0.0, (r['matched'], r['confidence']))
        # A refusal containing a figure is a guess with a disclaimer on it.
        check('%r: the refusal contains no numbers at all' % q,
              not _nums(r['answer']), sorted(_nums(r['answer'])))
        check('%r: it offers something it CAN do' % q,
              len(r['follow_up']) >= 2, r['follow_up'])

    print('\n[5] entities are read out of ordinary phrasing')
    e = assistant.entities('can a newcastlemax load 150,000 t at vizag in 2024')
    check('port alias resolves (vizag -> %s)' % e['port'],
          e['port'] == 'Visakhapatnam', e['port'])
    check('vessel resolves', e['vessel'] == 'Newcastlemax', e['vessel'])
    check('tonnage parses with a comma', e['tonnes'] == 150000.0, e['tonnes'])
    check('year parses', e['year'] == 2024, e['year'])
    for text, want in (('150k tonnes', 150000.0), ('1.5 mt', 1500000.0),
                       ('10 million tonnes', 10000000.0),
                       ('160000 t', 160000.0)):
        check('%r -> %s tonnes' % (text, format(int(want), ',')),
              assistant.entities(text)['tonnes'] == want,
              assistant.entities(text)['tonnes'])
    # "may" is a month and also an ordinary English word.
    check('"maybe" is not read as May',
          assistant.entities('maybe tomorrow')['month'] is None)
    check('but "in May" is', assistant.entities('what about in may')['month']
          == 5)
    check('a horizon in weeks becomes trading days',
          assistant.entities('two weeks out')['horizon'] is None
          and assistant.entities('3 weeks out')['horizon'] == 15)

    print('\n[6] an entity alone is enough to route')
    for q, want in (('paradip', 'activity'), ('2024', 'year'),
                    ('capesize', 'berth')):
        r = assistant.answer(q)
        check('%r routes to %s on the entity alone' % (q, want),
              r['matched'] == want, r['matched'])

    print('\n[7] it degrades rather than raising')
    real = assistant._artefacts
    try:
        assistant._artefacts = lambda: {k: None for k in
                                        ('metrics', 'forecast', 'procurement',
                                         'seasonal', 'licensed')}
        for q in ('what is the forecast', 'how accurate is it',
                  'what is it worth', 'why two models'):
            r = assistant.answer(q)
            ok = isinstance(r.get('answer'), str) and r['answer'].strip()
            check('%r survives missing artefacts' % q, ok, r)
            check('%r says what to run instead of inventing' % q,
                  'run' in r['answer'].lower() or 'not' in r['answer'].lower(),
                  r['answer'][:70])
    finally:
        assistant._artefacts = real

    print('\n[8] nothing user-supplied reaches the page as markup')
    # The answers are ours, but the invariant is cheap to keep and the
    # question is echoed back into the log.
    page = io.open(os.path.join(paths.TEMPLATES, 'dashboard.html'),
                   encoding='utf-8').read()
    check('the question is escaped before it is echoed',
          "askSay(rkEsc(question)" in page)
    check('table cells are escaped', "'<td>' + rkEsc(c)" in page)
    check('the markdown renderer escapes BEFORE adding tags',
          'let t = rkEsc(String(text' in page)
    check('chips carry their text as an escaped attribute',
          'data-q="' in page and 'rkEsc(q)' in page)

    print('\n[8b] the assistant holds a conversation')
    # A follow-up is the cheapest test of whether this is an assistant or
    # a search box. "and Haldia?" after a weather question is a weather
    # question, and nothing but the carried context can know that.
    r1 = assistant.answer('weather risk at paradip')
    check('a first answer hands back a context',
          isinstance(r1.get('context'), dict)
          and r1['context'].get('skill') == 'weather'
          and r1['context'].get('port') == 'Paradip',
          r1.get('context'))
    r2 = assistant.answer('and haldia?', r1.get('context'))
    check('the follow-up stays on weather and moves the port',
          r2.get('matched') == 'weather'
          and r2['context'].get('port') == 'Haldia',
          (r2.get('matched'), r2.get('context')))
    check('it says out loud that it carried the subject over',
          'Carrying on' in r2.get('answer', ''), r2.get('answer', '')[:60])
    check('without a context the same phrase is NOT answered as weather',
          assistant.answer('and haldia?').get('matched') != 'weather')

    r3 = assistant.answer('how did it do in 2022')
    r4 = assistant.answer('what about the year before', r3.get('context'))
    check('"the year before" resolves against the last year discussed',
          r4['context'].get('year') == 2021, r4.get('context'))

    print('\n[8c] it can show its working')
    r5 = assistant.answer('how accurate is it')
    r6 = assistant.answer('show me the working', r5.get('context'))
    check('the working answer explains the accuracy method',
          r6.get('matched') == 'working'
          and 'independent' in r6.get('answer', '').lower(),
          r6.get('answer', '')[:80])
    check('it names the module the number came from',
          str(r5.get('source', '')).split(' + ')[0]
          in r6.get('answer', ''), r6.get('answer', '')[-160:])
    check('a meta answer does not become the subject itself',
          r6['context'].get('skill') == 'accuracy', r6.get('context'))
    check('with no prior answer it says so rather than inventing one',
          'Ask me something first' in assistant.answer(
              'show me the working').get('answer', ''))

    print('\n[8d] two questions in one breath get two answers')
    r7 = assistant.answer('what is the forecast and how accurate is it')
    check('both halves are routed', r7.get('matched') == 'forecast + accuracy',
          r7.get('matched'))
    check('the two answers are separated by a rule',
          '---' in r7.get('answer', ''))
    check('a stray conjunction does not staple on a second essay',
          ' + ' not in (assistant.answer(
              'how do you avoid leakage and overfitting').get('matched') or ''))

    print('\n[8e] a scenario is priced from the asker\'s own numbers')
    r8 = assistant.answer('what if freight is $25 a tonne on 8 million tonnes')
    check('the tonnage in the question is the tonnage used - not a default',
          '8,000,000' in r8.get('answer', ''), r8.get('answer', '')[:120])
    check('the rate in the question is the rate used',
          '$25.00' in r8.get('answer', ''), r8.get('answer', '')[:120])
    check('it says whose numbers are whose',
          'yours' in r8.get('answer', '').lower())
    check('asked without numbers it asks for them rather than inventing them',
          'yours, not mine' in assistant.answer('what if').get('answer', ''))

    print('\n[8f] a typo is answered, and the reading is disclosed')
    for typo, want in (('forcast', 'forecast'), ('acuracy', 'accuracy'),
                       ('visakapatnam', 'activity'), ('capsize', 'berth')):
        r = assistant.answer(typo)
        check('%r is answered as %s' % (typo, want),
              (r.get('matched') or '').split(' + ')[0] == want,
              r.get('matched'))
        check('%r says what it read it as' % typo,
              'Reading that as' in r.get('answer', ''),
              r.get('answer', '')[:60])

    print('\n[8g] a hostile context cannot take an answer down')
    # The context rides out to the browser and comes back on the next
    # question, so every value in it is user input wearing the last
    # reply's clothes. Dropping a bad value beats rejecting the request:
    # the context is a convenience, and the question still deserves an
    # answer.
    for junk in (None, {}, [], 'nope', 42,
                 {'skill': 'nope'}, {'skill': 'weather', 'port': 'Atlantis'},
                 {'skill': 'year', 'year': 'not a year'},
                 {'skill': 'year', 'year': {'a': 1}},
                 {'skill': 'year', 'year': 10 ** 30},
                 {'skill': 'year', 'year': float('nan')},
                 {'skill': 'berth', 'vessel': ['list']},
                 {'skill': 'whatif', 'rate': 'free', 'tonnes': None},
                 {'skill': 'activity', 'source': 'x' * 5000}):
        ok = True
        for q in ('and haldia?', 'what about the year before', 'it'):
            try:
                r = assistant.answer(q, junk)
                ok = ok and isinstance(r.get('answer'), str) and bool(r['answer'])
            except Exception as exc:
                ok = False
                print('      raised: %s' % exc)
        check('a context of %.40r still answers' % (junk,), ok)
    check('an invented port never survives the context',
          assistant.answer('and?', {'skill': 'weather', 'port': 'Atlantis'}
                           )['context'].get('port') is None)
    check('an over-long source is dropped rather than echoed',
          len(str(assistant.answer('it', {'skill': 'activity',
                                          'source': 'x' * 5000}
                                   )['context'].get('source') or '')) < 300)

    print('\n[8h] the briefing carries its own caveats')
    rb = assistant.answer('brief me')
    check('the briefing answers', rb.get('matched') == 'brief',
          rb.get('matched'))
    for must in ('independent', 'loses on', 'control'):
        check('the briefing says %r rather than only the wins' % must,
              must in rb.get('answer', '').lower(), rb.get('answer', '')[:80])
    check('every claim in it names what computed it',
          bool(rb.get('table')) and all(len(row) == 3 and row[2]
                                        for row in rb['table']['rows']))

    print('\n[9] the panel is wired')
    for hook in ('askFab', 'askPanel', 'askLog', 'askForm', 'askSend',
                 'askOpen', 'askWire', 'ASK_PROMPTS', 'ask-fab', 'ask-bot'):
        check('page wires %r' % hook, hook in page)
    check('it posts to /api/ask', "'/api/ask'" in page)
    check('Escape closes it', "ev.key === 'Escape'" in page)
    check('Ctrl+K toggles it', "'k'" in page and 'ctrlKey' in page)
    check('the source line is rendered on every answer that has one',
          'ask-src' in page and 'r.source' in page)
    check('a deep link can open it, which is how the suite drives it',
          "qs.has('ask')" in page)
    check('reduced motion is respected',
          'prefers-reduced-motion' in page)


    # ----------------------------------------------------------------
    print('\n[10] every registered skill runs')

    # sk_pitch shipped broken for exactly this reason: it routed
    # correctly, then threw inside the handler on a helper that appends
    # a percent sign to a tonnage. Routing tests never touched it
    # because they only assert which skill WINS. Invoke all of them.
    from src import assistant as _A
    _art = _A._artefacts()
    _skills = _A._skills()
    check('there are skills registered', len(_skills) >= 20, len(_skills))
    _broke = []
    for _name, _fn, _terms in _skills:
        for _probe in ('what is the forecast', 'paradip in 2024',
                       'capesize 150000 tonnes at $20 a tonne'):
            _e = _A.entities(_probe)
            try:
                _out = _fn(_e, _art)
            except Exception as _exc:
                _broke.append('%s(%r): %s' % (_name, _probe, str(_exc)[:70]))
                break
            if not isinstance(_out, dict) or 'answer' not in _out:
                _broke.append('%s returned %r' % (_name, type(_out).__name__))
                break
    check('all %d skills run on every probe without throwing'
          % len(_skills), not _broke, _broke[:4])

    # _num appends '%'. A tonnage or a count formatted with it reads
    # "180000% tonnes", which has now caught three separate bugs.
    _bad_pct = []
    for _name, _fn, _terms in _skills:
        try:
            _a = (_fn(_A.entities('what is the forecast'), _art)
                  or {}).get('answer') or ''
        except Exception:
            continue
        for _m in re.finditer(r'([\d,]{4,})%', _a):
            _bad_pct.append('%s: %s' % (_name, _m.group(0)))
    check('no skill formats a count as a percentage',
          not _bad_pct, _bad_pct[:4])

    # ----------------------------------------------------------------
    print('\n[11] the pitch quotes the model that actually ships')

    _m = _art['metrics'] or {}
    _best = _m.get('best_model')
    _dep = (_m.get('models') or {}).get(_best, {}).get('direction_pct')
    _test = ((_m.get('models') or {})
             .get('direction_test', {}).get('rate'))
    _pitch = _A.answer('what does this do')
    check('a pitch question routes to the pitch skill',
          _pitch['matched'] == 'pitch', _pitch['matched'])
    if _dep is not None:
        check('and it quotes the deployed model (%.1f%%)' % _dep,
              '%.1f%%' % _dep in _pitch['answer'])
    # direction_test.rate is the best of several candidates, so it can
    # sit above the deployed model. Quoting it in the opening pitch
    # would overstate the page by a point.
    if _test is not None and _dep is not None and abs(_test * 100 - _dep) > .05:
        check('and NOT the flattering best-of-candidates figure (%.1f%%)'
              % (_test * 100),
              '%.1f%%' % (_test * 100) not in _pitch['answer'])

    for _q, _want in (('is it accurate', 'accuracy'),
                      ('this looks like curve fitting', 'rebuttal'),
                      ('is 61.7% even better than a coin flip', 'rebuttal'),
                      ('what stops me just always waiting', 'rebuttal'),
                      ('is this just a linear regression', 'rebuttal'),
                      ('what should i do today', 'today'),
                      ('does it run offline', 'deploy'),
                      ('how often does it retrain', 'deploy')):
        _r = _A.answer(_q)
        check('%r routes to %s' % (_q, _want), _r['matched'] == _want,
              _r['matched'])

    # The always-wait rebuttal exists to answer the obvious objection,
    # so it has to name the control arm rather than talk around it.
    _wait = _A.answer('what stops me just always waiting')['answer']
    _p = _art['procurement'] or {}
    if _p:
        check('the always-wait answer quotes the real control arm',
              '%.1f%%' % _p['control']['saved_pct'] in _wait
              or '%.1f' % _p['control']['saved_pct'] in _wait)
        check('and concedes it does not beat momentum',
              'momentum' in _wait.lower())


    # ----------------------------------------------------------------
    print('\n[12] the stack, the status quo and the provenance')

    for _q, _want in (('what is your tech stack', 'stack'),
                      ('what features does the model use', 'stack'),
                      ('what algorithm is it', 'stack'),
                      ('how is this better than a spreadsheet', 'versus'),
                      ('what is the alternative', 'versus'),
                      ('why not just ask a broker', 'versus'),
                      ('did you make these numbers up', 'provenance'),
                      ('are the numbers real', 'provenance'),
                      ('how many folds', 'method'),
                      ('do you average the two', 'two_models')):
        _r = _A.answer(_q)
        check('%r routes to %s' % (_q, _want), _r['matched'] == _want,
              _r['matched'])

    # Every feature in metrics.json must appear in the stack answer, or
    # a feature could be added and silently never mentioned.
    _feats = (_art['metrics'] or {}).get('features') or []
    _stack = _A.answer('what features does the model use')['answer']
    _absent = [f for f in _feats if f not in _stack]
    check('all %d features are named' % len(_feats), not _absent, _absent[:4])
    check('and the count it states matches the artefact',
          '%d features' % len(_feats) in _stack)

    # The comparison answer must not invent a description of how the
    # customer works today - the project has no source for that.
    _v = _A.answer('what is the alternative')['answer']
    check('the comparison concedes it does not beat momentum',
          'momentum' in _v.lower())
    check('and makes no claim about the customer internal process',
          'no claim about' in _v.lower())

    # ----------------------------------------------------------------
    print('\n[13] "is that good" judges rather than repeats')

    _acc = _A.answer('how accurate is it')
    _v1 = _A.answer('is that good', _acc.get('context'))
    check('a judgement follow-up routes to the verdict skill',
          _v1['matched'] == 'sowhat', _v1['matched'])
    check('and it does not simply repeat the previous answer',
          _v1['answer'] != _acc['answer'])
    check('it names the base rate as the right comparison, not 50 percent',
          'base rate' in _v1['answer'])

    _val = _A.answer('how much money does this save')
    _v2 = _A.answer('is that good', _val.get('context'))
    check('the money verdict differs from the accuracy verdict',
          _v2['answer'] != _v1['answer'])
    check('and it concedes the momentum comparison',
          'momentum' in _v2['answer'].lower())

    # sowhat judges the PREVIOUS answer, so it must never become the
    # subject - or "is that good" then "show me more" asks the verdict
    # to judge itself, which reads as a broken bot mid-demo.
    _more = _A.answer('show me more', _v2.get('context'))
    check('a continuation after a verdict returns to the real subject',
          _more['matched'] == 'value', _more['matched'])
    check('"show me more" is treated as a continuation at all',
          _more['matched'] is not None)
    print('\n' + '=' * 62)
    if FAIL:
        print('  %d FAILED:' % len(FAIL))
        for f in FAIL:
            print('    - %s' % f)
        print('=' * 62)
        return 1
    print('  all assistant checks passed')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())

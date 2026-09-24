from pathlib import Path

import pytest

from callreplay import contract as contracts
from callreplay import formats
from callreplay.evaluate import evaluate, score_of, status_of
from callreplay.model import Conversation, ToolCall, Turn

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'dental'


@pytest.fixture(scope='module')
def dental():
    return contracts.load(EXAMPLE)


def conv(*turns, **kw):
    return Conversation(id=kw.pop('id', 't'), turns=list(turns), **kw)


def user(text):
    return Turn('user', text)


def agent(text='', *calls):
    return Turn('agent', text, list(calls))


def tool(call_id, name, result):
    return Turn('tool', tool_call_id=call_id, name=name, result=result)


def codes(result):
    return [f.code for f in result.findings]


def test_the_example_recordings_score_as_documented(dental):
    results = {c.id: evaluate(c, dental, context=dental.prompt) for c in formats.load(EXAMPLE)}
    by_status = {}
    for r in results.values():
        by_status[r.status] = by_status.get(r.status, 0) + 1
    assert by_status == {'pass': 27, 'warn': 5, 'fail': 4, 'review': 1, 'unscorable': 5}
    assert codes(results['call-006']) == ['ungrounded_value']          # "how about 4:30 pm?" on a full day
    assert codes(results['call-027']) == ['ungrounded_value']          # a price from memory
    assert codes(results['call-002']) == ['missing_end_call']          # "cheers" isn't a goodbye to v1
    assert codes(results['call-019']) == ['missing_preferred_tool']
    assert results['call-037'].status == 'review'                      # wrong number: no intent


@pytest.mark.parametrize('says, reason', [
    ([], 'the caller never spoke'),
    (['[inaudible]'], 'only noise from the caller'),
    (['...'], 'only noise from the caller'),
    (['Hello?'], 'too little speech to judge'),
])
def test_unscorable_is_kept_apart(dental, says, reason):
    r = evaluate(conv(agent('Hi, Harbor Dental.'), *[user(s) for s in says]), dental)
    assert (r.status, r.score, r.reason) == ('unscorable', None, reason)


def test_marked_unscorable(dental):
    r = evaluate(conv(user('Testing one two three'), metadata={'unscorable': 'internal test call'}), dental)
    assert r.status == 'unscorable' and r.reason == 'internal test call'


def test_intents_resolve_in_file_order(dental):
    # "appointment" is a book keyword too, but cancel comes first in the contract
    r = evaluate(conv(user('I need to cancel my appointment please')), dental)
    assert r.intent == 'cancel' and r.intent_source == 'keywords'
    assert evaluate(conv(user('Hi', ), user('when is my next appointment?')), dental).intent == 'lookup'
    assert evaluate(conv(user('I want to cancel it'), intent='book'), dental).intent == 'book'   # a label wins


def test_contract_checks(dental):
    c = conv(user('Can I book a cleaning on Friday?'),
             agent('', ToolCall('create_booking', {'name': 'Ann Lee', 'date': '2026-09-18', 'time': '10:00'}, 'a')),
             tool('a', 'create_booking', {'ok': True}),
             agent('Booked.'),
             agent('', ToolCall('check_availability', {'date': '2026-09-18', 'service': 'cleaning'}, 'b')),
             tool('b', 'check_availability', {'slots': ['10:00']}))
    r = evaluate(c, dental)
    assert r.intent == 'book' and r.status == 'fail'
    assert {'wrong_order', 'missing_args'} <= set(codes(r))
    missing = next(f for f in r.findings if f.code == 'missing_args')
    assert 'phone' in missing.message and 'service' in missing.message and missing.turn == 1


def test_writes_outside_their_intent(dental):
    c = conv(user('How much is a cleaning?'),
             agent('', ToolCall('cancel_booking', {'booking_id': 'X'}, 'a')), tool('a', 'cancel_booking', {'ok': True}),
             agent('A cleaning is 95 dollars.'))
    r = evaluate(c, dental)
    assert r.intent == 'info' and 'unexpected_write' in codes(r)


def test_grounding(dental):
    c = conv(user('Do you have anything on Friday for a cleaning?'),
             agent('', ToolCall('check_availability', {'date': '2026-09-18', 'service': 'cleaning'}, 'a')),
             tool('a', 'check_availability', {'slots': ['14:15', '17:30']}),
             agent('I have 2:15 pm, 5:30 or 4 pm.'),
             user('Is 9 am possible?'),
             agent("9 am isn't free, sorry. We open at 9 am. It's $95."))
    r = evaluate(c, dental, context=dental.prompt)
    bad = [f for f in r.findings if f.code == 'ungrounded_value']
    assert [f.message.split('"')[1] for f in bad] == ['4 pm', '$95']   # 5:30 is 17:30; 9 am the caller said
    assert all(f.turn in (3, 5) for f in bad)


def test_repeated_calls_and_silence(dental):
    call = lambda i: ToolCall('find_booking', {'phone': '1'}, f'c{i}')
    c = conv(user('when is my appointment'), agent('', call(1)), tool('c1', 'find_booking', {}),
             agent('', call(2)), tool('c2', 'find_booking', {}), agent('', call(3)), tool('c3', 'find_booking', {}),
             agent(''))
    r = evaluate(c, dental)
    assert 'repeated_call' in codes(r) and 'silent_turn' in codes(r)


def test_status_and_score_agree():
    from callreplay.checks import Finding
    fs = [Finding('warning', 'x', '')]
    assert status_of(fs) == 'warn' and score_of(fs, 'warn', {'warning': 10}) == 84
    fs = [Finding('error', 'x', '')]
    assert status_of(fs) == 'fail' and score_of(fs, 'fail', {'error': 25}) == 59
    assert status_of([]) == 'pass' and score_of([], 'pass', {}) == 100


def test_no_contract_runs_generic_checks():
    r = evaluate(conv(user('hello there'), agent('Hi!'), user('are you there?'), agent('')), contracts.Contract())
    assert r.status == 'warn' and codes(r) == ['silent_turn'] and r.intent is None
    r = evaluate(conv(user('hello there'), agent('')), contracts.Contract())
    assert r.status == 'fail' and codes(r) == ['no_agent_response', 'silent_turn']


def test_contract_errors(tmp_path):
    bad = tmp_path / 'c.toml'
    bad.write_text('[intents.book]\nrequire = ["x"]\n')
    with pytest.raises(contracts.ContractError, match='unknown key'):
        contracts.load(bad)
    bad.write_text('[grounding]\nkinds = ["date"]\n')
    with pytest.raises(contracts.ContractError, match='unknown kind'):
        contracts.load(bad)
    bad.write_text('[intents.book]\norder = [["a", "b", "c"]]\n')
    with pytest.raises(contracts.ContractError, match='first, then'):
        contracts.load(bad)

import pytest

from callreplay.grounding import known_values, mentions


@pytest.mark.parametrize('text, expected', [
    ('I have 10:30 or 2:15 pm', [{'10:30', '22:30'}, {'14:15'}]),
    ('how about 5 in the afternoon?', [{'17:00'}]),
    ('we open at 9am and close at 6 p.m.', [{'09:00'}, {'18:00'}]),
    ('noon works', [{'12:00'}]),
    ('17:45 is free', [{'17:45'}]),
    ('the slot 2026-09-17T14:30:00', [{'14:30'}]),
    ('call 555-0100 or room 12', []),
])
def test_times(text, expected):
    assert [set(m.readings) for m in mentions(text, ['time'])] == expected


@pytest.mark.parametrize('text, expected', [
    ('$45.50', ['45.50']), ('$1,200', ['1200.00']), ('80 euros', ['80.00']), ('€30', ['30.00']),
    ('it costs 12 dollars', ['12.00']), ('room 12', []),
])
def test_money(text, expected):
    assert [next(iter(m.readings)) for m in mentions(text, ['money'])] == expected


def test_price_is_not_read_as_a_time():
    assert [m.kind for m in mentions('That is $10.30 at 10:30')] == ['money', 'time']


def test_known_values_in_tool_results():
    known = known_values({'slots': ['09:30', '14:15'], 'price': 120})
    assert {'09:30', '14:15', '120.00'} <= known
    assert '120.00' not in known_values('room 120')        # a bare number in text isn't a price

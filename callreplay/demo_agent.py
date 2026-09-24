"""A scripted receptionist for the demo: no model, no network, same answers every time.

It exists so `callreplay demo` works without an API key. It reads the conversation the same way
a model would (only the messages it is sent), calls the same tools and phrases its answers like
a phone agent, for Harbor Dental, the fictional clinic in examples/dental.

Two versions, to show what a replay is for:

    v1  the agent that took the recorded calls. Its known problems:
        - when a day is fully booked it offers a time it never checked
        - it quotes the whitening price from memory, and gets it wrong
        - it doesn't take "cheers" or "that's it" as a goodbye, so the call never ends
        - asked to move an appointment to a given time, it moves it without checking that time
    v2  a "new prompt" that fixes those, and breaks two things that used to work:
        - when the caller asks for a specific time, it books straight away, without checking
        - "cancel my appointment tomorrow" cancels by date, without looking the booking up
        One call also fails on purpose (a 503 from the "provider"), to show that errors are
        reported apart from the agent's own failures.

Real runs point --agent at a real model; see docs/replay.md.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from .agents import AgentError, AgentReply
from .grounding import mentions
from .model import ToolCall

SERVICES = {'checkup': ('check-up', 'checkup'), 'cleaning': ('cleaning', 'clean'), 'whitening': ('whitening',),
            'filling': ('filling',)}
DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
FAILING_CALLER = '+1 555 0147'          # v2: this caller's call hits a provider error


@dataclass
class State:
    today: date
    caller: str
    users: list[str]
    intent: str = ''
    service: str = ''
    day: date | None = None
    part: str = ''                      # 'morning' | 'afternoon'
    wanted_time: str = ''               # a time the caller asked for, HH:MM
    name: str = ''
    offered: list[str] = field(default_factory=list)
    checked: dict[str, list[str]] = field(default_factory=dict)     # date -> slots
    booking: dict[str, Any] | None = None     # found with find_booking
    found_nothing: bool = False
    done: str = ''                      # 'booked' | 'cancelled' | 'moved' | 'answered' | 'transferred'


class DemoAgent:
    def __init__(self, version: str = 'v2', delay: float = 0.0):
        if version not in ('v1', 'v2'):
            raise ValueError('demo agent: version is v1 or v2')
        self.version, self.delay = version, delay
        self.name = f'demo:{version}'

    # ------------------------------------------------------------------ entry
    def reply(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AgentReply:
        if self.delay:
            time.sleep(self.delay)
        st = self._state(messages)
        if self.version == 'v2' and st.caller == FAILING_CALLER:
            raise AgentError('HTTP 503: {"error": "model overloaded, try again later"}')
        last = messages[-1]
        if last['role'] == 'tool':
            return self._after_tools(st, messages)
        return self._after_user(st, last.get('content') or '')

    # ------------------------------------------------------------------ reading the conversation
    def _state(self, messages: list[dict[str, Any]]) -> State:
        system = next((m['content'] for m in messages if m['role'] == 'system'), '')
        now = re.search(r'Current time: (\S+)', system)
        caller = re.search(r'Caller: (.+)', system)
        today = datetime.fromisoformat(now.group(1)).date() if now else date.today()
        st = State(today=today, caller=caller.group(1).strip() if caller else '',
                   users=[m.get('content') or '' for m in messages if m['role'] == 'user'])
        calls: dict[str, tuple[str, dict[str, Any]]] = {}
        for m in messages:
            if m['role'] == 'user':
                self._read_user(st, m.get('content') or '')
            elif m['role'] == 'assistant':
                for c in m.get('tool_calls') or []:
                    calls[c['id']] = (c['function']['name'], json.loads(c['function'].get('arguments') or '{}'))
            elif m['role'] == 'tool':
                name, args = calls.get(m['tool_call_id'], ('', {}))
                self._read_result(st, name, args, _json(m.get('content')))
        return st

    def _read_user(self, st: State, text: str) -> None:
        t = text.lower()
        if not st.intent:
            st.intent = _intent(t)
        for svc, words in SERVICES.items():
            if any(w in t for w in words):
                st.service = svc
        d = _day(t, st.today)
        target = re.search(rf"\bto ((?:next )?(?:{'|'.join(DAYS)})|tomorrow|today)", t)
        if target:                                  # "move it from Wednesday to Thursday"
            d = _day(target.group(1), st.today)
        if d:
            st.day = d
        if 'morning' in t:
            st.part = 'morning'
        elif 'afternoon' in t or 'evening' in t:
            st.part = 'afternoon'
        times = mentions(text, ['time'])
        if times:                                   # "5:30" on a phone is 17:30 at a clinic
            readings = sorted(times[0].readings)
            st.wanted_time = next((r for r in readings if r in st.offered), None) or \
                next((r for r in readings if '08:00' <= r < '20:00'), readings[0])
        name = _name(text)
        if name:
            st.name = name

    def _read_result(self, st: State, name: str, args: dict[str, Any], result: Any) -> None:
        r = result if isinstance(result, dict) else {}
        if name == 'check_availability':
            slots = [s for s in r.get('slots', []) if _part_ok(s, st.part)] or r.get('slots', [])
            st.checked[args.get('date', '')] = list(r.get('slots', []))
            st.offered = slots[:2]
            if args.get('date'):
                st.day = date.fromisoformat(args['date'])
        elif name == 'find_booking':
            found = r.get('bookings') or []
            st.booking, st.found_nothing = (found[0] if found else None), not found
        elif name == 'create_booking' and r.get('ok', True):
            st.done = 'booked'
        elif name == 'cancel_booking' and r.get('ok', True):
            st.done = 'cancelled'
        elif name == 'reschedule_booking' and r.get('ok', True):
            st.done = 'moved'
        elif name == 'transfer_call':
            st.done = 'transferred'

    # ------------------------------------------------------------------ after the caller speaks
    def _after_user(self, st: State, text: str) -> AgentReply:
        t = text.lower()
        v1 = self.version == 'v1'
        goodbye = r"\b(bye|goodbye|thanks|thank you|that's all)\b" if v1 else \
            r"\b(bye|goodbye|thanks|thank you|that's all|that's it|cheers|nothing else)\b"
        busy = st.intent in ('book', 'cancel', 'reschedule') and not st.done and st.day is not None
        if re.search(r'\b(bye|goodbye)\b', t) or (re.search(goodbye, t) and not busy and not mentions(text, ['time'])):
            return _say('Thanks for calling Harbor Dental. Have a good day!', call('end_call', reason='caller said goodbye'))
        if st.done and st.done != 'answered' and re.search(r'\b(no|nope)\b', t):
            return _say('Anything else I can help you with today?')

        if st.intent == 'transfer':
            return _say("I'll put you through to the team right away.", call('transfer_call', reason=_reason(t)))
        if st.intent == 'other':
            return _say("This is Harbor Dental, the dental clinic. Is there anything I can help you with?")
        if st.intent == 'info':
            return self._info(st, t)
        if st.intent == 'lookup':
            if st.booking is None and not st.found_nothing:
                return _say('', call('find_booking', phone=st.caller))
            return _say('Anything else I can help you with?')
        if st.intent == 'cancel':
            return self._cancel(st, t)
        if st.intent == 'reschedule':
            return self._reschedule(st, t)
        return self._book(st, t)

    def _book(self, st: State, t: str) -> AgentReply:
        if st.done == 'booked':
            return _say(f"You're all set for {_when(st.day, st.wanted_time, st.today)}. Anything else?")
        if not st.service:
            return _say('Sure. What would you like to come in for: a check-up, a cleaning, whitening or a filling?')
        if not st.day:
            return _say(f'Of course. What day would suit you for the {_svc(st.service)}?')
        iso = st.day.isoformat()
        direct = self.version == 'v2' and st.wanted_time and st.name and len(st.users) == 1
        if direct:                                   # v2 regression: books what was asked, unchecked
            return _say('', call('create_booking', name=st.name, phone=st.caller, date=iso,
                                 time=st.wanted_time, service=st.service))
        if iso not in st.checked:
            return _say('', call('check_availability', date=iso, service=st.service))
        if st.wanted_time and st.wanted_time not in st.checked[iso]:
            if st.offered:
                return _say(f"Sorry, {_clock(st.wanted_time)} isn't free. I have {_offer(st.offered)} {_on(st.day, st.today)}.")
            return _say(f"Sorry, {_clock(st.wanted_time)} isn't free.")
        if st.wanted_time:
            if not st.name:
                return _say('Great. Can I have your name, please?')
            return _say('', call('create_booking', name=st.name, phone=st.caller, date=iso,
                                 time=st.wanted_time, service=st.service))
        if st.offered:
            return _say(f'I have {_offer(st.offered)} {_on(st.day, st.today)}. Which suits you better?')
        return _say('Which time would suit you?')

    def _cancel(self, st: State, t: str) -> AgentReply:
        if st.done == 'cancelled':
            return _say("It's cancelled. Anything else I can help you with?")
        if self.version == 'v2' and st.booking is None and st.day and len(st.users) == 1:
            # v2 regression: cancels by date straight away, without finding the booking
            return _say('', call('cancel_booking', date=st.day.isoformat(), phone=st.caller))
        if st.booking is None:
            if st.found_nothing and st.name:
                return _say('', call('find_booking', name=st.name))
            if st.found_nothing:
                return _say("I can't find a booking for this number. What name is it under?")
            return _say('', call('find_booking', phone=st.caller))
        if re.search(r'\b(yes|yeah|please|go ahead|sure)\b', t):
            return _say('', call('cancel_booking', booking_id=st.booking['booking_id']))
        return _say('Shall I cancel it?')

    def _reschedule(self, st: State, t: str) -> AgentReply:
        if st.done == 'moved':
            return _say('All done. Anything else?')
        if st.booking is None:
            return _say('', call('find_booking', phone=st.caller))
        new_day = st.day if st.day and st.day.isoformat() != st.booking['date'] else None
        if not new_day:
            return _say('What day would you like instead?')
        iso = new_day.isoformat()
        if self.version == 'v1' and st.wanted_time:  # v1: moves it without checking the new time
            return _say('', call('reschedule_booking', booking_id=st.booking['booking_id'], date=iso, time=st.wanted_time))
        if iso not in st.checked:
            return _say('', call('check_availability', date=iso, service=st.booking.get('service', 'checkup')))
        if st.wanted_time and st.wanted_time in st.checked[iso]:
            return _say('', call('reschedule_booking', booking_id=st.booking['booking_id'], date=iso, time=st.wanted_time))
        return _say(f'{_on(new_day, st.today, True)} I have {_offer(st.offered)}. Which one would you like?')

    def _info(self, st: State, t: str) -> AgentReply:
        if re.search(r'\b(how much|price|cost)', t) and st.service:
            if self.version == 'v1' and st.service == 'whitening':   # v1: from memory, and wrong
                return _say('Whitening is around $200. Anything else?')
            return _say('', call('get_price', service=st.service))
        if 'saturday' in t:
            return _say("On Saturdays we're open from 9 am to 1 pm. Anything else?")
        if re.search(r'\b(open|close|hours)\b', t):
            return _say("We're open Monday to Friday, 9 am to 6 pm, and Saturday mornings. Anything else?")
        if re.search(r'\b(park|parking|address|where)\b', t):
            return _say("We're at 12 Harbor Street, with free parking behind the building. Anything else?")
        return _say('Is there anything else I can help you with?')

    # ------------------------------------------------------------------ after a tool answers
    def _after_tools(self, st: State, messages: list[dict[str, Any]]) -> AgentReply:
        asked = next(m for m in reversed(messages) if m['role'] == 'assistant')
        name = asked['tool_calls'][-1]['function']['name']
        result = _json(messages[-1].get('content'))
        r = result if isinstance(result, dict) else {}
        if name == 'check_availability':
            day = st.day
            slots = r.get('slots') or []
            if not slots and day:
                if self.version == 'v1':             # v1: offers a time it never checked
                    return _say(f"{_day_name(day, st.today).capitalize()} is quite full, but how about 4:30 pm?")
                nxt = _next_open(day)
                if nxt.isoformat() not in st.checked:
                    return _say(f"{_day_name(day, st.today).capitalize()} is fully booked. Let me check the next day.",
                                call('check_availability', date=nxt.isoformat(), service=st.service or 'checkup'))
            if st.intent == 'reschedule' and st.booking:
                if st.wanted_time in slots:
                    return _say('', call('reschedule_booking', booking_id=st.booking['booking_id'],
                                         date=day.isoformat(), time=st.wanted_time))
                return _say(f'{_on(day, st.today, True)} I have {_offer(st.offered)}. Which one would you like?')
            if st.wanted_time and st.wanted_time in slots:
                if st.name:
                    return _say('', call('create_booking', name=st.name, phone=st.caller, date=day.isoformat(),
                                         time=st.wanted_time, service=st.service))
                return _say(f'{_clock(st.wanted_time).capitalize()} {_on(day, st.today)} is free. '
                            'Can I have your name, please?')
            if st.wanted_time and slots:
                return _say(f"Sorry, {_clock(st.wanted_time)} is taken. I have {_offer(st.offered)} "
                            f"{_on(day, st.today)}.")
            if not slots:
                return _say(f"Sorry, there's nothing free {_on(day, st.today)}.")
            return _say(f'{_on(day, st.today, True)} I have {_offer(st.offered)}. Which suits you better?')
        if name == 'create_booking':
            b = r
            when = _when(date.fromisoformat(b['date']), b['time'], st.today) if b.get('date') else _when(st.day, st.wanted_time, st.today)
            return _say(f"You're booked for a {_svc(b.get('service', st.service))} {when}. Anything else?")
        if name == 'find_booking':
            b = st.booking
            if not b:
                return _say("I can't find a booking for this number. What name is it under?")
            when = _when(date.fromisoformat(b['date']), b['time'], st.today)
            if st.intent == 'cancel':
                return _say(f'I can see a {_svc(b["service"])} {when}. Shall I cancel it?')
            if st.intent == 'reschedule':
                new_day = st.day if st.day and st.day.isoformat() != b['date'] else None
                if new_day and self.version == 'v1' and st.wanted_time:
                    return _say('', call('reschedule_booking', booking_id=b['booking_id'],
                                         date=new_day.isoformat(), time=st.wanted_time))
                if new_day:
                    return _say('', call('check_availability', date=new_day.isoformat(), service=b['service']))
                return _say(f'I can see your {_svc(b["service"])} {when}. What day would you like instead?')
            return _say(f'Your next appointment is a {_svc(b["service"])} {when}. Anything else?')
        if name == 'cancel_booking':
            return _say("Done, that's cancelled. Anything else I can help you with?")
        if name == 'reschedule_booking':
            moved = date.fromisoformat(r.get('date', st.day.isoformat()))
            return _say(f"Done. It's now {_when(moved, r.get('time', st.wanted_time), st.today)}. Anything else?")
        if name == 'get_price':
            return _say(f"A {_svc(r.get('service', st.service))} is ${r.get('price')}. Anything else?")
        if name == 'transfer_call':
            return _say('')
        return _say('Anything else I can help you with?')


# --------------------------------------------------------------------------- helpers
def call(tool: str, /, **args: Any) -> ToolCall:
    return ToolCall(tool, {k: v for k, v in args.items() if v not in (None, '')})


def _say(text: str, *calls: ToolCall) -> AgentReply:
    return AgentReply(text=text, tool_calls=list(calls))


def _json(v: Any) -> Any:
    try:
        return json.loads(v) if isinstance(v, str) else v
    except json.JSONDecodeError:
        return v


def _intent(t: str) -> str:
    if re.search(r'emergenc|really bad|swollen|bleeding|\bbill|invoice|insurance|speak to (a|some)', t):
        return 'transfer'
    if re.search(r'\bcancel', t):
        return 'cancel'
    if re.search(r'\b(move|reschedule|change)\b', t):
        return 'reschedule'
    if re.search(r'when is my|what time is my|do i have|confirm my', t):
        return 'lookup'
    if re.search(r'how much|price|cost|open|close|hours|parking|address', t):
        return 'info'
    if re.search(r'book|appointment|come in|available|free|slot|check-?up|cleaning|whitening|filling', t):
        return 'book'
    if re.search(r'pharmacy|pizza|wrong number|is this', t):
        return 'other'
    return ''


def _day(t: str, today: date) -> date | None:
    if 'today' in t or 'this afternoon' in t or 'this morning' in t:
        return today
    if 'tomorrow' in t:
        return today + timedelta(days=1)
    for i, name in enumerate(DAYS):
        if re.search(rf'\b{name}', t):
            found = today + timedelta(days=(i - today.weekday()) % 7 or 7)
            if re.search(rf'next {name}', t) and found.isocalendar()[1] == today.isocalendar()[1]:
                found += timedelta(days=7)              # "next Monday": the Monday of next week
            return found
    return None


def _name(text: str) -> str:
    m = re.search(r"(?i:it's|this is|my name is|name's|i'm|under)\s+([A-Z][a-z]+(?: [A-Z][a-z']+)?)", text)
    if m and m.group(1).lower() not in ('calling', 'looking', 'not', 'just', 'here'):
        return m.group(1)
    m = re.fullmatch(r"\s*([A-Z][a-z]+ [A-Z][a-z']+)\.?\s*", text)
    return m.group(1) if m else ''


def _reason(t: str) -> str:
    if re.search(r'emergenc|really bad|swollen|bleeding', t):
        return 'dental emergency'
    if re.search(r'bill|invoice|insurance', t):
        return 'billing question'
    return 'caller asked for a person'


def _part_ok(slot: str, part: str) -> bool:
    return not part or (slot < '12:00') == (part == 'morning')


def _next_open(d: date) -> date:
    n = d + timedelta(days=1)
    while n.weekday() == 6:
        n += timedelta(days=1)
    return n


def _clock(hhmm: str) -> str:
    h, m = int(hhmm[:2]), int(hhmm[3:5])
    suffix = 'am' if h < 12 else 'pm'
    h12 = h % 12 or 12
    return f'{h12} {suffix}' if m == 0 else f'{h12}:{m:02d} {suffix}'


def _offer(slots: list[str]) -> str:
    return ' or '.join(_clock(s) for s in slots)


def _day_name(d: date, today: date) -> str:
    if d == today:
        return 'today'
    if d == today + timedelta(days=1):
        return 'tomorrow'
    name = DAYS[d.weekday()].capitalize()
    return name if (d - today).days < 7 else f'{name} the {d.day}{_nth(d.day)}'


def _on(d: date | None, today: date, start: bool = False) -> str:
    """'tomorrow', 'on Friday': how a day goes in a sentence."""
    if not d:
        return ''
    day = _day_name(d, today)
    out = day if day in ('today', 'tomorrow') else 'on ' + day
    return out[0].upper() + out[1:] if start else out


def _when(d: date | None, hhmm: str, today: date) -> str:
    if not d:
        return ''
    return f'{_on(d, today)} at {_clock(hhmm)}' if hhmm else _on(d, today)


def _nth(n: int) -> str:
    return 'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')


def _svc(s: str) -> str:
    return {'checkup': 'check-up'}.get(s, s)

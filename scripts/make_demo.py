#!/usr/bin/env python3
"""Generate examples/dental/conversations: a week of synthetic phone calls to Harbor Dental.

Every name, number and appointment is invented. Each call is a scripted caller talking to the
demo agent's v1 (the "agent in production" of the story) against a simulated calendar, so the
recordings contain real tool calls with real results, and v1's known mistakes where they
happen. Latencies are drawn from a seeded range, like a phone agent's.

    python scripts/make_demo.py
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from callreplay.contract import load as load_contract  # noqa: E402
from callreplay.demo_agent import DemoAgent  # noqa: E402
from callreplay.formats import save  # noqa: E402
from callreplay.model import Conversation, ToolCall, Turn  # noqa: E402
from callreplay.replay import render_prompt  # noqa: E402

OUT = ROOT / 'examples' / 'dental'
GREETING = 'Thanks for calling Harbor Dental, this is Ava. How can I help?'
PRICES = {'checkup': 60, 'cleaning': 95, 'whitening': 250, 'filling': 140}
GRID = [f'{h:02d}:{m:02d}' for h in range(9, 18) for m in (0, 30)] + ['10:30', '14:15', '16:30']


@dataclass
class Call:
    id: str
    when: str                                   # started_at, local time
    caller: str
    says: list[str]
    slots: dict[str, list[str]] = field(default_factory=dict)       # date -> free times (overrides)
    bookings: list[dict[str, Any]] = field(default_factory=list)    # the caller's appointments
    by_name_only: bool = False                  # find_booking by phone finds nothing
    meta: dict[str, Any] = field(default_factory=dict)


def d(day: int) -> str:
    return f'2026-09-{day:02d}'


CALLS = [
    # --- booking ------------------------------------------------------------------------
    Call('call-001', '2026-09-14T09:12', '+1 555 0101',
         ["Hi, I'd like to book a cleaning on Thursday.", '2:15 works for me.', "It's Maria Lopez.",
          "No, that's all, thanks."], slots={d(17): ['10:30', '14:15', '16:00']}),
    Call('call-002', '2026-09-14T10:40', '+1 555 0102',
         ['Hello, do you have anything tomorrow morning for a check-up?', '9:30 is perfect.', 'Tom Baker.',
          "That's it, cheers."], slots={d(15): ['09:30', '11:00', '15:00']}),
    Call('call-003', '2026-09-14T12:05', '+1 555 0103',
         ['Hi, this is Priya Shah. Can I get a cleaning on Friday at 3 pm?',
          "No, that's everything, thank you. Bye."], slots={d(18): ['10:00', '15:00', '16:30']}),
    Call('call-004', '2026-09-15T09:03', '+1 555 0104',
         ["Hello, it's James Carter. Do you have 10:30 tomorrow for a filling?", 'Great, thank you, bye.'],
         slots={d(16): ['09:00', '10:30', '12:00']}),
    Call('call-005', '2026-09-15T11:26', '+1 555 0105',
         ["Hi, I'm Aisha Khan. Could I come in for whitening on Wednesday at 11 am?", '11:30 is fine then.',
          "No, that's all. Bye."], slots={d(16): ['11:30', '16:00', '17:00']}),
    Call('call-006', '2026-09-15T15:48', '+1 555 0106',
         ['Hi, can I book a check-up for Thursday afternoon?', 'Hmm, no. Friday would be better actually.',
          '3:30, please.', 'Daniel Kim.', 'No, thank you. Bye.'],
         slots={d(17): [], d(18): ['14:00', '15:30', '17:00']}),
    Call('call-007', '2026-09-16T09:31', '+1 555 0107',
         ['Hello, I need a filling appointment tomorrow.', 'Oh. What about Friday then?', '9:30 is good.',
          'Sofia Rossi.', "That's all, thank you. Bye."],
         slots={d(17): [], d(18): ['09:30', '11:00', '15:00']}),
    Call('call-008', '2026-09-16T10:15', '+1 555 0147',
         ['Hi, can I book a cleaning for Monday?', '9 am works.', "It's Liam Walsh.", 'No, thanks, bye.'],
         slots={d(21): ['09:00', '12:30', '16:00']}),
    Call('call-009', '2026-09-16T13:02', '+1 555 0109',
         ["Hi, I'd like to make an appointment.", 'Just a check-up.', 'Tomorrow afternoon if you can.',
          '4 pm, please.', 'Emma Novak.', 'Nothing else, thanks. Bye.'],
         slots={d(17): ['10:00', '16:00', '17:30']}),
    Call('call-010', '2026-09-16T16:44', '+1 555 0110',
         ['Hello, is there anything available today for a cleaning?', '5:30 please.', 'Lucas Silva.',
          'Perfect, thank you. Bye.'], slots={d(16): ['17:00', '17:30']}),
    Call('call-011', '2026-09-17T10:09', '+1 555 0111',
         ['Hi, can I book whitening next Monday in the morning?', '10:30 sounds good.', "It's Grace Chen.",
          "That's all, thanks. Bye."], slots={d(21): ['10:30', '11:30', '15:00', '16:00']}),
    Call('call-012', '2026-09-17T14:37', '+1 555 0112',
         ['Hello, do you have a slot on Saturday for a check-up?', '11 am is fine.', 'Omar Haddad.',
          'Great, thanks, bye.'], slots={d(19): ['09:30', '11:00', '12:00']}),
    # --- cancelling ---------------------------------------------------------------------
    Call('call-013', '2026-09-14T11:10', '+1 555 0113',
         ['Hi, I need to cancel my appointment.', 'Yes, please.', "No, that's all, thanks. Bye."],
         bookings=[{'booking_id': 'HD-4102', 'date': d(17), 'time': '10:00', 'service': 'cleaning'}]),
    Call('call-014', '2026-09-15T08:58', '+1 555 0114',
         ['Hello, I need to cancel my appointment tomorrow.', 'Yes, please.', 'No thanks, bye.'],
         bookings=[{'booking_id': 'HD-4117', 'date': d(16), 'time': '10:00', 'service': 'cleaning'}]),
    Call('call-015', '2026-09-16T12:20', '+1 555 0115',
         ['Hi, can I cancel my check-up on Friday?', 'Yes, go ahead.', 'Thank you, bye.'],
         bookings=[{'booking_id': 'HD-4125', 'date': d(18), 'time': '12:30', 'service': 'checkup'}]),
    Call('call-016', '2026-09-17T09:44', '+1 555 0116',
         ["Hi, I have to cancel an appointment, I can't make it.", "It's under Hannah Weber.", 'Yes, go ahead.',
          'Thank you, bye.'],
         bookings=[{'booking_id': 'HD-4131', 'date': d(21), 'time': '11:00', 'service': 'whitening'}],
         by_name_only=True),
    Call('call-017', '2026-09-18T15:05', '+1 555 0117',
         ['Hello, I need to cancel, something came up.', 'Yes, cancel it please.', 'No, bye.'],
         bookings=[{'booking_id': 'HD-4140', 'date': d(22), 'time': '15:30', 'service': 'filling'}]),
    # --- moving -------------------------------------------------------------------------
    Call('call-018', '2026-09-14T14:30', '+1 555 0118',
         ['Hi, can I move my appointment?', 'Friday afternoon if possible.', '4:30 is good.',
          "No, that's all. Bye."],
         slots={d(18): ['10:00', '16:30', '17:30']},
         bookings=[{'booking_id': 'HD-4108', 'date': d(17), 'time': '09:30', 'service': 'checkup'}]),
    Call('call-019', '2026-09-15T10:02', '+1 555 0119',
         ['Hello, could you move my cleaning on Wednesday to Thursday at 11 am?', "No, that's all, thanks."],
         slots={d(17): ['09:00', '11:00', '15:00']},
         bookings=[{'booking_id': 'HD-4119', 'date': d(16), 'time': '12:00', 'service': 'cleaning'}]),
    Call('call-020', '2026-09-16T11:11', '+1 555 0120',
         ['Hi, I need to change my filling appointment to Friday at 3:30 pm.', 'Thank you, bye.'],
         slots={d(18): ['10:30', '15:30', '16:00']},
         bookings=[{'booking_id': 'HD-4122', 'date': d(17), 'time': '09:00', 'service': 'filling'}]),
    Call('call-021', '2026-09-17T12:48', '+1 555 0121',
         ["Hi, I'd like to reschedule my check-up.", 'Monday morning?', '9:30.', 'Thanks, bye.'],
         slots={d(21): ['09:30', '10:00', '15:00']},
         bookings=[{'booking_id': 'HD-4133', 'date': d(18), 'time': '15:00', 'service': 'checkup'}]),
    Call('call-022', '2026-09-18T09:20', '+1 555 0122',
         ['Hello, can I move my whitening appointment please?', 'Next Tuesday afternoon.', '3 pm works.',
          "That's all, thank you. Bye."],
         slots={d(22): ['11:00', '15:00', '16:00']},
         bookings=[{'booking_id': 'HD-4144', 'date': d(21), 'time': '10:00', 'service': 'whitening'}]),
    # --- looking up ---------------------------------------------------------------------
    Call('call-023', '2026-09-14T16:02', '+1 555 0123',
         ['Hi, when is my next appointment?', 'Great, thanks. Bye.'],
         bookings=[{'booking_id': 'HD-4111', 'date': d(18), 'time': '11:00', 'service': 'cleaning'}]),
    Call('call-024', '2026-09-15T13:40', '+1 555 0124',
         ['Hello, do I have an appointment this week?', "Perfect, that's it, cheers."],
         bookings=[{'booking_id': 'HD-4120', 'date': d(17), 'time': '14:30', 'service': 'checkup'}]),
    Call('call-025', '2026-09-16T17:05', '+1 555 0125',
         ['Hi, what time is my appointment tomorrow?', 'Thank you, bye.'],
         bookings=[{'booking_id': 'HD-4128', 'date': d(17), 'time': '09:00', 'service': 'filling'}]),
    Call('call-026', '2026-09-18T11:34', '+1 555 0126',
         ['Hello, could you confirm my appointment?', 'Lovely, thanks. Bye.'],
         bookings=[{'booking_id': 'HD-4146', 'date': d(21), 'time': '16:00', 'service': 'cleaning'}]),
    # --- questions ----------------------------------------------------------------------
    Call('call-027', '2026-09-14T09:55', '+1 555 0127', ['Hi, how much is teeth whitening?', 'OK, thank you. Bye.']),
    Call('call-028', '2026-09-15T16:20', '+1 555 0128', ['Hello, what does whitening cost?', 'Alright, thanks, bye.']),
    Call('call-029', '2026-09-16T09:48', '+1 555 0129', ['Hi, how much is a cleaning?', 'Thanks, bye.']),
    Call('call-030', '2026-09-16T15:12', '+1 555 0130', ['What time do you close on Saturday?', 'Great, thank you. Bye.']),
    Call('call-031', '2026-09-17T11:58', '+1 555 0131', ['Hi, where are you and is there parking?', 'Perfect, thanks, bye.']),
    Call('call-032', '2026-09-17T16:30', '+1 555 0132', ['Hello, what are your opening hours?', "OK, that's it, cheers."]),
    Call('call-033', '2026-09-18T10:42', '+1 555 0133', ['Hi, how much does a filling cost?', 'OK, thanks. Bye.']),
    # --- to a person --------------------------------------------------------------------
    Call('call-034', '2026-09-14T08:51', '+1 555 0134',
         ["Hi, I've got a really bad toothache and my face is swollen, it's an emergency."]),
    Call('call-035', '2026-09-16T14:03', '+1 555 0135', ['Hello, I have a question about my bill.']),
    Call('call-036', '2026-09-18T13:27', '+1 555 0136', ['Can I speak to someone about my insurance?']),
    # --- not for us ---------------------------------------------------------------------
    Call('call-037', '2026-09-17T13:15', '+1 555 0137', ['Hi, is this the pharmacy?', 'Oh, sorry, wrong number. Bye.']),
    # --- nothing to judge ---------------------------------------------------------------
    Call('call-038', '2026-09-14T13:33', '+1 555 0138', [], meta={'duration_s': 4}),
    Call('call-039', '2026-09-15T12:51', '+1 555 0139', ['[inaudible]'], meta={'duration_s': 9}),
    Call('call-040', '2026-09-16T08:47', '+1 555 0140', ['...'], meta={'duration_s': 6}),
    Call('call-041', '2026-09-17T17:52', '+1 555 0141', ['Hello?'], meta={'duration_s': 7}),
    Call('call-042', '2026-09-18T08:30', '+1 555 0142', ['Testing, testing, one two three.'],
         meta={'unscorable': 'internal test call', 'duration_s': 15}),
]


class Clinic:
    """The simulated back office one call talks to."""

    def __init__(self, call: Call):
        self.call = call
        self.rng = random.Random(call.id)
        self.next_id = 5000 + int(call.id.split('-')[1]) * 7

    def __call__(self, name: str, args: dict[str, Any]) -> Any:
        c = self.call
        if name == 'check_availability':
            day = args.get('date', '')
            if day in c.slots:
                return {'date': day, 'service': args.get('service'), 'slots': c.slots[day]}
            weekday = date.fromisoformat(day).weekday()
            grid = [] if weekday == 6 else [t for t in sorted(set(GRID)) if weekday < 5 or t < '13:00']
            free = sorted(self.rng.sample(grid, k=min(4, len(grid))))
            return {'date': day, 'service': args.get('service'), 'slots': free}
        if name == 'create_booking':
            self.next_id += 1
            return {'ok': True, 'booking_id': f'HD-{self.next_id}', **{k: args.get(k) for k in ('date', 'time', 'service')}}
        if name == 'find_booking':
            if c.by_name_only and not args.get('name'):
                return {'bookings': []}
            return {'bookings': c.bookings}
        if name == 'reschedule_booking':
            return {'ok': True, 'booking_id': args.get('booking_id'), 'date': args.get('date'), 'time': args.get('time')}
        if name == 'cancel_booking':
            return {'ok': True, 'booking_id': args.get('booking_id')}
        if name == 'get_price':
            return {'service': args.get('service'), 'price': PRICES.get(args.get('service', ''), 0)}
        return {'ok': True}


def record(call: Call, prompt: str, tools: list[dict[str, Any]]) -> Conversation:
    agent = DemoAgent('v1')
    rng = random.Random('latency-' + call.id)
    conv = Conversation(id=call.id, caller=call.caller, started_at=call.when + ':00-04:00',
                        metadata={'channel': 'phone', **call.meta})
    conv.metadata.setdefault('duration_s', 20 + 11 * len(call.says) + rng.randint(0, 25))
    messages = [{'role': 'system', 'content': render_prompt(prompt, conv)},
                {'role': 'assistant', 'content': GREETING}]
    conv.turns.append(Turn('agent', GREETING))
    clinic = Clinic(call)
    n = 0
    for said in call.says:
        messages.append({'role': 'user', 'content': said})
        conv.turns.append(Turn('user', said))
        if call.meta.get('duration_s', 99) < 10 or call.meta.get('unscorable'):
            conv.turns.append(Turn('agent', 'Sorry, could you say that again?', latency_ms=rng.uniform(700, 1100)))
            break
        ended = False
        for _ in range(5):
            reply = agent.reply(messages, tools)
            calls = []
            for c in reply.tool_calls:
                n += 1
                calls.append(ToolCall(c.name, c.args, f'call_{n}'))
            latency = rng.uniform(380, 720) if calls and not reply.text else rng.uniform(640, 1480)
            conv.turns.append(Turn('agent', reply.text, calls, latency_ms=latency))
            messages.append({'role': 'assistant', 'content': reply.text,
                             **({'tool_calls': [{'id': c.id, 'type': 'function', 'function': {
                                 'name': c.name, 'arguments': json.dumps(c.args)}} for c in calls]} if calls else {})})
            if not calls:
                break
            for c in calls:
                result = clinic(c.name, c.args)
                conv.turns.append(Turn('tool', tool_call_id=c.id, name=c.name, result=result))
                messages.append({'role': 'tool', 'tool_call_id': c.id, 'content': json.dumps(result)})
                ended = ended or c.name in ('end_call', 'transfer_call')
            if ended:
                break
        if ended:
            break
    return conv


def main() -> None:
    contract = load_contract(OUT)
    folder = OUT / 'conversations'
    for old in folder.glob('*.json'):
        old.unlink()
    convs = [record(c, contract.prompt, contract.tools_schema) for c in CALLS]
    save(convs, folder)
    print(f'{len(convs)} conversations in {folder.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
